# -*- coding: utf-8 -*-
"""任务1: tgr(nCRT) 拆队列统计与零重训重评。5折OOF L2 LR, 锁死 master 折表。"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "data", "master_5fold_356.csv")
RAW = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
LOG = os.path.join(ROOT, "logs")
X929 = os.path.join(ROOT, "data", "Hos", "结直肠癌多模态患者数据收集表9.29.xlsx")
XFULL = os.path.join(ROOT, "data", "Hos", "结直肠癌多模态患者数据收集表_完整版.xlsx")


def load_tgr_map():
    """batch2: 9.29 编号1-208 → pid=编号+200; batch1: 完整版行序1-163 → pid=行号"""
    d929 = pd.read_excel(X929)
    dfull = pd.read_excel(XFULL)
    m = {}
    for _, r in d929.iterrows():
        t = r["病理TGR"]
        if pd.notna(t):
            m[int(r["编号"]) + 200] = t
    for i, r in dfull.iterrows():
        t = r["病理TGR"]
        if pd.notna(t):
            m[i + 1] = t
    return m


def oof_lr(X, y, folds, C_grid=(0.01, 0.1, 1.0, 10.0)):
    oof = np.zeros(len(y))
    fold_aucs = []
    for fi in sorted(set(folds)):
        tr = folds != fi
        va = folds == fi
        if y[tr].sum() == 0 or y[va].sum() == 0 or y[va].sum() == len(y[va]):
            oof[va] = 0.5
            fold_aucs.append(0.5)
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        best, besta = None, -1
        for C in C_grid:
            mm = LogisticRegression(C=C, max_iter=3000, solver="lbfgs").fit(Xtr, y[tr])
            a = roc_auc_score(y[tr], mm.predict_proba(Xtr)[:, 1])
            if a > besta:
                besta, best = a, mm
        oof[va] = best.predict_proba(Xva)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y[va], oof[va]))
        except ValueError:
            fold_aucs.append(0.5)
    try:
        pooled = roc_auc_score(y, oof)
    except ValueError:
        pooled = 0.5
    return pooled, np.array(fold_aucs)


def report(X, y, folds, label):
    if len(y) < 8 or y.sum() == 0 or y.sum() == len(y):
        print(f"  [{label}] n={len(y)} 样本不足/单类, 跳过")
        return None
    pooled, folds_auc = oof_lr(X, y, folds)
    print(f"  [{label}] n={len(y)} pos={int(y.sum())} OOF AUC {pooled:.4f} "
          f"(折均值 {folds_auc.mean():.4f}±{folds_auc.std():.4f})")
    return {"n": int(len(y)), "pos": int(y.sum()),
            "pooled_auc": round(float(pooled), 4),
            "fold_mean": round(float(folds_auc.mean()), 4),
            "fold_std": round(float(folds_auc.std()), 4),
            "fold_aucs": [round(float(a), 4) for a in folds_auc]}


def main():
    master = pd.read_csv(MASTER)
    raw = pd.read_csv(RAW)
    tgr_map = load_tgr_map()

    # ── 1. 队列分布审计 (MRI-t 可用子集) ──
    rsub = raw[(raw["mri_available_t"] == True) | (raw["mri_available"] == True)].copy()
    rsub = rsub.merge(master[["patient_id", "fold"]], on="patient_id", how="inner")
    rsub["tgr"] = rsub["patient_id"].map(tgr_map)
    rsub["grp"] = np.where(rsub["tgr"].notna(), "ncrt", "direct")
    rsub["batch"] = np.where(rsub["patient_id"] <= 200, "batch1", "batch2")

    audit = {"all_master_n": len(master), "mri_t_sub_n": int(len(rsub))}
    audit["grp_counts"] = {"direct": int((rsub["grp"] == "direct").sum()),
                           "ncrt": int((rsub["grp"] == "ncrt").sum())}
    audit["tgr_value_counts"] = {str(k): int(v) for k, v in rsub["tgr"].value_counts(dropna=False).items()}
    print("== tgr 队列分布审计 (MRI-t 可用子集) ==")
    print(f"  全队列 n={len(master)} | MRI-t 子集 n={len(rsub)}")
    print(f"  直接手术(无tgr) {audit['grp_counts']['direct']} | 放化疗后(有tgr) {audit['grp_counts']['ncrt']}")
    print("  tgr 等级:", audit["tgr_value_counts"])
    for grp, gname in (("direct", "直接手术"), ("ncrt", "放化疗后")):
        g = rsub[rsub["grp"] == grp]
        audit[f"{grp}_pos"] = {"T+": round(float(g["label_T"].mean()), 4),
                               "N+": round(float(g["label_N"].mean()), 4),
                               "Tpos": int(g["label_T"].sum()), "Npos": int(g["label_N"].sum())}
        print(f"  [{gname}] T+率 {g['label_T'].mean():.3f} ({int(g['label_T'].sum())}/{len(g)}) "
              f"N+率 {g['label_N'].mean():.3f} ({int(g['label_N'].sum())}/{len(g)})")
    for b in ("batch1", "batch2"):
        g = rsub[rsub["batch"] == b]
        audit[f"{b}_grp"] = {"direct": int((g["grp"] == "direct").sum()),
                             "ncrt": int((g["grp"] == "ncrt").sum())}
        print(f"  [{b}] direct {(g['grp']=='direct').sum()} / ncrt {(g['grp']=='ncrt').sum()}")

    # ── 2. 零重训分群重评 ──
    print("\n== 分群重评 (L2 LR, 5折 OOF) ==")
    res = {}

    def feats_of(d, kind):
        if kind == "crc":
            return [c for c in d.columns if c.startswith("crc_")]
        return [c for c in d.columns if c not in {
            "patient_id", "mask_type", "ok", "label_T", "label_N", "fold",
            "split", "has_ct", "tgr", "grp", "batch"}]

    def group_report(X, y, folds, grp, tag):
        r = report(X, y, folds, f"{tag}|{grp}")
        if r:
            res.setdefault(tag, {})[grp] = r

    def per_grp(X, y, folds, grpmask, tag, tgt):
        for gname, gm in (("direct", ~grpmask), ("ncrt", grpmask)):
            if gm.sum() >= 8:
                group_report(X[gm], y[gm], folds[gm], f"{gname}_{tgt}", tag)

    # route1 body/ring (每例 2 行, 按 mask_type 拆)
    r1 = pd.read_csv(os.path.join(LOG, "route1_radiomics_features.csv"))
    r1["patient_id"] = r1["patient_id"].astype(int)
    r1["tgr"] = r1["patient_id"].map(tgr_map)
    r1["grp"] = np.where(r1["tgr"].notna(), "ncrt", "direct")
    for mt in ("body", "ring"):
        sub = r1[(r1["mask_type"] == mt) & (r1["ok"] == 1)].copy()
        feats = feats_of(sub, "py")
        X = sub[feats].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        grpmask = sub["grp"].values == "ncrt"
        for tgt, col in (("T", "label_T"), ("N", "label_N")):
            y = sub[col].values.astype(int)
            per_grp(X, y, sub["fold"].values, grpmask, f"route1_{mt}", tgt)
        # TRG 二分类 (放化疗组内)
        ok = sub["tgr"].notna()
        if ok.sum() >= 4:
            tt = pd.to_numeric(sub.loc[ok, "tgr"], errors="coerce")
            valid = tt.notna() & (tt >= 1) & (tt <= 3)
            if valid.sum() >= 4:
                subv = sub[ok][valid.values]
                y_trg = (pd.to_numeric(subv["tgr"], errors="coerce") <= 1).astype(int).values
                Xv = X[ok.values][valid.values]
                group_report(Xv, y_trg, subv["fold"].values, "trg_good<=1", f"route1_{mt}")

    # route2-MRI mean/max (280 例)
    feats_dir = os.path.join(LOG, "route2_feats")
    pids = sorted(int(x[:-4]) for x in os.listdir(feats_dir) if x.endswith(".npy"))
    meta = master[["patient_id", "fold", "label_T", "label_N"]].set_index("patient_id").loc[pids].reset_index()
    meta["tgr"] = meta["patient_id"].map(tgr_map)
    meta["grp"] = np.where(meta["tgr"].notna(), "ncrt", "direct")
    Xm = np.stack([np.mean(np.load(os.path.join(feats_dir, f"{p}.npy")), axis=0) for p in pids])
    Xx = np.stack([np.max(np.load(os.path.join(feats_dir, f"{p}.npy")), axis=0) for p in pids])
    for nm, X in (("mean", Xm), ("max", Xx)):
        grpmask = meta["grp"].values == "ncrt"
        for tgt, col in (("T", "label_T"), ("N", "label_N")):
            y = meta[col].values.astype(int)
            per_grp(X, y, meta["fold"].values, grpmask, f"r2mri_{nm}", tgt)
        ok = meta["tgr"].notna()
        if ok.sum() >= 4:
            tt = pd.to_numeric(meta.loc[ok, "tgr"], errors="coerce")
            valid = tt.notna() & (tt >= 1) & (tt <= 3)
            if valid.sum() >= 4:
                y_trg = (pd.to_numeric(meta.loc[ok, "tgr"][valid.values], errors="coerce") <= 1).astype(int).values
                group_report(X[ok.values][valid.values], y_trg,
                             meta.loc[ok, "fold"].values[valid.values], "trg_good<=1", f"r2mri_{nm}")

    # route2-CT cls (CRCFound, 145 例)
    cf = pd.read_csv(os.path.join(LOG, "crcfound_full_features.csv"))
    cf["patient_id"] = cf["patient_id"].astype(int)
    cf["tgr"] = cf["patient_id"].map(tgr_map)
    cf["grp"] = np.where(cf["tgr"].notna(), "ncrt", "direct")
    subc = cf[cf["has_ct"] == True].merge(master[["patient_id", "fold"]], on="patient_id", how="inner")
    feats_c = feats_of(subc, "crc")
    Xc = subc[feats_c].values.astype(np.float64)
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    grpmask_c = subc["grp"].values == "ncrt"
    for tgt, col in (("T", "label_T"), ("N", "label_N")):
        y = subc[col].values.astype(int)
        per_grp(Xc, y, subc["fold"].values, grpmask_c, "r2ct_cls", tgt)

    with open(os.path.join(LOG, "audit_tgr_task1.json"), "w", encoding="utf-8") as f:
        json.dump({"audit": audit, "res": res}, f, ensure_ascii=False, indent=2, default=str)
    print("\nDONE → logs/audit_tgr_task1.json")


if __name__ == "__main__":
    main()
