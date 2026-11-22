# -*- coding: utf-8 -*-
"""终极探针: 用路线1/路线2 已落盘特征预测 sex + T 分期, 并打印管线参数/间距/方向。
判据(用户 m03195): sex OOF AUC<0.80 → 代码/对齐/ID bug, 此前结论作废;
sex ≥0.85-0.95 而 T/N 仍 0.48-0.50 → 特征管线无误, 问题在标签口径或信号稀释。
"""
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


def oof_lr(X, y, folds, C_grid=(0.01, 0.1, 1.0, 10.0)):
    oof = np.zeros(len(y))
    for fi in sorted(set(folds)):
        tr = folds != fi
        va = folds == fi
        if y[tr].sum() == 0 or y[va].sum() == 0:
            oof[va] = 0.5
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xva = sc.transform(X[va])
        best = None
        besta = -1
        for C in C_grid:
            m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs").fit(Xtr, y[tr])
            a = roc_auc_score(y[tr], m.predict_proba(Xtr)[:, 1])
            if a > besta:
                besta, best = a, m
        oof[va] = best.predict_proba(Xva)[:, 1]
    try:
        return roc_auc_score(y, oof), int(len(y)), int(y.sum())
    except ValueError:
        return 0.5, int(len(y)), int(y.sum())


def main():
    out = {}
    master = pd.read_csv(MASTER)
    raw = pd.read_csv(RAW)
    raw["patient_id"] = raw["patient_id"].astype(int)
    meta = master[["patient_id", "fold", "label_T", "label_N"]].merge(
        raw[["patient_id", "sex"]], on="patient_id", how="left"
    )
    print("meta rows:", len(meta), "sex non-null:", int(meta["sex"].notna().sum()))

    # ── route1 影像组学特征 → sex ──
    r1 = pd.read_csv(os.path.join(LOG, "route1_radiomics_features.csv"))
    r1["patient_id"] = r1["patient_id"].astype(int)
    feats1 = [c for c in r1.columns if c not in {
        "patient_id", "mask_type", "ok", "label_T", "label_N", "fold"}]
    for mt, tgt in (("body", "sex"), ("ring", "sex")):
        sub = r1[(r1["mask_type"] == mt) & (r1["ok"] == 1)].merge(
            meta[["patient_id", "sex"]], on="patient_id", how="left")
        sub = sub[sub["sex"].notna()]
        X = sub[feats1].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        auc, n, pos = oof_lr(X, sub["sex"].values.astype(int), sub["fold"].values)
        print(f"[route1 {mt}] sex LR OOF AUC {auc:.4f} (n={n}, M={pos})")
        out[f"route1_{mt}_sex"] = {"auc": auc, "n": n, "pos": pos}

    # ── route2-MRI ConvNeXt 特征 → sex / T ──
    feats_dir = os.path.join(LOG, "route2_feats")
    pids = sorted(int(p[:-4]) for p in os.listdir(feats_dir) if p.endswith(".npy"))
    Xm = np.stack([np.mean(np.load(os.path.join(feats_dir, f"{p}.npy")), axis=0) for p in pids])
    Xmmax = np.stack([np.max(np.load(os.path.join(feats_dir, f"{p}.npy")), axis=0) for p in pids])
    sub = meta[meta["patient_id"].isin(pids)].set_index("patient_id").loc[pids].reset_index()
    for nm, X in (("mean", Xm), ("max", Xmmax)):
        for tgt in ("sex", "label_T", "label_N"):
            y = sub[tgt].values.astype(int)
            auc, n, pos = oof_lr(X, y, sub["fold"].values)
            print(f"[route2-MRI {nm}] {tgt} LR OOF AUC {auc:.4f} (n={n}, pos={pos})")
            out[f"r2mri_{nm}_{tgt}"] = {"auc": auc, "n": n, "pos": pos}

    # ── route2-CT CRCFound cls 特征 → sex (仅 has_ct 行) ──
    cf = pd.read_csv(os.path.join(LOG, "crcfound_full_features.csv"))
    cf["patient_id"] = cf["patient_id"].astype(int)
    cfcols = [c for c in cf.columns if c.startswith("crc_")]
    subc = cf[cf["has_ct"] == True].merge(meta, on="patient_id", suffixes=("", "_m"))
    subc = subc[subc["sex"].notna()]
    Xc = subc[cfcols].values.astype(np.float64)
    auc, n, pos = oof_lr(Xc, subc["sex"].values.astype(int), subc["fold"].values)
    print(f"[route2-CT cls] sex LR OOF AUC {auc:.4f} (n={n}, M={pos})")
    out["r2ct_cls_sex"] = {"auc": auc, "n": n, "pos": pos}

    with open(os.path.join(LOG, "probe_sex.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── route1 参数打印 ──
    print("\n[route1 参数核查]")
    print("  normalize=True (zscore), binWidth=25, force2D=False, resampledPixelSpacing=None(原生分辨率)")
    print("  特征类: firstorder/glcm/glrlm/glszm/gldm/ngtdm/shape; 未做直肠下段 15cm 截断(全结肠 mask)")
    n_feat_cols = len(feats1)
    X1 = r1[r1["ok"] == 1][feats1].values.astype(np.float64)
    X1 = np.nan_to_num(X1, nan=0.0, posinf=0.0, neginf=0.0)
    nz = int((np.nanstd(X1, axis=0) > 1e-4).sum())
    print(f"  特征列数={n_feat_cols}, 非零方差特征数(std>1e-4)={nz}")

    # ── mri_path_t spacing / axcodes 分布 ──
    print("\n[mri_path_t spacing/方向分布]")
    import nibabel as nib
    sp_list, ax_list = [], []
    for p in raw[raw["mri_path_t"].notna()]["mri_path_t"].unique():
        if not os.path.exists(p):
            continue
        try:
            img = nib.load(p)
            sp_list.append(tuple(round(float(x), 1) for x in img.header.get_zooms()[:3]))
            ax_list.append("".join(nib.aff2axcodes(img.affine)))
        except Exception as e:
            print(f"  load fail {p}: {e}")
    from collections import Counter
    print("  spacing 计数 top10:", Counter(sp_list).most_common(10))
    print("  axcodes 计数:", Counter(ax_list))


if __name__ == "__main__":
    main()
