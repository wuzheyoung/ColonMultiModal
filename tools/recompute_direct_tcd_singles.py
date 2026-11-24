# -*- coding: utf-8 -*-
"""recompute_direct_tcd_singles.py — direct_tcd (n=160) 严格同口径单路基准重算。

背景 (m00003 任务1): 论文主表单路与融合需 100% 同样本口径。audit_fusion_debug.json 的
direct_tcd 单路疑似复用了全集结果。本脚本:
  (A) repro: 逐字镜像 audit_fusion_debug.py 的全部逻辑 (特征路径改指 features/logs,
      tgr 映射改读 --tgr-csv), 重新产出同结构 JSON → 与 audit_fusion_debug.json 逐字段
      diff, 相等即证明原表数值的生成机制 (哪些是真重算、哪些是特征覆盖集合恒等导致的
      "复用" —— 后者是机制正确而非抄写错误);
  (B) strict_n160_singles: 在完全相同的 direct_tcd n=160 交集上, 六路单路各自严格同折重算
      (clin_clean 用折内 SimpleImputer(median); 影像分支无缺失用 nan_to_num 等价路径),
      输出 Pooled OOF AUC + Fold-Mean ± SD, 供论文主表直接引用。

用法 (Windows 多进程安全, 含 if __name__ == "__main__" 保护):
  python tools/recompute_direct_tcd_singles.py --tgr-csv data/tgr_pids.csv
  python tools/recompute_direct_tcd_singles.py --no-tgr        # 机制预演 (n=all_tcd=220)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MASTER_CSV = "data/master_5fold_356.csv"
PAIRS_RAW = "data/hos_pairs_raw.csv"
FEAT_ROOT = "features/logs"
R3A_CSV = os.path.join(FEAT_ROOT, "route3a_radiomics_features.csv")
R3B = {"c": os.path.join(FEAT_ROOT, "route3b_c_features.csv"),
       "d": os.path.join(FEAT_ROOT, "route3b_d_features.csv")}
R2T_DIR = os.path.join(FEAT_ROOT, "route2_feats")
R2CD_DIR = os.path.join(FEAT_ROOT, "route2_cd_feats")
CLIN_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
OUT_JSON = "logs/recompute_direct_tcd_singles.json"


# ---------------- 与 audit_fusion_debug.py 逐字一致的特征加载 ----------------
def rad_feats(csv, mtype="body"):
    df = pd.read_csv(csv)
    df = df[(df["mask_type"] == mtype) & (df["ok"] == 1)]
    feat = [c for c in df.columns if c not in
            ("patient_id", "mask_type", "ok", "label_T", "label_N", "fold")]
    return df.set_index("patient_id")[feat]


def conv_feats_dir(d):
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".npy"):
            continue
        pid = int(fn.replace(".npy", ""))
        rows.append((pid, np.load(os.path.join(d, fn)).mean(axis=0)))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])


def conv_feats_npy():
    d_all = np.load(os.path.join(R2CD_DIR, "d.npy"))
    d_pids = pd.read_csv(os.path.join(R2CD_DIR, "d_pids.csv"))["patient_id"].values
    c_all = np.load(os.path.join(R2CD_DIR, "c.npy"))
    c_pids = pd.read_csv(os.path.join(R2CD_DIR, "c_pids.csv"))["patient_id"].values
    return (pd.DataFrame(d_all, index=d_pids), pd.DataFrame(c_all, index=c_pids))


# ---------------- 与 audit_fusion_debug.py 逐字一致的评估函数 ----------------
def per_fold_lr(X, y, folds, C_grid=(0.01, 0.1, 1.0, 10.0), imp=None):
    """折内 imp(median 或 None=nan_to_num) → scaler → 折内选 C → 返回逐折明细+oof。"""
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    detail = []
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 2 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            detail.append({"fold": int(k), "n_tr": int(tr.sum()), "n_va": int(va.sum()),
                           "skip": "degenerate"})
            continue
        if imp is not None:
            ip = SimpleImputer(strategy="median").fit(X[tr])
            Xtr = ip.transform(X[tr]); Xva = ip.transform(X[va])
        else:
            Xtr = np.nan_to_num(X[tr].astype(float), nan=0.0)
            Xva = np.nan_to_num(X[va].astype(float), nan=0.0)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
        best_c, best_a, best_m = C_grid[0], -1, None
        for c in C_grid:
            m = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(Xtr_s, y[tr])
            a = roc_auc_score(y[tr], m.predict_proba(Xtr_s)[:, 1])
            if a > best_a:
                best_a, best_c, best_m = a, c, m
        pv = best_m.predict_proba(Xva_s)[:, 1]
        oof[va] = pv
        try:
            va_auc = roc_auc_score(y[va], pv)
        except ValueError:
            va_auc = None
        detail.append({"fold": int(k), "n_tr": int(tr.sum()), "n_va": int(va.sum()),
                       "tr_pos": float(y[tr].mean()), "va_pos": float(y[va].mean()),
                       "va_p_mean": float(pv.mean()), "va_p_std": float(pv.std()),
                       "C": best_c, "val_auc": va_auc})
    return oof, detail


def pooled_auc(y, p):
    if p is None or np.isnan(p).any() or len(np.unique(y)) < 2:
        return None
    try:
        return round(float(roc_auc_score(y, p)), 4)
    except ValueError:
        return None


def fold_mean_auc(y, p, folds):
    fa = []
    for k in sorted(set(folds)):
        va = folds == k
        if va.sum() < 2 or len(np.unique(y[va])) < 2:
            continue
        try:
            fa.append(roc_auc_score(y[va], p[va]))
        except ValueError:
            pass
    fa = np.array(fa)
    if len(fa) == 0:
        return None, None
    return round(float(fa.mean()), 4), round(float(fa.std()), 4)


# ---------------- 核心: 与 audit 相同的 oof_for_branch / fusion_variants ----------------
def oof_for_branch(name, ids, branches, clin_clean_oof, d_ids, master):
    """audit 同款: clin_clean 用 direct_all 预计算 OOF 按 searchsorted 切片; 其余分支
    在 scope∩branch 上折内重训 (imp=None)。"""
    if name == "clin_clean":
        return clin_clean_oof[np.searchsorted(d_ids, ids)], ids
    fdf = branches[name]
    ids = np.array(sorted(set(ids) & set(fdf.index)))
    if len(ids) < 8:
        return None, None
    yl = master.set_index("patient_id").loc[ids, "label_T"].values.astype(int)
    fl = master.set_index("patient_id").loc[ids, "fold"].values.astype(int)
    X = fdf.loc[ids].values.astype(float)
    p, _ = per_fold_lr(X, yl, fl, imp=None)
    return p, ids


def singles_table(scope_ids, branch_names, branches, clin_clean_oof, d_ids, master):
    """与 audit fusion_variants 中 s1_singles 完全一致的单路输出结构。"""
    ids = np.array(sorted(scope_ids))
    out = {"n": int(len(ids)),
           "T+": int(master.set_index("patient_id").loc[ids, "label_T"].sum())}
    for nm in branch_names:
        p, bid = oof_for_branch(nm, ids, branches, clin_clean_oof, d_ids, master)
        if p is None:
            out[nm] = {"pooled": None, "fold_mean": None, "fold_sd": None,
                       "n": 0}
            continue
        yl = master.set_index("patient_id").loc[bid, "label_T"].values.astype(int)
        fl = master.set_index("patient_id").loc[bid, "fold"].values.astype(int)
        out[nm] = {"pooled": pooled_auc(yl, p),
                   "fold_mean": None, "fold_sd": None, "n": int(len(bid))}
        fm, sd = fold_mean_auc(yl, p, fl)
        out[nm]["fold_mean"] = fm
        out[nm]["fold_sd"] = sd
    return out


def strict_n160_singles(scope_ids, branch_names, branches, clin_raw, master):
    """(B) 严格同口径: 在 direct_tcd 自身折结构上, 每路单路重训 (clin_clean 折内 median;
    其余 imp=None)。返回 {name: {pooled, fold_mean, fold_sd, n, per_fold}}。"""
    m = master.set_index("patient_id")
    out = {}
    for nm in branch_names:
        if nm == "clin_clean":
            idsb = np.array(sorted(set(scope_ids) & set(clin_raw.index)))
            X = clin_raw.loc[idsb].values.astype(float)
            imp = "median"
        else:
            fdf = branches[nm]
            idsb = np.array(sorted(set(scope_ids) & set(fdf.index)))
            if len(idsb) < 8:
                out[nm] = {"pooled": None, "fold_mean": None, "fold_sd": None,
                           "n": 0, "per_fold": []}
                continue
            X = fdf.loc[idsb].values.astype(float)
            imp = None
        y = m.loc[idsb, "label_T"].values.astype(int)
        fl = m.loc[idsb, "fold"].values.astype(int)
        p, det = per_fold_lr(X, y, fl, imp=imp)
        out[nm] = {"pooled": pooled_auc(y, p), "n": int(len(idsb))}
        fm, sd = fold_mean_auc(y, p, fl)
        out[nm]["fold_mean"] = fm
        out[nm]["fold_sd"] = sd
        out[nm]["per_fold"] = [{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in d.items()} for d in det]
    return out


def load_tgr_from_csv(path):
    df = pd.read_csv(path)
    col = "patient_id" if "patient_id" in df.columns else df.columns[0]
    return {int(v) for v in df[col].dropna()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tgr-csv", default=None,
                    help="tgr (nCRT) 患者 patient_id 列表 csv; 缺省与 --no-tgr 二选一")
    ap.add_argument("--no-tgr", action="store_true",
                    help="不使用 tgr 过滤 (机制预演, direct=全量 355)")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    if args.tgr_csv is None and not args.no_tgr:
        ap.error("必须提供 --tgr-csv 或 --no-tgr")

    master = pd.read_csv(MASTER_CSV)
    raw = pd.read_csv(PAIRS_RAW).set_index("patient_id")
    tgr = load_tgr_from_csv(args.tgr_csv) if args.tgr_csv else set()
    direct = master[~master["patient_id"].isin(tgr)].copy()
    d_ids = np.sort(direct["patient_id"].values.astype(np.int64))
    print(f"tgr 患者 {len(tgr)} 例 → direct n={len(d_ids)} (tgr_cv={args.tgr_csv or 'none'})")

    rad_t = rad_feats(R3A_CSV, "body")
    rad_d = rad_feats(R3B["d"], "body")
    rad_c = rad_feats(R3B["c"], "body")
    conv_t = conv_feats_dir(R2T_DIR)
    conv_d, conv_c = conv_feats_npy()
    clin_raw = raw[CLIN_COLS].replace([np.inf, -np.inf], np.nan).astype(float)
    branches = {"t_rad": rad_t, "t_conv": conv_t, "d_rad": rad_d, "d_conv": conv_d,
                "c_rad": rad_c, "c_conv": conv_c, "clin": clin_raw}
    print(f"特征覆盖: t_rad {len(rad_t)} t_conv {len(conv_t)} d_rad {len(rad_d)} "
          f"d_conv {len(conv_d)} c_rad {len(rad_c)} c_conv {len(conv_c)}")

    res = {}
    # ---------- 任务1 (clin 核查, audit 同款) ----------
    t1 = {}
    Xc = clin_raw.loc[d_ids].values.astype(float)
    yT = master.set_index("patient_id").loc[d_ids, "label_T"].values.astype(int)
    folds = master.set_index("patient_id").loc[d_ids, "fold"].values.astype(int)
    p_old, _ = per_fold_lr(Xc, yT, folds, imp=None)
    p_clean, _ = per_fold_lr(Xc, yT, folds, imp="median")
    fm_o, sd_o = fold_mean_auc(yT, p_old, folds)
    fm_c, sd_c = fold_mean_auc(yT, p_clean, folds)
    t1["direct_n"] = len(d_ids)
    t1["nan_to_num"] = {"pooled": pooled_auc(yT, p_old), "fold_mean": fm_o, "fold_sd": sd_o}
    t1["imputer_median"] = {"pooled": pooled_auc(yT, p_clean), "fold_mean": fm_c, "fold_sd": sd_c}
    res["task1_clin"] = t1

    # ---------- 任务4 消融 (audit 同款) ----------
    clin_clean_oof, _ = per_fold_lr(Xc, yT, folds, imp="median")
    rp = raw.copy()
    has_p = {s: set(rp.index[rp[f"mri_path_{s}"].notna()]) for s in ("t", "c", "d")}
    scope_defs = {
        "direct_all": set(d_ids),
        "direct_td": set(d_ids) & has_p["t"] & has_p["d"],
        "direct_tcd": set(d_ids) & has_p["t"] & has_p["c"] & has_p["d"],
    }
    t4 = {"ablation": {}}
    steps = {
        "s1_singles": ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"],
        "s2_clin_dconv": ["clin_clean", "d_conv"],
        "s3_img_3plane": ["d_conv", "t_conv", "c_conv"],
        "s4_clin_3plane": ["clin_clean", "d_conv", "t_conv", "c_conv"],
        "s5_six": ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"],
    }
    for scope_name in ("direct_all", "direct_td", "direct_tcd"):
        t4["ablation"][scope_name] = {}
        for step, names in steps.items():
            t4["ablation"][scope_name][step] = singles_table(
                scope_defs[scope_name], names, branches, clin_clean_oof, d_ids, master)
    res["task4_fusion_debug"] = t4

    # ---------- (B) 严格 n=160 同口径单路 ----------
    res["strict_n160_singles"] = strict_n160_singles(
        scope_defs["direct_tcd"], ["clin_clean", "d_conv", "t_conv", "c_conv",
                                   "t_rad", "c_rad"], branches, clin_raw, master)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nDONE → {args.out}")
    st = res["strict_n160_singles"]
    print("\n[strict n=160 单路] (Pooled OOF / Fold-Mean±SD, n):")
    for nm, e in st.items():
        print(f"  {nm:10s} {e['pooled']} / {e['fold_mean']}±{e['fold_sd']}  n={e['n']}")


if __name__ == "__main__":
    main()
