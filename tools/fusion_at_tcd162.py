# -*- coding: utf-8 -*-
"""fusion_at_tcd162.py — 在提供的 tgr 清单口径 (direct_tcd n=162) 上, 用 audit_fusion_debug.py
的 fusion_variants 原版逻辑重算 s1-s5 融合 (f1 z-logit 校准均值 / f2 rank 均值),
供与 audit 的 n=160 融合基线 (0.6889/0.6887/0.6425/0.6896) 对照, 判断口径差异影响。
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("rc", os.path.join(ROOT, "tools", "recompute_direct_tcd_singles.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

master = pd.read_csv(rc.MASTER_CSV)
raw = pd.read_csv(rc.PAIRS_RAW).set_index("patient_id")
tgr = rc.load_tgr_from_csv("data/tgr_pids.csv")
direct = master[~master["patient_id"].isin(tgr)].copy()
d_ids = np.sort(direct["patient_id"].values.astype(np.int64))

rad_t = rc.rad_feats(rc.R3A_CSV, "body")
rad_d = rc.rad_feats(rc.R3B["d"], "body")
rad_c = rc.rad_feats(rc.R3B["c"], "body")
conv_t = rc.conv_feats_dir(rc.R2T_DIR)
conv_d, conv_c = rc.conv_feats_npy()
clin_raw = raw[rc.CLIN_COLS].replace([np.inf, -np.inf], np.nan).astype(float)
branches = {"t_rad": rad_t, "t_conv": conv_t, "d_rad": rad_d, "d_conv": conv_d,
            "c_rad": rad_c, "c_conv": conv_c, "clin": clin_raw}

has_p = {s: set(raw.index[raw[f"mri_path_{s}"].notna()]) for s in ("t", "c", "d")}
scope = np.array(sorted(set(d_ids) & has_p["t"] & has_p["c"] & has_p["d"]))
print(f"tgr {len(tgr)} → direct n={len(d_ids)} → direct_tcd n={len(scope)}")

m = master.set_index("patient_id")
y = m.loc[scope, "label_T"].values.astype(int)
fl = m.loc[scope, "fold"].values.astype(int)
print(f"T+={int(y.sum())}  folds={dict(zip(*np.unique(fl, return_counts=True)))}")

# clin_clean OOF: 与 audit 完全一致 = direct_all 折内 median 预计算后 searchsorted 切片
Xc = clin_raw.loc[d_ids].values.astype(float)
yT = m.loc[d_ids, "label_T"].values.astype(int)
fT = m.loc[d_ids, "fold"].values.astype(int)
clin_clean_oof, _ = rc.per_fold_lr(Xc, yT, fT, imp="median")
clin_sliced = clin_clean_oof[np.searchsorted(d_ids, scope)]


def logit_clip(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def oof_strict(name, ids):
    """strict 口径: 每分支在 scope 自身折上重训 (clin_clean imp=median, 其余 imp=None)。"""
    if name == "clin_clean":
        idsb = np.array(sorted(set(ids) & set(clin_raw.index)))
        X = clin_raw.loc[idsb].values.astype(float)
        imp = "median"
    else:
        fdf = branches[name]
        idsb = np.array(sorted(set(ids) & set(fdf.index)))
        X = fdf.loc[idsb].values.astype(float)
        imp = None
    yl = m.loc[idsb, "label_T"].values.astype(int)
    floc = m.loc[idsb, "fold"].values.astype(int)
    p, _ = rc.per_fold_lr(X, yl, floc, imp=imp)
    return p, idsb


def fusion(scope_ids, branch_names, clin_oof_mode):
    """audit fusion_variants 原版: f1 z-logit 折内校准均值 / f2 rank 均值。"""
    ids = scope_ids
    n = len(ids)
    yg = m.loc[ids, "label_T"].values.astype(int)
    single = {}
    for nm in branch_names:
        if clin_oof_mode == "audit_slice" and nm == "clin_clean":
            p = clin_sliced[np.searchsorted(scope, ids)]  # direct_all 预计算切片
            bid = ids
        else:
            p, bid = oof_strict(nm, ids)
        if p is not None:
            single[nm] = (bid, p)
    F1 = np.full(n, np.nan); C1 = np.zeros(n)
    for nm, (bid, p) in single.items():
        floc = m.loc[bid, "fold"].values.astype(int)
        yloc = m.loc[bid, "label_T"].values.astype(int)
        Lva = logit_clip(p)
        cal = np.full(len(bid), np.nan)
        for k in np.unique(floc):
            va = floc == k
            tr = ~va
            if tr.sum() < 2 or len(np.unique(yloc[tr])) < 2:
                cal[va] = Lva[va]
                continue
            if nm == "clin_clean":
                Xb = SimpleImputer(strategy="median").fit_transform(clin_raw.loc[bid].values.astype(float))
                ip = SimpleImputer(strategy="median").fit(Xb[tr])
                Xb_tr = ip.transform(Xb[tr]); Xb_va = ip.transform(Xb[va])
            else:
                Xb = branches[nm].loc[bid].values.astype(float)
                Xb_tr, Xb_va = Xb[tr], Xb[va]
            sc = StandardScaler().fit(Xb_tr)
            mdl = LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs").fit(sc.transform(Xb_tr), yloc[tr])
            Ptr = mdl.predict_proba(sc.transform(Xb_tr))[:, 1]
            Ltr = logit_clip(Ptr)
            m_tr, s_tr = Ltr.mean(), Ltr.std() + 1e-9
            cal[va] = (Lva[va] - m_tr) / s_tr
        idx = np.searchsorted(ids, bid)
        for ii, i in enumerate(idx):
            if not np.isnan(cal[ii]):
                F1[i] = cal[ii] if np.isnan(F1[i]) else F1[i] + cal[ii]
                C1[i] += 1
    m1 = C1 > 0
    F1[m1] /= C1[m1]
    F2 = np.full(n, np.nan); C2 = np.zeros(n)
    for nm, (bid, p) in single.items():
        r = rankdata(p) / len(p)
        idx = np.searchsorted(ids, bid)
        for ii, i in enumerate(idx):
            if not np.isnan(p[ii]):
                F2[i] = r[ii] if np.isnan(F2[i]) else F2[i] + r[ii]
                C2[i] += 1
    m2 = C2 > 0
    F2[m2] /= C2[m2]

    def auc_ignoring_nan(yv, pv, fv):
        ok = ~np.isnan(pv)
        if ok.sum() < 10 or len(np.unique(yv[ok])) < 2:
            return None, None, None
        a = rc.pooled_auc(yv[ok], pv[ok])
        fm, sd = rc.fold_mean_auc(yv[ok], pv[ok], fv[ok])
        return a, fm, sd

    out = {"n": int(n), "T+": int(yg.sum())}
    for nm, (bid, p) in single.items():
        yl = m.loc[bid, "label_T"].values.astype(int)
        out[nm] = {"pooled": rc.pooled_auc(yl, p)}
    fv = m.loc[ids, "fold"].values.astype(int)
    a, fm, sd = auc_ignoring_nan(yg, F1, fv)
    out["f1_zlogit_avg"] = {"pooled": a, "fold_mean": fm, "fold_sd": sd,
                            "n_valid": int((~np.isnan(F1)).sum())}
    a, fm, sd = auc_ignoring_nan(yg, F2, fv)
    out["f2_rank_avg"] = {"pooled": a, "fold_mean": fm, "fold_sd": sd,
                          "n_valid": int((~np.isnan(F2)).sum())}
    return out


steps = {
    "s1_singles": ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"],
    "s2_clin_dconv": ["clin_clean", "d_conv"],
    "s3_img_3plane": ["d_conv", "t_conv", "c_conv"],
    "s4_clin_3plane": ["clin_clean", "d_conv", "t_conv", "c_conv"],
    "s5_six": ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"],
}
res = {"scope": {"direct_n": int(len(d_ids)), "direct_tcd_n": int(len(scope)), "T+": int(y.sum())},
       "audit_slice_clin": {}, "strict_clin": {}}
for mode in ("audit_slice_clin", "strict_clin"):
    for step, names in steps.items():
        out = fusion(scope, names, "audit_slice" if mode == "audit_slice_clin" else "strict")
        res[mode][step] = {"n": out["n"], "T+": out["T+"],
                           "f1": out["f1_zlogit_avg"], "f2": out["f2_rank_avg"]}

with open(os.path.join(ROOT, "logs", "fusion_at_tcd162.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)

print("\n[direct_tcd n=162 融合, audit 同款 f1 z-logit 均值] (pooled / fold±sd, n_valid)")
for step in steps:
    e = res["audit_slice_clin"][step]["f1"]
    print(f"  {step:15s} {e['pooled']} / {e['fold_mean']}±{e['fold_sd']}  n_valid={e['n_valid']}")
print("\n[strict clin (折内重训) f1]")
for step in steps:
    e = res["strict_clin"][step]["f1"]
    print(f"  {step:15s} {e['pooled']} / {e['fold_mean']}±{e['fold_sd']}  n_valid={e['n_valid']}")
print("\n[audit n=160 对照] s1 0.6889 / s2 0.6887 / s3 0.6425 / s4 0.6896 (f1)")
