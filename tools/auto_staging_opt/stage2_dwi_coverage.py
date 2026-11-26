# -*- coding: utf-8 -*-
"""stage2_dwi_coverage.py — 阶段2: DWI 掩码覆盖率恢复 (from_t 纳回消融)
TRIPOD 1b, 10×5 重复分层 CV (seed0-9, bonly), 开发队列 direct_td n=214。

假设 H2: 将 d 分支覆盖从 d_self(n=104) 扩展至 d_self + from_t(方案C 审核通过),
对比是否拼接 is_d_self 0/1 指示变量; 分别在 Baseline_0(t120) 与 H1.1(t60) 上评估。

输出: logs/auto_staging_opt/stage2_results.json + 追加 progress_log.md。
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

_orig_chdir = os.chdir
def _safe_chdir(p):
    if os.path.isdir(p):
        return _orig_chdir(p)
os.chdir = _safe_chdir
try:
    import exp_v2_validation_suite as S   # noqa
    import exp_v2_final_calibrated as CALC  # noqa
finally:
    os.chdir = _orig_chdir

import train_slice_mil as tsm
import exp_v2_round2_ablation as abl
import recompute_direct_tcd_singles as rc

OUTDIR = "logs/auto_staging_opt"
os.makedirs(OUTDIR, exist_ok=True)
BACKBONE = "dinov2"
T120 = "features/slice_feats_v2"
T60 = "features/slice_feats_v2_tight60_m10"
C_GRID = S.PROBE_C_GRID
C_LR = (0.01, 0.1, 1.0, 10.0)
SEEDS = range(10)

md = S.md
md_i = S.md_i
raw = S.raw
all_direct = list(S.all_direct)
td = sorted(S.td)
tcd = sorted(S.tcd)
d_self = sorted(S.self_d)
pos_of = {int(p): i for i, p in enumerate(all_direct)}
y_all = md_i.loc[all_direct, "label_T"].values.astype(int)

def fold_of_all(seed):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fo = np.empty(len(all_direct), dtype=int)
    for k, (_, va) in enumerate(skf.split(all_direct, y_all)):
        fo[va] = k
    return fo

_pool_cache = {}
def pool_branch(root, seq, pool, pids):
    pids = list(pids)
    key = (root, seq, pool, tuple(pids))
    if key in _pool_cache:
        return _pool_cache[key]
    have = tsm.collect_plane_pids(md, root, BACKBONE, seq, set(pids))
    bags = tsm.preload_plane(md, root, BACKBONE, seq, have, "label_T", "none", torch.device("cpu"))
    X, y = abl.pool_bags(bags, have, pool)
    _pool_cache[key] = (have, np.asarray(X, dtype=np.float64), np.asarray(y, dtype=float))
    return _pool_cache[key]

def clin_oof(pids, folds):
    pids = np.asarray(pids, dtype=np.int64)
    folds = np.asarray(folds, dtype=int)
    Xc = raw.loc[pids, rc.CLIN_COLS].values
    yc = md_i.loc[pids, "label_T"].values.astype(float)
    oof = np.full(len(pids), np.nan)
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(yc[tr])) < 2:
            oof[va] = 0.5
            continue
        Xtr, ytr = Xc[tr], yc[tr]
        impm = SimpleImputer(strategy="median").fit(Xtr)
        Xtr_i = impm.transform(Xtr)
        inner = folds[tr] % 3
        best_c, best_a = None, -1.0
        for c in C_LR:
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(np.unique(ytr[itr])) < 2:
                    continue
                sc = StandardScaler().fit(Xtr_i[itr])
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr_i[itr]), ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(sc.transform(Xtr_i[iva]))[:, 1]))
            if aucs and np.mean(aucs) > best_a:
                best_a, best_c = float(np.mean(aucs)), c
        sc = StandardScaler().fit(Xtr_i)
        mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr_i), ytr)
        Xva = SimpleImputer(strategy="median").fit(Xc[tr]).transform(Xc[va])
        oof[va] = mm.predict_proba(sc.transform(Xva))[:, 1]
    rows = [{"patient_id": int(p), "fold": int(f), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp, f in zip(pids, yc, oof, folds)]
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)

def probe_oof(have, X, y, folds):
    folds = np.asarray(folds, dtype=int)
    oof = np.full(len(have), np.nan)
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            continue
        Xtr, ytr = X[tr], y[tr]
        inner = folds[tr] % 3
        best_c, best_a = None, -1.0
        for c in C_GRID:
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(np.unique(ytr[itr])) < 2:
                    continue
                sc = StandardScaler().fit(Xtr[itr])
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr[itr]), ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(sc.transform(Xtr[iva]))[:, 1]))
            if aucs and np.mean(aucs) > best_a:
                best_a, best_c = float(np.mean(aucs)), c
        sc = StandardScaler().fit(Xtr)
        mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
        oof[va] = mm.predict_proba(sc.transform(X[va]))[:, 1]
    rows = [{"patient_id": int(p), "fold": int(f), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp, f in zip(have, y, oof, folds)]
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)

def evals10x5(make_oof):
    per_seed = {"td": [], "tcd": []}
    for seed in SEEDS:
        fo = fold_of_all(seed)
        df = make_oof(seed, fo)
        df["fold"] = df["fold"].astype(int)
        for scope_name, scope in (("td", set(td)), ("tcd", set(tcd))):
            g = df[df.patient_id.isin(scope)].copy()
            per_seed[scope_name].append(round(float(roc_auc_score(g["label_T"], g["prob"])), 4))
    out = {}
    for scope_name, vals in per_seed.items():
        v = np.array(vals)
        out[scope_name] = {"repeated_mean": round(float(v.mean()), 4),
                           "repeated_sd": round(float(v.std(ddof=1)), 4),
                           "per_seed": [round(float(x), 4) for x in v]}
    return out

def report(tag, make_oof):
    res = evals10x5(make_oof)
    line = f"- stage2 {tag}: " + " | ".join(f"{k} {v['repeated_mean']}±{v['repeated_sd']}" for k, v in res.items())
    with open(os.path.join(OUTDIR, "progress_log.md"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return res

# d 分支构建: mode='self' | 'all' (self+from_t) | 'all+ind'
# from_t pids = manifest mask_source=='from_t' & has_valid==1 & cohort_direct
def d_pids_sets():
    man = pd.read_csv("features/dwi_mask_remap_manifest.csv")
    man["pid"] = man["pid"].astype(int)
    ft = set(man[(man.mask_source == "from_t") & (man.has_valid_mask == 1) & (man.cohort_direct == 1)]["pid"])
    return d_self, ft

def make_oof(d_root, d_mode, t_root, use_ind):
    def _inner(seed, fo):
        t_have, t_X, t_y = pool_branch(t_root, "t", "area_w", all_direct)
        dself, ft = d_pids_sets()
        if d_mode == "self":
            d_pids = dself
        else:
            d_pids = sorted(set(dself) | ft)
        d_have, d_X, d_y = pool_branch(d_root, "d", "meanmax", d_pids)
        clin = clin_oof(all_direct, fo)
        tf = np.array([fo[pos_of[p]] for p in t_have])
        df_ = np.array([fo[pos_of[p]] for p in d_have])
        t_df = probe_oof(t_have, t_X, t_y, tf)
        d_df = probe_oof(d_have, d_X, d_y, df_)
        if use_ind:
            # is_d_self 指示符: d_have 中属 self 为 1, from_t 为 0; 无 d 的例不参与影像融合
            ind = np.array([1.0 if p in set(dself) else 0.0 for p in d_have])
            # 作为第 769 维附加特征与 d 特征拼接
            d_X2 = np.concatenate([d_X, ind[:, None]], axis=1)
            d_df2 = probe_oof(d_have, d_X2, d_y, df_)
            d_df = d_df2
        fuse = S.F.fuse_zlogit_strategy({"clin": clin, "t": t_df, "d": d_df}, strategy="B")
        cal, _ = S.calibrate_cv(fuse, mode="bonly")
        return cal
    return _inner

def main():
    dself, ft = d_pids_sets()
    print(f"d_self={len(dself)} from_t={len(ft)} union={len(set(dself)|ft)}")
    results = {}
    # Baseline_0 (t120) 上: self vs all vs all+ind
    results["b0_dself"] = report("b0 t120 dself+clin+t", make_oof(T120, "self", T120, False))
    results["b0_dall"] = report("b0 t120 dall(self+ft)+clin+t", make_oof(T120, "all", T120, False))
    results["b0_dall_ind"] = report("b0 t120 dall+is_d_self ind+clin+t", make_oof(T120, "all", T120, True))
    # H1.1 (t60) 上: self vs all vs all+ind
    results["h11_dself"] = report("h11 t60 dself+clin+t", make_oof(T120, "self", T60, False))
    results["h11_dall"] = report("h11 t60 dall(self+ft)+clin+t", make_oof(T120, "all", T60, False))
    results["h11_dall_ind"] = report("h11 t60 dall+is_d_self ind+clin+t", make_oof(T120, "all", T60, True))
    with open(os.path.join(OUTDIR, "stage2_results.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "d_self_n": len(dself), "from_t_n": len(ft)}, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUTDIR, "stage2_results.json"))

if __name__ == "__main__":
    main()
