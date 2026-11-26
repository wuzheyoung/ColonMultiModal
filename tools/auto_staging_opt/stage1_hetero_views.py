# -*- coding: utf-8 -*-
"""stage1_hetero_views.py — 阶段1: 异构视野 + 病灶物理尺度 + 最深浸润聚合
TRIPOD 1b, 开发队列 direct_td n=214 的 10×5 重复分层 CV (seed0-9, bonly 校准)。
Baseline_0 = clin_clean + d_self(MeanMax) + t(AWM, slice_feats_v2)  → 0.6763±0.0302 (direct_td)

H1.1 异构视野: t 改用 tight60_m10 + d_self(slice_feats_v2) + clin
H1.2 物理尺度标量: 从 tight60_m10 npz (mask_area_mm2/rel_z/slice_idx) 计算
     max_area_mm2 / total_area_mm2 / z_span_slices / area_concentration 及 d:t 面积比,
     折内 log1p+标准化, 加入 Baseline_0 与 H1.1。
H1.3 最深浸润聚合: t(tight60_m10) 上 AWM vs Top3 vs MaxArea vs 双尺度拼接(t60+t120 AWM, PCA/L2)。

所有评估: 10×5 (50 fold) Mean AUC ± SD(ddof1) + Pooled OOF AUC + ECE + Brier。
输出: logs/auto_staging_opt/stage1_*.json + 追加 progress_log.md。
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

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
T60 = "features/slice_feats_v2_tight60_m10"
T120 = "features/slice_feats_v2"
C_GRID = S.PROBE_C_GRID
C_LR = (0.01, 0.1, 1.0, 10.0)
SEEDS = range(10)

# ---------------- 数据装载 ----------------
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

# 池化缓存: (seq, root, pool, pids_tuple) -> (have, X)
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

# 物理尺度标量 (从 npz 直接读, 不依赖池化)
def phys_scalars(root, seq, pids, want_ratio_dt=False):
    """返回 DataFrame: patient_id + [max_area_mm2, total_area_mm2, z_span_slices, area_concentration(, t_area, d_area)]"""
    rows = []
    for p in pids:
        fp = os.path.join(root, BACKBONE, seq, f"{p}.npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp)
        mm2 = np.asarray(z["mask_area_mm2"], dtype=np.float64)
        relz = np.asarray(z["rel_z"], dtype=np.float64)
        rows.append({
            "patient_id": int(p),
            "max_area_mm2": float(mm2.max()),
            "total_area_mm2": float(mm2.sum()),
            "z_span_slices": int(len(mm2)),
            "area_concentration": float(mm2.max() / mm2.sum()),
            "area_sum": float(mm2.sum()),
        })
    df = pd.DataFrame(rows).set_index("patient_id")
    return df

# ---------------- 10×5 评估骨架 ----------------
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

def ece10_of(df):
    p = np.clip(df["prob"].values, 0, 1)
    y = df["label_T"].values
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)

def evals10x5(make_oof):
    """make_oof(seed, fold_of) -> pd.DataFrame(patient_id, fold, label_T, prob)。
    返回按 scope 汇总的 10×5 指标。"""
    per_seed = {}
    for seed in SEEDS:
        fo = fold_of_all(seed)
        df = make_oof(seed, fo)
        df["fold"] = df["fold"].astype(int)
        for scope_name, scope in (("td", set(td)), ("tcd", set(tcd))):
            g = df[df.patient_id.isin(scope)].copy()
            a = roc_auc_score(g["label_T"], g["prob"])
            per_seed.setdefault(scope_name, []).append(round(float(a), 4))
    out = {}
    for scope_name, vals in per_seed.items():
        v = np.array(vals)
        out[scope_name] = {
            "repeated_mean": round(float(v.mean()), 4),
            "repeated_sd": round(float(v.std(ddof=1)), 4),
            "per_seed": [round(float(x), 4) for x in v],
        }
    return out

def report(tag, make_oof, extra=None):
    res = evals10x5(make_oof)
    line = f"- {tag}: " + " | ".join(f"{k} {v['repeated_mean']}±{v['repeated_sd']}" for k, v in res.items())
    with open(os.path.join(OUTDIR, "progress_log.md"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return {"tag": tag, "res": res, "extra": extra or {}}

# ---------------- 组合构造 ----------------
def make_oof_baseline(seed, fo):
    t_have, t_X, t_y = pool_branch(T120, "t", "area_w", all_direct)
    d_have, d_X, d_y = pool_branch(T120, "d", "meanmax", d_self)
    tf = np.array([fo[pos_of[p]] for p in t_have])
    df = np.array([fo[pos_of[p]] for p in d_have])
    clin = clin_oof(all_direct, fo)
    t_df = probe_oof(t_have, t_X, t_y, tf)
    d_df = probe_oof(d_have, d_X, d_y, df)
    fuse = S.F.fuse_zlogit_strategy({"clin": clin, "t": t_df, "d": d_df}, strategy="B")
    cal, _ = S.calibrate_cv(fuse, mode="bonly")
    return cal

def make_oof_h11(seed, fo):
    """H1.1: t 用 tight60_m10 AWM + d_self(T120 meanmax) + clin"""
    t_have, t_X, t_y = pool_branch(T60, "t", "area_w", all_direct)
    d_have, d_X, d_y = pool_branch(T120, "d", "meanmax", d_self)
    tf = np.array([fo[pos_of[p]] for p in t_have])
    df = np.array([fo[pos_of[p]] for p in d_have])
    clin = clin_oof(all_direct, fo)
    t_df = probe_oof(t_have, t_X, t_y, tf)
    d_df = probe_oof(d_have, d_X, d_y, df)
    fuse = S.F.fuse_zlogit_strategy({"clin": clin, "t": t_df, "d": d_df}, strategy="B")
    cal, _ = S.calibrate_cv(fuse, mode="bonly")
    return cal

def make_oof_h12_base(seed, fo):
    """Baseline_0 + 物理尺度标量 (t 侧, tight60)"""
    t_have, t_X, t_y = pool_branch(T120, "t", "area_w", all_direct)
    d_have, d_X, d_y = pool_branch(T120, "d", "meanmax", d_self)
    clin = clin_oof(all_direct, fo)
    sc_t = phys_scalars(T60, "t", all_direct)
    feats = sc_t[["max_area_mm2", "total_area_mm2", "z_span_slices", "area_concentration"]].copy()
    feats = feats.reindex(all_direct)
    feats = feats.fillna(feats.median())
    feats_log = np.log1p(feats.values)
    feats_log[:, 2] = feats["z_span_slices"].fillna(0).values  # 层数不 log
    # 折内 OOF probe (标量)
    tf = np.array([fo[pos_of[p]] for p in t_have])
    df = np.array([fo[pos_of[p]] for p in d_have])
    t_df = probe_oof(t_have, t_X, t_y, tf)
    d_df = probe_oof(d_have, d_X, d_y, df)
    sc_df = probe_oof(all_direct, feats_log, y_all.astype(float), fo)
    fuse = S.F.fuse_zlogit_strategy({"clin": clin, "t": t_df, "d": d_df, "sc": sc_df}, strategy="B")
    cal, _ = S.calibrate_cv(fuse, mode="bonly")
    return cal

def make_oof_h12_h11(seed, fo):
    """H1.1 + 物理尺度标量"""
    t_have, t_X, t_y = pool_branch(T60, "t", "area_w", all_direct)
    d_have, d_X, d_y = pool_branch(T120, "d", "meanmax", d_self)
    clin = clin_oof(all_direct, fo)
    sc_t = phys_scalars(T60, "t", all_direct)
    feats = sc_t[["max_area_mm2", "total_area_mm2", "z_span_slices", "area_concentration"]].reindex(all_direct)
    feats = feats.fillna(feats.median())
    feats_log = np.log1p(feats.values)
    feats_log[:, 2] = feats["z_span_slices"].fillna(0).values
    tf = np.array([fo[pos_of[p]] for p in t_have])
    df = np.array([fo[pos_of[p]] for p in d_have])
    t_df = probe_oof(t_have, t_X, t_y, tf)
    d_df = probe_oof(d_have, d_X, d_y, df)
    sc_df = probe_oof(all_direct, feats_log, y_all.astype(float), fo)
    fuse = S.F.fuse_zlogit_strategy({"clin": clin, "t": t_df, "d": d_df, "sc": sc_df}, strategy="B")
    cal, _ = S.calibrate_cv(fuse, mode="bonly")
    return cal

def make_oof_h13_pools(seed, fo):
    """H1.3: t(tight60) 三池化 (AWM/Top3/MaxArea) 各自 + d_self + clin"""
    t_have, t_X, t_y = pool_branch(T60, "t", "area_w", all_direct)
    t3_have, t3_X, t3_y = pool_branch(T60, "t", "top3", all_direct)
    d_have, d_X, d_y = pool_branch(T120, "d", "meanmax", d_self)
    clin = clin_oof(all_direct, fo)
    tf = np.array([fo[pos_of[p]] for p in t_have])
    t3f = np.array([fo[pos_of[p]] for p in t3_have])
    df = np.array([fo[pos_of[p]] for p in d_have])
    t_df = probe_oof(t_have, t_X, t_y, tf)
    t3_df = probe_oof(t3_have, t3_X, t3_y, t3f)
    d_df = probe_oof(d_have, d_X, d_y, df)
    fuse = S.F.fuse_zlogit_strategy({"clin": clin, "t": t3_df, "d": d_df}, strategy="B")
    cal, _ = S.calibrate_cv(fuse, mode="bonly")
    return cal

def main():
    results = {}
    results["baseline_0"] = report("stage1 baseline_0 (t120awm+dself+clin, td)", make_oof_baseline)
    results["h11"] = report("stage1 H1.1 (t60awm+dself+clin)", make_oof_h11)
    results["h12_base"] = report("stage1 H1.2 baseline+phys_scalars(t60)", make_oof_h12_base)
    results["h12_h11"] = report("stage1 H1.2 H1.1+phys_scalars(t60)", make_oof_h12_h11)
    results["h13_top3"] = report("stage1 H1.3 t60 Top3+dself+clin", make_oof_h13_pools)
    with open(os.path.join(OUTDIR, "stage1_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUTDIR, "stage1_results.json"))

if __name__ == "__main__":
    main()
