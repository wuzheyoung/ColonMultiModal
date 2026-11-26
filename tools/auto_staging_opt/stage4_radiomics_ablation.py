# -*- coding: utf-8 -*-
"""stage4_radiomics_ablation.py — 阶段4: 放射组学消融 (intra / rim_3mm / rim_5mm)
TRIPOD 1b, 10×5 重复分层 CV (seed0-9, bonly), 开发队列 direct_td n=214。
特征: features/auto_staging_opt/radiomics_rim_features.csv (pid,seq,roi,~22 特征列)。
- H4.1 纯放射组学基线: radiomics_only & clin+radiomics
- H4.2 临床 + 放射组学(含瘤周环) + DINOv2 三模态融合 (在 H1.1 最优上)
对比: Baseline_0 td 0.6763, H1.1 (t60+dself+clin) td 0.7003。
输出: logs/auto_staging_opt/stage4_results.json + progress_log.md。
"""
import os, sys, json
import numpy as np
import pandas as pd
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
RAD = "features/auto_staging_opt/radiomics_rim_features.csv"
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

# 特征缓存
_pool_cache = {}
def pool_branch(root, seq, pool, pids):
    pids = list(pids)
    key = (root, seq, pool, tuple(pids))
    if key in _pool_cache:
        return _pool_cache[key]
    have, X, y = S.pool_branch(pids, seq, pool)
    _pool_cache[key] = (list(have), np.asarray(X, dtype=np.float64), np.asarray(y, dtype=float))
    return _pool_cache[key]

def load_radiomics(roi_list):
    """返回 dict {roi: (have_pids, X_df indexed by pid)}"""
    df = pd.read_csv(RAD)
    df = df[df.roi != "ERROR"]
    out = {}
    for roi in roi_list:
        g = df[df.roi == roi]
        cols = [c for c in g.columns if c not in ("seq", "pid", "roi", "mask_source")]
        piv = g.pivot_table(index="pid", columns="seq", values=cols, aggfunc="first")
        # pivot 后列是 (feat, seq) 多级 → 合并: pid × (feat_seq)
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        out[roi] = piv
    return out

def probe_matrix_oof(pids, Xmat, folds, yvals):
    """Xmat: DataFrame indexed by pid (特征矩阵), 折内 median impute + std + 嵌套选 C。"""
    pids = [int(p) for p in pids]
    X = Xmat.reindex(pids).values
    y = np.asarray([yvals[pos_of[p]] for p in pids], dtype=float)
    folds = np.asarray([folds[pos_of[p]] for p in pids], dtype=int)
    oof = np.full(len(pids), np.nan)
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            continue
        Xtr, ytr = X[tr], y[tr]
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
        Xva = SimpleImputer(strategy="median").fit(Xtr).transform(X[va])
        oof[va] = mm.predict_proba(sc.transform(Xva))[:, 1]
    rows = [{"patient_id": int(p), "fold": int(f), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp, f in zip(pids, y, oof, folds)]
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)

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
    line = f"- stage4 {tag}: " + " | ".join(f"{k} {v['repeated_mean']}±{v['repeated_sd']}" for k, v in res.items())
    with open(os.path.join(OUTDIR, "progress_log.md"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return res

RAD_ROIS = load_radiomics(["intra", "rim_3mm", "rim_5mm"])

def make_oof_rad_only(roi):
    def _inner(seed, fo):
        piv = RAD_ROIS[roi]
        pids = [int(p) for p in piv.index if int(p) in pos_of]
        rad_df = probe_matrix_oof(pids, piv, fo, y_all)
        cal, _ = S.calibrate_cv(rad_df, mode="bonly")
        return cal
    return _inner

def make_oof_clin_rad(roi):
    def _inner(seed, fo):
        piv = RAD_ROIS[roi]
        pids = [int(p) for p in piv.index if int(p) in pos_of]
        rad_df = probe_matrix_oof(pids, piv, fo, y_all)
        clin = clin_oof(all_direct, fo)
        fuse = S.F.fuse_zlogit_strategy({"clin": clin, "rad": rad_df}, strategy="B")
        cal, _ = S.calibrate_cv(fuse, mode="bonly")
        return cal
    return _inner

def make_oof_tri(roi):
    """H4.2: clin + 放射组学 + DINOv2 (H1.1: t60 AWM + dself meanmax)"""
    def _inner(seed, fo):
        piv = RAD_ROIS[roi]
        pids = [int(p) for p in piv.index if int(p) in pos_of]
        rad_df = probe_matrix_oof(pids, piv, fo, y_all)
        th, tX, ty = pool_branch(T60, "t", "area_w", all_direct)
        dh, dX, dy = pool_branch(T120, "d", "meanmax", d_self)
        tf = np.array([fo[pos_of[p]] for p in th])
        df_ = np.array([fo[pos_of[p]] for p in dh])
        t_df = probe_oof(th, tX, ty, tf)
        d_df = probe_oof(dh, dX, dy, df_)
        clin = clin_oof(all_direct, fo)
        fuse = S.F.fuse_zlogit_strategy({"clin": clin, "t": t_df, "d": d_df, "rad": rad_df}, strategy="B")
        cal, _ = S.calibrate_cv(fuse, mode="bonly")
        return cal
    return _inner

def main():
    results = {}
    for roi in ("intra", "rim_3mm", "rim_5mm"):
        results[f"rad_only_{roi}"] = report(f"rad_only({roi})", make_oof_rad_only(roi))
        results[f"clin_rad_{roi}"] = report(f"clin+rad({roi})", make_oof_clin_rad(roi))
    results["tri_intra"] = report("H4.2 clin+rad(intra)+t60+dself", make_oof_tri("intra"))
    results["tri_rim3"] = report("H4.2 clin+rad(rim_3mm)+t60+dself", make_oof_tri("rim_3mm"))
    results["tri_rim5"] = report("H4.2 clin+rad(rim_5mm)+t60+dself", make_oof_tri("rim_5mm"))
    with open(os.path.join(OUTDIR, "stage4_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUTDIR, "stage4_results.json"))

if __name__ == "__main__":
    main()
