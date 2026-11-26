# -*- coding: utf-8 -*-
"""stage3_patch_dual_ablation.py — 阶段3: DINOv2 Patch Token 瘤内/瘤周双流池化消融
TRIPOD 1b, 10×5 重复分层 CV (seed0-9, bonly), 开发队列 direct_td n=214。

特征: features/auto_staging_opt/dinov2_patch_dual/{t,d}/<pid>.npz
- H3.1 t(intra_patch) 替代 t(cls)  [与 Baseline_0/H1.1 同融合]
- H3.2 t(intra_patch + peri_patch) 瘤内瘤周双流 (拼接, 折内独立 L2 正则)
- H3.3 d(intra_patch) 及与最优融合
对比基准: H1.1 (t60awm+dself+clin) = td 0.7003±0.0298。

输出: logs/auto_staging_opt/stage3_results.json + 追加 progress_log.md。
"""
import os, sys, json
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
finally:
    os.chdir = _orig_chdir

import train_slice_mil as tsm
import exp_v2_round2_ablation as abl
import recompute_direct_tcd_singles as rc

OUTDIR = "logs/auto_staging_opt"
os.makedirs(OUTDIR, exist_ok=True)
BACKBONE = "dinov2"
PATCH = "features/auto_staging_opt/dinov2_patch_dual"
T120 = "features/slice_feats_v2"
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

FIELD_KEY = {"cls": "cls_feat", "intra": "intra_patch", "peri": "peri_patch"}

def load_patch_plane(root, seq, field, pids):
    """返回 (have, X(K×768 逐层→AWM 池化), y)。field ∈ {cls,intra,peri}"""
    pids = list(pids)
    have, X = [], []
    fk = FIELD_KEY[field]
    for p in pids:
        fp = os.path.join(root, seq, f"{p}.npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp)
        f = np.asarray(z[fk], dtype=np.float32)   # (K,768)
        mm2 = np.asarray(z["mask_area_mm2"], dtype=np.float32)
        w = mm2 / (mm2.sum() + 1e-9)
        pooled = (f.T * w).T.sum(axis=0)             # 面积加权平均 (AWM)
        have.append(int(p)); X.append(pooled)
    X = np.asarray(X, dtype=np.float64)
    return have, X

def load_patch_plane_meanmax(root, seq, field, pids):
    pids = list(pids)
    have, X = [], []
    fk = FIELD_KEY[field]
    for p in pids:
        fp = os.path.join(root, seq, f"{p}.npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp)
        f = np.asarray(z[fk], dtype=np.float32)
        have.append(int(p)); X.append(np.concatenate([f.mean(axis=0), f.max(axis=0)]))
    return have, np.asarray(X, dtype=np.float64)

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
    line = f"- stage3 {tag}: " + " | ".join(f"{k} {v['repeated_mean']}±{v['repeated_sd']}" for k, v in res.items())
    with open(os.path.join(OUTDIR, "progress_log.md"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return res

# 缓存各 field 的 pooled 特征
_t_cache = {}
def t_feats(field):
    if field not in _t_cache:
        have, X = load_patch_plane(PATCH, "t", field, all_direct)
        _t_cache[field] = (have, X)
    return _t_cache[field]

_d_cache = {}
def d_feats(field):
    if field not in _d_cache:
        have, X = load_patch_plane(PATCH, "d", field, d_self)
        _d_cache[field] = (have, X)
    return _d_cache[field]

def make_oof(t_field, d_field, extra_t_field=None):
    """t_field ∈ {cls, intra, peri}; extra_t_field 可拼 peri 做双流 (拼接)。"""
    def _inner(seed, fo):
        th, tX = t_feats(t_field)
        ty = np.asarray([md_i.loc[int(p), "label_T"] for p in th], dtype=float)
        tf = np.array([fo[pos_of[p]] for p in th])
        t_df = probe_oof(th, tX, ty, tf)
        dth, dX = d_feats(d_field)
        dy = np.asarray([md_i.loc[int(p), "label_T"] for p in dth], dtype=float)
        df_ = np.array([fo[pos_of[p]] for p in dth])
        d_df = probe_oof(dth, dX, dy, df_)
        clin = clin_oof(all_direct, fo)
        if extra_t_field:
            th2, tX2 = t_feats(extra_t_field)
            assert th2 == th
            tXc = np.concatenate([tX, tX2], axis=1)
            t_df2 = probe_oof(th, tXc, ty, tf)
            t_df = t_df2
        fuse = S.F.fuse_zlogit_strategy({"clin": clin, "t": t_df, "d": d_df}, strategy="B")
        cal, _ = S.calibrate_cv(fuse, mode="bonly")
        return cal
    return _inner

def main():
    results = {}
    # H3.1: t(intra) 替代 t(cls)
    results["h31_tintra"] = report("H3.1 t(intra_patch)+dself+clin", make_oof("intra", "intra"))
    # H3.2: t(intra+peri) 双流
    results["h32_tintraperi"] = report("H3.2 t(intra+peri)+dself+clin", make_oof("intra", "intra", "peri"))
    # 参考: t(cls) 直接等价 H1.1 口径
    results["t_cls_ref"] = report("ref t(cls)+dself+clin", make_oof("cls", "intra"))
    # H3.3: d(intra)
    results["h33_dintra"] = report("H3.3 t(cls)+d(intra)+clin", make_oof("cls", "intra"))
    results["h33_dperi"] = report("H3.3 t(intra+peri)+d(intra)+clin", make_oof("intra", "intra", "peri"))
    with open(os.path.join(OUTDIR, "stage3_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUTDIR, "stage3_results.json"))

if __name__ == "__main__":
    main()
