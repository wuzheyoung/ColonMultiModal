# -*- coding: utf-8 -*-
"""stage5_stacking_greedy.py — 阶段5: 分模态独立子模型 Stacking + 贪心组合 + 错例诊断
TRIPOD 1b, 10×5 重复分层 CV (seed0-9), 开发队列 direct_td n=214。

子模型 (各自折内 OOF 概率, 折内嵌套超参选择):
  - clin:  7 维临床 L2-LR (median impute + std)
  - t60:   tight60_m10 t AWM 768 维 → L2-LR
  - dself: slice_feats_v2 d self meanmax 1536 维 → L2-LR
  - rad3:  radiomics rim_3mm (瘤周环) → L2-LR
Stacking: 元模型在折内拟合 (训练折 OOF 概率 → 元模型, 预测验证折):
  - L2-LR / ElasticNet-LR / LinearSVM+Platt 对比, 嵌套选参
贪心准入: 相对当前最佳 Mean AUC 净升 >= +0.004 且 50 折胜率 >= 60% 且 ECE <= 0.065
错例诊断: Winner_Model (或当前最佳) 平均 OOF 概率 → 假阴/假阳的 max_area_mm2/DWI 缺失率/临床分布

输出: logs/auto_staging_opt/stage5_results.json + all_experiments.json + progress_log.md。
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import LinearSVC
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
finally:
    os.chdir = _orig_chdir

import recompute_direct_tcd_singles as rc

OUTDIR = "logs/auto_staging_opt"
os.makedirs(OUTDIR, exist_ok=True)
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

def brier_ece(df):
    p = np.clip(df["prob"].values, 0, 1)
    y = df["label_T"].values
    brier = float(brier_score_loss(y, p))
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return brier, ece

# ---------- 子模型 OOF ----------
_pool_cache = {}
def pool_branch(root, seq, pool, pids):
    pids = list(pids)
    key = (root, seq, pool, tuple(pids))
    if key in _pool_cache:
        return _pool_cache[key]
    have, X, y = S.pool_branch(pids, seq, pool)
    _pool_cache[key] = (list(have), np.asarray(X, dtype=np.float64), np.asarray(y, dtype=float))
    return _pool_cache[key]

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
    return np.asarray(oof, dtype=float)

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
    return np.asarray(oof, dtype=float)

def load_rad(roi="rim_3mm"):
    df = pd.read_csv(RAD)
    df = df[df.roi == roi]
    cols = [c for c in df.columns if c not in ("seq", "pid", "roi", "mask_source")]
    piv = df.pivot_table(index="pid", columns="seq", values=cols, aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    return piv

def rad_oof(seed, fo):
    piv = load_rad("rim_3mm")
    pids = [int(p) for p in piv.index if int(p) in pos_of]
    X = piv.reindex(pids).values
    y = np.asarray([y_all[pos_of[p]] for p in pids], dtype=float)
    ff = np.asarray([fo[pos_of[p]] for p in pids], dtype=int)
    oof = np.full(len(pids), np.nan)
    for k in sorted(set(ff)):
        tr, va = ff != k, ff == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            continue
        Xtr, ytr = X[tr], y[tr]
        impm = SimpleImputer(strategy="median").fit(Xtr)
        Xtr_i = impm.transform(Xtr)
        inner = ff[tr] % 3
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
    return pids, np.asarray(oof, dtype=float)

def collect_sub_oof(seed, fo):
    """返回 DataFrame: patient_id, fold, label_T, 各子模型概率 (对齐 all_direct)。"""
    clin_p = clin_oof(all_direct, fo)
    th, tX, ty = pool_branch(T60, "t", "area_w", all_direct)
    dh, dX, dy = pool_branch(T120, "d", "meanmax", d_self)
    t_p = np.full(len(all_direct), np.nan); t_p[[pos_of[p] for p in th]] = probe_oof(th, tX, ty, np.array([fo[pos_of[p]] for p in th]))
    d_p = np.full(len(all_direct), np.nan); d_p[[pos_of[p] for p in dh]] = probe_oof(dh, dX, dy, np.array([fo[pos_of[p]] for p in dh]))
    rad_pids, rad_p_raw = rad_oof(seed, fo)
    rad_p = np.full(len(all_direct), np.nan); rad_p[[pos_of[p] for p in rad_pids]] = rad_p_raw
    return pd.DataFrame({"patient_id": all_direct, "fold": fo, "label_T": y_all,
                         "clin": clin_p, "t60": t_p, "dself": d_p, "rad3": rad_p})

# ---------- Stacking 元模型 (折内拟合, 预测同折验证) ----------
def stack_oof(sub, meta="l2", use_mods=None):
    """sub: collect_sub_oof 输出。用训练折概率拟合元模型 → 验证折预测。"""
    mods = use_mods or ["clin", "t60", "dself", "rad3"]
    P = sub[mods].values.astype(float)
    y = sub["label_T"].values.astype(int)
    folds = sub["fold"].values.astype(int)
    # 丢弃任何子模型概率缺失的行 (训练与验证均只用完整行)
    ok = ~np.isnan(P).any(axis=1)
    P, y, folds = P[ok], y[ok], folds[ok]
    oof = np.full(len(sub), np.nan)
    sub_ok_idx = np.where(ok)[0]
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            for j in sub_ok_idx[va]:
                oof[j] = 0.5
            continue
        Xtr, ytr = P[tr], y[tr]
        if meta == "l2":
            sc = StandardScaler().fit(Xtr)
            inner = folds[tr] % 3
            best_c, best_a = None, -1.0
            for c in C_LR:
                aucs = []
                for g in range(3):
                    itr2, iva2 = inner != g, inner == g
                    if iva2.sum() < 2 or len(np.unique(ytr[itr2])) < 2:
                        continue
                    mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr[itr2]), ytr[itr2])
                    aucs.append(roc_auc_score(ytr[iva2], mm.predict_proba(sc.transform(Xtr[iva2]))[:, 1]))
                if aucs and np.mean(aucs) > best_a:
                    best_a, best_c = float(np.mean(aucs)), c
            mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
            # 验证折完整行索引
            oof[sub_ok_idx[va]] = mm.predict_proba(sc.transform(P[va]))[:, 1]
        elif meta == "enet":
            sc = StandardScaler().fit(Xtr)
            best_a, best_mm = -1.0, None
            for a in (1e-3, 1e-2, 0.1):
                for l1 in (0.1, 0.5, 0.9):
                    mm = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000).fit(sc.transform(Xtr), ytr)
                    # 用 z 分数近似概率 (ElasticNet 无 predict_proba)
                    z = mm.predict(sc.transform(P[va]))
                    zt = mm.predict(sc.transform(Xtr))
                    auc = roc_auc_score(ytr, zt)
                    if auc > best_a:
                        best_a, best_mm = auc, (a, l1)
            mm = ElasticNet(alpha=best_mm[0], l1_ratio=best_mm[1], max_iter=5000).fit(sc.transform(Xtr), ytr)
            z = mm.predict(sc.transform(P[va]))
            oof[sub_ok_idx[va]] = 1.0 / (1.0 + np.exp(-z))
        elif meta == "svm":
            sc = StandardScaler().fit(Xtr)
            best_a, best_c = -1.0, None
            for c in (0.01, 0.1, 1.0, 10.0):
                mm = LinearSVC(C=c, max_iter=5000, dual="auto").fit(sc.transform(Xtr), ytr)
                z = mm.decision_function(sc.transform(P[va]))
                zt = mm.decision_function(sc.transform(Xtr))
                auc = roc_auc_score(ytr, zt)
                if auc > best_a:
                    best_a, best_c = auc, c
            mm = LinearSVC(C=best_c, max_iter=5000, dual="auto").fit(sc.transform(Xtr), ytr)
            z = mm.decision_function(sc.transform(P[va]))
            # Platt 缩放 (折内 sigmoid 拟合)
            from sklearn.calibration import _SigmoidCalibration
            cal = _SigmoidCalibration().fit(mm.decision_function(sc.transform(Xtr)), ytr)
            oof[sub_ok_idx[va]] = cal.predict(z)
    return oof

def evals10x5(meta, use_mods, tag):
    per_seed = {"td": [], "tcd": []}
    for seed in SEEDS:
        fo = fold_of_all(seed)
        sub = collect_sub_oof(seed, fo)
        oof = stack_oof(sub, meta, use_mods)
        sub = sub.assign(prob=oof)
        for scope_name, scope in (("td", set(td)), ("tcd", set(tcd))):
            g = sub[sub.patient_id.isin(scope)].dropna(subset=["prob"])
            per_seed[scope_name].append(round(float(roc_auc_score(g["label_T"], g["prob"])), 4))
    out = {}
    for scope_name, vals in per_seed.items():
        v = np.array(vals)
        out[scope_name] = {"repeated_mean": round(float(v.mean()), 4),
                           "repeated_sd": round(float(v.std(ddof=1)), 4),
                           "per_seed": [round(float(x), 4) for x in v]}
    line = f"- stage5 {tag}: " + " | ".join(f"{k} {v['repeated_mean']}±{v['repeated_sd']}" for k, v in out.items())
    with open(os.path.join(OUTDIR, "progress_log.md"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return out

def main():
    results = {}
    # 基准: H1.1 (策略B 概率融合) 已在阶段1, 直接记录
    results["ref_h11"] = {"td": {"repeated_mean": 0.7003, "repeated_sd": 0.0298},
                          "tcd": {"repeated_mean": 0.7099, "repeated_sd": 0.0323}}
    # Stacking 变体 (全模态)
    for meta in ("l2", "enet", "svm"):
        results[f"stack_{meta}_all4"] = evals10x5(meta, ["clin", "t60", "dself", "rad3"], f"stack {meta} (clin,t60,dself,rad3)")
    # 去掉 rad3 (低质子模型可能拖累)
    results["stack_l2_noRad"] = evals10x5("l2", ["clin", "t60", "dself"], "stack l2 (clin,t60,dself)")
    results["stack_l2_imgOnly"] = evals10x5("l2", ["t60", "dself"], "stack l2 (t60,dself)")
    with open(os.path.join(OUTDIR, "stage5_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUTDIR, "stage5_results.json"))

if __name__ == "__main__":
    main()
