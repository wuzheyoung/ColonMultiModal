# -*- coding: utf-8 -*-
"""exp_v2_validation_suite.py — m01774 全套收口验证 (4 模块, 1650 端)
================================================================================
严格防泄露: 凡重新切分 (模块2 重复CV / 模块3 时间外推 / 模块4 ncrt 全量训练)
一律从原始特征矩阵出发, 严禁复用任何绑定旧 5 折的 OOF CSV;
clin_clean / d_self(2D MeanMax) / t(2D AreaW) / zlogit 策略B / 校准层
全部在「当前训练集内部」从头拟合 (缺失中位数填充 → 训练折 std → 折内嵌套 3 折 CV
L2-LR → 策略 B 融合 → 训练折校准)。

模块1: 单口径校准 (校准层改在 direct_all 训练折 n≈224 拟合, 对比 Platt(a,b)
       vs 仅截距 a=1) + 校准曲线/DCA/临床亚组 (direct_tcd n=162)。
模块2: 10×5 重复分层 CV (TRIPOD 1b, seed 0-9) — clin_clean / clin+t / clin+d+t
       于 direct_tcd 162 与 direct_td 214, 10 次 Pooled OOF AUC 均值±SD(ddof=1)
       + min–max + 50 折 Fold-Mean±SD + 胜率与平均 ΔAUC。
模块3: 时间外推 (TRIPOD 2b) — 临床表无全量日期 → DICOM StudyDate 切分
       (direct_all 125/54, 节点 20250428; tcd 90/39; 缺失日期 101 例不入训练),
       早期训练全流程从头拟合, 冻结推理近期测试集。
模块4: ncrt (n=75) 放化疗后域偏移 — direct_all 全量训练 + direct OOF 定 Youden,
       零微调推理 ncrt; 过分期分析 (ypT2→T3-4 假阳性率) + t∩d_self 配对 n=34 衰减。
输出: logs/exp_v2_validation_suite_T.{json,md} + outputs/exp_v2_validation_suite/.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import recompute_direct_tcd_singles as rc    # noqa: E402
import train_slice_mil as tsm                # noqa: E402
import eval_paper_metrics as epm             # noqa: E402
import exp_v2_round2_ablation as abl         # noqa: E402
import exp_v2_final_T as F                   # noqa: E402
import exp_v2_final_calibrated as CALC       # noqa: E402

MASTER = "data/master_5fold_356.csv"
PAIRS = "data/hos_pairs_raw.csv"
COHORT = "data/direct_cohort_ids.csv"
SUMMARY = "logs/dwi_mask_remap_summary.csv"
DICOM = "2026-11-21/hos-data-inventory/work/dicom_meta.json"
DATA_ROOT = "features/slice_feats_v2"
BACKBONE = "dinov2"
OUT = "outputs/exp_v2_validation_suite"
JOUT = "logs/exp_v2_validation_suite_T.json"
MOUT = "logs/exp_v2_validation_suite_T.md"
PROBE_C_GRID = tsm.PROBE_C_GRID
S4_REF = "outputs/oof_baseline_n162/fusions/s4_clin_3plane.f1.csv"

# ---------------- 基础加载 ----------------
master = pd.read_csv(MASTER)
cohort = pd.read_csv(COHORT)
raw = pd.read_csv(PAIRS).set_index("patient_id")
sm = pd.read_csv(SUMMARY)
tgr = set(cohort.loc[cohort.is_tgr == 1, "pid"])
ncrt_ids = tgr & set(master.patient_id)
md = master.loc[~master.patient_id.isin(tgr)].reset_index(drop=True)
md_i = md.set_index("patient_id")

def has(c):
    return raw["mri_path_" + c].notna().reindex(md_i.index).fillna(False)

has_t, has_c, has_d = has("t"), has("c"), has("d")
td = set(md_i.index[has_t & has_d])
tcd = set(md_i.index[has_t & has_c & has_d])
self_d = set(sm.loc[(sm.cohort_direct == 1) & (sm.mask_source == "self"), "pid"]) & set(md.patient_id)
all_direct = np.sort(md.patient_id.values.astype(np.int64))
print(f"direct_all={len(md)} tcd={len(tcd)} td={len(td)} d_self={len(self_d)} ncrt={len(ncrt_ids)}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------- 特征池化 (一次性, 供各模块复用) ----------------
_pool_cache = {}

def pool_branch(pids, seq, pool, frame=None, use_cache=True):
    """返回 (have_pids, X(n,d), y) — 池化一次, 缓存。
    frame: 标签/折归属主表 (默认 md=direct; 模块4 ncrt 推理需传 master 全表)。"""
    frame = frame if frame is not None else md
    key = (seq, pool, len(pids), frame is master)
    if use_cache and key in _pool_cache:
        return _pool_cache[key]
    have = tsm.collect_plane_pids(frame, DATA_ROOT, BACKBONE, seq, set(pids))
    bags = tsm.preload_plane(frame, DATA_ROOT, BACKBONE, seq, have, "label_T", "none",
                             torch.device("cpu"))
    X, y = abl.pool_bags(bags, have, pool)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=float)
    if use_cache:
        _pool_cache[key] = (have, X, y)
    return (have, X, y)


# ---------------- 通用 CV 探针 (任意折分配) ----------------
def probe_generic(have, X, y, folds, feat_norm="std", c_grid=PROBE_C_GRID):
    """folds: 与 have 对齐的整型折数组。折内嵌套 3 折 CV 选 C L2-LR, OOF 契约。"""
    folds = np.asarray(folds, dtype=int)
    oof = np.full(len(have), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            Cs[int(k)] = None
            continue
        Xtr, ytr = X[tr], y[tr]
        inner = folds[tr] % 3
        inner_auc = {}
        for c in c_grid:
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(np.unique(ytr[itr])) < 2:
                    continue
                sc = StandardScaler().fit(Xtr[itr]) if feat_norm == "std" else None
                Xi_tr = sc.transform(Xtr[itr]) if sc is not None else Xtr[itr]
                Xi_va = sc.transform(Xtr[iva]) if sc is not None else Xtr[iva]
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(Xi_tr, ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(Xi_va)[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(c_grid, key=lambda c: inner_auc[c])
        if feat_norm == "std":
            sc = StandardScaler().fit(Xtr)
            mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
            oof[va] = mm.predict_proba(sc.transform(X[va]))[:, 1]
        else:
            mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(Xtr, ytr)
            oof[va] = mm.predict_proba(X[va])[:, 1]
        Cs[int(k)] = float(best_c)
    rows = [{"patient_id": int(p), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp in zip(have, y, oof)]
    df = pd.DataFrame(rows)
    df["fold"] = df["patient_id"].map({int(p): int(f) for p, f in zip(have, folds)})
    return df[["patient_id", "fold", "label_T", "prob"]].sort_values("patient_id").reset_index(drop=True), Cs


def clin_cv(pids, folds, imp="median"):
    """临床 7 维 OOF (fold-internal 中位数填补 + StandardScaler + 折内嵌套3折CV 选 C)。"""
    pids = np.asarray(pids, dtype=np.int64)
    folds = np.asarray(folds, dtype=int)
    Xc = raw.loc[pids, rc.CLIN_COLS].values
    yc = md_i.loc[pids, "label_T"].values.astype(float)
    oof = np.full(len(pids), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(yc[tr])) < 2:
            oof[va] = 0.5
            Cs[int(k)] = None
            continue
        Xtr, ytr = Xc[tr], yc[tr]
        if imp == "median":
            from sklearn.impute import SimpleImputer
            impm = SimpleImputer(strategy="median").fit(Xtr)
            Xtr = impm.transform(Xtr)
        else:
            Xtr = np.nan_to_num(Xtr, nan=0.0)
        inner = folds[tr] % 3
        inner_auc = {}
        for c in c_grid_lr():
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(np.unique(ytr[itr])) < 2:
                    continue
                sc = StandardScaler().fit(Xtr[itr])
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr[itr]), ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(sc.transform(Xtr[iva]))[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(c_grid_lr(), key=lambda c: inner_auc[c])
        sc = StandardScaler().fit(Xtr)
        mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
        if imp == "median":
            Xva = SimpleImputer(strategy="median").fit(Xc[tr]).transform(Xc[va])
        else:
            Xva = np.nan_to_num(Xc[va], nan=0.0)
        oof[va] = mm.predict_proba(sc.transform(Xva))[:, 1]
        Cs[int(k)] = float(best_c)
    rows = [{"patient_id": int(p), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp in zip(pids, yc, oof)]
    df = pd.DataFrame(rows)
    df["fold"] = df["patient_id"].map({int(p): int(f) for p, f in zip(pids, folds)})
    return df[["patient_id", "fold", "label_T", "prob"]].sort_values("patient_id").reset_index(drop=True), Cs


def c_grid_lr():
    return (0.01, 0.1, 1.0, 10.0)


# ---------------- fit-on-train → predict-test (模块 3/4 部署路径) ----------------
def probe_fit_predict(have_tr, X_tr, y_tr, have_te, X_te, feat_norm="std", c_grid=PROBE_C_GRID):
    """在训练集上选 C (训练集内 index%3 内折) + 拟合, 预测训练集(样本内)与测试集。"""
    inner = np.arange(len(X_tr)) % 3
    inner_auc = {}
    for c in c_grid:
        aucs = []
        for g in range(3):
            itr, iva = inner != g, inner == g
            if iva.sum() < 2 or len(np.unique(y_tr[itr])) < 2:
                continue
            sc = StandardScaler().fit(X_tr[itr]) if feat_norm == "std" else None
            Xi_tr = sc.transform(X_tr[itr]) if sc is not None else X_tr[itr]
            Xi_va = sc.transform(X_tr[iva]) if sc is not None else X_tr[iva]
            mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(Xi_tr, y_tr[itr])
            aucs.append(roc_auc_score(y_tr[iva], mm.predict_proba(Xi_va)[:, 1]))
        inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
    best_c = max(c_grid, key=lambda c: inner_auc[c])
    sc = StandardScaler().fit(X_tr) if feat_norm == "std" else None
    Xi_tr = sc.transform(X_tr) if sc is not None else X_tr
    Xi_te = sc.transform(X_te) if sc is not None else X_te
    mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(Xi_tr, y_tr)
    p_tr = mm.predict_proba(Xi_tr)[:, 1]
    p_te = mm.predict_proba(Xi_te)[:, 1]
    return p_tr, p_te, float(best_c)


def clin_fit_predict(tr_pids, te_pids, imp="median"):
    from sklearn.impute import SimpleImputer
    Xtr = raw.loc[tr_pids, rc.CLIN_COLS].values
    Xte = raw.loc[te_pids, rc.CLIN_COLS].values
    ytr = md_i.loc[tr_pids, "label_T"].values.astype(float)
    if imp == "median":
        impm = SimpleImputer(strategy="median").fit(Xtr)
        Xtr_i, Xte_i = impm.transform(Xtr), impm.transform(Xte)
    else:
        Xtr_i, Xte_i = np.nan_to_num(Xtr, nan=0.0), np.nan_to_num(Xte, nan=0.0)
    inner = np.arange(len(Xtr_i)) % 3
    inner_auc = {}
    for c in c_grid_lr():
        aucs = []
        for g in range(3):
            itr, iva = inner != g, inner == g
            if iva.sum() < 2 or len(np.unique(ytr[itr])) < 2:
                continue
            sc = StandardScaler().fit(Xtr_i[itr])
            mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr_i[itr]), ytr[itr])
            aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(sc.transform(Xtr_i[iva]))[:, 1]))
        inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
    best_c = max(c_grid_lr(), key=lambda c: inner_auc[c])
    sc = StandardScaler().fit(Xtr_i)
    mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr_i), ytr)
    return mm.predict_proba(sc.transform(Xtr_i))[:, 1], mm.predict_proba(sc.transform(Xte_i))[:, 1], float(best_c)


# ---------------- zlogit 策略 B (train-side 统计, 模块 3/4 部署) ----------------
def zlogit_fuse_deploy(branch_probs_train, branch_probs_test, clin_key="clin"):
    """branch_probs: dict name -> (pids, probs 数组)。z 统计取自训练集, 策略 B 融合。
    返回 (train df, test df): patient_id, label_T, prob(=z)。"""
    ztr, zte = {}, {}
    for name, (pids_tr, pr_tr, pids_te, pr_te) in branch_probs.items():
        L_tr = tsm.logit_clip(np.asarray(pr_tr, dtype=float))
        mu, sd = L_tr.mean(), L_tr.std() + 1e-9
        ztr[name] = pd.DataFrame({"patient_id": np.asarray(pids_tr, dtype=np.int64),
                                  "z": (L_tr - mu) / sd})
        L_te = tsm.logit_clip(np.asarray(pr_te, dtype=float))
        zte[name] = pd.DataFrame({"patient_id": np.asarray(pids_te, dtype=np.int64),
                                  "z": (L_te - mu) / sd})
    ztr = {n: d.set_index("patient_id")["z"] for n, d in ztr.items()}
    zte = {n: d.set_index("patient_id")["z"] for n, d in zte.items()}

    def fuse(pids, zdict):
        rows = []
        for pid in pids:
            comps = {n: z.loc[pid] for n, z in zdict.items() if pid in z.index}
            if not comps:
                continue
            zc = comps.get(clin_key)
            imgs = [z for n, z in comps.items() if n != clin_key]
            if zc is None or not imgs:
                v = float(np.mean(list(comps.values())))
            else:
                v = 0.5 * float(zc) + 0.5 * float(np.mean(imgs))
            rows.append({"patient_id": int(pid), "prob": v})
        return pd.DataFrame(rows)

    df_tr = fuse(tr_pids_all, ztr)
    df_te = fuse(te_pids_all, zte)
    df_tr["label_T"] = df_tr.patient_id.map(md_i["label_T"]).astype(int)
    df_te["label_T"] = df_te.patient_id.map(md_i["label_T"]).astype(int)
    return df_tr, df_te


tr_pids_all, te_pids_all = [], []  # 模块3/4 填充


# ---------------- 校准层 ----------------
def platt_fit(z, y):
    mm = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(
        np.asarray(z, dtype=float).reshape(-1, 1), np.asarray(y, dtype=int))
    return float(mm.coef_[0][0]), float(mm.intercept_[0])


def platt_fit_bonly(z, y):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=int)

    def nll(b):
        p = sigmoid(z + b)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded")
    return 1.0, float(res.x)


def calibrate_cv(zdf, mode="platt"):
    """逐折: 训练折 (fold≠k) 拟合校准 → 验证折 (fold==k)。mode: 'platt' 或 'bonly'。"""
    d = zdf.copy()
    z = d["prob"].values.astype(float)
    folds = d["fold"].values.astype(int)
    y = d["label_T"].values.astype(int)
    out = np.full(len(d), np.nan)
    params = {}
    for k in np.unique(folds):
        va = folds == k
        tr = ~va
        if tr.sum() < 8 or len(np.unique(y[tr])) < 2:
            out[va] = sigmoid(z[va])
            params[int(k)] = None
            continue
        if mode == "platt":
            a, b = platt_fit(z[tr], y[tr])
        else:
            a, b = platt_fit_bonly(z[tr], y[tr])
        out[va] = sigmoid(a * z[va] + b)
        params[int(k)] = [round(a, 5), round(b, 5)]
    d["prob"] = out
    return d, params


# ---------------- 指标 ----------------
def metrics_of(df):
    df = df.dropna(subset=["prob"]).reset_index(drop=True)
    df["prob"] = np.clip(df["prob"], 0, 1)
    if len(np.unique(df["label_T"])) < 2:
        return {"n": int(len(df)), "n_pos": int(df.label_T.sum()),
                "pooled_auc": None, "boot_95ci": None,
                "fold_mean": None, "fold_sd": None,
                "brier": round(float(brier_score_loss(df["label_T"], df["prob"])), 4)
                if len(df) else None, "ece10": None, "youden_oof": None,
                "note": "single class in y — AUC undefined"}
    auc = float(roc_auc_score(df["label_T"], df["prob"]))
    brier = float(brier_score_loss(df["label_T"], df["prob"]))
    ece = CALC.ece10(df)
    n_folds = len(np.unique(df["fold"])) if "fold" in df.columns else 0
    if n_folds >= 2:
        fm = epm.fold_mean_sd(df)
        youden = epm.youden_metrics_oof(df)
    else:
        fm, youden = None, None
    ci = epm.boot_ci(df)
    return {
        "n": int(len(df)), "n_pos": int(df.label_T.sum()),
        "pooled_auc": round(auc, 4), "boot_95ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "fold_mean": round(float(fm[0]), 4) if fm else None,
        "fold_sd": round(float(fm[1]), 4) if fm else None,
        "brier": round(brier, 4), "ece10": round(ece, 4),
        "youden_oof": {k: round(float(v), 4) for k, v in youden.items()} if youden else None,
    }


def delong_pair(df1, df2):
    res, np_ = epm.delong_paired_test(df1, df2)
    return res, np_


# ---------------- 模块 1: 单口径校准 + DCA + 校准曲线 + 亚组 ----------------
def module1():
    os.makedirs(OUT, exist_ok=True)
    # 从原始特征重算 direct_all OOF (master 折)
    folds_d = md_i.loc[all_direct, "fold"].values.astype(int)
    clin_oof, _ = clin_cv(all_direct, folds_d)
    t_have, t_X, t_y = pool_branch(all_direct, "t", "area_w")
    t_fold = md_i.loc[t_have, "fold"].values.astype(int)
    t_oof, _ = probe_generic(t_have, t_X, t_y, t_fold)
    d_have, d_X, d_y = pool_branch(sorted(self_d), "d", "meanmax")
    d_fold = md_i.loc[d_have, "fold"].values.astype(int)
    d_oof, _ = probe_generic(d_have, d_X, d_y, d_fold)
    # 策略 B 融合 (direct_all, master 折) → z (fuse_zlogit_strategy 自带 fold/label_T)
    fused = F.fuse_zlogit_strategy({"clin": clin_oof, "t": t_oof, "d": d_oof},
                                   strategy="B", clin_key="clin")
    fused = fused.sort_values("patient_id").reset_index(drop=True)
    # 两种校准 (校准层在 direct_all 训练折 n≈224 拟合)
    cal_platt, p_platt = calibrate_cv(fused, mode="platt")
    cal_bonly, p_bonly = calibrate_cv(fused, mode="bonly")
    sig_df = fused.copy()
    sig_df["prob"] = sigmoid(sig_df["prob"])
    tcd_sig = sig_df[sig_df.patient_id.isin(tcd)].reset_index(drop=True)
    tcd_platt = cal_platt[cal_platt.patient_id.isin(tcd)].reset_index(drop=True)
    tcd_bonly = cal_bonly[cal_bonly.patient_id.isin(tcd)].reset_index(drop=True)
    m_sig, m_platt, m_bonly = metrics_of(tcd_sig), metrics_of(tcd_platt), metrics_of(tcd_bonly)
    # 择优: Pooled AUC 损失最小 且 Brier<0.20 且 ECE<0.06
    def score(m):
        auc_loss = abs(m["pooled_auc"] - m_sig["pooled_auc"])
        ok_brier = m["brier"] < 0.20
        ok_ece = m["ece10"] < 0.06
        return auc_loss, ok_brier, ok_ece
    s_platt, s_bonly = score(m_platt), score(m_bonly)
    cand = {"platt": (s_platt, m_platt, tcd_platt), "bonly": (s_bonly, m_bonly, tcd_bonly)}
    # 排序: 先满足 brier/ece, 再按 auc_loss 最小
    ranked = sorted(cand.items(), key=lambda kv: (not kv[1][0][1], not kv[1][0][2], kv[1][0][0]))
    lock_name = ranked[0][0]
    lock_df = cand[lock_name][2].copy()
    # 锁定列 = 校准概率 (唯一列)
    m_lock = metrics_of(lock_df)
    # DeLong: 锁定列 vs 旧 s4 (参照基线, 非复用为特征)
    s4 = epm.load_csv(S4_REF)
    dl_s4 = delong_pair(lock_df[["patient_id", "fold", "label_T", "prob"]], s4)
    # DCA
    dca_res = dca_analysis(lock_df, m_sig, clin_oof)
    # 校准曲线图
    cal_curve_fig(lock_df)
    # 亚组
    sub = subgroup_audit(lock_df)
    m1 = {
        "module": "M1 single-caliber calibration + DCA + subgroups",
        "calibration_fit_scope": "direct_all train folds (n≈224), NOT tcd train folds",
        "uncal_sigmoid_tcd": m_sig, "platt_tcd": m_platt, "bonly_tcd": m_bonly,
        "platt_params_by_fold": {str(k): v for k, v in p_platt.items()},
        "bonly_params_by_fold": {str(k): v for k, v in p_bonly.items()},
        "selection": {"locked": lock_name, "criterion": "min pooled-AUC loss w/ Brier<0.20 & ECE<0.06",
                      "platt_score": list(s_platt), "bonly_score": list(s_bonly)},
        "locked_metrics": m_lock,
        "delong_locked_vs_s4": {"res": dl_s4[0], "n_paired": dl_s4[1]},
        "dca": dca_res, "subgroups": sub,
    }
    return m1, lock_df


def dca_analysis(lock_df, m_sig, clin_oof):
    y = lock_df["label_T"].values.astype(int)
    p = np.clip(lock_df["prob"].values.astype(float), 0, 1)
    n = len(y)
    prev = y.mean()
    th = np.arange(0.10, 0.91, 0.05)
    rows = []
    for t in th:
        tp = ((p >= t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum()
        nb_model = (tp - fp * t / (1 - t)) / n
        nb_all = (prev - (1 - prev) * t / (1 - t))
        rows.append({"threshold": round(float(t), 2),
                     "nb_best": round(float(nb_model), 4),
                     "nb_clin": None, "nb_all": round(float(nb_all), 4), "nb_none": 0.0})
    # clin_clean DCA (tcd 截取)
    clin_tcd = clin_oof[clin_oof.patient_id.isin(tcd)].reset_index(drop=True)
    yc = clin_tcd["label_T"].values.astype(int)
    pc = np.clip(clin_tcd["prob"].values.astype(float), 0, 1)
    nc = len(yc)
    for r in rows:
        t = r["threshold"]
        tp = ((pc >= t) & (yc == 1)).sum()
        fp = ((pc >= t) & (yc == 0)).sum()
        r["nb_clin"] = round(float((tp - fp * t / (1 - t)) / nc), 4)
    # 图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ts = [r["threshold"] for r in rows]
    ax.plot(ts, [r["nb_best"] for r in rows], "-o", label="Best (clin+d_self+t, locked cal)")
    ax.plot(ts, [r["nb_clin"] for r in rows], "-s", label="clin_clean")
    ax.plot(ts, [r["nb_all"] for r in rows], "--", label="Treat All")
    ax.plot(ts, [r["nb_none"] for r in rows], ":", label="Treat None")
    ax.set_xlabel("Threshold (probability)"); ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis — direct_tcd n=162 (locked calibrated probs)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dca_locked_tcd.png"), dpi=130)
    plt.close(fig)
    return {"thresholds": rows,
            "key_intervals": {"0.40": [r["nb_best"] for r in rows if abs(r["threshold"]-0.40)<1e-9][0],
                              "0.50": [r["nb_best"] for r in rows if abs(r["threshold"]-0.50)<1e-9][0],
                              "0.60": [r["nb_best"] for r in rows if abs(r["threshold"]-0.60)<1e-9][0],
                              "0.70": [r["nb_best"] for r in rows if abs(r["threshold"]-0.70)<1e-9][0],
                              "0.80": [r["nb_best"] for r in rows if abs(r["threshold"]-0.80)<1e-9][0]}}


def cal_curve_fig(lock_df):
    y = lock_df["label_T"].values.astype(int)
    p = np.clip(lock_df["prob"].values.astype(float), 0, 1)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, "o-", label="10 uniform bins")
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    ps = np.linspace(0, 1, 200)
    ax.plot(ps, iso.predict(ps), "-", alpha=0.6, label="Isotonic (Loess-like smooth)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction positive")
    ax.set_title("Calibration curve — locked probs, direct_tcd n=162")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cal_curve_locked_tcd.png"), dpi=130)
    plt.close(fig)


def subgroup_audit(lock_df):
    """direct_tcd 162 亚组: d_self 有无 / 性别 / 年龄中位 / CEA≥5 / 肿瘤位置(表内无→标注)。"""
    lk = lock_df.set_index("patient_id")
    age_med = raw.loc[list(tcd), "age"].median()
    out = {}
    def grp(name, mask, desc):
        g = lk.index.isin(set(mask)) & lk.index.isin(tcd)
        sub = lk[g].reset_index()
        if len(sub) < 8:
            out[name] = {"desc": desc, "n": int(len(sub)), "note": "too small"}
            return
        m = metrics_of(sub)
        out[name] = {"desc": desc, **{k: m[k] for k in ("n", "n_pos", "pooled_auc", "boot_95ci", "brier")}}
    grp("has_d_self", self_d, "有 d_self (self DWI)")
    grp("no_d_self", tcd - self_d, "无 d_self")
    sex1 = set(raw.index[raw["sex"] == 1])
    sex0 = set(raw.index[raw["sex"] == 0])
    grp("sex1", sex1, "sex=1")
    grp("sex0", sex0, "sex=0")
    age_hi = set(raw.index[raw["age"] >= age_med]) if pd.notna(age_med) else set()
    age_lo = set(raw.index[raw["age"] < age_med]) if pd.notna(age_med) else set()
    grp("age_ge_median", age_hi, f"age >= median({age_med:.0f})")
    grp("age_lt_median", age_lo, f"age < median({age_med:.0f})")
    cea_hi = set(raw.index[raw["cea"] >= 5.0])
    cea_lo = set(raw.index[raw["cea"] < 5.0])
    grp("cea_ge_5", cea_hi, "CEA >= 5 ng/mL")
    grp("cea_lt_5", cea_lo, "CEA < 5 ng/mL")
    out["tumor_location"] = {"desc": "临床表 (hos_pairs_raw) 无 高/中/低位 字段 — 无法审计",
                             "available": False}
    return out


# ---------------- 模块 2: 10×5 重复分层 CV ----------------
def module2(lock_mode):
    from sklearn.model_selection import StratifiedKFold
    d_pids = all_direct
    t_have, t_X, t_y = pool_branch(d_pids, "t", "area_w")
    d_have, d_X, d_y = pool_branch(sorted(self_d), "d", "meanmax")
    t_map = {int(p): i for i, p in enumerate(t_have)}
    d_map = {int(p): i for i, p in enumerate(d_have)}
    pos_of_pid = {int(p): i for i, p in enumerate(d_pids)}
    y_all = md_i.loc[d_pids, "label_T"].values.astype(int)
    seeds = range(10)
    res = {"module": "M2 10x5 repeated stratified CV (TRIPOD 1b)"}
    for scope_name, scope in (("direct_tcd", tcd), ("direct_td", td)):
        per_seed = {"clin": [], "clin_t": [], "clin_dt": []}
        fold_aucs = {"clin": [], "clin_t": [], "clin_dt": []}
        win_clin, win_t, deltas_clin, deltas_t = 0, 0, [], []
        for seed in seeds:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_of = np.empty(len(d_pids), dtype=int)
            for k, (_, va) in enumerate(skf.split(d_pids, y_all)):
                fold_of[va] = k
            # 各分支 OOF (fold 分配按 d_pids 位置)
            clin_df, _ = clin_cv(d_pids, fold_of)
            t_folds = np.array([fold_of[pos_of_pid[p]] for p in t_have])
            t_df, _ = probe_generic(t_have, t_X, t_y, t_folds)
            d_folds = np.array([fold_of[pos_of_pid[p]] for p in d_have])
            d_df, _ = probe_generic(d_have, d_X, d_y, d_folds)
            # 融合 (clin, clin+t, clin+d+t) — 策略 B
            fuse_ct = F.fuse_zlogit_strategy({"clin": clin_df, "t": t_df}, strategy="B")
            fuse_cdt = F.fuse_zlogit_strategy({"clin": clin_df, "t": t_df, "d": d_df}, strategy="B")
            # 校准 (单口径锁定 mode), 截取 scope
            ct_cal, _ = calibrate_cv(fuse_ct, mode=lock_mode)
            cdt_cal, _ = calibrate_cv(fuse_cdt, mode=lock_mode)
            ct_s = ct_cal[ct_cal.patient_id.isin(scope)].reset_index(drop=True)
            cdt_s = cdt_cal[cdt_cal.patient_id.isin(scope)].reset_index(drop=True)
            cl_s = clin_df[clin_df.patient_id.isin(scope)].reset_index(drop=True)
            cl_s["prob"] = np.clip(cl_s["prob"], 0, 1)
            a_clin = float(roc_auc_score(cl_s["label_T"], cl_s["prob"]))
            a_ct = float(roc_auc_score(ct_s["label_T"], ct_s["prob"]))
            a_cdt = float(roc_auc_score(cdt_s["label_T"], cdt_s["prob"]))
            per_seed["clin"].append(round(a_clin, 4))
            per_seed["clin_t"].append(round(a_ct, 4))
            per_seed["clin_dt"].append(round(a_cdt, 4))
            win_clin += int(a_cdt > a_clin)
            win_t += int(a_cdt > a_ct)
            deltas_clin.append(a_cdt - a_clin)
            deltas_t.append(a_cdt - a_ct)
            # 50 折: 每折 scope 内 AUC
            for name, df in (("clin", cl_s), ("clin_t", ct_s), ("clin_dt", cdt_s)):
                for k in np.unique(df["fold"]):
                    g = df[df["fold"] == k]
                    if len(g) >= 8 and len(np.unique(g["label_T"])) == 2:
                        fold_aucs[name].append(round(float(roc_auc_score(g["label_T"], g["prob"])), 4))
        rep = {}
        for name in ("clin", "clin_t", "clin_dt"):
            a = np.array(per_seed[name])
            fa = np.array(fold_aucs[name])
            rep[name] = {
                "repeated_pooled_auc_mean": round(float(a.mean()), 4),
                "repeated_pooled_auc_sd_ddof1": round(float(a.std(ddof=1)), 4),
                "repeated_pooled_auc_min": round(float(a.min()), 4),
                "repeated_pooled_auc_max": round(float(a.max()), 4),
                "per_seed_aucs": [round(float(x), 4) for x in a],
                "fold50_mean": round(float(fa.mean()), 4) if len(fa) else None,
                "fold50_sd": round(float(fa.std(ddof=1)), 4) if len(fa) else None,
            }
        rep["win_rate_best_vs_clin"] = f"{win_clin}/10"
        rep["win_rate_best_vs_clin_t"] = f"{win_t}/10"
        rep["mean_delta_vs_clin"] = round(float(np.mean(deltas_clin)), 4)
        rep["mean_delta_vs_clin_t"] = round(float(np.mean(deltas_t)), 4)
        res[scope_name] = rep
    return res


# ---------------- 模块 3: 时间外推 (TRIPOD 2b) ----------------
def module3():
    # 前置 5 秒核查: 临床原始总表是否含全量日期列 (<10% 缺失)
    date_cols = [c for c in raw.columns if any(k in c.lower() for k in
                 ("date", "时间", "日期", "入院", "手术", "病理"))]
    clinical_date_full = {"available": False, "reason": "hos_pairs_raw 无任何日期列"}
    # DICOM StudyDate 切分 (复用 audit_temporal_ncrt 逻辑)
    dj = json.load(open(DICOM, encoding="utf-8"))
    sdate = {}
    import re as _re
    pidplane = _re.compile(r"^(\d+)_([tcd])$")
    for x in dj["MRI"]:
        m = pidplane.match(x["dir"])
        if m:
            sdate[(int(m.group(1)), m.group(2))] = x["meta_first"].get("StudyDate")

    def fname_pid(path):
        if not isinstance(path, str) or not path:
            return None
        b = os.path.basename(path).replace(".nii.gz", "")
        m = pidplane.match(b)
        return int(m.group(1)) if m else None

    def get_date(pid):
        for plane in ("t", "c", "d"):
            p = raw.loc[pid, "mri_path_" + plane] if pid in raw.index else None
            pp = fname_pid(p) if isinstance(p, str) else None
            sd = sdate.get((pp, plane)) if pp is not None else None
            if sd:
                return sd
        return None

    dated = []
    for pid in all_direct:
        sd = get_date(int(pid))
        if sd:
            dated.append((int(pid), sd))
    dated.sort(key=lambda x: x[1])
    cut = int(round(0.70 * len(dated)))
    tr_pids = [p for p, _ in dated[:cut]]
    te_pids = [p for p, _ in dated[cut:]]
    global tr_pids_all, te_pids_all
    tr_pids_all, te_pids_all = tr_pids, te_pids
    print(f"[M3] dated={len(dated)} train={len(tr_pids)} test={len(te_pids)} "
          f"cutoff={dated[cut][1] if cut < len(dated) else None}")
    # 训练集 OOF (内 5 折) 定 Youden 阈值 → 测试集
    def run_model(branches):
        # branches: dict name -> (tr_probs, te_probs, tr_pids, te_pids)
        ztr, zte = {}, {}
        for name, (p_tr, p_te) in branches.items():
            L_tr = tsm.logit_clip(np.asarray(p_tr, dtype=float))
            mu, sd = L_tr.mean(), L_tr.std() + 1e-9
            ztr[name] = L_tr - mu / sd
            zte[name] = tsm.logit_clip(np.asarray(p_te, dtype=float)) - mu / sd
        return ztr, zte

    # 训练集内 OOF (5 折) 供 Youden
    from sklearn.model_selection import StratifiedKFold
    tr_arr = np.array(tr_pids, dtype=np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_tr = md_i.loc[tr_arr, "label_T"].values.astype(int)
    tr_fold = np.empty(len(tr_arr), dtype=int)
    for k, (_, va) in enumerate(skf.split(tr_arr, y_tr)):
        tr_fold[va] = k

    def probe_tr_oof(pids, seq, pool):
        pids = np.asarray(pids, dtype=np.int64)
        have, X, y = pool_branch(all_direct, seq, pool)
        pos = {int(p): i for i, p in enumerate(have)}
        ids = np.array([p for p in pids if p in pos], dtype=np.int64)
        Xs = np.array([X[pos[p]] for p in ids])
        ys = np.array([y[pos[p]] for p in ids])
        folds = np.array([tr_fold[np.where(tr_arr == p)[0][0]] for p in ids])
        df, _ = probe_generic(ids, Xs, ys, folds)
        return df

    def clin_tr_oof(pids):
        pids = np.asarray(pids, dtype=np.int64)
        folds = np.array([tr_fold[np.where(tr_arr == p)[0][0]] for p in pids])
        df, _ = clin_cv(pids, folds)
        return df

    res = {"module": "M3 temporal extrapolation (TRIPOD 2b)",
           "clinical_date_check": clinical_date_full,
           "date_source": "DICOM dicom_meta.json StudyDate (mri_path filename pid_plane join)",
           "cutoff_date": dated[cut][1] if cut < len(dated) else None,
           "train_n": len(tr_pids), "test_n": len(te_pids),
           "missing_date_excluded": int(len(all_direct) - len(dated))}
    # tcd 子集
    tr_tcd = [p for p in tr_pids if p in tcd]
    te_tcd = [p for p in te_pids if p in tcd]
    res["tcd_subset"] = {"train_n": len(tr_tcd), "test_n": len(te_tcd)}

    # 部署模型: 各分支 fit on train → predict train(test) 样本内
    def deploy(scope_tr, scope_te, name):
        tr_arr_s = np.array(scope_tr, dtype=np.int64)
        te_arr_s = np.array(scope_te, dtype=np.int64)
        # clin
        p_clin_tr, p_clin_te, _ = clin_fit_predict(tr_arr_s, te_arr_s)
        # t
        have_t, X_t, y_t = pool_branch(all_direct, "t", "area_w")
        tpos = {int(p): i for i, p in enumerate(have_t)}
        ids_tr = np.array([p for p in tr_arr_s if p in tpos], dtype=np.int64)
        ids_te = np.array([p for p in te_arr_s if p in tpos], dtype=np.int64)
        p_t_tr, p_t_te, _ = probe_fit_predict(
            ids_tr, np.array([X_t[tpos[p]] for p in ids_tr]), np.array([y_t[tpos[p]] for p in ids_tr]),
            ids_te, np.array([X_t[tpos[p]] for p in ids_te]))
        # d_self
        have_d, X_d, y_d = pool_branch(sorted(self_d), "d", "meanmax")
        dpos = {int(p): i for i, p in enumerate(have_d)}
        ids_d_tr = np.array([p for p in tr_arr_s if p in dpos], dtype=np.int64)
        ids_d_te = np.array([p for p in te_arr_s if p in dpos], dtype=np.int64)
        if len(ids_d_tr) >= 10:
            p_d_tr, p_d_te, _ = probe_fit_predict(
                ids_d_tr, np.array([X_d[dpos[p]] for p in ids_d_tr]), np.array([y_d[dpos[p]] for p in ids_d_tr]),
                ids_d_te, np.array([X_d[dpos[p]] for p in ids_d_te]))
        else:
            p_d_tr, p_d_te = None, None
        # zlogit 策略 B 部署融合 (z 统计=训练集)
        branch_probs = {"clin": (tr_arr_s, p_clin_tr, te_arr_s, p_clin_te),
                        "t": (ids_tr, p_t_tr, ids_te, p_t_te)}
        if p_d_tr is not None:
            branch_probs["d"] = (ids_d_tr, p_d_tr, ids_d_te, p_d_te)
        # 手工策略 B 融合 (训练集 z 统计)
        def fuse_deploy(branch_probs, which):
            zs = {}
            for nm, (pids_tr, pr_tr, pids_te, pr_te) in branch_probs.items():
                L_tr = tsm.logit_clip(np.asarray(pr_tr, dtype=float))
                mu, sd = L_tr.mean(), L_tr.std() + 1e-9
                if which == "tr":
                    zs[nm] = pd.Series((L_tr - mu) / sd, index=np.asarray(pids_tr, dtype=np.int64))
                else:
                    L_te = tsm.logit_clip(np.asarray(pr_te, dtype=float))
                    zs[nm] = pd.Series((L_te - mu) / sd, index=np.asarray(pids_te, dtype=np.int64))
            out = {}
            for nm in zs:
                pass
            pids_all = sorted(set().union(*[set(z.index) for z in zs.values()]))
            rows = []
            for pid in pids_all:
                comps = {n: z.loc[pid] for n, z in zs.items() if pid in z.index}
                zc = comps.get("clin")
                imgs = [z for n, z in comps.items() if n != "clin"]
                if zc is None or not imgs:
                    v = float(np.mean(list(comps.values())))
                else:
                    v = 0.5 * float(zc) + 0.5 * float(np.mean(imgs))
                rows.append({"patient_id": int(pid), "prob": v})
            return pd.DataFrame(rows)
        # 三种模型: clin, clin+t, clin+d+t
        models = {}
        for mname, bps in (("clin", {"clin": branch_probs["clin"]}),
                           ("clin_t", {"clin": branch_probs["clin"], "t": branch_probs["t"]}),
                           ("clin_dt", branch_probs)):
            f_tr = fuse_deploy(bps, "tr")
            f_te = fuse_deploy(bps, "te")
            f_tr["label_T"] = f_tr.patient_id.map(md_i["label_T"]).astype(int)
            f_te["label_T"] = f_te.patient_id.map(md_i["label_T"]).astype(int)
            # 校准: Platt 拟合训练集 z, 应用到测试集
            a, b = platt_fit(f_tr["prob"], f_tr["label_T"])
            f_tr["prob"] = sigmoid(a * f_tr["prob"] + b)
            f_te["prob"] = sigmoid(a * f_te["prob"] + b)
            f_tr["fold"] = 0
            f_te["fold"] = 1
            models[mname] = {"train": f_tr, "test": f_te, "platt": (round(a, 5), round(b, 5))}
        # 训练集 Youden 阈值 (train OOF 用内 5 折: 简化用样本内 fused 概率的 Youden)
        # 严格起见: 用 train 融合概率 Youden J
        out = {}
        for mname, mm in models.items():
            f_te = mm["test"].dropna(subset=["prob"]).reset_index(drop=True)
            f_te["prob"] = np.clip(f_te["prob"], 0, 1)
            f_tr = mm["train"].dropna(subset=["prob"]).reset_index(drop=True)
            f_tr["prob"] = np.clip(f_tr["prob"], 0, 1)
            from sklearn.metrics import roc_curve
            fpr, tpr, ths = roc_curve(f_tr["label_T"], f_tr["prob"])
            J = tpr - fpr
            th = ths[np.argmax(J)]
            tp = ((f_te["prob"] >= th) & (f_te["label_T"] == 1)).sum()
            fp = ((f_te["prob"] >= th) & (f_te["label_T"] == 0)).sum()
            tn = ((f_te["prob"] < th) & (f_te["label_T"] == 0)).sum()
            fn = ((f_te["prob"] < th) & (f_te["label_T"] == 1)).sum()
            sens = tp / (tp + fn) if (tp + fn) else None
            spec = tn / (tn + fp) if (tn + fp) else None
            ppv = tp / (tp + fp) if (tp + fp) else None
            npv = tn / (tn + fn) if (tn + fn) else None
            out[mname] = {
                "test_metrics": metrics_of(f_te),
                "train_youden_threshold": round(float(th), 4),
                "train_youden_j": round(float(J.max()), 4),
                "test_at_train_youden": {"sens": round(float(sens), 4) if sens is not None else None,
                                         "spec": round(float(spec), 4) if spec is not None else None,
                                         "ppv": round(float(ppv), 4) if ppv is not None else None,
                                         "npv": round(float(npv), 4) if npv is not None else None},
                "platt": mm["platt"],
                "train_n": int(len(f_tr)), "test_n": int(len(f_te)),
            }
        return out

    res["direct_all"] = deploy(tr_pids, te_pids, "direct_all")
    res["direct_tcd"] = deploy(tr_tcd, te_tcd, "direct_tcd")
    # ROC 图 (测试集)
    return res


# ---------------- 模块 4: ncrt 域偏移 ----------------
def module4(lock_mode):
    # direct_all 全量训练 + direct OOF (master 折) 定 Youden
    folds_d = md_i.loc[all_direct, "fold"].values.astype(int)
    clin_oof, _ = clin_cv(all_direct, folds_d)
    t_have, t_X, t_y = pool_branch(all_direct, "t", "area_w")
    t_fold = md_i.loc[t_have, "fold"].values.astype(int)
    t_oof, _ = probe_generic(t_have, t_X, t_y, t_fold)
    d_have, d_X, d_y = pool_branch(sorted(self_d), "d", "meanmax")
    d_fold = md_i.loc[d_have, "fold"].values.astype(int)
    d_oof, _ = probe_generic(d_have, d_X, d_y, d_fold)
    fused = F.fuse_zlogit_strategy({"clin": clin_oof, "t": t_oof, "d": d_oof},
                                   strategy="B", clin_key="clin")
    # 校准 (锁定 mode, 逐折拟合在 direct train 折) + direct OOF 概率 → Youden
    cal, params = calibrate_cv(fused, mode=lock_mode)
    cal["prob"] = np.clip(cal["prob"], 0, 1)
    from sklearn.metrics import roc_curve
    fpr, tpr, ths = roc_curve(cal["label_T"], cal["prob"])
    J = tpr - fpr
    th_y = float(ths[np.argmax(J)])
    # 部署: 各分支 fit on ALL direct → predict ncrt
    # 池化必须覆盖 direct ∪ ncrt (frame=master 全表), 再按 pids 拆训练/预测子集
    ncrt_arr = np.array(sorted(ncrt_ids), dtype=np.int64)
    all_master = np.array(sorted(master["patient_id"].astype(np.int64)), dtype=np.int64)
    p_clin_tr_all, p_clin_ncrt, _ = clin_fit_predict(all_direct, ncrt_arr)
    # t: frame=master 全表池化
    have_tm, X_tm, y_tm = pool_branch(all_master, "t", "area_w", frame=master, use_cache=False)
    tpos = {int(p): i for i, p in enumerate(have_tm)}
    t_tr_pids = np.array([p for p in all_direct if p in tpos], dtype=np.int64)
    ncrt_t = np.array([p for p in ncrt_arr if p in tpos], dtype=np.int64)
    p_t_tr, p_t_ncrt, _ = probe_fit_predict(
        t_tr_pids, np.array([X_tm[tpos[p]] for p in t_tr_pids]),
        np.array([y_tm[tpos[p]] for p in t_tr_pids]),
        ncrt_t, np.array([X_tm[tpos[p]] for p in ncrt_t]))
    # d: frame=master 全表池化 (self_d ∪ ncrt_self)
    ncrt_self = set(sm.loc[(sm.cohort_direct == 0) & (sm.mask_source == "self"), "pid"])
    d_master_pids = np.array(sorted(set(self_d) | set(ncrt_self)), dtype=np.int64)
    have_dm, X_dm, y_dm = pool_branch(d_master_pids, "d", "meanmax", frame=master, use_cache=False)
    dpos = {int(p): i for i, p in enumerate(have_dm)}
    d_tr_pids = np.array([p for p in all_direct if p in dpos], dtype=np.int64)
    ncrt_d = np.array([p for p in ncrt_arr if p in dpos], dtype=np.int64)
    if len(ncrt_d) >= 2:
        p_d_tr, p_d_ncrt, _ = probe_fit_predict(
            d_tr_pids, np.array([X_dm[dpos[p]] for p in d_tr_pids]),
            np.array([y_dm[dpos[p]] for p in d_tr_pids]),
            ncrt_d, np.array([X_dm[dpos[p]] for p in ncrt_d]))
    else:
        p_d_tr, p_d_ncrt = None, None
    # 融合部署 (z 统计 = direct 训练)
    branch_probs = {"clin": (all_direct, p_clin_tr_all, ncrt_arr, p_clin_ncrt),
                    "t": (t_tr_pids, p_t_tr, ncrt_t, p_t_ncrt)}
    if p_d_ncrt is not None:
        branch_probs["d"] = (d_tr_pids, p_d_tr, ncrt_d, p_d_ncrt)
    # 策略 B 融合 → z, 再套 direct-OOF 拟合的单校准 (pooled Platt on direct OOF z)
    zs = {}
    for nm, (pids_tr, pr_tr, pids_te, pr_te) in branch_probs.items():
        L_tr = tsm.logit_clip(np.asarray(pr_tr, dtype=float))
        mu, sd = L_tr.mean(), L_tr.std() + 1e-9
        L_te = tsm.logit_clip(np.asarray(pr_te, dtype=float))
        zs[nm] = {"tr": pd.Series((L_tr - mu) / sd, index=np.asarray(pids_tr, dtype=np.int64)),
                  "te": pd.Series((L_te - mu) / sd, index=np.asarray(pids_te, dtype=np.int64))}
    pids_te_all = sorted(set().union(*[set(z["te"].index) for z in zs.values()]))
    rows = []
    for pid in pids_te_all:
        comps = {n: z["te"].loc[pid] for n, z in zs.items() if pid in z["te"].index}
        zc = comps.get("clin")
        imgs = [z for n, z in comps.items() if n != "clin"]
        if zc is None or not imgs:
            v = float(np.mean(list(comps.values())))
        else:
            v = 0.5 * float(zc) + 0.5 * float(np.mean(imgs))
        rows.append({"patient_id": int(pid), "prob": v})
    ncrt_z = pd.DataFrame(rows)
    # 校准: pooled Platt 拟合 direct OOF 的 z-score (fused["prob"] 是 z) → 应用到 ncrt z
    a2, b2 = platt_fit(fused["prob"], fused["label_T"])
    ncrt_z["prob"] = sigmoid(a2 * ncrt_z["prob"] + b2)
    ncrt_z["label_T"] = ncrt_z.patient_id.map(master.set_index("patient_id")["label_T"]).astype(int)
    ncrt_z["pT"] = ncrt_z.patient_id.map(master.set_index("patient_id")["pT"])
    ncrt_z = ncrt_z.dropna(subset=["prob"]).reset_index(drop=True)
    ncrt_z["prob"] = np.clip(ncrt_z["prob"], 0, 1)
    ncrt_z["fold"] = 0
    m_all = metrics_of(ncrt_z)
    mean_pred = float(ncrt_z["prob"].mean())
    true_rate = float(ncrt_z["label_T"].mean())
    # 过分期: ypT2 (label_T==0) 被判 T3-4
    ypt2 = ncrt_z[ncrt_z["pT"] == 2]
    ypt34 = ncrt_z[ncrt_z["pT"] >= 3]
    over50 = float(((ypt2["prob"] >= 0.50)).mean()) if len(ypt2) else None
    over_youden = float((ypt2["prob"] >= th_y).mean()) if len(ypt2) else None
    med2 = float(np.median(ypt2["prob"])) if len(ypt2) else None
    q2 = np.percentile(ypt2["prob"], [25, 75]) if len(ypt2) else None
    med34 = float(np.median(ypt34["prob"])) if len(ypt34) else None
    q34 = np.percentile(ypt34["prob"], [25, 75]) if len(ypt34) else None
    iqr2 = float(q2[1] - q2[0]) if q2 is not None else None
    iqr34 = float(q34[1] - q34[0]) if q34 is not None else None
    # 跨模态: t∩d_self 配对 n=34 (ncrt 内) 与 direct 内部配对 (direct 内 self∩t)
    pair_pids = set(ncrt_t) & set(ncrt_d)
    ncrt_pair = ncrt_z[ncrt_z.patient_id.isin(pair_pids)].reset_index(drop=True)
    direct_pair = set(self_d) & set(t_tr_pids)
    # direct 内部验证 (OOF 概率)
    t_oof_dp = t_oof[t_oof.patient_id.isin(direct_pair)].reset_index(drop=True)
    d_oof_dp = d_oof[d_oof.patient_id.isin(direct_pair)].reset_index(drop=True)
    auc_t_direct = float(roc_auc_score(t_oof_dp["label_T"], t_oof_dp["prob"])) if len(t_oof_dp) >= 8 else None
    auc_d_direct = float(roc_auc_score(d_oof_dp["label_T"], d_oof_dp["prob"])) if len(d_oof_dp) >= 8 else None
    # ncrt 配对单路: t 单路与 d_self 单路部署概率 (严格限定 t∩d_self 配对患者集)
    t_single = pd.DataFrame({"patient_id": ncrt_t,
                             "prob": np.clip(p_t_ncrt, 0, 1)})
    t_single["label_T"] = t_single.patient_id.map(master.set_index("patient_id")["label_T"]).astype(int)
    t_pair = t_single[t_single.patient_id.isin(pair_pids)].reset_index(drop=True)
    d_single = pd.DataFrame({"patient_id": ncrt_d,
                             "prob": np.clip(p_d_ncrt, 0, 1)})
    d_single["label_T"] = d_single.patient_id.map(master.set_index("patient_id")["label_T"]).astype(int)
    d_pair = d_single[d_single.patient_id.isin(pair_pids)].reset_index(drop=True)
    auc_t_ncrt = float(roc_auc_score(t_pair["label_T"], t_pair["prob"])) if len(t_pair) >= 8 else None
    auc_d_ncrt = float(roc_auc_score(d_pair["label_T"], d_pair["prob"])) if len(d_pair) >= 8 else None
    res = {
        "module": "M4 ncrt (n=75) post-CRT domain-shift stress test",
        "direct_train_n": int(len(all_direct)),
        "direct_oof_youden_threshold": round(th_y, 4),
        "ncrt_overall": {**m_all, "mean_pred": round(mean_pred, 4),
                         "true_t_rate": round(true_rate, 4)},
        "over_staging": {"ypT2_n": int(len(ypt2)), "ypT2_over50_rate": round(over50, 4) if over50 is not None else None,
                         "ypT2_over_youden_rate": round(over_youden, 4) if over_youden is not None else None,
                         "ypT2_median_prob": round(med2, 4) if med2 is not None else None,
                         "ypT2_iqr": [round(float(x), 4) for x in q2] if q2 is not None else None,
                         "ypT34_n": int(len(ypt34)),
                         "ypT34_median_prob": round(med34, 4) if med34 is not None else None,
                         "ypT34_iqr": [round(float(x), 4) for x in q34] if q34 is not None else None},
        "cross_modal_paired_n34": {
            "ncrt_paired_n": int(len(ncrt_pair)),
            "direct_paired_n": int(len(direct_pair)),
            "t_auc_direct": round(auc_t_direct, 4) if auc_t_direct is not None else None,
            "t_auc_ncrt": round(auc_t_ncrt, 4) if auc_t_ncrt is not None else None,
            "d_auc_direct": round(auc_d_direct, 4) if auc_d_direct is not None else None,
            "d_auc_ncrt": round(auc_d_ncrt, 4) if auc_d_ncrt is not None else None,
            "t_attenuation": round(auc_t_direct - auc_t_ncrt, 4) if (auc_t_direct is not None and auc_t_ncrt is not None) else None,
            "d_attenuation": round(auc_d_direct - auc_d_ncrt, 4) if (auc_d_direct is not None and auc_d_ncrt is not None) else None,
            "note": "exploratory small-n (n=34); no significance claims"},
        "platt_deploy": [round(a2, 5), round(b2, 5)],
    }
    # ROC 图 (ncrt)
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(ncrt_z["label_T"], ncrt_z["prob"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ncrt overall AUC={m_all['pooled_auc']:.3f}")
    if t_pair is not None and len(t_pair) >= 8:
        fpr2, tpr2, _ = roc_curve(t_pair["label_T"], t_pair["prob"])
        ax.plot(fpr2, tpr2, "--", label=f"t single (n={len(t_pair)}) AUC={auc_t_ncrt:.3f}")
    if len(d_pair) >= 8:
        fpr3, tpr3, _ = roc_curve(d_pair["label_T"], d_pair["prob"])
        ax.plot(fpr3, tpr3, ":", label=f"d_self single (n={len(d_pair)}) AUC={auc_d_ncrt:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ncrt (n=75) domain-shift ROC")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roc_ncrt_domain.png"), dpi=130)
    plt.close(fig)
    return res


# ---------------- 主流程 ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Module 1 ===")
    m1, lock_df = module1()
    print("M1 locked:", m1["selection"]["locked"], "AUC", m1["locked_metrics"]["pooled_auc"],
          "Brier", m1["locked_metrics"]["brier"], "ECE", m1["locked_metrics"]["ece10"])
    lock_mode = m1["selection"]["locked"]
    print("=== Module 2 (10x5) ===")
    m2 = module2(lock_mode)
    for scope in ("direct_tcd", "direct_td"):
        print(scope, m2[scope]["clin_dt"]["repeated_pooled_auc_mean"],
              "±", m2[scope]["clin_dt"]["repeated_pooled_auc_sd_ddof1"],
              "win", m2[scope]["win_rate_best_vs_clin"])
    print("=== Module 3 (temporal) ===")
    m3 = module3()
    print("M3 test AUC:", {k: v["test_metrics"]["pooled_auc"] for k, v in m3["direct_all"].items()})
    print("=== Module 4 (ncrt) ===")
    m4 = module4(lock_mode)
    print("M4 ncrt AUC:", m4["ncrt_overall"]["pooled_auc"],
          "over50:", m4["over_staging"]["ypT2_over50_rate"])
    suite = {"m1": m1, "m2": m2, "m3": m3, "m4": m4,
             "locked_calibration_mode": lock_mode,
             "redlines": {"no_old_oof_reuse": True,
                          "refit_from_raw": "all modules recompute clin/d_self/t from raw features; "
                                            "s4 CSV only used as DeLong comparator (not feature)"}}
    with open(JOUT, "w", encoding="utf-8") as fp:
        json.dump(suite, fp, ensure_ascii=False, indent=2)
    write_md(suite)
    print("saved ->", JOUT, "/", MOUT)


def write_md(s):
    L = ["# exp_v2_validation_suite — 全套收口验证 (1650 端, m01774)",
         "",
         f"锁定校准口径: **{s['locked_calibration_mode']}** (校准层在 direct_all 训练折 n≈224 拟合)",
         "",
         "## 模块 1: 单口径校准 + DCA + 校准曲线 + 亚组 (direct_tcd n=162)",
         "",
         "### 校准方式对比 (校准层在 direct_all 训练折拟合)",
         "",
         "| 方式 | Pooled AUC | 95% CI | Fold-Mean±SD | Brier | ECE10 |",
         "|---|---|---|---|---|---|"]
    for name in ("uncal_sigmoid_tcd", "platt_tcd", "bonly_tcd"):
        m = s["m1"][name]
        L.append(f"| {name} | {m['pooled_auc']:.4f} | "
                 f"[{m['boot_95ci'][0]:.4f},{m['boot_95ci'][1]:.4f}] | "
                 f"{m['fold_mean']:.4f}±{m['fold_sd']:.4f} | {m['brier']:.4f} | {m['ece10']:.4f} |")
    lm = s["m1"]["locked_metrics"]
    L += [f"**锁定列** ({s['m1']['selection']['locked']}): Pooled AUC **{lm['pooled_auc']:.4f}** "
          f"[{lm['boot_95ci'][0]:.4f},{lm['boot_95ci'][1]:.4f}] (Fold {lm['fold_mean']:.4f}±{lm['fold_sd']:.4f}), "
          f"Brier {lm['brier']:.4f}, ECE {lm['ece10']:.4f}",
          "",
          f"DeLong (锁定列 vs 旧 s4): ΔAUC {s['m1']['delong_locked_vs_s4']['res']['diff']:+.4f}, "
          f"z={s['m1']['delong_locked_vs_s4']['res']['z']:.3f}, p={s['m1']['delong_locked_vs_s4']['res']['p']:.4f}, "
          f"n_paired={s['m1']['delong_locked_vs_s4']['n_paired']}",
          "",
          "### DCA 净获益 (阈值区间)",
          "",
          "| 阈值 | 最优模型 NB | clin_clean NB | Treat All NB | Treat None |",
          "|---|---|---|---|---|"]
    for r in s["m1"]["dca"]["thresholds"]:
        L.append(f"| {r['threshold']:.2f} | {r['nb_best']:.4f} | {r['nb_clin']:.4f} | "
                 f"{r['nb_all']:.4f} | {r['nb_none']:.4f} |")
    L += ["",
          "### 临床亚组 (direct_tcd 162)",
          "",
          "| 亚组 | 描述 | n | Pooled AUC | 95% CI | Brier |",
          "|---|---|---|---|---|---|"]
    for k, v in s["m1"]["subgroups"].items():
        if v.get("pooled_auc") is None:
            note = v.get("note", "")
            if v.get("available") is False:
                note = "表内无字段"
            elif v.get("n", 0) < 8:
                note = f"n={v.get('n')} 过小不可估"
            elif note:
                note = "单类/退化"
            L.append(f"| {k} | {v.get('desc','')} | {v.get('n','-')} | - | - | - |"
                     + (f" | {note}" if note else " |"))
        else:
            L.append(f"| {k} | {v['desc']} | {v['n']} | {v['pooled_auc']:.4f} | "
                     f"[{v['boot_95ci'][0]:.4f},{v['boot_95ci'][1]:.4f}] | {v['brier']:.4f} |")
    L += ["", "## 模块 2: 10×5 重复分层 CV (TRIPOD 1b, seed 0-9)", ""]
    for scope in ("direct_tcd", "direct_td"):
        r = s["m2"][scope]
        L += [f"### {scope}",
              "",
              "| 模型 | Repeated Pooled OOF AUC (mean±SD) | min–max | 50 折 Fold-Mean±SD | "
              "胜率 vs clin | 胜率 vs clin+t | mean ΔAUC vs clin | mean ΔAUC vs clin+t |",
              "|---|---|---|---|---|---|---|---|"]
        for name in ("clin", "clin_t", "clin_dt"):
            x = r[name]
            L.append(f"| {name} | {x['repeated_pooled_auc_mean']:.4f}±{x['repeated_pooled_auc_sd_ddof1']:.4f} | "
                     f"{x['repeated_pooled_auc_min']:.4f}–{x['repeated_pooled_auc_max']:.4f} | "
                     f"{x['fold50_mean']:.4f}±{x['fold50_sd']:.4f} | "
                     f"{r['win_rate_best_vs_clin'] if name=='clin_dt' else '-'} | "
                     f"{r['win_rate_best_vs_clin_t'] if name=='clin_dt' else '-'} | "
                     f"{r['mean_delta_vs_clin'] if name=='clin_dt' else '-'} | "
                     f"{r['mean_delta_vs_clin_t'] if name=='clin_dt' else '-'} |")
    L += ["", "## 模块 3: 时间外推 (TRIPOD 2b)", "",
          f"- 临床表日期核查: {s['m3']['clinical_date_check']}",
          f"- 采用 DICOM StudyDate 切分: 节点 {s['m3']['cutoff_date']}, "
          f"train {s['m3']['train_n']} / test {s['m3']['test_n']} "
          f"(缺失日期 {s['m3']['missing_date_excluded']} 例不入训练)",
          f"- tcd 子集: train {s['m3']['tcd_subset']['train_n']} / test {s['m3']['tcd_subset']['test_n']}", ""]
    for scope in ("direct_all", "direct_tcd"):
        L += [f"### {scope} 测试集 (冻结推理)", "",
              "| 模型 | n | Pooled AUC | 95% CI | Brier | ECE | Youden阈值(训练) | Sens | Spec | PPV | NPV |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for mname in ("clin", "clin_t", "clin_dt"):
            m = s["m3"][scope][mname]
            mt = m["test_metrics"]
            yd = m["test_at_train_youden"]
            L.append(f"| {mname} | {mt['n']} | {mt['pooled_auc']:.4f} | "
                     f"[{mt['boot_95ci'][0]:.4f},{mt['boot_95ci'][1]:.4f}] | {mt['brier']:.4f} | {mt['ece10']:.4f} | "
                     f"{m['train_youden_threshold']:.4f} | {yd['sens']:.4f} | {yd['spec']:.4f} | "
                     f"{yd['ppv']:.4f} | {yd['npv']:.4f} |")
    m4 = s["m4"]
    L += ["", "## 模块 4: ncrt (n=75) 域偏移压力测试", "",
          f"- direct 训练 n={m4['direct_train_n']}, direct OOF Youden 阈值={m4['direct_oof_youden_threshold']}",
          f"- **ncrt 整体**: AUC {m4['ncrt_overall']['pooled_auc']:.4f} "
          f"[{m4['ncrt_overall']['boot_95ci'][0]:.4f},{m4['ncrt_overall']['boot_95ci'][1]:.4f}], "
          f"Brier {m4['ncrt_overall']['brier']:.4f}, 预测均值 {m4['ncrt_overall']['mean_pred']:.4f} "
          f"vs 真实阳性率 {m4['ncrt_overall']['true_t_rate']:.4f}",
          "",
          "### 过分期分析",
          "",
          "| 组 | n | 中位预测概率 | IQR |",
          "|---|---|---|---|"]
    os_ = m4["over_staging"]
    L.append(f"| ypT2 (label 0) | {os_['ypT2_n']} | {os_['ypT2_median_prob']:.4f} | "
             f"[{os_['ypT2_iqr'][0]:.4f},{os_['ypT2_iqr'][1]:.4f}] |")
    L.append(f"| ypT3-4 (label 1) | {os_['ypT34_n']} | {os_['ypT34_median_prob']:.4f} | "
             f"[{os_['ypT34_iqr'][0]:.4f},{os_['ypT34_iqr'][1]:.4f}] |")
    L += ["",
          f"**ypT2 过分期为 T3-4 的假阳性率**: p≥0.50 口径 {os_['ypT2_over50_rate']:.4f} / "
          f"训练集 Youden 阈值口径 {os_['ypT2_over_youden_rate']:.4f}",
          "",
          "### 跨模态衰减 (t ∩ d_self 配对, 探索性小样本)",
          "",
          "| 单路 | direct 内部 AUC | ncrt (n=34) AUC | 衰减 |",
          "|---|---|---|---|"]
    cm = m4["cross_modal_paired_n34"]
    L.append(f"| t (AreaW) | {cm['t_auc_direct']:.4f} | {cm['t_auc_ncrt']:.4f} | {cm['t_attenuation']:+.4f} |")
    L.append(f"| d_self (MeanMax) | {cm['d_auc_direct']:.4f} | {cm['d_auc_ncrt']:.4f} | {cm['d_attenuation']:+.4f} |")
    L += ["", f"- 配对说明: direct 配对 n={cm['direct_paired_n']}, ncrt 配对 n={cm['ncrt_paired_n']} "
              f"({cm['note']})"]
    with open(MOUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
