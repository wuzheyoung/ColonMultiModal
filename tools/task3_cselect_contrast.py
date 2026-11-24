# -*- coding: utf-8 -*-
"""task3_cselect_contrast.py — 只读高维选 C 对照 (m00622 第3项)。

d_conv / t_conv / c_conv 在 n=162 切片上的 OOF AUC:
  (a) 原版 TRAIN-AUC 选 C  (audit per_fold_lr, imp=None)
  (b) 折内嵌套 3 折 CV 选 C (每折训练侧再划 3 折, 内层 AUC 均值选 C, 全训练侧重拟合)
逐折打印实际选中的 C, 供 MIL 线性探针基线参考。仅打印, 不覆写任何已锁定产物。
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location(
    "rc", os.path.join(ROOT, "tools", "recompute_direct_tcd_singles.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

C_GRID = (0.01, 0.1, 1.0, 10.0)
BRANCHES = {
    "d_conv": ("features/logs/route2_cd_feats/d.npy", "features/logs/route2_cd_feats/d_pids.csv"),
    "c_conv": ("features/logs/route2_cd_feats/c.npy", "features/logs/route2_cd_feats/c_pids.csv"),
    "t_conv": ("features/logs/route2_feats", None),
}


def conv_feats_npy(path_npy, path_pids):
    X = np.load(path_npy)
    pids = pd.read_csv(path_pids)["patient_id"].astype(np.int64).values
    return X, pids


def conv_feats_dir(d):
    pids, mats = [], []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npy"):
            pids.append(int(fn[:-4]))
            mats.append(np.load(os.path.join(d, fn)).mean(axis=0))
    pids = np.array(pids)
    return np.stack(mats), pids


def load_branch(name):
    if name == "t_conv":
        X, pids = conv_feats_dir(BRANCHES[name][0])
    else:
        X, pids = conv_feats_npy(*BRANCHES[name])
    return X, pids


def outer_train_auc(X, y, folds):
    """原版: 每折 TRAIN-AUC 选 C。返回 (oof_probs, per_fold_C)。"""
    oof = np.full(len(y), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 2 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5; Cs[k] = None; continue
        Xtr, Xva = np.nan_to_num(X[tr].astype(float), nan=0.0), np.nan_to_num(X[va].astype(float), nan=0.0)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
        best_c, best_a, best_m = C_GRID[0], -1, None
        for c in C_GRID:
            m = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(Xtr_s, y[tr])
            a = roc_auc_score(y[tr], m.predict_proba(Xtr_s)[:, 1])
            if a > best_a:
                best_a, best_c, best_m = a, c, m
        oof[va] = best_m.predict_proba(Xva_s)[:, 1]
        Cs[k] = best_c
    return oof, Cs


def outer_nested_cv(X, y, folds):
    """折内嵌套 3 折 CV 选 C: 训练侧按 fold%3 划内层, 内层均值 AUC 选 C → 全训练重拟合。"""
    oof = np.full(len(y), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 2 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5; Cs[k] = None; continue
        Xtr = np.nan_to_num(X[tr].astype(float), nan=0.0)
        Xva = np.nan_to_num(X[va].astype(float), nan=0.0)
        ytr = y[tr]
        inner = (folds[tr] % 3)
        inner_auc = {}
        for c in C_GRID:
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(np.unique(ytr[itr])) < 2 or iva.sum() < 2:
                    continue
                sc = StandardScaler().fit(Xtr[itr])
                m = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr[itr]), ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], m.predict_proba(sc.transform(Xtr[iva]))[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(C_GRID, key=lambda c: inner_auc[c])
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
        oof[va] = m.predict_proba(sc.transform(Xva))[:, 1]
        Cs[k] = best_c
    return oof, Cs


def main():
    master = pd.read_csv("data/master_5fold_356.csv")
    cohort = pd.read_csv("data/direct_cohort_ids.csv")
    raw = pd.read_csv("data/hos_pairs_raw.csv")
    has_t = set(raw.loc[raw["mri_path_t"].notna(), "patient_id"])
    has_c = set(raw.loc[raw["mri_path_c"].notna(), "patient_id"])
    has_d = set(raw.loc[raw["mri_path_d"].notna(), "patient_id"])
    raw = raw.set_index("patient_id")
    tgr = set(cohort.loc[cohort["is_tgr"] == 1, "pid"].astype(np.int64))
    d_ids = np.sort(master.loc[~master["patient_id"].isin(tgr), "patient_id"].values.astype(np.int64))
    m = master.set_index("patient_id")
    tcd = np.array(sorted(set(d_ids) & has_t & has_c & has_d))

    print(f"direct_all={len(d_ids)}  direct_tcd eval={len(tcd)}  T+={int(m.loc[tcd,'label_T'].sum())}")
    print("=" * 88)
    for name in BRANCHES:
        X, pids = load_branch(name)
        ids = np.array(sorted(set(d_ids) & set(pids)))
        pos = {int(p): i for i, p in enumerate(pids)}
        X_ids = X[np.array([pos[int(p)] for p in ids])]     # 按 ids 对齐行 (d.npy 含 ncrt 患者)
        mask = np.isin(ids, tcd)
        ids_e = ids[mask]
        X_e = X_ids[mask]
        y_e = m.loc[ids_e, "label_T"].values.astype(int)
        f_e = m.loc[ids_e, "fold"].values.astype(int)
        y_all = m.loc[ids, "label_T"].values.astype(int)
        f_all = m.loc[ids, "fold"].values.astype(int)

        pA, CA = outer_train_auc(X_ids, y_all, f_all)
        pB, CB = outer_nested_cv(X_ids, y_all, f_all)
        pA_e = pA[mask]; pB_e = pB[mask]
        aA = rc.pooled_auc(y_e, pA_e)
        aB = rc.pooled_auc(y_e, pB_e)
        print(f"\n[{name}]  n_train(all direct)={len(ids)}  n_eval(tcd)={len(ids_e)}")
        print(f"  TRAIN-AUC选C : pooled OOF AUC={aA}  每折C={[CB.get(k) for k in sorted(set(f_e))] and [CA.get(k) for k in sorted(set(f_e))]}")
        print(f"  nested3CV选C : pooled OOF AUC={aB}  每折C={[CB.get(k) for k in sorted(set(f_e))]}")
        # 逐折明细
        for k in sorted(set(f_e)):
            va = f_e == k
            yv, pA_v, pB_v = y_e[va], pA_e[va], pB_e[va]
            if len(np.unique(yv)) < 2:
                continue
            print(f"    fold {k}: trainC={CA.get(k)} auc={roc_auc_score(yv, pA_v):.4f}  |  "
                  f"nestC={CB.get(k)} auc={roc_auc_score(yv, pB_v):.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
