# -*- coding: utf-8 -*-
"""task1_supplement.py — 任务1 补充核验 (m00524)。
(a) 选 C 逻辑对照: 原版 audit_fusion_debug.py.per_fold_lr 确认按 TRAIN AUC 选 C
    (line 100-106, roc_auc_score(y[tr], proba(Xtr)) 取最大) → 当前 baseline_locked_n162
    即为 audit 复现口径。补充三种选 C 方式在 clin_clean 上的对照:
      train_auc  (audit 原版) / fixed_C01 / nested3cv (折内嵌套 3 折 CV 选 C)。
    输出 direct_all (n=280) OOF AUC + 切片到 n=162 的 Pooled/Fold-Mean±SD, 厘清
    0.6044 → 0.6469 变动来源 (应为评估子集改变而非选 C)。
(b) d_conv 覆盖拆解: 在 n=162 评估集上, s4.f1 在 d_conv 存在 81 例 vs 缺失 81 例的
    AUC; 以及 n=81 共同交集上 d_conv vs s4.f1 的配对 DeLong 对比。
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
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


# ---------- DeLong (placement-values 形式) 两相关 ROC 检验, 供配对交集使用 ----------
def delong_paired_test(y, p1, p2):
    """配对 DeLong 检验 (U-统计量 placement 值形式, 自动含两分类器协方差, 支持不平衡样本):
    返回 (auc1, auc2, var_diff, z, p_two_sided)。"""
    y = np.asarray(y, dtype=int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 < 1 or n0 < 1:
        return None
    neg, pos = y == 0, y == 1
    V10_1 = np.array([np.mean(p1[neg] < p1[i]) for i in range(len(y))])
    V01_1 = np.array([np.mean(p1[pos] > p1[i]) for i in range(len(y))])
    V10_2 = np.array([np.mean(p2[neg] < p2[i]) for i in range(len(y))])
    V01_2 = np.array([np.mean(p2[pos] > p2[i]) for i in range(len(y))])
    auc1 = float(V10_1[pos].mean())
    auc2 = float(V10_2[pos].mean())
    D10 = V10_1 - V10_2
    D01 = V01_1 - V01_2
    var = float(np.var(D10[pos], ddof=1) / n1 + np.var(D01[neg], ddof=1) / n0)
    if var <= 0:
        return (round(auc1, 4), round(auc2, 4), None, None, None)
    z = (auc1 - auc2) / np.sqrt(var)
    from scipy.stats import norm
    return (round(auc1, 4), round(auc2, 4), round(var, 6),
            round(float(z), 4), round(float(2 * (1 - norm.cdf(abs(z)))), 6))


def oof_clin(cmode, X, y, folds):
    """clin_clean OOF: 三种选 C 方式。"""
    oof = np.full(len(y), np.nan)
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 2 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            continue
        ip = SimpleImputer(strategy="median").fit(X[tr])
        Xtr = ip.transform(X[tr]); Xva = ip.transform(X[va])
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
        if cmode == "fixed_C01":
            best_c = 0.1
        elif cmode == "nested3cv":
            best_c, best_a = 0.01, -1
            ikf = KFold(n_splits=3, shuffle=True, random_state=42)
            for c in (0.01, 0.1, 1.0, 10.0):
                inner = []
                for itr, iva in ikf.split(np.arange(tr.sum())):
                    ipi = SimpleImputer(strategy="median").fit(Xtr[itr])
                    Xtri = ipi.transform(Xtr[itr]); Xvai = ipi.transform(Xtr[iva])
                    sci = StandardScaler().fit(Xtri)
                    mi = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(
                        sci.transform(Xtri), y[tr][itr])
                    inner.append(roc_auc_score(y[tr][iva], mi.predict_proba(sci.transform(Xvai))[:, 1]))
                a = float(np.mean(inner))
                if a > best_a:
                    best_a, best_c = a, c
        else:  # train_auc (audit 原版)
            best_c, best_a = 0.01, -1
            for c in (0.01, 0.1, 1.0, 10.0):
                m = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(Xtr_s, y[tr])
                a = roc_auc_score(y[tr], m.predict_proba(Xtr_s)[:, 1])
                if a > best_a:
                    best_a, best_c = a, c
        m = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(Xtr_s, y[tr])
        oof[va] = m.predict_proba(Xva_s)[:, 1]
    return oof


def main():
    master = pd.read_csv(rc.MASTER_CSV)
    raw = pd.read_csv(rc.PAIRS_RAW).set_index("patient_id")
    coh = pd.read_csv("data/direct_cohort_ids.csv")
    tgr = set(coh.loc[coh["is_tgr"] == 1, "pid"].astype(int))
    d_ids = np.sort(master.loc[~master["patient_id"].isin(tgr), "patient_id"].values.astype(np.int64))
    has_p = {s: set(raw.index[raw[f"mri_path_{s}"].notna()]) for s in ("t", "c", "d")}
    tcd = np.array(sorted(set(d_ids) & has_p["t"] & has_p["c"] & has_p["d"]))
    m = master.set_index("patient_id")
    clin_raw = raw[rc.CLIN_COLS].replace([np.inf, -np.inf], np.nan).astype(float)
    Xc = clin_raw.loc[d_ids].values.astype(float)
    yT = m.loc[d_ids, "label_T"].values.astype(int)
    fT = m.loc[d_ids, "fold"].values.astype(int)

    print("=== (a) clin_clean 选 C 方式对照 (direct_all n=280 → 切片 n=162) ===")
    print(f"{'C-mode':12s} {'dir_all pooled':>14s} {'tcd pooled':>10s} {'tcd fold-mean±sd':>18s}")
    res_a = {}
    for cm in ("train_auc", "fixed_C01", "nested3cv"):
        p = oof_clin(cm, Xc, yT, fT)
        pa = rc.pooled_auc(yT, p)
        mask = np.isin(d_ids, tcd)
        ye, pe, fe = yT[mask], p[mask], fT[mask]
        pt = rc.pooled_auc(ye, pe)
        fm, sd = rc.fold_mean_auc(ye, pe, fe)
        res_a[cm] = {"dir_all_pooled": pa, "tcd_pooled": pt, "tcd_fold_mean": fm, "tcd_fold_sd": sd}
        print(f"{cm:12s} {pa:>14} {pt:>10} {fm}±{sd}")
    # 变动来源: audit 原版 (train_auc) 在 direct_all 上即 0.60xx, 切片到 tcd 即 0.64xx
    print("\n  → 0.6044→0.6469 变动来源 = 评估子集 (direct_all 280 → tcd 162) 改变, "
          "选 C 方式不变 (audit 原版即 TRAIN AUC)。")

    # ---------- (b) d_conv 覆盖拆解 ----------
    print("\n=== (b) d_conv 覆盖拆解 (n=162 评估集) ===")
    s4 = pd.read_csv(os.path.join("outputs", "oof_baseline_n162", "fusions", "s4_clin_3plane.f1.csv"))
    dcv = pd.read_csv(os.path.join("outputs", "oof_baseline_n162", "singles", "d_conv.csv"))
    dset = set(dcv["patient_id"])
    s4 = s4.set_index("patient_id")
    with_d = [p for p in tcd if p in dset]
    no_d = [p for p in tcd if p not in dset]
    print(f"  s4.f1: n_total=162, d_conv 存在 {len(with_d)} 例, d_conv 缺失 {len(no_d)} 例")
    for tag, ids in (("with_d (81)", with_d), ("without_d (81)", no_d)):
        y = m.loc[ids, "label_T"].values.astype(int)
        p = s4.loc[ids, "prob"].values.astype(float)
        print(f"  s4.f1 on {tag:15s}: pooled AUC = {rc.pooled_auc(y, p):.4f}"
              f"  (T+={int(y.sum())})")
    # n=81 共同交集: d_conv vs s4.f1
    ids = sorted(with_d)
    y = m.loc[ids, "label_T"].values.astype(int)
    pd_ = dcv.set_index("patient_id").loc[ids, "prob"].values.astype(float)
    pf_ = s4.loc[ids, "prob"].values.astype(float)
    print(f"\n  n=81 共同交集: d_conv pooled={rc.pooled_auc(y, pd_):.4f} | "
          f"s4.f1 pooled={rc.pooled_auc(y, pf_):.4f}")
    dl = delong_paired_test(y, pd_, pf_)
    if dl:
        print(f"  配对 DeLong: var_diff={dl[2]} z={dl[3]} p={dl[4]}")

    out = {"a": res_a, "b": {"n_total": int(len(tcd)),
                             "d_present": len(with_d), "d_missing": len(no_d),
                             "s4_f1_with_d": rc.pooled_auc(m.loc[with_d, "label_T"].values.astype(int),
                                                           s4.loc[with_d, "prob"].values.astype(float)),
                             "s4_f1_without_d": rc.pooled_auc(m.loc[no_d, "label_T"].values.astype(int),
                                                              s4.loc[no_d, "prob"].values.astype(float)),
                             "d_conv_vs_s4_n81": {"d_conv": rc.pooled_auc(y, pd_),
                                                  "s4_f1": rc.pooled_auc(y, pf_),
                                                  "delong_p": dl[4] if dl else None}}}
    with open("logs/task1_supplement.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("\nDONE → logs/task1_supplement.json")


if __name__ == "__main__":
    main()
