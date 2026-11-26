# -*- coding: utf-8 -*-
"""stage5b_misdiagnosis_final.py — 错例诊断 + 最终报告生成
Winner_Model = H1.1 (t60 AWM + dself meanmax + clin 策略B + bonly 校准), td 0.7003±0.0298。
错例分析: 10×5 平均 OOF 概率 → 假阴/假阳 (Youden 阈值) 的
  max_area_mm2 / DWI 缺失率 / 临床指标分布差异。
输出: logs/auto_staging_opt/final_autonomous_report.md + all_experiments.json + progress_log.md 末节。
"""
import os, sys, json, datetime
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

import recompute_direct_tcd_singles as rc

OUTDIR = "logs/auto_staging_opt"
os.makedirs(OUTDIR, exist_ok=True)
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

def winner_oof(seed, fo):
    th, tX, ty = pool_branch(T60, "t", "area_w", all_direct)
    dh, dX, dy = pool_branch(T120, "d", "meanmax", d_self)
    t_p = np.full(len(all_direct), np.nan); t_p[[pos_of[p] for p in th]] = probe_oof(th, tX, ty, np.array([fo[pos_of[p]] for p in th]))
    d_p = np.full(len(all_direct), np.nan); d_p[[pos_of[p] for p in dh]] = probe_oof(dh, dX, dy, np.array([fo[pos_of[p]] for p in dh]))
    clin_p = clin_oof(all_direct, fo)
    sub = pd.DataFrame({"patient_id": all_direct, "fold": fo, "label_T": y_all,
                        "clin": clin_p, "t60": t_p, "dself": d_p})
    # 策略B: 0.5*z_clin + 0.5*mean(z_t, z_d)
    z_clin = np.log(sub["clin"] / (1 - sub["clin"] + 1e-9))
    z_t = np.log(sub["t60"] / (1 - sub["t60"] + 1e-9))
    z_d = np.log(sub["dself"] / (1 - sub["dself"] + 1e-9))
    z_im = np.nanmean(np.vstack([z_t, z_d]), axis=0)
    # 双影像全缺的样本: z_im NaN → 中性 (z=0)
    z_im = np.where(np.isnan(z_im), 0.0, z_im)
    z_fuse = 0.5 * z_clin + 0.5 * z_im
    prob = 1.0 / (1.0 + np.exp(-z_fuse))
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    sub["prob_raw"] = prob
    # bonly 校准 (需要列名 patient_id/fold/label_T/prob)
    cdf = pd.DataFrame({"patient_id": sub["patient_id"], "fold": sub["fold"],
                        "label_T": sub["label_T"], "prob": prob})
    cal, _ = S.calibrate_cv(cdf, mode="bonly")
    return cal

def main():
    # 收集 10×5 全部 OOF (平均概率)
    agg = {}
    for seed in SEEDS:
        fo = fold_of_all(seed)
        df = winner_oof(seed, fo)
        for _, r in df.iterrows():
            pid = int(r["patient_id"])
            a = agg.setdefault(pid, {"n": 0, "sum": 0.0, "label": int(r["label_T"])})
            a["n"] += 1
            a["sum"] += float(r["prob"])
    rows = []
    for pid, a in agg.items():
        rows.append({"patient_id": pid, "prob": a["sum"] / a["n"], "label_T": a["label"]})
    oof_all = pd.DataFrame(rows)
    td_mask = oof_all.patient_id.isin(td)
    oof_td = oof_all[td_mask].copy()

    # Youden 阈值 (td)
    auc = float(roc_auc_score(oof_td["label_T"], oof_td["prob"]))
    thresh, best_youden = 0.5, -1.0
    for t in np.linspace(0, 1, 101):
        pred = (oof_td["prob"] >= t).astype(int)
        tp = ((pred == 1) & (oof_td["label_T"] == 1)).sum()
        fn = ((pred == 0) & (oof_td["label_T"] == 1)).sum()
        fp = ((pred == 1) & (oof_td["label_T"] == 0)).sum()
        tn = ((pred == 0) & (oof_td["label_T"] == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        yd = sens + spec - 1
        if yd > best_youden:
            best_youden, thresh = yd, t
    oof_td["pred"] = (oof_td["prob"] >= thresh).astype(int)
    fn_df = oof_td[(oof_td["label_T"] == 1) & (oof_td["pred"] == 0)]
    fp_df = oof_td[(oof_td["label_T"] == 0) & (oof_td["pred"] == 1)]
    tp_df = oof_td[(oof_td["label_T"] == 1) & (oof_td["pred"] == 1)]
    tn_df = oof_td[(oof_td["label_T"] == 0) & (oof_td["pred"] == 0)]

    # 病灶物理尺度 (tight60_m10 npz)
    def phys(pid):
        fp = os.path.join(T60, "dinov2", "t", f"{pid}.npz")
        if not os.path.exists(fp):
            return None
        z = np.load(fp)
        mm2 = np.asarray(z["mask_area_mm2"], dtype=np.float64)
        return {"max_area_mm2": float(mm2.max()), "total_area_mm2": float(mm2.sum()),
                "z_span": int(len(mm2))}
    for df_ in (fn_df, fp_df, tp_df, tn_df):
        df_["phys"] = df_["patient_id"].apply(phys)
    def stat(df_, key):
        vals = df_["phys"].dropna().apply(lambda d: d[key] if d else np.nan)
        return (round(float(vals.median()), 1) if len(vals) else None,
                round(float(vals.mean()), 1) if len(vals) else None, int(vals.notna().sum()))
    # DWI 缺失率
    d_have = set(d_self)
    def dwi_miss(df_):
        return round(float((~df_["patient_id"].isin(d_have)).mean()), 3)
    # 临床分布
    clincols = rc.CLIN_COLS
    def clin_med(df_):
        X = raw.loc[df_["patient_id"].values, clincols].median(numeric_only=True)
        return {k: round(float(v), 2) if pd.notna(v) else None for k, v in X.items()}

    diag = {
        "td_n": int(len(oof_td)), "auc": round(auc, 4),
        "youden_threshold": round(float(thresh), 2),
        "confusion": {"TP": int(len(tp_df)), "FP": int(len(fp_df)),
                      "FN": int(len(fn_df)), "TN": int(len(tn_df))},
        "fn": {"n": int(len(fn_df)), "max_area_mm2_med/mean/n": stat(fn_df, "max_area_mm2"),
               "total_area_mm2_med/mean/n": stat(fn_df, "total_area_mm2"),
               "dwi_missing_rate": dwi_miss(fn_df), "clin_median": clin_med(fn_df)},
        "fp": {"n": int(len(fp_df)), "max_area_mm2_med/mean/n": stat(fp_df, "max_area_mm2"),
               "total_area_mm2_med/mean/n": stat(fp_df, "total_area_mm2"),
               "dwi_missing_rate": dwi_miss(fp_df), "clin_median": clin_med(fp_df)},
        "tp": {"n": int(len(tp_df)), "max_area_mm2_med/mean/n": stat(tp_df, "max_area_mm2"),
               "dwi_missing_rate": dwi_miss(tp_df), "clin_median": clin_med(tp_df)},
        "tn": {"n": int(len(tn_df)), "max_area_mm2_med/mean/n": stat(tn_df, "max_area_mm2"),
               "dwi_missing_rate": dwi_miss(tn_df), "clin_median": clin_med(tn_df)},
    }
    with open(os.path.join(OUTDIR, "stage5_misdiagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(json.dumps(diag, ensure_ascii=False, indent=1))

    # ---------- all_experiments.json ----------
    exps = {}
    for fn in ("stage1_results.json", "stage2_results.json", "stage3_results.json",
               "stage4_results.json", "stage5_results.json", "stage5_misdiagnosis.json"):
        p = os.path.join(OUTDIR, fn)
        if os.path.exists(p):
            exps[fn.replace(".json", "")] = json.load(open(p, encoding="utf-8"))
    with open(os.path.join(OUTDIR, "all_experiments.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.datetime.now().isoformat(timespec="seconds"),
                   "lock_mode": "bonly", "cohort": "direct_td n=214 (dev), direct_tcd n=162 (ref)",
                   "winner": "H1.1: clin_clean + d_self(MeanMax) + t(tight60_m10 AWM), strategy B, bonly calib",
                   "winner_td_10x5": "0.7003 ± 0.0298 (tcd 0.7099 ± 0.0323)",
                   "experiments": exps}, f, ensure_ascii=False, indent=2)

    # ---------- final_autonomous_report.md ----------
    L = []
    L.append("# final_autonomous_report — 4060 端 6 阶段自主迭代 (10×5 重复 CV)")
    L.append("")
    L.append(f"生成时间: {datetime.datetime.now().isoformat(timespec='seconds')}")
    L.append("")
    L.append("## 0. 基线 (阶段0)")
    L.append("- 锁定基线数字澄清: **direct_td n=214 基线 = 0.6763±0.0302**; 用户指令所称 0.6935±0.0357 实为 direct_tcd n=162 口径 (两口径均精确复现 match<0.001)。")
    L.append("- Baseline_0 (direct_td): clin_clean + d_self(MeanMax) + t(120mm AWM), 策略B + bonly 校准 → **0.6763±0.0302**")
    L.append("")
    L.append("## 1. 阶段1 异构视野/物理尺度/浸润聚合")
    L.append("| 方案 | td 10×5 | tcd 10×5 | 判定 |")
    L.append("|---|---|---|---|")
    L.append("| Baseline_0 | 0.6763±0.0302 | 0.6935±0.0357 | 基准 |")
    L.append("| **H1.1 t→tight60_m10 AWM** | **0.7003±0.0298** | 0.7099±0.0323 | ✅ 采纳 (+0.024) |")
    L.append("| H1.2 baseline+物理标量 | 0.6767±0.0266 | 0.7081±0.0282 | ❌ 无增益 |")
    L.append("| H1.2 H1.1+物理标量 | 0.6910±0.0272 | 0.7141±0.0276 | ❌ 反而-0.009 |")
    L.append("| H1.3 t60 Top3 | 0.6953±0.0351 | 0.7004±0.0343 | ❌ 略逊 AWM |")
    L.append("")
    L.append("## 2. 阶段2 DWI from_t 纳回")
    L.append("| 方案 | td | 判定 |")
    L.append("|---|---|---|")
    L.append("| b0 d_self | 0.6763±0.0302 | 基准 |")
    L.append("| b0 +from_t | 0.6627±0.0316 | ❌ -0.014 |")
    L.append("| b0 +from_t+is_d_self | 0.6611±0.031 | ❌ -0.015 |")
    L.append("| H1.1 d_self | 0.7003±0.0298 | 基准 |")
    L.append("| H1.1 +from_t | 0.6902±0.0202 | ❌ -0.010 |")
    L.append("| H1.1 +from_t+is_d_self | 0.6898±0.022 | ❌ -0.011 |")
    L.append("→ **阶段2 拒绝**: from_t 借壳掩码纳回全面劣化, is_d_self 指示符无增益。")
    L.append("")
    L.append("## 3. 阶段3 DINOv2 Patch Token 瘤内/瘤周双流")
    L.append("| 方案 | td | tcd | 判定 |")
    L.append("|---|---|---|---|")
    L.append("| H3.1 t(intra) | 0.6849±0.0364 | 0.698 | ❌ 低于 H1.1 |")
    L.append("| **H3.2 t(intra+peri)** | 0.6985±0.0272 | 0.7076±0.0313 | ≈H1.1 持平 |")
    L.append("| ref t(cls) | 0.6526±0.0332 | 0.6587 | 参考 |")
    L.append("| H3.3 d(intra) | 0.6526 | 0.6587 | ❌ d 无增益 |")
    L.append("→ **阶段3 拒绝**: patch token 双流未超越 tight60 CLS AWM (H1.1)。")
    L.append("")
    L.append("## 4. 阶段4 放射组学 (3D 形态 + 瘤周环 3/5mm)")
    L.append("| 方案 | td | 判定 |")
    L.append("|---|---|---|")
    L.append("| rad_only(intra) | 0.5576±0.0261 | ❌ 弱 |")
    L.append("| clin+rad(rim_3mm) | 0.6494±0.0116 | 弱于 H1.1 |")
    L.append("| H4.2 clin+rad(rim_3mm)+t60+dself | 0.6816±0.0225 | ❌ 低于 H1.1 0.7003 |")
    L.append("| H4.2 clin+rad(rim_5mm)+t60+dself | 0.6804±0.0212 | ❌ 低于 H1.1 |")
    L.append("→ **阶段4 拒绝**: 放射组学在三模态融合无增益, 甚至拖累。")
    L.append("")
    L.append("## 5. 阶段5 分模态 Stacking + 贪心准入")
    L.append("| 元模型 | td | 判定 |")
    L.append("|---|---|---|")
    L.append("| stack l2 (4模态) | 0.627±0.0735 | ❌ 大幅劣化 |")
    L.append("| stack enet (4模态) | 0.6595±0.0634 | ❌ |")
    L.append("| stack svm+platt (4模态) | 0.6611±0.0599 | ❌ |")
    L.append("| stack l2 (clin,t60,dself) | 0.6472±0.0667 | ❌ |")
    L.append("| stack l2 (t60,dself) | 0.6017±0.0816 | ❌ |")
    L.append("→ **阶段5 拒绝**: 概率级 Stacking 全部劣化 (小样本过拟合+概率相关性), 策略B 简单融合最优。贪心准入: 无方案满足 +0.004/胜率60%/ECE≤0.065 → **Winner_Model = H1.1**。")
    L.append("")
    L.append("## 6. 错例诊断 (Winner_Model, td Youden 阈值)")
    L.append(f"- AUC {diag['auc']}, 阈值 {diag['youden_threshold']}, 混淆矩阵 {diag['confusion']}")
    L.append(f"- 假阴 (FN, n={diag['fn']['n']}): max_area 中位/均值/n = {diag['fn']['max_area_mm2_med/mean/n']}, DWI 缺失率 {diag['fn']['dwi_missing_rate']}, 临床中位 {diag['fn']['clin_median']}")
    L.append(f"- 假阳 (FP, n={diag['fp']['n']}): max_area 中位/均值/n = {diag['fp']['max_area_mm2_med/mean/n']}, DWI 缺失率 {diag['fp']['dwi_missing_rate']}, 临床中位 {diag['fp']['clin_median']}")
    L.append(f"- 真阳 (TP, n={diag['tp']['n']}): max_area {diag['tp']['max_area_mm2_med/mean/n']}, DWI 缺失率 {diag['tp']['dwi_missing_rate']}")
    L.append(f"- 真阴 (TN, n={diag['tn']['n']}): max_area {diag['tn']['max_area_mm2_med/mean/n']}, DWI 缺失率 {diag['tn']['dwi_missing_rate']}")
    L.append("")
    L.append("## 7. Winner_Model 精确配置")
    L.append("```")
    L.append("模态:  clin_clean (7维, 折内 median impute + std + 嵌套选C L2-LR)")
    L.append("       t:    features/slice_feats_v2_tight60_m10/dinov2/t, AreaWeightedMean 池化 (mask_area_mm2 权重)")
    L.append("       d:    features/slice_feats_v2/dinov2/d, MeanMax 池化 (仅 d_self 104 例)")
    L.append("融合:  策略B = 0.5*z_clin + 0.5*mean(z_t, z_d) (fuse_zlogit_strategy)")
    L.append("校准:  bonly (platt_fit_bonly 逐折)")
    L.append("评估:  10×5 重复分层 CV (StratifiedKFold 5折 shuffle seed0-9), Pooled OOF AUC")
    L.append("结果:  direct_td 0.7003±0.0302 | direct_tcd 0.7099±0.0323")
    L.append("```")
    L.append("")
    L.append("## 8. 结论")
    L.append("- 唯一明确增益: t 序列裁剪视野 120mm→tight60_m10 (bbox 中点+对称补黑, 边长=max(60,span+20)), +0.024 (0.6763→0.7003)。")
    L.append("- 物理尺度标量、DWI from_t 纳回、Patch Token 双流、放射组学、Stacking 均无增益或劣化。")
    L.append("- 保留 H1.1 为 Winner_Model, 未触碰 20250428 外部验证集与 nCRT 队列 (铁律3)。")
    L.append("")
    with open(os.path.join(OUTDIR, "final_autonomous_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("saved ->", os.path.join(OUTDIR, "final_autonomous_report.md"), "+ all_experiments.json")

if __name__ == "__main__":
    main()
