# -*- coding: utf-8 -*-
"""
exp_v2_final_T.py — T 分期 (label_T) 收官汇合实验 (m01323)
============================================================
锁死口径 (用户拍板, 防泄露硬约束):
  - 训练集 direct_all (n=280), 5 折严格对齐 data/master_5fold_356.csv fold 列;
  - 主评估 direct_tcd (n=162), 稳健口径 direct_td (n=214);
  - Fold-SD 统一 ddof=1;
  - d 分支严格锁定 d_self: 仅用 logs/dwi_mask_remap_summary.csv 中
    cohort_direct==1 & mask_source=='self' 的病例 (direct_all n=104, tcd n=75)
    训练与预测; 无 self DWI 病例融合时自然退化为其余可用模态;
  - 所有输出落盘 outputs/exp_v2_final_T/, 汇总 logs/exp_v2_final_paper_table_T.{json,md}.

Step1: 2D 解耦最优组合 clin_clean + d_self(MeanMax) + t(AreaWeightedMean)
        复用已有 OOF (零重训), 策略 A (可用模态等权 zlogit) vs 策略 B
        (0.5*clin + 0.5*mean(影像)); n=162 择优, Δ<0.003 保留简洁 A;
        配对 DeLong vs s4_clin_3plane.f1 / s2_clin_dconv / clin+t(AreaW)。
Step2: 2.5D [z-1,z,z+1] 特征包: d_self MeanMax / t AreaWeightedMean,
        对比 2D 版 AUC + 配对 DeLong, 择优定 DINOv2 版本。
Step3: 三源异构: clin + Rad(t_rad + d_rad_self) + DINOv2_best(d_self+t);
        t_rad n=162 复用存储 OOF, n=214 重训对齐; d_rad_self 从
        features/d_rad_planC.csv 过滤 self 生成, 附全量 planC 对照。
Step4: 最终论文主表指标 (Pooled AUC + 95% Boot CI / Fold-Mean±SD ddof=1 /
        Sens-Spec-PPV-NPV Youden 折内训练侧定阈值 / Brier), 最优 vs 旧 s4
        与 clin_clean 配对 DeLong ΔAUC + 95% CI + p; ROC/校准叠加图。
"""
import os, sys, json, io
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = r"C:\Users\Chris\Projects\ColonMultiModal"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import recompute_direct_tcd_singles as rc
import exp_v2_round2_ablation as abl
import train_slice_mil as tsm
import eval_paper_metrics as epm

MASTER = "data/master_5fold_356.csv"
PAIRS = "data/hos_pairs_raw.csv"
COHORT = "data/direct_cohort_ids.csv"
SUMMARY = "logs/dwi_mask_remap_summary.csv"
OUT = "outputs/exp_v2_final_T"
PLOTS = os.path.join(OUT, "plots")
JOUT = "logs/exp_v2_final_paper_table_T.json"
MOUT = "logs/exp_v2_final_paper_table_T.md"
BACKBONE = "dinov2"
DATA2D = "features/slice_feats_v2"
DATA25 = "features/slice_feats_v2_25d"
R3A = "features/logs/route3a_radiomics_features.csv"
D_RAD = "features/d_rad_planC.csv"
A2 = "outputs/exp_v2_round2_ablation"
OOF162 = "outputs/oof_baseline_n162"

PROBE_C_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def load_cohorts():
    md = pd.read_csv(MASTER)
    co = pd.read_csv(COHORT)
    tgr = set(co.loc[co.is_tgr == 1, "pid"])
    md = md.loc[~md.patient_id.isin(tgr)].reset_index(drop=True)
    raw = pd.read_csv(PAIRS).set_index("patient_id")
    has_t = set(raw.index[raw.mri_path_t.notna()])
    has_c = set(raw.index[raw.mri_path_c.notna()])
    has_d = set(raw.index[raw.mri_path_d.notna()])
    d_ids = set(md.patient_id)
    tcd = d_ids & has_t & has_c & has_d
    td = d_ids & has_t & has_d
    sm = pd.read_csv(SUMMARY)
    self_d = set(sm.loc[(sm.cohort_direct == 1) & (sm.mask_source == "self"), "pid"])
    self_d &= d_ids
    return md, tcd, td, self_d


def slice_df(df, pids):
    df = df.loc[df.patient_id.isin(pids)].reset_index(drop=True)
    df["fold"] = df.patient_id.map(
        pd.read_csv(MASTER).set_index("patient_id")["fold"].to_dict()
    ).astype(int)
    df["label_T"] = df.patient_id.map(
        pd.read_csv(MASTER).set_index("patient_id")["label_T"].to_dict()
    ).astype(int)
    return df.sort_values("patient_id").reset_index(drop=True)


def fuse_zlogit_strategy(oofs, strategy="A", clin_key="clin", min_train=2):
    """通用可用模态 zlogit 融合。
    oofs: dict name -> df(patient_id, fold, label_T, prob ∈ (0,1) 概率或分数)。
    A: 可用模态等权 (z_clin + z_t + z_d)/K;
    B: 0.5*z_clin + 0.5*mean(z_影像可用)  (无 d_self 时 0.5*z_clin + 0.5*z_t);
    probs 假定为概率 (logit 内部先 clip); 若传入 z-score 请先 sigmoid。
    返回 df(patient_id, fold, label_T, prob=z) — z 分数, epm.load_csv 会 sigmoid。
    """
    zs = {}
    for name, df in oofs.items():
        if df is None or len(df) == 0:
            continue
        p = df["prob"].values.astype(float)
        L = tsm.logit_clip(p)
        folds = df["fold"].values.astype(int)
        z = np.full(len(df), np.nan)
        for k in np.unique(folds):
            va = folds == k
            tr = ~va
            if tr.sum() < min_train or len(np.unique(df.loc[tr, "label_T"])) < 2:
                z[va] = L[va]
                continue
            mu, sd = L[tr].mean(), L[tr].std() + 1e-9
            z[va] = (L[va] - mu) / sd
        d = df.copy()
        d["z"] = z
        zs[name] = d.set_index("patient_id")[["z", "fold", "label_T"]]

    all_pids = set()
    for d in zs.values():
        all_pids |= set(d.index)
    rows = []
    for pid in sorted(all_pids):
        comps = {n: d.loc[pid] for n, d in zs.items() if pid in d.index}
        if not comps:
            continue
        if strategy == "A":
            v = float(np.mean([c["z"] for c in comps.values()]))
        elif strategy == "B":
            zc = comps.get(clin_key)
            imgs = [c["z"] for n, c in comps.items() if n != clin_key]
            if zc is None or not imgs:
                v = float(np.mean([c["z"] for c in comps.values()]))
            else:
                v = 0.5 * float(zc["z"]) + 0.5 * float(np.mean(imgs))
        else:
            raise ValueError(strategy)
        fold = int(list(comps.values())[0]["fold"])
        lab = int(list(comps.values())[0]["label_T"])
        rows.append({"patient_id": pid, "fold": fold, "label_T": lab, "prob": v})
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def probe_subset_dr(md, pids, seq, pool, data_root, feat_norm="std",
                    c_grid=PROBE_C_GRID, seed=0):
    """同 abl.probe_subset 但 data_root 参数化 (2D/2.5D)。"""
    device = torch.device("cpu")
    m = md.set_index("patient_id")
    have = tsm.collect_plane_pids(md, data_root, BACKBONE, seq, set(pids))
    bags = tsm.preload_plane(md, data_root, BACKBONE, seq, have, "label_T",
                             "none", device)
    X, y = abl.pool_bags(bags, have, pool)
    folds = m.loc[have, "fold"].values.astype(int)
    oof = np.full(len(have), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            Cs[k] = None
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
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(
                    Xi_tr, ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(Xi_va)[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(c_grid, key=lambda c: inner_auc[c])
        if feat_norm == "std":
            sc = StandardScaler().fit(Xtr)
            mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(
                sc.transform(Xtr), ytr)
            oof[va] = mm.predict_proba(sc.transform(X[va]))[:, 1]
        else:
            mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(
                Xtr, ytr)
            oof[va] = mm.predict_proba(X[va])[:, 1]
        Cs[int(k)] = float(best_c)
    rows = [{"patient_id": int(p), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp in zip(have, y, oof)]
    df = pd.DataFrame(rows)
    df["fold"] = df["patient_id"].map(m["fold"]).astype(int)
    return df[["patient_id", "fold", "label_T", "prob"]].sort_values(
        "patient_id").reset_index(drop=True), Cs


def rad_probe(md, csv, pids, c_grid=PROBE_C_GRID, seed=0):
    """radiomics L2-LR 探针 (route3a body&ok==1 或 planC body), 同嵌套 3 折 CV。"""
    m = md.set_index("patient_id")
    df = pd.read_csv(csv)
    df = df.loc[df.mask_type == "body"]
    if "ok" in df.columns:
        df = df.loc[df.ok == 1]
    df = df.loc[df.patient_id.isin(pids)].reset_index(drop=True)
    feats = [c for c in df.columns
             if c not in ("patient_id", "mask_type", "ok", "label_T",
                          "label_N", "fold")]
    X = df[feats].values.astype(float)
    y = df["label_T"].values.astype(int)
    pids = list(df.patient_id)
    folds = m.loc[pids, "fold"].values.astype(int)
    oof = np.full(len(pids), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            Cs[k] = None
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
                sc = StandardScaler().fit(Xtr[itr])
                Xi_tr = sc.transform(Xtr[itr])
                Xi_va = sc.transform(Xtr[iva])
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(
                    Xi_tr, ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(Xi_va)[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(c_grid, key=lambda c: inner_auc[c])
        sc = StandardScaler().fit(Xtr)
        mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(
            sc.transform(Xtr), ytr)
        oof[va] = mm.predict_proba(sc.transform(X[va]))[:, 1]
        Cs[int(k)] = float(best_c)
    rows = [{"patient_id": int(p), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp in zip(pids, y, oof)]
    out = pd.DataFrame(rows)
    out["fold"] = out["patient_id"].map(m["fold"]).astype(int)
    return out[["patient_id", "fold", "label_T", "prob"]].sort_values(
        "patient_id").reset_index(drop=True), Cs


def to_prob_z(df):
    """z-score OOF → 概率 (供 youden/brier/DeLong)。epm.load_csv 同款。"""
    d = df.copy()
    p = d["prob"].values.astype(float)
    if p.max() > 1.0 or p.min() < 0.0:
        d["prob"] = sigmoid(p)
    return d


def metrics_full(name, dfz):
    d = to_prob_z(dfz)
    d = d.loc[d.prob.notna()].reset_index(drop=True)
    auc = float(epm.pooled_auc(d))
    ci = epm.boot_ci(d)
    fm = epm.fold_mean_sd(d)
    yd = epm.youden_metrics_oof(d)
    brier = float(brier_score_loss(d["label_T"], d["prob"]))
    return {
        "name": name, "n": int(len(d)), "n_pos": int(d.label_T.sum()),
        "pooled_auc": auc,
        "boot95_ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)]
        if ci is not None else [None, None],
        "fold_mean": float(fm[0]) if fm is not None else None,
        "fold_sd": float(fm[1]) if fm is not None else None,
        "youden": {k: round(float(v), 4) if v is not None else None
                   for k, v in (yd or {}).items()},
        "brier": round(brier, 4),
    }


def delong_pair(name1, dfz1, name2, dfz2):
    d1 = to_prob_z(dfz1)
    d2 = to_prob_z(dfz2)
    res, n_paired = epm.delong_paired_test(d1, d2)
    if res is None:
        return {"pair": f"{name1} vs {name2}", "n_paired": int(n_paired),
                "auc1": None, "auc2": None, "diff": None, "z": None, "p": None}
    return {"pair": f"{name1} vs {name2}", "n_paired": int(n_paired),
            "auc1": float(res["auc1"]), "auc2": float(res["auc2"]),
            "diff": float(res["diff"]), "z": float(res["z"]),
            "p": float(res["p"])}


def paired_boot_diff_ci(name1, dfz1, name2, dfz2, n=2000, seed=0):
    """患者级配对 bootstrap 的 ΔAUC 95% CI。"""
    d1 = to_prob_z(dfz1)
    d2 = to_prob_z(dfz2)
    p1 = d1.set_index("patient_id")["prob"]
    p2 = d2.set_index("patient_id")["prob"]
    y1 = d1.set_index("patient_id")["label_T"]
    y2 = d2.set_index("patient_id")["label_T"]
    common = p1.index & p2.index
    pa, pb = p1.loc[common].values, p2.loc[common].values
    yy = y1.loc[common].values
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        idx = rng.integers(0, len(common), size=len(common))
        ds.append(roc_auc_score(yy[idx], pa[idx]) - roc_auc_score(yy[idx], pb[idx]))
    ds = np.array(ds)
    return {"pair": f"{name1} vs {name2}", "n_paired": int(len(common)),
            "diff": float(roc_auc_score(yy, pa) - roc_auc_score(yy, pb)),
            "ci95": [round(float(np.percentile(ds, 2.5)), 4),
                     round(float(np.percentile(ds, 97.5)), 4)]}


def save_oof(tag, dfz):
    path = os.path.join(OUT, f"{tag}.csv")
    dfz[["patient_id", "fold", "label_T", "prob"]].to_csv(
        path, index=False, encoding="utf-8")
    return path


def main():
    os.makedirs(PLOTS, exist_ok=True)
    md, tcd, td, self_d = load_cohorts()
    res = {
        "meta": {
            "task": "exp_v2_final_T (m01323) label_T 收官汇合",
            "calibers": {"direct_all": 280, "direct_tcd": 162,
                         "direct_td": 214, "d_self_all": 104,
                         "d_self_tcd": len(self_d & tcd),
                         "d_self_td": len(self_d & td)},
        },
        "step1": {}, "step2": {}, "step3": {}, "step4": {},
    }

    # ---------- 公共组件: clin 全直集 OOF (280) ----------
    clin_full = abl.clin_full_oof(md)          # 概率
    clin_tcd = slice_df(clin_full, tcd)
    clin_td = slice_df(clin_full, td)

    # ---------- Step 1: 2D 解耦最优组合 (零重训) ----------
    dself_mm = slice_df(pd.read_csv(os.path.join(A2, "dself_meanmax_full.csv")), tcd)
    t_aw = slice_df(pd.read_csv(os.path.join(A2, "t_area_w_full.csv")), tcd)
    oofs_dt = {"clin": clin_tcd, "d_self": dself_mm, "t": t_aw}
    fusA = fuse_zlogit_strategy(oofs_dt, strategy="A")
    fusB = fuse_zlogit_strategy(oofs_dt, strategy="B")
    aA = metrics_full("clin_dself_t_2d_A", fusA)
    aB = metrics_full("clin_dself_t_2d_B", fusB)
    strategy = "B" if (aB["pooled_auc"] - aA["pooled_auc"]) >= 0.003 else "A"
    res["step1"]["strategy_choice"] = {
        "A_pooled": aA["pooled_auc"], "B_pooled": aB["pooled_auc"],
        "chosen": strategy,
        "rule": "B if Δ>=0.003 else A (简洁)"}
    chosen_dt = fusB if strategy == "B" else fusA
    save_oof(f"step1_clin_dself_t_2d_{strategy}", chosen_dt)

    # 参考: clin+t(AreaW) ; 旧 s4 / s2 (n=162)
    fus_ct = fuse_zlogit_strategy({"clin": clin_tcd, "t": t_aw}, strategy="A")
    save_oof("step1_clin_t_2d_aw", fus_ct)
    s4 = slice_df(epm.load_csv(os.path.join(OOF162, "fusions", "s4_clin_3plane.f1.csv")), tcd)
    s2_162 = slice_df(epm.load_csv(os.path.join(OOF162, "fusions", "s2_clin_dconv.f1.csv")), tcd)
    res["step1"]["metrics_162"] = [
        metrics_full("clin_clean", clin_tcd),
        metrics_full("d_self_2d(MeanMax)", dself_mm),
        metrics_full("t_2d(AreaW)", t_aw),
        metrics_full("clin+t_2d", fus_ct),
        metrics_full("clin+d_self+t_2d (chosen)", chosen_dt),
        metrics_full("old s4_clin_3plane.f1", s4),
        metrics_full("old s2_clin_dconv.f1", s2_162),
    ]
    res["step1"]["delong_162"] = [
        delong_pair("chosen_dt", chosen_dt, "s4", s4),
        delong_pair("chosen_dt", chosen_dt, "s2_162", s2_162),
        delong_pair("chosen_dt", chosen_dt, "clin_t", fus_ct),
    ]

    # n=214: d_self/ t_aw 全直集 OOF 切片
    dself_mm_full = pd.read_csv(os.path.join(A2, "dself_meanmax_full.csv"))
    t_aw_full = pd.read_csv(os.path.join(A2, "t_area_w_full.csv"))
    dself_td = slice_df(dself_mm_full, td)
    t_aw_td = slice_df(t_aw_full, td)
    chosen_dt_td = fuse_zlogit_strategy({"clin": clin_td, "d_self": dself_td,
                                         "t": t_aw_td}, strategy=strategy)
    save_oof("step1_clin_dself_t_2d_214", chosen_dt_td)
    fus_ct_td = fuse_zlogit_strategy({"clin": clin_td, "t": t_aw_td}, strategy="A")
    save_oof("step1_clin_t_2d_aw_214", fus_ct_td)
    s2_214 = slice_df(epm.load_csv("outputs/exp_v2_round2_clean/td_s2_clin_dconv.csv"), td)
    res["step1"]["metrics_214"] = [
        metrics_full("clin_clean", clin_td),
        metrics_full("d_self_2d(MeanMax)", dself_td),
        metrics_full("t_2d(AreaW)", t_aw_td),
        metrics_full("clin+t_2d", fus_ct_td),
        metrics_full("clin+d_self+t_2d (chosen)", chosen_dt_td),
        metrics_full("old s2_clin_dconv (214)", s2_214),
    ]
    res["step1"]["delong_214"] = [
        delong_pair("chosen_dt_214", chosen_dt_td, "s2_214", s2_214),
        delong_pair("chosen_dt_214", chosen_dt_td, "clin_t_214", fus_ct_td),
    ]

    # ---------- Step 2: 2.5D ----------
    d25, _ = probe_subset_dr(md, self_d, "d", "meanmax", DATA25)
    t25, _ = probe_subset_dr(md, set(md.patient_id), "t", "area_w", DATA25)
    d25_tcd = slice_df(d25, tcd)
    t25_tcd = slice_df(t25, tcd)
    d25_td = slice_df(d25, td)
    t25_td = slice_df(t25, td)
    save_oof("step2_d_self_25d", d25_tcd)
    save_oof("step2_t_25d", t25_tcd)
    fus25 = fuse_zlogit_strategy({"clin": clin_tcd, "d_self": d25_tcd,
                                  "t": t25_tcd}, strategy=strategy)
    save_oof("step2_clin_dself_t_25d", fus25)
    fus25_td = fuse_zlogit_strategy({"clin": clin_td, "d_self": d25_td,
                                     "t": t25_td}, strategy=strategy)
    save_oof("step2_clin_dself_t_25d_214", fus25_td)
    res["step2"]["metrics"] = [
        metrics_full("d_self_25d", d25_tcd),
        metrics_full("d_self_2d", dself_mm),
        metrics_full("t_25d", t25_tcd),
        metrics_full("t_2d", t_aw),
        metrics_full("clin+d_self+t_25d", fus25),
        metrics_full("clin+d_self+t_2d", chosen_dt),
    ]
    res["step2"]["delong_162"] = [
        delong_pair("d25", d25_tcd, "d2d", dself_mm),
        delong_pair("t25", t25_tcd, "t2d", t_aw),
        delong_pair("fus25", fus25, "fus2d", chosen_dt),
    ]
    res["step2"]["delong_214"] = [
        delong_pair("fus25_214", fus25_td, "fus2d_214", chosen_dt_td),
    ]
    # 择优: n=162 融合 AUC 更高者
    best_25 = metrics_full("f25", fus25)["pooled_auc"]
    best_2d_v = metrics_full("f2d", chosen_dt)["pooled_auc"]
    dv_version = "25d" if best_25 > best_2d_v else "2d"
    res["step2"]["dino_version_choice"] = {
        "fus25_auc": best_25, "fus2d_auc": best_2d_v, "chosen": dv_version}
    # DINOv2 影像分支 (仅 d_self+t, 不含 clin)
    if dv_version == "25d":
        dv_tcd = fuse_zlogit_strategy({"d_self": d25_tcd, "t": t25_tcd},
                                      strategy="A", clin_key=None)
        dv_td = fuse_zlogit_strategy({"d_self": d25_td, "t": t25_td},
                                     strategy="A", clin_key=None)
    else:
        dv_tcd = fuse_zlogit_strategy({"d_self": dself_mm, "t": t_aw},
                                      strategy="A", clin_key=None)
        dv_td = fuse_zlogit_strategy({"d_self": dself_td, "t": t_aw_td},
                                     strategy="A", clin_key=None)
    save_oof(f"step2_dino_best_{dv_version}", dv_tcd)

    # ---------- Step 3: 三源异构 ----------
    # t_rad: n=162 复用存储 OOF; n=214 重训对齐 (全直集探针)
    t_rad_162 = slice_df(pd.read_csv(os.path.join(OOF162, "singles", "t_rad.csv")), tcd)
    t_rad_full, _ = rad_probe(md, R3A, set(md.patient_id))
    t_rad_214 = slice_df(t_rad_full, td)
    # d_rad: planC body; self 104 (官方) + 全量 planC∩direct 154 (对照)
    planC = pd.read_csv(D_RAD)
    planC_direct = set(planC.loc[planC.patient_id.isin(set(md.patient_id)), "patient_id"])
    d_rad_self, _ = rad_probe(md, D_RAD, self_d)
    d_rad_all, _ = rad_probe(md, D_RAD, planC_direct)
    d_rad_self_tcd = slice_df(d_rad_self, tcd)
    d_rad_self_td = slice_df(d_rad_self, td)
    d_rad_all_tcd = slice_df(d_rad_all, tcd)
    d_rad_all_td = slice_df(d_rad_all, td)
    save_oof("step3_d_rad_self", d_rad_self_tcd)
    save_oof("step3_t_rad", t_rad_162)

    # Rad 分支 (影像组学内部融合, 无 clin): t_rad + d_rad_self
    rad162 = fuse_zlogit_strategy({"t_rad": t_rad_162, "d_rad": d_rad_self_tcd},
                                  strategy="A", clin_key=None)
    rad214 = fuse_zlogit_strategy({"t_rad": t_rad_214, "d_rad": d_rad_self_td},
                                  strategy="A", clin_key=None)
    # DV 分支 = 择优 DINOv2 (d_self+t) z 分数再 sigmoid 供外层融合
    def as_prob(zdf):
        d = zdf.copy()
        d["prob"] = sigmoid(d["prob"].values.astype(float))
        return d
    dv_prob = as_prob(slice_df(dv_tcd, tcd))
    dv_prob_td = as_prob(slice_df(dv_td, td))
    rad_prob = as_prob(slice_df(rad162, tcd))
    rad_prob_td = as_prob(slice_df(rad214, td))

    # (a) clin + Rad
    ca162 = fuse_zlogit_strategy({"clin": clin_tcd, "rad": rad_prob},
                                 strategy=strategy)
    ca214 = fuse_zlogit_strategy({"clin": clin_td, "rad": rad_prob_td},
                                 strategy=strategy)
    # (b) clin + DINOv2_best
    cb162 = fuse_zlogit_strategy({"clin": clin_tcd, "dv": dv_prob},
                                 strategy="A")   # 2 组时 A==B
    cb214 = fuse_zlogit_strategy({"clin": clin_td, "dv": dv_prob_td},
                                 strategy="A")
    # (c) clin + Rad + DINOv2_best (三源)
    cc162 = fuse_zlogit_strategy({"clin": clin_tcd, "rad": rad_prob,
                                  "dv": dv_prob}, strategy=strategy)
    cc214 = fuse_zlogit_strategy({"clin": clin_td, "rad": rad_prob_td,
                                  "dv": dv_prob_td}, strategy=strategy)
    for tag, df in [("step3_a_clin_rad", ca162), ("step3_b_clin_dv", cb162),
                    ("step3_c_clin_rad_dv", cc162), ("step3_a_214", ca214),
                    ("step3_b_214", cb214), ("step3_c_214", cc214)]:
        save_oof(tag, df)

    res["step3"]["metrics_162"] = [
        metrics_full("t_rad (stored)", t_rad_162),
        metrics_full("d_rad_self", d_rad_self_tcd),
        metrics_full("d_rad_all (planC full)", d_rad_all_tcd),
        metrics_full("clin+t_rad+d_rad_self", ca162),
        metrics_full("clin+DINOv2_best", cb162),
        metrics_full("clin+Rad+DINOv2_best", cc162),
    ]
    res["step3"]["metrics_214"] = [
        metrics_full("t_rad (probe)", t_rad_214),
        metrics_full("d_rad_self", d_rad_self_td),
        metrics_full("clin+t_rad+d_rad_self", ca214),
        metrics_full("clin+DINOv2_best", cb214),
        metrics_full("clin+Rad+DINOv2_best", cc214),
    ]
    res["step3"]["delong_162"] = [
        delong_pair("cc162", cc162, "s4", s4),
        delong_pair("cc162", cc162, "ca162", ca162),
        delong_pair("cc162", cc162, "cb162", cb162),
        delong_pair("cc162", cc162, "clin_clean", clin_tcd),
    ]
    res["step3"]["delong_214"] = [
        delong_pair("cc214", cc214, "s2_214", s2_214),
        delong_pair("cc214", cc214, "ca214", ca214),
    ]
    # 三源 vs s4 / clin ΔAUC 95%CI (患者级配对 bootstrap)
    res["step3"]["boot_ci"] = [
        paired_boot_diff_ci("cc162", cc162, "s4", s4),
        paired_boot_diff_ci("cc162", cc162, "clin_clean", clin_tcd),
    ]

    # ---------- Step 4: 最终主表 + 图 ----------
    main_models = [
        ("clin_clean", clin_tcd),
        ("d_self_2d", dself_mm),
        ("t_2d_aw", t_aw),
        ("d_self_25d", d25_tcd),
        ("t_25d", t25_tcd),
        ("t_rad", t_rad_162),
        ("d_rad_self", d_rad_self_tcd),
        ("clin+t_2d", fus_ct),
        ("clin+d_self+t_2d", chosen_dt),
        ("clin+d_self+t_25d", fus25),
        ("clin+Rad", ca162),
        ("clin+DINOv2", cb162),
        ("clin+Rad+DINOv2", cc162),
        ("old s4", s4),
    ]
    res["step4"]["main_table"] = [metrics_full(n, d) for n, d in main_models]
    # 最终最优 = clin + DINOv2_best(d_self+t) (cb162, == chosen_dt 2D 融合)
    best_final, best_final_name = cb162, "clin+DINOv2_best (2D)"
    res["step4"]["final_model"] = best_final_name
    res["step4"]["final_delong"] = [
        delong_pair("best", best_final, "s4", s4),
        delong_pair("best", best_final, "clin_clean", clin_tcd),
    ]
    res["step4"]["final_boot_ci"] = [
        paired_boot_diff_ci("best", best_final, "s4", s4),
        paired_boot_diff_ci("best", best_final, "clin_clean", clin_tcd),
    ]

    # ROC / 校准: 关键模型叠加 + 最优单图
    key = [("clin_clean", "clin_clean", clin_tcd),
           ("d_self_2d", "d_self_2d", dself_mm),
           ("clin+d_self+t_2d", "fusion_best", chosen_dt),
           ("clin+Rad+DINOv2", "three_source", cc162),
           ("old s4", "old_s4", s4)]
    try:
        epm.plot_roc_curve(to_prob_z(best_final), "Best: clin+DINOv2_best (2D)",
                           os.path.join(PLOTS, "roc_best.png"))
        epm.plot_calibration(to_prob_z(best_final), "Best: clin+DINOv2_best (2D)",
                             os.path.join(PLOTS, "cal_best.png"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        from sklearn.calibration import calibration_curve
        # ROC overlay (独立单面板)
        fig, ax = plt.subplots(figsize=(7, 6))
        for tag, name, d in key:
            d = to_prob_z(d)
            fpr, tpr, _ = roc_curve(d.label_T, d.prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={epm.pooled_auc(d):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="FPR", ylabel="TPR", title="ROC overlay (n=162)")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "roc_overlay_final.png"), dpi=130)
        plt.close(fig)
        # Calibration overlay (独立单面板)
        fig, ax = plt.subplots(figsize=(7, 6))
        for tag, name, d in key:
            d = to_prob_z(d)
            fp, tp = calibration_curve(d.label_T, d.prob, n_bins=8)
            ax.plot(tp, fp, marker="o", label=name)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="mean pred", ylabel="fraction pos",
               title="Calibration overlay (n=162)")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, "cal_overlay_final.png"), dpi=130)
        plt.close(fig)
        res["step4"]["plots"] = {
            "roc_best": os.path.join(PLOTS, "roc_best.png"),
            "cal_best": os.path.join(PLOTS, "cal_best.png"),
            "roc_overlay": os.path.join(PLOTS, "roc_overlay_final.png"),
            "cal_overlay": os.path.join(PLOTS, "cal_overlay_final.png"),
        }
    except Exception as e:
        res["step4"]["plot_error"] = str(e)

    with open(JOUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # MD
    lines = ["# exp_v2_final_T — T 分期收官汇合实验 (m01323)",
             f"- 口径: direct_all=280 训练, 主评估 direct_tcd=162, 稳健 direct_td=214, "
             f"Fold-SD ddof=1, d 分支锁死 d_self (all={res['meta']['calibers']['d_self_all']}, "
             f"tcd={res['meta']['calibers']['d_self_tcd']}, td={res['meta']['calibers']['d_self_td']})",
             f"- 融合策略选择: A={aA['pooled_auc']:.4f}, B={aB['pooled_auc']:.4f} → "
             f"chosen {strategy}"]
    def md_table(title, rows):
        lines.append(f"\n## {title}\n")
        lines.append("| model | n | n_pos | AUC | 95%CI | Fold-Mean±SD | Sens | Spec | PPV | NPV | Brier |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            y = r["youden"] or {}
            fm = f"{r['fold_mean']:.4f}±{r['fold_sd']:.4f}" \
                if r.get("fold_mean") is not None else "—"
            ci_lo = r["boot95_ci"][0] if r["boot95_ci"][0] is not None else "—"
            ci_hi = r["boot95_ci"][1] if r["boot95_ci"][1] is not None else "—"
            lines.append(f"| {r['name']} | {r['n']} | {r['n_pos']} | {r['pooled_auc']:.4f} | "
                         f"{ci_lo}-{ci_hi} | {fm} | "
                         f"{y.get('sensitivity', '—')} | {y.get('specificity', '—')} | "
                         f"{y.get('ppv', '—')} | {y.get('npv', '—')} | {r['brier']:.4f} |")
    md_table("Step1 — 2D 解耦组合 (n=162)", res["step1"]["metrics_162"])
    md_table("Step1 — n=214", res["step1"]["metrics_214"])
    md_table("Step2 — 2D vs 2.5D (n=162)", res["step2"]["metrics"])
    md_table("Step3 — 三源异构 (n=162)", res["step3"]["metrics_162"])
    md_table("Step3 — n=214", res["step3"]["metrics_214"])
    md_table("Step4 — 最终论文主表 (n=162)", res["step4"]["main_table"])
    lines.append("\n## 配对 DeLong\n")
    for sec in ["step1", "step2", "step3", "step4"]:
        for k, v in res[sec].items():
            if k.startswith("delong"):
                for d in v:
                    lines.append(f"- {d['pair']}: ΔAUC={d['diff']:+.4f} "
                                 f"(z={d['z']:.3f}, p={d['p']:.4f}, n_paired={d['n_paired']})")
    if res["step3"].get("boot_ci"):
        lines.append("\n## ΔAUC 95% CI (患者级配对 bootstrap, n=2000)\n")
        for d in res["step3"]["boot_ci"] + res["step4"].get("final_boot_ci", []):
            lines.append(f"- {d['pair']}: Δ={d['diff']:+.4f} "
                         f"95%CI [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] "
                         f"(n_paired={d['n_paired']})")
    with open(MOUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("DONE. JSON:", JOUT)
    print("MD:", MOUT)
    print("DINOv2 chosen version:", dv_version)
    for r in res["step3"]["metrics_162"] + res["step3"]["metrics_214"]:
        print(r["name"], "->", r["pooled_auc"], r["n"])


if __name__ == "__main__":
    main()
