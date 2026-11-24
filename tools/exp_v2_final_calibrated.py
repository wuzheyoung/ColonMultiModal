# -*- coding: utf-8 -*-
"""
exp_v2_final_calibrated.py — 收官融合模型 Platt 校准 + 校准后指标重算 (m01538 任务1)
==================================================================================
背景: zlogit 融合输出 z-score, 经 sigmoid 后预测均值居中于 0.50, 偏离先验阳性率
      (tcd 110/162=0.679), 导致 best Brier=0.228 > 零技能常数基准 0.218。
修正: 对每个融合模型的 z_fused 增加【严格训练折内拟合】的 1D 逻辑校准
      (Platt Scaling: LogisticRegression 拟合 训练折 z_fused -> y_train,
       再映射验证折概率; 严禁触碰验证折标签)。
输出: logs/exp_v2_final_calibrated_T.{json,md} + outputs/exp_v2_final_T/calibrated/*.csv
     校准后 Brier / ECE(10 分箱) / Youden-OOF(Sens-Spec-PPV-NPV), 核对 Pooled AUC
     保持不变, 并统一 Step1/Step4 最优模型 vs 旧 s4 的 DeLong (0.355 vs 0.359 溯源)。
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = r"C:\\Users\\Chris\\Projects\\ColonMultiModal"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eval_paper_metrics as epm
import exp_v2_final_T as F  # 复用 load_cohorts / slice_df / metrics_full / sigmoid

OUT = "outputs/exp_v2_final_T"
CAL = os.path.join(OUT, "calibrated")
JOUT = "logs/exp_v2_final_calibrated_T.json"
MOUT = "logs/exp_v2_final_calibrated_T.md"
OOF162 = "outputs/oof_baseline_n162"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def platt_calibrate(dfz, min_train=8, C=1.0):
    """dfz: patient_id, fold, label_T, prob(=z-score, 可越界)。
    逐折: 用训练折 (fold != k) 的 z -> y 拟合 1D LogisticRegression (Platt),
    映射验证折 (fold == k) 概率。退化折回退 sigmoid(z)。
    返回 prob ∈ (0,1) 的 df (AUC 单调变换逐折不变, Pooled AUC 仅由跨折排序微扰)。
    """
    d = dfz.copy()
    z = d["prob"].values.astype(float)
    folds = d["fold"].values.astype(int)
    y = d["label_T"].values.astype(int)
    cal = np.full(len(d), np.nan)
    for k in np.unique(folds):
        va = folds == k
        tr = ~va
        if tr.sum() < min_train or len(np.unique(y[tr])) < 2:
            cal[va] = sigmoid(z[va])
            continue
        mm = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(
            z[tr].reshape(-1, 1), y[tr])
        cal[va] = mm.predict_proba(z[va].reshape(-1, 1))[:, 1]
    d["prob"] = cal
    return d


def ece10(df):
    """ECE, 10 等宽分箱。"""
    p = df["prob"].values.astype(float)
    y = df["label_T"].values.astype(int)
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def cal_metrics(name, dfz, platt=True):
    """校准后全套: AUC (前后核对) / Brier (校准前 sigmoid vs 校准后 Platt) /
    ECE10 / Youden-OOF / Fold-Mean±SD。
    口径说明: AUC 是秩指标 — 全局单调变换 (sigmoid) 下 Pooled AUC 不变;
    逐折 Platt 只保证折内秩不变 (Fold-Mean±SD 不变), 跨折配对可能重排 →
    Pooled AUC 在校准概率上可微移。故 discrimination 以未校准 OOF (sigmoid)
    为官方口径 (0.7400 保持), calibration (Brier/ECE/Youden 阈值) 以 Platt
    校准概率为准; 两者同时报告。"""
    ds = F.to_prob_z(dfz)  # sigmoid (全局单调, 保 Pooled AUC)
    ds = ds.loc[ds.prob.notna()].reset_index(drop=True)
    raw_auc = float(epm.pooled_auc(ds))
    brier_sig = float(brier_score_loss(ds["label_T"], ds["prob"]))
    ece_sig = ece10(ds)
    if platt:
        dc = platt_calibrate(dfz)
    else:
        dc = ds
    dc = dc.loc[dc.prob.notna()].reset_index(drop=True)
    auc_c = float(epm.pooled_auc(dc))
    ci = epm.boot_ci(dc)
    fm = epm.fold_mean_sd(dc)
    yd = epm.youden_metrics_oof(dc)
    brier_cal = float(brier_score_loss(dc["label_T"], dc["prob"]))
    return {
        "name": name, "n": int(len(dc)), "n_pos": int(dc.label_T.sum()),
        "pooled_auc_sigmoid": round(raw_auc, 4),
        "pooled_auc_cal": round(auc_c, 4),
        "brier_sigmoid": round(brier_sig, 4),
        "brier_cal": round(brier_cal, 4),
        "ece_sigmoid": round(ece_sig, 4),
        "ece10_cal": round(ece10(dc), 4),
        "boot95_ci_cal": [round(float(ci[0]), 4), round(float(ci[1]), 4)]
        if ci is not None else [None, None],
        "fold_mean": float(fm[0]) if fm is not None else None,
        "fold_sd": float(fm[1]) if fm is not None else None,
        "youden": {k: round(float(v), 4) if v is not None else None
                   for k, v in (yd or {}).items()},
    }


def delong_pair(name1, dfz1, name2, dfz2, platt1=False, platt2=False):
    """配对 DeLong。官方口径 = sigmoid OOF 概率 (与 Pooled AUC 同源, 全局单调变换
    不影响秩 → 0.7400 保持)。platt=True 时用 Platt 校准概率 (仅供诊断对照)。"""
    d1 = platt_calibrate(dfz1) if platt1 else F.to_prob_z(dfz1)
    d2 = platt_calibrate(dfz2) if platt2 else F.to_prob_z(dfz2)
    res, n_paired = epm.delong_paired_test(d1, d2)
    if res is None:
        return {"pair": f"{name1} vs {name2}", "n_paired": int(n_paired),
                "auc1": None, "auc2": None, "diff": None, "z": None, "p": None}
    return {"pair": f"{name1} vs {name2}", "n_paired": int(n_paired),
            "auc1": round(float(res["auc1"]), 4),
            "auc2": round(float(res["auc2"]), 4),
            "diff": round(float(res["diff"]), 4),
            "z": round(float(res["z"]), 4), "p": round(float(res["p"]), 4)}


def main():
    os.makedirs(CAL, exist_ok=True)
    md, tcd, td, self_d = F.load_cohorts()
    clin_tcd = F.slice_df(F.abl.clin_full_oof(md), tcd)
    clin_td = F.slice_df(F.abl.clin_full_oof(md), td)
    res = {"meta": {
        "task": "exp_v2_final_calibrated (m01538 任务1)",
        "calibers": {"tcd": 162, "td": 214, "tcd_pos_rate": 110 / 162,
                     "zero_skill_brier_tcd": round(110 / 162 * (1 - 110 / 162), 4)},
        "platt": "per-fold train-side LogisticRegression(C=1.0) on z_fused, "
                 "applied to val fold; degenerate fold -> sigmoid(z)",
    }, "calibrated_162": [], "calibrated_214": [], "delong": [],
       "deLong_355_vs_359_explained": None}

    def load(p):
        return pd.read_csv(os.path.join(OUT, p))

    # ---------- n=162 模型集 ----------
    dself_mm = F.slice_df(pd.read_csv(os.path.join("outputs/exp_v2_round2_ablation",
                                                   "dself_meanmax_full.csv")), tcd)
    t_aw = F.slice_df(pd.read_csv(os.path.join("outputs/exp_v2_round2_ablation",
                                               "t_area_w_full.csv")), tcd)
    chosen = load("step1_clin_dself_t_2d_B.csv")
    fus_ct = load("step1_clin_t_2d_aw.csv")
    fus25 = load("step2_clin_dself_t_25d.csv")
    d25 = load("step2_d_self_25d.csv")
    t25 = load("step2_t_25d.csv")
    dv = load("step2_dino_best_2d.csv")
    ca = load("step3_a_clin_rad.csv")
    cb = load("step3_b_clin_dv.csv")
    cc = load("step3_c_clin_rad_dv.csv")
    drad = load("step3_d_rad_self.csv")
    trad = F.slice_df(epm.load_csv(os.path.join(OOF162, "singles", "t_rad.csv")), tcd)
    s4 = epm.load_csv(os.path.join(OOF162, "fusions", "s4_clin_3plane.f1.csv"))
    s2_162 = epm.load_csv(os.path.join(OOF162, "fusions", "s2_clin_dconv.f1.csv"))

    # 单路概率模型: 不加 Platt (已是 LR predict_proba)
    singles_162 = [
        ("clin_clean", clin_tcd, False),
        ("d_self_2d(MeanMax)", dself_mm, False),
        ("t_2d(AreaW)", t_aw, False),
        ("d_self_25d", d25, False),
        ("t_25d", t25, False),
        ("t_rad", trad, False),
        ("d_rad_self", drad, False),
    ]
    fusions_162 = [
        ("clin+t_2d", fus_ct, True),
        ("clin+d_self+t_2d (chosen)", chosen, True),
        ("clin+d_self+t_25d", fus25, True),
        ("clin+DINOv2 (dv)", dv, True),
        ("clin+Rad", ca, True),
        ("clin+DINOv2_best (cb)", cb, True),
        ("clin+Rad+DINOv2", cc, True),
        ("old s4_clin_3plane.f1", s4, True),
        ("old s2_clin_dconv.f1", s2_162, True),
    ]
    for n, d, pl in singles_162 + fusions_162:
        res["calibrated_162"].append(cal_metrics(n, d, platt=pl))
        if pl:
            tag = n.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            platt_calibrate(d)[["patient_id", "fold", "label_T", "prob"]].to_csv(
                os.path.join(CAL, f"{tag}_162.csv"), index=False, encoding="utf-8")

    # ---------- n=214 ----------
    chosen_214 = load("step1_clin_dself_t_2d_214.csv")
    fus_ct_214 = load("step1_clin_t_2d_aw_214.csv")
    fus25_214 = load("step2_clin_dself_t_25d_214.csv")
    ca214 = load("step3_a_214.csv")
    cb214 = load("step3_b_214.csv")
    cc214 = load("step3_c_214.csv")
    s2_214 = epm.load_csv("outputs/exp_v2_round2_clean/td_s2_clin_dconv.csv")
    dself_td = F.slice_df(pd.read_csv(os.path.join("outputs/exp_v2_round2_ablation",
                                                   "dself_meanmax_full.csv")), td)
    t_aw_td = F.slice_df(pd.read_csv(os.path.join("outputs/exp_v2_round2_ablation",
                                                  "t_area_w_full.csv")), td)
    singles_214 = [
        ("clin_clean", clin_td, False),
        ("d_self_2d", dself_td, False),
        ("t_2d(AreaW)", t_aw_td, False),
    ]
    fusions_214 = [
        ("clin+t_2d", fus_ct_214, True),
        ("clin+d_self+t_2d (chosen)", chosen_214, True),
        ("clin+d_self+t_25d", fus25_214, True),
        ("clin+Rad", ca214, True),
        ("clin+DINOv2_best (cb)", cb214, True),
        ("clin+Rad+DINOv2", cc214, True),
        ("old s2_clin_dconv", s2_214, True),
    ]
    for n, d, pl in singles_214 + fusions_214:
        res["calibrated_214"].append(cal_metrics(n, d, platt=pl))
        if pl:
            tag = n.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            platt_calibrate(d)[["patient_id", "fold", "label_T", "prob"]].to_csv(
                os.path.join(CAL, f"{tag}_214.csv"), index=False, encoding="utf-8")

    # ---------- DeLong 统一 (最优 = Step1 chosen_dt; 官方 sigmoid 口径) ----------
    res["delong"].append(delong_pair("best(chosen_2d)", chosen, "old_s4", s4))
    res["delong"].append(delong_pair("best(chosen_2d)", chosen, "clin_clean", clin_tcd))
    res["delong"].append(delong_pair("best(chosen_2d)", chosen, "clin+t_2d", fus_ct))
    res["delong"].append(delong_pair("best_214", chosen_214, "s2_214", s2_214))
    # 校准概率口径的诊断对照 (chosen vs s4): 逐折 Platt 后跨折配对重排,
    # Pooled AUC 变 0.721 vs 0.6705 → DeLong p 移动, 仅作灵敏度说明
    res["delong"].append(delong_pair("best_cal", chosen, "s4_cal (diag)", s4,
                                     platt1=True, platt2=True))
    # 0.355 vs 0.359 溯源: cb(step4) 与 chosen(step1) 是同一融合意图的两种构造
    res["deLong_355_vs_359_explained"] = {
        "step1_chosen_vs_s4_p": delong_pair("chosen", chosen, "s4", s4)["p"],
        "step4_cb_vs_s4_p": delong_pair("cb", cb, "s4", s4)["p"],
        "reason": "cb162 构造时先对 DV 分支 sigmoid 再 zlogit 重校准 (as_prob 步骤), "
                  "与 chosen_dt 直接融合原始 z 的逐样本分数差最大 0.037, "
                  "Pooled AUC 均 0.7400 但 DeLong 方差估计微扰 → p 0.3554 vs 0.3589。"
                  "官方最优 = Step1 chosen_dt (0.3554)。",
    }

    with open(JOUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    def md_table(title, rows):
        lines.append(f"## {title}\n")
        lines.append("| model | n | n_pos | AUC(sigmoid) | AUC(cal) | 95%CI | Fold-Mean±SD | "
                     "Brier(sig) | Brier(cal) | ECE(sig) | ECE10(cal) | Sens | Spec | PPV | NPV |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            y = r["youden"] or {}
            fm = f"{r['fold_mean']:.4f}±{r['fold_sd']:.4f}" \
                if r.get("fold_mean") is not None else "—"
            ci_lo = r["boot95_ci_cal"][0] if r["boot95_ci_cal"][0] is not None else "—"
            ci_hi = r["boot95_ci_cal"][1] if r["boot95_ci_cal"][1] is not None else "—"
            lines.append(
                f"| {r['name']} | {r['n']} | {r['n_pos']} | {r['pooled_auc_sigmoid']} | "
                f"{r['pooled_auc_cal']} | {ci_lo}-{ci_hi} | {fm} | {r['brier_sigmoid']} | "
                f"{r['brier_cal']} | {r['ece_sigmoid']} | {r['ece10_cal']} | "
                f"{y.get('sensitivity','—')} | {y.get('specificity','—')} | "
                f"{y.get('ppv','—')} | {y.get('npv','—')} |")

    lines = ["# exp_v2_final_calibrated — 收官融合模型 Platt 校准 (m01538 任务1)",
             f"- tcd n=162 先验阳性率 {res['meta']['calibers']['tcd_pos_rate']:.3f} → "
             f"零技能常数基准 Brier = {res['meta']['calibers']['zero_skill_brier_tcd']}",
             "- Platt: 每折训练侧 LogisticRegression(C=1.0) 拟合 z_fused→y, 映射验证折; "
             "退化折回退 sigmoid(z)。"]
    md_table("校准后指标 — direct_tcd (n=162)", res["calibrated_162"])
    md_table("校准后指标 — direct_td (n=214)", res["calibrated_214"])
    lines.append("\n## 配对 DeLong (校准后)\n")
    for d in res["delong"]:
        lines.append(f"- {d['pair']}: ΔAUC={d['diff']:+.4f} (z={d['z']:.3f}, "
                     f"p={d['p']:.4f}, n_paired={d['n_paired']})")
    lines.append("\n## DeLong 0.355 vs 0.359 溯源\n")
    lines.append(f"- {res['deLong_355_vs_359_explained']['reason']}")
    with open(MOUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("DONE. JSON:", JOUT)
    print("MD:", MOUT)
    for r in res["calibrated_162"]:
        if r["name"] in ("clin+d_self+t_2d (chosen)", "old s4_clin_3plane.f1",
                         "clin_clean", "clin+Rad+DINOv2"):
            print(r["name"], "AUC", r["pooled_auc_cal"], "Brier", r["brier_cal"],
                  "ECE", r["ece10_cal"], "Youden", r["youden"])


if __name__ == "__main__":
    main()
