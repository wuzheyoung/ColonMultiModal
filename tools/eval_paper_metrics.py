# -*- coding: utf-8 -*-
"""eval_paper_metrics.py — 论文级评估统计 (m00524 任务4)。

输入: OOF 概率 CSV (契约 patient_id,fold,label_T,prob), 自动发现
      outputs/oof_baseline_n162/singles/*.csv + fusions/*.csv (或 --dir 指定)。

输出:
  1) 每路: Pooled OOF AUC, Bootstrap 95% CI (患者级重采样 2000 次),
     Fold-Mean ± SD, Youden-OOF 临床指标 (见约束 2);
  2) 两两对比: DeLong 配对检验 — 自动按 patient_id 取共同交集, 显式输出 n_paired;
  3) ROC + 校准曲线 PNG → outputs/eval_paper_metrics_plots/;
  4) logs/eval_paper_metrics.json + markdown 汇总表。

硬性约束:
  1. DeLong 配对交集: 任意两路按 patient_id 交集配对 (n_paired 列明示);
  2. 阈值防泄露: Youden 阈值在折内训练侧概率上确定 (每折用其它折训练出的模型对训练折
     做 OOF/训练预测 → Youden J), 应用于对应验证折; 全折混淆矩阵汇总算临床指标;
     严禁在全集 OOF 上事后挑阈值。
"""
import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PLOT_DIR = "outputs/eval_paper_metrics_plots"
OUT_JSON = "logs/eval_paper_metrics.json"
N_BOOT = 2000
SEED = 0


def discover(dirpath):
    """自动发现 singles/ 与 fusions/ 下的 OOF CSV。"""
    found = {}
    for sub in ("singles", "fusions"):
        d = os.path.join(dirpath, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".csv"):
                name = f"{sub[:-1]}:{os.path.splitext(fn)[0]}"
                found[name] = os.path.join(d, fn)
    return found


def load_csv(path):
    df = pd.read_csv(path)
    df = df[["patient_id", "fold", "label_T", "prob"]].dropna(subset=["prob"])
    df["fold"] = df["fold"].astype(int)
    # f1 (zlogit 均值) 输出为 z 分数而非概率: 单调 sigmoid 映射到 [0,1]
    # (AUC/DeLong/Youden 对单调变换不变; 仅校准/Brier 需要真概率)
    p = df["prob"].values.astype(float)
    if p.max() > 1.0 or p.min() < 0.0:
        df["prob"] = 1.0 / (1.0 + np.exp(-p))
        df.attrs["was_score"] = True
    else:
        df.attrs["was_score"] = False
    return df


def pooled_auc(df):
    y = df["label_T"].values.astype(int)
    p = df["prob"].values.astype(float)
    if len(np.unique(y)) < 2:
        return None
    return roc_auc_score(y, p)


def boot_ci(df, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y = df["label_T"].values.astype(int)
    p = df["prob"].values.astype(float)
    ids = df["patient_id"].values
    if len(np.unique(y)) < 2:
        return None
    stats = []
    for _ in range(n):
        sel = rng.choice(len(df), size=len(df), replace=True)
        if len(np.unique(y[sel])) < 2:
            continue
        stats.append(roc_auc_score(y[sel], p[sel]))
    qs = np.percentile(stats, [2.5, 97.5])
    return float(qs[0]), float(qs[1])


def fold_mean_sd(df):
    y = df["label_T"].values.astype(int)
    p = df["prob"].values.astype(float)
    per = []
    for k, g in df.groupby("fold"):
        if len(np.unique(g["label_T"])) < 2:
            continue
        per.append(roc_auc_score(g["label_T"].values.astype(int), g["prob"].values.astype(float)))
    if not per:
        return None
    return float(np.mean(per)), float(np.std(per, ddof=1) if len(per) > 1 else 0.0)


def youden_metrics_oof(df):
    """折内训练侧 OOF 定 Youden 阈值 → 应用到对应验证折 → 全折混淆汇总。

    显式说明 (m00622): 对第 k 折, 阈值只由**其余 4 折**患者的 OOF 概率
    (roc_curve + Youden J) 确定, 再应用于第 k 折的概率; 逐折独立, 最后跨折
    汇总 TP/FP/TN/FN 计算 Sens/Spec/PPV/NPV。严禁在全集 OOF 上事后挑阈值。
    """
    y = df["label_T"].values.astype(int)
    p = df["prob"].values.astype(float)
    folds = sorted(df["fold"].unique())
    tp = fp = tn = fn = 0
    for k in folds:
        va = df["fold"].values == k
        tr = ~va
        if tr.sum() < 8 or len(np.unique(y[tr])) < 2 or va.sum() < 1:
            continue
        # 训练侧: 阈值来自训练折概率 (防泄露, 不在验证折上挑)
        fpr, tpr, th = roc_curve(y[tr], p[tr])
        j = tpr - fpr
        t0 = float(th[np.argmax(j)])
        pv, yv = p[va], y[va]
        pred = (pv >= t0).astype(int)
        tp += int(((pred == 1) & (yv == 1)).sum())
        fp += int(((pred == 1) & (yv == 0)).sum())
        tn += int(((pred == 0) & (yv == 0)).sum())
        fn += int(((pred == 0) & (yv == 1)).sum())
    if tp + fn == 0 or tn + fp == 0:
        return None
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp > 0 else None
    npv = tn / (tn + fn) if tn + fn > 0 else None
    return {"sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "ppv": round(ppv, 4) if ppv is not None else None,
            "npv": round(npv, 4) if npv is not None else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def delong_paired_test(df1, df2):
    """placement-value 配对 DeLong (不平衡安全, 含跨分类器协方差)。"""
    common = sorted(set(df1["patient_id"]) & set(df2["patient_id"]))
    if len(common) < 10 or len(set(df1["label_T"])) < 2:
        return None, len(common)
    a = df1.set_index("patient_id").loc[common]
    b = df2.set_index("patient_id").loc[common]
    y = a["label_T"].values.astype(int)
    pa, pb = a["prob"].values.astype(float), b["prob"].values.astype(float)
    pos = y == 1
    neg = y == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 < 2 or n0 < 2:
        return None, len(common)
    V10a = np.array([(pa[neg] < pa[i]).mean() + 0.5 * (pa[neg] == pa[i]).mean() for i in np.where(pos)[0]])
    V01a = np.array([(pa[pos] > pa[j]).mean() + 0.5 * (pa[pos] == pa[j]).mean() for j in np.where(neg)[0]])
    V10b = np.array([(pb[neg] < pb[i]).mean() + 0.5 * (pb[neg] == pb[i]).mean() for i in np.where(pos)[0]])
    V01b = np.array([(pb[pos] > pb[j]).mean() + 0.5 * (pb[pos] == pb[j]).mean() for j in np.where(neg)[0]])
    d10, d01 = V10a - V10b, V01a - V01b
    # 配对差的两样本方差 (含两分类器协方差), 双侧 z 检验
    var = d10.var(ddof=1) / n1 + d01.var(ddof=1) / n0
    if var <= 1e-12:
        return None, len(common)
    auc1 = roc_auc_score(y, pa)
    auc2 = roc_auc_score(y, pb)
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"auc1": round(float(auc1), 4), "auc2": round(float(auc2), 4),
            "diff": round(float(auc1 - auc2), 4), "z": round(float(z), 4),
            "p": round(float(p), 4)}, len(common)


def plot_roc_curve(df, name, path):
    y = df["label_T"].values.astype(int)
    p = df["prob"].values.astype(float)
    fpr, tpr, _ = roc_curve(y, p)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {name}"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_calibration(df, name, path):
    y = df["label_T"].values.astype(int)
    p = df["prob"].values.astype(float)
    bs = brier_score_loss(y, p)
    frac_pos, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pred, frac_pos, "o-", lw=2, label=f"{name} (Brier={bs:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration — {name}"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


MAIN_NAMES = ["fusion:s4_clin_3plane.f1", "fusion:s2_clin_dconv.f1",
              "single:clin_clean", "single:d_conv", "single:t_conv", "single:c_conv"]


def plot_overlay_main(frames, roc_path, cal_path):
    """6 路主模型叠加图: 图例标注 AUC (95% CI) 与 n。"""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in MAIN_NAMES:
        if name not in frames:
            continue
        df = frames[name]
        y = df["label_T"].values.astype(int)
        p = df["prob"].values.astype(float)
        fpr, tpr, _ = roc_curve(y, p)
        a = auc(fpr, tpr)
        ci = boot_ci(df)
        n = int(len(df))
        lab = f"{name} (AUC={a:.3f}, 95%CI {ci[0]:.3f}-{ci[1]:.3f}, n={n})"
        ax.plot(fpr, tpr, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC overlay — main models"); ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(roc_path, dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in MAIN_NAMES:
        if name not in frames:
            continue
        df = frames[name]
        y = df["label_T"].values.astype(int)
        p = df["prob"].values.astype(float)
        bs = brier_score_loss(y, p)
        frac_pos, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        n = int(len(df))
        lab = f"{name} (Brier={bs:.3f}, n={n})"
        ax.plot(pred, frac_pos, "o-", lw=2, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration overlay — main models")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(cal_path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="outputs/oof_baseline_n162")
    ap.add_argument("--pairs", default="all", help="对比对: all 或逗号分隔 'a vs b' 名")
    ap.add_argument("--plots", action="store_true", default=True, help="输出 ROC/校准曲线")
    args = ap.parse_args()

    found = discover(args.dir)
    if not found:
        print(f"[err] no OOF CSVs under {args.dir}"); sys.exit(1)
    frames = {k: load_csv(p) for k, p in found.items()}
    os.makedirs(PLOT_DIR, exist_ok=True)

    res = {}
    for name, df in frames.items():
        entry = {"n": int(len(df)), "n_pos": int(df["label_T"].sum()),
                 "was_score": df.attrs.get("was_score", False)}
        a = pooled_auc(df)
        ci = boot_ci(df)
        fms = fold_mean_sd(df)
        ym = youden_metrics_oof(df)
        entry["pooled_auc"] = round(a, 4) if a is not None else None
        entry["boot95_ci"] = [round(v, 4) for v in ci] if ci else None
        entry["fold_mean_auc"] = [round(v, 4) for v in fms] if fms else None
        entry["youden_oof"] = ym
        if args.plots:
            plot_roc_curve(df, name, os.path.join(PLOT_DIR, f"roc_{name.replace(':','_')}.png"))
            plot_calibration(df, name, os.path.join(PLOT_DIR, f"cal_{name.replace(':','_')}.png"))
        res[name] = entry
        print(f"[{name}] n={entry['n']} AUC={entry['pooled_auc']} "
              f"CI={entry['boot95_ci']} fold={entry['fold_mean_auc']} youden={ym and ym['sensitivity']}/{ym and ym['specificity']}")

    # 两两 DeLong (自动配对交集, 显式 n_paired)
    names = list(frames)
    pairs = []
    if args.pairs.lower() == "all":
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    else:
        for spec in args.pairs.split(","):
            a, b = [x.strip() for x in spec.split("vs")]
            pairs.append((a, b))
    delong = {}
    for na, nb in pairs:
        if na not in frames or nb not in frames:
            continue
        d, npd = delong_paired_test(frames[na], frames[nb])
        delong[f"{na} vs {nb}"] = {**d, "n_paired": npd} if d else {"n_paired": npd, "note": "insufficient"}
        print(f"  [DeLong] {na} vs {nb}: n_paired={npd} p={d and d['p']}")

    # 6 路主模型叠加图
    plot_overlay_main(frames, os.path.join(PLOT_DIR, "roc_overlay_main.png"),
                      os.path.join(PLOT_DIR, "cal_overlay_main.png"))

    out = {"n_boot": N_BOOT, "rows": res, "delong": delong}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"DONE → {OUT_JSON} + plots → {PLOT_DIR}")


if __name__ == "__main__":
    main()
