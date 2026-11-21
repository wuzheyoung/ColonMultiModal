# -*- coding: utf-8 -*-
"""离线汇总 ROI-DL 逐折 OOF (只读 CSV; 不加载模型/不读影像/不调用 GPU)。

输入:
  results/roidl_oof/{tag}_k{fold}_s{seed}.csv        (best ckpt 外折概率)
  results/roidl_oof/{tag}_k{fold}_s{seed}_last5.csv  (末 5 轮均值, 可选)
列: patient_id, label_T, prob_T, label_N, prob_N (--n-only/--t-only 时仅一列)

输出:
  logs/aggregate_roidl_{tag}_s{seed}.json  (不覆盖 kfold_summary_*)
计算:
  - 逐折 AUC / PR-AUC / macro-F1@0.5, 跨折 mean±std
  - Pooled OOF (可用折拼接) 的 AUC / PR-AUC / macro-F1@0.5
  - 跨折 LOFO 阈值: 第 i 折阈值由其余折 OOF 搜索, 再应用到第 i 折
缺失折 (如单折标定) 明确打印 found/missing 后跳过, 不报错。

用法:
  python tools/aggregate_roidl_kfold.py --tag p0_mri_t_bbox --seed 42
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OOF_DIR = os.path.join(ROOT, "results", "roidl_oof")
GRID = np.arange(0.05, 1.0, 0.05)


def _load_fold(path, task):
    """读单折 CSV; 缺文件/缺列/空表返回 None。"""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = [f"label_{task}", f"prob_{task}"]
    if "patient_id" not in df.columns or any(c not in df.columns for c in cols) or len(df) == 0:
        return None
    y = pd.to_numeric(df[cols[0]], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    p = pd.to_numeric(df[cols[1]], errors="coerce").to_numpy(dtype=np.float64)
    keep = (y >= 0) & np.isfinite(p)
    return (p[keep], y[keep], df["patient_id"].astype(str).to_numpy()[keep])


def _metrics(p, y):
    if len(y) == 0:
        return {"n": 0, "n_pos": 0, "roc_auc": None, "pr_auc": None, "macro_f1": None}
    pred = (p >= 0.5).astype(int)
    single = len(np.unique(y)) < 2
    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "roc_auc": None if single else float(roc_auc_score(y, p)),
        "pr_auc": None if single else float(average_precision_score(y, p)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def _threshold_search(p, y):
    """与 utils.calibrate_binary_threshold 同口径: 0.05–0.95 网格, macro-F1 最大。"""
    best_t, best_f = None, -1.0
    for t in GRID:
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


def _mean_std(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}


def _summarize(folds):
    """folds: [None | (p, y, pids)] → 逐折/均值/Pooled/LOFO。"""
    per_fold = []
    for i, f in enumerate(folds):
        if f is None:
            per_fold.append({"fold": i, "available": False})
            continue
        p, y = f[0], f[1]
        m = _metrics(p, y)
        m.update({"fold": i, "available": True, "n_unique_id": int(len(np.unique(f[2])))})
        per_fold.append(m)

    avail = [(i, f) for i, f in enumerate(folds) if f is not None]
    pooled = {"n_folds": len(avail)}
    if avail:
        p = np.concatenate([f[0] for _, f in avail])
        y = np.concatenate([f[1] for _, f in avail])
        pooled.update(_metrics(p, y))
        pooled["n_duplicate_ids"] = int(
            len(np.concatenate([f[2] for _, f in avail])) - len(np.unique(np.concatenate([f[2] for _, f in avail]))))

    lofo = []
    for i, f in enumerate(folds):
        if f is None:
            lofo.append({"fold": i, "available": False})
            continue
        others = [(j, g) for j, g in enumerate(folds) if j != i and g is not None]
        if not others:
            lofo.append({"fold": i, "available": True, "threshold": None,
                         "threshold_source_folds": [], "reason": "无其余折 OOF"})
            continue
        po = np.concatenate([g[0] for _, g in others])
        yo = np.concatenate([g[1] for _, g in others])
        if len(np.unique(yo)) < 2:
            lofo.append({"fold": i, "available": True, "threshold": None,
                         "threshold_source_folds": [j for j, _ in others],
                         "reason": "其余折 OOF 单类"})
            continue
        thr, _ = _threshold_search(po, yo)
        pred = (f[0] >= thr).astype(int)
        lofo.append({"fold": i, "available": True, "threshold": thr,
                     "threshold_source_folds": [j for j, _ in others],
                     "threshold_source_n": int(len(yo)),
                     "macro_f1": float(f1_score(f[1], pred, average="macro", zero_division=0)),
                     "recall": float(recall_score(f[1], pred, average="binary", zero_division=0))})

    return {
        "per_fold": per_fold,
        "mean_std": {
            "roc_auc": _mean_std([r.get("roc_auc") for r in per_fold if r.get("available")]),
            "pr_auc": _mean_std([r.get("pr_auc") for r in per_fold if r.get("available")]),
            "macro_f1": _mean_std([r.get("macro_f1") for r in per_fold if r.get("available")]),
        },
        "pooled": pooled,
        "lofo": lofo,
        "lofo_macro_f1_mean": _mean_std([r.get("macro_f1") for r in lofo if r.get("available")])["mean"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="out-tag (与训练 --out-tag 一致)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--oof-dir", default=DEFAULT_OOF_DIR)
    ap.add_argument("--out", default="", help="输出 JSON (默认 logs/aggregate_roidl_{tag}_s{seed}.json)")
    args = ap.parse_args()

    found, missing = [], []
    for i in range(args.folds):
        base = os.path.join(args.oof_dir, f"{args.tag}_k{i}_s{args.seed}")
        has = any((_load_fold(f"{base}.csv", t) is not None) or
                  (_load_fold(f"{base}_last5.csv", t) is not None) for t in ("T", "N"))
        (found if has else missing).append(i)

    print(f"[agg] tag={args.tag} seed={args.seed} folds={args.folds} "
          f"found={found} missing={missing}")
    if not found:
        print(f"[agg][warn] 未找到任何折 OOF ({args.oof_dir}/{args.tag}_k*_s{args.seed}.csv); "
              f"不生成汇总")
        return 0

    # 任务列可能只有 T 或只有 N: 逐折按可用任务分开装载
    out = {"tag": args.tag, "seed": args.seed, "folds_expected": args.folds,
           "folds_found": found, "folds_missing": missing, "oof_dir": args.oof_dir,
           "source": "best ckpt OOF"}

    for task in ("T", "N"):
        task_folds = []
        for i in range(args.folds):
            base = os.path.join(args.oof_dir, f"{args.tag}_k{i}_s{args.seed}")
            task_folds.append(_load_fold(f"{base}.csv", task))
        if any(f is not None for f in task_folds):
            out[f"task_{task}"] = _summarize(task_folds)
        task_l5 = []
        for i in range(args.folds):
            base = os.path.join(args.oof_dir, f"{args.tag}_k{i}_s{args.seed}")
            task_l5.append(_load_fold(f"{base}_last5.csv", task))
        if any(f is not None for f in task_l5):
            out[f"task_{task}_last5"] = _summarize(task_l5)

    op = args.out or os.path.join(ROOT, "logs", f"aggregate_roidl_{args.tag}_s{args.seed}.json")
    out_dir = os.path.dirname(op)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[agg] saved: {op}")
    for task in ("T", "N"):
        s = out.get(f"task_{task}")
        if s:
            ms = s["mean_std"]
            print(f"[agg] {task}: AUC {ms['roc_auc']['mean']}±{ms['roc_auc']['std']} "
                  f"AP {ms['pr_auc']['mean']}±{ms['pr_auc']['std']} "
                  f"F1@0.5 {ms['macro_f1']['mean']}±{ms['macro_f1']['std']} | "
                  f"pooled AUC {s['pooled'].get('roc_auc')} AP {s['pooled'].get('pr_auc')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
