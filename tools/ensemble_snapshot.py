# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""S9-⑤: Snapshot Ensemble 评估 (cosine warm restarts 周期快照的 softmax 平均)。

对给定快照 ckpt 列表 (T-best 各自权重即 snapshot_{tag}_s{seed}_ep{N}.pth),
每个快照对 val 前向取 softmax, 平均后输出 T/N macro_f1 + 校准。
用法: python tools/ensemble_snapshot.py --snapshots ep10,ep20,ep30,ep40,ep50
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.hos import HosConfig
from data_loader.loader import build_dataloaders
from utils import calibrate_binary_threshold, setup_logger
from tools.radiomics_experiment import _build_model  # noqa: E402
from tools.ensemble_seeds import _load_and_predict, _load_predict_n, _f1_at  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="ep10,ep20,ep30,ep40,ep50",
                    help="逗号分隔快照标签, 拼成 checkpoints_hos_full/snapshot_hos_full_s{seed}_{ep}.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--branch", default="full")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default="logs/ensemble_snapshot_full.json")
    args = ap.parse_args()

    eps = [s.strip() for s in args.snapshots.split(",")]
    tag = args.branch
    val_csv = f"data/hos_{tag}_val_s{args.split_seed}.csv"

    logger = setup_logger("./logs", "ensemble_snapshot")
    logger.info(f"[S9-5] eps={eps} seed={args.seed} val_csv={val_csv}")

    cfg = HosConfig()
    cfg.backbone_name = "medicalnet18"
    cfg.pool_mode = "gem"
    cfg.head_width = 128
    cfg.lr = 1e-4
    cfg.lr_n = 3e-4
    cfg.n_from_shallow = True
    cfg.n_shallow_idx = 3
    cfg.class_weight_mode = "inverse"
    cfg.save_dir = "./checkpoints_hos_full"
    cfg.plot_dir = "./plots_hos_full"
    cfg.use_augmentation = False
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    ens_t = None
    ens_n = None
    lt = ln = None
    for ep in eps:
        ckpt = f"checkpoints_hos_full/snapshot_hos_full_s{args.seed}_{ep}.pth"
        if not os.path.exists(ckpt):
            logger.warning(f"skip missing {ckpt}")
            continue
        pt, lt = _load_and_predict(cfg, val_csv, ckpt)
        pn, ln = _load_predict_n(cfg, val_csv, ckpt)
        rows.append({
            "ep": ep,
            "T_macro_f1@0.5": round(float(_f1_at(pt, lt, 0.5)[0]), 4),
            "N_macro_f1@0.5": round(float(_f1_at(pn, ln, 0.5)[0]), 4),
        })
        ens_t = pt if ens_t is None else ens_t + pt
        ens_n = pn if ens_n is None else ens_n + pn
        logger.info(f"[S9-5] {ep} T_f1={rows[-1]['T_macro_f1@0.5']} N_f1={rows[-1]['N_macro_f1@0.5']}")

    if ens_t is None:
        print("no snapshots found")
        sys.exit(1)
    ens_t = ens_t / len(rows)
    ens_n = ens_n / len(rows)

    out = {"meta": {"branch": tag, "seed": args.seed, "snapshots": [r["ep"] for r in rows],
                    "val_n": int(len(lt)), "method": "softmax average"},
           "single": rows}

    f1_t, rec_t = _f1_at(ens_t, lt, 0.5)
    cal_t = calibrate_binary_threshold(ens_t[:, 1], lt)
    f1_n, rec_n = _f1_at(ens_n, ln, 0.5)
    cal_n = calibrate_binary_threshold(ens_n[:, 1], ln)

    out["ensemble"] = {
        "T": {"thr0.5_f1": round(f1_t, 4), "recall": [round(x, 4) for x in rec_t],
              "cal_thr": round(float(cal_t["best"]["threshold"]), 4),
              "cal_f1": round(float(cal_t["best"]["metrics"]["macro_f1"]), 4),
              "cal_recall": [round(x, 4) for x in cal_t["best"]["metrics"]["per_class_recall"]]},
        "N": {"thr0.5_f1": round(f1_n, 4), "recall": [round(x, 4) for x in rec_n],
              "cal_thr": round(float(cal_n["best"]["threshold"]), 4),
              "cal_f1": round(float(cal_n["best"]["metrics"]["macro_f1"]), 4),
              "cal_recall": [round(x, 4) for x in cal_n["best"]["metrics"]["per_class_recall"]]},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write(f"branch={tag} seed={args.seed} snapshots={[r['ep'] for r in rows]} val_n={len(lt)}\n\n")
        f.write("ep | T_f1@0.5 | N_f1@0.5\n")
        for r in rows:
            f.write(f"{r['ep']} | {r['T_macro_f1@0.5']} | {r['N_macro_f1@0.5']}\n")
        f.write(f"\nENSEMBLE T: f1@0.5={out['ensemble']['T']['thr0.5_f1']} "
                f"recall={out['ensemble']['T']['recall']} | cal_thr={out['ensemble']['T']['cal_thr']} "
                f"cal_f1={out['ensemble']['T']['cal_f1']} cal_recall={out['ensemble']['T']['cal_recall']}\n")
        f.write(f"ENSEMBLE N: f1@0.5={out['ensemble']['N']['thr0.5_f1']} "
                f"recall={out['ensemble']['N']['recall']} | cal_thr={out['ensemble']['N']['cal_thr']} "
                f"cal_f1={out['ensemble']['N']['cal_f1']} cal_recall={out['ensemble']['N']['cal_recall']}\n")
    print("[S9-5] done ->", args.out)


if __name__ == "__main__":
    main()
