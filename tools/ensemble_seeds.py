# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""S0: 多 seed 概率集成评估 (R5 配置 ckpt, split_seed=42 同一划分)。

对每个 seed 分别加载 T-best 与 N-best checkpoint, 对 val 前向取 softmax,
再对 4 个 seed 的概率做平均, 输出集成后的 T/N macro_f1 + 校准阈值与 recall,
同时列出各单 seed 指标便于对比。
用法: python tools/ensemble_seeds.py [--seeds 42,0,1,7] [--branch full] [--split-seed 42]
"""
import argparse
import glob
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


def _load_and_predict(cfg, csv_path, ckpt_path):
    """加载一个 ckpt, 对 csv_path 对应数据前向, 返回 (softmax_probs, labels)。"""
    model = _build_model(cfg)
    sd = torch.load(ckpt_path, map_location=cfg.device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(cfg.device).eval()

    cfg.train_csv, cfg.val_csv = csv_path, csv_path
    _, loader = build_dataloaders(cfg, csv_path, csv_path)
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(cfg.device)
            mri = batch["mri"].to(cfg.device)
            clin = batch["clinical"].to(cfg.device)
            lbl = batch["label"].to(cfg.device)
            ct_mask = batch["ct_available"].to(cfg.device).bool()
            mri_mask = batch["mri_available"].to(cfg.device).bool()
            mri_seq_mask = batch.get("mri_seq_available")
            if mri_seq_mask is not None:
                mri_seq_mask = mri_seq_mask.to(cfg.device).bool()
            with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
                o = model(ct, mri, clin, ct_mask=ct_mask, mri_mask=mri_mask,
                          mri_seq_mask=mri_seq_mask)
            probs.append(torch.softmax(o["main"], dim=1).float().cpu().numpy())
            labels.append(lbl.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _load_predict_n(cfg, csv_path, ckpt_path):
    """加载 N-best ckpt 前向, 返回 (N softmax, N labels)。"""
    model = _build_model(cfg)
    sd = torch.load(ckpt_path, map_location=cfg.device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(cfg.device).eval()

    cfg.train_csv, cfg.val_csv = csv_path, csv_path
    _, loader = build_dataloaders(cfg, csv_path, csv_path)
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(cfg.device)
            mri = batch["mri"].to(cfg.device)
            clin = batch["clinical"].to(cfg.device)
            ct_mask = batch["ct_available"].to(cfg.device).bool()
            mri_mask = batch["mri_available"].to(cfg.device).bool()
            mri_seq_mask = batch.get("mri_seq_available")
            if mri_seq_mask is not None:
                mri_seq_mask = mri_seq_mask.to(cfg.device).bool()
            lbl_n = batch.get("label_secondary")
            with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
                o = model(ct, mri, clin, ct_mask=ct_mask, mri_mask=mri_mask,
                          mri_seq_mask=mri_seq_mask)
            if lbl_n is not None and "n" in o:
                probs.append(torch.softmax(o["n"], dim=1).float().cpu().numpy())
                labels.append(lbl_n.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def _f1_at(probs, labels, thr):
    pred = (probs[:, 1] >= thr).astype(int)
    cm = {}
    for t, p in zip(labels, pred):
        cm[(int(t), int(p))] = cm.get((int(t), int(p)), 0) + 1
    tp = cm.get((1, 1), 0); fn = cm.get((1, 0), 0); fp = cm.get((0, 1), 0)
    rec1 = tp / (tp + fn) if (tp + fn) else 0.0
    prec1 = tp / (tp + fp) if (tp + fp) else 0.0
    rec0 = cm.get((0, 0), 0) / (cm.get((0, 0), 0) + fp) if (cm.get((0, 0), 0) + fp) else 0.0
    prec0 = cm.get((0, 0), 0) / (cm.get((0, 0), 0) + fn) if (cm.get((0, 0), 0) + fn) else 0.0
    f1_1 = 2 * rec1 * prec1 / (rec1 + prec1) if (rec1 + prec1) else 0.0
    f1_0 = 2 * rec0 * prec0 / (rec0 + prec0) if (rec0 + prec0) else 0.0
    return (f1_1 + f1_0) / 2, [rec0, rec1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,0,1,7")
    ap.add_argument("--branch", default="full")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default="logs/ensemble_full.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tag = args.branch
    val_csv = f"data/hos_{tag}_val_s{args.split_seed}.csv"

    logger = setup_logger("./logs", "ensemble")
    logger.info(f"[S0] seeds={seeds} val_csv={val_csv}")

    cfg = HosConfig()
    cfg.backbone_name = "medicalnet18"
    cfg.pool_mode = "gem"
    cfg.head_width = 128
    cfg.lr = 1e-4
    cfg.lr_n = 3e-4
    cfg.n_from_shallow = True
    cfg.n_shallow_idx = 3
    cfg.class_weight_mode = "inverse"
    cfg.save_dir = f"./checkpoints_hos_{tag}"
    cfg.plot_dir = f"./plots_hos_{tag}"
    cfg.use_augmentation = False
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    ens_t = None
    ens_n = None
    ckpt_dir = f"./checkpoints_hos_{tag}"
    for seed in seeds:
        ckpt_t = f"{ckpt_dir}/best_model_hos_{tag}_s{seed}.pth"
        ckpt_n = f"{ckpt_dir}/best_model_n_hos_{tag}_s{seed}.pth"
        for p in (ckpt_t, ckpt_n):
            if not os.path.exists(p):
                raise SystemExit(
                    f"[S0] 缺少 ckpt: {p}\n"
                    f"     现有目录 {ckpt_dir}: {sorted(os.path.basename(x) for x in glob.glob(ckpt_dir + '/*.pth'))[:12]}\n"
                    "     注意: 历史 s42 权重已被后续实验覆盖, 复现 README 集成数字需用同配置重新训练对应 seed。")
        pt, lt = _load_and_predict(cfg, val_csv, ckpt_t)
        pn, ln = _load_predict_n(cfg, val_csv, ckpt_n)
        rows.append({
            "seed": seed,
            "T_macro_f1@0.5": round(float(_f1_at(pt, lt, 0.5)[0]), 4),
            "N_macro_f1@0.5": round(float(_f1_at(pn, ln, 0.5)[0]), 4),
        })
        ens_t = pt if ens_t is None else ens_t + pt
        ens_n = pn if ens_n is None else ens_n + pn
        logger.info(f"[S0] seed={seed} T_f1={rows[-1]['T_macro_f1@0.5']} N_f1={rows[-1]['N_macro_f1@0.5']}")

    ens_t = ens_t / len(seeds)
    ens_n = ens_n / len(seeds)

    out = {"meta": {"branch": tag, "seeds": seeds, "split_seed": args.split_seed,
                    "val_n": int(len(lt)), "method": "softmax average"},
           "single": rows}

    # 集成 T (阈值 0.5 + 校准)
    f1_t, rec_t = _f1_at(ens_t, lt, 0.5)
    cal_t = calibrate_binary_threshold(ens_t[:, 1], lt)
    # 集成 N
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
        f.write(f"branch={tag} seeds={seeds} split_seed={args.split_seed} val_n={len(lt)}\n\n")
        f.write("seed | T_f1@0.5 | N_f1@0.5\n")
        for r in rows:
            f.write(f"{r['seed']} | {r['T_macro_f1@0.5']} | {r['N_macro_f1@0.5']}\n")
        f.write(f"\nENSEMBLE T: f1@0.5={out['ensemble']['T']['thr0.5_f1']} "
                f"recall={out['ensemble']['T']['recall']} | cal_thr={out['ensemble']['T']['cal_thr']} "
                f"cal_f1={out['ensemble']['T']['cal_f1']} cal_recall={out['ensemble']['T']['cal_recall']}\n")
        f.write(f"ENSEMBLE N: f1@0.5={out['ensemble']['N']['thr0.5_f1']} "
                f"recall={out['ensemble']['N']['recall']} | cal_thr={out['ensemble']['N']['cal_thr']} "
                f"cal_f1={out['ensemble']['N']['cal_f1']} cal_recall={out['ensemble']['N']['cal_recall']}\n")
    print("[S0] done ->", args.out)


if __name__ == "__main__":
    main()
