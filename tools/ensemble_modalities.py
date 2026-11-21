# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""F2: 异构单模态模型概率融合评估 (round9 S8)。

CT-only (run_06 smct) 与 MRI-only (run_07 smmri) 各自独立训练的模型,
对同一 val 划分前向取 softmax, 做模态级概率融合 (平均 + 按单模型 val F1 加权),
输出融合后 T/N macro_f1 + 校准阈值与 recall, 同时列出各单模态指标。

区别 S0 (同构 seed 集成, 错误强相关→证伪): F2 是异构模态模型, 错误弱相关,
且 S1/B3b 证据显示 T 靠 MRI、N 靠 CT —— 融合可能互补。

用法: python tools/ensemble_modalities.py [--branch full] [--split-seed 42]
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


def _load_and_predict(cfg, csv_path, ckpt_path, task):
    """加载 ckpt, 按 cfg.single_modal 强制掩码, 前向返回 (softmax, labels)。

    task="T" 用 best_model_*.pth (main 头); task="N" 用 best_model_n_*.pth (n 头)。
    """
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
            B = ct.size(0)
            # round9 单模态: 训练与验证一致强制 (trainer._apply_single_modal 语义, mask 为 [B])
            if cfg.single_modal == "mri":
                ct_mask = torch.zeros(B, dtype=torch.bool, device=cfg.device)
            elif cfg.single_modal == "ct":
                mri_mask = torch.zeros(B, dtype=torch.bool, device=cfg.device)
            mri_seq_mask = batch.get("mri_seq_available")
            if mri_seq_mask is not None:
                mri_seq_mask = mri_seq_mask.to(cfg.device).bool()
            with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
                o = model(ct, mri, clin, ct_mask=ct_mask, mri_mask=mri_mask,
                          mri_seq_mask=mri_seq_mask)
            if task == "T":
                probs.append(torch.softmax(o["main"], dim=1).float().cpu().numpy())
                labels.append(batch["label"].to(cfg.device).cpu().numpy())
            else:
                if "n" in o and batch.get("label_secondary") is not None:
                    probs.append(torch.softmax(o["n"], dim=1).float().cpu().numpy())
                    labels.append(batch["label_secondary"].to(cfg.device).cpu().numpy())
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


def _cfg_for(single_modal):
    cfg = HosConfig()
    cfg.backbone_name = "medicalnet18"
    cfg.pool_mode = "gem"
    cfg.head_width = 128
    cfg.lr = 1e-4
    cfg.lr_n = 3e-4
    cfg.n_from_shallow = True
    cfg.n_shallow_idx = 3
    cfg.class_weight_mode = "inverse"
    cfg.single_modal = single_modal
    cfg.save_dir = "./checkpoints_hos_full"
    cfg.plot_dir = "./plots_hos_full"
    cfg.use_augmentation = False
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="full")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default="logs/ensemble_modalities_full.json")
    args = ap.parse_args()

    tag = args.branch
    val_csv = f"data/hos_{tag}_val_s{args.split_seed}.csv"
    logger = setup_logger("./logs", "ensemble_modalities")
    logger.info(f"[F2] val_csv={val_csv}")

    cfg_ct = _cfg_for("ct")
    cfg_mri = _cfg_for("mri")

    ckpt_t_ct = "checkpoints_hos_full/best_model_hos_full_s42_smct.pth"
    ckpt_n_ct = "checkpoints_hos_full/best_model_n_hos_full_s42_smct.pth"
    ckpt_t_mri = "checkpoints_hos_full/best_model_hos_full_s42_smmri.pth"
    ckpt_n_mri = "checkpoints_hos_full/best_model_n_hos_full_s42_smmri.pth"
    for p in (ckpt_t_ct, ckpt_n_ct, ckpt_t_mri, ckpt_n_mri):
        if not os.path.exists(p):
            raise SystemExit(f"[单模态集成] 缺少 ckpt: {p} (需先 --single-modal ct/mri 重训对应 seed)")

    pt_ct, lt = _load_and_predict(cfg_ct, val_csv, ckpt_t_ct, "T")
    pn_ct, ln = _load_and_predict(cfg_ct, val_csv, ckpt_n_ct, "N")
    pt_mri, _ = _load_and_predict(cfg_mri, val_csv, ckpt_t_mri, "T")
    pn_mri, _ = _load_and_predict(cfg_mri, val_csv, ckpt_n_mri, "N")

    def evl(probs, labels, name):
        f1, rec = _f1_at(probs, labels, 0.5)
        cal = calibrate_binary_threshold(probs[:, 1], labels)
        return {"name": name,
                "f1@0.5": round(f1, 4), "recall": [round(x, 4) for x in rec],
                "cal_thr": round(float(cal["best"]["threshold"]), 4),
                "cal_f1": round(float(cal["best"]["metrics"]["macro_f1"]), 4),
                "cal_recall": [round(x, 4) for x in cal["best"]["metrics"]["per_class_recall"]]}

    rows = {
        "T_ct_only": evl(pt_ct, lt, "T_ct_only"),
        "T_mri_only": evl(pt_mri, lt, "T_mri_only"),
        "N_ct_only": evl(pn_ct, ln, "N_ct_only"),
        "N_mri_only": evl(pn_mri, ln, "N_mri_only"),
    }

    # 平均融合
    pt_mean = (pt_ct + pt_mri) / 2
    pn_mean = (pn_ct + pn_mri) / 2
    rows["T_mean_fusion"] = evl(pt_mean, lt, "T_mean_fusion")
    rows["N_mean_fusion"] = evl(pn_mean, ln, "N_mean_fusion")

    # 按单模型 val F1@0.5 加权融合 (T: ct=0.6010, mri=0.6806 → 权重 ≈ 0.469:0.531)
    f1_t_ct = rows["T_ct_only"]["f1@0.5"]
    f1_t_mri = rows["T_mri_only"]["f1@0.5"]
    w = f1_t_ct / (f1_t_ct + f1_t_mri) if (f1_t_ct + f1_t_mri) > 0 else 0.5
    pt_w = w * pt_ct + (1 - w) * pt_mri
    f1_n_ct = rows["N_ct_only"]["f1@0.5"]
    f1_n_mri = rows["N_mri_only"]["f1@0.5"]
    w_n = f1_n_ct / (f1_n_ct + f1_n_mri) if (f1_n_ct + f1_n_mri) > 0 else 0.5
    pn_w = w_n * pn_ct + (1 - w_n) * pn_mri
    rows["T_weighted_fusion"] = evl(pt_w, lt, "T_weighted_fusion")
    rows["N_weighted_fusion"] = evl(pn_w, ln, "N_weighted_fusion")
    rows["_T_weight_ct"] = round(w, 4)
    rows["_N_weight_ct"] = round(w_n, 4)

    out = {"meta": {"branch": tag, "split_seed": args.split_seed, "val_n": int(len(lt)),
                    "method": "modality-level fusion (smct + smmri)"},
           "rows": rows}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write(f"[F2] branch={tag} split_seed={args.split_seed} val_n={len(lt)}\n\n")
        for k, v in rows.items():
            if k.startswith("_"):
                continue
            f.write(f"{k:20s} f1@0.5={v['f1@0.5']:7.4f} recall={v['recall']} | "
                    f"cal_thr={v['cal_thr']} cal_f1={v['cal_f1']} cal_recall={v['cal_recall']}\n")
    print("[F2] done ->", args.out)


if __name__ == "__main__":
    main()
