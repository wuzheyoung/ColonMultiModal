# -*- coding: utf-8 -*-
"""
data/Hos 多模态 (CT+MRI+临床+病理T/N) 训练脚本。

流程:
  1. 读取 data/hos_pairs.csv (由 utils/build_hos_dataset.py 生成)
  2. 按 病理T二分类 × 病理N二分类 组合标签分层划分 train/val (默认 80/20)
  3. 双任务多模态网络 (T 头 + N 头, 共享融合特征), 训练 30 轮
  4. 输出最佳模型 + 验证集 T/N 指标

用法:
  python train_hos.py [--epochs 30] [--val-ratio 0.2] [--seed 42]
"""
import os
import sys
import argparse
import warnings

warnings.simplefilter("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import HosConfig
from data_loader import build_dataloaders
from models import CRCMultiModalNet
from trainers import Trainer
from utils import setup_logger, calibrate_binary_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = HosConfig()
    cfg.epochs = args.epochs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = setup_logger("./logs")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
        logger.info("CUDA 不可用，回退到 CPU")

    # ── 数据划分 ──
    df = pd.read_csv(cfg.train_csv)
    df = df[df["label_T"] >= 0].copy()
    df = df[df["label_N"] >= 0].copy()
    df = df.reset_index(drop=True)
    logger.info(f"有效样本: {len(df)}  (label_T/label_N 均有效)")
    logger.info(f"label_T 分布: {df['label_T'].value_counts().to_dict()}")
    logger.info(f"label_N 分布: {df['label_N'].value_counts().to_dict()}")
    logger.info(f"双模态(CT+MRI): {int((df['ct_available'] & df['mri_available']).sum())}  "
                f"CT-only: {int((df['ct_available'] & ~df['mri_available']).sum())}  "
                f"MRI-only: {int((~df['ct_available'] & df['mri_available']).sum())}")

    stratify = (df["label_T"].astype(str) + "_" + df["label_N"].astype(str))
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, stratify=stratify, random_state=args.seed
    )
    os.makedirs("./data", exist_ok=True)
    train_df.to_csv("./data/hos_train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv("./data/hos_val.csv", index=False, encoding="utf-8-sig")
    logger.info(f"train={len(train_df)}  val={len(val_df)}")
    logger.info(f"val 中 label_T: {val_df['label_T'].value_counts().to_dict()}, "
                f"label_N: {val_df['label_N'].value_counts().to_dict()}")

    # ── 数据加载 ──
    train_loader, val_loader = build_dataloaders(cfg, "./data/hos_train.csv", "./data/hos_val.csv")

    # ── 模型 ──
    model = CRCMultiModalNet(
        backbone_name=cfg.backbone_name,
        fusion_mode=cfg.fusion_mode,
        num_clinical=cfg.num_clinical,
        num_classes=cfg.num_classes,
        num_tasks=cfg.num_tasks,
        swin_feature_size=cfg.swin_feature_size,
        num_fuse_stages=cfg.num_fuse_stages,
        img_size=cfg.spatial_size,
        use_self_attn=cfg.use_self_attn,
        self_attn_layers=cfg.self_attn_layers,
        self_attn_heads=cfg.self_attn_heads,
        use_imputer=cfg.use_imputer,
        imputer_hidden_dim=cfg.imputer_hidden_dim,
        head_dropout=getattr(cfg, "head_dropout", 0.3),
        pretrained_backbone=cfg.pretrained_backbone,
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── T 主任务类别权重 (按 class_weight_mode; N 分布均衡不加权) ──
    mode = getattr(cfg, "class_weight_mode", "inverse")
    class_weights = None
    if mode != "none":
        cl, cnt = np.unique(train_df["label_T"].values, return_counts=True)
        counts = np.zeros(cfg.num_classes, dtype=float)
        counts[cl.astype(int)] = cnt
        inv_freq = len(train_df) / (cfg.num_classes * counts)
        class_weights = inv_freq if mode == "inverse" else np.sqrt(inv_freq)
        mode_label = "满逆频率" if mode == "inverse" else "√逆频率"
        logger.info(f"T 类别权重 ({mode_label}): "
                    f"{dict(zip(cl.astype(int).tolist(), class_weights[cl.astype(int)].round(3).tolist()))}")
    use_focal = getattr(cfg, "use_focal_loss", False)
    logger.info(f"正则化: head_dropout={getattr(cfg, 'head_dropout', 0.3)}  "
                f"label_smoothing={getattr(cfg, 'label_smoothing', 0.0)}  "
                f"weight_decay={cfg.weight_decay}  "
                f"focal={'γ=' + str(getattr(cfg, 'focal_gamma', 2.0)) if use_focal else 'off'}  "
                f"freeze={getattr(cfg, 'freeze_backbone_layers', [])}")

    trainer = Trainer(model, train_loader, val_loader, cfg, logger,
                      class_weights=class_weights,
                      label_smoothing=getattr(cfg, "label_smoothing", 0.0),
                      use_focal_loss=use_focal,
                      focal_gamma=getattr(cfg, "focal_gamma", 2.0),
                      freeze_backbone_layers=getattr(cfg, "freeze_backbone_layers", None))
    result = trainer.fit(tag="hos")

    # ── 验证集 T / N 独立评估 ──
    if result.get("confusion_matrix"):
        logger.info("=" * 60)
        logger.info(f"验证集 (T 二分类, 阈值0.5) acc={result['accuracy']:.4f}  "
                    f"macro_f1={result['macro_f1']:.4f}")
        logger.info(f"per_class_recall={result['per_class_recall']}")

    # ── 阈值校准: 在验证集上扫描概率阈值, 寻找最优决策边界 ──
    if getattr(cfg, "calibrate_thresholds", False):
        proba = trainer.predict_proba()
        cal_t = calibrate_binary_threshold(proba["probs_t"], proba["labels_t"], cfg.num_classes)
        bt = cal_t["best"]
        logger.info("─" * 60)
        logger.info(f"[校准] T: 阈值0.5 → acc={cal_t['default']['accuracy']:.4f} "
                    f"macro_f1={cal_t['default']['macro_f1']:.4f} "
                    f"recall={cal_t['default']['per_class_recall']}")
        logger.info(f"[校准] T: 最优阈值={bt['threshold']} → "
                    f"acc={bt['metrics']['accuracy']:.4f} "
                    f"macro_f1={bt['metrics']['macro_f1']:.4f} "
                    f"recall={bt['metrics']['per_class_recall']}")
        result["calibrated_t"] = bt
        if "probs_n" in proba:
            proba_n = trainer.predict_proba(best_key="n")
            cal_n = calibrate_binary_threshold(proba_n["probs_n"], proba_n["labels_n"], cfg.num_classes)
            bn = cal_n["best"]
            logger.info(f"[校准] N: 阈值0.5 → acc={cal_n['default']['accuracy']:.4f} "
                        f"macro_f1={cal_n['default']['macro_f1']:.4f} "
                        f"recall={cal_n['default']['per_class_recall']}")
            logger.info(f"[校准] N: 最优阈值={bn['threshold']} → "
                        f"acc={bn['metrics']['accuracy']:.4f} "
                        f"macro_f1={bn['metrics']['macro_f1']:.4f} "
                        f"recall={bn['metrics']['per_class_recall']}")
            result["calibrated_n"] = bn


if __name__ == "__main__":
    main()
