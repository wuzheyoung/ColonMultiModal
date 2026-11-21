# -*- coding: utf-8 -*-
"""round8: 1-epoch 探针 —— 数据统计 + 单个训练/验证 epoch。

用途: 手动跑长训练前, 用 ~1-2 分钟快速摸清:
  - 数据规模 / T,N 标签分布 / CT、MRI 各序列可用性
  - 训练 1 个 epoch 的 loss 分量(main/aux/rec)、lr、梯度 L2 范数、T 各类别 acc
  - 验证集 acc / n_acc

用法（Pytorch 环境, 项目根目录下执行）:
  D:\\Development\\miniconda\\envs\\Pytorch\\python.exe tools/probe_epoch.py --branch full
  D:\\Development\\miniconda\\envs\\Pytorch\\python.exe tools/probe_epoch.py --branch full --lr 3e-4 --backbone resnet10 --share-backbone
探针产生的临时 csv 写到 logs/ (不写 data/)。
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")

from config import HosConfig
from data_loader import build_dataloaders
from trainers import Trainer
from train_hos_seeds import (_apply_branch, _apply_run_overrides, build_model,
                             compute_class_weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="full",
                    choices=["mri_t", "mri_d", "mri_c", "mri_tdc", "ct_p", "ct_z", "full"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--backbone", type=str, default=None,
                    choices=["resnet18", "resnet10", "medicalnet18", "medicalnet10"])
    ap.add_argument("--share-backbone", action="store_true")
    ap.add_argument("--freeze-low", action="store_true")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--class-weight-mode", type=str, default=None,
                    choices=["inverse", "sqrt", "none"])
    ap.add_argument("--no-aug", action="store_true",
                    help="关闭训练随机增强")
    ap.add_argument("--pool", type=str, default=None, choices=["avg", "gem"],
                    help="池化方式 (avg/gem)")
    ap.add_argument("--head-width", type=int, default=None,
                    help="主分类头隐层宽度 (如 128)")
    args = ap.parse_args()

    logger = logging.getLogger("probe")
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)

    cfg = HosConfig()
    cfg.epochs = 1
    freeze = _apply_run_overrides(
        cfg, None, args.backbone, False, None, args.mixup, None,
        args.freeze_low, args.share_backbone, lr=args.lr,
        class_weight_mode=args.class_weight_mode, no_aug=args.no_aug,
        pool=args.pool, head_width=args.head_width)
    tag = _apply_branch(cfg, args.branch, None, None)
    torch.manual_seed(42)
    np.random.seed(42)

    # ── 数据统计 ──
    df = pd.read_csv(cfg.train_csv)
    df = df[(df["label_T"] >= 0) & (df["label_N"] >= 0)].reset_index(drop=True)
    logger.info(f"== 探针 branch={args.branch} tag={tag} 有效样本={len(df)}")
    logger.info(f"label_T: {df['label_T'].value_counts().to_dict()}  "
                f"label_N: {df['label_N'].value_counts().to_dict()}")
    ct_avail = df["ct_available"].astype(bool) if "ct_available" in df else None
    mri_avail = df["mri_available"].astype(bool) if "mri_available" in df else None
    if ct_avail is not None and mri_avail is not None:
        logger.info(f"模态可用性: 双模态={int((ct_avail & mri_avail).sum())}  "
                    f"仅CT={int((ct_avail & ~mri_avail).sum())}  "
                    f"仅MRI={int((~ct_avail & mri_avail).sum())}  "
                    f"双缺={int((~ct_avail & ~mri_avail).sum())}")
    if "mri_available_t" in df:
        logger.info("MRI 序列可用: t="
                    f"{int(df['mri_available_t'].fillna(True).astype(bool).sum())} c="
                    f"{int(df['mri_available_c'].fillna(True).astype(bool).sum())} d="
                    f"{int(df['mri_available_d'].fillna(True).astype(bool).sum())}")

    # ── 划分 train/val（与 run_seed 相同协议: 先过滤 -1 标签, 再分层切分落盘）──
    # 注意: 分支 CSV 含 label=-1 样本, 直接喂 loader 会在 CrossEntropy 触发 CUDA device-side assert
    from sklearn.model_selection import train_test_split
    stratify = df["label_T"].astype(str) + "_" + df["label_N"].astype(str)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=stratify,
                                        random_state=42)
    os.makedirs("./logs", exist_ok=True)
    train_csv = f"./logs/hos_{tag}_probe_train.csv"
    val_csv = f"./logs/hos_{tag}_probe_val.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")
    logger.info(f"probe train={len(train_df)}  val={len(val_df)}")

    train_loader, val_loader = build_dataloaders(cfg, train_csv, val_csv)
    model = build_model(cfg)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    # backbone 预训练加载状态 (仅 medicalnet* 有 pretrain_msg)
    for _nm, _bb in (("ct", model.ct_backbone),
                     ("mri_shared", getattr(model, "mri_shared_backbone", None))):
        if _bb is not None and hasattr(_bb, "pretrain_msg"):
            logger.info(f"{_nm} backbone: {_bb.pretrain_msg}")
    for _seq, _bb in (model.mri_backbones or {}).items():
        if _bb is not None and hasattr(_bb, "pretrain_msg"):
            logger.info(f"mri_{_seq} backbone: {_bb.pretrain_msg}")

    class_weights = compute_class_weights(df, cfg.num_classes, cfg.class_weight_mode)
    trainer = Trainer(model, train_loader, val_loader, cfg, logger,
                      class_weights=class_weights,
                      label_smoothing=getattr(cfg, "label_smoothing", 0.0),
                      freeze_backbone_layers=freeze)
    logger.info(f"实验参数: backbone={cfg.backbone_name}, lr={cfg.lr}, "
                f"class_weight_mode={cfg.class_weight_mode}, "
                f"loss_weights={cfg.loss_weights}, mixup={getattr(cfg, 'mixup', 0.0)}, "
                f"use_aug={getattr(cfg, 'use_augmentation', True)}, freeze={freeze}")

    # ── 首 batch 模态 mask 采样比例（反映模态 dropout 实际生效情况）──
    try:
        batch0 = next(iter(train_loader))
        ct_m, mri_m = trainer._sample_modality_masks(
            batch0["label"].size(0), cfg.device,
            batch0.get("ct_available"), batch0.get("mri_available"))
        ct_avail_b = batch0.get("ct_available")
        mri_avail_b = batch0.get("mri_available")
        logger.info("首 batch 模态: "
                    f"ct_avail={ct_avail_b.float().mean().item():.2f} "
                    f"mri_avail={mri_avail_b.float().mean().item():.2f} | "
                    f"采样后 ct_mask={ct_m.float().mean().item():.2f} "
                    f"mri_mask={mri_m.float().mean().item():.2f}")
    except Exception as e:  # noqa: BLE001
        logger.info(f"mask 采样打印跳过: {e}")

    logger.info("== 训练 1 epoch ==")
    avg_loss, acc, comp = trainer.train_one_epoch(1)
    logger.info("== 验证 1 次 ==")
    vloss, vacc, _, _, n_acc, _, _ = trainer.validate(1)
    logger.info(f"探针结果: train_acc={acc:.4f}  val_acc={vacc:.4f}  val_n_acc={n_acc:.4f}")


if __name__ == "__main__":
    main()
