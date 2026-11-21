# -*- coding: utf-8 -*-
"""数据路径压力探针: 复现 V1 训练 epoch 7 后的偶发 CPU 卡死并定位肇事样本。

只跑数据集加载 + 随机增强 (与训练相同配置, 不碰 GPU 训练), 逐 batch 计时。
任何 batch 数据加载耗时超过 SLOW_S (默认 10s) 即 dump:
  - 该 batch 内各样本 idx / ct·mri 路径 / 体素统计 / 增强随机种子
  - 逐个样本单独计时, 定位具体卡死样本 (含增强 Compose 逐变换计时)

用法 (Pytorch 环境, CPU/GPU 均可):
  python tools/stress_data_path.py --branch full [--epochs 3] [--no-aug] [--backbone medicalnet18]
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HosConfig
from data_loader import build_dataloaders
from train_hos_seeds import _apply_run_overrides, _apply_branch  # noqa: E402

SLOW_S = float(os.environ.get("SLOW_S", "10.0"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="full")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--backbone", default=None,
                    choices=["resnet18", "resnet10", "medicalnet18", "medicalnet10"])
    args = ap.parse_args()

    logger = logging.getLogger("stress"); logger.setLevel(logging.INFO)
    h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)

    cfg = HosConfig(); cfg.epochs = 1
    _apply_run_overrides(cfg, None, args.backbone, False, None, 0.0, None,
                         False, False, no_aug=args.no_aug)
    _apply_branch(cfg, args.branch, None, None)
    torch.manual_seed(42); np.random.seed(42)
    loader, _ = build_dataloaders(cfg, cfg.train_csv, cfg.val_csv)

    aug = loader.dataset.augment if hasattr(loader.dataset, "augment") else None
    logger.info(f"数据路径压力探针: branch={args.branch} 增强={'ON' if aug else 'OFF'} "
                f"样本={len(loader.dataset)} 批次={len(loader)} epochs={args.epochs} "
                f"慢阈值={SLOW_S}s")

    for ep in range(1, args.epochs + 1):
        logger.info(f"== epoch {ep} ==")
        it = iter(loader)
        bidx = 0
        while True:
            t0 = time.time()   # 必须包住 next(loader) —— __getitem__+增强在 next 内同步执行
            try:
                batch = next(it)
            except StopIteration:
                break
            dt = time.time() - t0
            ct, mri = batch["ct"], batch["mri"]
            flag = ""
            if dt > SLOW_S:
                flag = "  <<<< SLOW"
                logger.info(f"batch {bidx:02d}: {dt:.2f}s 样本 {len(ct)} 个{flag}")
                _probe_slow_batch(loader.dataset, bidx, aug, logger)
            else:
                logger.info(f"batch {bidx:02d}: {dt:.2f}s")
            bidx += 1


def _probe_slow_batch(dataset, bidx, aug, logger):
    """对慢 batch 内的每个样本单独计时 (含逐变换计时), 定位肇事样本。"""
    import inspect
    comp = None
    if aug is not None and hasattr(aug, "transforms"):
        comp = aug.transforms
    n = len(dataset)
    start = bidx * (getattr(dataset, "batch_size", None) or 8)
    for i in range(start, min(start + 8, n)):
        t0 = time.time()
        _ = dataset[i]
        dt = time.time() - t0
        row = dataset.df.iloc[i]
        logger.info(f"  样本 idx={i} {dt:.2f}s ct={row.get('ct_path','')} "
                    f"mri_t={row.get('mri_path_t','')}")
        if dt > SLOW_S / 2 and comp is not None:
            # 逐变换计时 (重建输入统计, 不缓存)
            vol = dataset[i] if False else None
            _time_transforms(dataset, i, comp, logger)
    logger.info(f"  慢 batch {bidx} 逐样本排查完毕")


def _time_transforms(dataset, idx, comp, logger):
    """对疑似样本复现并逐变换计时。"""
    row = dataset.df.iloc[idx]
    ct_path = dataset._pick_ct_path(row)
    if isinstance(ct_path, str) and ct_path and os.path.exists(ct_path):
        x = dataset.disk.get(ct_path)[0] if dataset.disk is not None else None
        if x is None:
            x = dataset._load_det(ct_path, dataset.ct_det)
        for tr in comp:
            t0 = time.time()
            _ = tr(x.copy() if hasattr(x, "copy") else x)
            logger.info(f"    变换 {type(tr).__name__}: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
