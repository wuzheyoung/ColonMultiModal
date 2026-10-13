# -*- coding: utf-8 -*-
"""临时探针: 验证强化数据增强管线可跑通 (shape/NaN/DataLoader)。"""
import os
import sys
sys.path.insert(0, os.getcwd())

import torch
from config.hos import HosConfig
from data_loader.loader import CRCMultiModalDataset, build_dataloaders

cfg = HosConfig()
ds = CRCMultiModalDataset(
    cfg.train_csv, label_col=cfg.label_col, spatial_size=cfg.spatial_size,
    num_clinical=cfg.num_clinical, clinical_cols=getattr(cfg, "clinical_cols", None),
    multi_task=getattr(cfg, "multi_task", False),
    secondary_label_col=getattr(cfg, "secondary_label_col", "label_N"),
    is_train=True, disk_cache_dir="./cache_hos",
)
print("dataset len:", len(ds))
for k in range(8):
    d = ds[(k * 7) % len(ds)]
    ct, mri = d["ct"], d["mri"]
    assert ct.shape == (1, *cfg.spatial_size), ct.shape
    assert mri.shape == (1, *cfg.spatial_size), mri.shape
    assert torch.isfinite(ct).all() and torch.isfinite(mri).all()
    nnz = lambda t: (t.abs() > 1e-6).float().mean().item()
    print(f"idx={(k*7)%len(ds)}  ct[mean={ct.mean():.3f} std={ct.std():.3f} nnz={nnz(ct):.3f}]  "
          f"mri[mean={mri.mean():.3f} std={mri.std():.3f} nnz={nnz(mri):.3f}]  "
          f"label={d['label']}  N={d['label_secondary']}")

train_loader, val_loader = build_dataloaders(cfg)
batch = next(iter(train_loader))
print("train batch ct:", tuple(batch["ct"].shape), "labels:", tuple(batch["label"].shape),
      "secondary:", tuple(batch["label_secondary"].shape))
print("AUGMENT OK")
