# -*- coding: utf-8 -*-
"""临时探针: 验证 resnet10 模型可构建(含 MONAI pretrained 权重)。"""
import sys

sys.path.insert(0, r"D:\PyFile\ColonMultiModal")

from config import HosConfig
from train_hos_seeds import build_model

cfg = HosConfig()
cfg.backbone_name = "resnet10"
m = build_model(cfg)
n_all = sum(p.numel() for p in m.parameters())
n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"resnet10 build OK: total={n_all/1e6:.1f}M trainable={n_tr/1e6:.1f}M")
