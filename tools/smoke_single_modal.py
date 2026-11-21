# -*- coding: utf-8 -*-
"""round9-CAM 后: --single-modal 消融的纯逻辑冒烟（CPU 秒级, 不训练）。

验证:
1. _apply_single_modal(cfg, sm) → cfg.single_modal + modality_dropout p=1.0 + 后缀
2. Trainer._apply_single_modal 验证阶段掩码强制 (mri→丢弃CT / ct→丢弃MRI / None→原样)
"""
import sys, types, torch

sys.path.insert(0, ".")
from config import HosConfig
import train_hos_seeds as ths
from trainers.trainer import Trainer

# ── 1) cfg 设置 ──
cfg = HosConfig()
suf = ths._apply_single_modal(cfg, "mri")
assert suf == "_smmri", suf
assert cfg.single_modal == "mri"
assert cfg.modality_dropout == {"ct_only": 1.0, "mri_only": 0.0, "both": 0.0}

suf3 = ths._apply_single_modal(HosConfig(), "ct")
assert suf3 == "_smct"
assert cfg.modality_dropout == {"ct_only": 1.0, "mri_only": 0.0, "both": 0.0}  # 原 cfg 未被污染

c2 = HosConfig()
assert ths._apply_single_modal(c2, None) == ""
assert not hasattr(c2, "single_modal")
assert c2.modality_dropout == HosConfig().modality_dropout  # None 不改 dropout

# ── 2) 验证阶段掩码强制 ──
t = Trainer.__new__(Trainer)
B = 4
avail = torch.tensor([True, False, True, False])

t.cfg = types.SimpleNamespace(single_modal="mri", device="cpu")
ctm, mrim = t._apply_single_modal(None, None, B, "cpu")
assert not ctm.any() and mrim.all(), (ctm, mrim)
ctm2, mrim2 = t._apply_single_modal(avail, avail, B, "cpu")
assert not ctm2.any() and mrim2.tolist() == [True, False, True, False]

t.cfg = types.SimpleNamespace(single_modal="ct", device="cpu")
ctm3, mrim3 = t._apply_single_modal(None, None, B, "cpu")
assert ctm3.all() and not mrim3.any()

t.cfg = types.SimpleNamespace(single_modal=None, device="cpu")
ctm4, mrim4 = t._apply_single_modal(avail, avail, B, "cpu")
assert ctm4.tolist() == mrim4.tolist() == [True, False, True, False]

# ── 3) CLI 解析 ──
sys.argv = ["train_hos_seeds.py", "--help"]
try:
    ths.main()
except SystemExit:
    pass

print("SINGLE-MODAL SMOKE OK")
