# -*- coding: utf-8 -*-
"""round9 S9-1 不确定性加权最小单元测试(CPU): 验证 log_var 可学习 + main 加权生效"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from config.hos import HosConfig
from models.network import CRCMultiModalNet
from trainers.trainer import Trainer

def build_cfg():
    cfg = HosConfig()
    cfg.backbone_name = "resnet10"
    cfg.batch_size = 4
    cfg.epochs = 2
    cfg.device = "cpu"
    cfg.use_amp = False
    cfg.mri_sequences = ["t"]
    cfg.n_from_shallow = True
    cfg.n_modal = "both"
    cfg.uncertainty_weight = True   # 关键开关
    return cfg

def main():
    cfg = build_cfg()
    torch.manual_seed(0)
    # 最小模型: resnet10, 临床7, 双头
    model = CRCMultiModalNet(
        backbone_name="resnet10", fusion_mode="multi",
        num_clinical=len(cfg.clinical_cols), num_classes=2, num_tasks=2,
        swin_feature_size=None, num_fuse_stages=2, img_size=cfg.spatial_size,
        use_self_attn=getattr(cfg, "use_self_attn", True),
        self_attn_layers=2, self_attn_heads=4,
        use_imputer=True, imputer_hidden_dim=64,
        head_dropout=0.0, pretrained_backbone=False,
        mri_share_backbone=False, mri_sequences=cfg.mri_sequences,
        head_width=128, pool_mode="avg", gem_p=3.0,
        n_from_shallow=cfg.n_from_shallow, n_shallow_idx=2,
        n_head_width=None, n_dropout=None, n_modal=cfg.n_modal, n_cascade=False,
    )
    class_weights_t = torch.tensor([1.324, 0.836])
    trainer = Trainer(model, None, None, cfg, class_weights=class_weights_t)
    assert trainer.uncertainty_weight is True, "uncertainty_weight 未启用"
    assert hasattr(trainer, "log_var_t") and hasattr(trainer, "log_var_n"), "log_var 缺失"
    # 确认 log_var 在 optimizer 参数组中 (用身份比较避免 Tensor == 歧义)
    opt_params = [p for g in trainer.optimizer.param_groups for p in g["params"]]
    assert any(p is trainer.log_var_t for p in opt_params) and \
        any(p is trainer.log_var_n for p in opt_params), \
        "log_var 未加入 optimizer"
    print(f"[OK] log_var_t={trainer.log_var_t.item():.4f} log_var_n={trainer.log_var_n.item():.4f} "
          f"(requires_grad={trainer.log_var_t.requires_grad})")

    # 假 batch: [B,1,48,48,48] 极小体积省内存
    B = 4
    x = torch.randn(B, 1, 48, 48, 48)
    mri = torch.randn(B, 1, 48, 48, 48)
    clin = torch.randn(B, len(cfg.clinical_cols))
    label = torch.tensor([1, 0, 1, 0])
    label_n = torch.tensor([0, 1, 0, 1])
    ct_avail = torch.tensor([True, True, True, True])
    mri_avail = torch.tensor([True, True, True, True])

    # 模拟 train_one_epoch 的 loss 计算路径(直接调 _compute_loss 两次验证 log_var 有梯度)
    out = model(x, mri, clin, ct_mask=ct_avail, mri_mask=mri_avail)
    total, main, aux, rec = trainer._compute_loss(
        out, label, ct_avail, mri_avail, label_secondary=label_n)
    w_t = torch.exp(-trainer.log_var_t).item()
    w_n = torch.exp(-trainer.log_var_n).item()
    print(f"[OK] _compute_loss 加权路径 total={total.item():.4f} main={main.item():.4f} "
          f"w_t={w_t:.4f} w_n={w_n:.4f}")

    # 反向一次: 验证 log_var 梯度非空且 optimizer 能 step
    total.backward()
    g_t = trainer.log_var_t.grad
    g_n = trainer.log_var_n.grad
    assert g_t is not None and g_n is not None, "log_var 无梯度"
    print(f"[OK] log_var 梯度 grad_t={g_t.item():.4f} grad_n={g_n.item():.4f}")
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    print(f"[OK] optimizer.step 后 log_var_t={trainer.log_var_t.item():.4f} "
          f"log_var_n={trainer.log_var_n.item():.4f}")

    # 关闭开关时走原路径(回归)
    cfg2 = build_cfg(); cfg2.uncertainty_weight = False
    model2 = CRCMultiModalNet(
        backbone_name="resnet10", fusion_mode="multi",
        num_clinical=len(cfg2.clinical_cols), num_classes=2, num_tasks=2,
        swin_feature_size=None, num_fuse_stages=2, img_size=cfg2.spatial_size,
        use_self_attn=True, self_attn_layers=2, self_attn_heads=4,
        use_imputer=True, imputer_hidden_dim=64, head_dropout=0.0,
        pretrained_backbone=False, mri_share_backbone=False,
        mri_sequences=cfg2.mri_sequences, head_width=128, pool_mode="avg", gem_p=3.0,
        n_from_shallow=True, n_shallow_idx=2, n_head_width=None, n_dropout=None,
        n_modal="both", n_cascade=False,
    )
    t2 = Trainer(model2, None, None, cfg2, class_weights=class_weights_t)
    assert t2.uncertainty_weight is False and not hasattr(t2, "log_var_t")
    out2 = model2(x, mri, clin, ct_mask=ct_avail, mri_mask=mri_avail)
    total2, main2, _, _ = t2._compute_loss(
        out2, label, ct_avail, mri_avail, label_secondary=label_n)
    print(f"[OK] 关闭分支回归 total2={total2.item():.4f} main2={main2.item():.4f}")
    print("UNCERTAINTY-SMOKE-OK")

if __name__ == "__main__":
    main()
