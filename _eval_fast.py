# -*- coding: utf-8 -*-
"""轻量 GPU 评估: 加载指定 best checkpoint, 在 val 上输出 T/N 指标 + 阈值校准。

只迭代 val 集, 确定性预处理在 Dataset 内缓存, 前向放 GPU。
用法:
  python _eval_fast.py [--ckpt ./checkpoints_hos/best_model_hos.pth] [--val-csv ./data/hos_val.csv]
"""
import argparse
import sys
import time

sys.path.insert(0, 'D:/PyFile/ColonMultiModal')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn.functional as F

from config import HosConfig
from data_loader.loader import CRCMultiModalDataset
from models import CRCMultiModalNet
from utils import calibrate_binary_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./checkpoints_hos/best_model_hos.pth")
    ap.add_argument("--val-csv", type=str, default="./data/hos_val.csv")
    args = ap.parse_args()

    cfg = HosConfig()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={cfg.device}', flush=True)

    model = CRCMultiModalNet(
        backbone_name=cfg.backbone_name, fusion_mode=cfg.fusion_mode,
        num_clinical=cfg.num_clinical, num_classes=cfg.num_classes, num_tasks=cfg.num_tasks,
        swin_feature_size=cfg.swin_feature_size, num_fuse_stages=cfg.num_fuse_stages,
        img_size=cfg.spatial_size, use_self_attn=cfg.use_self_attn,
        self_attn_layers=cfg.self_attn_layers, self_attn_heads=cfg.self_attn_heads,
        use_imputer=cfg.use_imputer, imputer_hidden_dim=cfg.imputer_hidden_dim,
        head_dropout=getattr(cfg, 'head_dropout', 0.3), pretrained_backbone=False,
    ).to(cfg.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=cfg.device, weights_only=True))
    model.eval()
    print(f'Loaded {args.ckpt} OK', flush=True)

    ds = CRCMultiModalDataset(
        args.val_csv, label_col=cfg.label_col, spatial_size=cfg.spatial_size,
        num_clinical=cfg.num_clinical, clinical_cols=getattr(cfg, 'clinical_cols', None),
        multi_task=getattr(cfg, 'multi_task', False),
        secondary_label_col=getattr(cfg, 'secondary_label_col', 'label_N'),
        is_train=False, cache=True,
    )
    print(f'val samples: {len(ds)}', flush=True)

    t0 = time.time()
    probs_t, labels_t, probs_n, labels_n = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            ct = b['ct'].unsqueeze(0).to(cfg.device)
            mri = b['mri'].unsqueeze(0).to(cfg.device)
            clinical = b['clinical'].unsqueeze(0).to(cfg.device)
            label = b['label'].item()
            label_sec = b['label_secondary'].item()
            ct_mask = b['ct_available'].unsqueeze(0).to(cfg.device).bool()
            mri_mask = b['mri_available'].unsqueeze(0).to(cfg.device).bool()
            with torch.autocast(device_type='cuda', enabled=cfg.use_amp):
                out = model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask)
            probs_t.append(F.softmax(out['main'], dim=1).float().cpu().numpy())
            labels_t.append(label)
            if 'n' in out:
                probs_n.append(F.softmax(out['n'], dim=1).float().cpu().numpy())
                labels_n.append(label_sec)
            print(f'  [{i + 1}/{len(ds)}] elapsed={time.time() - t0:.1f}s', flush=True)

    # 逐样本收集的是 (1, C) 数组的列表, 必须先拼接成 (N, C) 再送入校准,
    # 否则 np.asarray 会得到 (N,1,C) 三维数组, sklearn 会报 mixed targets。
    probs_t = np.concatenate(probs_t, axis=0)
    labels_t = np.asarray(labels_t)
    if probs_n:
        probs_n = np.concatenate(probs_n, axis=0)
        labels_n = np.asarray(labels_n)

    cal_t = calibrate_binary_threshold(probs_t, labels_t, cfg.num_classes)
    bt, dt = cal_t['best'], cal_t['default']
    print(f'[val] T @0.5:     acc={dt["accuracy"]:.4f}  macro_f1={dt["macro_f1"]:.4f}  '
          f'recall={dt["per_class_recall"]}  cm={dt["confusion_matrix"]}', flush=True)
    print(f'[val] T @best={bt["threshold"]}: '
          f'acc={bt["metrics"]["accuracy"]:.4f}  macro_f1={bt["metrics"]["macro_f1"]:.4f}  '
          f'recall={bt["metrics"]["per_class_recall"]}  cm={bt["metrics"]["confusion_matrix"]}',
          flush=True)
    if len(probs_n) > 0:
        cal_n = calibrate_binary_threshold(probs_n, labels_n, cfg.num_classes)
        bn, dn = cal_n['best'], cal_n['default']
        print(f'[val] N @0.5:     acc={dn["accuracy"]:.4f}  macro_f1={dn["macro_f1"]:.4f}  '
              f'recall={dn["per_class_recall"]}  cm={dn["confusion_matrix"]}', flush=True)
        print(f'[val] N @best={bn["threshold"]}: '
              f'acc={bn["metrics"]["accuracy"]:.4f}  macro_f1={bn["metrics"]["macro_f1"]:.4f}  '
              f'recall={bn["metrics"]["per_class_recall"]}  cm={bn["metrics"]["confusion_matrix"]}',
              flush=True)
    print(f'EVAL DONE ({time.time() - t0:.1f}s)', flush=True)


if __name__ == "__main__":
    main()
