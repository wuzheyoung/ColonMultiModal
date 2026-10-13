# -*- coding: utf-8 -*-
"""评估 checkpoints_hos/best_model_hos.pth (epoch 11) 在 train/val 上的 T/N 指标。"""
import sys
sys.path.insert(0, 'D:/PyFile/ColonMultiModal')
sys.stdout.reconfigure(encoding='utf-8')
import torch
from config import HosConfig
from data_loader import build_dataloaders
from models import CRCMultiModalNet
from utils import compute_metrics

cfg = HosConfig()
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CRCMultiModalNet(
    backbone_name=cfg.backbone_name, fusion_mode=cfg.fusion_mode,
    num_clinical=cfg.num_clinical, num_classes=cfg.num_classes, num_tasks=cfg.num_tasks,
    swin_feature_size=cfg.swin_feature_size, num_fuse_stages=cfg.num_fuse_stages,
    img_size=cfg.spatial_size, use_self_attn=cfg.use_self_attn,
    self_attn_layers=cfg.self_attn_layers, self_attn_heads=cfg.self_attn_heads,
    use_imputer=cfg.use_imputer, imputer_hidden_dim=cfg.imputer_hidden_dim,
    head_dropout=getattr(cfg, 'head_dropout', 0.3), pretrained_backbone=False,
).to(cfg.device)
model.load_state_dict(torch.load('./checkpoints_hos/best_model_hos.pth', map_location=cfg.device))
model.eval()
print('Loaded best_model_hos.pth OK', flush=True)


def evaluate(csv_path, name):
    _, loader = build_dataloaders(cfg, csv_path, csv_path)
    preds_t, labels_t, preds_n, labels_n = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ct = batch['ct'].to(cfg.device)
            mri = batch['mri'].to(cfg.device)
            clinical = batch['clinical'].to(cfg.device)
            label = batch['label'].to(cfg.device)
            label_sec = batch.get('label_secondary')
            if label_sec is not None:
                label_sec = label_sec.to(cfg.device)
            ct_mask = batch.get('ct_available')
            mri_mask = batch.get('mri_available')
            if ct_mask is not None:
                ct_mask = ct_mask.to(cfg.device).bool()
            if mri_mask is not None:
                mri_mask = mri_mask.to(cfg.device).bool()
            with torch.autocast(device_type='cuda', enabled=cfg.use_amp):
                out = model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask)
            preds_t.extend(out['main'].argmax(1).cpu().tolist())
            labels_t.extend(label.cpu().tolist())
            if label_sec is not None and 'n' in out:
                preds_n.extend(out['n'].argmax(1).cpu().tolist())
                labels_n.extend(label_sec.cpu().tolist())
    mt = compute_metrics(preds_t, labels_t, cfg.num_classes)
    print(f'[{name}] T: acc={mt["accuracy"]:.4f}  macro_f1={mt["macro_f1"]:.4f}  '
          f'recall={mt["per_class_recall"]}  cm={mt["confusion_matrix"]}', flush=True)
    if preds_n:
        mn = compute_metrics(preds_n, labels_n, cfg.num_classes)
        print(f'[{name}] N: acc={mn["accuracy"]:.4f}  macro_f1={mn["macro_f1"]:.4f}  '
              f'recall={mn["per_class_recall"]}  cm={mn["confusion_matrix"]}', flush=True)
    else:
        print(f'[{name}] N: no N head', flush=True)


evaluate('./data/hos_val.csv', 'val')
evaluate('./data/hos_train.csv', 'train')
print('EVAL DONE', flush=True)
