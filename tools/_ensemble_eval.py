# -*- coding: utf-8 -*-
"""round6 A2: seed 集成评估 —— 多个 checkpoint 在固定 val 上平均 softmax 概率,
再校准阈值, 报告集成 T/N macro-F1 与各成员指标对比。

用法:
  python tools/_ensemble_eval.py --ckpts checkpoints_hos/best_model_hos_s42.pth,checkpoints_hos/best_model_hos_s0.pth \
      --names s42,s0 --val-csv data/hos_val_s42.csv
"""
import argparse
import json
import sys
import warnings

warnings.simplefilter("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F

from config import HosConfig
from data_loader import build_dataloaders
from models import CRCMultiModalNet
from utils import calibrate_binary_threshold, compute_metrics


def build_model(cfg):
    return CRCMultiModalNet(
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
        mri_share_backbone=getattr(cfg, "mri_share_backbone", False),
    )


@torch.no_grad()
def predict_probs(model, loader, cfg):
    model.eval()
    pts, pns, lts, lns = [], [], [], []
    for batch in loader:
        ct = batch["ct"].to(cfg.device)
        mri = batch["mri"].to(cfg.device)
        clinical = batch["clinical"].to(cfg.device)
        label = batch["label"].to(cfg.device)
        ct_mask = batch.get("ct_available")
        mri_mask = batch.get("mri_available")
        if ct_mask is not None:
            ct_mask = ct_mask.to(cfg.device).bool()
        if mri_mask is not None:
            mri_mask = mri_mask.to(cfg.device).bool()
        mri_seq_avail = batch.get("mri_seq_available")
        if mri_seq_avail is not None:
            mri_seq_avail = mri_seq_avail.to(cfg.device).bool()
        out = model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask,
                    mri_seq_mask=mri_seq_avail)
        pts.append(F.softmax(out["main"], dim=1).float().cpu().numpy())
        lts.append(label.cpu().numpy())
        if "n" in out:
            pns.append(F.softmax(out["n"], dim=1).float().cpu().numpy())
            lns.append(batch["label_secondary"].cpu().numpy())
    return (np.concatenate(pts), np.concatenate(lts),
            np.concatenate(pns) if pns else None,
            np.concatenate(lns) if lns else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, required=True, help="逗号分隔 checkpoint 路径")
    ap.add_argument("--val-csv", type=str, required=True)
    ap.add_argument("--names", type=str, default=None, help="逗号分隔成员名 (对应 ckpts)")
    ap.add_argument("--backbone", type=str, default=None)
    ap.add_argument("--fusion", type=str, default=None)
    ap.add_argument("--share-backbone", action="store_true")
    ap.add_argument("--no-clinical", action="store_true")
    ap.add_argument("--out", type=str, default="./logs/ensemble_summary.json")
    args = ap.parse_args()

    cfg = HosConfig()
    if args.backbone:
        cfg.backbone_name = args.backbone
    if args.fusion:
        cfg.fusion_mode = args.fusion
    if args.share_backbone:
        cfg.mri_share_backbone = True
    if args.no_clinical:
        cfg.clinical_cols = []
    cfg.val_csv = args.val_csv

    # 两个 dataset 都用 val 数据, 避免额外加载全量 train csv
    _, val_loader = build_dataloaders(cfg, train_csv=args.val_csv, val_csv=args.val_csv)

    ckpt_paths = [p.strip() for p in args.ckpts.split(",") if p.strip()]
    names = [n.strip() for n in args.names.split(",")] if args.names else \
        [f"m{i}" for i in range(len(ckpt_paths))]

    members = []
    stacked_t = stacked_n = None
    labels_t = labels_n = None
    for name, path in zip(names, ckpt_paths):
        model = build_model(cfg).to(cfg.device)
        state = torch.load(path, map_location=cfg.device)
        model.load_state_dict(state)
        pts, lts, pns, lns = predict_probs(model, val_loader, cfg)
        labels_t, labels_n = lts, lns
        m_t = compute_metrics(pts.argmax(1), lts, cfg.num_classes)
        m_n = compute_metrics(pns.argmax(1), lns, cfg.num_classes) if pns is not None else {}
        print(f"[{name}] T macro_f1={m_t['macro_f1']:.4f} acc={m_t['accuracy']:.4f}  "
              f"N macro_f1={m_n.get('macro_f1', float('nan')):.4f}")
        members.append({"name": name, "ckpt": path,
                        "t_macro_f1": float(m_t["macro_f1"]),
                        "t_acc": float(m_t["accuracy"]),
                        "n_macro_f1": float(m_n.get("macro_f1", float("nan")))})
        stacked_t = pts if stacked_t is None else stacked_t + pts
        if pns is not None:
            stacked_n = pns if stacked_n is None else stacked_n + pns

    res = {"members": members, "val_csv": args.val_csv}
    if stacked_t is not None:
        avg_t = stacked_t / len(ckpt_paths)
        cal_t = calibrate_binary_threshold(avg_t, labels_t, cfg.num_classes)["best"]
        print(f"[ENSEMBLE] T macro_f1={cal_t['metrics']['macro_f1']:.4f} "
              f"@ thr {cal_t['threshold']}  recall={cal_t['metrics']['per_class_recall']}")
        res["ensemble_t"] = {"calibrated": cal_t}
    if stacked_n is not None:
        avg_n = stacked_n / len(ckpt_paths)
        cal_n = calibrate_binary_threshold(avg_n, labels_n, cfg.num_classes)["best"]
        print(f"[ENSEMBLE] N macro_f1={cal_n['metrics']['macro_f1']:.4f} "
              f"@ thr {cal_n['threshold']}  recall={cal_n['metrics']['per_class_recall']}")
        res["ensemble_n"] = {"calibrated": cal_n}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"集成汇总已保存: {args.out}")


if __name__ == "__main__":
    main()
