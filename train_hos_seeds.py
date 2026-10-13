# -*- coding: utf-8 -*-
"""多 seed 训练: 对多个随机种子各自分层划分 + 训练 + 阈值校准, 汇总 best macro-F1。

用法:
  python train_hos_seeds.py [--epochs 30] [--val-ratio 0.2] [--seeds 42,0,1,7]

输出:
  - checkpoint: checkpoints_hos/best_model_hos_s{seed}.pth (每个 seed 一个)
  - 曲线/混淆矩阵: plots_hos/training_curves_hos_s{seed}.png 等
  - 汇总: logs/seed_summary.json + 控制台表格
"""
import argparse
import json
import os
import sys
import warnings

warnings.simplefilter("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import HosConfig
from data_loader import build_dataloaders
from models import CRCMultiModalNet
from trainers import Trainer
from utils import setup_logger, calibrate_binary_threshold


def compute_class_weights(train_df, num_classes, mode):
    """T 主任务类别权重: 'inverse' 满逆频率 / 'sqrt' √逆频率 / 'none' 不加权。"""
    if mode == "none":
        return None
    cl, cnt = np.unique(train_df["label_T"].values, return_counts=True)
    counts = np.zeros(num_classes, dtype=float)
    counts[cl.astype(int)] = cnt
    inv_freq = len(train_df) / (num_classes * counts)
    return inv_freq if mode == "inverse" else np.sqrt(inv_freq)


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


def _apply_run_overrides(cfg, fusion, backbone, no_clinical, split_seed,
                         mixup, label_smoothing, freeze_low, share_backbone):
    """round6: 将 CLI 覆盖写入 cfg，返回 freeze_layers 列表。"""
    if fusion is not None:
        cfg.fusion_mode = fusion
    if backbone is not None:
        cfg.backbone_name = backbone
    if no_clinical:
        # 临床特征置零消融：clinical_cols=[] → loader 返回全 0 向量，模型结构不变
        cfg.clinical_cols = []
        cfg.save_dir = "./logs/_tmp_noclinc_ckpt"
        cfg.plot_dir = "./logs/_tmp_noclinc_plots"
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.plot_dir, exist_ok=True)
    if split_seed is not None:
        cfg.split_seed = split_seed   # A2: 所有 seed 共用同一划分 (seed 集成的前提)
    if mixup > 0:
        cfg.mixup = mixup
    if label_smoothing is not None:
        cfg.label_smoothing = label_smoothing
    if share_backbone:
        cfg.mri_share_backbone = True
    # freeze 策略: --freeze-low 覆盖 config 默认冻结清单 (冻 stem+layer1+2, 保留 layer3/4)
    if freeze_low:
        freeze_layers = ["conv1", "bn1", "act", "maxpool", "layer1", "layer2"]
    else:
        freeze_layers = getattr(cfg, "freeze_backbone_layers", None)
        # resnet10 参数量已减半(42.6M vs 79.9M), 低层全部可训练, 不再冻结
        if backbone == "resnet10":
            freeze_layers = []
    return freeze_layers


def run_seed(seed, epochs, val_ratio, fusion=None, backbone=None, no_clinical=False,
             split_seed=None, mixup=0.0, label_smoothing=None, freeze_low=False,
             share_backbone=False):
    cfg = HosConfig()
    cfg.epochs = epochs
    freeze_layers = _apply_run_overrides(
        cfg, fusion, backbone, no_clinical, split_seed, mixup, label_smoothing,
        freeze_low, share_backbone)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger = setup_logger("./logs", name=f"crc_seed{seed}")
    logger.info(f"===== Seed {seed} 开始 (epochs={epochs}) =====")

    df = pd.read_csv(cfg.train_csv)
    df = df[df["label_T"] >= 0].copy()
    df = df[df["label_N"] >= 0].copy()
    df = df.reset_index(drop=True)
    logger.info(f"有效样本: {len(df)}  label_T: {df['label_T'].value_counts().to_dict()}  "
                f"label_N: {df['label_N'].value_counts().to_dict()}")

    stratify = df["label_T"].astype(str) + "_" + df["label_N"].astype(str)
    # A2: split_seed 固定时所有 seed 用同一划分 (可比较 seed 差异 / 做 logit 集成)
    random_state = split_seed if split_seed is not None else seed
    train_df, val_df = train_test_split(df, test_size=val_ratio, stratify=stratify,
                                        random_state=random_state)
    os.makedirs("./data", exist_ok=True)
    train_csv = f"./data/hos_train_s{seed}.csv"
    val_csv = f"./data/hos_val_s{seed}.csv"
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")
    logger.info(f"train={len(train_df)}  val={len(val_df)}  "
                f"val label_T: {val_df['label_T'].value_counts().to_dict()}")

    train_loader, val_loader = build_dataloaders(cfg, train_csv, val_csv)
    model = build_model(cfg)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    class_weights = compute_class_weights(
        train_df, cfg.num_classes, getattr(cfg, "class_weight_mode", "inverse"))
    use_focal = getattr(cfg, "use_focal_loss", False)
    logger.info(f"class_weight_mode={getattr(cfg, 'class_weight_mode', 'inverse')}  "
                f"weights={None if class_weights is None else class_weights.round(3).tolist()}  "
                f"focal={use_focal}")
    trainer = Trainer(model, train_loader, val_loader, cfg, logger,
                      class_weights=class_weights,
                      label_smoothing=getattr(cfg, "label_smoothing", 0.0),
                      use_focal_loss=use_focal,
                      focal_gamma=getattr(cfg, "focal_gamma", 2.0),
                      freeze_backbone_layers=freeze_layers)
    result = trainer.fit(tag=f"hos_s{seed}{'_noclinc' if no_clinical else ''}")

    cal_t = cal_n = None
    if getattr(cfg, "calibrate_thresholds", False):
        proba = trainer.predict_proba()
        cal_t = calibrate_binary_threshold(proba["probs_t"], proba["labels_t"],
                                           cfg.num_classes)["best"]
        if "probs_n" in proba:
            proba_n = trainer.predict_proba(best_key="n")
            cal_n = calibrate_binary_threshold(proba_n["probs_n"], proba_n["labels_n"],
                                               cfg.num_classes)["best"]
        logger.info(f"[seed {seed}] 校准T@阈值{cal_t['threshold']}: "
                    f"macro_f1={cal_t['metrics']['macro_f1']:.4f} "
                    f"recall={cal_t['metrics']['per_class_recall']}")
        if cal_n:
            logger.info(f"[seed {seed}] 校准N@阈值{cal_n['threshold']}: "
                        f"macro_f1={cal_n['metrics']['macro_f1']:.4f} "
                        f"recall={cal_n['metrics']['per_class_recall']}")

    return {
        "seed": seed,
        "best_macro_f1": result["best_macro_f1"],
        "best_acc": result["best_acc"],
        "best_epoch": result["best_epoch"],
        "best_macro_f1_n": result.get("best_macro_f1_n", 0.0),
        "best_epoch_n": result.get("best_epoch_n", 0),
        "calibrated_t": cal_t,
        "calibrated_n": cal_n,
    }


def run_kfold(epochs, n_splits=5, kfold_seed=42, fusion=None, backbone=None,
              no_clinical=False, split_seed=None, mixup=0.0, label_smoothing=None,
              freeze_low=False, share_backbone=False):
    """round6 A1: StratifiedKFold 交叉验证（替代 0.2 holdout）。

    每折在 train 子集上训练并保存, 在该折 val 子集上评估;
    汇总各折 best val 指标 mean±std。checkpoint: best_model_hos_k{fold}.pth
    """
    cfg = HosConfig()
    cfg.epochs = epochs
    freeze_layers = _apply_run_overrides(
        cfg, fusion, backbone, no_clinical, split_seed, mixup, label_smoothing,
        freeze_low, share_backbone)
    logger = setup_logger("./logs", name="crc_kfold")
    logger.info(f"===== K-Fold CV 开始: {n_splits} folds, seed={kfold_seed}, "
                f"epochs={epochs} =====")

    df = pd.read_csv(cfg.train_csv)
    df = df[df["label_T"] >= 0].copy()
    df = df[df["label_N"] >= 0].copy()
    df = df.reset_index(drop=True)
    logger.info(f"有效样本: {len(df)}  label_T: {df['label_T'].value_counts().to_dict()}  "
                f"label_N: {df['label_N'].value_counts().to_dict()}")

    stratify = df["label_T"].astype(str) + "_" + df["label_N"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=kfold_seed)
    os.makedirs("./data", exist_ok=True)
    fold_summaries = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, stratify)):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        logger.info(f"--- Fold {fold}: train={len(train_df)} val={len(val_df)}  "
                    f"val label_T: {val_df['label_T'].value_counts().to_dict()}")
        train_csv = f"./data/hos_kfold_f{fold}_train.csv"
        val_csv = f"./data/hos_kfold_f{fold}_val.csv"
        train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")

        torch.manual_seed(kfold_seed * 1000 + fold)
        np.random.seed(kfold_seed * 1000 + fold)
        train_loader, val_loader = build_dataloaders(cfg, train_csv, val_csv)
        model = build_model(cfg)
        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        class_weights = compute_class_weights(
            train_df, cfg.num_classes, getattr(cfg, "class_weight_mode", "inverse"))
        trainer = Trainer(model, train_loader, val_loader, cfg, logger,
                          class_weights=class_weights,
                          label_smoothing=getattr(cfg, "label_smoothing", 0.0),
                          use_focal_loss=getattr(cfg, "use_focal_loss", False),
                          focal_gamma=getattr(cfg, "focal_gamma", 2.0),
                          freeze_backbone_layers=freeze_layers)
        result = trainer.fit(tag=f"hos_k{fold}")

        cal_t = cal_n = None
        if getattr(cfg, "calibrate_thresholds", False):
            proba = trainer.predict_proba()
            cal_t = calibrate_binary_threshold(proba["probs_t"], proba["labels_t"],
                                               cfg.num_classes)["best"]
            if "probs_n" in proba:
                proba_n = trainer.predict_proba(best_key="n")
                cal_n = calibrate_binary_threshold(proba_n["probs_n"], proba_n["labels_n"],
                                                   cfg.num_classes)["best"]
            logger.info(f"[fold {fold}] 校准T@阈值{cal_t['threshold']}: "
                        f"macro_f1={cal_t['metrics']['macro_f1']:.4f}")
            if cal_n:
                logger.info(f"[fold {fold}] 校准N@阈值{cal_n['threshold']}: "
                            f"macro_f1={cal_n['metrics']['macro_f1']:.4f}")

        fold_summaries.append({
            "fold": fold,
            "best_macro_f1": result["best_macro_f1"],
            "best_acc": result["best_acc"],
            "best_epoch": result["best_epoch"],
            "best_macro_f1_n": result.get("best_macro_f1_n", 0.0),
            "best_epoch_n": result.get("best_epoch_n", 0),
            "calibrated_t": cal_t,
            "calibrated_n": cal_n,
        })

    # ── 汇总 ──
    agg = {}
    for key in ("best_macro_f1", "best_acc", "best_macro_f1_n"):
        vals = [s[key] for s in fold_summaries]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals))}
        print(f"kfold {key}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    os.makedirs("./logs", exist_ok=True)
    summary_path = "./logs/kfold_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"n_splits": n_splits, "kfold_seed": kfold_seed,
                   "folds": fold_summaries, "aggregate": agg},
                  f, ensure_ascii=False, indent=2)
    print(f"K-Fold 汇总已保存: {summary_path}")
    return fold_summaries, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seeds", type=str, default="42,0,1,7")
    ap.add_argument("--fusion", type=str, default=None,
                    help="覆盖 fusion_mode: multi/single")
    ap.add_argument("--backbone", type=str, default=None,
                    help="覆盖 backbone_name: resnet18/resnet10")
    ap.add_argument("--no-clinical", action="store_true",
                    help="临床特征置零消融（模型结构不变，clinical 输入全 0）")
    # ── round6 新增 ──
    ap.add_argument("--kfold", type=int, default=0,
                    help=">1 时启用 StratifiedKFold 交叉验证（替代 holdout）")
    ap.add_argument("--kfold-seed", type=int, default=42,
                    help="k-fold 打乱 seed")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="固定 train/val 划分 random_state；所有 seed 共用同一划分"
                         "（seed logit 集成的前提）")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="MixUp 概率（0=关闭，建议 0.2~0.5）")
    ap.add_argument("--label-smoothing", type=float, default=None,
                    help="覆盖 cfg.label_smoothing")
    ap.add_argument("--freeze-low", action="store_true",
                    help="冻结 backbone stem+layer1+2（保留 layer3/4 训练）")
    ap.add_argument("--share-backbone", action="store_true",
                    help="MRI 三序列共享同一 encoder（权重共享减参）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.kfold > 1:
        run_kfold(args.epochs, n_splits=args.kfold, kfold_seed=args.kfold_seed,
                  fusion=args.fusion, backbone=args.backbone,
                  no_clinical=args.no_clinical, split_seed=args.split_seed,
                  mixup=args.mixup, label_smoothing=args.label_smoothing,
                  freeze_low=args.freeze_low, share_backbone=args.share_backbone)
        return

    summaries = [run_seed(s, args.epochs, args.val_ratio,
                          fusion=args.fusion, backbone=args.backbone,
                          no_clinical=args.no_clinical, split_seed=args.split_seed,
                          mixup=args.mixup, label_smoothing=args.label_smoothing,
                          freeze_low=args.freeze_low, share_backbone=args.share_backbone)
                 for s in seeds]

    # ── 汇总输出 ──
    print("\n" + "=" * 110)
    header = (f"{'seed':>5}  {'best_macro_f1':>14}  {'best_acc':>9}  {'best_epoch':>10}  "
              f"{'n_best_mf1':>11}  {'n_epoch':>7}  "
              f"{'calT_thr':>9}  {'calT_mf1':>8}  {'calT_recall':>20}  "
              f"{'calN_thr':>9}  {'calN_mf1':>8}  {'calN_recall':>20}")
    print(header)
    print("-" * 130)
    for s in summaries:
        ct = s.get("calibrated_t") or {}
        cn = s.get("calibrated_n") or {}
        print(f"{s['seed']:>5}  {s['best_macro_f1']:>14.4f}  {s['best_acc']:>9.4f}  "
              f"{s['best_epoch']:>10}  {s.get('best_macro_f1_n', 0.0):>11.4f}  "
              f"{s.get('best_epoch_n', 0):>7}  {str(ct.get('threshold', '-')):>9}  "
              f"{ct.get('metrics', {}).get('macro_f1', float('nan')):>8.4f}  "
              f"{str(ct.get('metrics', {}).get('per_class_recall', '-')):>20}  "
              f"{str(cn.get('threshold', '-')):>9}  "
              f"{cn.get('metrics', {}).get('macro_f1', float('nan')):>8.4f}  "
              f"{str(cn.get('metrics', {}).get('per_class_recall', '-')):>20}")
    print("=" * 130)

    mf1s = [s["best_macro_f1"] for s in summaries]
    print(f"best_macro_f1: mean={np.mean(mf1s):.4f}  std={np.std(mf1s):.4f}  "
          f"min={np.min(mf1s):.4f}  max={np.max(mf1s):.4f}")

    os.makedirs("./logs", exist_ok=True)
    summary_path = ("./logs/seed_summary_noclinc.json" if args.no_clinical
                    else "./logs/seed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"seeds": summaries,
                   "macro_f1": {"mean": float(np.mean(mf1s)), "std": float(np.std(mf1s)),
                                "min": float(np.min(mf1s)), "max": float(np.max(mf1s))}},
                  f, ensure_ascii=False, indent=2)
    print(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
