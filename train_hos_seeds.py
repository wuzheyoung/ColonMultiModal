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
import glob
import json
import os
import random
import re
import sys
import warnings

warnings.simplefilter("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import HosConfig, HOS_BRANCHES
from data_loader import build_dataloaders
from models import CRCMultiModalNet
from trainers import Trainer
from utils import setup_logger, calibrate_binary_threshold, build_eval_report, save_eval_report


def _seed_everything(seed, deterministic=False):
    """统一播种: random / numpy / torch(CPU+CUDA), 可选 cudnn 确定性。

    trainer._sample_modality_masks 用 Python random 采样模态 dropout, 旧版从未 random.seed
    → 同 seed 的 dropout 序列不可复现; 这里一并覆盖。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_class_weights(train_df, num_classes, mode):
    """T 主任务类别权重: 'inverse' 满逆频率 / 'sqrt' √逆频率 / 'none' 不加权。"""
    if mode == "none":
        return None
    cl, cnt = np.unique(train_df["label_T"].values, return_counts=True)
    counts = np.zeros(num_classes, dtype=float)
    counts[cl.astype(int)] = cnt
    inv_freq = len(train_df) / (num_classes * counts)
    return inv_freq if mode == "inverse" else np.sqrt(inv_freq)


def compute_class_weights_n(train_df, num_classes, mode):
    """round9: N 次任务类别权重（label_N 分布）, 与 T 同一公式——N 短板优先:
    给 N 的少数类(pN+)更高 loss 权重, 缓解 N 头被 T 主导。"""
    if mode == "none":
        return None
    cl, cnt = np.unique(train_df["label_N"].values, return_counts=True)
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
        mri_sequences=getattr(cfg, "mri_sequences", ("t", "c", "d")),
        head_width=getattr(cfg, "head_width", 256),
        pool_mode=getattr(cfg, "pool_mode", "avg"),
        gem_p=getattr(cfg, "gem_p", 3.0),
        n_from_shallow=getattr(cfg, "n_from_shallow", False),
        n_shallow_idx=getattr(cfg, "n_shallow_idx", 3),
        n_head_width=getattr(cfg, "n_head_width", None),
        n_dropout=getattr(cfg, "n_dropout", None),
        n_modal=getattr(cfg, "n_modal", "both"),
        n_cascade=getattr(cfg, "n_cascade", False),
        t_modal=getattr(cfg, "t_modal", "both"),
        ct_backbone_name=getattr(cfg, "ct_backbone_name", None),
        ct_pretrained=getattr(cfg, "ct_pretrained", True),
        use_eca=getattr(cfg, "use_eca", False),
        lesion_fusion=getattr(cfg, "lesion_fusion", False),
        mri_lesion_fusion=getattr(cfg, "mri_lesion_fusion", False),
        mri_lesion_backbone=getattr(cfg, "mri_lesion_backbone", "medicalnet10"),
        mri_lesion_size=getattr(cfg, "mri_lesion_size", 64),
        mri_lesion_head_width=getattr(cfg, "mri_lesion_head_width", 256),
        n_lesion_fusion=getattr(cfg, "n_lesion_fusion", False),
        lesion_backbone=getattr(cfg, "lesion_backbone", "medicalnet10"),
        lesion_size=getattr(cfg, "lesion_size", 64),
        lesion_head_width=getattr(cfg, "lesion_head_width", 256),
    )


def _apply_run_overrides(cfg, fusion, backbone, no_clinical, split_seed,
                         mixup, label_smoothing, freeze_low, share_backbone,
                         lr=None, class_weight_mode=None, focal=False,
                         aux_weight=None, rec_weight=None, head_dropout=None,
                         no_aug=False, pool=None, head_width=None, gem_p=None,
                         freeze_all=False, lr_n=None, n_shallow=False,
                         n_head_width=None, n_dropout=None,
                         n_shallow_idx=None, n_modal=None, n_cascade=False,
                         t_modal=None, ema_decay=None, consist_modal=None,
                         consist_weight=None, uncertainty_weight=False,
                         use_eca=None, snapshot_k=None, warm_restarts=False, md_both=None,
                         lesion_fusion=False, lesion_backbone=None,
                         lesion_size=None, lesion_head_width=None,
                         mri_lesion_fusion=False, mri_lesion_backbone=None,
                         mri_lesion_size=None, mri_lesion_head_width=None,
                         n_lesion_fusion=False, pcgrad=False, n_ring_fusion=False):
    """round6+: 将 CLI 覆盖写入 cfg，返回 freeze_layers 列表。

    round8 新增实验参数: lr / class_weight_mode / focal / aux_weight /
    rec_weight / head_dropout / no_aug / pool / head_width / gem_p，
    默认 None/False = 沿用 config，行为不变。
    """
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
    # ── round8 实验参数 ──
    if lr is not None:
        cfg.lr = lr
    if lr_n is not None:
        cfg.lr_n = lr_n   # round9 R4: N 头独立学习率
    if n_shallow:
        cfg.n_from_shallow = True   # round9 R5: N 特征解耦(浅层)
    if n_head_width is not None:
        cfg.n_head_width = n_head_width   # round9 R6: N 头独立宽度
    if n_dropout is not None:
        cfg.n_dropout = n_dropout         # round9 R6: N 头独立 dropout
    if n_shallow_idx is not None:
        cfg.n_shallow_idx = n_shallow_idx # round9 B1: N 浅层 index (layer2/3)
    if n_modal is not None:
        cfg.n_modal = n_modal             # round9 B3: N 特征来源 ct/mri/both
    if n_cascade:
        cfg.n_cascade = True              # round9 B2: T→N 级联
    if t_modal is not None:
        cfg.t_modal = t_modal             # round9 D1b: T 头特征来源 both/mri
    if ema_decay is not None:
        cfg.ema_decay = ema_decay         # round9 S3: EMA 影子权重 decay
    if consist_modal is not None:
        cfg.consist_modal = consist_modal # round9 C2: 一致性正则模态
    if consist_weight is not None:
        cfg.consist_weight = consist_weight  # round9 C2: 一致性 loss 权重
    if uncertainty_weight:
        cfg.uncertainty_weight = True  # round9 S9-1: T/N 不确定性加权
    if use_eca:
        cfg.use_eca = True  # round9 S9-③: 融合前 ECA 通道注意力
    if snapshot_k is not None:
        cfg.snapshot_k = snapshot_k  # round9 S9-⑤: 快照周期
    if warm_restarts:
        cfg.warm_restarts = True  # round9: 纯 warm restarts 不存快照
    if md_both is not None:
        cfg.modality_dropout["both"] = md_both  # round9: 模态 dropout both 概率调优
    if lesion_fusion:
        cfg.lesion_fusion = True  # round9 F1: 病灶融合分支
    if lesion_backbone is not None:
        cfg.lesion_backbone = lesion_backbone
    if lesion_size is not None:
        cfg.lesion_size = lesion_size
    if lesion_head_width is not None:
        cfg.lesion_head_width = lesion_head_width
    if mri_lesion_fusion:
        cfg.mri_lesion_fusion = True  # round9 MRI-t ROI: 结肠质心裁剪融合分支
    if mri_lesion_backbone is not None:
        cfg.mri_lesion_backbone = mri_lesion_backbone
    if mri_lesion_size is not None:
        cfg.mri_lesion_size = mri_lesion_size
    if mri_lesion_head_width is not None:
        cfg.mri_lesion_head_width = mri_lesion_head_width
    if n_lesion_fusion:
        cfg.n_lesion_fusion = True  # round9 ②: MRI-t ROI 嵌入只进 N 头 (n_parts), 不进 T 头
    if n_ring_fusion:
        # Day1⑤: ring (肠周/系膜) 版本 —— loader 的 mri_lesion 卷改用膨胀 2 层取环质心裁剪
        cfg.n_lesion_fusion = True
        cfg.mri_ring_dir = cfg.mri_lesion_dir
    if pcgrad:
        cfg.pcgrad = True  # round9 ④: PCGrad 梯度解耦 (T/N 梯度冲突投影)
    if class_weight_mode is not None:
        cfg.class_weight_mode = class_weight_mode
    if focal:
        cfg.use_focal_loss = True
    if aux_weight is not None:
        cfg.loss_weights["aux"] = aux_weight
    if rec_weight is not None:
        cfg.loss_weights["rec"] = rec_weight
    if head_dropout is not None:
        cfg.head_dropout = head_dropout
    if no_aug:
        cfg.use_augmentation = False
    # ── round8 A3/A4: 池化方式 / 头宽度 / GeM 指数 ──
    if pool is not None:
        cfg.pool_mode = pool
    if head_width is not None:
        cfg.head_width = head_width
    if gem_p is not None:
        cfg.gem_p = gem_p
    # freeze 策略: --freeze-all 全冻结 backbone (特征提取器不动, 只训融合/头, 治本: 参数与 284 样本匹配)
    if freeze_all:
        freeze_layers = ["conv1", "bn1", "relu", "act", "maxpool",
                         "layer1", "layer2", "layer3", "layer4"]
    elif freeze_low:
        freeze_layers = ["conv1", "bn1", "act", "maxpool", "layer1", "layer2"]
    else:
        freeze_layers = getattr(cfg, "freeze_backbone_layers", None)
        # resnet10/medicalnet10 参数量已减半, 低层全部可训练, 不再冻结
        if backbone in ("resnet10", "medicalnet10"):
            freeze_layers = []
    return freeze_layers


def _apply_branch(cfg, branch, mri_seq, ct_seq):
    """round7: 分支训练配置（定义见 config.hos.HOS_BRANCHES）。

    - MRI 分支: 强制 mri_only 模态路径（CT 走 imputer 伪 CT），mri_sequences 取对应子集
    - CT 分支: 强制 ct_only 模态路径（MRI 走 imputer 伪 MRI），ct_seq 取 plain/enhanced
    - full: 仅切换 csv（保留原模态 dropout 与三序列）
    mri_seq / ct_seq CLI 显式参数优先于分支默认。
    返回分支 tag（None=非分支）。
    """
    if mri_seq:
        seqs = tuple(s.strip() for s in mri_seq.split(",") if s.strip())
        cfg.mri_sequences = seqs
    if ct_seq:
        cfg.ct_seq = ct_seq
    if not branch:
        return None

    cfg.branch = branch
    for key, val in HOS_BRANCHES.get(branch, {}).items():
        setattr(cfg, key, val)
    cfg.train_csv = f"./data/hos_branch_{branch}.csv"
    cfg.val_csv = cfg.train_csv
    cfg.save_dir = f"./checkpoints_hos_{branch}"
    cfg.plot_dir = f"./plots_hos_{branch}"
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)
    return branch


def _apply_single_modal(cfg, sm):
    """round9-CAM 后: --single-modal 同批数据单模态消融（356 样本同 holdout，只喂单模态）。

    - 训练: modality_dropout p=1.0 → _sample_modality_masks 每样本必丢弃该模态
      (走 imputer 伪输入), 与旧 branch 实验同一机制
    - 验证: trainer._apply_single_modal 强制 val/predict_proba 掩码（与训练一致）
    - 返回标签后缀 (如 "_smct") 用于 ckpt/报告/日志隔离, 防覆盖双模态产物
    - sm="mri" → 丢弃 CT (ct_only=1.0);  sm="ct" → 丢弃 MRI (mri_only=1.0)
    """
    if not sm:
        return ""
    cfg.single_modal = sm
    if sm == "mri":
        cfg.modality_dropout = {"ct_only": 1.0, "mri_only": 0.0, "both": 0.0}
    else:
        cfg.modality_dropout = {"ct_only": 0.0, "mri_only": 1.0, "both": 0.0}
    return f"_sm{sm}"


def artifact_paths(cfg, fit_tag):
    """round7 整改: checkpoint / 曲线 / 混淆矩阵输出路径单一出口。

    命名与 trainers.Trainer.fit 的 suffix 完全对应:
      best_model_{fit_tag}.pth / best_model_n{fit_tag}.pth /
      training_curves_{fit_tag}.png / confusion_matrix_{fit_tag}.png
    """
    return {
        "ckpt": os.path.join(cfg.save_dir, f"best_model_{fit_tag}.pth"),
        "ckpt_n": os.path.join(cfg.save_dir, f"best_model_n{fit_tag}.pth"),
        "curves": os.path.join(cfg.plot_dir, f"training_curves_{fit_tag}.png"),
        "cm": os.path.join(cfg.plot_dir, f"confusion_matrix_{fit_tag}.png"),
    }


def _archive_run(logger, fit_tag, run_label, ap, rep_files):
    """round9: 每轮训练/评估结束后归档全部产物 → results/round9/run_NN_<fit_tag>/。

    复制不移动（保留原路径, 兼容 kfold-resume / --eval-only 的路径检查）。
    归档内容: T/N ckpt + 训练曲线 + 混淆矩阵 + 报告 json/txt/png + 训练日志。
    返回归档目录路径; 失败时打 warning 不中断流程。
    """
    import shutil
    root = os.path.join("results", "round9")
    try:
        os.makedirs(root, exist_ok=True)
        seq = 0
        if os.path.isdir(root):
            for d in os.listdir(root):
                m = re.match(r"^run_(\d+)_", d)
                if m:
                    seq = max(seq, int(m.group(1)))
        dest = os.path.join(root, f"run_{seq + 1:02d}_{fit_tag}")
        while True:  # 原子占位, 避免并发训练撞号 (run_93 曾重复)
            try:
                os.makedirs(dest)
                break
            except FileExistsError:
                seq += 1
                dest = os.path.join(root, f"run_{seq + 1:02d}_{fit_tag}")
        copied = []
        srcs = [ap["ckpt"], ap["ckpt_n"], ap["curves"], ap["cm"]]
        if rep_files:
            srcs += [rep_files.get("json"), rep_files.get("txt"), rep_files.get("png")]
        for s in srcs:
            if s and os.path.exists(s):
                shutil.copy2(s, os.path.join(dest, os.path.basename(s)))
                copied.append(os.path.basename(s))
        log_path = None
        for h in getattr(logger, "handlers", []):
            base = getattr(h, "baseFilename", None)
            if base:
                log_path = base
                break
        if log_path and os.path.exists(log_path):
            shutil.copy2(log_path, os.path.join(dest, os.path.basename(log_path)))
            copied.append(os.path.basename(log_path))
        logger.info(f"[{run_label}] 归档完成: {dest} ({len(copied)} 个文件: {', '.join(copied)})")
        return dest
    except Exception as e:  # 归档失败不影响训练结果
        logger.warning(f"[{run_label}] 归档失败: {e}")
        return None


def _train_eval_report(cfg, tag, logger, train_df, val_df, train_csv, val_csv,
                       fit_tag, freeze_layers, split_seed, run_label):
    """round7 整改: run_seed / run_kfold 共用的「训练→评估→报告」流水线。

    返回 (result, cal_t, cal_n, rep_files)。
    """
    train_loader, val_loader = build_dataloaders(cfg, train_csv, val_csv)
    model = build_model(cfg)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    # round8 A2: MedicalNet 预训练加载状态 (仅 medicalnet* backbone 有 pretrain_msg)
    _bb_msgs = []
    for _name, _bb in (("ct", model.ct_backbone),
                       ("mri_shared", getattr(model, "mri_shared_backbone", None))):
        if _bb is not None and hasattr(_bb, "pretrain_msg"):
            _bb_msgs.append(f"{_name}: {_bb.pretrain_msg}")
    for _seq, _bb in (getattr(model, "mri_backbones", None) or {}).items():
        if _bb is not None and hasattr(_bb, "pretrain_msg"):
            _bb_msgs.append(f"mri_{_seq}: {_bb.pretrain_msg}")
    if _bb_msgs:
        logger.info("backbone 预训练状态: " + " | ".join(_bb_msgs))

    class_weights = compute_class_weights(
        train_df, cfg.num_classes, getattr(cfg, "class_weight_mode", "inverse"))
    use_focal = getattr(cfg, "use_focal_loss", False)
    logger.info(f"class_weight_mode={getattr(cfg, 'class_weight_mode', 'inverse')}  "
                f"weights={None if class_weights is None else class_weights.round(3).tolist()}  "
                f"focal={use_focal}")
    trainer = Trainer(model, train_loader, val_loader, cfg, logger,
                      class_weights=class_weights,
                      class_weights_n=compute_class_weights_n(
                          train_df, cfg.num_classes,
                          getattr(cfg, "class_weight_mode", "inverse")),
                      label_smoothing=getattr(cfg, "label_smoothing", 0.0),
                      use_focal_loss=use_focal,
                      focal_gamma=getattr(cfg, "focal_gamma", 2.0),
                      freeze_backbone_layers=freeze_layers,
                      lr_n=getattr(cfg, "lr_n", None),
                      ema_decay=getattr(cfg, "ema_decay", None))
    result = trainer.fit(tag=fit_tag)

    # ── 每次训练完整评估: 全指标 + 阈值校准 + 文本/JSON/ROC-PR 图记录 ──
    proba = trainer.predict_proba()
    if "probs_n" in proba:
        # N 头指标用 N-best 权重重算, 不受 T 的 best epoch 绑架
        proba_n = trainer.predict_proba(best_key="n")
        proba["probs_n"], proba["labels_n"] = proba_n["probs_n"], proba_n["labels_n"]
    cal_t = cal_n = None
    if getattr(cfg, "calibrate_thresholds", False):
        cal_t = calibrate_binary_threshold(proba["probs_t"], proba["labels_t"],
                                           cfg.num_classes)["best"]
        if "probs_n" in proba:
            cal_n = calibrate_binary_threshold(proba["probs_n"], proba["labels_n"],
                                               cfg.num_classes)["best"]
        logger.info(f"[{run_label}] 校准T@阈值{cal_t['threshold']}: "
                    f"macro_f1={cal_t['metrics']['macro_f1']:.4f} "
                    f"recall={cal_t['metrics']['per_class_recall']}")
        if cal_n:
            logger.info(f"[{run_label}] 校准N@阈值{cal_n['threshold']}: "
                        f"macro_f1={cal_n['metrics']['macro_f1']:.4f} "
                        f"recall={cal_n['metrics']['per_class_recall']}")

    ap = artifact_paths(cfg, fit_tag)
    report = build_eval_report(
        cfg, tag, run_label, result, proba, cal_t, cal_n,
        train_df, val_df, train_csv, val_csv,
        split_seed=split_seed, freeze_layers=freeze_layers,
        ckpt_path=ap["ckpt"], curves_path=ap["curves"], cm_path=ap["cm"])
    rep_files = save_eval_report(report, tag, run_label)
    logger.info(f"[{run_label}] 评估报告已保存: txt={rep_files['txt']}  "
                f"json={rep_files['json']}  png={rep_files['png']}")
    _archive_run(logger, fit_tag, run_label, ap, rep_files)
    return result, cal_t, cal_n, rep_files


def _load_previous_result(tag, run_label):
    """--eval-only: 从已生成的报告 JSON 读训练 best 指标（不依赖训练日志）。

    报告 JSON 缺失时（如训练尾段崩溃未写报告）回退解析训练日志:
      - run_label="s{seed}" → logs/crc_{tag}_seed{seed}_*.log
      - run_label="k{fold}" → logs/crc_{tag}_kfold*.log 中该 Fold 的段落
    再按 "Saved best model"/"Saved best N model" 行取值, best_epoch 取前一条 [Val] Epoch。
    """
    tag_str = tag or "hos"
    path = os.path.join("./logs", f"eval_report_{tag_str}_{run_label}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("result", {})
    best = {"best_epoch": 0, "best_macro_f1": 0.0, "best_acc": 0.0,
            "best_epoch_n": 0, "best_macro_f1_n": 0.0}
    m_k = re.match(r"^k(\d+)", run_label)
    if m_k:  # kfold: 单一日志内按 Fold 段落截取 (旧正则不匹配 "k0" → 永远查不到日志)
        fold = int(m_k.group(1))
        logs = sorted(glob.glob(os.path.join("./logs", f"crc_{tag_str}_kfold*.log")),
                      key=os.path.getmtime)
        if not logs:
            return best
        full = open(logs[-1], encoding="utf-8", errors="replace").read()
        m = re.search(rf"(?ms)^--- Fold {fold}:.*?(?=^--- Fold \d+:|\Z)", full)
        text = m.group(0) if m else ""
    else:
        seed = re.sub(r"^s", "", run_label)
        logs = sorted(glob.glob(os.path.join("./logs", f"crc_{tag_str}_seed{seed}*.log")),
                      key=os.path.getmtime)
        if not logs:
            return best
        text = open(logs[-1], encoding="utf-8", errors="replace").read()
    cur_ep = 0
    for ln in text.splitlines():
        m = re.search(r"\[Val\]\s+Epoch\s+(\d+)", ln)
        if m:
            cur_ep = int(m.group(1))
        m = re.search(r"Saved best model \(macro_f1=([\d.]+), acc=([\d.]+)\)", ln)
        if m:
            best["best_epoch"] = cur_ep
            best["best_macro_f1"] = float(m.group(1))
            best["best_acc"] = float(m.group(2))
        m = re.search(r"Saved best N model \(n_macro_f1=([\d.]+), n_acc=([\d.]+)\)", ln)
        if m:
            best["best_epoch_n"] = cur_ep
            best["best_macro_f1_n"] = float(m.group(1))
    return best


def _eval_one_existing(cfg, tag, logger, train_csv, val_csv, fit_tag, run_label,
                       freeze_layers, split_seed):
    """--eval-only: 单个 seed/fold 的只读评估（不训练、不更新权重）。"""
    ap = artifact_paths(cfg, fit_tag)
    if (not os.path.exists(ap["ckpt"]) or not os.path.exists(train_csv)
            or not os.path.exists(val_csv)):
        logger.warning(f"跳过 {run_label}: checkpoint 或 split csv 缺失 "
                       f"({ap['ckpt']}, {train_csv})")
        return False
    logger.info(f"--- {run_label}: ckpt={ap['ckpt']}")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_loader, val_loader = build_dataloaders(cfg, train_csv, val_csv)
    model = build_model(cfg)
    try:
        model.load_state_dict(torch.load(ap["ckpt"], map_location=cfg.device))
    except (RuntimeError, KeyError) as e:
        logger.warning(f"跳过 {run_label}: ckpt 结构不匹配当前参数 "
                       f"(请确认 --branch/--fusion/--share-backbone/--backbone/"
                       f"--mri-seq 与训练时一致): {str(e)[:200]}")
        return False
    trainer = Trainer(model, train_loader, val_loader, cfg, logger,
                      freeze_backbone_layers=freeze_layers,
                      ema_decay=getattr(cfg, "ema_decay", None))
    result = _load_previous_result(tag, run_label)
    proba = trainer.predict_proba()
    if "probs_n" in proba:
        proba_n = trainer.predict_proba(best_key="n")
        proba["probs_n"], proba["labels_n"] = proba_n["probs_n"], proba_n["labels_n"]
    cal_t = cal_n = None
    if getattr(cfg, "calibrate_thresholds", False):
        cal_t = calibrate_binary_threshold(proba["probs_t"], proba["labels_t"],
                                           cfg.num_classes)["best"]
        if "probs_n" in proba:
            cal_n = calibrate_binary_threshold(proba["probs_n"], proba["labels_n"],
                                               cfg.num_classes)["best"]
        logger.info(f"[{run_label}] 校准T@阈值{cal_t['threshold']}: "
                    f"macro_f1={cal_t['metrics']['macro_f1']:.4f} "
                    f"recall={cal_t['metrics']['per_class_recall']}")
    report = build_eval_report(
        cfg, tag, run_label, result, proba, cal_t, cal_n,
        train_df, val_df, train_csv, val_csv,
        split_seed=split_seed, freeze_layers=freeze_layers,
        ckpt_path=ap["ckpt"], curves_path=ap["curves"], cm_path=ap["cm"])
    rep_files = save_eval_report(report, tag, run_label)
    logger.info(f"[{run_label}] 评估报告已保存: txt={rep_files['txt']}  "
                f"json={rep_files['json']}  png={rep_files['png']}")
    _archive_run(logger, fit_tag, run_label, ap, rep_files)
    return True


def eval_existing(seeds, epochs, kfold, kfold_seed, fusion, backbone, no_clinical,
                  split_seed, mixup, label_smoothing, freeze_low, share_backbone,
                  branch, mri_seq, ct_seq, freeze_all=False, single_modal=None,
                  lr=None, class_weight_mode=None, focal=False,
                  aux_weight=None, rec_weight=None, head_dropout=None, no_aug=False,
                  pool=None, head_width=None, gem_p=None, lr_n=None,
                  n_shallow=False, n_head_width=None, n_dropout=None,
                  n_shallow_idx=None, n_modal=None, n_cascade=False,
                  t_modal=None, ema_decay=None, ct_backbone=None, ct_pretrained=True,
                  batch_size=None, consist_modal=None, consist_weight=None,
                  uncertainty_weight=False, use_eca=None, snapshot_k=None,
                   warm_restarts=False, md_both=None,
                   lesion_fusion=False, lesion_backbone=None,
                   lesion_size=None, lesion_head_width=None,
                   mri_lesion_fusion=False, mri_lesion_backbone=None,
                   mri_lesion_size=None, mri_lesion_head_width=None,
                   n_lesion_fusion=False, pcgrad=False, n_ring_fusion=False):
    """round7 整改: 不训练, 用已有 checkpoint + split csv 重出完整报告。

    round9 修复: 完整接收 round8/9 实验参数（lr/n_shallow/pool/head_width/lr_n/...），
    使 --eval-only 构建的模型结构与训练时完全一致（否则 R5 等 ckpt load 失败）。

    holdout:  python train_hos_seeds.py --eval-only --branch mri_t --seeds 42
    kfold:    python train_hos_seeds.py --eval-only --kfold 5 --branch ct_z
    """
    cfg = HosConfig()
    cfg.epochs = epochs
    if batch_size:
        cfg.batch_size = batch_size
    if ct_backbone == "swin":
        cfg.ct_backbone_name = "swin"
        cfg.ct_pretrained = ct_pretrained
    freeze_layers = _apply_run_overrides(
        cfg, fusion, backbone, no_clinical, split_seed, mixup, label_smoothing,
        freeze_low, share_backbone, lr=lr, class_weight_mode=class_weight_mode,
        focal=focal, aux_weight=aux_weight, rec_weight=rec_weight,
        head_dropout=head_dropout, no_aug=no_aug, pool=pool,
        head_width=head_width, gem_p=gem_p, freeze_all=freeze_all, lr_n=lr_n,
        n_shallow=n_shallow, n_head_width=n_head_width, n_dropout=n_dropout,
        n_shallow_idx=n_shallow_idx, n_modal=n_modal, n_cascade=n_cascade,
        t_modal=t_modal, ema_decay=ema_decay,
        consist_modal=consist_modal, consist_weight=consist_weight,
        uncertainty_weight=uncertainty_weight, use_eca=use_eca,
        snapshot_k=snapshot_k, warm_restarts=warm_restarts, md_both=md_both,
        lesion_fusion=lesion_fusion, lesion_backbone=lesion_backbone,
        lesion_size=lesion_size, lesion_head_width=lesion_head_width,
        mri_lesion_fusion=mri_lesion_fusion, mri_lesion_backbone=mri_lesion_backbone,
        mri_lesion_size=mri_lesion_size, mri_lesion_head_width=mri_lesion_head_width,
        n_lesion_fusion=n_lesion_fusion, pcgrad=pcgrad, n_ring_fusion=n_ring_fusion)
    tag = _apply_branch(cfg, branch, mri_seq, ct_seq)
    sm_suffix = _apply_single_modal(cfg, single_modal)
    logger = setup_logger("./logs", name=f"crc_{tag}_evalonly{sm_suffix}" if tag else f"crc_evalonly{sm_suffix}")
    logger.info(f"===== 仅评估已有 checkpoint (branch={branch}, "
                f"mri_seq={cfg.mri_sequences}, ct_seq={cfg.ct_seq}, "
                f"single_modal={single_modal}) =====")
    done = []
    if kfold > 1:
        for fold in range(kfold):
            train_csv = (f"./data/hos_{tag}_kfold_f{fold}_train.csv" if tag
                         else f"./data/hos_kfold_f{fold}_train.csv")
            val_csv = (f"./data/hos_{tag}_kfold_f{fold}_val.csv" if tag
                       else f"./data/hos_kfold_f{fold}_val.csv")
            fit_tag = f"hos_{tag}_k{fold}{sm_suffix}" if tag else f"hos_k{fold}{sm_suffix}"
            if _eval_one_existing(cfg, tag, logger, train_csv, val_csv, fit_tag,
                                  f"k{fold}{sm_suffix}", freeze_layers, kfold_seed):
                done.append(f"k{fold}{sm_suffix}")
    else:
        for seed in seeds:
            train_csv = (f"./data/hos_{tag}_train_s{seed}.csv" if tag
                         else f"./data/hos_train_s{seed}.csv")
            val_csv = (f"./data/hos_{tag}_val_s{seed}.csv" if tag
                       else f"./data/hos_val_s{seed}.csv")
            fit_tag = f"hos_{tag}_s{seed}{sm_suffix}" if tag else f"hos_s{seed}{sm_suffix}"
            if _eval_one_existing(cfg, tag, logger, train_csv, val_csv, fit_tag,
                                  f"s{seed}{sm_suffix}", freeze_layers, split_seed):
                done.append(f"s{seed}{sm_suffix}")
    logger.info(f"仅评估完成: {done if done else '(无可用 checkpoint / split csv)'}")


def run_seed(seed, epochs, val_ratio, fusion=None, backbone=None, no_clinical=False,
             split_seed=None, mixup=0.0, label_smoothing=None, freeze_low=False,
             share_backbone=False, branch=None, mri_seq=None, ct_seq=None,
             lr=None, class_weight_mode=None, focal=False,
             aux_weight=None, rec_weight=None, head_dropout=None, no_aug=False,
             pool=None, head_width=None, gem_p=None, freeze_all=False, lr_n=None,
             n_shallow=False, n_head_width=None, n_dropout=None,
             n_shallow_idx=None, n_modal=None, n_cascade=False, single_modal=None,
             t_modal=None, ema_decay=None, ct_backbone=None, ct_pretrained=True,
             batch_size=None, consist_modal=None, consist_weight=None,
             uncertainty_weight=False, use_eca=None, snapshot_k=None,
                   warm_restarts=False, md_both=None,
                   lesion_fusion=False, lesion_backbone=None,
                   lesion_size=None, lesion_head_width=None,
                   mri_lesion_fusion=False, mri_lesion_backbone=None,
                   mri_lesion_size=None, mri_lesion_head_width=None,
                   n_lesion_fusion=False, pcgrad=False, n_ring_fusion=False):
    cfg = HosConfig()
    cfg.epochs = epochs
    if batch_size:
        cfg.batch_size = batch_size
    if ct_backbone == "swin":
        cfg.ct_backbone_name = "swin"
        cfg.ct_pretrained = ct_pretrained
    freeze_layers = _apply_run_overrides(
        cfg, fusion, backbone, no_clinical, split_seed, mixup, label_smoothing,
        freeze_low, share_backbone, lr=lr, class_weight_mode=class_weight_mode,
        focal=focal, aux_weight=aux_weight, rec_weight=rec_weight,
        head_dropout=head_dropout, no_aug=no_aug, pool=pool,
        head_width=head_width, gem_p=gem_p, freeze_all=freeze_all, lr_n=lr_n,
        n_shallow=n_shallow, n_head_width=n_head_width, n_dropout=n_dropout,
        n_shallow_idx=n_shallow_idx, n_modal=n_modal, n_cascade=n_cascade,
        t_modal=t_modal, ema_decay=ema_decay,
        consist_modal=consist_modal, consist_weight=consist_weight,
        uncertainty_weight=uncertainty_weight, use_eca=use_eca,
        snapshot_k=snapshot_k, warm_restarts=warm_restarts, md_both=md_both,
        lesion_fusion=lesion_fusion, lesion_backbone=lesion_backbone,
        lesion_size=lesion_size, lesion_head_width=lesion_head_width,
        mri_lesion_fusion=mri_lesion_fusion, mri_lesion_backbone=mri_lesion_backbone,
        mri_lesion_size=mri_lesion_size, mri_lesion_head_width=mri_lesion_head_width,
        n_lesion_fusion=n_lesion_fusion, pcgrad=pcgrad, n_ring_fusion=n_ring_fusion)
    tag = _apply_branch(cfg, branch, mri_seq, ct_seq)
    sm_suffix = _apply_single_modal(cfg, single_modal)
    _seed_everything(seed, bool(getattr(cfg, "deterministic", False)))
    logger = setup_logger(
        "./logs", name=f"crc_{tag}_seed{seed}{sm_suffix}" if tag else f"crc_seed{seed}{sm_suffix}")
    logger.info(f"===== Seed {seed} (branch={branch}, mri_seq={cfg.mri_sequences}, "
                f"ct_seq={cfg.ct_seq}, epochs={epochs}, single_modal={single_modal}) =====")
    logger.info(f"实验参数: backbone={cfg.backbone_name}, lr={cfg.lr}, "
                f"class_weight_mode={cfg.class_weight_mode}, focal={cfg.use_focal_loss}, "
                f"loss_weights={cfg.loss_weights}, head_dropout={cfg.head_dropout}, "
                f"head_width={getattr(cfg, 'head_width', 256)}, "
                f"pool={getattr(cfg, 'pool_mode', 'avg')}, "
                f"mixup={getattr(cfg, 'mixup', 0.0)}, "
                f"use_aug={getattr(cfg, 'use_augmentation', True)}, "
                f"label_smoothing={cfg.label_smoothing}, freeze={freeze_layers}")

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
    # 划分文件名带上真正使用的 split_seed (与 seed 不同才加后缀):
    # 旧命名只含 seed, 换 --split-seed 会静默覆盖同名 csv, 导致 ckpt 与划分错配;
    # 默认 split_seed 为空/等于 seed 时保持旧文件名, 不破坏现有脚本引用。
    sp_suffix = f"_sp{split_seed}" if (split_seed is not None and split_seed != seed) else ""
    train_csv = (f"./data/hos_{tag}_train_s{seed}{sp_suffix}.csv" if tag
                 else f"./data/hos_train_s{seed}{sp_suffix}.csv")
    val_csv = (f"./data/hos_{tag}_val_s{seed}{sp_suffix}.csv" if tag
               else f"./data/hos_val_s{seed}{sp_suffix}.csv")
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")
    logger.info(f"train={len(train_df)}  val={len(val_df)}  "
                f"val label_T: {val_df['label_T'].value_counts().to_dict()}")

    fit_tag = (f"hos_{tag}_s{seed}{sm_suffix}{'_noclinc' if no_clinical else ''}" if tag
               else f"hos_s{seed}{sm_suffix}{'_noclinc' if no_clinical else ''}")
    result, cal_t, cal_n, rep_files = _train_eval_report(
        cfg, tag, logger, train_df, val_df, train_csv, val_csv, fit_tag,
        freeze_layers, split_seed, f"s{seed}{sm_suffix}")

    return {
        "seed": seed,
        "best_macro_f1": result["best_macro_f1"],
        "best_acc": result["best_acc"],
        "best_epoch": result["best_epoch"],
        "best_macro_f1_n": result.get("best_macro_f1_n", 0.0),
        "best_epoch_n": result.get("best_epoch_n", 0),
        "calibrated_t": cal_t,
        "calibrated_n": cal_n,
        "report": rep_files,
    }


def run_kfold(epochs, n_splits=5, kfold_seed=42, fusion=None, backbone=None,
              no_clinical=False, split_seed=None, mixup=0.0, label_smoothing=None,
              freeze_low=False, share_backbone=False, branch=None, mri_seq=None,
              ct_seq=None, lr=None, class_weight_mode=None, focal=False,
              aux_weight=None, rec_weight=None, head_dropout=None, no_aug=False,
              pool=None, head_width=None, gem_p=None, freeze_all=False, lr_n=None,
              n_shallow=False, n_head_width=None, n_dropout=None,
              n_shallow_idx=None, n_modal=None, n_cascade=False, single_modal=None,
              kfold_resume=False, kfold_max_folds=None, ema_decay=None,
              ct_backbone=None, ct_pretrained=True, t_modal=None,
              batch_size=None, consist_modal=None, consist_weight=None,
              uncertainty_weight=False, use_eca=None, snapshot_k=None,
                   warm_restarts=False, md_both=None,
                   lesion_fusion=False, lesion_backbone=None,
                   lesion_size=None, lesion_head_width=None,
                   mri_lesion_fusion=False, mri_lesion_backbone=None,
                   mri_lesion_size=None, mri_lesion_head_width=None,
                   n_lesion_fusion=False, pcgrad=False, n_ring_fusion=False):
    """round6 A1: StratifiedKFold 交叉验证（替代 0.2 holdout）。

    每折在 train 子集上训练并保存, 在该折 val 子集上评估;
    汇总各折 best val 指标 mean±std。checkpoint: best_model_hos_k{fold}.pth

    round9-kfold-resume: 已有 eval_report_{tag}_k{fold}.json 的折跳过训练,
    从 JSON 读 best 指标入汇总 (cal_t/cal_n=None)。kfold_max_folds=N: 本进程
    最多训练 N 个新折后停止 —— 用于 GPU 环境卡死时逐折分进程跑(每 job 一折,
    进程重启规避累积性硬卡死; resume 折不占配额)。
    """
    cfg = HosConfig()
    cfg.epochs = epochs
    if batch_size:
        cfg.batch_size = batch_size
    if ct_backbone == "swin":
        cfg.ct_backbone_name = "swin"
        cfg.ct_pretrained = ct_pretrained
    freeze_layers = _apply_run_overrides(
        cfg, fusion, backbone, no_clinical, split_seed, mixup, label_smoothing,
        freeze_low, share_backbone, lr=lr, class_weight_mode=class_weight_mode,
        focal=focal, aux_weight=aux_weight, rec_weight=rec_weight,
        head_dropout=head_dropout, no_aug=no_aug, pool=pool,
        head_width=head_width, gem_p=gem_p, freeze_all=freeze_all, lr_n=lr_n,
        n_shallow=n_shallow, n_head_width=n_head_width, n_dropout=n_dropout,
        n_shallow_idx=n_shallow_idx, n_modal=n_modal, n_cascade=n_cascade,
        t_modal=t_modal, ema_decay=ema_decay,
        consist_modal=consist_modal, consist_weight=consist_weight,
        uncertainty_weight=uncertainty_weight, use_eca=use_eca,
        snapshot_k=snapshot_k, warm_restarts=warm_restarts, md_both=md_both,
        lesion_fusion=lesion_fusion, lesion_backbone=lesion_backbone,
        lesion_size=lesion_size, lesion_head_width=lesion_head_width,
        mri_lesion_fusion=mri_lesion_fusion, mri_lesion_backbone=mri_lesion_backbone,
        mri_lesion_size=mri_lesion_size, mri_lesion_head_width=mri_lesion_head_width,
        n_lesion_fusion=n_lesion_fusion, pcgrad=pcgrad, n_ring_fusion=n_ring_fusion)
    tag = _apply_branch(cfg, branch, mri_seq, ct_seq)
    sm_suffix = _apply_single_modal(cfg, single_modal)
    logger = setup_logger("./logs", name=f"crc_{tag}_kfold{sm_suffix}" if tag else f"crc_kfold{sm_suffix}")
    logger.info(f"===== K-Fold CV 开始: {n_splits} folds, seed={kfold_seed}, "
                f"branch={branch}, mri_seq={cfg.mri_sequences}, ct_seq={cfg.ct_seq}, "
                f"epochs={epochs}, single_modal={single_modal} =====")

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
    trained_this_run = 0
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, stratify)):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        logger.info(f"--- Fold {fold}: train={len(train_df)} val={len(val_df)}  "
                    f"val label_T: {val_df['label_T'].value_counts().to_dict()}")
        train_csv = f"./data/hos_{tag}_kfold_f{fold}_train.csv" if tag else f"./data/hos_kfold_f{fold}_train.csv"
        val_csv = f"./data/hos_{tag}_kfold_f{fold}_val.csv" if tag else f"./data/hos_kfold_f{fold}_val.csv"
        train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")

        fit_tag = f"hos_{tag}_k{fold}{sm_suffix}" if tag else f"hos_k{fold}{sm_suffix}"
        rep_json = (f"./logs/eval_report_{tag}_k{fold}{sm_suffix}.json" if tag
                    else f"./logs/eval_report_k{fold}{sm_suffix}.json")
        if kfold_resume and os.path.exists(rep_json):
            with open(rep_json, encoding="utf-8") as f:
                rep = json.load(f)
            res = rep.get("result", {})
            logger.info(f"--- Fold {fold}: 已有报告 {rep_json}, 跳过训练 (resume)")
            fold_summaries.append({
                "fold": fold, "resumed": True,
                "best_macro_f1": res.get("best_macro_f1", 0.0),
                "best_acc": res.get("best_acc", 0.0),
                "best_epoch": res.get("best_epoch", 0),
                "best_macro_f1_n": res.get("best_macro_f1_n", 0.0),
                "best_epoch_n": res.get("best_epoch_n", 0),
                "calibrated_t": None, "calibrated_n": None,
                "report": [rep_json],
            })
            continue
        if kfold_max_folds is not None and trained_this_run >= kfold_max_folds:
            logger.info(f"本进程已达 kfold_max_folds={kfold_max_folds}, 停止后续折训练")
            break

        _seed_everything(kfold_seed * 1000 + fold, bool(getattr(cfg, "deterministic", False)))
        result, cal_t, cal_n, rep_files = _train_eval_report(
            cfg, tag, logger, train_df, val_df, train_csv, val_csv, fit_tag,
            freeze_layers, kfold_seed, f"k{fold}{sm_suffix}")
        trained_this_run += 1

        fold_summaries.append({
            "fold": fold, "resumed": False,
            "best_macro_f1": result["best_macro_f1"],
            "best_acc": result["best_acc"],
            "best_epoch": result["best_epoch"],
            "best_macro_f1_n": result.get("best_macro_f1_n", 0.0),
            "best_epoch_n": result.get("best_epoch_n", 0),
            "calibrated_t": cal_t,
            "calibrated_n": cal_n,
            "report": rep_files,
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
    summary_path = (f"./logs/kfold_summary_{tag}{sm_suffix}.json" if tag
                    else f"./logs/kfold_summary{sm_suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"n_splits": n_splits, "kfold_seed": kfold_seed,
                   "folds": fold_summaries, "aggregate": agg},
                  f, ensure_ascii=False, indent=2)
    print(f"K-Fold 汇总已保存: {summary_path}")
    return fold_summaries, agg


def _save_summary(args, summaries):
    """round7 整改: 逐 seed 增量落盘 seed_summary，中途崩溃不丢已完成结果。"""
    if not summaries:
        return None
    os.makedirs("./logs", exist_ok=True)
    if args.branch:
        summary_path = f"./logs/seed_summary_{args.branch}.json"
    else:
        summary_path = ("./logs/seed_summary_noclinc.json" if args.no_clinical
                        else "./logs/seed_summary.json")
    mf1s = [s["best_macro_f1"] for s in summaries]
    payload = {"seeds": summaries,
               "macro_f1": {"mean": float(np.mean(mf1s)), "std": float(np.std(mf1s)),
                            "min": float(np.min(mf1s)), "max": float(np.max(mf1s))}}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return summary_path


def _parse_nidx(v):
    """round9 S9-④: '3' → 3; '3,4' → (3, 4) —— N 头多 stage 拼接。"""
    parts = [int(x.strip()) for x in v.split(",") if x.strip()]
    return parts[0] if len(parts) == 1 else tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60,
                    help="训练轮数（默认与 config.hos.epochs 一致=60）")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seeds", type=str, default="42,0,1,7")
    ap.add_argument("--eval-only", action="store_true",
                    help="round7 整改: 不训练, 用已有 checkpoint + split csv 只读重出评估报告")
    ap.add_argument("--fusion", type=str, default=None,
                    help="覆盖 fusion_mode: multi/single/concat (concat=round9 F1 消融交叉注意力)")
    ap.add_argument("--backbone", type=str, default=None,
                    choices=["resnet18", "resnet10", "medicalnet18", "medicalnet10"],
                    help="覆盖 backbone_name: resnet18/resnet10(MONAI, ImageNet-2D→3D广播) / "
                         "medicalnet18/medicalnet10(MedicalNet 23数据集 3D 医学预训练, "
                         "权重放 pretrained/)")
    ap.add_argument("--no-clinical", action="store_true",
                    help="临床特征置零消融（模型结构不变，clinical 输入全 0）")
    ap.add_argument("--deterministic", action="store_true",
                    help="开启 cudnn 确定性 (deterministic=True/benchmark=False), 同 seed 严格复现; 速度略降")
    # ── round6 新增 ──
    ap.add_argument("--kfold", type=int, default=0,
                    help=">1 时启用 StratifiedKFold 交叉验证（替代 holdout）")
    ap.add_argument("--kfold-seed", type=int, default=42,
                    help="k-fold 打乱 seed")
    ap.add_argument("--kfold-resume", action="store_true",
                    help="round9-kfold-resume: 已有 eval_report_{tag}_k{fold}.json 的折跳过训练"
                         "(从 JSON 读 best 指标), 配合 --kfold-max-folds 逐折分进程续跑")
    ap.add_argument("--kfold-max-folds", type=int, default=None,
                    help="round9-kfold-resume: 本进程最多训练 N 个新折后停止"
                         "(GPU 环境硬卡死时每 job 一折, 进程重启规避累积崩溃; resume 折不占配额)")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="固定 train/val 划分 random_state；所有 seed 共用同一划分"
                         "（seed logit 集成的前提）")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="MixUp 概率（0=关闭，建议 0.2~0.5）")
    ap.add_argument("--label-smoothing", type=float, default=None,
                    help="覆盖 cfg.label_smoothing")
    ap.add_argument("--freeze-low", action="store_true",
                    help="冻结 backbone stem+layer1+2（保留 layer3/4 训练）")
    ap.add_argument("--freeze-all", action="store_true",
                    help="round8 治本: 全冻结 backbone(含 layer4, 特征提取器不动), "
                         "只训练融合层+双头 (可训练参数 ~2M, 适合 284 小样本)")
    ap.add_argument("--share-backbone", action="store_true",
                    help="MRI 三序列共享同一 encoder（权重共享减参）")
    # ── round7 分支训练 ──
    ap.add_argument("--branch", type=str, default=None,
                    choices=["mri_t", "mri_d", "mri_c", "mri_tdc", "ct_p", "ct_z", "full"],
                    help="分支子集: MRI 单序列 / CT 平扫 / CT 增强 / 全模态（对应 data/hos_branch_*.csv）")
    ap.add_argument("--mri-seq", type=str, default=None,
                    help="MRI 序列子集, 逗号分隔（如 't' / 't,c,d'；覆盖分支默认）")
    ap.add_argument("--ct-seq", type=str, default=None,
                    choices=["auto", "plain", "enhanced"],
                    help="CT 序列选择: auto(原优先级) / plain(平扫) / enhanced(增强)")
    # ── round8 实验参数（默认 None/False=沿用 config.hos，行为不变）──
    ap.add_argument("--lr", type=float, default=None,
                    help="覆盖 cfg.lr（如 3e-4 / 1e-3）")
    ap.add_argument("--lr-n", type=float, default=None,
                    help="round9 R4: N 头(classifier_n)独立学习率——解决双任务 lr 冲突"
                         "（lr3e-4 偏好 N / lr1e-4 偏好 T）")
    ap.add_argument("--n-shallow", action="store_true",
                    help="round9 R5: N 特征解耦——N 头从浅层特征(CT/MRI layer3)+临床取输入,"
                         "不经 T 主导的融合通道, 解除 T/N 互相干扰")
    ap.add_argument("--n-width", type=int, default=None,
                    help="round9 R6: N 头独立宽度(默认=主头宽度)——防 N 早期过拟合")
    ap.add_argument("--n-dropout", type=float, default=None,
                    help="round9 R6: N 头独立 dropout(默认=主头 dropout)——防 N 早期过拟合")
    ap.add_argument("--n-shallow-idx", type=_parse_nidx, default=None,
                    help="round9 B1: N 浅层特征 index (0-4, 默认3=layer3; 2=layer2 更原始纹理); "
                         "round9 S9-④: 支持逗号分隔多 stage 拼接, 如 '3,4' = layer3+4 (纹理+语义)")
    ap.add_argument("--n-modal", type=str, default=None, choices=["ct", "mri", "both"],
                    help="round9 B3: N 特征来源——ct/mri/both(默认, 需配合 --n-shallow)")
    ap.add_argument("--t-modal", type=str, default=None, choices=["mri", "both"],
                    help="round9 D1b: T 头特征来源——mri=MRI 以 CT 为上下文的增强特征(调换 cross-attn "
                         "query/context), both=默认(CT 作 query)")
    ap.add_argument("--single-modal", type=str, default=None, choices=["ct", "mri"],
                    help="round9-CAM 后: 同批数据单模态消融——ct=只喂CT(MRI走imputer伪输入), "
                         "mri=只喂MRI(CT走imputer); 训练与验证一致强制, 产物带 _smct/_smmri 后缀隔离")
    ap.add_argument("--n-cascade", action="store_true",
                    help="round9 B2: T→N 级联——N 头输入追加 T 预测概率(softmax, detach),"
                         "医学先验: pT 深→pN+ 概率高")
    ap.add_argument("--ct-backbone", type=str, default=None, choices=["same", "swin"],
                    help="round9 SwinUNETR: CT 分支 backbone——same=跟随 --backbone(默认), "
                         "swin=CT 用 SwinUNETR(Swim ViT, MONAI SSL 预训练 719MB), MRI 保持 --backbone")
    ap.add_argument("--lesion-fusion", action="store_true",
                    help="round9 F1: 病灶融合分支——CT Task010 mask 质心 64³ 裁剪 → medicalnet10 → GeM, "
                         "特征与全体积融合特征 concat (空 mask 样本零卷 + 存在标志)")
    ap.add_argument("--lesion-backbone", type=str, default=None,
                    help="round9 F1: 病灶分支 backbone(默认 medicalnet10, 轻量避免显存超)")
    ap.add_argument("--lesion-size", type=int, default=None,
                    help="round9 F1: 病灶质心裁剪边长(默认 64)")
    ap.add_argument("--lesion-head-width", type=int, default=None,
                    help="round9 F1: 病灶嵌入宽度(默认 256)")
    ap.add_argument("--mri-lesion-fusion", action="store_true",
                        help="round9 MRI-t ROI: MRI-t 结肠质心裁剪卷分支 (total_mr mask)")
    ap.add_argument("--mri-lesion-backbone", default="medicalnet10",
                        help="MRI-t ROI 分支骨干 (默认 medicalnet10)")
    ap.add_argument("--mri-lesion-size", type=int, default=64,
                        help="MRI-t ROI 裁剪边长 (默认 64)")
    ap.add_argument("--mri-lesion-head-width", type=int, default=256,
                        help="MRI-t ROI 分支 MLP 宽度 (默认 256)")
    ap.add_argument("--n-lesion-fusion", action="store_true",
                        help="round9 ②: MRI-t 结肠 ROI 嵌入只进 N 头 (n_parts), 不进 T 头 (main_logits) "
                             "——利用 ROI DL 帮 N、融合伤 N 的双向证据, N 专项病灶局部信息")
    ap.add_argument("--n-ring-fusion", action="store_true",
                        help="Day1⑤: ring 版本 N 头嵌入 —— 结肠 mask 膨胀 2 层取环 (肠周/系膜) 质心裁剪 "
                             "只进 N 头 (N 分期信号在系膜区域的病理依据)")
    ap.add_argument("--pcgrad", action="store_true",
                        help="round9 ④: PCGrad 梯度解耦 —— T/N 两任务梯度冲突时投影到彼此法平面, "
                             "缓解共享表征的'帮 T 伤 N / 帮 N 伤 T'冲突 (Yu et al. 2020)")
    ap.add_argument("--no-ct-pretrained", action="store_true",
                    help="round9 SwinUNETR: CT 分支不用 SSL 预训练(仅当 --ct-backbone swin 时生效)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="覆盖 batch size (SwinUNETR 显存大, 8GB 卡建议 4 或 2)")
    ap.add_argument("--class-weight-mode", type=str, default=None,
                    choices=["inverse", "sqrt", "none"],
                    help="覆盖 cfg.class_weight_mode: inverse(满逆频) / sqrt(√逆频) / none(不加权)")
    ap.add_argument("--focal", action="store_true",
                    help="T 主任务改用 Focal Loss（alpha=类别权重）")
    ap.add_argument("--aux-weight", type=float, default=None,
                    help="覆盖 loss_weights.aux")
    ap.add_argument("--rec-weight", type=float, default=None,
                    help="覆盖 loss_weights.rec（重建损失权重，怀疑主导梯度时可调低）")
    ap.add_argument("--head-dropout", type=float, default=None,
                    help="覆盖 cfg.head_dropout")
    ap.add_argument("--no-aug", action="store_true",
                    help="round8: 关闭训练随机增强（对照「增强过强导致 T 学不动」）")
    # ── round8 A3/A4: 模型本体优化 ──
    ap.add_argument("--pool", type=str, default=None, choices=["avg", "gem"],
                    help="覆盖 cfg.pool_mode: avg(全局平均) / gem(GeM 广义均值池化)")
    ap.add_argument("--head-width", type=int, default=None,
                    help="覆盖 cfg.head_width: 主分类头(T/N)隐层宽度, 小样本可减窄至 128")
    ap.add_argument("--gem-p", type=float, default=None,
                    help="覆盖 cfg.gem_p: GeM 幂指数 (pool=gem 时生效)")
    # ── round9 S3: EMA 参数平均 ──
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="round9 S3: EMA 影子权重 decay (0~1, 如 0.999; 默认 None=关闭)。"
                         "每 batch 更新影子权重, 验证/保存 best 用 EMA 权重(小数据抗噪)")
    ap.add_argument("--consist-modal", type=str, default=None, choices=["mri", "ct"],
                    help="round9 C2: 一致性正则模态 (mri/ct), 训练时对该模态卷做第二次增强")
    ap.add_argument("--consist-weight", type=float, default=None,
                    help="round9 C2: 一致性 loss 权重 (默认 0 关闭)")
    # ── round9 S9-5: Snapshot Ensemble (cosine warm restarts 周期快照) ──
    ap.add_argument("--warm-restarts", action="store_true", default=False,
                    help="round9: 纯 cosine warm restarts 不存快照 (省磁盘)")
    ap.add_argument("--md-both", type=float, default=None,
                    help="round9: 覆盖 modality_dropout both 概率 (默认 0.65)")
    ap.add_argument("--snapshot-k", type=int, default=None,
                    help="round9 S9-⑤: 每 k epoch 存一次快照权重 (cosine restarts, 默认 None 关闭)")
    # ── round9 S9-1: T/N 异方差不确定性加权 ──
    ap.add_argument("--uncertainty-weight", action="store_true",
                    help="round9 S9-1: 用 Kendall 异方差不确定性加权替代手工等权"
                         "T/N 损失 (log_var 可学习, 初始权重=1.0 与手工等权一致)")
    # ── round9 S9-③: ECA 通道注意力 ──
    ap.add_argument("--eca", action="store_true",
                    help="round9 S9-③: 融合前对每 stage 特征图加 ECA 通道注意力"
                         "(核 k 随通道数自适应, 轻量; eval 时需同传以匹配结构)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.deterministic:
        HosConfig.deterministic = True  # 影响后续 HosConfig() 实例 (cudnn 确定性开关)

    if args.eval_only:
        # round7 整改: 只读重出报告（不训练、不更新权重）
        eval_existing(seeds, args.epochs, args.kfold, args.kfold_seed,
                      fusion=args.fusion, backbone=args.backbone,
                      no_clinical=args.no_clinical, split_seed=args.split_seed,
                      mixup=args.mixup, label_smoothing=args.label_smoothing,
                      freeze_low=args.freeze_low, share_backbone=args.share_backbone,
                      branch=args.branch, mri_seq=args.mri_seq, ct_seq=args.ct_seq,
                      freeze_all=args.freeze_all, single_modal=args.single_modal,
                      lr=args.lr, class_weight_mode=args.class_weight_mode,
                      focal=args.focal, aux_weight=args.aux_weight,
                      rec_weight=args.rec_weight, head_dropout=args.head_dropout,
                      no_aug=args.no_aug, pool=args.pool, head_width=args.head_width,
                      gem_p=args.gem_p, lr_n=args.lr_n,
                      n_shallow=args.n_shallow, n_head_width=args.n_width,
                      n_dropout=args.n_dropout, n_shallow_idx=args.n_shallow_idx,
                      n_modal=args.n_modal, n_cascade=args.n_cascade,
                      t_modal=args.t_modal,
                      ema_decay=args.ema_decay, ct_backbone=args.ct_backbone,
                      ct_pretrained=not args.no_ct_pretrained,
                      batch_size=args.batch_size,
                        consist_modal=args.consist_modal, consist_weight=args.consist_weight,
                        uncertainty_weight=args.uncertainty_weight, use_eca=args.eca,
                        snapshot_k=args.snapshot_k, warm_restarts=args.warm_restarts, md_both=args.md_both,
                        lesion_fusion=args.lesion_fusion, lesion_backbone=args.lesion_backbone,
                        lesion_size=args.lesion_size, lesion_head_width=args.lesion_head_width,
                        mri_lesion_fusion=args.mri_lesion_fusion,
                        mri_lesion_backbone=args.mri_lesion_backbone,
                        mri_lesion_size=args.mri_lesion_size,
                        mri_lesion_head_width=args.mri_lesion_head_width,
                        n_lesion_fusion=args.n_lesion_fusion, pcgrad=args.pcgrad, n_ring_fusion=args.n_ring_fusion)
        return

    if args.kfold > 1:
        run_kfold(args.epochs, n_splits=args.kfold, kfold_seed=args.kfold_seed,
                  fusion=args.fusion, backbone=args.backbone,
                  no_clinical=args.no_clinical, split_seed=args.split_seed,
                  mixup=args.mixup, label_smoothing=args.label_smoothing,
                  freeze_low=args.freeze_low, share_backbone=args.share_backbone,
                  branch=args.branch, mri_seq=args.mri_seq, ct_seq=args.ct_seq,
                  lr=args.lr, class_weight_mode=args.class_weight_mode,
                  focal=args.focal, aux_weight=args.aux_weight,
                  rec_weight=args.rec_weight, head_dropout=args.head_dropout,
                  no_aug=args.no_aug, pool=args.pool, head_width=args.head_width,
                  gem_p=args.gem_p, freeze_all=args.freeze_all, lr_n=args.lr_n,
                  n_shallow=args.n_shallow, n_head_width=args.n_width,
                  n_dropout=args.n_dropout, n_shallow_idx=args.n_shallow_idx,
                  n_modal=args.n_modal, n_cascade=args.n_cascade,
                  single_modal=args.single_modal,
                  kfold_resume=args.kfold_resume,
                  kfold_max_folds=args.kfold_max_folds,
                  ema_decay=args.ema_decay, ct_backbone=args.ct_backbone,
                  ct_pretrained=not args.no_ct_pretrained, t_modal=args.t_modal,
                  batch_size=args.batch_size,
                        consist_modal=args.consist_modal, consist_weight=args.consist_weight,
                        uncertainty_weight=args.uncertainty_weight, use_eca=args.eca,
                        snapshot_k=args.snapshot_k, warm_restarts=args.warm_restarts, md_both=args.md_both,
                        lesion_fusion=args.lesion_fusion, lesion_backbone=args.lesion_backbone,
                        lesion_size=args.lesion_size, lesion_head_width=args.lesion_head_width,
                        mri_lesion_fusion=args.mri_lesion_fusion,
                        mri_lesion_backbone=args.mri_lesion_backbone,
                        mri_lesion_size=args.mri_lesion_size,
                        mri_lesion_head_width=args.mri_lesion_head_width,
                        n_lesion_fusion=args.n_lesion_fusion, pcgrad=args.pcgrad, n_ring_fusion=args.n_ring_fusion)
        return

    # ── 逐 seed 训练, 每完成一个立即增量落盘 summary（崩溃不丢）──
    summaries = []
    summary_path = None
    for s in seeds:
        sm = run_seed(s, args.epochs, args.val_ratio,
                      fusion=args.fusion, backbone=args.backbone,
                      no_clinical=args.no_clinical, split_seed=args.split_seed,
                      mixup=args.mixup, label_smoothing=args.label_smoothing,
                      freeze_low=args.freeze_low, share_backbone=args.share_backbone,
                      branch=args.branch, mri_seq=args.mri_seq, ct_seq=args.ct_seq,
                      lr=args.lr, class_weight_mode=args.class_weight_mode,
                      focal=args.focal, aux_weight=args.aux_weight,
                      rec_weight=args.rec_weight, head_dropout=args.head_dropout,
                      no_aug=args.no_aug, pool=args.pool, head_width=args.head_width,
                      gem_p=args.gem_p, freeze_all=args.freeze_all, lr_n=args.lr_n,
                      n_shallow=args.n_shallow, n_head_width=args.n_width,
                      n_dropout=args.n_dropout, n_shallow_idx=args.n_shallow_idx,
                      n_modal=args.n_modal, n_cascade=args.n_cascade,
                      t_modal=args.t_modal,
                      single_modal=args.single_modal,
                      ema_decay=args.ema_decay, ct_backbone=args.ct_backbone,
                      ct_pretrained=not args.no_ct_pretrained,
                      batch_size=args.batch_size,
                        consist_modal=args.consist_modal, consist_weight=args.consist_weight,
                        uncertainty_weight=args.uncertainty_weight, use_eca=args.eca,
                        snapshot_k=args.snapshot_k, warm_restarts=args.warm_restarts, md_both=args.md_both,
                        lesion_fusion=args.lesion_fusion, lesion_backbone=args.lesion_backbone,
                        lesion_size=args.lesion_size, lesion_head_width=args.lesion_head_width,
                        mri_lesion_fusion=args.mri_lesion_fusion,
                        mri_lesion_backbone=args.mri_lesion_backbone,
                        mri_lesion_size=args.mri_lesion_size,
                        mri_lesion_head_width=args.mri_lesion_head_width,
                        n_lesion_fusion=args.n_lesion_fusion, pcgrad=args.pcgrad, n_ring_fusion=args.n_ring_fusion)
        summaries.append(sm)
        summary_path = _save_summary(args, summaries)

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

    if summary_path:
        print(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
