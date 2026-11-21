# -*- coding: utf-8 -*-
"""单次训练结果评估报告：全指标计算 + ROC/PR 图 + 文本/JSON 记录。

每次训练（每个 seed / 每折 kfold）生成三份记录：
  - logs/eval_report_{tag}_{run}.txt    人类可读文本（含全部训练参数 + 指标）
  - logs/eval_report_{tag}_{run}.json   结构化指标（AUC/PR/混淆矩阵/校准）
  - plots_hos_{branch}/roc_pr_{tag}_{run}.png  ROC + PR 双面板图

train_hos_seeds.py 在 run_seed / run_kfold 中集成调用。
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


# ──────────────────────────────────────────────
# 1. 训练参数收集（每次实验完整记录）
# ──────────────────────────────────────────────
def collect_params(cfg, freeze_layers=None):
    """从 cfg 提取本次实验的全部训练/模型/数据参数。"""
    g = lambda k, d=None: getattr(cfg, k, d)
    return {
        # 模型结构
        "backbone_name": g("backbone_name"),
        "pretrained_backbone": g("pretrained_backbone"),
        "fusion_mode": g("fusion_mode"),
        "num_fuse_stages": g("num_fuse_stages"),
        "use_self_attn": g("use_self_attn"),
        "self_attn_layers": g("self_attn_layers"),
        "self_attn_heads": g("self_attn_heads"),
        "use_imputer": g("use_imputer"),
        "imputer_hidden_dim": g("imputer_hidden_dim"),
        "mri_share_backbone": g("mri_share_backbone", False),
        "mri_sequences": list(g("mri_sequences", ("t", "c", "d"))),
        "ct_seq": g("ct_seq", "auto"),
        "spatial_size": list(g("spatial_size")),
        "num_clinical": g("num_clinical"),
        # 训练超参
        "epochs": g("epochs"),
        "batch_size": g("batch_size"),
        "lr": g("lr"),
        "weight_decay": g("weight_decay"),
        "optimizer": g("optimizer_name", "AdamW"),
        "scheduler": g("scheduler_name", "CosineAnnealingLR"),
        "use_amp": g("use_amp"),
        "device": str(g("device")),
        # 正则 / 类不平衡
        "mixup": g("mixup", 0.0),
        "mixup_alpha": g("mixup_alpha", 0.2),
        "label_smoothing": g("label_smoothing", 0.0),
        "use_focal_loss": g("use_focal_loss", False),
        "focal_gamma": g("focal_gamma", 2.0),
        "class_weight_mode": g("class_weight_mode", "inverse"),
        "head_dropout": g("head_dropout", 0.5),
        # 多任务
        "loss_weights": dict(g("loss_weights", {"aux": 0.3, "rec": 0.5})),
        "modality_dropout": dict(g("modality_dropout",
                                   {"ct_only": 0.15, "mri_only": 0.15, "both": 0.7})),
        "single_modal": g("single_modal", None),
        "num_tasks": g("num_tasks", 2),
        # 冻结策略
        "freeze_backbone_layers": list(freeze_layers or g("freeze_backbone_layers", None) or []),
        # 评估
        "calibrate_thresholds": g("calibrate_thresholds", False),
    }


# ──────────────────────────────────────────────
# 2. 单任务全指标评估
# ──────────────────────────────────────────────
def eval_task(probs, labels, num_classes, class_names, calibrated=None):
    """计算一个任务(如 T/N)的完整评估指标。

    probs: [N, C] softmax 概率;  labels: [N] 真值
    calibrated: calibrate_binary_threshold()["best"] 的可选结果
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)
    preds = probs.argmax(1)
    classes = list(range(num_classes))

    cm = confusion_matrix(labels, preds, labels=classes).tolist()
    per_p = precision_score(labels, preds, labels=classes, average=None, zero_division=0).tolist()
    per_r = recall_score(labels, preds, labels=classes, average=None, zero_division=0).tolist()
    per_f = f1_score(labels, preds, labels=classes, average=None, zero_division=0).tolist()
    per_class = [{"class": str(class_names[i]), "precision": float(per_p[i]),
                  "recall": float(per_r[i]), "f1": float(per_f[i])}
                 for i in range(num_classes)]

    # AUC(ROC) / AUC(PR): 二分类用正类概率; 多分类用 one-vs-rest 宏平均
    try:
        if num_classes == 2:
            auc_roc = float(roc_auc_score(labels, probs[:, 1]))
            auc_pr = float(average_precision_score(labels, probs[:, 1]))
        else:
            auc_roc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
            auc_pr = float(average_precision_score(labels, probs, average="macro"))
    except ValueError:
        auc_roc = auc_pr = float("nan")

    default = compute_default_metrics(probs, labels, num_classes)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "per_class": per_class,
        "confusion_matrix": cm,
        "n_samples": int(len(labels)),
        "class_dist": {str(class_names[i]): int((labels == i).sum()) for i in classes},
        "default_threshold_0.5": default,
        "calibrated": calibrated,
    }


def compute_default_metrics(probs, labels, num_classes):
    """阈值 0.5 下的指标（不重新扫描）。"""
    preds = (probs[:, 1] >= 0.5).astype(int) if num_classes == 2 else probs.argmax(1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "per_class_recall": [round(float(x), 4) for x in recall_score(
            labels, preds, labels=list(range(num_classes)), average=None, zero_division=0)],
    }


# ──────────────────────────────────────────────
# 3. ROC / PR 曲线图
# ──────────────────────────────────────────────
def plot_roc_pr(tasks, out_dir, filename):
    """tasks: {task名: {"probs":[N,C], "labels":[N], "auc_roc":.., "auc_pr":..}}
    双面板: 左 ROC, 右 PR。"""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, t in tasks.items():
        labels = np.asarray(t["labels"]).astype(int)
        p1 = np.asarray(t["probs"])[:, 1]
        fpr, tpr, _ = roc_curve(labels, p1)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC={t['auc_roc']:.4f})")
        prec, rec, _ = precision_recall_curve(labels, p1)
        axes[1].plot(rec, prec, lw=2, label=f"{name} (AP={t['auc_pr']:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves"); axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend(loc="lower left"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# 4. 组装 + 保存报告
# ──────────────────────────────────────────────
def build_eval_report(cfg, tag, run_label, result, proba, cal_t, cal_n,
                      train_df, val_df, train_csv, val_csv,
                      split_seed=None, freeze_layers=None,
                      ckpt_path=None, curves_path=None, cm_path=None):
    """组装一次训练(seed/fold)的完整报告 dict。proba 为 None 时跳过任务评估。"""
    class_names = list(getattr(cfg, "class_names", None)
                       or [f"class{i}" for i in range(cfg.num_classes)])
    n_class_names = list(getattr(cfg, "n_class_names", None) or class_names)
    num_classes = cfg.num_classes
    report = {
        "meta": {
            "branch": getattr(cfg, "branch", None),
            "run": run_label,
            "tag": tag,
            "split_seed": split_seed,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "params": collect_params(cfg, freeze_layers),
        "data": {
            "train_csv": train_csv,
            "val_csv": val_csv,
            "train_n": len(train_df),
            "val_n": len(val_df),
            "train_label_T": train_df["label_T"].value_counts().to_dict() if train_df is not None else {},
            "val_label_T": val_df["label_T"].value_counts().to_dict() if val_df is not None else {},
        },
        "result": {
            "best_epoch": result["best_epoch"],
            "best_macro_f1": float(result["best_macro_f1"]),
            "best_acc": float(result["best_acc"]),
            "best_epoch_n": int(result.get("best_epoch_n", 0)),
            "best_macro_f1_n": float(result.get("best_macro_f1_n", 0.0)),
        },
        "tasks": {},
        "files": {
            "checkpoint": ckpt_path,
            "training_curves": curves_path,
            "confusion_matrix": cm_path,
        },
    }
    if proba is not None:
        report["tasks"]["T"] = eval_task(proba["probs_t"], proba["labels_t"],
                                         num_classes, class_names, calibrated=cal_t)
        if "probs_n" in proba:
            report["tasks"]["N"] = eval_task(proba["probs_n"], proba["labels_n"],
                                             num_classes, n_class_names, calibrated=cal_n)
    # 供画图用（不写入 JSON）
    report["_curve_data"] = {}
    if proba is not None:
        report["_curve_data"]["T"] = {"probs": proba["probs_t"], "labels": proba["labels_t"],
                                      "auc_roc": report["tasks"]["T"]["auc_roc"],
                                      "auc_pr": report["tasks"]["T"]["auc_pr"]}
        if "probs_n" in proba:
            report["_curve_data"]["N"] = {"probs": proba["probs_n"], "labels": proba["labels_n"],
                                          "auc_roc": report["tasks"]["N"]["auc_roc"],
                                          "auc_pr": report["tasks"]["N"]["auc_pr"]}
    return report


def save_eval_report(report, tag, run_label, logs_dir="./logs"):
    """保存 txt + json + ROC/PR 图，返回路径 dict。"""
    tag_str = tag or "hos"
    branch = report["meta"].get("branch")
    plot_dir = f"./plots_hos_{branch}" if branch else "./plots_hos"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    base = f"eval_report_{tag_str}_{run_label}"
    txt_path = os.path.join(logs_dir, base + ".txt")
    json_path = os.path.join(logs_dir, base + ".json")
    png_path = os.path.join(plot_dir, f"roc_pr_{tag_str}_{run_label}.png")

    # 先画图，再写文本/JSON（files 记录完整路径）
    if report.get("_curve_data"):
        plot_roc_pr(report["_curve_data"], plot_dir, os.path.basename(png_path))
    report["files"]["roc_pr"] = png_path
    report["files"]["report_txt"] = txt_path
    report["files"]["report_json"] = json_path

    write_text_report(report, txt_path)
    report_out = {k: v for k, v in report.items() if k != "_curve_data"}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_out, f, ensure_ascii=False, indent=2)
    return {"txt": txt_path, "json": json_path, "png": png_path}


def _fmt_cal(cal):
    if not cal:
        return "（未开启阈值校准）"
    m = cal["metrics"]
    return (f"阈值={cal['threshold']}  macro_f1={m['macro_f1']:.4f}  "
            f"accuracy={m['accuracy']:.4f}  "
            f"per_class_recall={[round(x, 4) for x in m['per_class_recall']]}")


def write_text_report(report, path):
    """人类可读文本报告：训练参数 + 数据 + 结果 + 逐任务指标。"""
    meta, params, data = report["meta"], report["params"], report["data"]
    lines = []
    sep = "=" * 78
    lines.append(sep)
    lines.append("HOS 分支训练评估报告")
    lines.append(sep)
    lines.append(f"  分支      : {meta['branch'] or '(非分支)'}")
    lines.append(f"  实验标识  : {meta['run']}   tag={meta['tag']}")
    lines.append(f"  split_seed: {meta['split_seed']}")
    lines.append(f"  生成时间  : {meta['time']}")
    lines.append("")
    lines.append(sep)
    lines.append("[1] 训练参数")
    lines.append(sep)
    lines.append(f"  模型: backbone={params['backbone_name']}  pretrained={params['pretrained_backbone']}  "
                 f"fusion={params['fusion_mode']}(stages={params['num_fuse_stages']})")
    lines.append(f"         self_attn={params['use_self_attn']}({params['self_attn_layers']}层/{params['self_attn_heads']}头)  "
                 f"imputer={params['use_imputer']}(dim={params['imputer_hidden_dim']})")
    lines.append(f"  MRI  : sequences={params['mri_sequences']}  share_backbone={params['mri_share_backbone']}")
    lines.append(f"  CT   : ct_seq={params['ct_seq']}  spatial={params['spatial_size']}  num_clinical={params['num_clinical']}")
    lines.append(f"  优化 : epochs={params['epochs']}  batch_size={params['batch_size']}  "
                 f"lr={params['lr']}  weight_decay={params['weight_decay']}  optimizer={params['optimizer']}")
    lines.append(f"  调度 : {params['scheduler']}  amp={params['use_amp']}  device={params['device']}")
    lines.append(f"  正则 : mixup={params['mixup']}(α={params['mixup_alpha']})  label_smoothing={params['label_smoothing']}  "
                 f"focal={params['use_focal_loss']}(γ={params['focal_gamma']})  head_dropout={params['head_dropout']}")
    lines.append(f"  加权 : class_weight={params['class_weight_mode']}")
    lines.append(f"  多任务: loss_weights={params['loss_weights']}  modality_dropout={params['modality_dropout']}")
    lines.append(f"  冻结 : {params['freeze_backbone_layers'] or '(无)'}")
    lines.append("")
    lines.append(sep)
    lines.append("[2] 数据划分")
    lines.append(sep)
    lines.append(f"  train_csv = {data['train_csv']}   n={data['train_n']}  "
                 f"label_T={data['train_label_T']}")
    lines.append(f"  val_csv   = {data['val_csv']}   n={data['val_n']}  "
                 f"label_T={data['val_label_T']}")
    lines.append("")
    lines.append(sep)
    lines.append("[3] 训练结果 (best 模型)")
    lines.append(sep)
    r = report["result"]
    lines.append(f"  T: best_epoch={r['best_epoch']}  macro_f1={r['best_macro_f1']:.4f}  "
                 f"acc={r['best_acc']:.4f}")
    if r["best_epoch_n"]:
        lines.append(f"  N: best_epoch={r['best_epoch_n']}  macro_f1={r['best_macro_f1_n']:.4f}")
    for task_name in ("T", "N"):
        if task_name not in report["tasks"]:
            continue
        t = report["tasks"][task_name]
        lines.append("")
        lines.append(sep)
        lines.append(f"[4.{'0' if task_name == 'T' else '1'}] 任务 {task_name} 评估 "
                     f"({t['n_samples']} 例, 分布 {t['class_dist']})")
        lines.append(sep)
        lines.append(f"  accuracy={t['accuracy']:.4f}  macro_f1={t['macro_f1']:.4f}  "
                     f"macro_precision={t['macro_precision']:.4f}  macro_recall={t['macro_recall']:.4f}")
        lines.append(f"  AUC_ROC={t['auc_roc']:.4f}  AUC_PR(AP)={t['auc_pr']:.4f}  "
                     f"weighted_f1={t['weighted_f1']:.4f}")
        lines.append("  每类: " + "  ".join(
            f"{pc['class']}(P={pc['precision']:.3f}/R={pc['recall']:.3f}/F1={pc['f1']:.3f})"
            for pc in t["per_class"]))
        lines.append(f"  混淆矩阵(行=真, 列=预测): {t['confusion_matrix']}")
        lines.append(f"  阈值0.5: macro_f1={t['default_threshold_0.5']['macro_f1']:.4f}  "
                     f"acc={t['default_threshold_0.5']['accuracy']:.4f}  "
                     f"recall={t['default_threshold_0.5']['per_class_recall']}")
        lines.append(f"  阈值校准: {_fmt_cal(t['calibrated'])}")
    lines.append("")
    lines.append(sep)
    lines.append("[5] 输出文件")
    lines.append(sep)
    for k, v in report["files"].items():
        lines.append(f"  {k:16s}: {v}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
