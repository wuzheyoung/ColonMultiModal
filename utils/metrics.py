import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def compute_metrics(preds, labels, num_classes):
    """
    preds, labels: 1D tensor 或 numpy array
    返回: dict 含 accuracy, macro_f1, per_class_recall, confusion_matrix
    """
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    return {
        "accuracy":         accuracy_score(labels, preds),
        "macro_f1":         f1_score(labels, preds, average="macro", zero_division=0),
        "per_class_recall": recall_score(labels, preds, average=None,
                                         labels=list(range(num_classes)), zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, preds,
                                             labels=list(range(num_classes))).tolist(),
    }


def calibrate_binary_threshold(probs, labels, num_classes=2, grid=None, metric="macro_f1"):
    """二分类概率阈值校准。

    probs : [N, num_classes] 的 softmax 概率（或 [N] 的 class-1 概率）
    labels: [N] 0/1 真值
    grid  : 扫描的阈值网格，默认 np.arange(0.05, 1.0, 0.05)
    返回: dict
        best   : {threshold, metrics} 在给定 metric 上最优的阈值
        default: 阈值 0.5 下的指标
        curve  : 各阈值下的指标列表
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim == 2:
        p1 = probs[:, 1]
    else:
        p1 = probs
    grid = grid if grid is not None else np.arange(0.05, 1.0, 0.05)

    best = None
    curve = []
    for t in grid:
        preds = (p1 >= t).astype(int)
        m = compute_metrics(preds, labels, num_classes)
        curve.append({"threshold": round(float(t), 3), **m})
        if best is None or m[metric] > best["metrics"][metric]:
            best = {"threshold": round(float(t), 3), "metrics": m}

    default = compute_metrics((p1 >= 0.5).astype(int), labels, num_classes)
    return {"best": best, "default": default, "curve": curve}

