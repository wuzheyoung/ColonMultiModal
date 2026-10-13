"""训练过程与结果可视化：loss/acc 曲线、混淆矩阵。"""

import os
import matplotlib
matplotlib.use("Agg")  # 无界面后端，服务器/脚本环境安全
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(history, out_dir, tag=""):
    """
    history: dict，含以下等长列表
        epoch, train_loss, val_loss, train_acc, val_acc,
        train_main, train_aux, train_rec
    输出: training_curves[_tag].png
    """
    os.makedirs(out_dir, exist_ok=True)
    ep = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) 总 loss
    axes[0].plot(ep, history["train_loss"], "-o", label="Train", color="#1f77b4")
    axes[0].plot(ep, history["val_loss"], "-s", label="Val", color="#ff7f0e")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # (2) 准确率
    axes[1].plot(ep, history["train_acc"], "-o", label="Train", color="#1f77b4")
    axes[1].plot(ep, history["val_acc"], "-s", label="Val", color="#ff7f0e")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # (3) loss 分项
    axes[2].plot(ep, history["train_main"], "-o", label="main (cls)", color="#2ca02c")
    axes[2].plot(ep, history["train_aux"], "-^", label="aux", color="#d62728")
    axes[2].plot(ep, history["train_rec"], "-v", label="rec (impute)", color="#9467bd")
    axes[2].set_title("Loss Components (Train)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fname = f"training_curves_{tag}.png" if tag else "training_curves.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, class_names, out_dir, title="Confusion Matrix",
                          filename=None, tag=""):
    """cm: 2D list/array. 输出到 out_dir/filename（未指定则用 confusion_matrix[_tag].png）"""
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"confusion_matrix_{tag}.png" if tag else "confusion_matrix.png"
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
