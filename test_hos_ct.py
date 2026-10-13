"""
用训练好的模型在 data/Hos/CT 外部数据上测试 CT 分支。

对每个病例输出两种 CT 口径的预测:
  - CT-aux : 纯 CT 分支 (ct_aux_head)，不经融合
  - CT-only: 主网络，MRI 置缺失由 imputer 补

stage3 (label 0) 计入准确率；stage2 (label -1, OOD) 仅显示预测不计分。

用法: python test_hos_ct.py
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd

from config import Config
from models import CRCMultiModalNet
from data_loader.loader import CRCMultiModalDataset
from torch.utils.data import DataLoader
from utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix

TEST_CSV = "./data/hos_ct_test.csv"


def build_model(cfg, device):
    model = CRCMultiModalNet(
        backbone_name=cfg.backbone_name, fusion_mode=cfg.fusion_mode,
        num_clinical=cfg.num_clinical, num_classes=cfg.num_classes,
        swin_feature_size=cfg.swin_feature_size, num_fuse_stages=cfg.num_fuse_stages,
        img_size=cfg.spatial_size, use_self_attn=cfg.use_self_attn,
        self_attn_layers=cfg.self_attn_layers, self_attn_heads=cfg.self_attn_heads,
        use_imputer=cfg.use_imputer, imputer_hidden_dim=cfg.imputer_hidden_dim,
    ).to(device)
    ckpt = os.path.join(cfg.save_dir, "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"已加载权重: {ckpt}\n")
    return model


@torch.no_grad()
def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CRCMultiModalDataset(
        TEST_CSV, label_col="label", spatial_size=cfg.spatial_size,
        num_clinical=cfg.num_clinical, is_train=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(cfg, device)
    class_names = cfg.class_names  # ["stage3","stage4"]

    rows = []
    for batch in loader:
        pid = batch["patient_id"][0]
        ct = batch["ct"].to(device)
        mri = batch["mri"].to(device)
        clinical = batch["clinical"].to(device)
        label = int(batch["label"][0])
        ct_avail = batch["ct_available"].to(device).bool()
        B = ct.size(0)

        out = model(ct, mri, clinical, ct_mask=ct_avail,
                    mri_mask=torch.zeros(B, dtype=torch.bool, device=device))

        ct_aux_p = F.softmax(out["ct_aux"], dim=1)[0]
        main_p = F.softmax(out["main"], dim=1)[0]
        ct_aux_pred = int(ct_aux_p.argmax())
        main_pred = int(main_p.argmax())

        rows.append({
            "patient_id": pid,
            "true_label": label,
            "true_stage": "stage2(OOD)" if label == -1 else class_names[label],
            "ct_aux_pred": class_names[ct_aux_pred],
            "ct_aux_prob": f"{ct_aux_p[ct_aux_pred]:.3f}",
            "ct_only_pred": class_names[main_pred],
            "ct_only_prob": f"{main_p[main_pred]:.3f}",
        })

    df = pd.DataFrame(rows)
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    # 仅对 label>=0（stage3）计算准确率
    scored = df[df["true_label"] >= 0].copy()
    if len(scored):
        y_true = [class_names[l] for l in scored["true_label"]]
        for branch in ["ct_aux_pred", "ct_only_pred"]:
            acc = accuracy_score(y_true, scored[branch].tolist())
            print(f"{branch:<14} 准确率 (仅 stage3, n={len(scored)}): {acc:.3f}")

        # CT-only 混淆矩阵（2 类空间）
        idx = {n: i for i, n in enumerate(class_names)}
        yt = [idx[s] for s in y_true]
        yp = [idx[s] for s in scored["ct_only_pred"]]
        cm = confusion_matrix(yt, yp, labels=[0, 1]).tolist()
        path = plot_confusion_matrix(cm, class_names, cfg.plot_dir,
                                     title="Hos CT-only (stage3 only)",
                                     filename="confusion_matrix_hos_ct.png")
        print(f"混淆矩阵已保存: {path}")

    print("\n注: stage2 病例训练时未见过(OOD)，仅显示预测、不计入准确率。样本量极小，结果仅供演示。")


if __name__ == "__main__":
    main()
