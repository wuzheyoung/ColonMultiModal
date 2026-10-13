"""
按患者分层划分 TCGA CT 数据为 train / val（防数据泄漏）。

关键设计:
  - 划分单位是【患者】，不是 series。同一患者的所有 series 只会进同一侧，
    避免"同病人 series A 进 train、series B 进 val"的泄漏导致指标虚高。
  - train 侧【展开】该患者的所有诊断级 series（来自 tcga_ct_series.csv），
    作为同患者数据增强，扩充训练样本。
  - val 侧每患者只取【主体积】（来自 tcga_ct_pairs.csv），评估干净、按患者。
  - 仅使用有分期 label 的患者；无 label 者写入 tcga_unlabeled.csv 备查。
  - 患者级按 label 分层，保持四类比例；seed 固定以复现。

输出:
  data/tcga_train.csv       (展开 series, 训练用)
  data/tcga_val.csv         (每患者主体积, 验证用)
  data/tcga_unlabeled.csv   (无分期患者, 备查)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

VAL_RATIO = 0.2
SEED = 42
LABEL_COL = "label_bin"   # 划分/训练用的 label 列: "label"(四分类) 或 "label_bin"(二分类)


def _labeled(df):
    m = df[LABEL_COL].notna() & (df[LABEL_COL].astype(str).str.strip() != "")
    out = df[m].copy()
    out[LABEL_COL] = out[LABEL_COL].astype(int)
    return out


def split(base):
    data = os.path.join(base, "data")
    pairs = pd.read_csv(os.path.join(data, "tcga_ct_pairs.csv"))   # 每患者主体积
    series_path = os.path.join(data, "tcga_ct_series.csv")         # 展开诊断级 series
    series = pd.read_csv(series_path) if os.path.isfile(series_path) else pairs.copy()

    labeled = _labeled(pairs)
    unlabeled = pairs[~pairs.index.isin(labeled.index)]

    # 患者级分层划分（每患者一行 -> 一个 label）
    pat_ids = labeled["patient_id"].tolist()
    pat_labels = labeled[LABEL_COL].tolist()
    train_ids, val_ids = train_test_split(
        pat_ids, test_size=VAL_RATIO, random_state=SEED, stratify=pat_labels,
    )
    train_ids, val_ids = set(train_ids), set(val_ids)

    # train: 展开 series 表中属于 train 患者的所有诊断级 series
    series_labeled = _labeled(series)
    train_df = series_labeled[series_labeled["patient_id"].isin(train_ids)].copy()
    # val: 主体积表中属于 val 患者的行（每患者一个）
    val_df = labeled[labeled["patient_id"].isin(val_ids)].copy()

    for name, d in {
        "tcga_train.csv": train_df,
        "tcga_val.csv": val_df,
        "tcga_unlabeled.csv": unlabeled,
    }.items():
        d.to_csv(os.path.join(data, name), index=False, encoding="utf-8-sig")

    print(f"label 列={LABEL_COL}  有 label 患者: {len(labeled)}  无 label: {len(unlabeled)}")
    print(f"train 患者={len(train_ids)} -> 展开 {len(train_df)} series")
    print(f"val   患者={len(val_ids)} -> {len(val_df)} 主体积")
    print("train label 分布(series级):", train_df[LABEL_COL].value_counts().sort_index().to_dict())
    print("val   label 分布(患者级):", val_df[LABEL_COL].value_counts().sort_index().to_dict())
    return train_df, val_df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split(base)
