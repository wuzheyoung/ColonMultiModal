"""
按患者分层生成 K 折交叉验证的 train/val csv（防数据泄漏）。

设计（与 split_tcga.py 一致的防泄漏原则）:
  - 划分单位是【患者】。StratifiedKFold 按患者的 label 分层，保证每折类别比例接近。
  - 同一患者的所有 series 只会出现在某一折的 train 或 val，绝不跨折泄漏。
  - 每折 train 侧【展开】该折训练患者的所有诊断级 series（tcga_ct_series.csv）；
    val 侧每个验证患者只取【主体积】（tcga_ct_pairs.csv），评估干净、按患者。
  - 仅使用有 label 的患者（由 LABEL_COL 指定，默认 label_bin 二分类）。

输出:
  data/kfold/fold{k}_train.csv
  data/kfold/fold{k}_val.csv
  (k = 0..K-1)
"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
SEED = 42
LABEL_COL = "label_bin"   # "label"(四分类) 或 "label_bin"(二分类)


def _labeled(df):
    m = df[LABEL_COL].notna() & (df[LABEL_COL].astype(str).str.strip() != "")
    out = df[m].copy()
    out[LABEL_COL] = out[LABEL_COL].astype(int)
    return out


def make(base, n_splits=N_SPLITS):
    data = os.path.join(base, "data")
    pairs = pd.read_csv(os.path.join(data, "tcga_ct_pairs.csv"))
    series_path = os.path.join(data, "tcga_ct_series.csv")
    series = pd.read_csv(series_path) if os.path.isfile(series_path) else pairs.copy()

    pairs_l = _labeled(pairs)
    series_l = _labeled(series)

    pids = pairs_l["patient_id"].to_numpy()
    labels = pairs_l[LABEL_COL].to_numpy()

    out_dir = os.path.join(data, "kfold")
    os.makedirs(out_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    print(f"label 列={LABEL_COL}  有 label 患者={len(pids)}  折数={n_splits}")
    for k, (tr_idx, va_idx) in enumerate(skf.split(pids, labels)):
        train_ids = set(pids[tr_idx])
        val_ids = set(pids[va_idx])

        train_df = series_l[series_l["patient_id"].isin(train_ids)].copy()
        val_df = pairs_l[pairs_l["patient_id"].isin(val_ids)].copy()

        train_df.to_csv(os.path.join(out_dir, f"fold{k}_train.csv"),
                        index=False, encoding="utf-8-sig")
        val_df.to_csv(os.path.join(out_dir, f"fold{k}_val.csv"),
                      index=False, encoding="utf-8-sig")
        print(f"[fold {k}] train 患者={len(train_ids)}->{len(train_df)}series  "
              f"val 患者={len(val_ids)}  "
              f"val分布={val_df[LABEL_COL].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    make(base)
