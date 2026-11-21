# -*- coding: utf-8 -*-
"""生成 7 个分支训练子集 csv（round7 分支实验数据准备）。

分支定义（基于 hos_pairs.csv 现有列，只过滤行、保留全部列）:
  mri_t / mri_d / mri_c   : MRI 单序列可用
  mri_tdc                 : MRI t/c/d 三序列齐全
  ct_p                    : CT 平扫可用（ct_avail_plain）
  ct_z                    : CT 增强可用（ct_avail_enhanced）
  full                    : 有任一模态（CT 或 MRI）

注意: label>=0 过滤与 train/val 分层划分由 train_hos_seeds.py 内部完成，
      这里只产出"该分支可用的患者集合"。

用法:
  python tools/make_branch_subsets.py
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "hos_pairs.csv")
OUT_DIR = os.path.join(BASE, "data")

BRANCHES = {
    "mri_t": lambda d: d["mri_available_t"].astype(bool),
    "mri_d": lambda d: d["mri_available_d"].astype(bool),
    "mri_c": lambda d: d["mri_available_c"].astype(bool),
    "mri_tdc": lambda d: (d["mri_available_t"].astype(bool)
                          & d["mri_available_c"].astype(bool)
                          & d["mri_available_d"].astype(bool)),
    "ct_p": lambda d: d["ct_avail_plain"].astype(bool),
    "ct_z": lambda d: d["ct_avail_enhanced"].astype(bool),
    "full": lambda d: (d["ct_available"].astype(bool)
                       | d["mri_available"].astype(bool)),
}


def main():
    df = pd.read_csv(CSV)
    for name, cond in BRANCHES.items():
        sub = df[cond(df)].copy()
        out = os.path.join(OUT_DIR, f"hos_branch_{name}.csv")
        sub.to_csv(out, index=False, encoding="utf-8-sig")
        lab_t = sub[sub["label_T"] >= 0]["label_T"].value_counts().to_dict()
        lab_n = sub[sub["label_N"] >= 0]["label_N"].value_counts().to_dict()
        print(f"{name:<8} n={len(sub):<4} 有标签T={sum(lab_t.values())} "
              f"T分布={lab_t}  N分布={lab_n}")


if __name__ == "__main__":
    main()
