# -*- coding: utf-8 -*-
"""为 hos_pairs.csv 扩充 CT 平扫/增强双序列列（分支训练数据准备）。

背景: 原 csv 每行只存一个 ct_path（pick_series 优先 z > p > 特殊名），
      95 例平扫&增强双齐全患者的另一个序列被弃选，无法支撑
      "CT 平扫分支 / CT 增强分支" 训练。

本脚本只新增 4 列，不改动现有 361 行其余任何列:
  - ct_path_plain    : 平扫 CT 路径（{num}_p；第一批若缺则用特殊名如 {num}_QG* = 平扫）
  - ct_path_enhanced : 增强 CT 路径（{num}_z）
  - ct_avail_plain   : 平扫是否可用
  - ct_avail_enhanced: 增强是否可用

编号映射（与 build_hos_batch2_csv.py / convert_batch2_to_nifti.py 一致）:
  patient_id <= 200 : 第一批, 目录 _nifti/CT,     num = patient_id
  patient_id >  200 : 第二批, 目录 _nifti/batch2, num = patient_id - 200

用法:
  python utils/expand_ct_series_csv.py
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT_DIR = os.path.join(BASE, "data", "Hos", "_nifti", "CT")
B2_DIR = os.path.join(BASE, "data", "Hos", "_nifti", "batch2")
CSV = os.path.join(BASE, "data", "hos_pairs.csv")


def _dir_for(patient_id):
    if patient_id <= 200:
        return CT_DIR, patient_id
    return B2_DIR, patient_id - 200


def _find_special_plain(ct_dir, num):
    """第一批特殊名（如 1_QG12455983 / 2_N01022751）视为平扫 CT。"""
    prefix = f"{num}_"
    for name in sorted(os.listdir(ct_dir)):
        if not name.endswith(".nii.gz"):
            continue
        base = name[:-len(".nii.gz")]
        if base.startswith(prefix) and not base.endswith(("_p", "_z")):
            return os.path.join(ct_dir, name)
    return ""


def main():
    df = pd.read_csv(CSV)
    rows_plain = rows_enh = both = 0
    for i, r in df.iterrows():
        pid = int(r["patient_id"])
        d, num = _dir_for(pid)

        p_path = os.path.join(d, f"{num}_p.nii.gz")
        if not os.path.exists(p_path):
            p_path = _find_special_plain(CT_DIR, num) if d == CT_DIR else ""

        z_path = os.path.join(d, f"{num}_z.nii.gz")
        if not os.path.exists(z_path):
            z_path = ""

        df.at[i, "ct_path_plain"] = p_path
        df.at[i, "ct_path_enhanced"] = z_path
        df.at[i, "ct_avail_plain"] = bool(p_path)
        df.at[i, "ct_avail_enhanced"] = bool(z_path)
        rows_plain += bool(p_path)
        rows_enh += bool(z_path)
        both += bool(p_path) and bool(z_path)

    df.to_csv(CSV, index=False, encoding="utf-8-sig")
    print("=" * 60)
    print(f"已更新: {CSV} ({len(df)} 行)")
    print(f"平扫可用: {rows_plain}  增强可用: {rows_enh}  双齐全: {both}")
    print("label_T 分布不变:", df["label_T"].value_counts().to_dict())


if __name__ == "__main__":
    main()
