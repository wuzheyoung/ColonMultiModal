"""
构建 data/Hos/CT 的 CT-only 测试集。

每个病例目录形如  <序号>_<病历号>/ 下含一个 series 的 .dcm 文件；
诊断（AJCC 分期）由用户根据 <序号>-BL.txt 判定后在下方 STAGE_MAP 给出。

标签空间说明:
  训练模型是 stage3/stage4 两类 (label 0=stage3, 1=stage4)。
  本测试集为 4×stage3 + 1×stage2。
    - stage3 -> label 0，模型可正常评估
    - stage2 -> 训练时未见过 (OOD)，标 label=-1，推理照跑但不计入准确率

输出: data/hos_ct_test.csv
    列: patient_id, ct_path, mri_path, ct_available, mri_available, stage, label
"""

import os
import glob
import pandas as pd

HOS_CT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "Hos", "CT")

# 序号 -> (影像文件夹名, AJCC 分期)   3 号肠镜活检无分期，已剔除
CASES = {
    "1": ("1_QG12455983", 3),
    "2": ("2_N01022751", 3),
    "4": ("4_QG12425897", 3),
    "5": ("5_QG12447866", 2),
    "6": ("6_QG12446259", 3),
}

# 训练标签映射；OOD（未训练过的分期）标 -1
STAGE_TO_LABEL = {3: 0, 4: 1}


def build_csv(out_path):
    rows = []
    for idx, (folder, stage) in CASES.items():
        ct_dir = os.path.join(HOS_CT_DIR, folder)
        n_dcm = len(glob.glob(os.path.join(ct_dir, "*.dcm")))
        rows.append({
            "patient_id": folder,
            "ct_path": ct_dir,
            "mri_path": "",
            "ct_available": True,
            "mri_available": False,
            "stage": stage,
            "label": STAGE_TO_LABEL.get(stage, -1),
            "n_ct_slices": n_dcm,
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已生成: {out_path}  ({len(df)} 例)")
    print(df[["patient_id", "stage", "label", "ct_available", "n_ct_slices"]].to_string(index=False))
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_csv(os.path.join(base, "data", "hos_ct_test.csv"))
