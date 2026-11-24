# -*- coding: utf-8 -*-
"""第1步核查: c 序列为什么 K 中位 61 层 (m05840 方案C 第1步)。
读 NIfTI size/spacing/direction + c 掩码三轴物理包围盒, 判断真实解剖切片轴。
"""
import os, sys, json
import numpy as np
import SimpleITK as sitk
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RAW = "data/hos_pairs_raw.csv"
MASK_C = "data/masks_totalseg_mr_c"
FEATS_D2 = "features/slice_feats_v2/dinov2/c"


def arr_axis_dir(direction):
    """direction 3x3 行主序, 返回三个 array 轴的单位方向向量(近似主分量)。"""
    d = np.array(direction).reshape(3, 3)
    axes = []
    labels = {0: "R/L", 1: "A/P", 2: "S/I"}
    for a in range(3):
        col = d[:, a]
        idx = int(np.argmax(np.abs(col)))
        axes.append(f"{labels[idx]}{'+' if col[idx] > 0 else '-'}(axis{a})")
    return axes


def main():
    raw = pd.read_csv(RAW).set_index("patient_id")
    # 1) 找 K 最大与最小的 c 例 (dinov2/c)
    ks = {}
    for fn in os.listdir(FEATS_D2):
        if fn.endswith(".npz"):
            pid = int(fn[:-4])
            with np.load(os.path.join(FEATS_D2, fn), mmap_mode="r") as z:
                ks[pid] = int(z["feats"].shape[0])
    max_pid = max(ks, key=ks.get)
    print(f"c K 分布: n={len(ks)} min={min(ks.values())} max={ks[max_pid]}(pid{max_pid}) p50={int(np.median(list(ks.values())))}")
    # 抽查 pid1 + K最大 (若相同则补 K 第二大的)
    pids = [1, max_pid]
    if max_pid == 1:
        second = sorted(ks, key=ks.get)[-2]
        pids.append(second)
    out = {}
    for pid in pids:
        mp = raw.loc[pid, "mri_path_c"]
        v = sitk.ReadImage(mp)
        va = sitk.GetArrayFromImage(v)
        mimg_p = os.path.join(MASK_C, str(pid), "colon.nii.gz")
        has_mask = os.path.exists(mimg_p)
        rec = {
            "pid": pid, "K_dinov2": ks.get(pid),
            "mri_size_zyx": list(v.GetSize()[::-1]), "mri_spacing_zyx": [round(x, 4) for x in v.GetSpacing()[::-1]],
            "mri_direction_zyx": [round(x, 3) for x in v.GetDirection()],
            "mri_arr_axes": arr_axis_dir(v.GetDirection()),
            "npy_shape": list(va.shape),
        }
        if has_mask:
            m = sitk.ReadImage(mimg_p)
            ma = sitk.GetArrayFromImage(m)
            nz = np.nonzero(ma)
            span_idx = [(int(nz[i].min()), int(nz[i].max())) for i in range(3)]
            sp = m.GetSpacing()
            span_mm = [round((b - a + 1) * sp[i], 1) for i, (a, b) in enumerate(span_idx)]
            rec.update({
                "mask_size_zyx": list(m.GetSize()[::-1]),
                "mask_spacing_zyx": [round(x, 4) for x in m.GetSpacing()[::-1]],
                "mask_axis_span_idx": span_idx,
                "mask_axis_span_mm": span_mm,
                "mask_unique": [int(x) for x in np.unique(ma)],
                "mask_axis_dirs": arr_axis_dir(m.GetDirection()),
            })
        else:
            rec["mask"] = "MISSING"
        out[pid] = rec
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    with open("logs/step1_check_c_slices.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("DONE → logs/step1_check_c_slices.json")


if __name__ == "__main__":
    main()
