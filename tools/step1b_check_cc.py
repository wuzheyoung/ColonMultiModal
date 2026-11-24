# -*- coding: utf-8 -*-
"""补充必查项 (m05893): pid226 底部对称游离轮廓来源排查。
- 多标签明细: np.unique + 各标签体素数
- 3D 连通域拆解: 总数 + 前5大 (体素, 质心物理mm)
排查病例: pid 226 / 1 / K最大3例。
"""
import os, json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MASK_C = "data/masks_totalseg_mr_c"
FEATS_D2 = "features/slice_feats_v2/dinov2/c"


def centroid_phys(img, mask_bool):
    """连通域体素质心物理坐标 (mm)。"""
    idx = np.argwhere(mask_bool)  # (N,3) in z,y,x
    if len(idx) == 0:
        return None
    pts = []
    for z, y, x in idx[:: max(1, len(idx) // 2000)]:
        pts.append(img.TransformIndexToPhysicalPoint((int(x), int(y), int(z))))
    pts = np.array(pts)
    return [round(float(v), 1) for v in pts.mean(axis=0)]


def main():
    # K 最大 3 例
    ks = {}
    for fn in os.listdir(FEATS_D2):
        if fn.endswith(".npz"):
            pid = int(fn[:-4])
            with np.load(os.path.join(FEATS_D2, fn), mmap_mode="r") as z:
                ks[pid] = int(z["feats"].shape[0])
    top = sorted(ks, key=ks.get, reverse=True)[:3]
    pids = [226, 1] + top
    pids = list(dict.fromkeys(pids))
    print("排查病例:", pids, "| K:", {p: ks.get(p) for p in pids})

    out = {}
    for pid in pids:
        mp = os.path.join(MASK_C, str(pid), "colon.nii.gz")
        rec = {"pid": pid, "K_dinov2": ks.get(pid), "mask_path": mp, "exists": os.path.exists(mp)}
        if not os.path.exists(mp):
            print(f"pid{pid}: mask MISSING")
            out[pid] = rec
            continue
        mimg = sitk.ReadImage(mp)
        arr = sitk.GetArrayFromImage(mimg)
        vals, counts = np.unique(arr, return_counts=True)
        rec["n_voxels"] = int(arr.sum())
        rec["labels"] = {int(v): int(c) for v, c in zip(vals, counts)}
        # 3D 连通域 (6-邻接)
        lab, nlab = ndimage.label(arr > 0, structure=np.ones((3, 3, 3)))
        sizes = ndimage.sum_labels(arr > 0, lab, index=np.arange(1, nlab + 1)).astype(int)
        order = np.argsort(-sizes)
        top5 = []
        for i in order[:5]:
            li = int(i + 1)
            mask_i = lab == li
            top5.append({"cc_id": li, "voxels": int(sizes[i]),
                         "centroid_mm": centroid_phys(mimg, mask_i),
                         "z_span_idx": (int(np.argwhere(mask_i)[:, 0].min()), int(np.argwhere(mask_i)[:, 0].max()))})
        rec["n_cc"] = int(nlab)
        rec["top5_cc"] = top5
        rec["cc_vox_sorted"] = sorted(sizes.tolist(), reverse=True)[:5]
        out[pid] = rec
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    with open("logs/step1b_check_cc.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("DONE → logs/step1b_check_cc.json")


if __name__ == "__main__":
    main()
