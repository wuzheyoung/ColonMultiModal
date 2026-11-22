# -*- coding: utf-8 -*-
"""任务2a: 平面审计 — 核实 t/c/d 切面方向, 列出异常 axcodes pid, 导出 2 例 PIR 三视图 PNG。"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
QC = os.path.join(ROOT, "outputs", "plane_qc")
os.makedirs(QC, exist_ok=True)

raw = pd.read_csv(RAW)

# ── 1. axcodes / spacing 汇总 + 异常件 ──
print("== 平面审计汇总 ==")
anomalies = {}
for seq in ["t", "c", "d"]:
    col = f"mri_path_{seq}"
    rows = raw[raw[col].notna()]
    axc = Counter()
    for p in rows[col]:
        try:
            axc["".join(nib.aff2axcodes(nib.load(p).affine))] += 1
        except Exception:
            pass
    print(f"mri_{seq}: axcodes {dict(axc)}")
    for _, r in rows.iterrows():
        try:
            img = nib.load(r[col])
            ax = "".join(nib.aff2axcodes(img.affine))
            if (seq == "t" and ax == "LPS") or (seq == "c" and ax == "PIR") or (seq == "d" and ax != "LPS"):
                anomalies[f"mri_{seq}"] = (int(r["patient_id"]), ax,
                                           tuple(round(float(x), 2) for x in img.header.get_zooms()[:3]),
                                           tuple(img.shape))
        except Exception:
            pass
print("异常件:", anomalies)

# ── 2. 导出 2 例 PIR 样本(mri_t)三视图 ──
def save_views(pid, seq="t"):
    col = f"mri_path_{seq}"
    p = raw[raw["patient_id"] == pid][col].iloc[0]
    img = nib.load(p)
    arr = img.get_fdata()
    print(f"  pid={pid} mri_{seq}: shape={arr.shape} spacing={img.header.get_zooms()[:3]} axcodes={''.join(nib.aff2axcodes(img.affine))}")
    # 三视图: 中间层各轴
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, axis in zip(axes, (0, 1, 2)):
        sl = [arr.shape[i] // 2 for i in range(3)]
        sl[axis] = slice(None)
        ax.imshow(np.take(arr, arr.shape[axis] // 2, axis=axis).T, cmap="gray", origin="lower")
        ax.set_title(f"axis {axis} (sl={arr.shape[axis]//2})")
        ax.axis("off")
    fig.suptitle(f"pid {pid} mri_{seq} axcodes={''.join(nib.aff2axcodes(img.affine))}")
    out = os.path.join(QC, f"plane_p{pid}_mri{seq}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

pirs = []
for _, r in raw[raw["mri_path_t"].notna()].iterrows():
    try:
        ax = "".join(nib.aff2axcodes(nib.load(r["mri_path_t"]).affine))
        if ax == "PIR":
            pirs.append(int(r["patient_id"]))
    except Exception:
        pass
print(f"\nPIR mri_t 样本数: {len(pirs)}, 选 2 例: {pirs[:2]}")
for pid in pirs[:2]:
    save_views(pid, "t")

# ── 3. 顺带: mri_c/mri_d 的轴位性判定 ──
print("\n== 真轴位判定 ==")
for seq in ["c", "d"]:
    col = f"mri_path_{seq}"
    rows = raw[raw[col].notna()]
    print(f"mri_{seq}: {len(rows)} 例; axcodes LPS(轴位,厚层轴=S/I 头足) 占比 "
          f"{sum(1 for p in rows[col] if ''.join(nib.aff2axcodes(nib.load(p).affine))=='LPS')}/{len(rows)}")
print("\nDONE → outputs/plane_qc/")
