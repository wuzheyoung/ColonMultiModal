# -*- coding: utf-8 -*-
"""1650 同步前只读核验 (m05728): 磁盘体积/张量契约 + t 缺失16例 + 覆盖率/质心偏移"""
import numpy as np, pandas as pd, glob, os, nibabel as nib
from collections import Counter

FEAT = "features/slice_feats_v2"
MASK_BASE = {s: f"data/masks_totalseg_mr_{s}" for s in ("t", "c", "d")}
MASK_REMAP = "data/masks_dwi_remapped"

def npz_exists(bb, seq, pid):
    return os.path.exists(os.path.join(FEAT, bb, seq, f"{pid}.npz"))

def npz_pids(seq, bb="dinov2"):
    return {int(os.path.basename(p)[:-4]) for p in glob.glob(f"{FEAT}/{bb}/{seq}/*.npz")}

# ---------- Part 1: 磁盘体积 + 张量契约 ----------
print("=" * 60)
print("[Part 1] 磁盘体积与张量契约")
total = 0; n = 0
for p in glob.glob(f"{FEAT}/*/*/*.npz"):
    total += os.path.getsize(p); n += 1
print(f"总大小: {total/1e6:.1f} MB | 文件数: {n} | 平均: {total/n/1e3:.2f} KB | 全量估算: {total/1e9:.2f} GB")
s = sorted(glob.glob(f"{FEAT}/*/*/*.npz"))[0]
d = np.load(s)
print(f"示例: {s}")
for k in d.files:
    print(f"  {k}: shape={d[k].shape} dtype={d[k].dtype}")

# ---------- Part 2: t 序列缺失 16 例 ----------
print("=" * 60)
print("[Part 2] t 序列缺失 16 例过滤原因")
raw = pd.read_csv("data/hos_pairs_raw.csv")
t_npz = npz_pids("t")
miss = []
for _, r in raw.iterrows():
    pid = int(r["patient_id"])
    if isinstance(r["mri_path_t"], str) and pid not in t_npz:
        miss.append((pid, r["mri_path_t"]))
print(f"raw 中 mri_path_t 为 str 但无 t npz: {len(miss)} 例")
for pid, tp in miss:
    fe = os.path.exists(tp) if isinstance(tp, str) else False
    mp = os.path.join(MASK_BASE["t"], str(pid), "colon.nii.gz")
    mfe = os.path.exists(mp)
    v = -1
    if mfe:
        try:
            v = int((nib.load(mp).get_fdata() > 0).sum())
        except Exception:
            v = -2
    print(f"  pid{pid}: mri_path_t存在={fe} mask文件存在={mfe} mask体素={v}")

# ---------- Part 3: 核心评估集覆盖率 + 质心偏移 ----------
print("=" * 60)
print("[Part 3] direct 队列覆盖率与质心偏移")
cohort = pd.read_csv("data/direct_cohort_ids.csv")
direct = set(cohort.loc[cohort["is_tgr"] == 0, "pid"].astype(int))
ncrt = set(cohort.loc[cohort["is_tgr"] == 1, "pid"].astype(int))
print(f"direct_all={len(direct)} ncrt={len(ncrt)}")
# direct_tcd = direct 且 t/c/d 影像路径都非空
rawi = raw.set_index("patient_id")
def has_img(pid, s):
    try:
        v = rawi.loc[pid, f"mri_path_{s}"]
        return isinstance(v, str) and len(str(v)) > 0
    except Exception:
        return False
direct_tcd = {p for p in direct if all(has_img(p, s) for s in "tcd")}
print(f"direct_tcd (影像三模态都有) = {len(direct_tcd)}")
for s in "tcd":
    fs = npz_pids(s)
    print(f"  {s}: npz 覆盖 direct_all {len(fs & direct)}/{len(direct)} | direct_tcd {len(fs & direct_tcd)}/{len(direct_tcd)}")
# d self vs remapped (仅 direct_all 内)
man = pd.read_csv("features/dwi_mask_remap_manifest.csv").set_index("pid")
d_npz = npz_pids("d")
ddir = d_npz & direct
msrc = Counter(man.loc[list(ddir), "mask_source"] if len(ddir) else [])
print(f"  d 有效 npz direct_all: {len(ddir)} 例, mask_source 分布: {dict(msrc)}")
# 质心偏移: mask 物理质心 vs DWI 卷中心 (mm)
offs = {}
for pid in sorted(ddir):
    ms = man.loc[pid, "mask_source"]
    grp = "remapped" if ms != "self" else "self"
    mp = os.path.join(MASK_REMAP if grp == "remapped" else MASK_BASE["d"], str(pid), "colon.nii.gz")
    if not os.path.exists(mp):
        continue
    try:
        mimg = nib.load(mp); md = mimg.get_fdata()
        nz = np.argwhere(md > 0)
        if len(nz) == 0:
            continue
        cen = np.mean(nz, axis=0)  # (x,y,z)
        dp = rawi.loc[pid, "mri_path_d"]
        if not isinstance(dp, str) or not os.path.exists(dp):
            continue
        vimg = nib.load(dp); vc = (np.asarray(vimg.shape) - 1) / 2.0
        aff = mimg.affine
        pc = aff @ np.r_[cen, 1.0]; pv = aff @ np.r_[vc, 1.0]
        offs.setdefault(grp, []).append((float(np.linalg.norm(pc[:3] - pv[:3])), pid, ms))
    except Exception as e:
        print(f"  pid{pid} 质心计算失败: {e}")
for k in ("self", "remapped"):
    v = np.array([x[0] for x in offs.get(k, [])])
    if len(v):
        i = int(np.argmax(v))
        print(f"  d/{k}: n={len(v)} 质心偏移(相对DWI卷中心) 中位数={np.median(v):.1f}mm 最大={v[i]:.1f}mm (pid={offs[k][i][1]}, src={offs[k][i][2]})")
    else:
        print(f"  d/{k}: 无数据")
allv = [x[0] for x in offs.get("self", []) + offs.get("remapped", [])]
print(f"  全部 d 例: n={len(allv)} 质心偏移中位数={np.median(allv):.1f}mm 最大={max(allv):.1f}mm" if allv else "  N/A")
