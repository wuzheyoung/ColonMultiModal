# -*- coding: utf-8 -*-
"""只读核验 features/slice_feats_v2_60mm/dinov2/{t,d} 419 个 npz 完整性 (m06922 检查1)。
- 文件数与 t=280/d=139 对齐 (t 源=slice_feats_v2/dinov2/t 文件名, d 源=manifest self&has_valid=1)
- keys/shape/dtype/K 分布 (min/P50/max), 0 切片/维度异常/NaN/Inf
- 冒烟 3 文件 (pid329-t, pid1-t, pid1-d) K 与原掩码有效切片数比对, 确认非 --limit 5 截断
结果打印到 stdout + logs/verify_60mm_npz.json
"""
import os, sys, json
import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = r"D:\PyFile\ColonMultiModal"
OUT_ROOT = os.path.join(ROOT, "features", "slice_feats_v2_60mm", "dinov2")
SRC_ROOT = os.path.join(ROOT, "features", "slice_feats_v2", "dinov2")
MANIFEST = os.path.join(ROOT, "features", "dwi_mask_remap_manifest.csv")
MASK_BASE = {s: os.path.join(ROOT, "data", f"masks_totalseg_mr_{s}") for s in ("t", "d")}
RAW = os.path.join(ROOT, "data", "hos_pairs_raw.csv")

EXPECTED_KEYS = {"feats", "slice_idx", "rel_z", "mask_area_ratio", "mask_source"}
EXPECTED_SHAPES = {
    "feats": 2, "slice_idx": 1, "rel_z": 1, "mask_area_ratio": 1, "mask_source": 1,
}
EXPECTED_DTYPES = {
    "feats": "float16", "slice_idx": "int32", "rel_z": "float32",
    "mask_area_ratio": "float32", "mask_source": "S1",  # np.bytes_ 单元素数组
}


def load_mask_self(pid, seq):
    orig_p = os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz")
    if not os.path.exists(orig_p):
        return None
    try:
        a = sitk.GetArrayFromImage(sitk.ReadImage(orig_p))
        if int(a.sum()) >= 50:
            return a
    except Exception:
        return None
    return None


def n_eff_slices(pid, seq, mri_path):
    """复现 extract_slices 的有效切片数: 重采样到 MRI 网格后 z_min..z_max 内含掩码层数。"""
    vimg = sitk.ReadImage(mri_path)
    mimg = sitk.ReadImage(os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz"))
    mres = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
    ma = sitk.GetArrayFromImage(mres)
    nz = np.nonzero(ma)
    if len(nz[0]) == 0:
        return 0
    z_min, z_max = nz[0].min(), nz[0].max()
    cnt = 0
    for z in range(z_min, z_max + 1):
        if ma[z].sum() > 0:
            cnt += 1
    return cnt


def main():
    raw = pd.read_csv(RAW).set_index("patient_id")
    t_pids = sorted(int(x[:-4]) for x in os.listdir(os.path.join(SRC_ROOT, "t")) if x.endswith(".npz"))
    man = pd.read_csv(MANIFEST)
    d_pids = sorted(man[(man["mask_source"] == "self") & (man["has_valid_mask"] == 1)]["pid"].astype(int).tolist())
    print(f"t_pids={len(t_pids)} d_self_pids={len(d_pids)}")

    problems = []
    k_stats = {"t": [], "d": []}
    for seq, pids in (("t", t_pids), ("d", d_pids)):
        d_ = os.path.join(OUT_ROOT, seq)
        have = sorted(int(x[:-4]) for x in os.listdir(d_) if x.endswith(".npz"))
        # 对齐
        only_have = sorted(set(have) - set(pids))
        only_want = sorted(set(pids) - set(have))
        print(f"[{seq}] files_on_disk={len(have)} expected={len(pids)} "
              f"extra={len(only_have)} missing={len(only_want)}")
        if only_have:
            print("  EXTRA:", only_have[:20])
            problems.append(f"{seq}: extra files {only_have[:20]}")
        if only_want:
            print("  MISSING:", only_want[:20])
            problems.append(f"{seq}: missing {only_want[:20]}")
        for pid in pids:
            p = os.path.join(d_, f"{pid}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            keys = set(z.files)
            if keys != EXPECTED_KEYS:
                problems.append(f"{seq}/{pid}: keys={sorted(keys)}")
                continue
            k = int(z["feats"].shape[0])
            k_stats[seq].append(k)
            if k == 0:
                problems.append(f"{seq}/{pid}: K=0")
            if z["feats"].shape[1] != 768:
                problems.append(f"{seq}/{pid}: feats shape {z['feats'].shape}")
            if z["feats"].dtype != np.float16:
                problems.append(f"{seq}/{pid}: feats dtype {z['feats'].dtype}")
            if len(z["slice_idx"]) != k or len(z["rel_z"]) != k or len(z["mask_area_ratio"]) != k:
                problems.append(f"{seq}/{pid}: length mismatch K={k} "
                                f"idx={len(z['slice_idx'])} relz={len(z['rel_z'])} area={len(z['mask_area_ratio'])}")
            if not np.isfinite(z["feats"]).all():
                problems.append(f"{seq}/{pid}: feats has NaN/Inf")
            if not np.isfinite(z["rel_z"]).all():
                problems.append(f"{seq}/{pid}: rel_z has NaN/Inf")
            if not np.isfinite(z["mask_area_ratio"]).all():
                problems.append(f"{seq}/{pid}: mask_area_ratio has NaN/Inf")
            ms = z["mask_source"]
            msv = ms[0].decode() if isinstance(ms[0], (bytes, np.bytes_)) else str(ms[0])
            if ms.size != 1 or msv not in ("self", "missing"):
                problems.append(f"{seq}/{pid}: mask_source={ms!r}")
    for seq in ("t", "d"):
        ks = np.array(k_stats[seq], dtype=float)
        if len(ks):
            print(f"[{seq}] K: n={len(ks)} min={int(ks.min())} P50={int(np.median(ks))} "
                  f"max={int(ks.max())} mean={ks.mean():.1f}")

    # 冒烟 3 文件切片数比对 (pid329-t, pid1-t, pid1-d)
    print("\n-- smoke 3 files K vs original effective slices --")
    for seq, pid in (("t", 329), ("t", 1), ("d", 1)):
        p = os.path.join(OUT_ROOT, seq, f"{pid}.npz")
        if not os.path.exists(p):
            print(f"  {seq}/{pid}: NPZ MISSING on disk")
            continue
        z = np.load(p)
        k = int(z["feats"].shape[0])
        col = {"t": "mri_path_t", "d": "mri_path_d"}[seq]
        mp = raw.loc[pid, col]
        if isinstance(mp, str) and os.path.exists(mp):
            neff = n_eff_slices(pid, seq, mp)
            flag = "OK" if k == neff else "MISMATCH"
            print(f"  {seq}/{pid}: K={k} original_eff={neff} {flag}")
            if k != neff:
                problems.append(f"smoke {seq}/{pid}: K={k} != eff={neff}")
        else:
            print(f"  {seq}/{pid}: K={k} (mri_path missing)")

    out = {
        "t_pids": len(t_pids), "d_pids": len(d_pids),
        "k_dist": {s: {"n": len(v), "min": int(np.min(v)), "P50": int(np.median(v)),
                        "max": int(np.max(v))} for s, v in k_stats.items() if len(v)},
        "problems": problems,
        "n_problems": len(problems),
    }
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "verify_60mm_npz.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nproblems={len(problems)}")
    for pr in problems[:50]:
        print("  !", pr)
    print("DONE -> logs/verify_60mm_npz.json")


if __name__ == "__main__":
    main()
