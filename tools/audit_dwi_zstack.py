# -*- coding: utf-8 -*-
"""
只读数据审计: 核查 mri_d 疑似"双 b 值沿 Z 轴拼接"假设 (用户 m05038 指令)
不修改任何文件, 不跑特征提取。落盘 logs/audit_dwi_zstack.json
"""
import os, json, sys
import numpy as np
import pandas as pd
import nibabel as nib

ROOT = r"D:\PyFile\ColonMultiModal"
PAIRS = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
MR_D = os.path.join(ROOT, "data", "masks_totalseg_mr_d")
MR_C = os.path.join(ROOT, "data", "masks_totalseg_mr_c")
MR_T = os.path.join(ROOT, "data", "masks_totalseg_mr_t")
REMAP = os.path.join(ROOT, "data", "masks_dwi_remapped")
OUT = os.path.join(ROOT, "logs", "audit_dwi_zstack.json")

NCC_TH = 0.65  # 前后半段对应层 NCC 阈值 -> 疑似双周期拼接


def load_vol(path):
    img = nib.load(path)
    arr = img.get_fdata()
    zooms = img.header.get_zooms()
    affine = img.affine
    # nibabel 数组 shape = (i,j,k) = (x,y,z); 轴位 LPS 下 k 是 slice 轴
    origin = affine[:3, 3].tolist()
    # 物理 z 覆盖: 沿数组第 3 轴 (k) 的世界 z 分量
    col3 = affine[:3, 2]  # (dx/dk, dy/dk, dz/dk)
    z_step = col3[2]
    nz = arr.shape[2]
    z_min = origin[2] + 0.0
    z_max = origin[2] + z_step * nz
    # 层间 NCC (数组第3轴相邻层) 辅助
    return {
        "path": path, "shape": list(arr.shape), "spacing": list(zooms),
        "origin": origin, "direction_affine_col3": col3.tolist(),
        "z_step": float(z_step), "z_min": round(float(z_min), 2), "z_max": round(float(z_max), 2),
        "nz": int(nz), "arr": arr,
    }


def ncc_pair(arr, z_a, z_b):
    a = arr[:, :, z_a].astype(np.float32).ravel()
    b = arr[:, :, z_b].astype(np.float32).ravel()
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def zstack_stats(arr):
    """前后半段对应层 NCC 序列 + 是否疑似双周期"""
    nz = arr.shape[2]
    half = nz // 2
    if half < 2:
        return None
    corr = [ncc_pair(arr, z, z + half) for z in range(half)]
    # 对照: 相邻层 (z, z+1) 前半段平均 (应有高相关)
    adj = [ncc_pair(arr, z, z + 1) for z in range(half - 1)]
    return {
        "nz": nz, "half": half,
        "halfpair_ncc": [round(c, 3) for c in corr],
        "halfpair_ncc_mean": round(float(np.mean(corr)), 3),
        "adjacent_ncc_mean_half1": round(float(np.mean(adj)), 3),
        "suspect_double_z": bool(np.mean(corr) > NCC_TH),
    }


def header_summary(pid, pairs):
    row = pairs[pairs.patient_id == pid].iloc[0]
    res = {}
    for seq in ("t", "c", "d"):
        key = f"mri_path_{seq}"
        p = row[key]
        if pd.isna(p) or not os.path.exists(str(p)):
            res[seq] = {"exists": False}
            continue
        v = load_vol(str(p))
        zs = zstack_stats(v["arr"])
        res[seq] = {k: v[k] for k in ("shape", "spacing", "origin", "z_step", "z_min", "z_max", "nz")}
        if zs:
            res[seq]["zstack"] = zs
        del v["arr"]
        v["arr"] = None
        res[seq]["_vol_loaded"] = True
    return res


def main():
    pairs = pd.read_csv(PAIRS)
    out = {"ncc_threshold": NCC_TH, "pids": {}}

    # ---- 1. pid=46 三序列头文件 + from_t 映射非零体素 ----
    h46 = header_summary(46, pairs)
    out["pids"]["46"] = {"headers": h46}
    # from_t 映射
    from_t = os.path.join(REMAP, "46", "colon_from_t.nii.gz")
    from_c = os.path.join(REMAP, "46", "colon_from_c.nii.gz")
    for tag, fp in (("from_t", from_t), ("from_c", from_c)):
        if os.path.exists(fp):
            a = nib.load(fp).get_fdata()
            out["pids"]["46"][tag] = {"exists": True, "nonzero_voxels": int((a > 0).sum()),
                                      "shape": list(a.shape)}
        else:
            out["pids"]["46"][tag] = {"exists": False}

    # ---- 2. pid=46 mri_d 前后半段 NCC + 灰度均值/99分位 ----
    arr_d = load_vol(os.path.join(ROOT, "data", "Hos", "_nifti", "MRI", "46_d.nii.gz"))["arr"]
    half = arr_d.shape[2] // 2
    stats46 = zstack_stats(arr_d)
    lo = arr_d[:, :, :half].astype(np.float32)
    hi = arr_d[:, :, half:].astype(np.float32)
    out["pids"]["46"]["mri_d_hist"] = {
        "zstack": stats46,
        "half1_mean": round(float(lo.mean()), 4), "half2_mean": round(float(hi.mean()), 4),
        "half1_p99": round(float(np.percentile(lo, 99)), 4), "half2_p99": round(float(np.percentile(hi, 99)), 4),
        "half1_p50": round(float(np.median(lo)), 4), "half2_p50": round(float(np.median(hi)), 4),
    }
    del arr_d

    # ---- 3. 全队列 mri_d shape/z跨度分布 + 双周期检测 + 与138全零mask重合 ----
    all_rows = pairs[pd.notna(pairs["mri_path_d"]) & pairs["mri_path_d"].notna()].copy()
    shape_counter = {}
    zspan_list = []
    suspect_ids = []
    n_checked = 0
    maskzero_ids = []
    # 全零 mask: 扫描 MR_D 目录
    for d in sorted(os.listdir(MR_D)):
        mp = os.path.join(MR_D, d, "colon.nii.gz")
        if os.path.exists(mp):
            try:
                if int((nib.load(mp).get_fdata() > 0).sum()) == 0:
                    maskzero_ids.append(int(d))
            except Exception:
                pass
    out["mask_zero_mr_d_count"] = len(maskzero_ids)
    for _, r in all_rows.iterrows():
        p = str(r["mri_path_d"])
        pid = int(r["patient_id"])
        if not os.path.exists(p):
            continue
        try:
            v = load_vol(p)
        except Exception as e:
            out.setdefault("errors", []).append({"pid": pid, "msg": str(e)})
            continue
        n_checked += 1
        shp = tuple(v["shape"])
        shape_counter[str(shp)] = shape_counter.get(str(shp), 0) + 1
        zspan = round(v["z_max"] - v["z_min"], 1)
        zspan_list.append(zspan)
        zs = zstack_stats(v["arr"])
        if zs and zs["suspect_double_z"]:
            suspect_ids.append({"pid": pid, "nz": v["nz"], "halfpair_ncc_mean": zs["halfpair_ncc_mean"],
                                "zspan": zspan, "mask_zero": pid in maskzero_ids})
        v["arr"] = None
    out["mri_d_global"] = {
        "n_scanned": n_checked,
        "shape_dist": shape_counter,
        "zspan_min": round(float(np.min(zspan_list)), 1) if zspan_list else None,
        "zspan_max": round(float(np.max(zspan_list)), 1) if zspan_list else None,
        "zspan_mean": round(float(np.mean(zspan_list)), 1) if zspan_list else None,
        "zspan_hist": sorted({round(x, 1): zspan_list.count(round(x, 1)) for x in zspan_list}.items())[:40],
        "suspect_double_z_count": len(suspect_ids),
        "suspect_double_z_ids": suspect_ids,
    }
    sus_set = {s["pid"] for s in suspect_ids}
    overlap = sorted(sus_set & set(maskzero_ids))
    out["mri_d_global"]["suspect_and_maskzero_overlap"] = {
        "n_overlap": len(overlap), "ids": overlap,
        "n_maskzero_total": len(maskzero_ids),
        "maskzero_pct_among_suspect": round(100 * len(overlap) / len(sus_set), 1) if sus_set else None,
    }

    # ---- 4. pid 51/88/67/212/162: mri_d vs mri_c 物理 z + 双周期 + from_c 后半段 ----
    for pid in (51, 88, 67, 212, 162):
        row = pairs[pairs.patient_id == pid].iloc[0]
        rec = {}
        for seq in ("d", "c"):
            key = f"mri_path_{seq}"
            p = row[key]
            if pd.isna(p) or not os.path.exists(str(p)):
                rec[seq] = {"exists": False}
                continue
            v = load_vol(str(p))
            zs = zstack_stats(v["arr"])
            rec[seq] = {k: v[k] for k in ("shape", "spacing", "origin", "z_step", "z_min", "z_max", "nz")}
            if zs:
                rec[seq]["zstack"] = {"suspect_double_z": zs["suspect_double_z"],
                                      "halfpair_ncc_mean": zs["halfpair_ncc_mean"]}
            v["arr"] = None
        # from_c 映射
        fc = os.path.join(REMAP, str(pid), "colon_from_c.nii.gz")
        if os.path.exists(fc):
            a = nib.load(fc).get_fdata()
            nz_vox = int((a > 0).sum())
            zprof = np.where(a > 0)[2]
            rec["from_c"] = {"exists": True, "nonzero_voxels": nz_vox,
                             "z_min_layer": int(zprof.min()) if zprof.size else None,
                             "z_max_layer": int(zprof.max()) if zprof.size else None,
                             "nz_layers": len(set(zprof.tolist())) if zprof.size else 0}
        else:
            rec["from_c"] = {"exists": False}
        out["pids"][str(pid)] = {"headers": rec}

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, float):
            return round(o, 4) if abs(o) > 1e-6 else o
        return o

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_clean(out), f, ensure_ascii=False, indent=1)
    print(f"DONE -> {OUT}")
    print(json.dumps({k: v for k, v in out.items() if k != "pids"}, ensure_ascii=False, indent=1)[:2000])


if __name__ == "__main__":
    main()
