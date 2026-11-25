# -*- coding: utf-8 -*-
"""方案B 双全量完整性校验 (m07012 步骤4): 120mm_clean 与 tight60_m10 两套各 419 npz。

检查项:
1. 文件对齐: 每套 t=280/d=139, extra/missing=0;
2. NaN/Inf=0, 每文件 isfinite 全通过;
3. 切片数与旧版 slice_feats_v2 (120mm 基线) 逐例严格一致 (t/d 分别比对);
4. 新增字段完整性: crop_size_mm (K,) 有值, mask_area_mm2 (K,) >0, mask_area_ratio = mask_area_mm2/14400 恒等;
5. crop_size_mm 分位数统计 (两模式各自: 120mm_clean 应全=120, tight60_m10 应 >=60 且 >120 有比例);
6. K 分布 (min/p50/p95/max) 对比旧版。
输出: logs/verify_clean_npz.json
"""
import os, json
import numpy as np
import pandas as pd

ROOT = "D:\\PyFile\\ColonMultiModal"
os.chdir(ROOT)
MODES = {
    "120mm_clean": "features/slice_feats_v2_120mm_clean",
    "tight60_m10": "features/slice_feats_v2_tight60_m10",
}
OLD = "features/slice_feats_v2"  # 120mm 基线 (只读)


def stats_k(seq):
    ks = []
    for f in os.listdir(os.path.join(OLD, "dinov2", seq)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(OLD, "dinov2", seq, f))
            ks.append(int(z["slice_idx"].shape[0]))
    return np.array(ks, dtype=np.int64)


def main():
    out = {}
    man = pd.read_csv("features/dwi_mask_remap_manifest.csv")
    ref_pids = {
        "t": sorted(int(x[:-4]) for x in os.listdir(os.path.join(OLD, "dinov2", "t"))
                    if x.endswith(".npz")),
        "d": sorted(int(x) for x in man[(man["mask_source"] == "self")
                                        & (man["has_valid_mask"] == 1)]["pid"]),
    }
    old_k = {s: {} for s in ("t", "d")}
    for pid in ref_pids["t"]:
        old_k["t"][pid] = int(np.load(os.path.join(OLD, "dinov2", "t", f"{pid}.npz"))["slice_idx"].shape[0])
    for pid in ref_pids["d"]:
        old_k["d"][pid] = int(np.load(os.path.join(OLD, "dinov2", "d", f"{pid}.npz"))["slice_idx"].shape[0])
    for mode, root in MODES.items():
        d = {"t": {}, "d": {}}
        for seq in ("t", "d"):
            pids = ref_pids[seq]
            cur_pids = sorted(int(x[:-4]) for x in os.listdir(os.path.join(root, "dinov2", seq))
                              if x.endswith(".npz"))
            extra = sorted(set(cur_pids) - set(pids))
            missing = sorted(set(pids) - set(cur_pids))
            assert not extra and not missing, (seq, extra[:5], missing[:5])
            ks_cur, ks_old, nan_cnt, ratio_bad, mm2_bad, crop_bad, crop_vals = [], [], 0, 0, 0, 0, []
            for pid in pids:
                zc = np.load(os.path.join(root, "dinov2", seq, f"{pid}.npz"))
                zo = np.load(os.path.join(OLD, "dinov2", seq, f"{pid}.npz"))
                assert set(zc.files) == set(zo.files) | {"crop_size_mm", "mask_area_mm2"}, (mode, seq, pid, zc.files)
                assert np.isfinite(zc["feats"]).all(), (mode, seq, pid, "feats nonfinite")
                kc = int(zc["slice_idx"].shape[0])
                ko = int(zo["slice_idx"].shape[0])
                ks_cur.append(kc); ks_old.append(ko)
                assert kc == ko, (mode, seq, pid, kc, ko)
                assert (zc["slice_idx"] == zo["slice_idx"]).all(), (mode, seq, pid, "slice_idx mismatch")
                assert (zc["rel_z"] == zo["rel_z"]).all(), (mode, seq, pid, "rel_z mismatch")
                a = zc["mask_area_ratio"].astype(np.float64)
                m2 = zc["mask_area_mm2"].astype(np.float64)
                assert np.allclose(a * 14400.0, m2, rtol=1e-6), (mode, seq, pid, "ratio!=mm2/14400")
                assert (m2 > 0).all(), (mode, seq, pid, "mm2<=0")
                c = zc["crop_size_mm"].astype(np.float32)
                assert np.isfinite(c).all() and (c > 0).all(), (mode, seq, pid, "crop bad")
                crop_vals.extend(c.tolist())
                if np.isnan(zc["feats"]).any():
                    nan_cnt += 1
                if not np.allclose(a * 14400.0, m2, rtol=1e-6):
                    ratio_bad += 1
                if not (m2 > 0).all():
                    mm2_bad += 1
                if not (c > 0).all():
                    crop_bad += 1
            cv = np.array(crop_vals, dtype=np.float32)
            d[seq] = {
                "n": len(pids),
                "k_match_old": int((np.array(ks_cur) == np.array(ks_old)).sum()),
                "K_cur": {"min": int(np.min(ks_cur)), "p50": int(np.median(ks_cur)),
                          "p95": int(np.percentile(ks_cur, 95)), "max": int(np.max(ks_cur))},
                "nan_files": nan_cnt,
                "ratio_bad": ratio_bad, "mm2_bad": mm2_bad, "crop_bad": crop_bad,
                "crop_size_mm": {"min": float(cv.min()), "p50": float(np.median(cv)),
                                 "p90": float(np.percentile(cv, 90)), "max": float(cv.max())},
            }
        out[mode] = d
    with open("logs/verify_clean_npz.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("VERIFY_OK")


if __name__ == "__main__":
    main()
