# -*- coding: utf-8 -*-
"""round8: 核对 cache_hos 是否覆盖 CSV 中所有样本的 CT/MRI 确定性缓存。

用法（Pytorch 环境）:
  D:\\Development\\miniconda\\envs\\Pytorch\\python.exe tools/check_cache_coverage.py
  D:\\Development\\miniconda\\envs\\Pytorch\\python.exe tools/check_cache_coverage.py --csv data/hos_branch_ct_z.csv --ct-seq enhanced

覆盖情况说明:
- 主入口 train_hos_seeds.py / loader 在缓存缺失时会实时计算并落盘（首轮 ~8 分钟）；
- tools/train_2p5d.py 与 tools/train_mil.py 不做实时计算，缓存缺失样本会被跳过并打印计数；
- tools/train_roi_dl.py 同样跳过并计数（--cache-dir 可指定 cache_hos_n4）。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data_loader.loader import _det_cache_key, _det_cache_key_mri

sys.stdout.reconfigure(encoding="utf-8")

SPATIAL = (96, 96, 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/hos_pairs.csv")
    ap.add_argument("--ct-seq", default="auto", choices=["auto", "plain", "enhanced"])
    ap.add_argument("--cache-dir", default="cache_hos")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cache_dir = os.path.abspath(args.cache_dir)
    key_col = {"plain": "ct_path_plain", "enhanced": "ct_path_enhanced",
               "auto": "ct_path"}[args.ct_seq]

    missing_ct, missing_mri = [], []
    need_ct = need_mri = checked = 0
    for _, row in df.iterrows():
        # ── CT ──
        ct_path = row.get(key_col, "")
        if isinstance(ct_path, str) and ct_path and os.path.exists(ct_path):
            need_ct += 1
            k = _det_cache_key(ct_path, SPATIAL)
            if not os.path.exists(os.path.join(cache_dir, f"{k}.npy")):
                missing_ct.append((row.get("patient_id"), ct_path, k))

        # ── MRI 三序列卷 ──
        t_path = row.get("mri_path_t", row.get("mri_path", ""))
        c_path = row.get("mri_path_c", "")
        d_path = row.get("mri_path_d", "")
        seq_avail = [
            bool(row.get("mri_available_t", True)) and isinstance(t_path, str) and os.path.exists(t_path),
            bool(row.get("mri_available_c", True)) and isinstance(c_path, str) and os.path.exists(c_path),
            bool(row.get("mri_available_d", True)) and isinstance(d_path, str) and os.path.exists(d_path),
        ]
        if any(seq_avail):
            need_mri += 1
            mk = _det_cache_key_mri(t_path, c_path, d_path, SPATIAL)
            if not os.path.exists(os.path.join(cache_dir, f"{mk}.npy")):
                missing_mri.append((row.get("patient_id"), [t_path, c_path, d_path], mk))
        checked += 1

    print(f"检查 {args.csv} (ct_seq={args.ct_seq}): 共 {checked} 行")
    print(f"  CT 需缓存 {need_ct} 例, 缺 {len(missing_ct)} 例")
    for pid, p, k in missing_ct[:10]:
        print(f"    missing CT : {pid}  {p}  key={k}")
    print(f"  MRI 需缓存 {need_mri} 例, 缺 {len(missing_mri)} 例")
    for pid, ps, k in missing_mri[:10]:
        print(f"    missing MRI: {pid}  {ps}  key={k}")
    ok = not missing_ct and not missing_mri
    print("结果: " + ("✓ 缓存完整" if ok else
                      f"✗ 缺 {len(missing_ct) + len(missing_mri)} 个缓存（训练时会实时计算并落盘）"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
