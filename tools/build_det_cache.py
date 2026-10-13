# -*- coding: utf-8 -*-
"""构建 Hos 确定性预处理磁盘缓存 (cache_hos/*.npy)。

确定性管线 (LoadImage->Spacing->归一化/裁剪->Resize->NormalizeIntensity) 输出与
运行时完全一致, 缓存后训练/验证直接读 .npy, 跳过每 seed 重复的解压与重采样。

用法:
  python tools/build_det_cache.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# tools/ 子目录运行, 需要把项目根加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import HosConfig
from data_loader.loader import (_ct_deterministic, _mri_seqs_deterministic,
                                _det_cache_key_mri, _DetDiskCache)


def main():
    cfg = HosConfig()
    cache_dir = getattr(cfg, "disk_cache_dir", None) or "./cache_hos"
    disk = _DetDiskCache(cache_dir, cfg.spatial_size)
    ct_det = _ct_deterministic(cfg.spatial_size)

    df = pd.read_csv(cfg.train_csv)
    print(f"样本数: {len(df)}  缓存目录: {cache_dir}", flush=True)

    total = t0 = 0
    t0 = time.time()
    for i, row in df.iterrows():
        # CT: 单卷单路径缓存
        ct_path = row.get("ct_path", "")
        if isinstance(ct_path, str) and ct_path and os.path.exists(ct_path):
            cached, hit = disk.get(ct_path)
            if not hit:
                t = torch.as_tensor(np.ascontiguousarray(ct_det(ct_path)), dtype=torch.float32)
                disk.put(ct_path, t)
                total += 1
        # MRI: 三序列对齐卷缓存（T2 锚定，round5）
        t_path = row.get("mri_path_t", row.get("mri_path", ""))
        c_path = row.get("mri_path_c", "")
        d_path = row.get("mri_path_d", "")
        seq_avail = [
            bool(row.get("mri_available_t", True)) and isinstance(t_path, str) and os.path.exists(t_path),
            bool(row.get("mri_available_c", True)) and isinstance(c_path, str) and os.path.exists(c_path),
            bool(row.get("mri_available_d", True)) and isinstance(d_path, str) and os.path.exists(d_path),
        ]
        if any(seq_avail):
            mri_key = _det_cache_key_mri(t_path, c_path, d_path, cfg.spatial_size)
            cached, hit = disk.get_vol(mri_key)
            if not hit:
                vol = _mri_seqs_deterministic(t_path, c_path, d_path, seq_avail, cfg.spatial_size)
                t = torch.as_tensor(np.ascontiguousarray(vol), dtype=torch.float32)
                disk.put_vol(mri_key, t)
                total += 1
        if total % 20 == 0:
            print(f"[{time.time() - t0:.0f}s] cached {total} volumes", flush=True)
    print(f"done. total cached = {total}  elapsed = {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
