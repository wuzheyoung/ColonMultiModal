"""⑤ N4 偏置场校正缓存预生成 (Day1 梯队-1, 2026-11-20)。

目的: 对全量 MRI 样本做 N4 偏置场校正 (SimpleITK N4BiasFieldCorrectionImageFilter),
      再走与 loader 完全相同的 1mm 锚定网格/统一 bbox/zscore 管线, 输出到独立缓存目录
      cache_hos_n4/, 供 train_roi_dl.py --cache-dir cache_hos_n4 使用。

关键: 缓存文件名直接复用 loader._det_cache_key_mri, 否则 DataROI 永远读不到本目录产物;
      参考网格直接复用 loader._mri_reference_grid (1mm 各向同性/继承原图 origin+direction),
      不能自己造 origin=0/spacing=1 的空网格。

用法: python tools/build_n4_cache.py [--csv data/hos_full_train_s42.csv] [--shrink 2] [--out-dir cache_hos_n4]
已存在的 key 跳过 (可断点续跑)。
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_loader"))

import SimpleITK as sitk  # noqa: E402
from monai.transforms import Resize  # noqa: E402

from loader import (_det_cache_key_mri, _mri_bbox, _mri_reference_grid,  # noqa: E402
                    _resample_to_ref, _zscore_nonzero)

SPATIAL = (96, 96, 96)
_SEQ_ORDER = ("t", "c", "d")

CACHE_DIR = os.path.join(ROOT, "cache_hos_n4")


def _n4_correct(img: sitk.Image, shrink: int) -> sitk.Image:
    """N4 偏置场校正 (全图无 mask, float32)。"""
    img = sitk.Cast(img, sitk.sitkFloat32)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations([50] * 4)
    corr.SetConvergenceThreshold(0.001)
    corr.SetShrinkFactor(int(shrink))  # 必须调用方法; 属性赋值只会挂一个无效实例属性
    try:
        return corr.Execute(img)
    except Exception as e:  # 单样本失败 → 返回原图 (不阻塞)
        print(f"    N4 failed ({e}), fallback original", flush=True)
        return img


def _process_case(t_path, c_path, d_path, shrink):
    """N4 校正 + 锚定重采样 + bbox 裁剪 + zscore → [3,96,96,96] float32。"""
    paths = (t_path, c_path, d_path)
    imgs = [sitk.ReadImage(p) if (isinstance(p, str) and p and os.path.exists(p)) else None
            for p in paths]
    anchor_idx = next((i for i in range(3) if imgs[i] is not None), None)
    if anchor_idx is None:
        return np.zeros((3,) + tuple(SPATIAL), dtype=np.float32)
    # N4 校正 (仅对存在的序列)
    imgs = [_n4_correct(img, shrink) if img is not None else None for img in imgs]
    ref = _mri_reference_grid(imgs[anchor_idx])
    aligned = [None if img is None else sitk.GetArrayFromImage(_resample_to_ref(img, ref))
               for img in imgs]
    zs = [(_zscore_nonzero(a) if a is not None else None) for a in aligned]
    mask_ref = None
    for z in zs:
        if z is not None:
            mask_ref = z != 0
            break
    bbox = _mri_bbox(mask_ref)
    resize = Resize(spatial_size=SPATIAL, mode="trilinear")
    out = []
    for z in zs:
        if z is None:
            out.append(None)
        else:
            d0, d1, h0, h1, w0, w1 = bbox
            out.append(np.ascontiguousarray(resize(z[d0:d1, h0:h1, w0:w1][None])[0]))
    vols = []
    for i in range(3):
        vols.append(np.zeros(SPATIAL, dtype=np.float32) if out[i] is None else out[i].astype(np.float32))
    return np.stack(vols, axis=0)


def _key(t_path, c_path, d_path):
    """与 DataROI/loader 完全同一 key 格式 (N4 与原始缓存放在不同目录, 文件名一致)。"""
    return _det_cache_key_mri(t_path, c_path, d_path, SPATIAL)


def main():
    import argparse
    import pandas as pd
    import tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "hos_full_train_s42.csv"))
    ap.add_argument("--shrink", type=int, default=2)
    ap.add_argument("--out-dir", default=CACHE_DIR, help="N4 缓存输出目录 (默认 cache_hos_n4)")
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    rows = [r for _, r in df.iterrows() if not (isinstance(r.get("mri_path_t", ""), float) and np.isnan(r.get("mri_path_t", "")))]
    done = skip = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        paths = []
        for s in _SEQ_ORDER:
            v = row.get(f"mri_path_{s}", "")
            v = v.strip() if isinstance(v, str) else ""
            paths.append(os.path.join(ROOT, v) if v and not os.path.isabs(v) else v)
        t_path, c_path, d_path = paths
        key = _key(t_path, c_path, d_path)
        p = os.path.join(args.out_dir, f"{key}.npy")
        if os.path.exists(p):
            skip += 1
            continue
        vol = _process_case(t_path, c_path, d_path, args.shrink)
        fd, tmp = tempfile.mkstemp(suffix=".npy.tmp", dir=args.out_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, vol)
            os.replace(tmp, p)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        done += 1
        if done % 10 == 0 or i == len(rows) - 1:
            el = time.time() - t0
            print(f"[N4] {i + 1}/{len(rows)} done={done} skip={skip} elapsed={el:.0f}s ({(i + 1) / max(el, 1):.2f}/s)", flush=True)
    print(f"[N4] COMPLETE done={done} skip={skip} total={len(rows)} elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
