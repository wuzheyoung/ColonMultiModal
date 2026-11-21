# -*- coding: utf-8 -*-
"""Task010_Colon 病灶分割推理 (round9-S10, nnU-Net v1 预训练权重)

对 full 分支全部有 CT 的样本跑 Task010_Colon 3d_fullres 5 折集成分割,
输出病灶 mask 到 data/masks_task010/{patient_id}.nii.gz, 并统计:
  - 每样本 mask 体素数 / 最大连通域 / 是否检出病灶
  - 汇总报告 logs/task010_seg_full.{json,txt}

用法:
  python tools/run_task010_seg.py                 # 全量 (GPU)
  python tools/run_task010_seg.py --limit 2       # 冒烟 (前 2 个有 CT 样本)
  python tools/run_task010_seg.py --folds 0       # 只跑 fold_0 (更快, 精度略降)
  python tools/run_task010_seg.py --no-tta        # 关闭 TTA (更快)
"""
import os
import sys
import json
import time
import argparse

# ---- 环境变量必须在 import nnunet 之前设置 ----
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["RESULTS_FOLDER"] = os.path.join(PROJ, "pretrained")
os.environ.setdefault("nnUNet_raw_data_base", os.path.join(PROJ, "pretrained", "_nnunet_tmp"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(PROJ, "pretrained", "_nnunet_tmp", "preprocessed"))
os.environ.setdefault("TOTALSEG_HOME_DIR", os.path.join(PROJ, "pretrained", "totalsegmentator"))

NNUNET_SRC = os.path.join(PROJ, "third_party", "nnUNet-1.7.1")
if os.path.isdir(NNUNET_SRC) and NNUNET_SRC not in sys.path:
    sys.path.insert(0, NNUNET_SRC)

import numpy as np
import nibabel as nib
import pandas as pd

MODEL_DIR = os.path.join(PROJ, "pretrained", "nnUNet", "3d_fullres",
                        "Task010_Colon", "nnUNetTrainerV2__nnUNetPlansv2.1")
OUT_DIR = os.path.join(PROJ, "data", "masks_task010")
CSV = os.path.join(PROJ, "data", "hos_pairs.csv")


def collect_ct_samples(limit=None):
    df = pd.read_csv(CSV)
    df = df[df["ct_available"] == True]  # noqa: E712  (full 分支有真 CT 的样本)
    df = df.reset_index(drop=True)
    if limit:
        df = df.head(limit)
    # 过滤文件存在
    df = df[df["ct_path"].apply(lambda p: os.path.isfile(p) and p.strip() != "")]
    print(f"[CT samples] total={len(df)} (ct_available=True, files exist)")
    return df


def main():
    ap = argparse.ArgumentParser(description="Task010_Colon 病灶分割")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个样本(冒烟)")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4", help="使用的折, 逗号分隔")
    ap.add_argument("--no-tta", action="store_true", help="关闭 TTA")
    ap.add_argument("--num-threads", type=int, default=4, help="预处理线程数")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的 mask")
    args = ap.parse_args()

    folds = tuple(int(f) for f in args.folds.split(",") if f.strip())
    tta = not args.no_tta

    os.makedirs(OUT_DIR, exist_ok=True)
    assert os.path.isdir(MODEL_DIR), f"model dir not found: {MODEL_DIR}"

    df = collect_ct_samples(args.limit)

    # 待处理样本
    todo = []
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        out_p = os.path.join(OUT_DIR, f"{pid}.nii.gz")
        if os.path.isfile(out_p) and not args.overwrite:
            continue
        todo.append((pid, row["ct_path"], out_p))
    print(f"[to process] {len(todo)} samples (existing skipped: {len(df)-len(todo)})")
    if not todo:
        print("nothing to do")
        return

    list_of_lists = [[p] for _, p, _ in todo]
    out_files = [o for _, _, o in todo]
    pids = [i for i, _, _ in todo]

    t0 = time.time()
    from nnunet.inference.predict import predict_cases
    print(f"[predict] folds={folds} tta={tta} model={os.path.basename(os.path.dirname(MODEL_DIR))}")
    predict_cases(
        model=MODEL_DIR,
        list_of_lists=list_of_lists,
        output_filenames=out_files,
        folds=folds,
        save_npz=False,
        num_threads_preprocessing=args.num_threads,
        num_threads_nifti_save=2,
        do_tta=tta,
        mixed_precision=True,
        all_in_gpu=False,
        step_size=0.5,
        checkpoint_name="model_final_checkpoint",
        disable_postprocessing=False,
    )
    dt = time.time() - t0
    print(f"[predict done] {len(todo)} samples in {dt/60:.1f} min")

    # ---- 统计 ----
    rows = []
    for pid, out_p in zip(pids, out_files):
        if not os.path.isfile(out_p):
            rows.append({"patient_id": pid, "mask_exists": False})
            continue
        try:
            arr = nib.load(out_p).get_fdata()
        except Exception as e:  # noqa
            rows.append({"patient_id": pid, "mask_exists": True, "read_error": str(e)})
            continue
        lab = arr > 0.5
        n_vox = int(lab.sum())
        if n_vox > 0:
            from scipy import ndimage
            cc, n_cc = ndimage.label(lab)
            sizes = np.bincount(cc.ravel())[1:]
            max_cc = int(sizes.max())
            centroid = np.array(ndimage.center_of_mass(lab)) if n_vox else [np.nan] * 3
            bbox = np.argwhere(lab)
            bbox_lo, bbox_hi = bbox.min(0), bbox.max(0)
            rows.append({
                "patient_id": pid, "mask_exists": True, "n_voxels": n_vox,
                "n_components": int(n_cc), "max_component_voxels": max_cc,
                "centroid_zxy": [float(c) for c in centroid],
                "bbox_zxy": [[int(x) for x in bbox_lo], [int(x) for x in bbox_hi]],
                "bbox_voxels": int(np.prod(bbox_hi - bbox_lo + 1)),
            })
        else:
            rows.append({"patient_id": pid, "mask_exists": True, "n_voxels": 0,
                         "n_components": 0, "max_component_voxels": 0, "centroid_zxy": None})

    rep = {
        "folds": list(folds), "tta": tta, "samples_requested": len(todo),
        "samples_done": len(rows), "time_min": round(dt / 60, 2),
        "samples": rows,
        "summary": {
            "n_with_mask": sum(1 for r in rows if r.get("mask_exists") and r.get("n_voxels", 0) > 0),
            "n_empty": sum(1 for r in rows if r.get("n_voxels", -1) == 0),
            "total_voxels": sum(r.get("n_voxels", 0) for r in rows),
        },
    }
    rep_path = os.path.join(PROJ, "logs", "task010_seg_full.json")
    os.makedirs(os.path.dirname(rep_path), exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    txt = os.path.join(PROJ, "logs", "task010_seg_full.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"Task010_Colon 分割结果 (folds={folds} tta={tta})\n")
        f.write(f"样本数: {len(rows)}  用时: {dt/60:.1f} min\n")
        f.write(f"检出病灶: {rep['summary']['n_with_mask']}  空mask: {rep['summary']['n_empty']}  总体素: {rep['summary']['total_voxels']}\n")
        f.write("-" * 60 + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[report] {rep_path}")
    print(json.dumps(rep["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
