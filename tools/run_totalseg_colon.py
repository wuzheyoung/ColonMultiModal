# -*- coding: utf-8 -*-
"""TotalSegmentator 结肠分割 (round9-ROI): 对 145 个有 CT 的样本分割结肠器官 mask。
用 total 任务(117 类) roi_subset=['colon'] (index 20), 权重在 pretrained/totalsegmentator
(TOTALSEG_HOME_DIR)。输出 data/masks_totalseg/{pid}/colon.nii.gz + 统计 logs/totalseg_colon_full.{json,txt}。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

os.environ.setdefault("TOTALSEG_HOME_DIR",
                      os.path.join(ROOT, "pretrained", "totalsegmentator"))

MASK_DIR = os.path.join(ROOT, "data", "masks_totalseg")
CSV = os.path.join(ROOT, "data", "hos_pairs.csv")


def collect_ct_samples(csv_path, limit=None):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        ct_path = str(r.get("ct_path", ""))
        avail = str(r.get("ct_available", "")).strip().lower() == "true"
        if avail and ct_path and os.path.exists(ct_path):
            rows.append({"patient_id": str(r["patient_id"]), "ct_path": ct_path,
                         "label_T": int(r["label_T"]), "label_N": int(r["label_N"])})
    if limit:
        rows = rows[:limit]
    return rows


def collect_mr_samples(csv_path, seq, limit=None):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        key = "mri_path_" + seq
        mpath = str(r.get(key, ""))
        avail_key = "mri_available_" + seq
        avail = str(r.get(avail_key, "")).strip().lower() == "true"
        if avail and mpath and os.path.exists(mpath):
            rows.append({"patient_id": str(r["patient_id"]), "path": mpath,
                         "label_T": int(r["label_T"]), "label_N": int(r["label_N"])})
    if limit:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser(description="TotalSegmentator 结肠分割 (round9-ROI)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--roi", default="colon", help="roi_subset, 默认 colon")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"], help="TotalSegmentator 设备: gpu 或 cpu (注意: 不是 'cuda')")
    ap.add_argument("--seq", choices=["t", "c", "d", None], default=None,
                    help="MRI 序列 (t/c/d): 给定时用 total_mr 任务对 MRI 分割; None = CT 用 total 任务")
    args = ap.parse_args()

    if args.seq is not None:
        mask_dir = os.path.join(ROOT, "data", "masks_totalseg_mr_" + args.seq)
        task = "total_mr"
        samples = collect_mr_samples(CSV, args.seq, limit=args.limit)
        log_name = "totalseg_mr_" + args.seq
    else:
        mask_dir = MASK_DIR
        task = "total"
        samples = collect_ct_samples(CSV, limit=args.limit)
        log_name = "totalseg_colon"
    os.makedirs(mask_dir, exist_ok=True)
    print(f"[samples] total={len(samples)} (seq={args.seq or 'CT'})")

    from totalsegmentator.python_api import totalsegmentator

    done, skipped, failed = [], [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        pid, inp = s["patient_id"], s["path"]
        # 实际产物是 {mask_dir}/{pid}/colon.nii.gz (TotalSegmentator 目录语义), 跳过判定必须查该路径
        out = os.path.join(mask_dir, pid, "colon.nii.gz")
        if os.path.exists(out) and not args.overwrite:
            skipped.append(pid)
            continue
        ts = time.time()
        try:
            # TotalSegmentator 2.x: output 为目录语义, 在 {out_dir}/ 下写 colon.nii.gz
            out_dir = os.path.join(mask_dir, pid)
            os.makedirs(out_dir, exist_ok=True)
            totalsegmentator(inp, out_dir, roi_subset=[args.roi], device=args.device,
                             task=task, verbose=False)
            done.append(pid)
            print(f"[{i+1}/{len(samples)}] {pid} colon mask in {time.time()-ts:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append({"patient_id": pid, "error": str(e)})
            print(f"[{i+1}/{len(samples)}] {pid} FAILED: {e}", flush=True)

    # 统计
    stats = []
    for s in samples:
        pid = s["patient_id"]
        mpath = os.path.join(mask_dir, pid, "colon.nii.gz")
        if not os.path.exists(mpath):
            stats.append({"patient_id": pid, "mask_exists": False, "n_voxels": 0,
                          "label_T": s["label_T"], "label_N": s["label_N"]})
            continue
        import nibabel as nib
        arr = np.asarray(nib.load(mpath).get_fdata())
        stats.append({"patient_id": pid, "mask_exists": True,
                      "n_voxels": int((arr > 0).sum()),
                      "label_T": s["label_T"], "label_N": s["label_N"]})

    report = {"samples_requested": len(samples), "done": len(done),
              "skipped": len(skipped), "failed": len(failed),
              "time_min": round((time.time() - t0) / 60, 2),
              "n_with_mask": sum(1 for x in stats if x["mask_exists"]),
              "n_empty": sum(1 for x in stats if x["mask_exists"] and x["n_voxels"] == 0),
              "total_voxels": int(sum(x["n_voxels"] for x in stats)),
              "samples": stats, "errors": failed}
    jp = os.path.join(ROOT, "logs", log_name + "_full.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    tp = jp.replace(".json", ".txt")
    with open(tp, "w", encoding="utf-8") as f:
        f.write(f"TotalSegmentator colon (roi={args.roi}) 样本 {len(samples)} "
                f"用时 {report['time_min']} min\n"
                f"检出 {report['n_with_mask']} / 空 {report['n_empty']} "
                f"总体素 {report['total_voxels']}\n")
    print(f"[report] {tp}")


if __name__ == "__main__":
    main()
