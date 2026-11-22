# -*- coding: utf-8 -*-
"""A3: S3 多器官排斥直肠系膜 (nnUNet Dataset850 MRI 分割 → 排斥 mask → 验证)
规则(用户 m00745 A3): 查 TotalSegmentator total_mr 盆腔标签 (sacrum/urinary_bladder/prostate 均有,
Dataset850 labels: colon=15, urinary_bladder=16, prostate=17, sacrum=18, vertebrae=19, spinal_cord=21)。
→ 296 例 T2 分割 (~2.1h) + 3seed 验证 (~0.7h)。

流程:
1) nnUNetv2 (Dataset850_TotalSegMRI_part1_organs_1088subj, fold_0, checkpoint_final.pth)
   逐例推理 hos_pairs.mri_path_t (绝对路径, batch1 _nifti/MRI + batch2 _nifti/batch2)
   → tools/phase2_relay/outputs/a3_mesorectum/seg/{pid}.nii.gz (多标签 0-29, 与输入同网格)
2) 排斥 mask: 原 masks_totseg_mr_t/{pid}/colon.nii.gz 减去其他器官标签(排除colon/background)
   膨胀 2 体素的区域 → data/masks_totalseg_mr_t_excl/{pid}/colon.nii.gz
   (作用: ring/SDF 不再与膀胱/前列腺/骶骨/椎骨等非肠器官接触 → 排除多器官伪信号)
3) 验证: train_roi_dl.py --mode mri_t --sdf-channel --roi-mask bbox --mask-mr-root data/masks_totalseg_mr_t_excl
   seed 42/0/7, epochs 30, 聚合 last5 OOF N AUC + label_T 分层 → README
增量: seg/mask 已存在跳过 (超时被杀后可续跑; 分割>验证优先)。
输出: logs/step_a3_mesorectum.json + README
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import nibabel as nib
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = r"D:\Development\miniconda\envs\Pytorch\python.exe"
SEG_DIR = os.path.join(ROOT, "tools", "phase2_relay", "outputs", "a3_mesorectum", "seg")
EXCL_DIR = os.path.join(ROOT, "data", "masks_totalseg_mr_t_excl")
SRC_DIR = os.path.join(ROOT, "data", "masks_totalseg_mr_t")
NIFTI = os.path.join(ROOT, "data", "Hos", "_nifti", "MRI")
RESULTS_FOLDER = os.path.join(ROOT, "pretrained", "totalsegmentator", "nnunet", "results")
MODEL_DIR = os.path.join(RESULTS_FOLDER, "Dataset850_TotalSegMRI_part1_organs_1088subj",
                         "nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres")
# 需从 ring/SDF 排斥的器官 (排除 colon=15 与 background=0)
EXCLUDE_LABELS = set(range(1, 15)) | {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}


def _load_pids():
    """从 hos_pairs_raw.csv 取 mri_path_t 绝对路径 (batch1 _nifti/MRI + batch2 _nifti/batch2),
    与 master + 结肠 mask 交集。返回 [(pid, t2_abs_path)]。"""
    import pandas as pd
    master = pd.read_csv(os.path.join(ROOT, "data", "master_5fold_356.csv"), encoding="utf-8-sig")
    pairs = pd.read_csv(os.path.join(ROOT, "data", "hos_pairs_raw.csv"), encoding="utf-8-sig")
    master_pids = set(master["patient_id"].astype(str))
    rows = []
    for _, r in pairs.iterrows():
        pid = str(r["patient_id"])
        t2 = str(r.get("mri_path_t") or "")
        if not t2 or pid not in master_pids:
            continue
        if not os.path.isabs(t2):
            t2 = os.path.join(ROOT, t2)
        if not os.path.exists(t2):
            continue
        if os.path.exists(os.path.join(SRC_DIR, pid, "colon.nii.gz")):
            rows.append((pid, t2))
    return sorted(rows, key=lambda x: int(x[0]) if x[0].isdigit() else x[0])


def _segment(items, limit=None):
    """nnUNetv2 逐例推理 (Dataset850 fold_0 checkpoint_final.pth, GPU)。增量跳过。
    items = [(pid, t2_abs_path)]; seg 输出与输入同网格 (512x512xD)。"""
    os.makedirs(SEG_DIR, exist_ok=True)
    done, todo = 0, 0
    os.environ["nnUNet_results"] = RESULTS_FOLDER
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(device=torch.device("cuda"), allow_tqdm=False)
    pred.initialize_from_trained_model_folder(MODEL_DIR, use_folds=(0,),
                                              checkpoint_name="checkpoint_final.pth")
    for pid, t2 in items:
        outp = os.path.join(SEG_DIR, f"{pid}.nii.gz")
        if os.path.exists(outp):
            done += 1
            continue
        if limit is not None and todo >= limit:
            break
        todo += 1
        trunc = os.path.join(SEG_DIR, pid)  # nnUNetv2 输出 {pid}.nii.gz
        pred.predict_from_files([[t2]], [trunc], save_probabilities=False, overwrite=True,
                                num_processes_preprocessing=1, num_processes_segmentation_export=1)
        done += 1
        print(f"[A3] seg {pid} ({done}/{len(items)})", flush=True)
    return done


def _build_excl_masks(pids):
    """colon mask 减去其他器官(膨胀2体素) → 排斥 mask。增量跳过。"""
    os.makedirs(EXCL_DIR, exist_ok=True)
    n = 0
    for pid in pids:
        outp = os.path.join(EXCL_DIR, pid, "colon.nii.gz")
        if os.path.exists(outp):
            n += 1
            continue
        seg_p = os.path.join(SEG_DIR, f"{pid}.nii.gz")
        src_p = os.path.join(SRC_DIR, pid, "colon.nii.gz")
        if not (os.path.exists(seg_p) and os.path.exists(src_p)):
            continue
        seg = nib.load(seg_p).get_fdata()  # [H,W,D] 多标签 (nnUNet 输出 3D)
        colon = nib.load(src_p).get_fdata()
        from scipy import ndimage as ndi
        excl = np.zeros_like(seg, dtype=np.uint8)
        for lab in EXCLUDE_LABELS:
            m = seg == lab
            if m.any():
                excl |= ndi.binary_dilation(m, iterations=2)
        out = colon.copy()
        out[excl > 0] = 0
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        aff = nib.load(src_p).affine
        nib.save(nib.Nifti1Image(out.astype(np.uint8), aff), outp)
        n += 1
    return n


def _verify(pids, tag_prefix):
    """train_roi_dl mri_t+SDF bbox + 排斥 mask, 3 seed, 聚合 + label_T 分层。"""
    tag = f"{tag_prefix}_a3_excl"
    for seed in (42, 0, 7):
        outp = os.path.join(ROOT, "results", "roidl_oof", f"{tag}_k0_s{seed}.csv")
        if os.path.exists(outp):
            continue
        r = subprocess.run([PY, os.path.join(ROOT, "tools", "train_roi_dl.py"),
                            "--kfold", "5", "--kfold-csv", "data/master_5fold_356.csv",
                            "--full-csv", "data/hos_pairs.csv", "--seed", str(seed),
                            "--mode", "mri_t", "--sdf-channel", "--roi-mask", "bbox",
                            "--epochs", "30", "--self-written", "--no-clinical",
                            "--head-dropout", "0.5", "--mask-mr-root", "data/masks_totalseg_mr_t_excl",
                            "--out-tag", tag],
                           cwd=ROOT, capture_output=True, text=True)
        print(f"[A3] verify seed {seed} rc={r.returncode}")
    aggs = {}
    for seed in (42, 0, 7):
        subprocess.run([PY, os.path.join(ROOT, "tools", "aggregate_roidl_kfold.py"),
                        "--tag", tag, "--seed", str(seed),
                        "--out", os.path.join(ROOT, "logs", f"aggregate_roidl_{tag}_s{seed}.json")],
                       cwd=ROOT, capture_output=True, text=True)
        apath = os.path.join(ROOT, "logs", f"aggregate_roidl_{tag}_s{seed}.json")
        if os.path.exists(apath):
            with open(apath, encoding="utf-8") as f:
                aggs[f"s{seed}"] = json.load(f)
    return aggs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="dc")
    ap.add_argument("--seg-only", action="store_true", help="只分割, 不生成 mask/验证")
    ap.add_argument("--seg-limit", type=int, default=None, help="调试: 只分割前 N 例")
    args = ap.parse_args()

    t0 = time.time()
    items = _load_pids()  # [(pid, t2_abs_path)]
    pids = [p for p, _ in items]
    print(f"[A3] {len(pids)} 例 (master∩mr_t mask∩T2 nifti)")
    n_seg = _segment(items, limit=args.seg_limit)
    print(f"[A3] seg done={n_seg}/{len(pids)} ({time.time()-t0:.0f}s)")
    out = {"gen": time.strftime("%Y-%m-%d %H:%M:%S"), "n_pids": len(pids), "n_seg": n_seg}
    if args.seg_only:
        with open(os.path.join(ROOT, "logs", "step_a3_mesorectum.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return 0
    n_excl = _build_excl_masks(pids)
    print(f"[A3] excl mask done={n_excl}/{len(pids)}")
    out["n_excl"] = n_excl
    seg_ok = [p for p in pids if os.path.exists(os.path.join(EXCL_DIR, p, "colon.nii.gz"))]
    if len(seg_ok) < 100:
        print(f"[A3] 排斥 mask 仅 {len(seg_ok)} 例 (<100), 验证跳过 (分割未完可续跑)")
        out["verify"] = "skipped: masks < 100"
    else:
        aggs = _verify(seg_ok, args.tag_prefix)
        summary = {}
        for s, v in aggs.items():
            nn = v.get("task_N_last5", {}).get("pooled", {}).get("roc_auc")
            summary[s] = nn
        out["verify"] = {"last5_N_auc_per_seed": summary}
        # label_T 分层 (T3-4 vs T1-2)
        import pandas as pd
        from sklearn.metrics import roc_auc_score
        allp = []
        tag = f"{args.tag_prefix}_a3_excl"
        for seed in (42, 0, 7):
            ps = []
            for fi in range(5):
                pp = os.path.join(ROOT, "results", "roidl_oof", f"{tag}_k{fi}_s{seed}_last5.csv")
                if not os.path.exists(pp):
                    pp = os.path.join(ROOT, "results", "roidl_oof", f"{tag}_k{fi}_s{seed}.csv")
                if os.path.exists(pp):
                    ps.append(pd.read_csv(pp, encoding="utf-8-sig"))
            if ps:
                d = pd.concat(ps, ignore_index=True)
                d["probN"] = d["prob_N"]
                allp.append(d)
        if allp:
            base = allp[0][["patient_id"]].copy()
            for i, d in enumerate(allp):
                base[f"p{i}"] = d.set_index("patient_id")["probN"]
            base["prob"] = base[[c for c in base.columns if c.startswith("p")]].mean(1)
            master = pd.read_csv(os.path.join(ROOT, "data", "master_5fold_356.csv"), encoding="utf-8-sig")
            m = master[["patient_id", "label_T", "label_N"]].copy()
            dd = base.merge(m, on="patient_id")
            t34 = dd[dd["label_T"] == 1]
            t12 = dd[dd["label_T"] == 0]
            out["verify"]["T34_N_auc"] = float(roc_auc_score(t34["label_N"], t34["prob"])) if len(t34) > 5 else None
            out["verify"]["T12_N_auc"] = float(roc_auc_score(t12["label_N"], t12["prob"])) if len(t12) > 5 else None
            out["verify"]["n"] = len(dd)
    with open(os.path.join(ROOT, "logs", "step_a3_mesorectum.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "README.md"), "a", encoding="utf-8") as f:
        v = out.get("verify", {})
        s = v.get("last5_N_auc_per_seed", {})
        f.write(f"\n## A3 多器官排斥直肠系膜 (自动接力 {time.strftime('%Y-%m-%d %H:%M')})\n"
                f"- Dataset850 MRI 分割 {n_seg} 例, 排斥 mask {n_excl} 例; "
                f"验证 last5 N AUC/seed={s}, T3-4={v.get('T34_N_auc')}, T1-2={v.get('T12_N_auc')} "
                f"(logs/step_a3_mesorectum.json)\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
