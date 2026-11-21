# -*- coding: utf-8 -*-
# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""ROI 特征实验 (round9-ROI): 用 Task010 病灶 mask 提取 CT ROI 特征 + 临床 -> ML 判别力评估。
复用 radiomics_experiment 的 run_ml / _first_order / _glcm3d / run_stacking。
mask 与 CT 同为原始 nii.gz 空间(nnU-Net 输出=输入空间), 直接在原始空间提取, 无重采样对齐误差。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from radiomics_experiment import (_first_order, _glcm3d, run_ml,   # noqa: E402
                                  run_stacking, load_dl_probs, subset_dl_probs)
from data_loader.loader import _SEQ_ORDER  # noqa: E402

MASK_DIR = os.path.join(os.path.dirname(HERE), "data", "masks_task010")
MASK_DIR_TS = os.path.join(os.path.dirname(HERE), "data", "masks_totalseg")
CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]


def _read_nii(path):
    """nibabel 读 nii.gz -> float32 数组 [D,H,W]。"""
    import nibabel as nib
    if not (isinstance(path, str) and path and os.path.exists(path)):
        return None
    try:
        return np.ascontiguousarray(nib.load(path).get_fdata()).astype(np.float32)
    except Exception:
        return None


def _roi_feats(vol, mask):
    """ROI 特征: 一阶(mask 内) + GLCM + 形态。mask 外体素置 0, 复用 _first_order 的 vol[vol!=0] 语义。
    返回 11(一阶) + 6(GLCM) + 3(形态) = 20 维。"""
    if vol is None or mask is None:
        return [0.0] * 20
    mask = (mask > 0).astype(np.float32)
    if mask.sum() < 8:
        return [0.0] * 20
    roi = vol * mask                                   # ROI 外 = 0
    feats = _first_order(roi) + _glcm3d(roi)
    # 形态: 归一化体素占比 / 最大连通域占比 / bbox 对角长度
    import scipy.ndimage as ndi
    n_roi = int(mask.sum())
    lbl, nlab = ndi.label(mask)
    sizes = np.bincount(lbl.ravel())[1:]
    max_cc = int(sizes.max()) if nlab > 0 else 0
    nz = np.argwhere(mask > 0)
    mn, mx = nz.min(0).astype(float), nz.max(0).astype(float)
    bbox_diag = float(np.linalg.norm(mx - mn))
    feats += [n_roi / max(mask.size, 1), max_cc / max(n_roi, 1),
              bbox_diag / max(vol.shape[0], 1)]
    return feats


def _read_mask(pid, mask_kind):
    """mask_kind: 'task010' -> masks_task010/{pid}.nii.gz; 'totalseg' -> masks_totalseg/{pid}/colon.nii.gz;
    'mr_t'/'mr_c'/'mr_d' -> masks_totalseg_mr_{seq}/{pid}/colon.nii.gz (MRI 结肠)"""
    if mask_kind in ("mr_t", "mr_c", "mr_d"):
        seq = mask_kind[3]
        p = os.path.join(os.path.dirname(HERE), "data", f"masks_totalseg_mr_{seq}", pid, "colon.nii.gz")
    elif mask_kind == "totalseg":
        p = os.path.join(MASK_DIR_TS, pid, "colon.nii.gz")
    else:
        p = os.path.join(MASK_DIR, f"{pid}.nii.gz")
    return _read_nii(p) if os.path.exists(p) else None


def extract_roi_features(train_csv, val_csv, out_csv, smoke=False, mask_kind="task010",
                         mri_seq=None):
    """提取 ROI 特征表 (train/val, 与 DL seed42 划分一致)。
    mask_kind='mr_t/c/d' + mri_seq 时: 影像源=MRI 序列卷, mask=同序列结肠 mask (MRI ROI 实验)。"""
    dfs = []
    for split, csv_path in (("train", train_csv), ("val", val_csv)):
        df = pd.read_csv(csv_path)
        if smoke:
            df = df.head(2)
        rows = []
        for _, r in df.iterrows():
            t0 = time.time()
            pid = str(r.get("patient_id", ""))
            if mri_seq:
                seq = mri_seq
                vol_path = r.get(f"mri_path_{seq}", "")
                avail = str(r.get(f"mri_available_{seq}", "")).strip().lower() == "true"
                has_modal = avail and bool(vol_path)
                vol = _read_nii(vol_path) if has_modal else None
                mask = _read_mask(pid, f"mr_{seq}") if has_modal else None
            else:
                ct_path = r.get("ct_path", "")
                has_modal = str(r.get("ct_available", "")).strip().lower() == "true" and bool(ct_path)
                vol = _read_nii(ct_path) if has_modal else None
                mask = _read_mask(pid, mask_kind) if has_modal else None
            feats = _roi_feats(vol, mask)
            row = {"patient_id": pid, "split": split,
                   "has_ct": int(has_modal), "has_roi": int(mask is not None and mask.sum() > 0),
                   "label_T": int(r["label_T"]), "label_N": int(r["label_N"])}
            for i, v in enumerate(feats):
                row[f"roi_{i}"] = v
            for c in CLINICAL_COLS:
                v = r.get(c)
                row[f"clin_{c}"] = float(v) if v is not None else 0.0
            rows.append(row)
            if smoke:
                print(f"[roi] {pid} has_ct={row['has_ct']} has_roi={row['has_roi']} "
                      f"t={time.time()-t0:.1f}s", flush=True)
        dfs.append(pd.DataFrame(rows))
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(out_csv, index=False)
    return out


def _feat_columns(df):
    return [c for c in df.columns if c.startswith("roi_") or c.startswith("clin_")]


def main():
    ap = argparse.ArgumentParser(description="ROI 特征实验 (round9-ROI)")
    ap.add_argument("--train-csv", default="data/hos_full_train_s42.csv")
    ap.add_argument("--val-csv", default="data/hos_full_val_s42.csv")
    ap.add_argument("--out-csv", default="logs/roi_full_features.csv")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stacking", action="store_true",
                    help="加 R5-DL 概率 stacking (需 GPU, Task010 推理期间勿用)")
    ap.add_argument("--ckpt-t", default="checkpoints_hos_full/best_model_hos_full_s0.pth")
    ap.add_argument("--ckpt-n", default="checkpoints_hos_full/best_model_n_hos_full_s0.pth")
    ap.add_argument("--mask-kind", default="task010", choices=["task010", "totalseg", "mr_t", "mr_c", "mr_d"],
                    help="使用病灶 mask(task010) / 器官 mask(totalseg) / MRI 结肠 mask(mr_t/c/d) 提取 ROI 特征")
    ap.add_argument("--mri-seq", choices=["t", "c", "d"], default=None,
                    help="MRI 模式: 从 MRI 序列卷 + 同序列结肠 mask 提取 ROI 特征")
    args = ap.parse_args()

    if args.mri_seq:
        args.mask_kind = f"mr_{args.mri_seq}"
        args.out_csv = f"logs/roi_mri_{args.mri_seq}_features.csv"
        report_name = f"roi_mri_{args.mri_seq}_experiment_full.json"

    t0 = time.time()
    df = extract_roi_features(args.train_csv, args.val_csv, args.out_csv,
                              smoke=args.smoke, mask_kind=args.mask_kind,
                              mri_seq=args.mri_seq)
    print(f"[extract] {len(df)} rows ({time.time()-t0:.0f}s), roi features -> {args.out_csv}")

    # 只评估有模态子集 (CT 或 MRI 序列)
    df_ct = df[df["has_ct"] == 1].copy()
    print(f"[subset] has-modal: {len(df_ct)}/{len(df)} "
          f"(train {(df_ct['split']=='train').sum()}, val {(df_ct['split']=='val').sum()})")
    if len(df_ct) < 10:
        print("[smoke] CT 子集样本不足, 跳过 ML (仅验证特征提取)")
        return
    feats = _feat_columns(df_ct)
    print(f"[feats] {len(feats)} cols: roi_*={sum(c.startswith('roi_') for c in feats)}, "
          f"clin_*={sum(c.startswith('clin_') for c in feats)}")

    results = {}
    for task, label_col in (("T", "label_T"), ("N", "label_N")):  # noqa: N806
        ml = run_ml(df_ct, label_col, feats)
        results[task] = {"ml": ml}
        # stacking (可选): 特征 + DL 概率 -> LR
        if args.stacking:
            print(f"[stacking] loading DL probs for task {task}...", flush=True)
            dl = load_dl_probs(args.train_csv, args.val_csv, args.ckpt_t, args.ckpt_n)
            # DL 概率覆盖全 csv, 而 df_ct 只是有模态子集 → 必须按 patient_id 对齐后再 stack
            dl = subset_dl_probs(dl, df_ct["patient_id"].astype(str).tolist())
            st = run_stacking(df_ct, dl, label_col, feats)
            results[task]["stack"] = st

    out_path = os.path.join(os.path.dirname(args.out_csv), report_name if "report_name" in dir() else "roi_experiment_full.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_ct": len(df_ct), "results": results}, f, ensure_ascii=False, indent=2,
                  default=str)
    txt = [f"ROI 特征实验 (round9-ROI)  有模态子集 {len(df_ct)} 样本", ""]
    for task, res in results.items():
        txt.append(f"--- {task} ---")
        for mname, mres in res["ml"].items():
            txt.append(f"{mname}: CV-F1 {mres.get('cv_macro_f1'):.4f}±{mres.get('cv_f1_std', 0):.4f} "
                       f"AUC {mres.get('cv_auc', 0):.4f} | val F1 {mres.get('val_macro_f1'):.4f} "
                       f"AUC {mres.get('val_auc', 0):.4f}")
        if "stack" in res:
            s = res["stack"]
            txt.append(f"stack: CV-F1 {s.get('cv_macro_f1'):.4f}±{s.get('cv_f1_std', 0):.4f} "
                       f"AUC {s.get('cv_auc', 0):.4f} | val F1 {s.get('val_macro_f1'):.4f} "
                       f"AUC {s.get('val_auc', 0):.4f}")
            txt.append(f"  (注: {s.get('note', '')})")
    txt_path = out_path.replace(".json", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    print("\n".join(txt))
    print(f"[report] {txt_path}")


if __name__ == "__main__":
    main()
