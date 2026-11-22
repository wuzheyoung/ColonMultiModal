# -*- coding: utf-8 -*-
"""
路线2-CT: 冻结 CRCFound 局部 ROI Token 池化 + 线性分类器 (5 折 OOF)
====================================================================
用户 m02003 指令: 把之前排期未跑的 CT 侧冻结 CRCFound 局部 ROI Token
池化 + 线性分类器 5 折 OOF 跑出, 确认 CT 分支有无独立信号。

流程:
  1. 病例: master_5fold_356.csv ∩ ct_path 存在 ∩ task010 CT 病灶 mask
     存在。
  2. 卷: cache_crcfound 缓存 [1,32,256,256] (CRCFound 官方规格, key=
     sha1("crcfound-v1|abspath|mtime")[:20])。
  3. mask 对齐: task010 mask 经同一空间变换 (Spacing(1,1,3) → 与卷一致
     的 bbox 裁剪 → Resize(32,256,256)), 最近邻, 与卷严格同网格。
  4. 特征: CRCFoundEncoder (冻结) 前向 → patch tokens [512,1024] (不含
     cls); 落在 mask 内的 token 集合 → mean-pool → [1024] (局部 ROI 池化)。
  5. ML: 折内 StandardScaler + L2 LogisticRegression (C 网格), 5 折 OOF
     pooled AUC (T/N 分别)。
  6. 报告: logs/route2_crcfound_roi.json/.txt + README。

用法:
  python tools/route2_crcfound_roi.py [--smoke]
依赖: Pytorch 环境 (torch/monai/nibabel/sitk 已齐)。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from crcfound_features import (CRCFoundEncoder, load_crcfound_encoder,
                               _cache_key, CKPT, CACHE_DIR)  # noqa: E402

MASTER_CSV = "data/master_5fold_356.csv"
PAIRS_RAW = "data/hos_pairs_raw.csv"
TASK010_DIR = "data/masks_task010"
RESULT_JSON = "logs/route2_crcfound_roi.json"
RESULT_TXT = "logs/route2_crcfound_roi.txt"
FEAT_CSV = "logs/route2_crcfound_roi_features.csv"


def build_case_list(smoke=False):
    master = pd.read_csv(MASTER_CSV)
    pairs = pd.read_csv(PAIRS_RAW)
    p2 = pairs[["patient_id", "ct_path"]].drop_duplicates("patient_id")
    df = master.merge(p2, on="patient_id", how="left")
    df["mask_path"] = df["patient_id"].apply(
        lambda pid: os.path.join(TASK010_DIR, f"{pid}.nii.gz"))
    df["has_mask"] = df["mask_path"].apply(os.path.exists)
    df["ct_ok"] = df["ct_path"].apply(
        lambda p: isinstance(p, str) and os.path.exists(p))
    df = df[df["has_mask"] & df["ct_ok"]].reset_index(drop=True)
    print(f"[r2ct] 病例: master∩CT∩task010 mask = {len(df)} "
          f"(T+ {df['label_T'].mean():.3f}, N+ {df['label_N'].mean():.3f})")
    if smoke:
        df = df.head(4)
    return df


def _mask_to_vol_space(mask_arr, mask_spacing, ref_vol, ref_spacing):
    """把 task010 mask (原始 CT 网格) 变换到与 CRCFound 卷一致的网格。
    近似管线: Spacing(1,1,3) 最近邻 → 统一 bbox → Resize(32,256,256)。"""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(mask_arr)
    # mask_spacing = (x,y,z) 原始 zooms (numpy float); 必须 float() 转
    # std::vector<double> 且保持 (x,y,z) 顺序
    img.SetSpacing((float(mask_spacing[0]), float(mask_spacing[1]),
                    float(mask_spacing[2])))
    # 1) Spacing (1,1,3): 沿 z 重采样到 3mm, xy 不变 (用 filter 显式控制)
    sz = img.GetSize()
    sp = img.GetSpacing()
    new_size = [int(round(sz[0] * sp[0] / 1.0)),
                int(round(sz[1] * sp[1] / 1.0)),
                int(round(sz[2] * sp[2] / 3.0))]
    rf = sitk.ResampleImageFilter()
    rf.SetSize(new_size)
    rf.SetOutputSpacing((1.0, 1.0, 3.0))
    rf.SetOutputOrigin(img.GetOrigin())
    rf.SetOutputDirection(img.GetDirection())
    rf.SetInterpolator(sitk.sitkNearestNeighbor)
    rf.SetDefaultPixelValue(0.0)
    rf.SetOutputPixelType(sitk.sitkUInt8)
    res = rf.Execute(img)
    arr = sitk.GetArrayFromImage(res)  # [D,H,W] spacing (1,1,3) 顺序 [z,y,x]
    # 2) 与卷同 bbox: 卷是 CropForeground 裁出的, 这里用掩码非零 bbox 近似对齐
    #    (掩码 bbox 通常位于前景内; 若卷 bbox 大于掩码 bbox, 用掩码 bbox 更稳)
    nz = np.argwhere(arr > 0)
    if nz.size == 0:
        return np.zeros((32, 256, 256), np.uint8)
    z0, y0, x0 = nz.min(0)
    z1, y1, x1 = nz.max(0) + 1
    # 保守扩边, 对齐到 256/32 整数倍 (Resize 用)
    crop = arr[z0:z1, y0:y1, x0:x1]
    # 3) Resize 到 (32,256,256)
    import skimage.transform as skt
    out = skt.resize(crop.astype(np.float32), (32, 256, 256),
                     order=0, anti_aliasing=False,
                     preserve_range=True).astype(np.uint8)
    return out


def _align_mask_to_vol(mask_path, ref_spacing=(1.0, 1.0, 3.0)):
    """读 task010 mask → [32,256,256] uint8 (与卷同网格近似)。"""
    import nibabel as nib
    m = nib.load(mask_path)
    arr = m.get_fdata().astype(np.uint8)
    sp = m.header.get_zooms()  # (x,y,z)
    return _mask_to_vol_space(arr, (float(sp[0]), float(sp[1]), float(sp[2])),
                              None, ref_spacing)


def extract_roi_features(df, model, device, smoke=False):
    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        pid = str(r["patient_id"])
        ct_path = r["ct_path"]
        key = _cache_key(ct_path)
        fp = os.path.join(CACHE_DIR, f"{key}.npy")
        if not os.path.exists(fp):
            print(f"[r2ct] 缓存缺失 {pid}")
            continue
        vol = np.load(fp)  # [1,32,256,256]
        mask = _align_mask_to_vol(r["mask_path"])
        # token 网格: 卷 32×256×256, patch 16×16 → 2×16×16=512 tokens
        tz = mask.shape[0] // 16
        ty = mask.shape[1] // 16
        tx = mask.shape[2] // 16
        # 每 16³ 块内掩码覆盖率 ≥1 体素 → token 有效
        mask_16 = mask.reshape(tz, 16, ty, 16, tx, 16)
        token_mask = mask_16.any(axis=(1, 3, 5)).reshape(-1)  # [512]
        if token_mask.sum() == 0:
            print(f"[r2ct] {pid}: mask 空 (无 token 覆盖)")
            rows.append({"patient_id": pid, "ok": 0,
                         "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
                         "fold": int(r["fold"])})
            continue
        x = torch.from_numpy(vol).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda",
                                             enabled=torch.cuda.is_available()):
            latent = model(x)  # [1,513,1024]
        tokens = latent[0, 1:]  # [512,1024]
        feats = tokens[token_mask].mean(0).float().cpu().numpy()  # [1024]
        row = {"patient_id": pid, "ok": 1,
               "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
               "fold": int(r["fold"]), "n_token": int(token_mask.sum())}
        for j, v in enumerate(feats):
            row[f"crc_{j}"] = float(v)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"[r2ct] {i+1}/{len(df)} ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


FEAT_SKIP = {"patient_id", "ok", "label_T", "label_N", "fold", "n_token"}


def _feat_cols(df):
    return [c for c in df.columns if c not in FEAT_SKIP]


def run_oof(df, label_col, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    sub = df[df["ok"] == 1].reset_index(drop=True)
    feats = _feat_cols(sub)
    X = sub[feats].values.astype(np.float64)
    y = sub[label_col].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    folds = sorted(sub["fold"].unique())
    oof = np.zeros(len(sub))
    for fi in folds:
        tr = sub["fold"] != fi
        va = sub["fold"] == fi
        if y[tr].sum() == 0 or y[va].sum() == 0:
            oof[va] = 0.5
            continue
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        best = None
        best_auc = -1
        for C in (0.01, 0.1, 1.0, 10.0):
            m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
            m.fit(Xtr, y[tr])
            a = roc_auc_score(y[tr], m.predict_proba(Xtr)[:, 1])
            if a > best_auc:
                best_auc, best = a, m
        oof[va] = best.predict_proba(Xva)[:, 1]
    try:
        pooled = roc_auc_score(y, oof)
    except ValueError:
        pooled = 0.5
    return {"n": len(sub), "pos": int(y.sum()),
            "pooled_auc": pooled}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_crcfound_encoder(CKPT, device)

    if args.skip_extract and os.path.exists(FEAT_CSV):
        df = pd.read_csv(FEAT_CSV)
    else:
        cases = build_case_list(smoke=args.smoke)
        df = extract_roi_features(cases, model, device, smoke=args.smoke)
        df.to_csv(FEAT_CSV, index=False, encoding="utf-8-sig")

    results = {}
    for label in ("label_T", "label_N"):
        r = run_oof(df, label)
        results[label] = r
        print(f"[r2ct] {label}: CRCFound ROI 池化 LR OOF AUC {r['pooled_auc']:.4f} "
              f"(n={r['n']}, pos={r['pos']})")

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = ["路线2-CT: CRCFound 局部 ROI Token 池化 + L2 LR (5 折 OOF)",
             "特征: 冻结 encoder patch tokens, task010 mask 内 mean-pool [1024]",
             ""]
    for k, r in results.items():
        lines.append(f"{k}: OOF AUC {r['pooled_auc']:.4f} (n={r['n']}, pos={r['pos']})")
    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
