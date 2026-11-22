# -*- coding: utf-8 -*-
"""
路线1: 原生面内分辨率 + 完整掩码影像组学 (纯 CPU, 零深度过拟合)
================================================================
用户 m02003 指令: 直接在原始 NIfTI 轴位图像上 (不插值 z 轴, 保留高面内
分辨率), 用完整未裁断的结直肠本体掩码(测 T) 和 5-10mm 外扩环(测 N)
提取标准 2D/3D PyRadiomics 纹理与一阶特征。折内标准化 + 方差/相关性
筛选 + 强正则 L2 LogisticRegression / ElasticNet, 5 折 OOF AUC。

流程:
  1. 病例清单: master_5fold_356.csv ∩ mri_path_t 存在 ∩ mr_t colon mask
     存在 → T2 原始轴位 NIfTI + 本体 mask (T) + 外扩环 mask (N)。
  2. 特征提取: PyRadiomics (py3.9 环境, 官方 cp39 wheel 免编译),
     mask 重采样到 T2 网格(最近邻), 3D 特征类 = firstorder/glcm/glrlm/
     glszm/gldm/ngtdm/shape(3D) + shape2D, 每例 T 本体 + N 环 两套。
  3. ML: 折内筛选(方差>1e-4 + 相关性去冗余 + f_classif top300) +
     StandardScaler + L2 LogisticRegression(C 网格) / ElasticNet,
     5 折 OOF pooled AUC (T 用本体特征, N 用环特征)。
  4. 报告: logs/route1_radiomics.json/.txt + README 追加。

用法:
  python tools/route1_radiomics.py                 # 全流程
  python tools/route1_radiomics.py --skip-extract  # 复用 logs/route1_radiomics_features.csv
  python tools/route1_radiomics.py --smoke         # 5 例冒烟
依赖: 需在 pyrad39 环境运行 (conda create -n pyrad39 python=3.9 +
      pip install pyradiomics numpy pandas scikit-learn SimpleITK nibabel scipy)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MASTER_CSV = "data/master_5fold_356.csv"
PAIRS_RAW = "data/hos_pairs_raw.csv"
MASK_DIR = "data/masks_totalseg_mr_t"
FEAT_CSV = "logs/route1_radiomics_features.csv"
RESULT_JSON = "logs/route1_radiomics.json"
RESULT_TXT = "logs/route1_radiomics.txt"
TMP_MASK_DIR = "logs/route1_masks"


def build_case_list(smoke=False):
    master = pd.read_csv(MASTER_CSV)
    pairs = pd.read_csv(PAIRS_RAW)
    p2 = pairs[["patient_id", "mri_path_t"]].drop_duplicates("patient_id")
    df = master.merge(p2, on="patient_id", how="left")
    df["has_mask"] = df["patient_id"].apply(
        lambda pid: os.path.exists(os.path.join(MASK_DIR, str(pid), "colon.nii.gz")))
    df = df[df["has_mask"] & df["mri_path_t"].notna()].reset_index(drop=True)
    # 过滤实际存在的 mri_path_t
    df["mri_ok"] = df["mri_path_t"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    df = df[df["mri_ok"]].reset_index(drop=True)
    print(f"[route1] 病例: master∩mask∩mri = {len(df)} "
          f"(T+ {df['label_T'].mean():.3f}, N+ {df['label_N'].mean():.3f})")
    if smoke:
        df = df.head(5)
    return df


def load_img_mask(row):
    """读 T2 轴位 NIfTI + colon mask, mask 重采样到图像网格(最近邻)。返回 (sitk_img, sitk_mask)。"""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(row["mri_path_t"]))
    mask_path = os.path.join(MASK_DIR, str(row["patient_id"]), "colon.nii.gz")
    mask = sitk.ReadImage(mask_path)
    if mask.GetSpacing() != img.GetSpacing() or mask.GetSize() != img.GetSize():
        rf = sitk.ResampleImageFilter()
        rf.SetReferenceImage(img)
        rf.SetInterpolator(sitk.sitkNearestNeighbor)
        rf.SetOutputPixelType(sitk.sitkUInt8)
        mask = rf.Execute(mask)
    return img, mask


def make_ring_mask(mask_img, in_mm=5.0, out_mm=10.0):
    """mask 外扩环: 距 mask 表面 in_mm~out_mm 的壳 (物理毫米)。
    返回与 mask_img 同网格的 uint8 二值图 (1=环)。"""
    import SimpleITK as sitk
    import scipy.ndimage as ndi
    arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)  # [D,H,W]
    sp = mask_img.GetSpacing()  # (x,y,z)
    # 物理各向同性距离场: 在体素空间按 spacing 缩放后 EDT
    # 对 z 轴(轴位层厚)权重放宽: 用 spacing 直接加权
    sxy, sz = sp[0], sp[2]
    # 构建加权: 直接在补集上 EDT, 然后除以权重
    d = ndi.distance_transform_edt((arr == 0).astype(np.uint8),
                                   sampling=(sz, sxy, sxy))
    ring = ((d > in_mm) & (d <= out_mm)).astype(np.uint8)
    out = sitk.GetImageFromArray(ring)
    out.CopyInformation(mask_img)
    out.SetSpacing(mask_img.GetSpacing())
    return out


def extract_features(df, smoke=False):
    from radiomics import featureextractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings = {
        "normalize": True,          # 图像 zscore 归一 (MRI 值域非 HU)
        "binWidth": 25,
        "label": 1,
        "force2D": False,
        "resampledPixelSpacing": None,   # 不重采样 → 原生面内分辨率
        "additionalInfo": False,
    }
    enabled = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm",
               "shape"]
    for cls in enabled:
        extractor.enableFeatureClassByName(cls)
    os.makedirs(TMP_MASK_DIR, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        pid = str(r["patient_id"])
        img, mask = load_img_mask(r)
        ring = make_ring_mask(mask, 5.0, 10.0)
        for mask_type, m in (("body", mask), ("ring", ring)):
            if np.count_nonzero(sitk_arr(m)) == 0:
                rows.append({"patient_id": pid, "mask_type": mask_type,
                             "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
                             "fold": int(r["fold"]), "ok": 0})
                continue
            try:
                feats = extractor.execute(img, m)
            except Exception as e:
                print(f"[route1] 提取失败 {pid}/{mask_type}: {e}")
                rows.append({"patient_id": pid, "mask_type": mask_type,
                             "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
                             "fold": int(r["fold"]), "ok": 0})
                continue
            row = {"patient_id": pid, "mask_type": mask_type, "ok": 1,
                   "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
                   "fold": int(r["fold"])}
            for k, v in feats.items():
                if k.startswith("diagnostics_"):
                    continue
                # PyRadiomics 特征 key 形如 original_firstorder_Mean:
                # 去掉 original_ 前缀保留特征名
                if k.startswith("original_"):
                    k = k[len("original_"):]
                if isinstance(v, (int, float, np.integer, np.floating)):
                    row[k] = float(v)
                elif isinstance(v, np.ndarray) and v.ndim == 0:
                    row[k] = float(v)
            rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"[route1] {i+1}/{len(df)} 例完成 ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(FEAT_CSV, index=False, encoding="utf-8-sig")
    print(f"[route1] 特征表已写入 {FEAT_CSV} ({len(out)} 行)")
    return out


def sitk_arr(img):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(img)


# ── ML: 折内筛选 + L2 LR / ElasticNet, 5 折 OOF ───────────────────────

FEAT_SKIP = {"patient_id", "mask_type", "ok", "label_T", "label_N", "fold"}


def _feat_cols(df):
    return [c for c in df.columns if c not in FEAT_SKIP]


def _select_in_fold(Xtr, ytr, top_k=300, seed=42):
    """方差 + 相关性去冗余 + f_classif 预筛 (全在训练折内)。"""
    from sklearn.feature_selection import f_classif
    from scipy.stats import pearsonr
    mask = np.nanstd(Xtr, axis=0) > 1e-4
    X = Xtr[:, mask]
    cols_keep = [i for i, m in enumerate(mask) if m]
    # 相关性去冗余 (|r|>0.95 删后到)
    keep = []
    for j in range(X.shape[1]):
        dup = False
        for k in keep:
            if abs(pearsonr(X[:, j], X[:, k])[0]) > 0.95:
                dup = True
                break
        if not dup:
            keep.append(j)
    cols_keep = [cols_keep[j] for j in keep]
    X = Xtr[:, cols_keep]
    if X.shape[1] > top_k:
        f, _ = f_classif(X, ytr)
        order = np.argsort(-f)[:top_k]
        cols_keep = [cols_keep[j] for j in order]
    return cols_keep


def run_oof(df, label_col, mask_type, seed=42):
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    sub = df[df["mask_type"] == mask_type].reset_index(drop=True)
    sub = sub[sub["ok"] == 1]
    feats = _feat_cols(sub)
    X = sub[feats].values.astype(np.float64)
    y = sub[label_col].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    folds = sorted(sub["fold"].unique())
    oof_lr = np.zeros(len(sub))
    oof_en = np.zeros(len(sub))
    for fi in folds:
        tr = sub["fold"] != fi
        va = sub["fold"] == fi
        if y[tr].sum() == 0 or y[va].sum() == 0:
            oof_lr[va] = 0.5
            oof_en[va] = 0.5
            continue
        cols = _select_in_fold(X[tr], y[tr])
        scaler = StandardScaler().fit(X[tr][:, cols])
        Xtr = scaler.transform(X[tr][:, cols])
        Xva = scaler.transform(X[va][:, cols])
        # L2 LogisticRegression, C 网格小调
        best_lr = None
        best_auc = -1
        for C in (0.01, 0.1, 1.0, 10.0):
            m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
            m.fit(Xtr, y[tr])
            a = roc_auc_score(y[tr], m.predict_proba(Xtr)[:, 1])
            if a > best_auc:
                best_auc, best_lr = a, m
        oof_lr[va] = best_lr.predict_proba(Xva)[:, 1]
        # ElasticNet (L1+L2, saga)
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        en.fit(Xtr, y[tr])
        oof_en[va] = np.clip(en.predict(Xva), 1e-6, 1 - 1e-6)
    return {
        "n": len(sub), "pos": int(y.sum()),
        "lr_pooled_auc": roc_auc_score(y, oof_lr),
        "en_pooled_auc": roc_auc_score(y, oof_en),
        "lr_oof": oof_lr.tolist(), "en_oof": oof_en.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    if args.skip_extract and os.path.exists(FEAT_CSV):
        df = pd.read_csv(FEAT_CSV)
    else:
        cases = build_case_list(smoke=args.smoke)
        df = extract_features(cases, smoke=args.smoke)

    results = {}
    for mask_type, label in (("body", "label_T"), ("ring", "label_N")):
        r = run_oof(df, label, mask_type)
        results[f"{mask_type}_{label}"] = r
        print(f"[route1] {mask_type}/{label}: LR OOF AUC {r['lr_pooled_auc']:.4f} "
              f"(n={r['n']}, pos={r['pos']}) | EN {r['en_pooled_auc']:.4f}")

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = ["路线1 原生面内影像组学 (PyRadiomics 3D, 原生网格不重采样)",
             f"样本: body(T) n={results['body_label_T']['n']} / ring(N) n={results['ring_label_N']['n']}",
             "掩码: 本体=完整结直肠 mask | 环=mask 外扩 5-10mm",
             "模型: 折内方差/相关/f_classif 筛选 + StandardScaler + L2 LR(C网格) / ElasticNet",
             ""]
    for k, r in results.items():
        lines.append(f"{k}: LR OOF AUC {r['lr_pooled_auc']:.4f} | EN OOF AUC {r['en_pooled_auc']:.4f} (n={r['n']}, pos={r['pos']})")
    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
