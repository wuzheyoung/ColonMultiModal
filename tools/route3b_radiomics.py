# -*- coding: utf-8 -*-
"""任务2(修正版): 真轴位 mri_c / mri_d 的 2D PyRadiomics。
- 分序列保真 spacing (实测中位数):
  - mri_c: 面内中位数 0.625mm (p25-p75 0.625-0.742) → resampledPixelSpacing=[0.625,0.625,0]
  - mri_d: 面内中位数 1.258mm, p25-p75 1.25-1.367 高度一致 → resampledPixelSpacing=None 跳过重采样
  - 厚层轴一律 0 (保留原生层厚, 严禁插值); force2Ddimension=0 (真轴位 LPS, 数组 axis0=z=厚层轴)
- preCrop=True, padDistance=5, interpolator='sitkLinear', normalize=True, binCount=32
- ProcessPoolExecutor 8 worker (ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 + OMP_NUM_THREADS=1)
- ROI: body (测 T) / ring (测 N, 5-10mm EDT 壳)
- ML: 折内筛选+StandardScaler+L2 LR(C网格)/ElasticNet, 5折 OOF, 全队列/纯直接手术组 T/N
用法: python tools/route3b_radiomics.py --seq c|d [--smoke] [--skip-extract] [--only-extract]
输出: logs/route3b_{seq}_features.csv + logs/audit_tgr_task2c_rad_{seq}.json
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MASTER_CSV = "data/master_5fold_356.csv"
PAIRS_RAW = "data/hos_pairs_raw.csv"
X929 = "data/Hos/结直肠癌多模态患者数据收集表9.29.xlsx"
XFULL = "data/Hos/结直肠癌多模态患者数据收集表_完整版.xlsx"
MASK_DIR = {"c": "data/masks_totalseg_mr_c", "d": "data/masks_totalseg_mr_d"}
PATH_COL = {"c": "mri_path_c", "d": "mri_path_d"}
# 分序列目标面内 spacing (实测中位数; d 高度一致 → None 跳过重采样)
TARGET_XY = {"c": [0.625, 0.625, 0], "d": None}


def load_tgr_map():
    d929 = pd.read_excel(X929)
    dfull = pd.read_excel(XFULL)
    m = {}
    for _, r in d929.iterrows():
        if pd.notna(r["病理TGR"]):
            try:
                m[int(r["编号"]) + 200] = float(r["病理TGR"])
            except (ValueError, TypeError):
                pass
    for i, r in dfull.iterrows():
        if pd.notna(r["病理TGR"]):
            try:
                m[i + 1] = float(r["病理TGR"])
            except (ValueError, TypeError):
                pass
    return m


def _load_img_mask(row, seq):
    import SimpleITK as sitk
    mask_dir = os.path.join(ROOT, MASK_DIR[seq], str(int(row["patient_id"])))
    mask_p = os.path.join(mask_dir, "colon.nii.gz")
    if not os.path.exists(mask_p):
        return None, None
    img = sitk.ReadImage(row[PATH_COL[seq]])
    mask = sitk.ReadImage(mask_p)
    # 对齐 mask 到图像网格 (最近邻)
    ref = sitk.ResampleImageFilter()
    ref.SetReferenceImage(img)
    ref.SetInterpolator(sitk.sitkNearestNeighbor)
    ref.SetOutputPixelType(sitk.sitkUInt8)
    mask = ref.Execute(mask)
    return img, mask


def _make_ring_mask(mask_img, in_mm=5.0, out_mm=10.0):
    import SimpleITK as sitk
    import scipy.ndimage as ndi
    arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    sp = mask_img.GetSpacing()
    sxy, sz = sp[0], sp[2]
    d = ndi.distance_transform_edt((arr == 0).astype(np.uint8), sampling=(sz, sxy, sxy))
    ring = ((d > in_mm) & (d <= out_mm)).astype(np.uint8)
    out = sitk.GetImageFromArray(ring)
    out.CopyInformation(mask_img)
    return out


def _extract_one(args):
    """单例单 mask 提取 (worker 函数, 各自创建 extractor)"""
    row_dict, seq, mname, target_xy = args
    import SimpleITK as sitk
    from radiomics import featureextractor
    row = pd.Series(row_dict)
    try:
        img, mask = _load_img_mask(row, seq)
        if img is None or mask is None:
            return {"patient_id": int(row["patient_id"]), "mask_type": mname, "ok": 0,
                    "label_T": int(row["label_T"]), "label_N": int(row["label_N"]),
                    "fold": int(row["fold"])}
        m = mask if mname == "body" else _make_ring_mask(mask)
        _m_arr = sitk.GetArrayFromImage(m)
        # 注意: mask 全零时 EDT 会产出"背景壳"伪环, 必须前置判空
        _src_arr = sitk.GetArrayFromImage(mask)
        if _m_arr.sum() == 0 or _m_arr.ndim < 2 or _src_arr.sum() == 0:
            return {"patient_id": int(row["patient_id"]), "mask_type": mname, "ok": 0,
                    "label_T": int(row["label_T"]), "label_N": int(row["label_N"]),
                    "fold": int(row["fold"])}
        extractor = featureextractor.RadiomicsFeatureExtractor()
        settings = {
            "normalize": True,
            "binCount": 32,
            "force2D": True,
            "force2Ddimension": 0,
            "preCrop": True,
            "padDistance": 5,
            "interpolator": "sitkLinear",
            "correctMask": True,
            "label": 1,
        }
        if target_xy is not None:
            settings["resampledPixelSpacing"] = target_xy
        extractor.settings = settings
        for cls in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"):
            extractor.enableFeatureClassByName(cls)
        feats = extractor.execute(img, m)
        r = {"patient_id": int(row["patient_id"]), "mask_type": mname, "ok": 1,
             "label_T": int(row["label_T"]), "label_N": int(row["label_N"]), "fold": int(row["fold"])}
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                continue
            kk = k[len("original_"):] if k.startswith("original_") else k
            if isinstance(v, (int, float, np.integer, np.floating)):
                r[kk] = float(v)
            elif isinstance(v, np.ndarray) and v.ndim == 0:
                r[kk] = float(v)
        return r
    except Exception as e:
        return {"patient_id": int(row["patient_id"]), "mask_type": mname, "ok": 0,
                "label_T": int(row["label_T"]), "label_N": int(row["label_N"]),
                "fold": int(row["fold"]), "err": str(e)}


def extract_features(df, seq, smoke=False, workers=8):
    rows = []
    t0 = time.time()
    tasks = []
    for _, row in df.iterrows():
        for mname in ("body", "ring"):
            tasks.append((row.to_dict(), seq, mname, TARGET_XY[seq]))
    if smoke:
        tasks = tasks[:10]
        for t in tasks:
            rows.append(_extract_one(t))
        print(f"[route3b-{seq}] smoke {len(tasks)} tasks done ({time.time()-t0:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(_extract_one, tasks, chunksize=8)):
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f"[route3b-{seq}] {i+1}/{len(tasks)} tasks done ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    return out, time.time() - t0


def run_ml(df, seq, feats_cols, target="T", group="all"):
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr
    from sklearn.feature_selection import f_classif
    d = df[df["mask_type"] == "body" if target == "T" else df["mask_type"] == "ring"]
    d = d[d["ok"] == 1]
    ycol = "label_T" if target == "T" else "label_N"
    if group == "direct":
        tgr = load_tgr_map()
        d = d[~d["patient_id"].isin(tgr)]
    X = d[feats_cols].values.astype(float)
    y = d[ycol].values.astype(int)
    folds = d["fold"].values.astype(int)
    n, pos = len(y), int(y.sum())
    res = {}
    for model_name, mk in (("LR", lambda C: LogisticRegression(C=C, max_iter=2000)),
                           ("EN", lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=2000))):
        oof = np.full(n, np.nan)
        for k in np.unique(folds):
            va, tr = folds == k, folds != k
            if tr.sum() < 2 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
                oof[va] = 0.5
                continue
            # 折内特征筛选
            Xtr, ytr, Xva = X[tr], y[tr], X[va]
            sel = np.ones(Xtr.shape[1], dtype=bool)
            var = Xtr.var(axis=0)
            sel &= var > 1e-4
            if sel.sum() < 3:
                sel[:] = True
            # 相关性去冗余
            Xs = Xtr[:, sel]
            drop = set()
            for i in range(Xs.shape[1]):
                for j in range(i + 1, Xs.shape[1]):
                    if np.abs(pearsonr(Xs[:, i], Xs[:, j])[0]) > 0.95:
                        drop.add(j)
            keep = [c for ii, c in enumerate(np.where(sel)[0]) if ii not in drop]
            if len(keep) < 3:
                keep = list(np.where(sel)[0])
            # f_classif top300
            try:
                F, _ = f_classif(Xtr[:, keep], ytr)
                order = np.argsort(-F)
                keep = [keep[i] for i in order[:min(300, len(keep))]]
            except Exception:
                pass
            keep = np.array(keep, dtype=int)
            sc = StandardScaler().fit(Xtr[:, keep])
            Xtr_s, Xva_s = sc.transform(Xtr[:, keep]), sc.transform(Xva[:, keep])
            if model_name == "LR":
                best_c, best_a = 0.01, -1
                for c in (0.01, 0.1, 1.0, 10.0):
                    m = mk(c).fit(Xtr_s, ytr)
                    a = roc_auc_score(ytr, m.predict_proba(Xtr_s)[:, 1])
                    if a > best_a:
                        best_a, best_c = a, c
                m = mk(best_c).fit(Xtr_s, ytr)
            else:
                best_a, best_alpha = -1, 0.005
                for a in (0.005, 0.01, 0.05, 0.1):
                    m = mk(a).fit(Xtr_s, ytr)
                    try:
                        a_ = roc_auc_score(ytr, m.predict(Xtr_s))
                    except Exception:
                        a_ = 0.5
                    if a_ > best_a:
                        best_a, best_alpha = a_, a
                m = mk(best_alpha).fit(Xtr_s, ytr)
                oof[va] = m.predict(Xva_s)
                continue
            oof[va] = m.predict_proba(Xva_s)[:, 1]
        try:
            auc = float(roc_auc_score(y, oof))
        except Exception:
            auc = 0.5
        folds_arr = np.asarray(folds)
        per_fold = []
        for k in np.unique(folds_arr):
            m_ = folds_arr == k
            try:
                per_fold.append(round(float(roc_auc_score(y[m_], oof[m_])), 4))
            except Exception:
                per_fold.append(0.5)
        res[model_name] = {"n": n, "pos": pos, "auc": round(auc, 4),
                           "fold_mean": round(float(np.mean(per_fold)), 4),
                           "fold_std": round(float(np.std(per_fold)), 4),
                           "per_fold": per_fold}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, choices=["c", "d"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--only-extract", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    seq = args.seq
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    master = pd.read_csv(MASTER_CSV)
    raw = pd.read_csv(PAIRS_RAW)
    df = master.merge(raw[["patient_id", PATH_COL[seq]]], on="patient_id", how="left")
    df = df[df[PATH_COL[seq]].notna() & df["patient_id"].isin(
        [int(f) for f in os.listdir(MASK_DIR[seq]) if f.isdigit()])]
    df = df.reset_index(drop=True)
    print(f"[route3b-{seq}] 病例: master∩mask∩mri = {len(df)} (T+ {df['label_T'].mean():.3f}, N+ {df['label_N'].mean():.3f})")
    print(f"[route3b-{seq}] target spacing: {TARGET_XY[seq]}")

    feat_csv = f"logs/route3b_{seq}_features.csv"
    if not args.skip_extract:
        fdf, secs = extract_features(df, seq, args.smoke, args.workers)
        fdf.to_csv(feat_csv, index=False, encoding="utf-8")
        print(f"[route3b-{seq}] 特征表已写入 {feat_csv} ({len(fdf)} 行, 耗时 {secs:.0f}s)")
    else:
        fdf = pd.read_csv(feat_csv)
    if args.only_extract:
        print(f"[route3b-{seq}] only-extract DONE")
        return

    feats_cols = [c for c in fdf.columns if c not in
                  ("patient_id", "mask_type", "ok", "label_T", "label_N", "fold", "err")]
    print(f"[route3b-{seq}] n_feat={len(feats_cols)}")
    out = {"seq": seq, "target_spacing": TARGET_XY[seq], "n_feat": len(feats_cols)}
    for target in ("T", "N"):
        for group in ("all", "direct"):
            key = f"{target}_{group}"
            out[key] = run_ml(fdf, seq, feats_cols, target, group)
            r = out[key]["LR"]
            print(f"[route3b-{seq}] {key} LR n={r['n']} pos={r['pos']} AUC {r['auc']} ({r['fold_mean']}±{r['fold_std']})")
            r2 = out[key]["EN"]
            print(f"[route3b-{seq}] {key} EN n={r2['n']} pos={r2['pos']} AUC {r2['auc']} ({r2['fold_mean']}±{r2['fold_std']})")
    out_json = f"logs/audit_tgr_task2c_rad_{seq}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"DONE → {out_json}")


if __name__ == "__main__":
    main()
