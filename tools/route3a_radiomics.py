# -*- coding: utf-8 -*-
"""任务3a: PyRadiomics 参数修复重跑 (force2D + 0.5mm 面内重采样 + binCount=32, 全肠 vs 15cm 直肠段)。
修正 route1 的 binWidth=25 桶坍塌 (normalize zscore 后值域±4, binWidth 25 → 几乎 1 个 bin)。
- force2D=True, force2Ddimension=2 (mri_t 为矢状位 PIR, 面内=P×I, 厚层轴=R=axis2, 沿厚层轴切片)
- resampledPixelSpacing=[0.5,0.5,0] (面内各向同性 0.5mm, 0=厚层轴保持原值, 严禁插值)
- binCount=32 (替代 binWidth)
- ROI: (a) 完整 body/ring; (b) 沿真实头足轴自掩码尾侧端向上截 150mm 直肠段 (兼容 LPS 个案 pid=1)
ML: 折内筛选+StandardScaler+L2 LR(C网格)/ElasticNet, 5折 OOF, 全队列/直接手术(无tgr)/放化疗(有tgr) 三群组 T/N。
输出 logs/route3a_radiomics_features.csv + logs/audit_tgr_task3a.json + README 追加。
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MASTER_CSV = "data/master_5fold_356.csv"
PAIRS_RAW = "data/hos_pairs_raw.csv"
MASK_DIR = "data/masks_totalseg_mr_t"
FEAT_CSV = "logs/route3a_radiomics_features.csv"
RESULT_JSON = "logs/audit_tgr_task3a.json"
X929 = "data/Hos/结直肠癌多模态患者数据收集表9.29.xlsx"
XFULL = "data/Hos/结直肠癌多模态患者数据收集表_完整版.xlsx"


def load_tgr_map():
    import pandas as pd
    d929 = pd.read_excel(X929)
    dfull = pd.read_excel(XFULL)
    m = {}
    for _, r in d929.iterrows():
        if pd.notna(r["病理TGR"]):
            m[int(r["编号"]) + 200] = r["病理TGR"]
    for i, r in dfull.iterrows():
        if pd.notna(r["病理TGR"]):
            m[i + 1] = r["病理TGR"]
    return m


def build_case_list(smoke=False):
    master = pd.read_csv(MASTER_CSV)
    pairs = pd.read_csv(PAIRS_RAW)
    p2 = pairs[["patient_id", "mri_path_t"]].drop_duplicates("patient_id")
    df = master.merge(p2, on="patient_id", how="left")
    df["has_mask"] = df["patient_id"].apply(
        lambda pid: os.path.exists(os.path.join(MASK_DIR, str(pid), "colon.nii.gz")))
    df = df[df["has_mask"] & df["mri_path_t"].notna()]
    df["mri_ok"] = df["mri_path_t"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    df = df[df["mri_ok"]].reset_index(drop=True)
    print(f"[route3a] 病例: master∩mask∩mri = {len(df)} "
          f"(T+ {df['label_T'].mean():.3f}, N+ {df['label_N'].mean():.3f})")
    if smoke:
        df = df.head(5)
    return df


def load_img_mask(row):
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


def crop_rectum150(mask_img, headfoot_axis, axial_dir):
    """自掩码尾侧端沿头足轴向上截 150mm。mask_img SITK 数组 [z,y,x] (SITK axis0=z)。
    headfoot_axis: 头足轴在 SITK 数组中的索引 (mri_t PIR → 轴1=I 在 SITK 中? 见 _hf_axis)。"""
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)
    if arr.sum() == 0:
        return mask_img  # 空 mask 原样返回, 由调用方 ok=0 处理
    sp = mask_img.GetSpacing()
    hf = headfoot_axis  # SITK 数组轴: 0=z, 1=y, 2=x
    nz = np.nonzero(arr)
    lo = nz[hf].min()
    hi = nz[hf].max()
    thick = sp[2] if hf == 0 else sp[0]  # SITK z=sp[2], y/x=sp[0]/sp[1]
    if hf == 1:
        thick = sp[1]
    elif hf == 2:
        thick = sp[0]
    mm150 = int(round(150.0 / thick))
    # axial_dir: +1 = 数组索引增大方向朝尾侧(尾侧在 hi, 截 [hi-mm150, hi]); -1 = 尾侧在 lo, 截 [lo, lo+mm150]
    if axial_dir > 0:
        sl = [slice(None)] * 3
        sl[hf] = slice(max(0, hi - mm150), hi + 1)
    else:
        sl = [slice(None)] * 3
        sl[hf] = slice(lo, min(arr.shape[hf], lo + mm150))
    idx = np.zeros_like(arr, dtype=bool)
    idx[tuple(sl)] = True
    arr2 = np.where(idx, arr, 0)
    out = sitk.GetImageFromArray(arr2)
    out.CopyInformation(mask_img)
    return out


def hf_axis_for(img):
    """返回 (SITK 数组头足轴索引, 轴向方向符号)。
    mri_t 主规格 axcodes=PIR: SITK 数组 [z,y,x] = nibabel [x,y,z] 逆序。
    nibabel axcodes PIR → 轴0=P, 轴1=I, 轴2=R。SITK 数组轴0=nib 轴2(R), 轴1=nib 轴1(I), 轴2=nib 轴0(P)。
    故头足轴 I 在 SITK 数组 = 轴1。LPS 个案: nib 轴2=S → SITK 数组 = 轴0。"""
    import nibabel as nib
    import SimpleITK as sitk
    # 用 sitk 的 direction 判断更稳: 读图像 direction, 找到 S/I 分量最大的轴
    d = img.GetDirection()  # 9 元素 row-major (x行)
    # 世界坐标列向量 = direction 矩阵列
    # 头足 = S(上)/I(下): 世界 z 分量
    col_s = [d[6], d[7], d[8]]  # 世界 z 分量 × nib 轴 0,1,2
    # SITK 数组轴0=世界x? 需换算: SITK (z,y,x) 存储, direction 按 (x,y,z) 行
    # nibabel 轴 (0,1,2) = (x,y,z); SITK 数组轴 (0,1,2) = (z,y,x)
    # 世界 z 分量对应 nib 轴2 = SITK 数组轴0
    hf = 0
    axial_dir = 1  # 数组索引增大方向朝尾侧?
    # nib 轴2 世界 z 分量 >0 → 朝 S(上); 数组索引增大=世界z增大=朝头侧 → 尾侧在 lo
    wz2 = col_s[2]
    if wz2 > 0:
        axial_dir = -1  # 数组增大朝头侧, 尾侧在 lo
    else:
        axial_dir = 1   # 数组增大朝尾侧(I), 尾侧在 hi
    # 也核对 nib 轴1 (I) 是否存在更极端分量 (PIR 主规格轴1=I)
    wz1 = col_s[1]
    if abs(wz1) > abs(wz2):
        hf = 1  # SITK 数组轴1 = nib 轴1
        axial_dir = 1 if wz1 < 0 else -1  # nib 轴1 世界 z<0 → I(下) → 数组增大朝尾侧
    return hf, axial_dir


def extract_features(df, smoke=False):
    from radiomics import featureextractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings = {
        "normalize": True,
        "binCount": 32,               # 修复: 替代 binWidth=25 桶坍塌
        "label": 1,
        "force2D": True,              # 2D 特征
        "force2Ddimension": 2,        # 沿厚层轴(R=axis2)切片, 面=P×I 矢状面
        "resampledPixelSpacing": [0.5, 0.5, 0],  # 面内 0.5mm, 厚层轴 0=保持原值
        "additionalInfo": False,
    }
    for cls in ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"]:
        extractor.enableFeatureClassByName(cls)
    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        pid = str(r["patient_id"])
        img, mask = load_img_mask(r)
        ring = make_ring_mask(mask, 5.0, 10.0)
        hf, adir = hf_axis_for(img)
        mask_r = crop_rectum150(mask, hf, adir)
        ring_r = crop_rectum150(ring, hf, adir)
        for mask_type, m in (("body", mask), ("ring", ring), ("body_150", mask_r), ("ring_150", ring_r)):
            import SimpleITK as sitk
            if np.count_nonzero(sitk.GetArrayFromImage(m)) == 0:
                rows.append({"patient_id": pid, "mask_type": mask_type,
                             "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
                             "fold": int(r["fold"]), "ok": 0})
                continue
            try:
                feats = extractor.execute(img, m)
            except Exception as e:
                print(f"[route3a] 提取失败 {pid}/{mask_type}: {e}")
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
                if k.startswith("original_"):
                    k = k[len("original_"):]
                if isinstance(v, (int, float, np.integer, np.floating)):
                    row[k] = float(v)
                elif isinstance(v, np.ndarray) and v.ndim == 0:
                    row[k] = float(v)
            rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[route3a] {i+1}/{len(df)} 例完成 ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(FEAT_CSV, index=False, encoding="utf-8-sig")
    print(f"[route3a] 特征表已写入 {FEAT_CSV} ({len(out)} 行)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only-extract", action="store_true")
    args = ap.parse_args()

    df = build_case_list(args.smoke)
    if not args.skip_extract:
        extract_features(df, args.smoke)
        if args.only_extract:
            return
    feats_df = pd.read_csv(FEAT_CSV)
    feats_df["patient_id"] = feats_df["patient_id"].astype(int)
    tgr_map = load_tgr_map()
    feats_df["tgr"] = feats_df["patient_id"].map(tgr_map)
    feats_df["grp"] = np.where(feats_df["tgr"].notna(), "ncrt", "direct")
    feat_cols = [c for c in feats_df.columns if c not in {
        "patient_id", "mask_type", "ok", "label_T", "label_N", "fold", "tgr", "grp"}]

    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    def oof(X, y, folds, model="lr"):
        oofv = np.zeros(len(y))
        fa = []
        for fi in sorted(set(folds)):
            tr, va = folds != fi, folds == fi
            if y[tr].sum() == 0 or y[va].sum() == 0 or y[va].sum() == len(y[va]):
                oofv[va] = 0.5
                fa.append(0.5)
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
            best, ba = None, -1
            if model == "lr":
                for C in (0.01, 0.1, 1.0, 10.0):
                    mm = LogisticRegression(C=C, max_iter=3000, solver="lbfgs").fit(Xtr, y[tr])
                    a = roc_auc_score(y[tr], mm.predict_proba(Xtr)[:, 1])
                    if a > ba:
                        ba, best = a, mm
            else:
                for a_, l1 in ((0.005, 0.5), (0.01, 0.5), (0.05, 0.5), (0.1, 0.5)):
                    mm = ElasticNet(alpha=a_, l1_ratio=l1, max_iter=5000).fit(Xtr, y[tr])
                    try:
                        a = roc_auc_score(y[tr], mm.predict(Xtr))
                    except Exception:
                        a = -1
                    if a > ba:
                        ba, best = a, mm
            if best is None:
                oofv[va] = 0.5
                fa.append(0.5)
                continue
            oofv[va] = best.predict_proba(Xva)[:, 1] if model == "lr" else best.predict(Xva)
            try:
                fa.append(roc_auc_score(y[va], oofv[va]))
            except ValueError:
                fa.append(0.5)
        try:
            return roc_auc_score(y, oofv), fa
        except ValueError:
            return 0.5, fa

    def filt_feats(X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # 方差>1e-4
        v = X.std(axis=0)
        keep = v > 1e-4
        if keep.sum() == 0:
            keep = np.ones(X.shape[1], bool)
        return X[:, keep]

    out = {}
    for mt in ("body", "ring", "body_150", "ring_150"):
        sub = feats_df[(feats_df["mask_type"] == mt) & (feats_df["ok"] == 1)].copy()
        if len(sub) < 20:
            print(f"[{mt}] 样本不足 {len(sub)}, 跳过")
            continue
        X0 = sub[feat_cols].values.astype(np.float64)
        X0 = np.nan_to_num(X0, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"\n== {mt}: n={len(sub)} feats={X0.shape[1]} ==")
        out[mt] = {}
        for tgt, col in (("T", "label_T"), ("N", "label_N")):
            y = sub[col].values.astype(int)
            for gname, gm in (("all", np.ones(len(sub), bool)),
                              ("direct", sub["grp"].values == "direct"),
                              ("ncrt", sub["grp"].values == "ncrt")):
                if gm.sum() < 8 or y[gm].sum() == 0 or y[gm].sum() == len(y[gm]):
                    continue
                Xg = filt_feats(X0[gm])
                for mdl, nm in (("lr", "LR"), ("en", "EN")):
                    a, fa = oof(Xg, y[gm], sub["fold"].values[gm], mdl)
                    out[mt][f"{tgt}_{gname}_{nm}"] = {
                        "n": int(gm.sum()), "pos": int(y[gm].sum()),
                        "auc": round(float(a), 4),
                        "fold_mean": round(float(np.mean(fa)), 4),
                        "fold_std": round(float(np.std(fa)), 4)}
                    print(f"  [{tgt}|{gname}|{nm}] n={gm.sum()} pos={int(y[gm].sum())} "
                          f"AUC {a:.4f} ({np.mean(fa):.4f}±{np.std(fa):.4f})")
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("\nDONE → logs/audit_tgr_task3a.json")


if __name__ == "__main__":
    main()
