# -*- coding: utf-8 -*-
"""任务2c: 真轴位序列(mri_c/mri_d)冻结 ConvNeXt-Tiny 快速探针。
轴位切片逐层以 mask 质心裁 120mm 物理正方形(无拉伸), 冻结 convnext_tiny 提 [768],
MeanPool + 5折 L2 LR 测 sex / T / N(全队列+直接手术组+放化疗组) + TRG。
"""
import os, json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
MASTER = os.path.join(ROOT, "data", "master_5fold_356.csv")
LOG = os.path.join(ROOT, "logs")
MASK_C = os.path.join(ROOT, "data", "masks_totalseg_mr_c")
MASK_D = os.path.join(ROOT, "data", "masks_totalseg_mr_d")
OUT = os.path.join(LOG, "route2_cd_feats")
os.makedirs(OUT, exist_ok=True)

SEQ_COLS = {"c": ("mri_path_c", MASK_C), "d": ("mri_path_d", MASK_D)}
SIZE_MM = 120.0
IMG_SIZE = 224


def load_tgr_map():
    d929 = pd.read_excel(os.path.join(ROOT, "data", "Hos", "结直肠癌多模态患者数据收集表9.29.xlsx"))
    dfull = pd.read_excel(os.path.join(ROOT, "data", "Hos", "结直肠癌多模态患者数据收集表_完整版.xlsx"))
    m = {}
    for _, r in d929.iterrows():
        if pd.notna(r["病理TGR"]):
            m[int(r["编号"]) + 200] = r["病理TGR"]
    for i, r in dfull.iterrows():
        if pd.notna(r["病理TGR"]):
            m[i + 1] = r["病理TGR"]
    return m


def extract_case_feats(seq, pid, mri_path, model, device):
    """逐轴位层: mask 质心裁 120mm 正方形 → convnext [768]; 返回 mean 向量。"""
    mask_dir = SEQ_COLS[seq][1]
    mpath = os.path.join(mask_dir, str(pid), "colon.nii.gz")
    if not os.path.exists(mpath):
        return None
    mimg = sitk.ReadImage(mpath)
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()  # (x,y,z) 毫米
    msp = mimg.GetSpacing()
    # mask 重采样到 MRI 网格(最近邻)
    mres = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(),
                         sitk.sitkNearestNeighbor, vimg.GetOrigin(),
                         vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
    ma = sitk.GetArrayFromImage(mres)  # [z,y,x]
    va = sitk.GetArrayFromImage(vimg)
    # 轴位层: z 维为层; 每层 2D 面内 = y,x
    feats = []
    model.eval()
    with torch.no_grad():
        for z in range(ma.shape[0]):
            m2 = ma[z]
            if m2.sum() == 0:
                continue
            ys, xs = np.nonzero(m2)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            # 物理 120mm → 像素半径
            px, py = sp[0], sp[1]
            half = int(round(SIZE_MM / 2 / max(px, py)))
            # 保真不拉伸: 用最大各向同性半径
            ry = int(round(SIZE_MM / 2 / py))
            rx = int(round(SIZE_MM / 2 / px))
            y0, y1 = max(0, cy - ry), min(va.shape[1], cy + ry)
            x0, x1 = max(0, cx - rx), min(va.shape[2], cx + rx)
            if y1 - y0 < 4 or x1 - x0 < 4:
                continue
            crop = va[z, y0:y1, x0:x1].astype(np.float32)
            # pad 到正方形(边缘不足)
            H, W = crop.shape
            S = max(H, W)
            pad_img = np.zeros((S, S), dtype=np.float32)
            pad_img[:H, :W] = crop
            # 归一化 (zscore)
            m_ = pad_img.mean()
            s_ = pad_img.std() + 1e-6
            pad_img = (pad_img - m_) / s_
            # resize 224
            from PIL import Image
            im = Image.fromarray(pad_img)
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(im).astype(np.float32))[None, None]  # 1,1,224,224
            t = t.repeat(1, 3, 1, 1).to(device)
            f = model(t).cpu().numpy().ravel()  # [768]
            feats.append(f)
    if not feats:
        return None
    return np.stack(feats).mean(axis=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = None
    if not args.skip_extract:
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    tgr_map = load_tgr_map()
    raw = pd.read_csv(RAW)
    master = pd.read_csv(MASTER)
    meta = master[["patient_id", "fold", "label_T", "label_N"]].copy()
    meta["tgr"] = meta["patient_id"].map(tgr_map)
    meta["grp"] = np.where(meta["tgr"].notna(), "ncrt", "direct")
    sex_map = raw.set_index("patient_id")["sex"].to_dict()

    out = {}
    for seq in ("c", "d"):
        col, _ = SEQ_COLS[seq]
        cases = raw[raw[col].notna()][["patient_id", col]].merge(meta, on="patient_id", how="inner")
        print(f"== mri_{seq}: {len(cases)} 例 ==")
        if args.skip_extract and os.path.exists(os.path.join(OUT, f"{seq}.npy")):
            X = np.load(os.path.join(OUT, f"{seq}.npy"))
            ys_pid = pd.read_csv(os.path.join(OUT, f"{seq}_pids.csv"))["patient_id"].astype(int).tolist()
        else:
            if model is None:
                model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
                model = model.to(device).eval()
                for p in model.parameters():
                    p.requires_grad_(False)
            Xs, ys_pid = [], []
            for _, r in cases.iterrows():
                pid = int(r["patient_id"])
                f = extract_case_feats(seq, pid, r[col], model, device)
                if f is None:
                    continue
                Xs.append(f)
                ys_pid.append(pid)
            X = np.stack(Xs)
            np.save(os.path.join(OUT, f"{seq}.npy"), X)
            pd.DataFrame({"patient_id": ys_pid}).to_csv(os.path.join(OUT, f"{seq}_pids.csv"), index=False)
        sub = cases[cases["patient_id"].isin(ys_pid)].set_index("patient_id").loc[ys_pid].reset_index()
        print(f"  有效 {len(sub)} 例 (T+ {int(sub['label_T'].sum())}, N+ {int(sub['label_N'].sum())})")

        def oof_lr(X, y, folds, C_grid=(0.01, 0.1, 1.0, 10.0)):
            oof = np.zeros(len(y))
            fa = []
            for fi in sorted(set(folds)):
                tr, va = folds != fi, folds == fi
                if y[tr].sum() == 0 or y[va].sum() == 0:
                    oof[va] = 0.5
                    fa.append(0.5)
                    continue
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                best, ba = None, -1
                for C in C_grid:
                    mm = LogisticRegression(C=C, max_iter=3000, solver="lbfgs").fit(Xtr, y[tr])
                    a = roc_auc_score(y[tr], mm.predict_proba(Xtr)[:, 1])
                    if a > ba:
                        ba, best = a, mm
                oof[va] = best.predict_proba(Xva)[:, 1]
                try:
                    fa.append(roc_auc_score(y[va], oof[va]))
                except ValueError:
                    fa.append(0.5)
            try:
                return roc_auc_score(y, oof), np.array(fa)
            except ValueError:
                return 0.5, np.array(fa)

        results = {}
        for tgt, col in (("sex", None), ("T", "label_T"), ("N", "label_N")):
            if col is None:
                y = np.array([sex_map.get(p, np.nan) for p in sub["patient_id"].values], dtype=float)
            else:
                y = sub[col].values.astype(int)
            if np.isnan(y).any():
                y = np.nan_to_num(y, nan=0)
            for gname, mask in (("all", np.ones(len(sub), bool)),
                                ("direct", (sub["grp"] == "direct").values),
                                ("ncrt", (sub["grp"] == "ncrt").values)):
                if mask.sum() < 8 or y[mask].sum() == 0 or y[mask].sum() == len(y[mask]):
                    continue
                auc, fa = oof_lr(X[mask], y[mask].astype(int), sub["fold"].values[mask])
                results[f"{tgt}_{gname}"] = {"n": int(mask.sum()), "pos": int(y[mask].sum()),
                                             "auc": round(float(auc), 4),
                                             "fold_mean": round(float(fa.mean()), 4),
                                             "fold_std": round(float(fa.std()), 4)}
                print(f"  [{tgt}|{gname}] n={mask.sum()} pos={int(y[mask].sum())} "
                      f"AUC {auc:.4f} ({fa.mean():.4f}±{fa.std():.4f})")
        # TRG 二分类 (放化疗组内, TRG<=1 vs 2-3)
        msub = sub[sub["grp"] == "ncrt"]
        if len(msub) >= 8:
            tt = pd.to_numeric(msub["tgr"], errors="coerce")
            valid = tt.notna() & (tt >= 1) & (tt <= 3)
            if valid.sum() >= 4:
                y_trg = (tt[valid] <= 1).astype(int).values
                idx = np.where((sub["grp"] == "ncrt").values)[0][valid.values]
                auc, fa = oof_lr(X[idx], y_trg, sub["fold"].values[idx])
                results["trg"] = {"n": int(valid.sum()), "pos_good": int(y_trg.sum()),
                                  "auc": round(float(auc), 4),
                                  "fold_mean": round(float(fa.mean()), 4),
                                  "fold_std": round(float(fa.std()), 4)}
                print(f"  [TRG|ncrt] n={valid.sum()} good={int(y_trg.sum())} AUC {auc:.4f}")
        out[seq] = results

    with open(os.path.join(LOG, "audit_tgr_task2c.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nDONE → logs/audit_tgr_task2c.json")


if __name__ == "__main__":
    main()
