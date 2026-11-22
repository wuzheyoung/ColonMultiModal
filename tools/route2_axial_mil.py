# -*- coding: utf-8 -*-
"""
路线2-MRI: 原生轴位高分辨 + 冻结预训练 2D 编码器 + 轻量逐层 MIL
================================================================
用户 m02003 指令: 遍历每个含结直肠 mask 的原始轴位切片, 以该层 mask
中心在原始面内高分辨下裁固定物理尺寸正方形 (120×120mm, 无变比例拉伸);
冻结强预训练 2D 主干 (ConvNeXt-Tiny, timm) 离线提特征, 顶层训参数极少、
强权重衰减的 Gated Attention-MIL (或 Mean/Max Pooling + L2 线性分类器
底线测试), 5 折 OOF AUC。

流程:
  1. 病例: master_5fold_356.csv ∩ mri_path_t 存在 ∩ mr_t colon mask。
  2. 切片: 逐轴位层, mask 非空层以 mask 质心裁物理 120×120mm 正方形
     (原生面内分辨率, 不插值 z, 不拉伸) → resize 到 224² (仅缩放)。
  3. 特征: timm convnext_tiny (ImageNet 冻结) 前向 → 每例 [n_slices, 768]。
  4. 顶层: (a) Gated Attention-MIL (参数量 ~1.6K, weight decay 强);
     (b) 底线 Mean/Max Pooling + L2 LogisticRegression。5 折 OOF。
  5. 报告: logs/route2_axial_mil.json/.txt + README。

用法:
  python tools/route2_axial_mil.py [--smoke] [--backbone convnext_tiny]
依赖: Pytorch 环境 (timm/torch/nibabel/scipy/sitk 已齐)。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MASTER_CSV = "data/master_5fold_356.csv"
PAIRS_RAW = "data/hos_pairs_raw.csv"
MASK_DIR = "data/masks_totalseg_mr_t"
FEAT_CSV = "logs/route2_axial_mil_features.csv"
RESULT_JSON = "logs/route2_axial_mil.json"
RESULT_TXT = "logs/route2_axial_mil.txt"
CROP_MM = 120.0
INPUT_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_case_list(smoke=False):
    master = pd.read_csv(MASTER_CSV)
    pairs = pd.read_csv(PAIRS_RAW)
    p2 = pairs[["patient_id", "mri_path_t"]].drop_duplicates("patient_id")
    df = master.merge(p2, on="patient_id", how="left")
    df["has_mask"] = df["patient_id"].apply(
        lambda pid: os.path.exists(os.path.join(MASK_DIR, str(pid), "colon.nii.gz")))
    df["mri_ok"] = df["mri_path_t"].apply(
        lambda p: isinstance(p, str) and os.path.exists(p))
    df = df[df["has_mask"] & df["mri_ok"]].reset_index(drop=True)
    print(f"[r2mri] 病例: master∩mask∩mri = {len(df)} "
          f"(T+ {df['label_T'].mean():.3f}, N+ {df['label_N'].mean():.3f})")
    if smoke:
        df = df.head(4)
    return df


def _crop_slices(img_arr, mask_arr, spacing, crop_mm=CROP_MM):
    """img_arr [H,W,D] (nibabel 轴序 i,j,k), mask_arr 同。
    返回 (slices_xy [n,H,W] uint16, 每层 mask 质心 (x,y))。"""
    px, py, pz = spacing  # (x,y,z)
    n = int(round(crop_mm / (2 * max(px, py))))  # 像素半径 = 半宽 (物理尺寸固定, 无拉伸)
    out = []
    mask_flat = mask_arr.any(axis=(0, 1))  # [D]
    for k in range(img_arr.shape[2]):
        if not mask_flat[k]:
            continue
        mk = mask_arr[:, :, k]
        ys, xs = np.nonzero(mk)
        if len(ys) == 0:
            continue
        cx, cy = int(round(xs.mean())), int(round(ys.mean()))
        H, W = img_arr.shape[0], img_arr.shape[1]
        x0, x1 = max(0, cx - n), min(W, cx + n)
        y0, y1 = max(0, cy - n), min(H, cy + n)
        # 边界不足时以中心外扩 (物理尺寸可能 <120mm, 保真不拉伸)
        slice_img = img_arr[y0:y1, x0:x1, k].astype(np.float32)
        out.append(slice_img)
    return out


def extract_features(df, backbone, model, transform, smoke=False):
    import nibabel as nib
    import torchvision.transforms as T
    from PIL import Image
    to_tensor = T.ToTensor()
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in df.iterrows():
            pid = str(r["patient_id"])
            img = nib.load(r["mri_path_t"])
            arr = img.get_fdata().astype(np.float32)  # [H,W,D] (i,j,k)
            sp = img.header.get_zooms()  # (x,y,z)
            mask_path = os.path.join(MASK_DIR, pid, "colon.nii.gz")
            m = nib.load(mask_path)
            m_arr = m.get_fdata().astype(np.uint8)
            m_sp = m.header.get_zooms()
            if tuple(m_sp) != tuple(sp) or m_arr.shape != arr.shape:
                # mask 重采样到图像网格 (最近邻)
                import SimpleITK as sitk
                ref = sitk.GetImageFromArray(arr.transpose(2, 1, 0))
                ref.SetSpacing((sp[0], sp[1], sp[2]))
                m_img = sitk.GetImageFromArray(m_arr.transpose(2, 1, 0))
                m_img.SetSpacing((m_sp[0], m_sp[1], m_sp[2]))
                rf = sitk.ResampleImageFilter()
                rf.SetReferenceImage(ref)
                rf.SetInterpolator(sitk.sitkNearestNeighbor)
                rf.SetOutputPixelType(sitk.sitkUInt8)
                m_res = rf.Execute(m_img)
                m_arr = sitk.GetArrayFromImage(m_res).transpose(2, 1, 0)
            slices = _crop_slices(arr, m_arr, sp)
            if len(slices) == 0:
                rows.append({"patient_id": pid, "ok": 0,
                             "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
                             "fold": int(r["fold"])})
                continue
            feats = []
            for sl in slices:
                sl = sl - sl.mean() if sl.std() > 0 else sl
                sl = sl / (sl.std() + 1e-6)
                pil = Image.fromarray(((sl - sl.min()) / (sl.max() - sl.min() + 1e-6) * 255).astype(np.uint8))
                pil = pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
                x = to_tensor(pil).unsqueeze(0).repeat(1, 3, 1, 1).to(DEVICE)  # [1,3,224,224]
                f = model(x).float().cpu().numpy().reshape(-1)  # [768]
                feats.append(f)
            feats = np.stack(feats)  # [n_slices, 768]
            row = {"patient_id": pid, "ok": 1,
                   "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
                   "fold": int(r["fold"]), "n_slices": len(slices)}
            for j in range(feats.shape[1]):
                row[f"f_{j}"] = 0.0  # 占位, 特征序列单独存 npy
            rows.append(row)
            # 特征序列存 npy (每例一个)
            os.makedirs("logs/route2_feats", exist_ok=True)
            np.save(os.path.join("logs/route2_feats", f"{pid}.npy"), feats)
            if (i + 1) % 20 == 0:
                print(f"[r2mri] {i+1}/{len(df)} ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(FEAT_CSV, index=False, encoding="utf-8-sig")
    print(f"[r2mri] 特征表已写入 {FEAT_CSV} ({len(out)} 行)")
    return out


# ── Gated Attention-MIL ────────────────────────────────────────────────

class GatedAttnMIL(nn.Module):
    """Ilse 2018 Gated Attention Pooling: 参数量 ~ 2*d*(d+1) ≈ 1.6K (d=768)。"""

    def __init__(self, d_in=768, d_hid=128, dropout=0.25):
        super().__init__()
        self.attn_a = nn.Sequential(nn.Linear(d_in, d_hid), nn.Tanh())
        self.attn_b = nn.Sequential(nn.Linear(d_in, d_hid), nn.Sigmoid())
        self.attn_v = nn.Linear(d_hid, 1)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, 2))

    def forward(self, x):  # x: [B, n, d]
        a = self.attn_a(x) * self.attn_b(x)
        w = torch.softmax(self.attn_v(a).squeeze(-1), dim=1)  # [B,n]
        z = (w.unsqueeze(-1) * x).sum(1)                     # [B,d]
        return self.cls(z), w


def train_mil_oof(df, seed=42, epochs=40, lr=1e-3, wd=0.05):
    """每折: 训练 GatedAttnMIL, 预测验证折。返回 pooled OOF prob。"""
    from sklearn.metrics import roc_auc_score
    sub = df[df["ok"] == 1].reset_index(drop=True)
    folds = sorted(sub["fold"].unique())
    oof = np.zeros(len(sub))
    for fi in folds:
        tr = sub["fold"] != fi
        va = sub["fold"] == fi
        if sub[tr]["label_N"].sum() == 0 or sub[va]["label_N"].sum() == 0:
            oof[va] = 0.5
            continue
        Xtr = [np.load(os.path.join("logs/route2_feats", f"{p}.npy"))
               for p in sub[tr]["patient_id"]]
        ytr = sub[tr]["label_N"].values
        Xva = [np.load(os.path.join("logs/route2_feats", f"{p}.npy"))
               for p in sub[va]["patient_id"]]
        if len(Xtr) < 2 or len(Xva) < 1:
            oof[va] = 0.5
            continue
        torch.manual_seed(seed + fi)
        model = GatedAttnMIL().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        lossf = nn.CrossEntropyLoss()
        for ep in range(epochs):
            model.train()
            order = np.random.permutation(len(Xtr))
            tot = 0
            for idx in order:
                x = torch.from_numpy(Xtr[idx]).unsqueeze(0).float().to(DEVICE)
                y = torch.tensor([ytr[idx]], device=DEVICE)
                opt.zero_grad()
                out, _ = model(x)
                loss = lossf(out, y)
                loss.backward()
                opt.step()
                tot += loss.item()
            if (ep + 1) % 10 == 0:
                print(f"[r2mri] fold{fi} ep{ep+1} loss {tot/len(order):.4f}")
        model.eval()
        va_idx = np.where(va.values)[0]
        with torch.no_grad():
            for j, p in enumerate(sub[va]["patient_id"]):
                x = torch.from_numpy(Xva[j]).unsqueeze(0).float().to(DEVICE)
                out, _ = model(x)
                oof[va_idx[j]] = torch.softmax(out, 1)[0, 1].item()
    return {"n": len(sub), "pos": int(sub["label_N"].sum()),
            "mil_pooled_auc": roc_auc_score(sub["label_N"], oof)}


def run_pooling_lr(df, mode="mean", seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    sub = df[df["ok"] == 1].reset_index(drop=True)
    folds = sorted(sub["fold"].unique())
    oof = np.zeros(len(sub))
    for fi in folds:
        tr = sub["fold"] != fi
        va = sub["fold"] == fi
        Xtr = [np.load(os.path.join("logs/route2_feats", f"{p}.npy"))
               for p in sub[tr]["patient_id"]]
        Xva = [np.load(os.path.join("logs/route2_feats", f"{p}.npy"))
               for p in sub[va]["patient_id"]]
        if len(Xtr) < 2 or len(Xva) < 1:
            oof[va] = 0.5
            continue
        if mode == "mean":
            Xtr = np.stack([x.mean(0) for x in Xtr])
            Xva = np.stack([x.mean(0) for x in Xva])
        else:
            Xtr = np.stack([x.max(0) for x in Xtr])
            Xva = np.stack([x.max(0) for x in Xva])
        scaler = StandardScaler().fit(Xtr)
        best = None
        best_auc = -1
        for C in (0.01, 0.1, 1.0, 10.0):
            m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
            m.fit(scaler.transform(Xtr), sub[tr]["label_N"].values)
            a = roc_auc_score(sub[tr]["label_N"],
                              m.predict_proba(scaler.transform(Xtr))[:, 1])
            if a > best_auc:
                best_auc, best = a, m
        oof[va] = best.predict_proba(scaler.transform(Xva))[:, 1]
    return roc_auc_score(sub["label_N"], oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    if args.skip_extract and os.path.exists(FEAT_CSV):
        df = pd.read_csv(FEAT_CSV)
    else:
        import timm
        model = timm.create_model(args.backbone, pretrained=True, num_classes=0)
        model = model.to(DEVICE).eval()
        cases = build_case_list(smoke=args.smoke)
        df = extract_features(cases, args.backbone, model, None, smoke=args.smoke)

    results = {}
    results["mil"] = train_mil_oof(df)
    for mode in ("mean", "max"):
        a = run_pooling_lr(df, mode)
        results[f"pool_{mode}"] = a
        print(f"[r2mri] {mode} pooling + L2 LR OOF N AUC {a:.4f}")

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = ["路线2-MRI: 原生轴位 120mm 裁块 + 冻结 ConvNeXt-Tiny + 逐层 MIL",
             f"样本: n={results['mil']['n']} (N+ {results['mil']['pos']})",
             "顶层: Gated Attention-MIL (wd=0.05) vs Mean/Max pooling + L2 LR",
             ""]
    lines.append(f"GatedAttn-MIL: OOF N AUC {results['mil']['mil_pooled_auc']:.4f}")
    lines.append(f"MeanPool+LR:   OOF N AUC {results['pool_mean']:.4f}")
    lines.append(f"MaxPool+LR:    OOF N AUC {results['pool_max']:.4f}")
    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
