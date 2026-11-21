# -*- coding: utf-8 -*-
# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""
CRCFound 特征判别力验证 (路径①, ~30min)
=======================================
用 CRCFound MAE 预训练 3D ViT (checkpoint-999.pth) 对 CT 提取全局特征,
再以经典 ML (LASSO-LR / HistGB) + DL 概率 stacking 评估 T/N 判别力。

输入规格 (官方): spacing (1,1,3), size (32,256,256), CT HU 窗 [-150,250]
特征: encoder 的 cls token [1024]
对比: 与 DL 基线 (R5 s0 ckpt) 的 T/N 概率 stacking

用法:
  python tools/crcfound_features.py --smoke          # 冒烟 (2 样本)
  python tools/crcfound_features.py                  # 全量
  python tools/crcfound_features.py --skip-extract   # 复用已有特征表
输出:
  logs/crcfound_full_features.csv   特征表 (patient_id/split/label_T/label_N + crc_0..1023)
  logs/crcfound_features.json/.txt  报告
"""
import os
import sys
import json
import time
import argparse
import hashlib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

CKPT = "./pretrained/CRCFound/checkpoint-999.pth"
CACHE_DIR = "./cache_crcfound"
OUT_CSV = "./logs/crcfound_full_features.csv"
REPORT_JSON = "./logs/crcfound_features.json"
REPORT_TXT = "./logs/crcfound_features.txt"
TRAIN_CSV = "./data/hos_full_train_s42.csv"
VAL_CSV = "./data/hos_full_val_s42.csv"
CKPT_T = "./checkpoints_hos_full/best_model_hos_full_s0.pth"
CKPT_N = "./checkpoints_hos_full/best_model_n_hos_full_s0.pth"


# ── 1. CRCFound encoder (vit_pretrain 结构, 无 timm 依赖) ──────────────

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # [B,N,H,D]
        q = q.transpose(1, 2)    # [B,H,N,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim=1024, num_heads=16, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """官方 vit_pretrain PatchEmbed: self.proj = Conv3d(...)。"""

    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, num_frames=32, tubelet_size=16):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size))
        self.num_patches = (num_frames // tubelet_size) * \
                           (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)             # [B, D, T', H', W']
        x = x.flatten(2).transpose(1, 2)   # [B, N, D]
        return x


class CRCFoundEncoder(nn.Module):
    """MAE encoder only: patch_embed + pos_embedding + cls_token + 24 blocks + norm。
    输入 [B,1,32,256,256] -> 输出 [B,513,1024] (cls 在 0 位)。"""

    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 num_frames=32, tubelet_size=16):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      num_frames=num_frames,
                                      tubelet_size=tubelet_size)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)            # [B, N, D]
        x = x + self.pos_embedding[:, 1:]
        x = torch.cat([self.cls_token + self.pos_embedding[:, :1], x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                # [B, 513, 1024]


def load_crcfound_encoder(ckpt, device):
    """加载 checkpoint['model'] (module.* 前缀), 只取 encoder 键。"""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_sd = sd["model"]
    model = CRCFoundEncoder()
    stripped = {k.replace("module.", ""): v for k, v in model_sd.items()}
    enc_keys = set(model.state_dict().keys())
    picked = {k: v for k, v in stripped.items() if k in enc_keys}
    missing = enc_keys - set(picked.keys())
    if missing:
        raise RuntimeError(f"encoder 缺失键: {sorted(missing)[:5]} ...")
    model.load_state_dict(picked, strict=True)
    model.to(device).eval()
    print(f"[crcfound] encoder 加载 {len(picked)}/{len(enc_keys)} 键 (strict OK)")
    return model


# ── 2. CT 预处理管线 (CRCFound 官方规格) ───────────────────────────────

def _build_crc_pipeline():
    from monai.transforms import (Compose, LoadImage, EnsureChannelFirst,
                                  Spacing, ScaleIntensityRange, CropForeground,
                                  Resize, NormalizeIntensity)
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Spacing(pixdim=(1.0, 1.0, 3.0), mode="bilinear"),
        ScaleIntensityRange(a_min=-150.0, a_max=250.0, b_min=0.0, b_max=1.0,
                            clip=True),
        CropForeground(),
        Resize(spatial_size=(32, 256, 256), mode="trilinear"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])


def _cache_key(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = "?"
    raw = f"crcfound-v1|{os.path.abspath(path)}|{mtime}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _load_crc_ct(ct_path, pipe):
    """读 CT -> [1,32,256,256] float32, 磁盘缓存 cache_crcfound。"""
    if not (isinstance(ct_path, str) and ct_path and os.path.exists(ct_path)):
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(ct_path)
    fp = os.path.join(CACHE_DIR, f"{key}.npy")
    if os.path.exists(fp):
        return np.load(fp)
    try:
        vol = pipe(ct_path)                  # [1,32,256,256]
        arr = np.ascontiguousarray(vol).astype(np.float32)
        np.save(fp, arr)
        return arr
    except Exception as e:
        print(f"[crcfound] 预处理失败 {ct_path}: {e}")
        return None


# ── 3. 特征提取 ─────────────────────────────────────────────────────────

def extract_features(train_csv, val_csv, out_csv, smoke=False):
    model = load_crcfound_encoder(CKPT, "cuda" if torch.cuda.is_available() else "cpu")
    pipe = _build_crc_pipeline()
    dfs = []
    for split, csv_path in (("train", train_csv), ("val", val_csv)):
        df = pd.read_csv(csv_path)
        if smoke:
            df = df.head(2)
        rows = []
        t0 = time.time()
        for _, r in df.iterrows():
            ct_path = r.get("ct_path", "")
            vol = _load_crc_ct(ct_path, pipe)
            has_ct = vol is not None
            if not has_ct:
                feats = np.zeros(1024, dtype=np.float32)
            else:
                x = torch.from_numpy(vol).unsqueeze(0).to(model.patch_embed.proj.weight.device)
                with torch.no_grad(), torch.autocast(device_type="cuda",
                                                     enabled=torch.cuda.is_available()):
                    latent = model(x)
                feats = latent[0, 0].float().cpu().numpy()  # cls token [1024]
            row = {"patient_id": str(r.get("patient_id", "")), "split": split,
                   "label_T": int(r["label_T"]), "label_N": int(r["label_N"]),
                   "has_ct": bool(has_ct)}
            for i, v in enumerate(feats):
                row[f"crc_{i}"] = float(v)
            rows.append(row)
        dfs.append(pd.DataFrame(rows))
        print(f"[extract] {split}: {len(rows)} 样本完成 ({time.time()-t0:.1f}s)")
    feats_df = pd.concat(dfs, ignore_index=True)
    feats_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[extract] 特征表已写入 {out_csv} ({len(feats_df)} 样本)")
    return feats_df


# ── 4. 建模 ─────────────────────────────────────────────────────────────

def _feat_columns(df):
    return [c for c in df.columns if c not in
            {"patient_id", "split", "label_T", "label_N", "has_ct"}]


def load_dl_probs_with_ids(train_csv, val_csv, ckpt_t, ckpt_n):
    """复用 radiomics_experiment.load_dl_probs, 但额外返回 patient_id 供对齐。"""
    from radiomics_experiment import load_dl_probs
    base = load_dl_probs(train_csv, val_csv, ckpt_t, ckpt_n)
    # 重新用 build_dataloaders 拿 patient_id (与 load_dl_probs 相同数据顺序)
    from config.hos import HosConfig
    from data_loader.loader import build_dataloaders
    import torch
    cfg = HosConfig()
    cfg.backbone_name = "medicalnet18"
    cfg.use_augmentation = False
    cfg.device = "cpu"
    out = {}
    for split, csv in (("train", train_csv), ("val", val_csv)):
        _, loader = build_dataloaders(cfg, csv, csv)
        ids = []
        for batch in loader:
            ids.extend(str(p) for p in batch["patient_id"])
        out[split] = {"patient_id": ids, **base[split]}
    return out


def _select_feats(df, feats, label_col, top_k=300, seed=42):
    """f_classif 预筛 (train 内), 防 saga 在 1024 维上过慢。"""
    from sklearn.feature_selection import f_classif
    tr = df[df.split == "train"]
    X = tr[feats].values.astype(np.float64)
    y = tr[label_col].values
    f, _ = f_classif(X, y)
    order = np.argsort(-f)
    return [feats[i] for i in order[:top_k]]


def run_ml(df, label_col, feats, n_splits=5, seed=42):
    from radiomics_experiment import run_ml as _run_ml
    return _run_ml(df, label_col, feats, n_splits=n_splits, seed=seed)


def run_stacking(df, dl, label_col, base_feats, seed=42):
    from radiomics_experiment import run_stacking as _run_stacking
    return _run_stacking(df, dl, label_col, base_feats, seed=seed)


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    if args.skip_extract and os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV)
        if "has_ct" not in df.columns:
            df["has_ct"] = df[[f"crc_{i}" for i in range(1024)]].abs().sum(axis=1) > 1e-9
    else:
        df = extract_features(TRAIN_CSV, VAL_CSV, OUT_CSV, smoke=args.smoke)

    crc_feats = [c for c in _feat_columns(df) if c.startswith("crc_")]

    # 只用有 CT 的样本评估 (全 0 特征行会污染 ML)
    df_ct = df[df["has_ct"]].reset_index(drop=True)
    n_skip = len(df) - len(df_ct)
    print(f"[main] 总样本 {len(df)}, CT 可用 {len(df_ct)} (跳过 {n_skip} 无CT)")

    results = {}
    if not args.smoke and len(df_ct) >= 20:
        dl = load_dl_probs_with_ids(TRAIN_CSV, VAL_CSV, CKPT_T, CKPT_N)
        # 对齐 dl 到 df_ct 的 patient_id 顺序 (保持 numpy 数组供 run_stacking)
        import numpy as _np
        for split in ("train", "val"):
            sub = df_ct[df_ct.split == split]
            ids = [str(i) for i in sub["patient_id"]]
            d = dl[split]
            idx = [d["patient_id"].index(i) for i in ids]
            dl[split] = {k: _np.asarray(d[k])[idx]
                         for k in d if k != "patient_id"}
            dl[split]["patient_id"] = ids
        for label in ("label_T", "label_N"):
            sel = _select_feats(df_ct, crc_feats, label)
            ml = run_ml(df_ct, label, sel)   # {lasso_lr: {...}, histgb: {...}}
            for mname, mres in ml.items():
                results[f"{mname}_{label[-1]}"] = mres
            results[f"stack_{label[-1]}"] = run_stacking(df_ct, dl, label, sel)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": results, "n_feats": len(crc_feats),
                   "n_samples": len(df_ct), "n_skip": n_skip,
                   "smoke": args.smoke}, f, indent=2, ensure_ascii=False)

    lines = ["CRCFound 特征判别力验证 (路径①, 仅 CT 可用样本)",
             f"特征: {len(crc_feats)} 维 cls token (预筛≤300) | CT样本: {len(df_ct)} (跳过 {n_skip}) | smoke={args.smoke}", ""]
    for k, v in results.items():
        lines.append(
            f"{k}: CV-F1 {v.get('cv_macro_f1', float('nan')):.4f}±{v.get('cv_f1_std', 0):.4f} "
            f"AUC {v.get('cv_auc', float('nan')):.4f} | "
            f"val F1 {v.get('val_macro_f1', float('nan')):.4f} "
            f"acc {v.get('val_acc', float('nan')):.4f} "
            f"AUC {v.get('val_auc', float('nan')):.4f}")
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
