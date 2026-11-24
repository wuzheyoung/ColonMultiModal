# tools/finetune_dinov2_e2e.py
# 任务4 (m05965): 端到端微调 DINOv2 (2.5D [z-1,z,z+1] 三通道切片袋, max_slices=24,
#                 120mm 肿瘤中心裁剪框) + GatedAttention MIL 直训 label_T。
# - 序列: t (原生掩码) / d (干净: 优先原生 mr_d>=50 self, 其次 dwi remapped from_t)。
# - 5 折 CV 严格对齐 master fold; 训练集锁死 direct_all n=280; 评估 direct_tcd n=162 (附报 direct_td n=214)。
# - 落盘: outputs/exp_4060_overnight/oof_e2e_{seq}.csv + e2e_{seq}.png(可选) + summary 并入 exp_4060_overnight_summary.json。
# 用法: python tools/finetune_dinov2_e2e.py --seq t|d [--smoke 1 --epochs 2 --limit 40]
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MASTER = "data/master_5fold_356.csv"
RAW = "data/hos_pairs_raw.csv"
DIRECT = "data/direct_cohort_ids.csv"
MASK_BASE = {"t": "data/masks_totalseg_mr_t", "d": "data/masks_totalseg_mr_d"}
MASK_REMAP = "data/masks_dwi_remapped"
CACHE = "features/e2e_cache"
SIZE_MM = 120
IMG_SIZE = 224
MAX_SLICES = 24
MIN_VOX = 50
OUT = "outputs/exp_4060_overnight"
SUMMARY_JSON = "logs/exp_4060_overnight_summary.json"


def load_dinov2(device):
    import timm
    from safetensors.torch import load_file as _sl
    model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False,
                              num_classes=0, dynamic_img_size=True)
    st = _sl("pretrained/dinov2_vitb14_reg4/model.safetensors", device="cpu")
    model.load_state_dict(st, strict=True)
    return model.to(device)


def load_mask(pid, seq):
    orig_p = os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz")
    if os.path.exists(orig_p):
        try:
            a = sitk.GetArrayFromImage(sitk.ReadImage(orig_p))
            if int(a.sum()) >= MIN_VOX:
                return a, "self"
        except Exception:
            pass
    if seq == "d":
        remap_p = os.path.join(MASK_REMAP, str(pid), "colon.nii.gz")
        if os.path.exists(remap_p):
            try:
                a = sitk.GetArrayFromImage(sitk.ReadImage(remap_p))
                if int(a.sum()) >= MIN_VOX:
                    return a, "remapped"
            except Exception:
                pass
    return None, "missing"


def build_bag(pid, seq, mri_path, max_slices=MAX_SLICES):
    """返回 (img_stack, rel_z, area, K) 或 None。
    img_stack: (K,3,224,224) float16, zscore 归一化, 2.5D 三通道 [z-1,z,z+1]。
    """
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()
    va = sitk.GetArrayFromImage(vimg)
    ma, source = load_mask(pid, seq)
    if ma is None:
        return None
    mimg = sitk.ReadImage(os.path.join(MASK_REMAP if source == "remapped" else MASK_BASE[seq],
                                       str(pid), "colon.nii.gz"))
    mres = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
    ma = sitk.GetArrayFromImage(mres)
    nz = np.nonzero(ma)
    if len(nz[0]) == 0:
        return None
    z_min, z_max = nz[0].min(), nz[0].max()
    n_slices = int(z_max - z_min + 1)
    # 均匀选取 <= max_slices 层 (保覆盖)
    zs_all = [z for z in range(z_min, z_max + 1) if ma[z].sum() > 0]
    if not zs_all:
        return None
    if len(zs_all) > max_slices:
        step = len(zs_all) / max_slices
        zs = [zs_all[min(len(zs_all) - 1, int(i * step))] for i in range(max_slices)]
    else:
        zs = zs_all
    px, py = sp[0], sp[1]
    ry = int(round(SIZE_MM / 2 / py)); rx = int(round(SIZE_MM / 2 / px))
    imgs = []; relzs = []; areas = []
    for z in zs:
        m2 = ma[z]
        ys, xs = np.nonzero(m2)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        y0, y1 = max(0, cy - ry), min(va.shape[1], cy + ry)
        x0, x1 = max(0, cx - rx), min(va.shape[2], cx + rx)
        if y1 - y0 < 4 or x1 - x0 < 4:
            continue
        chans = []
        for dz in (-1, 0, 1):
            zz = max(0, min(va.shape[0] - 1, z + dz))
            crop = va[zz, y0:y1, x0:x1].astype(np.float32)
            H, W = crop.shape
            S = max(H, W)
            pad_img = np.zeros((S, S), dtype=np.float32)
            pad_img[:H, :W] = crop
            m_ = pad_img.mean(); s_ = pad_img.std() + 1e-6
            pad_img = (pad_img - m_) / s_
            im = Image.fromarray(pad_img).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            chans.append(np.asarray(im).astype(np.float32))
        img = np.stack(chans, axis=0)                       # (3,224,224)
        imgs.append(img)
        relzs.append((z - z_min) / max(1, (z_max - z_min)))
        areas.append(float(m2.sum()) / float(S * S))
    if not imgs:
        return None
    return (np.stack(imgs).astype(np.float16),                # (K,3,224,224)
            np.asarray(relzs, dtype=np.float32),
            np.asarray(areas, dtype=np.float32),
            len(imgs))


def cache_bag(pid, seq, mri_path):
    d = os.path.join(CACHE, seq)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{pid}.npz")
    if os.path.exists(p):
        return p
    r = build_bag(pid, seq, mri_path)
    if r is None:
        return None
    imgs, relz, area, K = r
    np.savez_compressed(p, imgs=imgs, rel_z=relz, area=area, K=K)
    return p


def get_mri_path(pid, seq):
    raw = pd.read_csv(RAW)
    row = raw[raw["patient_id"] == pid]
    if row.empty:
        return None
    return row.iloc[0].get(f"mri_path_{seq}")


class GatedAttention(nn.Module):
    def __init__(self, h, dropout=0.4):
        super().__init__()
        self.att_a = nn.Linear(h, h)
        self.att_b = nn.Linear(h, h)
        self.att_v = nn.Linear(h, 1)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, x, mask):  # (B,K,h), mask (B,K)
        a = torch.tanh(self.att_a(x)) * torch.sigmoid(self.att_b(x))
        w = torch.softmax(self.att_v(a).squeeze(-1).masked_fill(mask == 0, -float("inf")), dim=1)
        bag = (w.unsqueeze(-1) * x).sum(dim=1)
        return self.head(bag).squeeze(-1)


class E2EMIL(nn.Module):
    def __init__(self, backbone, h=256, dropout=0.4, spatial_prior=False):
        super().__init__()
        self.backbone = backbone
        self.spatial_prior = spatial_prior
        in_dim = 768 + (2 if spatial_prior else 0)
        self.proj = nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout))
        self.gated = GatedAttention(h, dropout)

    def forward(self, imgs, mask, rel_z=None, area=None):
        # imgs: (B,K,3,224,224) float16
        B, K, C, H, W = imgs.shape
        x = imgs.reshape(B * K, C, H, W)
        with torch.autocast("cuda", dtype=torch.float16):
            f = self.backbone(x)                                    # (B*K,768)
        f = f.float().reshape(B, K, -1)
        if self.spatial_prior:
            extra = torch.stack([rel_z, area], dim=-1).float()
            f = torch.cat([f, extra], dim=-1)
        h = self.proj(f)
        return self.gated(h, mask)


def train_fold(bags, tr_pids, va_pids, model, device, epochs, lr, wd, seed, spatial_prior):
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr * 0.1},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")],
         "lr": lr},
    ], weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    # 按 K 分桶
    bins = {}
    for p in tr_pids:
        K = bags[p][0].shape[0]
        b = int(K - 1) // 4
        bins.setdefault(b, []).append(p)
    KMAX = MAX_SLICES
    model.train()
    for ep in range(epochs):
        g = torch.Generator().manual_seed(int(seed) + ep)
        ks = list(bins)
        order = torch.randperm(len(ks), generator=g).tolist()
        for i in order:
            b = ks[i]
            ps = bins[b]
            for j in range(0, len(ps), 4):
                psb = ps[j:j + 4]
                imgs = [bags[p][0].to(device) for p in psb]             # (K,3,224,224)
                Ks = [x.shape[0] for x in imgs]
                Kpad = max(Ks)
                B = len(psb)
                ib = torch.zeros(B, Kpad, 3, IMG_SIZE, IMG_SIZE, device=device, dtype=torch.float16)
                mask = torch.zeros(B, Kpad, device=device)
                for bi, (x, K_) in enumerate(zip(imgs, Ks)):
                    ib[bi, :K_] = x
                    mask[bi, :K_] = 1.0
                y = torch.tensor([bags[p][3] for p in psb], device=device).float()
                def _pad1(x):
                    K_ = x.shape[0]
                    if K_ < Kpad:
                        return torch.cat([x, torch.zeros(Kpad - K_, device=device)], dim=0)
                    return x
                rel_z = torch.stack([_pad1(bags[p][1].to(device)) for p in psb], dim=0)
                area = torch.stack([_pad1(bags[p][2].to(device)) for p in psb], dim=0)
                logit = model(ib, mask, rel_z, area)
                loss = crit(logit, y)
                opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    # 验证折预测
    model.eval()
    yv, pv, ids = [], [], []
    with torch.no_grad():
        for p in va_pids:
            if p not in bags:
                continue
            imgs = bags[p][0].unsqueeze(0).to(device)               # (1,K,3,224,224)
            K_ = imgs.shape[1]
            mask = torch.ones(1, K_, device=device)
            rel_z = torch.tensor(bags[p][1], device=device).unsqueeze(0)
            area = torch.tensor(bags[p][2], device=device).unsqueeze(0)
            logit = model(imgs.float(), mask, rel_z, area)
            pv.append(torch.sigmoid(logit).item())
            yv.append(bags[p][3])
            ids.append(p)
    return pd.DataFrame({"patient_id": ids, "prob": pv, "label": yv})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="t")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--spatial-prior", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", type=int, default=0, help="只跑 1 折 + 前 N 例")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    master = pd.read_csv(MASTER)
    direct = pd.read_csv(DIRECT)
    direct_pids = sorted(direct.loc[direct["is_tgr"] == 0, "pid"].tolist())
    # 影像路径存在性子集
    raw = pd.read_csv(RAW).set_index("patient_id")
    def has_imgs(p, seqs):
        for s in seqs:
            v = raw.loc[p, f"mri_path_{s}"]
            if not (isinstance(v, str) and os.path.exists(v)):
                return False
        return True
    tcd = sorted([p for p in direct_pids if has_imgs(p, ("t", "c", "d"))])
    td = sorted([p for p in direct_pids if has_imgs(p, ("t", "d"))])
    eval_pids = sorted(set(tcd) | set(td))
    print(f"[e2e {args.seq}] direct_all={len(direct_pids)} tcd={len(tcd)} td={len(td)} eval={len(eval_pids)}")
    t0 = time.time()
    # 构建/加载缓存袋
    bags = {}
    mp_col = f"mri_path_{args.seq}"
    for pid in direct_pids:
        mp = raw.loc[pid, mp_col] if pid in raw.index else None
        if not (isinstance(mp, str) and os.path.exists(mp)):
            continue
        cp = cache_bag(pid, args.seq, mp)
        if cp is None:
            continue
        z = np.load(cp)
        bags[pid] = (torch.from_numpy(z["imgs"]), torch.from_numpy(z["rel_z"]),
                     torch.from_numpy(z["area"]), float(master.loc[master["patient_id"] == pid, "label_T"].iloc[0]))
        if args.smoke and len(bags) >= args.smoke:
            break
    print(f"[e2e {args.seq}] bags={len(bags)} 缓存/构建 {time.time()-t0:.0f}s")
    # 5 折
    model = E2EMIL(load_dinov2(device), h=args.h, dropout=args.dropout,
                   spatial_prior=args.spatial_prior).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[e2e {args.seq}] params={n_param/1e6:.1f}M")
    oofs = []
    folds = sorted(master["fold"].unique())
    for k in folds:
        tr = [p for p in bags if int(master.loc[master["patient_id"] == p, "fold"].iloc[0]) != k]
        va = [p for p in eval_pids if p in bags and int(master.loc[master["patient_id"] == p, "fold"].iloc[0]) == k]
        if not va:
            continue
        print(f"  fold {k}: train={len(tr)} val={len(va)}")
        df = train_fold(bags, tr, va, model, device, args.epochs, args.lr, args.wd, args.seed,
                        args.spatial_prior)
        df["fold"] = k
        oofs.append(df)
        if args.smoke:
            break
    oof = pd.concat(oofs, ignore_index=True)
    oof = oof.merge(master[["patient_id", "fold"]], on="patient_id", how="left",
                    suffixes=("", "_master"))
    os.makedirs(OUT, exist_ok=True)
    tag = f"e2e_{args.seq}" + ("_sp" if args.spatial_prior else "")
    oof.to_csv(os.path.join(OUT, f"oof_{tag}.csv"), index=False)
    # 评估 tcd / td
    def eval_view(sub):
        d = oof[oof["patient_id"].isin(sub)]
        if len(d) < 10 or d["label"].nunique() < 2:
            return None
        auc = roc_auc_score(d["label"], d["prob"])
        n_pos = int(d["label"].sum())
        rng = np.random.default_rng(7)
        boots = [roc_auc_score(d["label"].sample(frac=1.0, replace=True, random_state=int(s)),
                               d["prob"].sample(frac=1.0, replace=True, random_state=int(s)))
                 for s in rng.integers(0, 100000, 2000)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        fm = oof[oof["patient_id"].isin(sub)].groupby("fold")["prob"].apply(
            lambda s: roc_auc_score(oof.loc[s.index, "label"], s) if len(set(oof.loc[s.index, "label"])) == 2 else np.nan)
        return {"n": int(len(d)), "n_pos": n_pos, "pooled_auc": float(auc),
                "ci95": ci, "fold_mean": float(fm.mean()), "fold_sd": float(fm.std(ddof=1))}
    res = {"tcd": eval_view(tcd), "td": eval_view(td)}
    print(f"[e2e {args.seq}] tcd={res['tcd']} td={res['td']}")
    # 并入 summary
    if os.path.exists(SUMMARY_JSON):
        summary = json.load(open(SUMMARY_JSON, encoding="utf-8"))
    else:
        summary = {"tasks": {}}
    summary["tasks"][f"task4_{tag}"] = {"seq": args.seq, "epochs": args.epochs,
                                        "lr": args.lr, "wd": args.wd, "h": args.h,
                                        "spatial_prior": args.spatial_prior,
                                        "bags": len(bags), "params": n_param,
                                        "res": res, "time_s": round(time.time() - t0, 1)}
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"DONE → outputs/exp_4060_overnight/oof_{tag}.csv (total={time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
