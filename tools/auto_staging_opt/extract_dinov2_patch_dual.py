# -*- coding: utf-8 -*-
"""extract_dinov2_patch_dual.py — 阶段3: DINOv2 ViT-B/14 空间 Patch Token 瘤内/瘤周双流池化

复用 extract_slice_feats_clean.py 的 tight60_m10 (bbox 中点 + 真对称补黑) 精确裁剪逻辑,
对 t=280 (原生掩码) 与 d=201 (d_self 原生 ≥50 + from_t 重映射) 提取:
- cls_feat      (K,768) float16  — [CLS] token
- intra_patch   (K,768) float16  — 瘤内 16×16 软权重加权平均 patch token (w_intra, sum=1)
- peri_patch    (K,768) float16  — 瘤周 6mm 膨胀环带软权重加权平均 (w_peri, sum=1)
- mask_area_mm2 (K,)    float32  — 每层掩码物理面积
- slice_idx / rel_z (与旧 npz 契约一致)

w_intra: 二值掩码 → 224×224 (NEAREST) → avg_pool 14×14 → 16×16 软权重, 归一化 sum=1
w_peri : 掩码物理膨胀 6mm → 减原掩码 → 同下采样 → 归一化

输出: features/auto_staging_opt/dinov2_patch_dual/{t,d}/<pid>.npz
用法: python -B tools/auto_staging_opt/extract_dinov2_patch_dual.py --all [--smoke-pids a,b --limit N]
"""
import os, sys, argparse, json
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
import timm
from safetensors.torch import load_file as _sl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from extract_slice_feats_clean import (load_mask_self, bbox_mid, crop_clean,
                                       MODES, IMG_SIZE, MIN_VOX, MASK_BASE)  # noqa

BACKBONE = "dinov2"
SEQ_MASK = MASK_BASE   # {"t": "data/masks_totalseg_mr_t", "d": "data/masks_totalseg_mr_d"}
REMAPPED = "data/masks_dwi_remapped"
OUT_ROOT = "features/auto_staging_opt/dinov2_patch_dual"
MODE = "tight60_m10"   # 边长 = max(60, span+20), bbox 中点 + 真对称补黑
PERI_MM = 6.0          # 瘤周环带物理膨胀距离 (mm)


def load_mask_d(pid):
    """d 序列: 优先原生 mr_d ≥ MIN_VOX (self), 其次 remapped (from_t)。返回 (arr, source)。"""
    orig_p = os.path.join(SEQ_MASK["d"], str(pid), "colon.nii.gz")
    if os.path.exists(orig_p):
        try:
            a = sitk.GetArrayFromImage(sitk.ReadImage(orig_p))
            if int(a.sum()) >= MIN_VOX:
                return a, "self"
        except Exception:
            pass
    rp = os.path.join(REMAPPED, str(pid), "colon.nii.gz")
    if os.path.exists(rp):
        try:
            a = sitk.GetArrayFromImage(sitk.ReadImage(rp))
            if int(a.sum()) >= MIN_VOX:
                return a, "from_t"
        except Exception:
            pass
    return None, "missing"


def load_mask(pid, seq):
    if seq == "d":
        return load_mask_d(pid)
    return load_mask_self(pid, seq)


def weight_map_16(mbox224, px, py, dil_mm=0.0):
    """二值掩码 → 16×16 软权重 (NEAREST resize 224 → avg_pool 14×14 → 16×16), 归一化 sum=1 (极小病灶也非零)。"""
    if mbox224.shape != (IMG_SIZE, IMG_SIZE):
        from PIL import Image
        mbox224 = np.asarray(Image.fromarray(mbox224).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
    m = torch.from_numpy(mbox224.astype(np.float32))[None, None]
    if dil_mm > 0:
        # 物理膨胀: 半径(px) = dil/px, 半径(py) = dil/py → 各向异性结构元
        ry, rx = max(1, int(round(dil_mm / py))), max(1, int(round(dil_mm / px)))
        se = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
        mbox224 = binary_dilation(mbox224, structure=se).astype(np.float32)
        m = torch.from_numpy(mbox224)[None, None]
    w = F.avg_pool2d(m, kernel_size=14, stride=14)     # (1,1,16,16)
    w = w.reshape(16, 16)
    s = w.sum()
    if s <= 0:
        return w * 0.0
    return (w / s).float()   # sum=1


def patch_weighted(patch_tokens, w):
    """patch_tokens (196,768) × 16×16 软权重 → (768,) 加权平均。"""
    p = patch_tokens.reshape(16, 16, 768)              # (16,16,768)
    wv = w.reshape(16, 16, 1)
    return (p * wv).sum(dim=(0, 1))                    # (768,)


def process_pid(pid, seq, mri_path, model, device):
    """返回 npz dict 或 None (无有效层)。"""
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()
    px, py = sp[0], sp[1]
    va = sitk.GetArrayFromImage(vimg)                  # (Z,Y,X)
    ma, source = load_mask(pid, seq)
    if ma is None:
        return None
    if source == "from_t":
        mimg = sitk.ReadImage(os.path.join(REMAPPED, str(pid), "colon.nii.gz"))
    else:
        mimg = sitk.ReadImage(os.path.join(SEQ_MASK[seq], str(pid), "colon.nii.gz"))
    mres = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
    ma = sitk.GetArrayFromImage(mres)
    nz = np.nonzero(ma)
    if len(nz[0]) == 0:
        return None
    z_min, z_max = nz[0].min(), nz[0].max()
    # 收集各层裁剪
    crops, idxs, relzs, areas = [], [], [], []
    for z in range(z_min, z_max + 1):
        m2 = ma[z]
        if m2.sum() == 0:
            continue
        ys, xs = np.nonzero(m2)
        cy, cx = bbox_mid(ys, xs)
        span_mm = max((xs.max() - xs.min() + 1) * px, (ys.max() - ys.min() + 1) * py)
        size_mm, margin = MODES[MODE][1], MODES[MODE][2]
        if margin is not None:
            size_mm = max(size_mm, span_mm + 2 * margin)
        t, S, size_px, mbox = crop_clean(va[z], m2, cy, cx, size_mm, px, py)
        if S < 4:
            continue
        crops.append(t); idxs.append(z); relzs.append((z - z_min) / max(1, (z_max - z_min)))
        areas.append(float(m2.sum()) * px * py)
    if not crops:
        return None
    K = len(crops)
    imgs = torch.from_numpy(np.stack(crops))[:, None].repeat(1, 3, 1, 1).to(device).half()
    mboxes = [crop_clean(np.zeros_like(va[0]), m, cy, cx, size_mm, px, py)[3]
              for (m, cy, cx, size_mm) in _mbox_pairs(va, ma, z_min, z_max, px, py)]
    with torch.inference_mode():
        x = model.forward_features(imgs)      # (K, 1+4+196, 768)  fp16
        cls = x[:, 0]                          # (K,768)
        patch = x[:, 5:]                       # (K,196,768)  (跳过 1 cls + 4 reg)
        out = {}
        cls_a = cls.float().cpu().numpy()
        for i in range(K):
            w_intra = weight_map_16(mboxes[i], px, py, 0.0).to(device)
            w_peri = weight_map_16(mboxes[i], px, py, PERI_MM).to(device)
            intra = patch_weighted(patch[i].float(), w_intra)
            peri = patch_weighted(patch[i].float(), w_peri)
            if i == 0:
                intra_all = np.zeros((K, 768), dtype=np.float16)
                peri_all = np.zeros((K, 768), dtype=np.float16)
            intra_all[i] = intra.cpu().numpy().astype(np.float16)
            peri_all[i] = peri.cpu().numpy().astype(np.float16)
    res = {
        "cls_feat": cls_a.astype(np.float16),
        "intra_patch": intra_all,
        "peri_patch": peri_all,
        "mask_area_mm2": np.asarray(areas, dtype=np.float32),
        "slice_idx": np.asarray(idxs, dtype=np.int32),
        "rel_z": np.asarray(relzs, dtype=np.float32),
        "mask_source": np.asarray([source], dtype="S8"),
    }
    # isfinite 断言 (P3 防溢出)
    for k in ("cls_feat", "intra_patch", "peri_patch"):
        assert np.isfinite(res[k]).all(), (pid, seq, k)
    return res


def _mbox_pairs(va, ma, z_min, z_max, px, py):
    """重建每层裁剪用的 (mask, cy, cx, size_mm) — 用于 mbox 渲染 (与主循环一致)。"""
    pairs = []
    for z in range(z_min, z_max + 1):
        m2 = ma[z]
        if m2.sum() == 0:
            continue
        ys, xs = np.nonzero(m2)
        cy, cx = bbox_mid(ys, xs)
        span_mm = max((xs.max() - xs.min() + 1) * px, (ys.max() - ys.min() + 1) * py)
        size_mm, margin = MODES[MODE][1], MODES[MODE][2]
        if margin is not None:
            size_mm = max(size_mm, span_mm + 2 * margin)
        pairs.append((m2, cy, cx, size_mm))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke-pids", type=str, default="")
    ap.add_argument("--seq", type=str, default="t")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone(device)

    import pandas as pd
    md = pd.read_csv("data/master_5fold_356.csv")
    if args.smoke_pids:
        pids = [int(x) for x in args.smoke_pids.split(",")]
        seqs = [args.seq]
    else:
        pids = sorted(md.patient_id.unique().tolist())
        seqs = ["t", "d"]
    if args.limit:
        pids = pids[:args.limit]

    done = skipped = nomask = 0
    t0 = __import__("time").time()
    for seq in seqs:
        # t: 全 master 有原生掩码者; d: manifest self/from_t 有效 (has_valid=1)
        if seq == "d" and not args.smoke_pids:
            man = pd.read_csv("features/dwi_mask_remap_manifest.csv")
            man["pid"] = man["pid"].astype(int)
            valid = set(man[(man.has_valid_mask == 1) & (man.mask_source.isin(["self", "from_t"]))]["pid"])
            pids = sorted(p for p in md.patient_id.unique() if int(p) in valid)
        for pid in pids:
            out_p = os.path.join(OUT_ROOT, seq, f"{pid}.npz")
            if os.path.exists(out_p) and os.path.getsize(out_p) > 100 and not args.force:
                skipped += 1
                continue
            mri = None
            import pandas as pd
            rw = pd.read_csv("data/hos_pairs_raw.csv")
            col = f"mri_path_{seq}"
            row = rw[rw.patient_id == pid]
            if len(row) and isinstance(row.iloc[0][col], str):
                mri = row.iloc[0][col]
            if mri is None or not os.path.exists(mri):
                nomask += 1
                continue
            res = process_pid(pid, seq, mri, model, device)
            if res is None:
                nomask += 1
                continue
            os.makedirs(os.path.dirname(out_p), exist_ok=True)
            tmp = out_p.replace(".npz", "") + ".tmp"   # savez 会追加 .npz → x.tmp.npz
            np.savez_compressed(tmp, **res)
            os.replace(tmp + ".npz", out_p)
            done += 1
            if done % 50 == 0:
                print(f"[{seq}] done={done} skipped={skipped} nomask={nomask} "
                      f"elapsed={__import__('time').time()-t0:.0f}s")
    print(f"DONE seqs={seqs} done={done} skipped={skipped} nomask={nomask} "
          f"total={__import__('time').time()-t0:.1f}s")
    if args.all:
        summary = {"done": done, "skipped": skipped, "nomask": nomask,
                   "total_s": round(__import__("time").time() - t0, 1), "seqs": seqs}
        with open("logs/auto_staging_opt/extract_patch_dual_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def load_backbone(device):
    model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False,
                              num_classes=0, dynamic_img_size=True)
    st = _sl("pretrained/dinov2_vitb14_reg4/model.safetensors", device="cpu")
    model.load_state_dict(st, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).half()


if __name__ == "__main__":
    main()
