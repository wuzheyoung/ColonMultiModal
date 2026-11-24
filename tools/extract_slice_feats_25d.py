# -*- coding: utf-8 -*-
"""m05965 任务1b: 2.5D [z-1, z, z+1] DINOv2 特征提取 (夜间挂机实验)。

与 extract_slice_feats_v2.py 完全同契约 (.npz 字段/120mm 质心裁剪/zscore/BILINEAR 224):
  每有效掩码切片 z 以该层质心裁 120mm 方框, 取 va[max(z-1,0)], va[z], va[min(z+1,Z-1)]
  三层各自 zscore 后拼为 3 通道 (层序 z-1, z, z+1) → dinov2 → 768 维。
掩码: t 只读原生 masks_totalseg_mr_t; d 优先原生 mr_d>=MIN_VOX(self), 其次
  data/masks_dwi_remapped (方案C 后目录仅含 from_t, from_c 已禁用)。
输出: features/slice_feats_v2_25d/dinov2/{t,d}/{pid}.npz (feats/slice_idx/rel_z/
  mask_area_ratio/mask_source, 落盘前 np.isfinite 断言)。
用法: python tools/extract_slice_feats_25d.py --seq t,d [--smoke-pids 51] [--limit N]
"""
import os, sys, gc, json, time, argparse
import numpy as np
import SimpleITK as sitk
import torch
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RAW = "data/hos_pairs_raw.csv"
MASTER = "data/master_5fold_356.csv"
OUT_ROOT = "features/slice_feats_v2_25d"
MASK_REMAP = "data/masks_dwi_remapped"
MASK_BASE = {s: f"data/masks_totalseg_mr_{s}" for s in ("t", "c", "d")}
SIZE_MM = 120.0
IMG_SIZE = 224
MIN_VOX = 50
SEQ_COL = {"t": "mri_path_t", "c": "mri_path_c", "d": "mri_path_d"}


def load_backbone(device):
    model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False,
                              num_classes=0, dynamic_img_size=True)
    from safetensors.torch import load_file as _sl
    st = _sl("pretrained/dinov2_vitb14_reg4/model.safetensors", device="cpu")
    model.load_state_dict(st, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).half()


def load_mask(pid, seq):
    """与 extract_slice_feats_v2.load_mask 同逻辑 (P0 序列隔离)。"""
    orig_p = os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz")
    if seq in ("t", "c"):
        if os.path.exists(orig_p):
            try:
                a = sitk.GetArrayFromImage(sitk.ReadImage(orig_p))
                if int(a.sum()) >= MIN_VOX:
                    return a, "self"
            except Exception:
                pass
        return None, "missing"
    if os.path.exists(orig_p):
        try:
            a = sitk.GetArrayFromImage(sitk.ReadImage(orig_p))
            if int(a.sum()) >= MIN_VOX:
                return a, "self"
        except Exception:
            pass
    remap_p = os.path.join(MASK_REMAP, str(pid), "colon.nii.gz")
    if os.path.exists(remap_p):
        try:
            a = sitk.GetArrayFromImage(sitk.ReadImage(remap_p))
            if int(a.sum()) >= MIN_VOX:
                return a, "remapped"
        except Exception:
            pass
    return None, "missing"


def extract_slices_25d(pid, seq, mri_path, model, device):
    """2.5D 三层拼通道。返回 (feats[K,768], idx, relz, area, source)。"""
    from PIL import Image
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()
    va = sitk.GetArrayFromImage(vimg)
    Z = va.shape[0]
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
    feats_all = np.zeros((n_slices, 768), dtype=np.float16)
    idx_all = np.zeros(n_slices, dtype=np.int32)
    relz_all = np.zeros(n_slices, dtype=np.float32)
    area_all = np.zeros(n_slices, dtype=np.float32)
    px, py = sp[0], sp[1]
    ry = int(round(SIZE_MM / 2 / py)); rx = int(round(SIZE_MM / 2 / px))
    cnt = 0
    for z in range(z_min, z_max + 1):
        m2 = ma[z]
        if m2.sum() == 0:
            continue
        ys, xs = np.nonzero(m2)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        y0, y1 = max(0, cy - ry), min(va.shape[1], cy + ry)
        x0, x1 = max(0, cx - rx), min(va.shape[2], cx + rx)
        if y1 - y0 < 4 or x1 - x0 < 4:
            continue
        # 三层: [max(z-1,0), z, min(z+1,Z-1)], 各层同质心裁剪 → 各自 zscore → 224
        zs = [max(z - 1, 0), z, min(z + 1, Z - 1)]
        chans = []
        for zz in zs:
            crop = va[zz, y0:y1, x0:x1].astype(np.float32)
            H, W = crop.shape
            S = max(H, W)
            pad_img = np.zeros((S, S), dtype=np.float32)
            pad_img[:H, :W] = crop
            m_ = pad_img.mean(); s_ = pad_img.std() + 1e-6
            pad_img = (pad_img - m_) / s_
            im = Image.fromarray(pad_img)
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            chans.append(np.asarray(im).astype(np.float32))
        t = torch.from_numpy(np.stack(chans, axis=0))[None]  # (1,3,224,224)
        t = t.to(device).half()
        with torch.inference_mode():
            f = model(t).float().cpu().numpy().ravel()
        feats_all[cnt] = f.astype(np.float16)
        idx_all[cnt] = z
        relz_all[cnt] = (z - z_min) / max(1, (z_max - z_min))
        S = max(y1 - y0, x1 - x0)
        area_all[cnt] = float(m2.sum()) / float(S * S)
        cnt += 1
    if cnt == 0:
        return None
    return feats_all[:cnt], idx_all[:cnt], relz_all[:cnt], area_all[:cnt], source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="t,d", help="逗号分隔序列 (仅 t/d)")
    ap.add_argument("--smoke-pids", help="逗号分隔 pid (冒烟)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    t0 = time.time()
    model = load_backbone(device)
    print("dinov2 loaded in", round(time.time() - t0, 1), "s")
    master = pd.read_csv(MASTER)
    raw = pd.read_csv(RAW).set_index("patient_id")
    seqs = [s for s in args.seq.split(",") if s in ("t", "d")]
    if args.smoke_pids:
        pids = [int(x) for x in args.smoke_pids.split(",")]
    else:
        pids = sorted(master["patient_id"].astype(int).tolist())
    if args.limit:
        pids = pids[:args.limit]
    done, skipped, nan_n = 0, 0, 0
    t_start = time.time()
    peak = 0
    for pid in pids:
        if pid not in raw.index:
            skipped += 1
            continue
        for seq in seqs:
            col = SEQ_COL[seq]
            mp = raw.loc[pid, col]
            if not (isinstance(mp, str) and os.path.exists(mp)):
                continue
            out_p = os.path.join(OUT_ROOT, "dinov2", seq, f"{pid}.npz")
            if os.path.exists(out_p) and os.path.getsize(out_p) > 100:
                skipped += 1
                continue
            r = extract_slices_25d(pid, seq, mp, model, device)
            if r is None:
                if os.path.exists(out_p):
                    os.remove(out_p)
                skipped += 1
                continue
            feats, idx, relz, area, source = r
            if not np.isfinite(feats).all():
                if os.path.exists(out_p):
                    os.remove(out_p)
                nan_n += 1
                continue
            os.makedirs(os.path.dirname(out_p), exist_ok=True)
            tmp = os.path.join(os.path.dirname(out_p), f"{pid}_tmp.npz")
            np.savez_compressed(tmp, feats=feats, slice_idx=idx, rel_z=relz,
                                mask_area_ratio=area, mask_source=np.array([source]))
            os.replace(tmp, out_p)
            done += 1
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        if (done + skipped) % 40 == 0:
            torch.cuda.empty_cache(); gc.collect()
        if args.smoke_pids:
            print(f"  pid{pid}: done={done} skipped={skipped} (elapsed {round(time.time()-t_start,1)}s)")
    total = time.time() - t_start
    print(f"done={done} skipped={skipped} nan={nan_n} total={round(total,1)}s "
          f"({round(total/max(1,done),2)}s/case) peak_cuda={round(peak,2)}GB")
    with open("logs/exp_4060_extract_25d_summary.json", "w", encoding="utf-8") as f:
        json.dump({"done": done, "skipped": skipped, "nan": nan_n, "total_s": round(total, 1),
                   "per_case_s": round(total / max(1, done), 2), "peak_cuda_gb": round(peak, 2)},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
