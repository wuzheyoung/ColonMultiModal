# -*- coding: utf-8 -*-
"""DINOv2 60mm 局部视野冻结逐层特征提取 (m06672, 物理尺度消融)。

与 extract_slice_feats_v2.py 预处理/灰度归一化/切片筛选/.npz 契约完全一致,
唯一改动: 裁剪物理窗口 SIZE_MM 120.0 -> 60.0 (仍以每层掩码质心为中心, pad 正方, BILINEAR 224)。

范围限定 (仅 dinov2 骨干, 全队列 direct+ncrt):
- t 序列: pid 集合 = 现有 features/slice_feats_v2/dinov2/t/ 文件名 (与现有完全一致, ~280)。
- d 序列: 仅 mask_source=='self' 的原生掩码病例 (manifest self 且 has_valid=1, ~139), 不提 from_t。
产物: features/slice_feats_v2_60mm/dinov2/{t,d}/{pid}.npz, 落盘前断言 np.isfinite(feats).all()。

QC: --qc-pids 对给定 pid 渲染 120mm vs 60mm 裁剪并排图至 outputs/crop_60mm_qc/。
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
OUT_ROOT = "features/slice_feats_v2_60mm"
SRC_ROOT = "features/slice_feats_v2"
MANIFEST = "features/dwi_mask_remap_manifest.csv"
MASK_BASE = {s: f"data/masks_totalseg_mr_{s}" for s in ("t", "d")}
SIZE_MM = 60.0
SIZE_MM_OLD = 120.0  # QC 对比用
IMG_SIZE = 224
BATCH = 16
MIN_VOX = 50

SEQ_COL = {"t": "mri_path_t", "d": "mri_path_d"}


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


def load_mask_self(pid, seq):
    """仅原生掩码 (self): t/d 都只读 MASK_BASE, 不回退 remapped (60mm 版 d 仅 self)。"""
    orig_p = os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz")
    if os.path.exists(orig_p):
        try:
            a = sitk.GetArrayFromImage(sitk.ReadImage(orig_p))
            if int(a.sum()) >= MIN_VOX:
                return a, "self"
        except Exception:
            pass
    return None, "missing"


def _crop_slice(va, ma, z, size_mm, px, py):
    """以该层掩码质心为中心, size_mm 物理方框裁剪 (pad 正方, zscore, 返回 224 张量)。"""
    m2 = ma[z]
    ys, xs = np.nonzero(m2)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    ry = int(round(size_mm / 2 / py)); rx = int(round(size_mm / 2 / px))
    y0, y1 = max(0, cy - ry), min(va.shape[1], cy + ry)
    x0, x1 = max(0, cx - rx), min(va.shape[2], cx + rx)
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    crop = va[z, y0:y1, x0:x1].astype(np.float32)
    H, W = crop.shape
    S = max(H, W)
    pad_img = np.zeros((S, S), dtype=np.float32)
    pad_img[:H, :W] = crop
    m_ = pad_img.mean(); s_ = pad_img.std() + 1e-6
    pad_img = (pad_img - m_) / s_
    from PIL import Image
    im = Image.fromarray(pad_img)
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(im).astype(np.float32)


def extract_slices(pid, seq, mri_path, model, device):
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()
    va = sitk.GetArrayFromImage(vimg)
    ma, source = load_mask_self(pid, seq)
    if ma is None:
        return None
    mimg = sitk.ReadImage(os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz"))
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
    cnt = 0
    for z in range(z_min, z_max + 1):
        m2 = ma[z]
        if m2.sum() == 0:
            continue
        t = _crop_slice(va, ma, z, SIZE_MM, px, py)
        if t is None:
            continue
        S_sq = float(max(t.shape))  # 224
        t = torch.from_numpy(t)[None, None]
        t = t.repeat(1, 3, 1, 1).to(device).half()
        with torch.inference_mode():
            f = model(t).float().cpu().numpy().ravel()
        feats_all[cnt] = f.astype(np.float16)
        idx_all[cnt] = z
        relz_all[cnt] = (z - z_min) / max(1, (z_max - z_min))
        area_all[cnt] = float(m2.sum()) / float(S_sq * S_sq)
        cnt += 1
    if cnt == 0:
        return None
    return feats_all[:cnt], idx_all[:cnt], relz_all[:cnt], area_all[:cnt], source


def process_pid(pid, seq, mri_path, model, device, force=False):
    out_p = os.path.join(OUT_ROOT, "dinov2", seq, f"{pid}.npz")
    if not force and os.path.exists(out_p) and os.path.getsize(out_p) > 100:
        return "skip"
    r = extract_slices(pid, seq, mri_path, model, device)
    if r is None:
        if force and os.path.exists(out_p):
            os.remove(out_p)
        return "nomask"
    feats, idx, relz, area, source = r
    if not np.isfinite(feats).all():
        if force and os.path.exists(out_p):
            os.remove(out_p)
        return "nan"
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    tmp = os.path.join(os.path.dirname(out_p), f"{pid}_tmp.npz")
    np.savez_compressed(tmp, feats=feats, slice_idx=idx, rel_z=relz,
                        mask_area_ratio=area, mask_source=np.array([source]))
    os.replace(tmp, out_p)
    return f"{feats.shape[0]}sl"


def qc_render(pid, seq, mri_path, out_dir):
    """渲染 120mm vs 60mm 裁剪并排图 (含 mask 轮廓), 用于肉眼确认 60mm 框完整性。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()
    va = sitk.GetArrayFromImage(vimg)
    ma, _ = load_mask_self(pid, seq)
    if ma is None:
        print(f"  qc pid{pid}: no mask, skip"); return
    mimg = sitk.ReadImage(os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz"))
    mres = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
    ma = sitk.GetArrayFromImage(mres)
    z = int(np.argmax(ma.sum(axis=(1, 2))))  # mask 覆盖最大层
    m2 = ma[z]
    ys, xs = np.nonzero(m2)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    px, py = sp[0], sp[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, size_mm in zip(axes, (SIZE_MM_OLD, SIZE_MM)):
        ry = int(round(size_mm / 2 / py)); rx = int(round(size_mm / 2 / px))
        y0, y1 = max(0, cy - ry), min(va.shape[1], cy + ry)
        x0, x1 = max(0, cx - rx), min(va.shape[2], cx + rx)
        crop = va[z, y0:y1, x0:x1]
        H, W = crop.shape
        S = max(H, W)
        pad_img = np.zeros((S, S), dtype=np.float32)
        pad_img[:H, :W] = crop
        # mask 轮廓 (同坐标系)
        mbox = np.zeros((S, S), dtype=np.uint8)
        mbox[:H, :W] = m2[y0:y1, x0:x1]
        ax.imshow(pad_img, cmap="gray")
        ax.contour(mbox, levels=[0.5], colors="r", linewidths=1.2)
        ax.set_title(f"{int(size_mm)}mm (pad S={S}px)", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"pid{pid} seq={seq} z={z} mask_vox={int(m2.sum())} (pT QC)", fontsize=12)
    os.makedirs(out_dir, exist_ok=True)
    out_p = os.path.join(out_dir, f"{pid}_{seq}_120vs60.png")
    fig.savefig(out_p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  qc pid{pid}: {out_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-pids", help="逗号分隔 pid (端到端冒烟)")
    ap.add_argument("--all", action="store_true", help="全量 (阶段2)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qc-pids", help="逗号分隔 pid 渲染 120vs60 QC 图")
    ap.add_argument("--qc-seq", default="t", help="QC 渲染序列 (t/d)")
    args = ap.parse_args()
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    t0 = time.time()
    model = load_backbone(device)
    print("dinov2 loaded in", round(time.time() - t0, 1), "s")

    if args.qc_pids:
        raw = pd.read_csv(RAW).set_index("patient_id")
        for pid in [int(x) for x in args.qc_pids.split(",")]:
            col = SEQ_COL[args.qc_seq]
            mp = raw.loc[pid, col]
            if isinstance(mp, str) and os.path.exists(mp):
                qc_render(pid, args.qc_seq, mp, "outputs/crop_60mm_qc")
            else:
                print(f"  qc pid{pid}: mri_path missing")
        return

    raw = pd.read_csv(RAW).set_index("patient_id")
    if args.smoke_pids:
        pids = [int(x) for x in args.smoke_pids.split(",")]
        seqs = ("t", "d")
    elif args.all:
        # t: 与现有 dinov2/t 完全一致 (文件名集合)
        t_pids = sorted(int(x[:-4]) for x in os.listdir(os.path.join(SRC_ROOT, "dinov2", "t"))
                        if x.endswith(".npz"))
        # d: manifest self 且 has_valid=1
        man = pd.read_csv(MANIFEST)
        d_pids = sorted(man[(man["mask_source"] == "self") & (man["has_valid_mask"] == 1)]["pid"].astype(int).tolist())
        pids = sorted(set(t_pids) | set(d_pids))
        seqs = ("t", "d")
        print(f"t_pids={len(t_pids)} d_self_pids={len(d_pids)} union={len(pids)}")
    else:
        print("需 --smoke-pids 或 --all"); return
    if args.limit:
        pids = pids[:args.limit]
    done, skipped, nomask = 0, 0, 0
    t_start = time.time()
    peak = 0
    for pid in pids:
        if pid not in raw.index:
            skipped += 1; continue
        for seq in seqs:
            # 集合过滤: t 只处理 t_pids, d 只处理 d_self_pids
            if args.all:
                if seq == "t" and pid not in t_pids:
                    continue
                if seq == "d" and pid not in d_pids:
                    continue
            col = SEQ_COL[seq]
            mp = raw.loc[pid, col]
            if not (isinstance(mp, str) and os.path.exists(mp)):
                continue
            info = process_pid(pid, seq, mp, model, device, force=args.force)
            if info == "skip":
                skipped += 1
            elif info == "nomask":
                nomask += 1
            else:
                done += 1
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        if (done + skipped + nomask) % 20 == 0:
            torch.cuda.empty_cache(); gc.collect()
        if args.smoke_pids:
            print(f"  pid{pid}: {info}  (elapsed {round(time.time()-t_start,1)}s)")
    total = time.time() - t_start
    print(f"done={done} skipped={skipped} nomask={nomask} total={round(total,1)}s "
          f"({round(total/max(1,done),2)}s/case) peak_cuda={round(peak,2)}GB")
    # 统计
    n_files = 0
    for seq in ("t", "d"):
        d_ = os.path.join(OUT_ROOT, "dinov2", seq)
        n_files += len([x for x in os.listdir(d_) if x.endswith(".npz")]) if os.path.isdir(d_) else 0
    size_mb = 0
    for root, _, fs in os.walk(OUT_ROOT):
        for f in fs:
            if f.endswith(".npz"):
                size_mb += os.path.getsize(os.path.join(root, f))
    print(f"OUT files={n_files} size={round(size_mb/1e6,2)}MB")
    if args.all:
        with open("logs/extract_60mm_summary.json", "w", encoding="utf-8") as f:
            json.dump({"done": done, "skipped": skipped, "nomask": nomask,
                       "total_s": round(total, 1), "per_case_s": round(total / max(1, done), 2),
                       "peak_cuda_gb": round(peak, 2), "n_files": n_files,
                       "size_mb": round(size_mb / 1e6, 2), "size_mm": SIZE_MM},
                      f, ensure_ascii=False, indent=2)
        print("DONE → logs/extract_60mm_summary.json")


if __name__ == "__main__":
    main()
