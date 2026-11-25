# -*- coding: utf-8 -*-
"""方案B: 自适应 Tight ROI 与 120mm_clean 同口径重提 (m07012, 只读不动旧特征目录)。

与 extract_slice_feats_60mm.py / extract_slice_feats_v2.py 的差异 (修复两大缺陷):
1. 定心: 2D bbox 中点 (ymin+ymax)/2, (xmin+xmax)/2 (精确居中, 非单层全掩码质心 mean);
2. 补黑: 真对称补黑 —— 围绕 (cy,cx) 按目标物理边长对应像素半宽构造正方形画布,
   越界区域按实际越界像素数在对应侧补 0, 病灶外接框几何中心严格位于裁剪图中心;
3. mask_area_ratio 修正: 分母改固定参考 (120mm)^2 物理面积, 新增 mask_area_mm2 与 crop_size_mm 字段;
4. 保留 .npz 原有 keys (feats/slice_idx/rel_z/mask_area_ratio/mask_source) 兼容下游 (train_slice_mil 等)。

两种模式 (独立目录, 不覆盖旧产物):
- --mode 120mm_clean  -> features/slice_feats_v2_120mm_clean/dinov2/{t,d}/  (固定 120mm 方框)
- --mode tight60_m10  -> features/slice_feats_v2_tight60_m10/dinov2/{t,d}/  (边长 = max(60mm, span+2*10mm))

范围: t = 与 slice_feats_v2/dinov2/t 文件名一致 (~280); d = manifest self 且 has_valid=1 (~139)。
QC: --qc-pids/--qc-seq 渲染三列并排图 (左=原120质心+clamp, 中=120mm_clean, 右=tight60_m10)
    至 outputs/crop_clean_qc/, 打印三种裁剪边长/像素尺寸(S)/掩码保留率。
"""
import os, sys, gc, json, time, argparse
import numpy as np
import SimpleITK as sitk
import torch
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RAW = "data/hos_pairs_raw.csv"
SRC_ROOT = "features/slice_feats_v2"
MANIFEST = "features/dwi_mask_remap_manifest.csv"
MASK_BASE = {s: f"data/masks_totalseg_mr_{s}" for s in ("t", "d")}
IMG_SIZE = 224
MIN_VOX = 50
REF_MM2 = 120.0 * 120.0  # mask_area_ratio 固定参考分母 (物理面积 mm²)

MODES = {
    "120mm_clean": ("features/slice_feats_v2_120mm_clean", 120.0, None),        # (out_root, size_mm, margin)
    "tight60_m10": ("features/slice_feats_v2_tight60_m10", 60.0, 10.0),         # size= max(60, span+2*10)
}
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
    orig_p = os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz")
    if os.path.exists(orig_p):
        try:
            a = sitk.GetArrayFromImage(sitk.ReadImage(orig_p))
            if int(a.sum()) >= MIN_VOX:
                return a, "self"
        except Exception:
            pass
    return None, "missing"


def bbox_mid(ys, xs):
    """2D bbox 中点 (精确居中, 不做单侧偏移)。"""
    cy = (int(ys.min()) + int(ys.max())) / 2.0
    cx = (int(xs.min()) + int(xs.max())) / 2.0
    return cy, cx


def crop_clean(va, m2, cy, cx, size_mm, px, py):
    """以 (cy,cx) 为中心, 物理边长 size_mm 的正方形裁剪 + 真对称补黑 (越界四边补 0)。

    m07209 规范: 显式 canvas=zeros((S,S)), 有效源切片范围 [src_y0:src_y1, src_x0:src_x1]
    按目标画布偏移 [dst_y0:dst_y1, dst_x0:dst_x1] 精确贴入
    (dst_y0 = src_y0 - (cy_int - half)), 保证病灶外接框几何中心严格位于裁剪图中心,
    保留率理论值 1.0000。末尾硬断言 img/mask 严格同形 (S,S)。

    返回 (crop224, S_px, size_px, mbox) — crop224=(224,224) float32 未 zscore,
    S_px=画布边长(px), size_px=(y像素,x像素) 画布裁剪边长, mbox=画布内掩码 (S_px,S_px) uint8。
    """
    ry = size_mm / 2.0 / py
    rx = size_mm / 2.0 / px
    half_y = int(np.ceil(ry)); half_x = int(np.ceil(rx))
    half = max(half_y, half_x)
    cy_int = int(round(cy)); cx_int = int(round(cx))
    S = 2 * half
    canvas = np.zeros((S, S), dtype=np.float32)
    mbox = np.zeros((S, S), dtype=np.uint8)
    # 原图内有效源范围 (越界侧 clamp)
    src_y0 = max(0, cy_int - half); src_y1 = min(va.shape[0], cy_int + half)
    src_x0 = max(0, cx_int - half); src_x1 = min(va.shape[1], cx_int + half)
    # 画布内放置偏移 (相对理论裁剪中心, 越界侧对应补 0)
    dst_y0 = src_y0 - (cy_int - half)
    dst_x0 = src_x0 - (cx_int - half)
    if src_y1 > src_y0 and src_x1 > src_x0:
        h = src_y1 - src_y0; w = src_x1 - src_x0
        canvas[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = va[src_y0:src_y1, src_x0:src_x1]
        mbox[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = m2[src_y0:src_y1, src_x0:src_x1]
    # 硬断言: 图像与掩码严格同形 (S,S)、居中对称、越界四边全部补 0
    assert canvas.shape == (S, S) and mbox.shape == (S, S), \
        (canvas.shape, mbox.shape, (cy_int, cx_int, half, size_mm))
    m_ = canvas.mean(); s_ = canvas.std() + 1e-6
    canvas = (canvas - m_) / s_
    from PIL import Image
    im = Image.fromarray(canvas)
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(im).astype(np.float32), S, (S, S), mbox


def slice_size_mm(span_mm, mode):
    """该层目标裁剪边长 (物理 mm)。mode 为 MODES 的字符串 key。"""
    size_mm, margin = MODES[mode][1], MODES[mode][2]
    if margin is None:
        return size_mm
    return max(size_mm, span_mm + 2 * margin)


def extract_slices(pid, seq, mri_path, model, device, mode_cfg):
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()
    px, py = sp[0], sp[1]
    va = sitk.GetArrayFromImage(vimg)  # (Z, Y, X)
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
    area_all = np.zeros(n_slices, dtype=np.float32)   # mask_area_ratio: 物理面积 / REF_MM2
    area_mm2 = np.zeros(n_slices, dtype=np.float32)   # mask_area_mm2: 掩码物理面积
    crop_mm = np.zeros(n_slices, dtype=np.float32)    # crop_size_mm: 该层裁剪边长
    cnt = 0
    for z in range(z_min, z_max + 1):
        m2 = ma[z]
        if m2.sum() == 0:
            continue
        ys, xs = np.nonzero(m2)
        cy, cx = bbox_mid(ys, xs)
        span_mm = max((xs.max() - xs.min() + 1) * px, (ys.max() - ys.min() + 1) * py)
        size_mm = slice_size_mm(span_mm, mode_cfg)
        t, S, size_px, _ = crop_clean(va[z], m2, cy, cx, size_mm, px, py)
        if S < 4:
            continue
        tt = torch.from_numpy(t)[None, None]
        tt = tt.repeat(1, 3, 1, 1).to(device).half()
        with torch.inference_mode():
            f = model(tt).float().cpu().numpy().ravel()
        feats_all[cnt] = f.astype(np.float16)
        idx_all[cnt] = z
        relz_all[cnt] = (z - z_min) / max(1, (z_max - z_min))
        mm2 = float(m2.sum()) * px * py
        area_mm2[cnt] = mm2
        area_all[cnt] = mm2 / REF_MM2
        crop_mm[cnt] = size_mm
        cnt += 1
    if cnt == 0:
        return None
    return (feats_all[:cnt], idx_all[:cnt], relz_all[:cnt], area_all[:cnt],
            area_mm2[:cnt], crop_mm[:cnt], source)


def process_pid(pid, seq, mri_path, model, device, mode_cfg, force=False):
    out_root, _, _ = MODES[mode_cfg]
    out_p = os.path.join(out_root, "dinov2", seq, f"{pid}.npz")
    if not force and os.path.exists(out_p) and os.path.getsize(out_p) > 100:
        return "skip"
    r = extract_slices(pid, seq, mri_path, model, device, mode_cfg)
    if r is None:
        if force and os.path.exists(out_p):
            os.remove(out_p)
        return "nomask"
    feats, idx, relz, area, mm2, cmm, source = r
    if not np.isfinite(feats).all():
        if force and os.path.exists(out_p):
            os.remove(out_p)
        return "nan"
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    tmp = os.path.join(os.path.dirname(out_p), f"{pid}_tmp.npz")
    np.savez_compressed(tmp, feats=feats, slice_idx=idx, rel_z=relz,
                        mask_area_ratio=area, mask_area_mm2=mm2, crop_size_mm=cmm,
                        mask_source=np.array([source]))
    os.replace(tmp, out_p)
    return f"{feats.shape[0]}sl"


def qc_render(pid, seq, mri_path, out_dir):
    """三列并排: 左=原120质心+clamp (复现 v2 算法), 中=120mm_clean, 右=tight60_m10。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()
    px, py = sp[0], sp[1]
    va = sitk.GetArrayFromImage(vimg)
    ma, _ = load_mask_self(pid, seq)
    if ma is None:
        print(f"  qc pid{pid}: no mask, skip"); return
    mimg = sitk.ReadImage(os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz"))
    mres = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
    ma = sitk.GetArrayFromImage(mres)
    z = int(np.argmax(ma.sum(axis=(1, 2))))
    m2 = ma[z]
    ys, xs = np.nonzero(m2)
    tot = int(m2.sum())
    # ---- 左: 原120质心+clamp (v2 算法) ----
    cy_old, cx_old = int(np.mean(ys)), int(np.mean(xs))
    ry_o = int(round(120 / 2 / py)); rx_o = int(round(120 / 2 / px))
    H, W = va.shape[1], va.shape[2]
    y0o, y1o = max(0, cy_old - ry_o), min(H, cy_old + ry_o)
    x0o, x1o = max(0, cx_old - rx_o), min(W, cx_old + rx_o)
    crop_o = va[z, y0o:y1o, x0o:x1o].astype(np.float32)
    Ho, Wo = crop_o.shape
    So = max(Ho, Wo)
    pad_o = np.zeros((So, So), dtype=np.float32)
    pad_o[:Ho, :Wo] = crop_o
    mbox_o = np.zeros((So, So), dtype=np.uint8)
    mbox_o[:Ho, :Wo] = m2[y0o:y1o, x0o:x1o]
    ret_o = int(m2[y0o:y1o, x0o:x1o].sum()) / tot
    # ---- 中: 120mm_clean ----
    cy_m, cx_m = bbox_mid(ys, xs)
    t_m, S_m, size_px_m, mbox_m = crop_clean(va[z], m2, cy_m, cx_m, 120.0, px, py)
    ret_m = int(mbox_m.sum()) / tot
    # ---- 右: tight60_m10 ----
    span_mm = max((xs.max() - xs.min() + 1) * px, (ys.max() - ys.min() + 1) * py)
    size_t = max(60.0, span_mm + 20.0)
    t_t, S_t, size_px_t, mbox_t = crop_clean(va[z], m2, cy_m, cx_m, size_t, px, py)
    ret_t = int(mbox_t.sum()) / tot
    # ---- 渲染 ----
    from PIL import Image as _PILImage
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (pad_o, mbox_o, f"OLD120 centroid+clamp\nS={So}px ret={ret_o:.4f}"),
        (t_m, mbox_m, f"120mm_clean bbox-mid\nS={S_m}px ret={ret_m:.4f}"),
        (t_t, mbox_t, f"tight60_m10 (span={span_mm:.1f})\nS={S_t}px ret={ret_t:.4f}"),
    ]
    for ax, (img, mb, title) in zip(axes, panels):
        mb_disp = np.asarray(_PILImage.fromarray(mb).resize((IMG_SIZE, IMG_SIZE), _PILImage.NEAREST))
        ax.imshow(img, cmap="gray")
        if mb.sum() > 0:
            ax.contour(mb_disp, levels=[0.5], colors="r", linewidths=1.2)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"pid{pid} seq={seq} z={z} mask_vox={tot} spacing=({px:.3f},{py:.3f})mm", fontsize=12)
    os.makedirs(out_dir, exist_ok=True)
    out_p = os.path.join(out_dir, f"{pid}_{seq}_clean_qc.png")
    fig.savefig(out_p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  qc pid{pid} {seq}: z={z} span_mm={span_mm:.1f} | old120 ret={ret_o:.4f} S={So}px "
          f"| clean120 ret={ret_m:.4f} S={S_m}px size_px={size_px_m} | tight60 ret={ret_t:.4f} S={S_t}px size_px={size_px_t}")
    print(f"    -> {out_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--smoke-pids", help="逗号分隔 pid")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qc-pids", help="逗号分隔 pid")
    ap.add_argument("--qc-seq", default="t")
    args = ap.parse_args()
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    t0 = time.time()
    model = load_backbone(device)
    print("dinov2 loaded in", round(time.time() - t0, 1), "s")
    mode_cfg = args.mode
    out_root, _, _ = MODES[mode_cfg]

    if args.qc_pids:
        raw = pd.read_csv(RAW).set_index("patient_id")
        for pid in [int(x) for x in args.qc_pids.split(",")]:
            col = SEQ_COL[args.qc_seq]
            mp = raw.loc[pid, col]
            if isinstance(mp, str) and os.path.exists(mp):
                qc_render(pid, args.qc_seq, mp, "outputs/crop_clean_qc")
            else:
                print(f"  qc pid{pid}: mri_path missing")
        return

    raw = pd.read_csv(RAW).set_index("patient_id")
    if args.smoke_pids:
        pids = [int(x) for x in args.smoke_pids.split(",")]
        seqs = ("t", "d")
    elif args.all:
        t_pids = sorted(int(x[:-4]) for x in os.listdir(os.path.join(SRC_ROOT, "dinov2", "t"))
                        if x.endswith(".npz"))
        man = pd.read_csv(MANIFEST)
        d_pids = sorted(man[(man["mask_source"] == "self") & (man["has_valid_mask"] == 1)]["pid"].astype(int).tolist())
        pids = sorted(set(t_pids) | set(d_pids))
        seqs = ("t", "d")
        print(f"[{args.mode}] t_pids={len(t_pids)} d_self_pids={len(d_pids)} union={len(pids)}")
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
            if args.all:
                if seq == "t" and pid not in t_pids:
                    continue
                if seq == "d" and pid not in d_pids:
                    continue
            col = SEQ_COL[seq]
            mp = raw.loc[pid, col]
            if not (isinstance(mp, str) and os.path.exists(mp)):
                continue
            info = process_pid(pid, seq, mp, model, device, mode_cfg, force=args.force)
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
    n_files = 0
    for seq in ("t", "d"):
        d_ = os.path.join(out_root, "dinov2", seq)
        n_files += len([x for x in os.listdir(d_) if x.endswith(".npz")]) if os.path.isdir(d_) else 0
    size_mb = 0
    for root, _, fs in os.walk(out_root):
        for f in fs:
            if f.endswith(".npz"):
                size_mb += os.path.getsize(os.path.join(root, f))
    print(f"OUT files={n_files} size={round(size_mb/1e6,2)}MB")
    if args.all:
        with open(f"logs/extract_{args.mode}_summary.json", "w", encoding="utf-8") as f:
            json.dump({"mode": args.mode, "done": done, "skipped": skipped, "nomask": nomask,
                       "total_s": round(total, 1), "per_case_s": round(total / max(1, done), 2),
                       "peak_cuda_gb": round(peak, 2), "n_files": n_files,
                       "size_mb": round(size_mb / 1e6, 2), "size_mm": MODES[args.mode][1]},
                      f, ensure_ascii=False, indent=2)
        print(f"DONE -> logs/extract_{args.mode}_summary.json")


if __name__ == "__main__":
    main()
