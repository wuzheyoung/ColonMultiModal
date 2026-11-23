# -*- coding: utf-8 -*-
"""DWI 三骨干冻结逐层特征离线提取 (m04879)。
- 严格单进程串行 backbone→seq→pid; num_workers=0; torch.inference_mode; autocast float16。
- 切片逻辑与 route2 (audit_tgr_task2c.extract_case_feats) 完全一致:
  120mm 物理方框 / 每层 mask 质心 / 保真不拉伸 / 边界不足 pad / zscore / PIL BILINEAR 224 / 单层复制 3 通道。
- 掩码: 优先 data/masks_dwi_remapped/{pid}/colon.nii.gz (借壳救回), 回退 masks_totalseg_mr_{seq}。
- 落盘 features/slice_feats_v2/{backbone}/{seq}/{pid}.npz 原子写(.tmp→os.replace), 已存在非空跳过。
  npz: feats(K,D)float16 逐层矩阵 / slice_idx int32 / rel_z float32 / mask_area_ratio float32 / mask_source str。
- 冒烟: --smoke-pids 51,88,46 (端到端, 报单例耗时/显存/无NaN); 阶段2: --all (晨间硬截断由外部控制)。
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
OUT_ROOT = "features/slice_feats_v2"
MASK_REMAP = "data/masks_dwi_remapped"
MASK_BASE = {s: f"data/masks_totalseg_mr_{s}" for s in ("t", "c", "d")}
SIZE_MM = 120.0
IMG_SIZE = 224
BATCH = 16

SEQ_COL = {"t": "mri_path_t", "c": "mri_path_c", "d": "mri_path_d"}


def load_backbone(name, device):
    """离线加载三骨干, 返回 (model, mean, std, norm_kind)。全部 eval 冻结。"""
    if name == "convnext_tiny":
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif name == "dinov2":
        model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False,
                                  num_classes=0, dynamic_img_size=True)
        from safetensors.torch import load_file as _sl
        st = _sl("pretrained/dinov2_vitb14_reg4/model.safetensors", device="cpu")
        model.load_state_dict(st, strict=True)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif name == "biomedclip":
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        sd = torch.load("pretrained/biomedclip_vitb16/open_clip_pytorch_model.bin", map_location="cpu")
        trunk = {k.replace("visual.trunk.", ""): v for k, v in sd.items() if k.startswith("visual.trunk.")}
        model.load_state_dict(trunk, strict=True)
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    else:
        raise ValueError(name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).half(), mean, std


def load_mask(pid, seq):
    """优先级: 借壳掩码 → 原掩码。返回 (mask_arr[z,y,x], source)。"""
    remap_p = os.path.join(MASK_REMAP, str(pid), "colon.nii.gz")
    if os.path.exists(remap_p):
        try:
            a = sitk.GetArrayFromImage(sitk.ReadImage(remap_p))
            if int(a.sum()) > 0:
                return a, "remapped"
        except Exception:
            pass
    orig_p = os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz")
    if os.path.exists(orig_p):
        a = sitk.GetArrayFromImage(sitk.ReadImage(orig_p))
        if int(a.sum()) > 0:
            return a, "self"
    return None, "missing"


def extract_slices(pid, seq, mri_path, model, mean, std, device):
    """返回 (feats_list[K,768 float16], slice_idx, rel_z, mask_area_ratio, source)。"""
    vimg = sitk.ReadImage(mri_path)
    sp = vimg.GetSpacing()
    va = sitk.GetArrayFromImage(vimg)
    ma, source = load_mask(pid, seq)
    if ma is None:
        return None
    # 重采样 mask 到 MRI 网格(route2 同款)
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
        t = torch.from_numpy(np.asarray(im).astype(np.float32))[None, None]
        t = t.repeat(1, 3, 1, 1)
        # 规范化: route2 用 zscore (已做), 三骨干 mean/std 不再叠加 (保持一致)
        t = t.to(device).half()
        with torch.inference_mode():
            f = model(t).float().cpu().numpy().ravel()
        feats_all[cnt] = f.astype(np.float16)
        idx_all[cnt] = z
        relz_all[cnt] = (z - z_min) / max(1, (z_max - z_min))
        area_all[cnt] = float(m2.sum()) / float(S * S)
        cnt += 1
    if cnt == 0:
        return None
    return feats_all[:cnt], idx_all[:cnt], relz_all[:cnt], area_all[:cnt], source


def process_pid(pid, seq, mri_path, backbones, device):
    """三骨干各存一个 npz (同一 pid/seq)。返回 (ok, info)。"""
    infos = []
    for name, (model, mean, std) in backbones.items():
        out_p = os.path.join(OUT_ROOT, name, seq, f"{pid}.npz")
        if os.path.exists(out_p) and os.path.getsize(out_p) > 100:
            infos.append(f"{name}:skip")
            continue
        r = extract_slices(pid, seq, mri_path, model, mean, std, device)
        if r is None:
            infos.append(f"{name}:nomask")
            continue
        feats, idx, relz, area, source = r
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        tmp = os.path.join(os.path.dirname(out_p), f"{pid}_tmp.npz")
        np.savez_compressed(tmp, feats=feats, slice_idx=idx, rel_z=relz,
                            mask_area_ratio=area, mask_source=np.array([source]))
        os.replace(tmp, out_p)
        infos.append(f"{name}:{feats.shape[0]}sl")
    return infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-pids", help="逗号分隔 pid (端到端冒烟)")
    ap.add_argument("--all", action="store_true", help="全量 (阶段2)")
    ap.add_argument("--limit", type=int, default=0, help="调试: 最多处理 pid 数")
    args = ap.parse_args()
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "mem:", torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == "cuda" else 0)
    t0 = time.time()
    backbones = {n: load_backbone(n, device) for n in ("convnext_tiny", "dinov2", "biomedclip")}
    print("backbones loaded in", round(time.time() - t0, 1), "s")
    master = pd.read_csv(MASTER)
    raw = pd.read_csv(RAW).set_index("patient_id")
    seqs = ("t", "c", "d")
    if args.smoke_pids:
        pids = [int(x) for x in args.smoke_pids.split(",")]
    elif args.all:
        pids = sorted(master["patient_id"].astype(int).tolist())
    else:
        print("需 --smoke-pids 或 --all"); return
    if args.limit:
        pids = pids[:args.limit]
    done, skipped = 0, 0
    t_start = time.time()
    peak = 0
    for pid in pids:
        if pid not in raw.index:
            skipped += 1; continue
        for seq in seqs:
            col = SEQ_COL[seq]
            mp = raw.loc[pid, col]
            if not (isinstance(mp, str) and os.path.exists(mp)):
                continue
            infos = process_pid(pid, seq, mp, backbones, device)
            if infos:
                done += 1
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        if (done + skipped) % 20 == 0:
            torch.cuda.empty_cache(); gc.collect()
        if args.smoke_pids:
            print(f"  pid{pid}: {infos}  (elapsed {round(time.time()-t_start,1)}s)")
    total = time.time() - t_start
    print(f"done={done} skipped={skipped} total={round(total,1)}s "
          f"({round(total/max(1,done),2)}s/case) peak_cuda={round(peak,2)}GB")
    if args.all:
        with open("logs/nightly_extract_v2_summary.json", "w", encoding="utf-8") as f:
            json.dump({"done": done, "skipped": skipped, "total_s": round(total, 1),
                       "per_case_s": round(total / max(1, done), 2),
                       "peak_cuda_gb": round(peak, 2), "backbones": list(backbones)},
                      f, ensure_ascii=False, indent=2)
        print("DONE → logs/nightly_extract_v2_summary.json")


if __name__ == "__main__":
    main()
