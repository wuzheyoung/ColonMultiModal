# -*- coding: utf-8 -*-
"""DWI 物理坐标掩码借壳救回 (m04879 阶段1/2)。
掩码优先级(方案C, m05840 2026-11-23): self(原 mr_d 有效,体素>=50) → from_t(mr_t 重采样) → center_fallback。
**from_c 已禁用**: c/d 卷 z-origin 系统性差 -166.8mm 且 d 斜位采集 → c/d 不共享物理坐标系, from_c 借壳系统性错位。
输出 data/masks_dwi_remapped/{pid}/colon.nii.gz (严禁覆盖原掩码) + features/dwi_mask_remap_manifest.csv + logs/dwi_mask_remap_summary.csv(含质心偏移/mask_ratio)。
冒烟: --pids 51,88,46 强制借壳算 Dice/质心偏移并导叠图; --sample-fallback 3 从 direct 全零组随机抽 3 例导图。
"""
import os, json, csv, random
import numpy as np
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MASTER = "data/master_5fold_356.csv"
RAW = "data/hos_pairs_raw.csv"
COHORT_CSV = "data/direct_cohort_ids.csv"  # P1(1650 审计): 勘误后锁定名单, 弃用 xlsx 逐例解析
MR_D = "data/masks_totalseg_mr_d"
MR_C = "data/masks_totalseg_mr_c"
MR_T = "data/masks_totalseg_mr_t"
OUT_DIR = "data/masks_dwi_remapped"
QC_DIR = "outputs/dwi_remap_qc"
MIN_VOX = 50


def mask_ok(mask_path, min_vox=MIN_VOX):
    if not os.path.exists(mask_path):
        return False
    try:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        return int(arr.sum()) >= min_vox
    except Exception:
        return False


def remap_mask(src_mask_path, target_img_path, out_path):
    """将 src mask 经物理世界坐标重采样至 target 影像网格 (最近邻, Identity)。"""
    mimg = sitk.ReadImage(src_mask_path)
    vimg = sitk.ReadImage(target_img_path)
    res = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                        vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(),
                        0, sitk.sitkUInt8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(res, out_path)
    return res


def centroid_mm(img):
    arr = sitk.GetArrayFromImage(img)
    if arr.sum() == 0:
        return None
    z, y, x = np.nonzero(arr)
    phys = img.TransformIndexToPhysicalPoint
    # 体素索引 (i=x, j=y, k=z) 与 array 轴对应: arr[z,y,x] -> idx (x,y,z)
    pts = np.array([phys((int(xx), int(yy), int(zz))) for zz, yy, xx in zip(z, y, x)])
    return pts.mean(axis=0)


def dice_arr(a, b):
    i = float(np.logical_and(a > 0, b > 0).sum())
    u = float(a.sum() + b.sum())
    return 2 * i / u if u > 0 else 1.0


def draw_qc(pid, mri_path, mask_path, tag, layer_fracs=(0.25, 0.5, 0.75)):
    """导出上/中/下 3 层 mri_d 叠画 mask 轮廓 PNG (matplotlib 若可用, 否则 PIL 灰度+红点)。"""
    os.makedirs(QC_DIR, exist_ok=True)
    vimg = sitk.ReadImage(mri_path)
    va = sitk.GetArrayFromImage(vimg)
    mimg = sitk.ReadImage(mask_path)
    ma = sitk.GetArrayFromImage(mimg)
    nz = np.nonzero(ma)
    if len(nz[0]) == 0:
        zs = np.linspace(0, va.shape[0] - 1, 3, dtype=int)
    else:
        zmin, zmax = nz[0].min(), nz[0].max()
        zs = [int(round(zmin + (zmax - zmin) * f)) for f in layer_fracs]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, z in zip(axes, zs):
            ax.imshow(va[z], cmap="gray", vmin=np.percentile(va[z], 1), vmax=np.percentile(va[z], 99))
            mz = ma[z] > 0
            if mz.any():
                ax.contour(mz, colors="r", linewidths=1.2)
            ax.set_title(f"z={z} {tag}")
            ax.axis("off")
        fig.suptitle(f"pid={pid} {tag} remap QC")
        fig.savefig(os.path.join(QC_DIR, f"qc_p{pid}_{tag}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        return os.path.join(QC_DIR, f"qc_p{pid}_{tag}.png")
    except Exception as e:
        from PIL import Image, ImageDraw
        print("matplotlib unavailable, PIL fallback:", e)
        out = []
        for z in zs:
            im = va[z].astype(np.float32)
            im = (im - im.min()) / (im.max() - im.min() + 1e-9) * 255
            img = Image.fromarray(im.astype(np.uint8)).convert("RGB")
            d = ImageDraw.Draw(img)
            mz = np.argwhere(ma[z] > 0)
            if len(mz):
                for yy, xx in mz[::max(1, len(mz) // 500)]:
                    d.point((int(xx), int(yy)), fill=(255, 0, 0))
            p = os.path.join(QC_DIR, f"qc_p{pid}_{tag}_z{z}.png")
            img.save(p)
            out.append(p)
        return ",".join(out)


def calc_qc(mask_path, mri_d_path):
    """mask 质心 vs mri_d 几何中心三轴偏移(mm) + mask 区域 DWI 信号比(mask_ratio)。
    返回 (dx, dy, dz, mask_ratio); 掩码缺失/异常返回 (None,)*4。"""
    if not mask_path or not os.path.exists(mask_path):
        return None, None, None, None
    try:
        vimg = sitk.ReadImage(mri_d_path)
        mimg = sitk.ReadImage(mask_path)
        if mimg.GetSize() != vimg.GetSize() or mimg.GetSpacing() != vimg.GetSpacing():
            mimg = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                                 vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
        c_mask = centroid_mm(mimg)
        if c_mask is None:
            return None, None, None, None
        sz = vimg.GetSize()
        c_vol = vimg.TransformIndexToPhysicalPoint((sz[0] // 2, sz[1] // 2, sz[2] // 2))
        d = np.array(c_mask) - np.array(c_vol)
        va = sitk.GetArrayFromImage(vimg).astype(np.float32)
        ma = sitk.GetArrayFromImage(mimg)
        nz = np.nonzero(ma)
        nz_all = va[va > 0]
        ratio = float(np.median(va[nz])) / float(np.median(nz_all)) if len(nz[0]) and len(nz_all) else None
        return float(d[0]), float(d[1]), float(d[2]), ratio
    except Exception:
        return None, None, None, None


def process_pid(pid, row, manifest, force_from=None, cohort_direct=-1):
    """force_from: None=按优先级; 't'=强制借壳(冒烟用)。返回 (source, ok, note)。
    P2(1650 审计): from_t 重采样后 <MIN_VOX 的残片不落盘为有效掩码(删除文件)。
    方案C(m05840): from_c 禁用, 仅 self → from_t → center_fallback。"""
    mri_d = row.get("mri_path_d")
    if not mri_d or not os.path.exists(mri_d):
        return None, False, "no_mri_d"
    out_path = os.path.join(OUT_DIR, str(pid), "colon.nii.gz")
    self_path = os.path.join(MR_D, str(pid), "colon.nii.gz")
    t_path = os.path.join(MR_T, str(pid), "colon.nii.gz")
    self_ok = mask_ok(self_path)
    t_ok = mask_ok(t_path)
    source, src_path = None, None
    if force_from == "t" and t_ok:
        source, src_path = "from_t", t_path
    elif self_ok:
        # 原样复制 (保持原掩码不变, 不重采样)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import shutil
        shutil.copyfile(self_path, out_path)
        source = "self"
    elif t_ok:
        source, src_path = "from_t", t_path
    else:
        source = "center_fallback"
    orig_vox, final_vox, has_valid = 0, 0, False
    if src_path is not None:
        orig_arr = sitk.GetArrayFromImage(sitk.ReadImage(self_path)) if os.path.exists(self_path) else None
        orig_vox = int(orig_arr.sum()) if orig_arr is not None else 0
        remap_mask(src_path, mri_d, out_path)
        final_vox = int(sitk.GetArrayFromImage(sitk.ReadImage(out_path)).sum())
        has_valid = final_vox >= MIN_VOX
        if not has_valid:
            # P2: 残片(<50 体素)不留有效掩码, 删除落盘, extract 端将视为 missing
            try:
                os.remove(out_path)
            except OSError:
                pass
    elif source == "self":
        orig_vox = final_vox = int(sitk.GetArrayFromImage(sitk.ReadImage(out_path)).sum())
        has_valid = final_vox >= MIN_VOX
    elif source == "center_fallback":
        # 兜底: 不写掩码文件, 特征提取时按盆腔几何中心固定物理框
        orig_vox = final_vox = 0
        has_valid = False
    dx, dy, dz, ratio = calc_qc(out_path if has_valid else None, mri_d)
    manifest.append({"pid": int(pid), "cohort_direct": int(cohort_direct),
                     "mask_source": source, "orig_voxels": orig_vox, "final_voxels": final_vox,
                     "has_valid_mask": int(has_valid),
                     "centroid_dx_mm": dx, "centroid_dy_mm": dy, "centroid_dz_mm": dz,
                     "mask_ratio": ratio})
    return source, has_valid, "ok"


def main():
    import argparse, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", help="逗号分隔 pid 列表 (冒烟强制借壳 + 叠图)")
    ap.add_argument("--sample-fallback", type=int, default=0, help="从 direct 全零组随机抽 N 例导叠图")
    ap.add_argument("--all", action="store_true", help="全量 (阶段2)")
    args = ap.parse_args()
    master = pd.read_csv(MASTER)
    raw = pd.read_csv(RAW).set_index("patient_id")
    # P1(1650 审计): 弃用 xlsx 逐例解析 load_tgr_map, 改读勘误后锁定名单 direct_cohort_ids.csv (列: pid,is_tgr)
    cohort = pd.read_csv(COHORT_CSV)
    _cm = dict(zip(cohort["pid"].astype(int), cohort["is_tgr"].astype(int)))
    cohort_map = {p: (1 - int(t)) for p, t in _cm.items()}  # 1=direct, 0=ncrt
    master["cohort_direct"] = master["patient_id"].astype(int).map(lambda p: cohort_map.get(p, 0))
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    force_pids = [int(x) for x in args.pids.split(",")] if args.pids else []

    if args.sample_fallback:
        # 找出 direct 且原 mr_d 掩码全零/缺失的病例
        cand = []
        for _, r in master.iterrows():
            pid = int(r["patient_id"])
            if not r["cohort_direct"]:
                continue
            if pid in raw.index and pd.notna(raw.loc[pid, "mri_path_d"]):
                p = os.path.join(MR_D, str(pid), "colon.nii.gz")
                if not mask_ok(p):
                    cand.append(pid)
        random.seed(0)
        sel = random.sample(cand, min(args.sample_fallback, len(cand)))
        print(f"fallback candidates={len(cand)} sampled={sel}")
        for pid in sel:
            row = raw.loc[pid]
            source, ok, note = process_pid(pid, row, manifest, cohort_direct=cohort_map.get(pid, -1))
            if source == "from_t":
                outp = os.path.join(OUT_DIR, str(pid), "colon.nii.gz")
                print(f"  pid{pid}: {source} ok={ok} → {draw_qc(pid, row['mri_path_d'], outp, source)}")
            else:
                print(f"  pid{pid}: {source} (skip qc)")

    if force_pids:
        print("== 强制借壳冒烟 (from_t 仅) ==")
        for pid in force_pids:
            if pid not in raw.index:
                print(f"  pid{pid} not in raw, skip")
                continue
            row = raw.loc[pid]
            self_path = os.path.join(MR_D, str(pid), "colon.nii.gz")
            self_img = sitk.ReadImage(self_path) if os.path.exists(self_path) else None
            src_path = os.path.join(MR_T, str(pid), "colon.nii.gz")
            if not os.path.exists(src_path):
                print(f"  pid{pid} from_t: src mask missing")
            else:
                out_path = os.path.join(OUT_DIR, str(pid), f"colon_from_t.nii.gz")
                rim = remap_mask(src_path, row["mri_path_d"], out_path)
                ra = sitk.GetArrayFromImage(rim)
                if self_img is not None:
                    sa = sitk.GetArrayFromImage(self_img)
                    dice = dice_arr(sa, ra)
                else:
                    dice = None
                c_self = centroid_mm(self_img) if self_img is not None else None
                c_map = centroid_mm(rim)
                off = None
                if c_self is not None and c_map is not None:
                    off = float(np.linalg.norm(c_map - c_self))
                off_str = "n/a" if off is None else f"{off:.2f}mm"
                print(f"  pid{pid} from_t: dice={dice if dice is None else round(dice,3)} "
                      f"centroid_off={off_str} vox={int(ra.sum())}")
                print(f"    → {draw_qc(pid, row['mri_path_d'], out_path, 'from_t')}")
                # 也导出 self 对照图
                if self_img is not None:
                    sp = os.path.join(OUT_DIR, str(pid), "colon_self.png")
                    os.makedirs(os.path.dirname(sp), exist_ok=True)
                    import shutil
                    shutil.copyfile(self_path, os.path.join(OUT_DIR, str(pid), "colon_self.nii.gz"))
                    print(f"    self qc → {draw_qc(pid, row['mri_path_d'], os.path.join(OUT_DIR, str(pid), 'colon_self.nii.gz'), 'self')}")
            # 冒烟同时写正式 manifest 行
            source, ok, note = process_pid(pid, row, manifest, cohort_direct=cohort_map.get(pid, -1))
            print(f"  pid{pid} formal: {source} ok={ok}")

    if args.all:
        print("== 全量借壳 (方案C: self → from_t, from_c 禁用) ==")
        # 方案C(m05840): 清理旧 from_c 写入的掩码, 避免残留
        old_man = "features/dwi_mask_remap_manifest.csv"
        if os.path.exists(old_man):
            om = pd.read_csv(old_man)
            if "mask_source" in om.columns:
                fc = om[om["mask_source"] == "from_c"]["pid"].astype(int).tolist()
                removed = 0
                for p in fc:
                    fp = os.path.join(OUT_DIR, str(p), "colon.nii.gz")
                    if os.path.exists(fp):
                        try:
                            os.remove(fp)
                            removed += 1
                        except OSError:
                            pass
                print(f"cleaned {removed}/{len(fc)} old from_c masks")
        n_self = n_t = n_fb = n_no = 0
        # P2(1650 审计): 按 direct_all(280) 与 direct_tcd(162) 分层统计
        n_self_da = n_t_da = n_fb_da = 0
        n_self_tcd = n_t_tcd = n_fb_tcd = 0
        for _, r in master.iterrows():
            pid = int(r["patient_id"])
            if pid not in raw.index or pd.isna(raw.loc[pid, "mri_path_d"]):
                continue
            row = raw.loc[pid]
            is_direct = bool(r["cohort_direct"])
            has_tcd = all(isinstance(raw.loc[pid, c], str) and os.path.exists(raw.loc[pid, c])
                          for c in ("mri_path_t", "mri_path_c", "mri_path_d"))
            source, ok, note = process_pid(pid, row, manifest, cohort_direct=int(r["cohort_direct"]))
            if source == "self": n_self += 1
            elif source == "from_t": n_t += 1
            elif source == "center_fallback": n_fb += 1
            else: n_no += 1
            if is_direct:
                if source == "self": n_self_da += 1
                elif source == "from_t": n_t_da += 1
                elif source == "center_fallback": n_fb_da += 1
            if is_direct and has_tcd:
                if source == "self": n_self_tcd += 1
                elif source == "from_t": n_t_tcd += 1
                elif source == "center_fallback": n_fb_tcd += 1
        print(f"all: self={n_self} from_t={n_t} center_fallback={n_fb} no_mri_d={n_no}")
        print(f"direct_all(280): self={n_self_da} from_t={n_t_da} center_fallback={n_fb_da}")
        print(f"direct_tcd(162): self={n_self_tcd} from_t={n_t_tcd} center_fallback={n_fb_tcd}")

    man_path = "features/dwi_mask_remap_manifest.csv"
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    if manifest:
        with open(man_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pid", "cohort_direct", "mask_source", "orig_voxels",
                                              "final_voxels", "has_valid_mask",
                                              "centroid_dx_mm", "centroid_dy_mm", "centroid_dz_mm",
                                              "mask_ratio"])
            w.writeheader()
            w.writerows(manifest)
        # 方案C(m05840): 额外导出精简 summary (pid, cohort_direct, mask_source, has_valid_mask, n_vox, centroid_shift_mm, mask_ratio)
        import numpy as _np
        sum_rows = []
        for r in manifest:
            dx, dy, dz = r.get("centroid_dx_mm"), r.get("centroid_dy_mm"), r.get("centroid_dz_mm")
            shift = None
            if dx is not None and dy is not None and dz is not None:
                shift = round(float(_np.linalg.norm([dx, dy, dz])), 2)
            sum_rows.append({"pid": r["pid"], "cohort_direct": r["cohort_direct"],
                             "mask_source": r["mask_source"], "has_valid_mask": r["has_valid_mask"],
                             "n_vox": r["final_voxels"], "centroid_shift_mm": shift,
                             "mask_ratio": r.get("mask_ratio")})
        sum_path = "logs/dwi_mask_remap_summary.csv"
        os.makedirs(os.path.dirname(sum_path), exist_ok=True)
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pid", "cohort_direct", "mask_source", "has_valid_mask",
                                              "n_vox", "centroid_shift_mm", "mask_ratio"])
            w.writeheader()
            w.writerows(sum_rows)
        print(f"manifest → {man_path} ({len(manifest)} rows)")
        print(f"summary → {sum_path} ({len(sum_rows)} rows)")


if __name__ == "__main__":
    main()
