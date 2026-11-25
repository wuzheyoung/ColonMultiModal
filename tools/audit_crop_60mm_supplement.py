# -*- coding: utf-8 -*-
"""补充核查 (m06997 Q1/Q2): bbox 中点+对称补黑保留率对照 + span>120mm 切片连通域成分分析。
- Q1: 固定60mm/TightROI 均用质心 mean 定心 (复现审计脚本); 新增 bbox 中点定心 + 真对称补黑(先算完整框, 越界侧 pad 不 clamp) 模拟, 报告保留率。
- Q2: pid240(d) 逐层连通域构成 + 全队列 span>120mm 切片的主/次连通域像素数与占比, 判断极端跨度成分。
只读, 不改特征目录。
"""
import os, json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

ROOT = r"D:\PyFile\ColonMultiModal"
SRC_ROOT = os.path.join(ROOT, "features", "slice_feats_v2", "dinov2")
MANIFEST = os.path.join(ROOT, "features", "dwi_mask_remap_manifest.csv")
MASK_BASE = {s: os.path.join(ROOT, "data", f"masks_totalseg_mr_{s}") for s in ("t", "d")}
RAW = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
SEQ_COL = {"t": "mri_path_t", "d": "mri_path_d"}


def get_ma(pid, seq, mri_path):
    mimg = sitk.ReadImage(os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz"))
    vimg = sitk.ReadImage(mri_path)
    mres = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(mres), vimg.GetSpacing()


def main():
    raw = pd.read_csv(RAW).set_index("patient_id")
    t_pids = sorted(int(x[:-4]) for x in os.listdir(os.path.join(SRC_ROOT, "t")) if x.endswith(".npz"))
    man = pd.read_csv(MANIFEST)
    d_pids = sorted(man[(man["mask_source"] == "self") & (man["has_valid_mask"] == 1)]["pid"].astype(int).tolist())

    # ---------- Q1: 三种定心/补黑模拟保留率 ----------
    sim = {"t": {}, "d": {}}
    for seq, pids in (("t", t_pids), ("d", d_pids)):
        # 只采样全部病例, 收集所有层
        ret_fix_cent = []   # 固定60 质心+clamp (现状)
        ret_tight_cent = []  # TightROI 质心+clamp (审计脚本现状)
        ret_tight_bbox = []  # TightROI bbox中点+真对称补黑
        for pid in pids:
            col = SEQ_COL[seq]
            mp = raw.loc[pid, col]
            if not (isinstance(mp, str) and os.path.exists(mp)):
                continue
            ma, sp = get_ma(pid, seq, mp)
            px, py = sp[0], sp[1]
            nz = np.nonzero(ma)
            if len(nz[0]) == 0:
                continue
            z_min, z_max = nz[0].min(), nz[0].max()
            for z in range(z_min, z_max + 1):
                m2 = ma[z]
                if m2.sum() == 0:
                    continue
                ys, xs = np.nonzero(m2)
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                ymin, ymax, xmin, xmax = ys.min(), ys.max(), xs.min(), xs.max()
                W_mm = (xmax - xmin + 1) * px
                H_mm = (ymax - ymin + 1) * py
                span = max(W_mm, H_mm)
                tot = int(m2.sum())
                # 固定 60 质心+clamp
                ry = int(round(60 / 2 / py)); rx = int(round(60 / 2 / px))
                y0, y1 = max(0, cy - ry), min(ma.shape[1], cy + ry)
                x0, x1 = max(0, cx - rx), min(ma.shape[2], cx + rx)
                ret_fix_cent.append(int(m2[y0:y1, x0:x1].sum()) / tot)
                # Tight ROI 质心+clamp (margin=10)
                Lmm = max(60.0, span + 20)
                ry_ = int(round(Lmm / 2 / py)); rx_ = int(round(Lmm / 2 / px))
                y0, y1 = max(0, cy - ry_), min(ma.shape[1], cy + ry_)
                x0, x1 = max(0, cx - rx_), min(ma.shape[2], cx + rx_)
                ret_tight_cent.append(int(m2[y0:y1, x0:x1].sum()) / tot)
                # Tight ROI bbox 中点 + 真对称补黑: 框覆盖整个 bbox+margin, 越界侧 pad(不 clamp)
                ymid = (ymin + ymax) / 2.0
                xmid = (xmin + xmax) / 2.0
                ry_b = int(round(Lmm / 2 / py)); rx_b = int(round(Lmm / 2 / px))
                y0 = max(0, int(round(ymid - ry_b))); y1 = min(ma.shape[1], int(round(ymid + ry_b)))
                x0 = max(0, int(round(xmid - rx_b))); x1 = min(ma.shape[2], int(round(xmid + rx_b)))
                # 掩码只要在 [ymin,ymax]x[xmin,xmax] 内且框半径>=span/2+margin 就应全含;
                # 但因 ymid/xmid 可能使一侧留白不足, 显式计算框内掩码
                ret_tight_bbox.append(int(m2[y0:y1, x0:x1].sum()) / tot)
        for name, arr in (("fix60_centroid", ret_fix_cent), ("tight10_centroid", ret_tight_cent),
                          ("tight10_bboxmid", ret_tight_bbox)):
            a = np.array(arr)
            sim[seq][name] = {"n": len(a), "mean": float(a.mean()), "P50": float(np.median(a)),
                              "lt1": int((a < 0.999).sum()), "lt95": int((a < 0.95).sum())}
            print(f"[{seq}] {name}: n={len(a)} mean={a.mean():.4f} P50={np.median(a):.4f} "
                  f"<0.999: {int((a<0.999).sum())} <0.95: {int((a<0.95).sum())}")

    # ---------- Q2: span>120mm 切片连通域成分 + pid240 明细 ----------
    big_cc_report = {"t": [], "d": []}
    pid240_detail = []
    for seq, pids in (("t", t_pids), ("d", d_pids)):
        for pid in pids:
            col = SEQ_COL[seq]
            mp = raw.loc[pid, col]
            if not (isinstance(mp, str) and os.path.exists(mp)):
                continue
            ma, sp = get_ma(pid, seq, mp)
            px, py = sp[0], sp[1]
            nz = np.nonzero(ma)
            if len(nz[0]) == 0:
                continue
            z_min, z_max = nz[0].min(), nz[0].max()
            for z in range(z_min, z_max + 1):
                m2 = ma[z]
                if m2.sum() == 0:
                    continue
                ys, xs = np.nonzero(m2)
                ymin, ymax, xmin, xmax = ys.min(), ys.max(), xs.min(), xs.max()
                span = max((xmax - xmin + 1) * px, (ymax - ymin + 1) * py)
                if span <= 120:
                    continue
                lab, n_cc = ndimage.label(m2, structure=np.ones((3, 3), dtype=int))
                ccs = []
                tot = int(m2.sum())
                for c in range(1, n_cc + 1):
                    n = int((lab == c).sum())
                    ccs.append({"cc": c, "vox": n, "pct": round(100 * n / tot, 2)})
                ccs_sorted = sorted(ccs, key=lambda c: -c["vox"])
                main = ccs_sorted[0]
                sec = ccs_sorted[1:] if len(ccs_sorted) > 1 else []
                sec_vox = sum(c["vox"] for c in sec)
                rec = {"pid": pid, "seq": seq, "z": z, "span_mm": round(span, 1),
                       "n_cc": n_cc, "tot_vox": tot, "main": main,
                       "sec_vox": sec_vox, "sec_pct": round(100 * sec_vox / tot, 2),
                       "secs": ccs_sorted[1:3]}
                big_cc_report[seq].append(rec)
                if pid == 240 and seq == "d":
                    pid240_detail.append(rec)
    print("\n=== pid240 (d) span>120mm 层明细 ===")
    for r in pid240_detail:
        print(f"  z={r['z']} span={r['span_mm']} n_cc={r['n_cc']} tot={r['tot_vox']} "
              f"main={r['main']} sec_vox={r['sec_vox']} sec_pct={r['sec_pct']}%")
    print("  pid240 全部层(不限span>120): 见下方单独输出")
    # pid240 所有层
    pid240_all = []
    mp = raw.loc[240, "mri_path_d"]
    ma, sp = get_ma(240, "d", mp)
    px, py = sp[0], sp[1]
    nz = np.nonzero(ma)
    z_min, z_max = nz[0].min(), nz[0].max()
    for z in range(z_min, z_max + 1):
        m2 = ma[z]
        if m2.sum() == 0:
            continue
        ys, xs = np.nonzero(m2)
        span = max((xs.max() - xs.min() + 1) * px, (ys.max() - ys.min() + 1) * py)
        lab, n_cc = ndimage.label(m2, structure=np.ones((3, 3), dtype=int))
        ccs = []
        tot = int(m2.sum())
        for c in range(1, n_cc + 1):
            n = int((lab == c).sum())
            ccs.append({"cc": c, "vox": n, "pct": round(100 * n / tot, 2)})
        ccs_sorted = sorted(ccs, key=lambda c: -c["vox"])
        pid240_all.append({"z": z, "span": round(span, 1), "n_cc": n_cc, "tot": tot,
                           "main": ccs_sorted[0], "secs": ccs_sorted[1:3]})
    for r in pid240_all:
        print(f"  z={r['z']} span={r['span']} n_cc={r['n_cc']} tot={r['tot']} main={r['main']} secs={r['secs']}")

    for seq in ("t", "d"):
        br = big_cc_report[seq]
        sec_vox_list = [r["sec_vox"] for r in br]
        sec_pct_list = [r["sec_pct"] for r in br]
        print(f"\n=== {seq}: span>120mm 切片数={len(br)} ===")
        if len(br):
            sv = np.array(sec_vox_list); spc = np.array(sec_pct_list)
            print(f"  次要连通域总像素 sec_vox: min={sv.min()} P50={np.median(sv):.0f} max={sv.max()}")
            print(f"  次要占比 sec_pct%: min={spc.min():.2f} P50={np.median(spc):.2f} max={spc.max():.2f}")
            print(f"  sec_vox<=100 切片数: {int((sv<=100).sum())}  ({100*(sv<=100).mean():.1f}%)")
            print(f"  sec_pct<=5% 切片数: {int((spc<=5).sum())}  ({100*(spc<=5).mean():.1f}%)")
            print(f"  最大 span 前 5 例:")
            for r in sorted(br, key=lambda x: -x["span_mm"])[:5]:
                print(f"    {r['seq']}/{r['pid']} z={r['z']} span={r['span_mm']} n_cc={r['n_cc']} "
                      f"tot={r['tot_vox']} main={r['main']} sec_vox={r['sec_vox']} sec_pct={r['sec_pct']}%")

    out = {"sim": sim, "pid240": pid240_all,
           "span_gt120": {s: {"n": len(big_cc_report[s]),
                               "sec_vox_min": int(np.min([r["sec_vox"] for r in big_cc_report[s]])) if big_cc_report[s] else None,
                               "sec_pct_median": float(np.median([r["sec_pct"] for r in big_cc_report[s]])) if big_cc_report[s] else None} for s in ("t", "d")}}
    with open(os.path.join(ROOT, "logs", "audit_crop_60mm_supplement.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("\nDONE -> logs/audit_crop_60mm_supplement.json")


if __name__ == "__main__":
    main()
