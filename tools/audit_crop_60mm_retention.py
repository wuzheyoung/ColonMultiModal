# -*- coding: utf-8 -*-
"""全队列 60mm 裁剪保留率与掩码形态只读审计 (m06922 检查3)。

严格复现 extract_slice_feats_60mm.py 的定心与裁剪:
- 定心: cy,cx = mean(nonzero(m2)) 单层 2D 质心 (int 截断)
- 裁剪: ry=round(60/2/py) rx=round(60/2/px); y0,y1=max(0,cy-ry),min(H,cy+ry); x 同理
- 补黑: pad 正方 S=max(H,W), pad_img[:H,:W]=crop (左上角, 不对称)

统计 (t/d 分列):
1) 重采样后掩码唯一标签值, 非 {0,1} 病例; pid1 t/d 连通域
2) 单层 2D 外接框物理跨度 max(W,H)mm 分布 + >60/80/120mm 切片与病例占比
3) 单层多连通域占比; 跨度>60mm 切片中 多分离连通域 vs 单个大连通域 拆分
4) 固定 60mm 框保留率 (切片级 + 病例级体积加权/最差层) + 阈值占比 + Top10 低保留病例
5) 对照: 自适应 Tight ROI 边长=max(60, span+2*margin) margin=10/15 对称补黑, 边长分位+保留率

结果: stdout + logs/audit_crop_60mm_retention.json
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
SIZE_MM = 60.0


def get_ma(pid, seq, mri_path):
    """复现 extract_slices: 原生 mask 重采样到 MRI 网格 (NN, UInt8), 返回 array。"""
    mimg = sitk.ReadImage(os.path.join(MASK_BASE[seq], str(pid), "colon.nii.gz"))
    vimg = sitk.ReadImage(mri_path)
    mres = sitk.Resample(mimg, vimg.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor,
                         vimg.GetOrigin(), vimg.GetSpacing(), vimg.GetDirection(), 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(mres), vimg.GetSpacing()


def crop_box(cy, cx, ry, rx, H, W):
    y0, y1 = max(0, cy - ry), min(H, cy + ry)
    x0, x1 = max(0, cx - rx), min(W, cx + rx)
    return y0, y1, x0, x1


def main():
    raw = pd.read_csv(RAW).set_index("patient_id")
    t_pids = sorted(int(x[:-4]) for x in os.listdir(os.path.join(SRC_ROOT, "t")) if x.endswith(".npz"))
    man = pd.read_csv(MANIFEST)
    d_pids = sorted(man[(man["mask_source"] == "self") & (man["has_valid_mask"] == 1)]["pid"].astype(int).tolist())

    slice_stat = {"t": {}, "d": {}}   # 切片级汇总
    case_stat = {"t": [], "d": []}    # 病例级记录
    label_issues = []                 # 非 {0,1} 标签病例
    for seq, pids in (("t", t_pids), ("d", d_pids)):
        slices = {k: [] for k in ("span_mm", "n_cc", "ret60", "cc_span_max", "tight10", "tight15")}
        px_l, py_l = [], []
        for pid in pids:
            col = SEQ_COL[seq]
            mp = raw.loc[pid, col]
            if not (isinstance(mp, str) and os.path.exists(mp)):
                continue
            ma, sp = get_ma(pid, seq, mp)
            px, py = sp[0], sp[1]
            px_l.append(px); py_l.append(py)
            uniq = np.unique(ma)
            if not set(uniq.tolist()) <= {0, 1}:
                label_issues.append({"pid": pid, "seq": seq, "labels": uniq.tolist()})
            nz = np.nonzero(ma)
            if len(nz[0]) == 0:
                case_stat[seq].append({"pid": pid, "seq": seq, "slices": 0, "max_span": 0.0,
                                        "max_cc": 0, "vweighted_ret": 0.0, "worst_ret": 0.0})
                continue
            z_min, z_max = nz[0].min(), nz[0].max()
            case_vol_in = case_vol_out60 = 0.0
            worst_ret = 1.0
            max_span = 0.0
            max_cc = 0
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
                max_span = max(max_span, span)
                # 连通域 (2D, 8-邻接)
                lab, n_cc = ndimage.label(m2, structure=np.ones((3, 3), dtype=int))
                max_cc = max(max_cc, n_cc)
                cc_span_max = 0.0
                if n_cc >= 1:
                    for c in range(1, n_cc + 1):
                        cyy, cxx = np.nonzero(lab == c)
                        c_span = max((cxx.max() - cxx.min() + 1) * px,
                                     (cyy.max() - cyy.min() + 1) * py)
                        cc_span_max = max(cc_span_max, c_span)
                # 固定 60mm 框
                ry = int(round(SIZE_MM / 2 / py)); rx = int(round(SIZE_MM / 2 / px))
                y0, y1, x0, x1 = crop_box(cy, cx, ry, rx, ma.shape[1], ma.shape[2])
                in60 = int(m2[y0:y1, x0:x1].sum())
                tot = int(m2.sum())
                ret60 = in60 / tot if tot > 0 else 0.0
                worst_ret = min(worst_ret, ret60)
                case_vol_out60 += in60
                case_vol_in += tot
                # Tight ROI: 边长=max(60, span+2*margin), 对称补黑
                for m_, key in ((10, "tight10"), (15, "tight15")):
                    Lmm = max(SIZE_MM, span + 2 * m_)
                    ry_ = int(round(Lmm / 2 / py)); rx_ = int(round(Lmm / 2 / px))
                    y0t, y1t, x0t, x1t = crop_box(cy, cx, ry_, rx_, ma.shape[1], ma.shape[2])
                    rett = int(m2[y0t:y1t, x0t:x1t].sum()) / tot if tot > 0 else 0.0
                    slices[key].append(rett)
                slices["span_mm"].append(span)
                slices["n_cc"].append(n_cc)
                slices["cc_span_max"].append(cc_span_max)
                slices["ret60"].append(ret60)
            case_stat[seq].append({
                "pid": pid, "seq": seq, "slices": int((z_max - z_min + 1)),
                "max_span": round(max_span, 1), "max_cc": max_cc,
                "vweighted_ret": round(case_vol_out60 / case_vol_in, 4) if case_vol_in else 0.0,
                "worst_ret": round(worst_ret, 4),
            })
        for k, v in slices.items():
            slice_stat[seq][k] = v

    # ---------- 汇总输出 ----------
    def pct(vals, q):
        return float(np.percentile(vals, q))

    out = {"label_issues": label_issues, "slice": {}, "case": {}}
    for seq in ("t", "d"):
        sv = slice_stat[seq]
        spans = np.array(sv["span_mm"]); rets = np.array(sv["ret60"])
        ncc = np.array(sv["n_cc"]); ccs = np.array(sv["cc_span_max"])
        t10 = np.array(sv["tight10"]); t15 = np.array(sv["tight15"])
        n_sl = len(spans)
        print(f"\n=== {seq}: slices={n_sl} ===")
        print(f"  span_mm P50/P75/P90/P95/P99/Max: "
              f"{pct(spans,50):.1f}/{pct(spans,75):.1f}/{pct(spans,90):.1f}/{pct(spans,95):.1f}/{pct(spans,99):.1f}/{spans.max():.1f}")
        for thr in (60, 80, 120):
            print(f"  span>{thr}mm: {int((spans>thr).sum())} sl ({100*(spans>thr).mean():.1f}%)")
        # 多连通域
        print(f"  n_cc>=2 slices: {int((ncc>=2).sum())} ({100*(ncc>=2).mean():.1f}%)")
        big = spans > 60
        if big.sum():
            multi = ncc[big] >= 2
            single_big = ccs[big] > 60
            print(f"  span>60mm slices={int(big.sum())}: "
                  f"multi-cc={int(multi.sum())} single-big-cc={int(single_big.sum())} "
                  f"(both={int((multi & single_big).sum())})")
        # 保留率
        print(f"  ret60 sl: min={rets.min():.3f} P5={pct(rets,5):.3f} P10={pct(rets,10):.3f} "
              f"P25={pct(rets,25):.3f} P50={pct(rets,50):.3f} mean={rets.mean():.3f}")
        for thr in (0.95, 0.80, 0.50, 0.0):
            print(f"    ret<{thr}: {int((rets<thr).sum())} sl ({100*(rets<thr).mean():.1f}%)")
        # 病例级
        cs = case_stat[seq]
        vr = np.array([c["vweighted_ret"] for c in cs]); wr = np.array([c["worst_ret"] for c in cs])
        print(f"  cases={len(cs)}: vweighted P50={pct(vr,50):.3f} mean={vr.mean():.3f}; "
              f"worst P50={pct(wr,50):.3f} mean={wr.mean():.3f}")
        for thr in (0.90, 0.80, 0.50):
            print(f"    vweighted<{thr}: {int((vr<thr).sum())} cases; worst<{thr}: {int((wr<thr).sum())} cases")
        # Top10 低保留
        top = sorted(cs, key=lambda c: c["vweighted_ret"])[:10]
        print("  TOP10 lowest vweighted_ret:")
        for c in top:
            print(f"    pid={c['pid']} seq={c['seq']} slices={c['slices']} max_span={c['max_span']} "
                  f"cc={c['max_cc']} vw={c['vweighted_ret']} worst={c['worst_ret']}")
        # Tight ROI
        print(f"  tight10 L_mm=P50 {pct([max(60, s+20) for s in spans],50):.1f} "
              f"P95 {pct([max(60, s+20) for s in spans],95):.1f}; ret P50 {pct(t10,50):.3f} mean {t10.mean():.3f}")
        print(f"  tight15 L_mm=P50 {pct([max(60, s+30) for s in spans],50):.1f} "
              f"P95 {pct([max(60, s+30) for s in spans],95):.1f}; ret P50 {pct(t15,50):.3f} mean {t15.mean():.3f}")
        out["slice"][seq] = {
            "n_slices": n_sl,
            "span": {"P50": pct(spans,50), "P75": pct(spans,75), "P90": pct(spans,90),
                     "P95": pct(spans,95), "P99": pct(spans,99), "max": float(spans.max()),
                     "gt60": int((spans>60).sum()), "gt80": int((spans>80).sum()),
                     "gt120": int((spans>120).sum())},
            "n_cc_ge2": int((ncc>=2).sum()),
            "ret60": {"min": float(rets.min()), "P5": pct(rets,5), "P10": pct(rets,10),
                      "P25": pct(rets,25), "P50": pct(rets,50), "mean": float(rets.mean()),
                      "lt95": int((rets<0.95).sum()), "lt80": int((rets<0.8).sum()),
                      "lt50": int((rets<0.5).sum()), "eq0": int((rets<=0).sum())},
            "tight10_ret": {"P50": pct(t10,50), "mean": float(t10.mean())},
            "tight15_ret": {"P50": pct(t15,50), "mean": float(t15.mean())},
        }
        out["case"][seq] = {
            "n": len(cs),
            "vweighted": {"P50": pct(vr,50), "mean": float(vr.mean()),
                          "lt90": int((vr<0.9).sum()), "lt80": int((vr<0.8).sum()),
                          "lt50": int((vr<0.5).sum())},
            "worst": {"P50": pct(wr,50), "mean": float(wr.mean()),
                      "lt90": int((wr<0.9).sum()), "lt80": int((wr<0.8).sum()),
                      "lt50": int((wr<0.5).sum())},
            "top10_low": top,
        }

    with open(os.path.join(ROOT, "logs", "audit_crop_60mm_retention.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nDONE -> logs/audit_crop_60mm_retention.json")


if __name__ == "__main__":
    main()
