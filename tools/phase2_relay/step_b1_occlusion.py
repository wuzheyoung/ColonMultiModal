# -*- coding: utf-8 -*-
"""B1: 3D 遮挡敏感度 (sdf-only 权重, a1 臂) —— 定位 N 信号空间来源
规则(用户 m00745 B1): 取 a1_sdfonly 权重, 20 例 (10 高置信 N+ / 10 N0),
5 张叠加图 + 量化高敏感体素 肠壁/肠腔 vs 裁剪边缘 比例。
实现: 每样本用其所在折的 best N 权重, 16³ 块遮挡滑窗 (4×4×4=64 次前向),
Δprob_N 绝对值→敏感度图; 高敏感体素(前20%)按 SDF 符号分区(壁|腔|外)与边缘环统计。
输出: tools/phase2_relay/outputs/b1_occlusion/*.png + logs/step_b1_occlusion.json + README
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "data_loader"))
import train_roi_dl as T  # noqa: E402


def _load_model_for_fold(tag, seed, fi, device, mode="mri_t", in_ch=2):
    ckpt = os.path.join(ROOT, "checkpoints_hos_full",
                        f"best_model_n_roidl_{tag}_k{fi}_s{seed}.pth")
    if not os.path.exists(ckpt):
        return None
    sd = torch.load(ckpt, map_location="cpu")
    # 训练时 num_clinical = ClinPrep 维度 (7 值 + 7 missing 指示位 = 14), 非 0
    fuse_w = sd.get("selfnet.fuse.0.weight")
    num_clinical = int(fuse_w.shape[1]) - 128 if fuse_w is not None else 14
    model = T.ROINet(num_clinical=num_clinical, head_width=256, in_channels=in_ch,
                     head_dropout=0.5, self_written=True).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, num_clinical


def _sensitivity(model, vol, device, block=16, num_clinical=14):
    """vol [C,64,64,64] → 敏感度图 [64,64,64] (Δprob_N, 逐 16³ 块遮挡)。
    返回 (sens_map, base_prob)"""
    C = vol.shape[0]
    x = torch.as_tensor(vol[None], dtype=torch.float32, device=device)
    clin = torch.zeros((1, num_clinical), dtype=torch.float32, device=device)
    with torch.no_grad():
        _, logit_n = model(x, clin)
        base = float(torch.softmax(logit_n, 1)[0, 1])
    sens = np.zeros(vol.shape[1:], dtype=np.float32)
    n = 64 // block
    for i in range(n):
        for j in range(n):
            for k in range(n):
                xm = x.clone()
                xm[0, :, i * block:(i + 1) * block, j * block:(j + 1) * block, k * block:(k + 1) * block] = 0.0
                with torch.no_grad():
                    _, ln = model(xm, clin)
                    pn = float(torch.softmax(ln, 1)[0, 1])
                sens[i * block:(i + 1) * block, j * block:(j + 1) * block, k * block:(k + 1) * block] = abs(pn - base)
    return sens, base


def _region_stats(sens, sdf_ch, mask_ch=None, top_frac=0.2):
    """按 SDF 通道分区统计高敏感体素占比。sdf_ch [64,64,64] 为 **归一化 SDF** (clip ±15mm → [-1,1]):
    阈值按 mm 换算: |s|≤2mm → 壁 (=2/15≈0.133), s<-0.133 腔, s>+0.133 外。"""
    WALL_SDF = 2.0 / 15.0  # 2mm / SDF_CLIP_MM
    flat = sens.flatten()
    thr = float(np.quantile(flat, 1.0 - top_frac)) if flat.max() > 0 else float("inf")
    hot = flat >= thr
    s = sdf_ch.flatten()
    wall = np.abs(s) <= WALL_SDF      # 归一化 SDF ±2mm ≈ 肠壁
    lumen = s < -WALL_SDF
    outer = s > WALL_SDF
    edge = np.zeros_like(s, dtype=bool).reshape(sens.shape)
    pad = 8
    edge[:pad, :, :] = edge[-pad:, :, :] = True
    edge[:, :pad, :] = edge[:, -pad:, :] = True
    edge[:, :, :pad] = edge[:, :, -pad:] = True
    edge = edge.flatten()
    tot = int(hot.sum())
    return {
        "n_hot": tot,
        "wall_frac": float(hot[wall].sum() / tot) if tot else 0.0,
        "lumen_frac": float(hot[lumen].sum() / tot) if tot else 0.0,
        "outer_frac": float(hot[outer].sum() / tot) if tot else 0.0,
        "edge_frac": float(hot[edge].sum() / tot) if tot else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="dc")
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--top-frac", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    tag = f"{args.tag_prefix}_a1_sdfonly_s{seed}"
    oof = None
    for k in range(5):
        p = os.path.join(ROOT, "results", "roidl_oof", f"{tag}_k{k}_s{seed}_last5.csv")
        if not os.path.exists(p):
            p = os.path.join(ROOT, "results", "roidl_oof", f"{tag}_k{k}_s{seed}.csv")
        if not os.path.exists(p):
            print(f"[B1] 缺少 {tag} 折{k} OOF, 跳过 (阶段1 是否完成?)")
            return 1
        d = __import__("pandas").read_csv(p, encoding="utf-8-sig")
        d["fold"] = k
        oof = d if oof is None else __import__("pandas").concat([oof, d], ignore_index=True)
    oof["pid"] = oof["patient_id"].astype(str).str.strip()
    pos = oof[oof["label_N"] == 1].sort_values("prob_N", ascending=False)
    neg = oof[oof["label_N"] == 0].sort_values("prob_N", ascending=True)
    sel_pos = pos.head(args.n_per_class)
    sel_neg = neg.head(args.n_per_class)
    sel = __import__("pandas").concat([sel_pos, sel_neg])

    # 数据: 复用 train_roi_dl 的 DataROI (mri_t + sdf-channel + sdf-only + bbox)
    ds = T.DataROI(os.path.join(ROOT, "data", "hos_pairs.csv"), mode="mri_t", augment=False,
                   mask_geometry="aligned", roi_mask="bbox", cache_dir="cache_hos",
                   sdf_channel=True, sdf_only=True)
    idx_by_pid = {p: i for i, p in enumerate(ds._pids)}

    outdir = os.path.join(ROOT, "tools", "phase2_relay", "outputs", "b1_occlusion")
    os.makedirs(outdir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results, images = [], []
    for _, row in sel.iterrows():
        pid, fi = str(row["pid"]), int(row["fold"])
        if pid not in idx_by_pid:
            continue
        loaded = _load_model_for_fold(tag, seed, fi, device)
        if loaded is None:
            continue
        model, num_clinical = loaded
        vol = ds._vols[idx_by_pid[pid]]  # [C,64,64,64]
        sens, base = _sensitivity(model, vol, device, num_clinical=num_clinical)
        sdf_ch = vol[-1]
        stats = _region_stats(sens, sdf_ch, top_frac=args.top_frac)
        stats.update({"pid": pid, "fold": fi, "label_N": int(row["label_N"]),
                      "prob_N": float(row["prob_N"]), "base_prob": base})
        results.append(stats)

        # 叠加图: 轴向 5 张 (z=8,16,24,32,40) sens 热图叠加到 SDF 灰度
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for zi, ax in zip(range(8, 48, 8), axes):
            ax.imshow(sdf_ch[zi], cmap="gray", vmin=-1, vmax=1)
            im = ax.imshow(sens[zi], cmap="hot", alpha=0.55)
            ax.set_title(f"z={zi} pN={row['prob_N']:.3f}")
            ax.axis("off")
        fig.suptitle(f"B1 occlusion pid={pid} N={int(row['label_N'])} fold={fi}")
        fig.colorbar(im, ax=axes, shrink=0.6)
        op = os.path.join(outdir, f"occ_p{pid}_n{int(row['label_N'])}.png")
        fig.tight_layout(); fig.savefig(op, dpi=90); plt.close(fig)
        images.append(op)

    mean = {k: float(np.mean([r[k] for r in results])) for k in
            ("wall_frac", "lumen_frac", "outer_frac", "edge_frac")}
    out = {"mean": mean, "per_patient": results, "images": images,
           "gen": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(ROOT, "logs", "step_b1_occlusion.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "README.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## B1 遮挡敏感度 (自动接力 {time.strftime('%Y-%m-%d %H:%M')})\n"
                f"- a1_sdfonly 权重, {len(results)} 例遮挡, 高敏感体素分区占比均值: "
                f"壁={mean['wall_frac']:.3f} 腔={mean['lumen_frac']:.3f} "
                f"外={mean['outer_frac']:.3f} 边缘={mean['edge_frac']:.3f}\n"
                f"- 叠加图: tools/phase2_relay/outputs/b1_occlusion/ (logs/step_b1_occlusion.json)\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
