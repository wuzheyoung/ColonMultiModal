# -*- coding: utf-8 -*-
# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""round9 TTA (test-time augmentation) 评估: 验证时 3 变体翻转 softmax 平均。
用法: python tools/eval_tta.py [--ckpt-t ...] [--ckpt-n ...] [--branch full] [--split-seed 42] [--out logs/tta_full_s42.json]
不训练, 只对已有 ckpt 评估 val, 对比 no-TTA vs TTA。
"""
import argparse
import json
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.hos import HosConfig
from data_loader.loader import build_dataloaders
from models.network import CRCMultiModalNet
from utils import setup_logger, calibrate_binary_threshold


def _build_model(cfg):
    from train_hos_seeds import build_model as _bm
    return _bm(cfg)


def _predict_variants(model, loader, cfg, task="t", variants=3):
    """对 val 每个 batch 的 ct/mri 卷做翻转增强, softmax 平均。
    variants=1 -> 无 TTA (identity only)。variants=3 -> identity + LR + UD。
    """
    model.eval()
    out_probs, out_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(cfg.device)
            mri = batch["mri"].to(cfg.device)
            clin = batch["clinical"].to(cfg.device)
            ct_mask = batch["ct_available"].to(cfg.device).bool()
            mri_mask = batch["mri_available"].to(cfg.device).bool()
            mri_seq_mask = batch.get("mri_seq_available")
            if mri_seq_mask is not None:
                mri_seq_mask = mri_seq_mask.to(cfg.device).bool()
            lbl = (batch["label"] if task == "t" else batch.get("label_secondary"))
            if lbl is None:
                continue
            lbl = lbl.to(cfg.device)

            # 变体张量构造 (有效变体数 = 实际拼接的份数, 必须与 mask.repeat 一致)
            def _var(x):
                xs = [x]
                if variants >= 2:
                    xs.append(torch.flip(x, dims=[-1]))       # L-R
                if variants >= 3:
                    xs.append(torch.flip(x, dims=[-2]))       # U-D
                if variants >= 4:
                    xs.append(torch.flip(x, dims=[-3]))       # 层向
                if variants >= 8:                             # 全组合 (1+7)
                    for dims in ([-1, -2], [-1, -3], [-2, -3], [-1, -2, -3]):
                        xs.append(torch.flip(x, dims=dims))
                return torch.cat(xs, dim=0), len(xs)
            ct_v, n_var = _var(ct)
            mri_v, n_var_m = _var(mri)
            if n_var != n_var_m:
                raise RuntimeError(f"[TTA] ct/mri 变体数不一致: {n_var} vs {n_var_m}")
            # mask 也要扩到变体数 (同一变体重复)
            ct_mask_v = ct_mask.repeat(n_var)
            mri_mask_v = mri_mask.repeat(n_var)
            mri_seq_mask_v = mri_seq_mask.repeat(n_var, 1) if mri_seq_mask is not None else None

            with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
                o = model(ct_v, mri_v, clin.repeat(n_var, 1),
                          ct_mask=ct_mask_v, mri_mask=mri_mask_v,
                          mri_seq_mask=mri_seq_mask_v)
            key = "main" if task == "t" else "n"
            p = torch.softmax(o[key], dim=1).float()
            # 还原每样本: 变体维平均
            B = ct.size(0)
            p = p.view(B, n_var, -1).mean(dim=1).cpu().numpy()
            out_probs.append(p)
            out_labels.append(lbl.cpu().numpy())
    return np.concatenate(out_probs), np.concatenate(out_labels)


def _f1_at(probs, labels, thr):
    pred = (probs[:, 1] >= thr).astype(int)
    cm = {}
    for t, p in zip(labels, pred):
        cm[(int(t), int(p))] = cm.get((int(t), int(p)), 0) + 1
    tp = cm.get((1, 1), 0); fn = cm.get((1, 0), 0); fp = cm.get((0, 1), 0)
    rec1 = tp / (tp + fn) if (tp + fn) else 0.0
    prec1 = tp / (tp + fp) if (tp + fp) else 0.0
    rec0 = cm.get((0, 0), 0) / (cm.get((0, 0), 0) + fp) if (cm.get((0, 0), 0) + fp) else 0.0
    prec0 = cm.get((0, 0), 0) / (cm.get((0, 0), 0) + fn) if (cm.get((0, 0), 0) + fn) else 0.0
    f1_1 = 2 * rec1 * prec1 / (rec1 + prec1) if (rec1 + prec1) else 0.0
    f1_0 = 2 * rec0 * prec0 / (rec0 + prec0) if (rec0 + prec0) else 0.0
    return (f1_1 + f1_0) / 2, [rec0, rec1]


def _row(probs, labels):
    f1, rec = _f1_at(probs, labels, 0.5)
    cal = calibrate_binary_threshold(probs[:, 1], labels)
    return {
        "f1@0.5": round(float(f1), 4),
        "recall@0.5": [round(x, 4) for x in rec],
        "cal_thr": round(float(cal["best"]["threshold"]), 4),
        "cal_f1": round(float(cal["best"]["metrics"]["macro_f1"]), 4),
        "cal_recall": [round(x, 4) for x in cal["best"]["metrics"]["per_class_recall"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-t", default=None,
                    help="默认 checkpoints_hos_{branch}/best_model_hos_{branch}_s{split_seed}.pth")
    ap.add_argument("--ckpt-n", default=None,
                    help="默认 checkpoints_hos_{branch}/best_model_n_hos_{branch}_s{split_seed}.pth")
    ap.add_argument("--branch", default="full")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--variants", type=int, default=3, choices=[1, 2, 3, 4, 8],
                    help="TTA 变体数: 1=无TTA 3=identity+LR+UD 4=+层向 8=全组合")
    ap.add_argument("--out", default="logs/tta_full_s42.json")
    args = ap.parse_args()

    tag = args.branch
    val_csv = f"data/hos_{tag}_val_s{args.split_seed}.csv"
    ckpt_dir = f"./checkpoints_hos_{tag}"
    if args.ckpt_t is None:
        args.ckpt_t = f"{ckpt_dir}/best_model_hos_{tag}_s{args.split_seed}.pth"
    if args.ckpt_n is None:
        args.ckpt_n = f"{ckpt_dir}/best_model_n_hos_{tag}_s{args.split_seed}.pth"
    for p in (args.ckpt_t, args.ckpt_n):
        if not os.path.exists(p):
            raise SystemExit(f"[TTA] 缺少 ckpt: {p} (历史 s42 已被后续实验覆盖, 需同配置重训)")
    logger = setup_logger("./logs", "tta")
    logger.info(f"[TTA] variants={args.variants} val_csv={val_csv}")

    cfg = HosConfig()
    cfg.backbone_name = "medicalnet18"
    cfg.pool_mode = "gem"
    cfg.head_width = 128
    cfg.lr = 1e-4
    cfg.lr_n = 3e-4
    cfg.n_from_shallow = True
    cfg.n_shallow_idx = 3
    cfg.class_weight_mode = "inverse"
    cfg.save_dir = f"./checkpoints_hos_{tag}"
    cfg.plot_dir = f"./plots_hos_{tag}"
    cfg.use_augmentation = False
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    # ---- T 任务 ----
    model = _build_model(cfg)
    sd = torch.load(args.ckpt_t, map_location=cfg.device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(cfg.device).eval()
    cfg.train_csv, cfg.val_csv = val_csv, val_csv
    _, loader = build_dataloaders(cfg, val_csv, val_csv)

    pt, lt = _predict_variants(model, loader, cfg, task="t", variants=1)
    pt3, lt3 = _predict_variants(model, loader, cfg, task="t", variants=args.variants)
    results["T"] = {"no_tta": _row(pt, lt), f"tta_v{args.variants}": _row(pt3, lt3)}
    logger.info(f"[TTA] T no_tta={results['T']['no_tta']} tta={results['T'][f'tta_v{args.variants}']}")

    # ---- N 任务 ----
    model = _build_model(cfg)
    sd = torch.load(args.ckpt_n, map_location=cfg.device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.to(cfg.device).eval()
    pn, ln = _predict_variants(model, loader, cfg, task="n", variants=1)
    pn3, ln3 = _predict_variants(model, loader, cfg, task="n", variants=args.variants)
    results["N"] = {"no_tta": _row(pn, ln), f"tta_v{args.variants}": _row(pn3, ln3)}
    logger.info(f"[TTA] N no_tta={results['N']['no_tta']} tta={results['N'][f'tta_v{args.variants}']}")

    out = {"meta": {"branch": tag, "split_seed": args.split_seed, "val_n": int(len(lt)),
                    "ckpt_t": args.ckpt_t, "ckpt_n": args.ckpt_n, "variants": args.variants},
           "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write(f"branch={tag} split_seed={args.split_seed} val_n={len(lt)} variants={args.variants}\n\n")
        for task in ("T", "N"):
            for mode in ("no_tta", f"tta_v{args.variants}"):
                r = results[task][mode]
                f.write(f"{task} {mode}: f1@0.5={r['f1@0.5']} recall={r['recall@0.5']} | "
                        f"cal_thr={r['cal_thr']} cal_f1={r['cal_f1']} cal_recall={r['cal_recall']}\n")
    print("[TTA] done ->", args.out)


if __name__ == "__main__":
    main()
