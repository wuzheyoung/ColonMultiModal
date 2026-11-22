# -*- coding: utf-8 -*-
"""B3: 高分辨三臂×3seed 决定性对照 (cache_hos_hr15, 64³ 固定物理视野 96mm³, 严禁拉伸)
规则(用户 m00745 B3): 高分辨三臂 × 3seed (~2.0h)。
- hr_a1: SDF-only (前3通道灰度置零, 仅 SDF 通道)  [in_ch=4]
- hr_a2: 完整 灰度+SDF 4 通道                     [in_ch=4]
- hr_a3: 纯灰度 3 通道 (无 SDF)                   [in_ch=3]
统一: 5-fold (master_5fold_356.csv), epochs 30, self-written, no-clinical,
seed 42/0/7, 主裁判 = 3-seed 均值 last5 OOF N AUC。
独立训练入口: train_roi_dl.py 的 DataROI 硬编码 SPATIAL=96³, 无法读 64³ hr15 卷,
本脚本自带 HRDataset (读 cache_hos_hr15 npy) + 复用 ROINet/_tn_loss/_bin_metrics。
输出: results/roidl_oof/{tag}_k{k}_s{seed}.csv(+_last5) + logs/step_b3_hr_train.json + README
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_roi_dl as T  # noqa: E402  (ROINet/_tn_loss/_bin_metrics/make_loader/_write_oof)

H15 = os.path.join(ROOT, "cache_hos_hr15")


def _key(pid):
    return hashlib.sha1(f"hr15v1|{pid}".encode()).hexdigest()[:20]


class HRDataset(torch.utils.data.Dataset):
    """读 cache_hos_hr15 卷 [4,64,64,64] (0,1,2=t/c/d 灰度 zscore, 3=SDF clip±15)。
    arm: 'sdf' 前3通道置零; 'full' 4通道; 'img' 仅前3通道。"""

    def __init__(self, master, pairs, arm, cache_dir=H15, clin_cols=0):
        self.arm = arm
        self.clin_cols = clin_cols
        rows = []
        for _, r in master.iterrows():
            pid = str(r["patient_id"])
            pp = os.path.join(cache_dir, f"{_key(pid)}.npy")
            if not os.path.exists(pp):
                continue
            rows.append((pid, pp, int(r["label_T"]), int(r["label_N"]), int(r["fold"])))
        self.rows = rows
        self._lab_t = np.asarray([r[2] for r in rows], dtype=np.int64)
        self._lab_n = np.asarray([r[3] for r in rows], dtype=np.int64)
        self._lab_pt = self._lab_t.copy()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pid, pp, yt, yn, _ = self.rows[i]
        v = np.load(pp).astype(np.float32)  # [4,64,64,64]
        if self.arm == "sdf":
            v[:3] = 0.0
        elif self.arm == "img":
            v = v[:3]
        clin = np.zeros(self.clin_cols, dtype=np.float32)
        return torch.as_tensor(v), torch.as_tensor(clin), int(yt), int(yn), int(yt)


def _train_one(ds_tr, ds_va, args, device, logger, fold_label):
    """一折训练: 30ep, cosine, AdamW 双 lr, 逆频权重, best-epoch val AUC 选模, last5 概率平均。
    返回 (best_t_auc, best_n_auc, hist_n_probs, hist_t_probs) 对齐 ds_va 顺序。"""
    T._seed_everything(args.seed)
    tr = T.make_loader(ds_tr, args.batch_size, True, smoke=args.smoke)
    va = T.make_loader(ds_va, args.batch_size, False, drop_last=False, smoke=args.smoke)
    in_ch = ds_tr[0][0].shape[0]
    model = T.ROINet(num_clinical=args.num_clinical, head_width=args.head_width,
                     in_channels=in_ch, head_dropout=args.head_dropout,
                     self_written=True).to(device)
    w_t = T._inv_weights(ds_tr._lab_t, 2, device)
    w_n = T._inv_weights(ds_tr._lab_n, 2, device)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "head_n" not in n], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if "head_n" in n], "lr": args.lr_n},
    ], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    crit_t = nn.CrossEntropyLoss(weight=w_t)
    crit_n = nn.CrossEntropyLoss(weight=w_n)
    best_t = {"score": -1.0, "epoch": -1}
    best_n = {"score": -1.0, "epoch": -1}
    hist_t, hist_n = [], []  # 每轮 val 概率 (用于 last5 平均)
    for ep in range(1, args.epochs + 1):
        model.train()
        it = iter(tr)
        while True:
            try:
                ct, clin, lt, ln, lpt = next(it)
            except StopIteration:
                break
            ct, clin, lt, ln = (x.to(device) for x in (ct, clin, lt, ln))
            vt = (lt >= 0)
            vn = (ln >= 0)
            if not (vt.any() or vn.any()):
                continue
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logit_t, logit_n = model(ct, clin)
                loss = T._tn_loss(crit_t, crit_n, logit_t, logit_n, lt, ln, vt, vn)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        with torch.no_grad():
            pt, pn, yt, yn = [], [], [], []
            for i in range(len(ds_va)):
                ct, clin, lt, ln, _ = ds_va[i]
                ct = ct[None].to(device); clin = clin[None].to(device)
                ot, on = model(ct, clin)
                pt.append(nn.functional.softmax(ot.float(), 1)[0].cpu().numpy())
                pn.append(nn.functional.softmax(on.float(), 1)[0].cpu().numpy())
                yt.append(int(lt)); yn.append(int(ln))
            pt = np.asarray(pt); pn = np.asarray(pn)
            mt = T._bin_metrics(pt, np.asarray(yt))
            mn = T._bin_metrics(pn, np.asarray(yn))
        if mt["roc_auc"] is not None and mt["roc_auc"] > best_t["score"]:
            best_t = {"score": mt["roc_auc"], "epoch": ep}
        if mn["roc_auc"] is not None and mn["roc_auc"] > best_n["score"]:
            best_n = {"score": mn["roc_auc"], "epoch": ep}
        hist_t.append(pt); hist_n.append(pn)
        if logger is not None:
            logger.info(f"[HR] {fold_label} Epoch {ep:03d} T_auc={mt['roc_auc']:.4f} N_auc={mn['roc_auc']:.4f}")
        else:
            print(f"[HR] {fold_label} Epoch {ep:03d} T_auc={mt['roc_auc']:.4f} N_auc={mn['roc_auc']:.4f}")
    # last5 = 末5轮概率平均
    last5_t = np.mean(hist_t[-5:], 0)
    last5_n = np.mean(hist_n[-5:], 0)
    return {"best_t": best_t, "best_n": best_n,
            "last5_t": last5_t, "last5_n": last5_n,
            "yt": np.asarray(yt), "yn": np.asarray(yn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="dc")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-n", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.num_clinical = 0
    args.head_width = 256
    args.head_dropout = 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    master = pd.read_csv(os.path.join(ROOT, "data", "master_5fold_356.csv"), encoding="utf-8-sig")
    pairs = pd.read_csv(os.path.join(ROOT, "data", "hos_pairs_raw.csv"), encoding="utf-8-sig")
    arms = [("hr_a1_sdfonly", "sdf"), ("hr_a2_imgsdf", "full"), ("hr_a3_img", "img")]
    seeds = [42, 0, 7]
    out = {"gen": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": {}}
    for arm_name, arm in arms:
        ds_all = HRDataset(master, pairs, arm)
        if len(ds_all) < 50:
            print(f"[B3] {arm_name}: 样本不足 ({len(ds_all)}), 跳过")
            out["arms"][arm_name] = {"n": len(ds_all), "skipped": True}
            continue
        seed_rows = {}
        for seed in seeds:
            args.seed = seed
            tag = f"{args.tag_prefix}_{arm_name}_s{seed}"
            oof_paths = []
            per_fold = []
            for fi in range(5):
                va_idx = [i for i, r in enumerate(ds_all.rows) if r[4] == fi]
                tr_idx = [i for i, r in enumerate(ds_all.rows) if r[4] != fi]
                ds_va = torch.utils.data.Subset(ds_all, va_idx)
                ds_tr = torch.utils.data.Subset(ds_all, tr_idx)
                ds_tr._lab_t = ds_all._lab_t[tr_idx]
                ds_tr._lab_n = ds_all._lab_n[tr_idx]
                ds_tr._lab_pt = ds_all._lab_pt[tr_idx]
                ds_va._lab_t = ds_all._lab_t[va_idx]
                ds_va._lab_n = ds_all._lab_n[va_idx]
                ds_va._lab_pt = ds_all._lab_pt[va_idx]
                r = _train_one(ds_tr, ds_va, args, device, None, f"k{fi}")
                mn = T._bin_metrics(r["last5_n"], r["yn"])
                mt = T._bin_metrics(r["last5_t"], r["yt"])
                per_fold.append({"fold": fi, "T_auc": mt["roc_auc"], "N_auc": mn["roc_auc"]})
                pids = [ds_all.rows[i][0] for i in va_idx]
                T._write_oof(os.path.join(ROOT, "results", "roidl_oof", f"{tag}_k{fi}_s{seed}.csv"),
                             pids, {"t": (r["last5_t"], r["yt"]), "n": (r["last5_n"], r["yn"])})
                oof_paths.append(os.path.join(ROOT, "results", "roidl_oof", f"{tag}_k{fi}_s{seed}.csv"))
            # 合并 OOF 全局 AUC (last5)
            allp = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in oof_paths], ignore_index=True)
            auc_n = float(roc_auc_score(allp["label_N"], allp["prob_N"]))
            auc_t = float(roc_auc_score(allp["label_T"], allp["prob_T"]))
            seed_rows[f"s{seed}"] = {"global_N_auc_last5": auc_n, "global_T_auc_last5": auc_t,
                                     "per_fold": per_fold, "n": len(ds_all)}
            print(f"[B3] {tag}: N_last5={auc_n:.4f} T_last5={auc_t:.4f}")
        out["arms"][arm_name] = {"n": len(ds_all), "seeds": seed_rows}
    # 汇总
    def mean_k(arm, k):
        ss = out["arms"].get(arm, {}).get("seeds", {})
        vals = [v[k] for v in ss.values()] if ss else []
        return float(np.mean(vals)) if vals else None
    summary = {}
    for arm_name, _, in arms:
        summary[arm_name] = {"N_last5_mean": mean_k(arm_name, "global_N_auc_last5"),
                             "T_last5_mean": mean_k(arm_name, "global_T_auc_last5")}
    out["summary"] = summary
    with open(os.path.join(ROOT, "logs", "step_b3_hr_train.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "README.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## B3 高分辨三臂对照 (自动接力 {time.strftime('%Y-%m-%d %H:%M')})\n"
                f"- 高分辨 64³ 固定物理视野(1.5mm³), 样本 n={out['arms'].get('hr_a2_imgsdf',{}).get('n')}\n"
                f"- last5 OOF N AUC 三seed均值: sdfonly={summary['hr_a1_sdfonly']['N_last5_mean']} "
                f"img+sdf={summary['hr_a2_imgsdf']['N_last5_mean']} "
                f"img={summary['hr_a3_img']['N_last5_mean']} (logs/step_b3_hr_train.json)\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
