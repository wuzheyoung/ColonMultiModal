# -*- coding: utf-8 -*-
"""exp_v2_round1_run.py — m00964 第一轮真实特征对比 (label_T, 评估口径 direct_tcd n=162, ddof=1)。

Step2: 强正则线性探针全矩阵 — 3 骨干 × 3 平面 × {l2, std} 归一化, MeanMax 池化 + 折内嵌套
       3 折 CV 选 C L2-LR (C∈{1e-4..1.0}); 融合: 3-plane zlogit + clin_clean+3-plane zlogit。
Step3: 切片级深度 MIL — 3 骨干 × {gated+prior, gated−prior, topk+prior}, --max-batch-size 16,
       float32, feat_norm=l2; 逐折 train/val 曲线记录 (过拟合诊断)。

训练集 = direct_all (n=280, master_5fold_356 折列锁死, 剔除 tgr); 评估统一截取 direct_tcd。
clin OOF 复用 outputs/oof_baseline_n162/singles/clin_clean.csv (tcd 162 行, 用户指定)。
全部评估 CSV (tcd 截取后) 落盘 outputs/exp_v2_round1/, 统计与消融由 exp_v2_round1_stats.py 完成。
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import train_slice_mil as tsm
import eval_paper_metrics as epm

MASTER = "data/master_5fold_356.csv"
COHORT = "data/direct_cohort_ids.csv"
PAIRS = "data/hos_pairs_raw.csv"
DATA_ROOT = "features/slice_feats_v2"
CLIN_OOF = "outputs/oof_baseline_n162/singles/clin_clean.csv"
OUT = "outputs/exp_v2_round1"
BACKBONES = ["convnext_tiny", "dinov2", "biomedclip"]
PROBE_NORMS = ["l2", "std"]
MIL_CFG = [("gated", True), ("gated", False), ("topk", True)]
MIL_KW = dict(h=256, k_frac=0.25, dropout=0.4, epochs=40, lr=1e-3, wd=1e-2,
              seed=0, target="label_T", feat_norm="l2", bin_w=4, max_batch_size=16)
SEED = 0


def get_cohorts():
    master = pd.read_csv(MASTER)
    cohort = pd.read_csv(COHORT)
    pid_col = "patient_id" if "patient_id" in cohort.columns else "pid"
    tgr = set(cohort.loc[cohort["is_tgr"] == 1, pid_col])
    md = master[~master["patient_id"].isin(tgr)].reset_index(drop=True)   # direct_all 280
    raw = pd.read_csv(PAIRS).set_index("patient_id")
    dp = md.set_index("patient_id")
    def has(c):
        return raw["mri_path_" + c].notna().reindex(dp.index).fillna(False)
    tcd = set(dp.index[has("t") & has("c") & has("d")])
    return master, md, tcd


def slice_tcd(df, tcd):
    return df[df["patient_id"].isin(tcd)].reset_index(drop=True)


def eval_metrics(df):
    df = df.dropna(subset=["prob"]).reset_index(drop=True)
    n = len(df); n_pos = int(df["label_T"].sum())
    if n < 10:
        return {"n": n, "n_pos": n_pos, "pooled_auc": None}
    auc = epm.pooled_auc(df)
    fm = epm.fold_mean_sd(df)
    ci = epm.boot_ci(df)
    return {"n": n, "n_pos": n_pos, "pooled_auc": round(float(auc), 4),
            "fold_mean": round(fm[0], 4) if fm else None,
            "fold_sd": round(fm[1], 4) if fm else None,
            "boot_95ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None}


def run_fusions(oofs, md, tcd, tag, clin_oof):
    res = {}
    for mode, fn in [("probmean", tsm.fuse_probmean), ("zlogit", tsm.fuse_zlogit)]:
        fd = fn(oofs) if mode == "probmean" else fn(oofs, md)
        if fd is not None and len(fd):
            fd["label_T"] = fd["patient_id"].map(md.set_index("patient_id")["label_T"])
            fd["fold"] = fd["patient_id"].map(md.set_index("patient_id")["fold"])
            fd = slice_tcd(fd, tcd)
            fd.to_csv(os.path.join(OUT, tag, f"fusion_{mode}.csv"), index=False)
            res[mode] = eval_metrics(fd)
    # clin + 3-plane
    oofs2 = dict(oofs)
    if clin_oof is not None:
        oofs2["clin"] = clin_oof
        for mode in ("probmean", "zlogit"):
            fd = (tsm.fuse_probmean if mode == "probmean" else tsm.fuse_zlogit)(oofs2, md) if mode == "zlogit" \
                else tsm.fuse_probmean(oofs2)
            if fd is not None and len(fd):
                fd["label_T"] = fd["patient_id"].map(md.set_index("patient_id")["label_T"])
                fd["fold"] = fd["patient_id"].map(md.set_index("patient_id")["fold"])
                fd = slice_tcd(fd, tcd)
                fd.to_csv(os.path.join(OUT, tag, f"fusion_clin_{mode}.csv"), index=False)
                res[f"clin_{mode}"] = eval_metrics(fd)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    master, md, tcd = get_cohorts()
    clin_oof = pd.read_csv(CLIN_OOF)
    clin_oof = clin_oof[["patient_id", "fold", "label_T", "prob"]].dropna(subset=["prob"])
    print(f"direct_all={len(md)} direct_tcd={len(tcd)} clin_oof={len(clin_oof)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {"direct_all": len(md), "direct_tcd": len(tcd), "device": str(device),
                "probes": {}, "mil": {}, "times": {}}
    t0 = time.time()

    # ---- Step 2: 线性探针矩阵 ----
    for bk in BACKBONES:
        for norm in PROBE_NORMS:
            tag = f"probe_{bk}_{norm}"
            d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
            oofs = {}
            t1 = time.time()
            for s in tsm.SEQS:
                df, cs = tsm.linear_probe_plane(master=md, data_root=DATA_ROOT, backbone=bk,
                                                seq=s, device=device, seed=SEED,
                                                target="label_T", c_grid=tsm.PROBE_C_GRID,
                                                feat_norm=norm)
                if df is None:
                    continue
                dft = slice_tcd(df, tcd)
                dft.to_csv(os.path.join(d, f"eval_{s}.csv"), index=False)
                oofs[s] = dft
                print(f"  [probe {bk}/{norm} {s}] n={len(dft)} auc={eval_metrics(dft)['pooled_auc']}")
            fus = run_fusions(oofs, md, tcd, tag, clin_oof)
            manifest["probes"][f"{bk}|{norm}"] = {
                "planes": {s: eval_metrics(o) for s, o in oofs.items()},
                "fusions": fus, "probe_Cs": {s: cs for s, cs in [(s, None)]} and None,
                "time_s": round(time.time() - t1, 1)}
            print(f"  [probe {bk}/{norm}] fusions={ {k: v['pooled_auc'] for k, v in fus.items()} }")

    # ---- Step 3: 深度 MIL ----
    for bk in BACKBONES:
        for agg, sp in MIL_CFG:
            tag = f"mil_{bk}_{agg}_{'sp' if sp else 'nosp'}"
            d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
            oofs = {}
            curves = {}
            t1 = time.time()
            for s in tsm.SEQS:
                df, cv, pt = tsm.oof_for_plane(master=md, data_root=DATA_ROOT, backbone=bk,
                                               seq=s, agg=agg, device=device, tag=tag,
                                               spatial_prior=sp, **MIL_KW)
                if df is None:
                    continue
                dft = slice_tcd(df, tcd)
                dft.to_csv(os.path.join(d, f"eval_{s}.csv"), index=False)
                oofs[s] = dft
                if cv is not None and len(cv):
                    cv.to_csv(os.path.join(d, f"curves_{s}.csv"), index=False)
                    curves[s] = cv
                print(f"  [mil {bk}/{agg}/sp={sp} {s}] n={len(dft)} "
                      f"auc={eval_metrics(dft)['pooled_auc']} ({pt:.1f}s)")
            fus = run_fusions(oofs, md, tcd, tag, clin_oof)
            # 过拟合诊断: 每折末 epoch train_loss vs val_loss / val_auc
            diag = {}
            for s, cv in curves.items():
                last = cv.groupby("fold").tail(1)
                diag[s] = {"folds": int(len(last)),
                           "train_loss_last": round(float(last["train_loss"].mean()), 4),
                           "val_loss_last": round(float(last["val_loss"].mean()), 4),
                           "val_auc_last": round(float(last["val_auc"].mean()), 4) if last["val_auc"].notna().any() else None,
                           "val_auc_first": round(float(cv.groupby("fold").head(1)["val_auc"].mean()), 4) if cv["val_auc"].notna().any() else None}
            manifest["mil"][f"{bk}|{agg}|{'sp' if sp else 'nosp'}"] = {
                "planes": {s: eval_metrics(o) for s, o in oofs.items()},
                "fusions": fus, "overfit": diag, "time_s": round(time.time() - t1, 1)}
            print(f"  [mil {bk}/{agg}/sp={sp}] fusions={ {k: v['pooled_auc'] for k, v in fus.items()} }")

    manifest["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nDONE → {OUT}/manifest.json (total={manifest['total_s']}s)")


if __name__ == "__main__":
    main()
