# -*- coding: utf-8 -*-
"""exp_roi_ablation_mil.py — ROI 消融 MIL std 单独成表 (m07302 指令)。

与 exp_roi_ablation.py 同口径 (direct 锁死/d_self 限定/td/tcd 子集), 但模型为
GatedAttentionMIL (feat_norm=std, h=32, dropout=0.6, wd=0.05, epochs=15, lr=3e-4,
seed 0, spatial_prior=False), 与夜间任务3 完全同配置。
组合: t_only / d_self_only / img_td(=probmean t+d) / full(=probmean clin_linear + t + d)。
产物: logs/exp_roi_ablation_mil.json + 追加到 logs/exp_roi_ablation_report.md "## MIL std (gated)"。
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_slice_mil as tsm
from exp_roi_ablation import (load_subsets, t_pids_common, d_self_pids_common,
                              clin_oof, pooled_auc, fold_mean_sd, stratified_boot_ci,
                              ece_brier)

REFS = {
    "baseline_120mm_old": "features/slice_feats_v2",
    "clean_120mm": "features/slice_feats_v2_120mm_clean",
    "tight60_m10": "features/slice_feats_v2_tight60_m10",
}
BACKBONE = "dinov2"
MIL_KW = dict(agg="gated", h=32, k_frac=0.25, dropout=0.6, epochs=15, lr=3e-4,
              wd=0.05, seed=0, feat_norm="std", spatial_prior=False)
JSON_OUT = "logs/exp_roi_ablation_mil.json"


def mil_oof(master, data_root, seq, pids, device):
    """对指定 pid 集合 (已限定 direct/d_self) 跑 gated MIL 5 折 OOF, 返回 df。"""
    m = master.set_index("patient_id")
    pids = [int(p) for p in pids]
    have = [p for p in pids if os.path.exists(
        os.path.join(data_root, BACKBONE, seq, f"{p}.npz"))]
    if len(have) < 12:
        return None
    t0 = time.time()
    # python int 键 (preload_plane/make_std_bags 的 dict 键均须 python int, np.int64 哈希不匹配)
    bags_raw = tsm.preload_plane(master, data_root, BACKBONE, seq, have, "label_T",
                                 MIL_KW["feat_norm"], device)
    fold_of = {p: int(m.loc[p, "fold"]) for p in have}
    oof = []
    for k in sorted(set(fold_of.values())):
        tr = [p for p in have if fold_of[p] != k]
        va = [p for p in have if fold_of[p] == k]
        if len(tr) < 8 or len(va) < 1:
            continue
        bags = tsm.make_std_bags(bags_raw, tr + va, tr)
        torch.manual_seed(int(MIL_KW["seed"] + k))
        model = tsm.SliceMIL(tsm.D + tsm.SPATIAL_DIM if MIL_KW["spatial_prior"] else tsm.D,
                             agg=MIL_KW["agg"], h=MIL_KW["h"], k_frac=MIL_KW["k_frac"],
                             dropout=MIL_KW["dropout"],
                             spatial_prior=MIL_KW["spatial_prior"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=MIL_KW["lr"], weight_decay=MIL_KW["wd"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MIL_KW["epochs"])
        rows, _ = tsm.train_plane(bags, tr, va, model, opt, sch, MIL_KW["epochs"],
                                  MIL_KW["spatial_prior"], device,
                                  int(MIL_KW["seed"] + k), "label_T",
                                  max_batch_size=16)
        oof.extend(rows)
        del model, opt, sch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(oof)
    df["patient_id"] = df["patient_id"].astype(int)
    df["fold"] = df["patient_id"].map(m["fold"]).astype(int)
    df = df.rename(columns={"label_T": "label"})
    df = df.sort_values("patient_id").reset_index(drop=True)
    df["n_train"] = len(have) - len(set(fold_of.values()))
    return df, time.time() - t0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    master, md, tcd, td = load_subsets()
    m = master.set_index("patient_id")
    direct_ids = set(md["patient_id"].astype(int))
    tP = np.array(sorted(set(t_pids_common()) & direct_ids))
    dP = np.array(sorted(set(d_self_pids_common()) & direct_ids))
    print(f"tP={len(tP)} dP={len(dP)}")

    clin_df = clin_oof(md, target="label_T")
    clin_map = {int(r["patient_id"]): float(r["prob"]) for _, r in clin_df.iterrows()}

    oofs = {}
    for ref, root in REFS.items():
        dt, _ = mil_oof(master, root, "t", tP, device)
        dd, _ = mil_oof(master, root, "d", dP, device)
        if dt is None or dd is None:
            print(f"[{ref}] skip (dt={dt is not None} dd={dd is not None})")
            continue
        dt["root"] = ref; dd["root"] = ref
        oofs[(ref, "t_only")] = dt
        oofs[(ref, "d_self_only")] = dd
        # img_td = probmean(t, d) 限定 t∩d 公共样本 (与线性探针 concat 交集口径一致)
        common = sorted(set(dt["patient_id"]) & set(dd["patient_id"]))
        dtc = dt[dt["patient_id"].isin(common)].reset_index(drop=True)
        ddc = dd[dd["patient_id"].isin(common)].reset_index(drop=True)
        fuse = tsm.fuse_probmean({"t": dtc, "d": ddc})
        fuse["fold"] = fuse["patient_id"].map(m["fold"]).astype(int)
        fuse["label"] = fuse["patient_id"].map(lambda p: int(m.loc[p, "label_T"]))
        oofs[(ref, "img_td")] = fuse
        # full = probmean(clin_linear, t, d)
        cl = pd.DataFrame({"patient_id": [int(p) for p in fuse["patient_id"]],
                           "fold": fuse["fold"].values,
                           "label": fuse["label"].values,
                           "prob": [clin_map.get(int(p), 0.5) for p in fuse["patient_id"]]})
        full = tsm.fuse_probmean({"clin": cl, "t": dtc, "d": ddc})
        full["fold"] = full["patient_id"].map(m["fold"]).astype(int)
        full["label"] = full["patient_id"].map(lambda p: int(m.loc[p, "label_T"]))
        oofs[(ref, "full_multimodal")] = full

    combos = ["t_only", "d_self_only", "img_td", "full_multimodal"]
    subsets = {"td": td, "tcd": tcd}
    res = {"config": MIL_KW, "n_subsets": {k: int(len(v)) for k, v in subsets.items()},
           "t_common_direct": int(len(tP)), "d_self_common_direct": int(len(dP)),
           "evaluation": {}}
    for combo in combos:
        res["evaluation"][combo] = {}
        for sname, ids in subsets.items():
            row = {}
            for ref in REFS:
                df = oofs.get((ref, combo))
                if df is None:
                    row[ref] = None
                    continue
                sub = df[df["patient_id"].isin(ids)]
                if len(sub) < 10:
                    row[ref] = None
                    continue
                y = sub["label"].values.astype(int)
                p = sub["prob"].values.astype(float)
                f = sub["fold"].values
                auc = pooled_auc(y, p)
                lo, hi = stratified_boot_ci(y, p, n_boot=2000)
                fm, fsd = fold_mean_sd(y, p, f)
                ece, br = ece_brier(y, p)
                row[ref] = {"n": int(len(sub)), "n_pos": int(y.sum()),
                            "pooled_auc": auc, "ci95": [lo, hi],
                            "fold_mean": fm, "fold_sd": fsd,
                            "ece": ece, "brier": br}
            res["evaluation"][combo][sname] = row
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)

    # 追加 md
    L = ["\n## MIL std (gated, feat_norm=std, h=32/dropout=0.6/wd=0.05/epochs=15/lr=3e-4)"]
    for combo in combos:
        L.append(f"\n### {combo}")
        for sname in ("td", "tcd"):
            L.append(f"\n#### {sname} (n={len(subsets[sname])})")
            L.append("| 特征目录 | n | n_pos | Pooled AUC | 95% CI | Fold-Mean±SD | ECE | Brier |")
            L.append("|---|---|---|---|---|---|---|---|")
            for ref in REFS:
                r = res["evaluation"][combo][sname][ref]
                if r is None:
                    L.append(f"| {ref} | - | - | - | - | - | - | - |")
                    continue
                ci = "–" if r["ci95"][0] is None else f"[{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}]"
                fs = "–" if r["fold_mean"] is None else f"{r['fold_mean']:.4f}±{r['fold_sd']:.4f}"
                L.append(f"| {ref} | {r['n']} | {r['n_pos']} | {r['pooled_auc']:.4f} | {ci} | {fs} "
                         f"| {r['ece']:.4f} | {r['brier']:.4f} |")
    # 写 md: 先删除旧的 MIL std 段再追加 (避免重复运行累积两份)
    RP = "logs/exp_roi_ablation_report.md"
    if os.path.exists(RP):
        prev = open(RP, encoding="utf-8").read().splitlines(keepends=True)
        kept, in_mil = [], False
        for l in prev:
            if l.startswith("## MIL std"):
                in_mil = True
                continue
            if in_mil:
                if l.startswith("## ") and not l.startswith("## MIL std"):
                    in_mil = False
                else:
                    continue
            kept.append(l)
        with open(RP, "w", encoding="utf-8") as f:
            f.writelines(kept)
    with open(RP, "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("DONE ->", JSON_OUT)


if __name__ == "__main__":
    main()
