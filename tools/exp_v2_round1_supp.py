# -*- coding: utf-8 -*-
"""exp_v2_round1_supp.py — m01114 补算消融与功效对照 (probe_dinov2_std, 无需重训骨干)。

1. 去 c 消融 (direct_tcd n=162): clin+d+t  vs  clin+d+t+c 的 Pooled OOF AUC、
   Fold-Mean±SD (ddof=1) 与配对 DeLong (验证全结肠 c 是增益还是拖后腿)。
   直接复用本轮已落盘 tcd 截取的 OOF (outputs/exp_v2_round1/probe_dinov2_std/eval_{t,c,d}.csv
   + outputs/oof_baseline_n162/singles/clin_clean.csv), 融合口径 fuse_zlogit (可用模态 z-score 均值)。
2. 更大样本口径 direct_td (n=214) 功效验证: probe_dinov2_std (clin+d+t) vs 旧锁定基线
   s2_clin_dconv (clin_clean + 旧 route2 d_conv) 的 Pooled OOF AUC、95% CI 与配对 DeLong。
   需在 td 交集上重出全直集 OOF: linear_probe_plane (dinov2, std, seed=0) 在 direct_all 上
   全量 OOF 后截取 td (与落盘 tcd OOF 同模型, 先校验 tcd 截取与落盘逐行一致)。
"""
import os, sys, json
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import recompute_direct_tcd_singles as rc
import train_slice_mil as tsm
import eval_paper_metrics as epm

MASTER = "data/master_5fold_356.csv"
COHORT = "data/direct_cohort_ids.csv"
PAIRS = "data/hos_pairs_raw.csv"
DATA_ROOT = "features/slice_feats_v2"
PROBE_DIR = "outputs/exp_v2_round1/probe_dinov2_std"
CLIN_OOF_TCD = "outputs/oof_baseline_n162/singles/clin_clean.csv"
OUT = "logs/exp_v2_round1_supp.json"


def load_master_cohorts():
    master = pd.read_csv(MASTER)
    cohort = pd.read_csv(COHORT)
    pid_col = "patient_id" if "patient_id" in cohort.columns else "pid"
    tgr = set(cohort.loc[cohort["is_tgr"] == 1, pid_col])
    md = master[~master["patient_id"].isin(tgr)].reset_index(drop=True)
    raw = pd.read_csv(PAIRS).set_index("patient_id")
    dp = md.set_index("patient_id")

    def has(c):
        return raw["mri_path_" + c].notna().reindex(dp.index).fillna(False)

    td = set(dp.index[has("t") & has("d")])
    tcd = set(dp.index[has("t") & has("c") & has("d")])
    return master, md, td, tcd


def slice_ids(df, ids):
    return df[df["patient_id"].isin(ids)].reset_index(drop=True)


def clf_metrics(df, tag, with_ci=False):
    df = df.dropna(subset=["prob"]).reset_index(drop=True)
    n = len(df)
    y = df["label_T"].values.astype(int)
    auc = epm.pooled_auc(df)
    fm = epm.fold_mean_sd(df)
    r = {"n": n, "n_pos": int(y.sum()), "pooled_auc": round(float(auc), 4),
         "fold_mean": round(float(fm[0]), 4) if fm else None,
         "fold_sd": round(float(fm[1]), 4) if fm else None}
    if with_ci:
        ci = epm.boot_ci(df)
        r["boot_95ci"] = [round(ci[0], 4), round(ci[1], 4)] if ci else None
    return r


def main():
    master, md, td, tcd = load_master_cohorts()
    print(f"direct_all={len(md)} direct_td={len(td)} direct_tcd={len(tcd)}")

    # ---------- Task 1: 去 c 消融 (tcd, 直接用落盘 OOF) ----------
    eval_oofs = {}
    for s in ("t", "c", "d"):
        p = os.path.join(PROBE_DIR, f"eval_{s}.csv")
        eval_oofs[s] = pd.read_csv(p)
    clin_tcd = pd.read_csv(CLIN_OOF_TCD)[["patient_id", "fold", "label_T", "prob"]]
    print(f"saved OOF rows: t={len(eval_oofs['t'])} c={len(eval_oofs['c'])} d={len(eval_oofs['d'])} clin={len(clin_tcd)}")

    tcd_all = {"clin": clin_tcd, **eval_oofs}
    tcd_no_c = {"clin": clin_tcd, "t": eval_oofs["t"], "d": eval_oofs["d"]}
    f_full = tsm.fuse_zlogit(tcd_all, md)
    f_noc = tsm.fuse_zlogit(tcd_no_c, md)
    for tag, f in (("clin+d+t+c", f_full), ("clin+d+t", f_noc)):
        f["label_T"] = f["patient_id"].map(md.set_index("patient_id")["label_T"])
        f["fold"] = f["patient_id"].map(md.set_index("patient_id")["fold"])
    f_full = slice_ids(f_full, tcd)
    f_noc = slice_ids(f_noc, tcd)
    m_full = clf_metrics(f_full, "tcd_full")
    m_noc = clf_metrics(f_noc, "tcd_noc")
    dl1 = epm.delong_paired_test(f_noc, f_full)   # clin+d+t vs clin+d+t+c
    print("\n== Task1 去c消融 (tcd n=162) ==")
    print(f"clin+d+t   : {m_noc}")
    print(f"clin+d+t+c : {m_full}")
    print(f"DeLong (clin+d+t vs clin+d+t+c): {dl1[0]} n_paired={dl1[1]}")

    # ---------- Task 2: direct_td 功效验证 ----------
    # 重出 dinov2/std 全直集 OOF (与落盘同模型), 校验 tcd 截取一致后截取 td
    device = torch_device_cpu()
    full_probe = {}
    for s in ("t", "c", "d"):
        df, cs = tsm.linear_probe_plane(master=md, data_root=DATA_ROOT, backbone="dinov2",
                                        seq=s, device=device, seed=0,
                                        target="label_T", c_grid=tsm.PROBE_C_GRID,
                                        feat_norm="std")
        full_probe[s] = df
        # 校验: tcd 截取与落盘 eval 逐行 prob 一致
        check = slice_ids(df, tcd).sort_values("patient_id").reset_index(drop=True)
        saved = eval_oofs[s].sort_values("patient_id").reset_index(drop=True)
        maxdiff = float(np.abs(check["prob"].values - saved["prob"].values).max()) if len(check) else 0.0
        print(f"[probe dinov2/std {s}] full={len(df)} td={len(slice_ids(df, td))} "
              f"tcd={len(check)} max_prob_diff_vs_saved={maxdiff:.2e}")
        if len(check) and (len(check) != len(saved) or maxdiff > 1e-9):
            print(f"  !! MISMATCH vs saved eval_{s}.csv")
    # 旧基线 d_conv (route2) + clin_clean 在 td 上的 OOF
    d_old_all, _ = rc.conv_feats_npy()
    d_pids = np.array([int(p) for p in d_old_all.index])
    d_ids = sorted(set(d_pids) & set(md["patient_id"]))
    dX = d_old_all.loc[d_ids].values
    m_ = md.set_index("patient_id")
    dy = m_.loc[d_ids, "label_T"].values.astype(float)
    dfl = m_.loc[d_ids, "fold"].values.astype(int)
    p_dold, _ = rc.per_fold_lr(dX, dy, dfl, imp=None)
    d_old_oof = pd.DataFrame({"patient_id": d_ids, "label_T": dy.astype(int),
                              "fold": dfl, "prob": p_dold})
    # clin_clean 在 direct_all 上的 OOF (fold-internal median), 截取 td
    raw = pd.read_csv(PAIRS).set_index("patient_id")
    Xc = raw.loc[md["patient_id"], rc.CLIN_COLS]
    yc = md["label_T"].values.astype(float)
    fc = md["fold"].values.astype(int)
    p_clin, _ = rc.per_fold_lr(Xc.values, yc, fc, imp="median")
    clin_full = pd.DataFrame({"patient_id": md["patient_id"].values, "label_T": yc.astype(int),
                              "fold": fc, "prob": p_clin})

    probe_td = {s: slice_ids(full_probe[s], td) for s in ("t", "d")}
    clin_td = slice_ids(clin_full, td)
    d_old_td = slice_ids(d_old_oof, td)
    f_probe_td = tsm.fuse_zlogit({"clin": clin_td, "t": probe_td["t"], "d": probe_td["d"]}, md)
    f_s2_td = tsm.fuse_zlogit({"clin": clin_td, "d": d_old_td}, md)
    for tag, f in (("probe_clin_d_t", f_probe_td), ("s2_clin_dconv", f_s2_td)):
        f["label_T"] = f["patient_id"].map(md.set_index("patient_id")["label_T"])
        f["fold"] = f["patient_id"].map(md.set_index("patient_id")["fold"])
    f_probe_td = slice_ids(f_probe_td, td)
    f_s2_td = slice_ids(f_s2_td, td)
    m_probe = clf_metrics(f_probe_td, "td_probe", with_ci=True)
    m_s2 = clf_metrics(f_s2_td, "td_s2", with_ci=True)
    dl2 = epm.delong_paired_test(f_probe_td, f_s2_td)
    print("\n== Task2 direct_td (n=214) 功效 ==")
    print(f"probe dinov2/std clin+d+t : {m_probe}")
    print(f"old s2 clin+d_conv        : {m_s2}")
    print(f"DeLong (probe vs s2): {dl2[0]} n_paired={dl2[1]}")

    out = {"task1_tcd_ablation": {"clin_d_t": m_noc, "clin_d_t_c": m_full,
                                  "delong": dl2[0], "n_paired": dl2[1]},
           "task2_direct_td": {"probe_clin_d_t": m_probe, "s2_clin_dconv": m_s2,
                               "delong": dl2[0], "n_paired": dl2[1]}}
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {OUT}")


def torch_device_cpu():
    import torch
    return torch.device("cpu")


if __name__ == "__main__":
    main()
