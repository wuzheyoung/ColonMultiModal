# -*- coding: utf-8 -*-
"""exp_v2_round2_clean.py — m01174 方案C 干净特征 (self→from_t, d 201 例, 清除 from_c) 上重跑
probe_dinov2_std (MeanMax 1536 + 训练折 std + 折内嵌套3折CV L2-LR, direct_all 280, label_T, ddof=1)。

1. 主评估 direct_tcd (n=162): 单路 d/t/c + 多模态 zlogit 融合 (clin_clean 复用落盘 OOF):
   clin+t / clin+d+t (主模型) / clin+d+t+c (三平面对照); Pooled AUC + 95%CI + Fold-Mean±SD;
   3 组配对 DeLong (n=162): ① 干净 clin+d+t vs 旧 s4_clin_3plane.f1 (0.7016);
   ② vs clin+t (救回 DWI 边际); ③ vs clin+d+t+c (剔 c 增益)。
2. from_t 救回实效拆解 (tcd): mask_source self(75) vs from_t(30) 在 d 单路与 clin+d+t 融合的 AUC;
   旧缺失 81 例拆为 新救回 from_t(30) vs 仍缺(51) 在 clin+d+t 下 AUC (对照旧 0.6448)。
3. direct_td (n=214): 干净 probe clin+t / clin+d+t vs 旧 s2_clin_dconv, Pooled+CI+Fold±SD + 配对 DeLong。
落盘 logs/exp_v2_round2_clean_T.{json,md}; OOF 存 outputs/exp_v2_round2_clean/。
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
CLIN_OOF_TCD = "outputs/oof_baseline_n162/singles/clin_clean.csv"
OLD_S4_F1 = "outputs/oof_baseline_n162/fusions/s4_clin_3plane.f1.csv"
OUT = "outputs/exp_v2_round2_clean"
JOUT = "logs/exp_v2_round2_clean_T.json"
MOUT = "logs/exp_v2_round2_clean_T.md"


def load_cohorts():
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


def metrics(df, with_ci=True):
    df = df.dropna(subset=["prob"]).reset_index(drop=True)
    n = len(df)
    auc = epm.pooled_auc(df)
    fm = epm.fold_mean_sd(df)
    r = {"n": n, "n_pos": int(df["label_T"].sum()), "pooled_auc": round(float(auc), 4),
         "fold_mean": round(float(fm[0]), 4) if fm else None,
         "fold_sd": round(float(fm[1]), 4) if fm else None}
    if with_ci:
        ci = epm.boot_ci(df)
        r["boot_95ci"] = [round(ci[0], 4), round(ci[1], 4)] if ci else None
    return r


def mask_source_map():
    """dinov2/d npz 的 mask_source 标签 → {pid: source} (self / from_t)。
    方案C 已清除 from_c, 剩余 'remapped' 标签均为干净 from_t。"""
    d = os.path.join(DATA_ROOT, "dinov2", "d")
    m = {}
    for f in os.listdir(d):
        if not f.endswith(".npz"):
            continue
        pid = int(f[:-4])
        z = np.load(os.path.join(d, f))
        tag = str(z["mask_source"][0])
        m[pid] = "self" if tag == "self" else "from_t"
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    master, md, td, tcd = load_cohorts()
    print(f"direct_all={len(md)} direct_td={len(td)} direct_tcd={len(tcd)}")
    md_i = md.set_index("patient_id")

    # ---- probe dinov2/std 全直集 OOF ----
    device = torch_device_cpu()
    full = {}
    for s in ("t", "c", "d"):
        df, cs = tsm.linear_probe_plane(master=md, data_root=DATA_ROOT, backbone="dinov2",
                                        seq=s, device=device, seed=0,
                                        target="label_T", c_grid=tsm.PROBE_C_GRID,
                                        feat_norm="std")
        full[s] = df
        print(f"[probe dinov2/std {s}] full={len(df)} tcd={len(slice_ids(df, tcd))} td={len(slice_ids(df, td))}")

    # ---- 单路 (tcd) ----
    singles = {}
    for s in ("d", "t", "c"):
        o = slice_ids(full[s], tcd)
        o.to_csv(os.path.join(OUT, f"single_{s}.csv"), index=False)
        singles[s] = metrics(o)
        print(f"single {s}: {singles[s]}")

    # ---- 融合 (tcd, zlogit 可用模态) ----
    clin = pd.read_csv(CLIN_OOF_TCD)[["patient_id", "fold", "label_T", "prob"]]
    oofs_t = {"clin": clin, "t": slice_ids(full["t"], tcd)}
    oofs_dt = {"clin": clin, "d": slice_ids(full["d"], tcd), "t": slice_ids(full["t"], tcd)}
    oofs_dtc = {"clin": clin, "d": slice_ids(full["d"], tcd), "t": slice_ids(full["t"], tcd),
                "c": slice_ids(full["c"], tcd)}
    fusions = {}
    for tag, oo in (("clin_t", oofs_t), ("clin_d_t", oofs_dt), ("clin_d_t_c", oofs_dtc)):
        f = tsm.fuse_zlogit(oo, md)
        f["label_T"] = f["patient_id"].map(md_i["label_T"])
        f["fold"] = f["patient_id"].map(md_i["fold"])
        f = slice_ids(f, tcd)
        f.to_csv(os.path.join(OUT, f"fusion_{tag}.csv"), index=False)
        fusions[tag] = metrics(f)
        print(f"fusion {tag}: {fusions[tag]}")

    # ---- DeLong ① ② ③ ----
    old_s4 = pd.read_csv(OLD_S4_F1)
    old_s4["label_T"] = old_s4["patient_id"].map(md_i["label_T"])
    old_s4["fold"] = old_s4["patient_id"].map(md_i["fold"])
    old_s4 = slice_ids(old_s4, tcd)
    f_dt = pd.read_csv(os.path.join(OUT, "fusion_clin_d_t.csv"))
    f_t = pd.read_csv(os.path.join(OUT, "fusion_clin_t.csv"))
    f_dtc = pd.read_csv(os.path.join(OUT, "fusion_clin_d_t_c.csv"))
    dl = {}
    dl["dt_vs_old_s4"] = {"res": epm.delong_paired_test(f_dt, old_s4)[0],
                          "n_paired": epm.delong_paired_test(f_dt, old_s4)[1]}
    dl["dt_vs_t"] = {"res": epm.delong_paired_test(f_dt, f_t)[0],
                     "n_paired": epm.delong_paired_test(f_dt, f_t)[1]}
    dl["dt_vs_dtc"] = {"res": epm.delong_paired_test(f_dt, f_dtc)[0],
                       "n_paired": epm.delong_paired_test(f_dt, f_dtc)[1]}
    for k, v in dl.items():
        print(f"DeLong {k}: {v['res']} n_paired={v['n_paired']}")

    # ---- from_t 救回实效拆解 (tcd) ----
    smap = mask_source_map()
    d_oof = pd.read_csv(os.path.join(OUT, "single_d.csv"))
    d_oof["src"] = d_oof["patient_id"].map(smap)
    breakdown = {}
    for src, label in (("self", "self"), ("from_t", "from_t")):
        sub = d_oof[d_oof["src"] == src]
        breakdown[label + "_d_single"] = metrics(sub, with_ci=False)
    # 融合子集
    fdt = pd.read_csv(os.path.join(OUT, "fusion_clin_d_t.csv"))
    fdt["src"] = fdt["patient_id"].map(smap).fillna("none")
    for src, label in (("self", "self"), ("from_t", "from_t")):
        sub = fdt[fdt["src"] == src]
        breakdown[label + "_fusion"] = metrics(sub, with_ci=False)
    # 旧 d_conv 覆盖集 (route2) 在 tcd
    d_old_all, _ = rc.conv_feats_npy()
    old_d_pids = set(int(p) for p in d_old_all.index)
    old_missing = sorted(tcd - old_d_pids)          # 81
    new_fromt = sorted(p for p in old_missing if smap.get(p) == "from_t")
    still_missing = sorted(p for p in old_missing if p not in new_fromt)
    print(f"old_missing={len(old_missing)} new_fromt={len(new_fromt)} still_missing={len(still_missing)}")
    fdt_oldmiss = fdt[fdt["patient_id"].isin(old_missing)]
    breakdown["old_missing_81_fusion"] = metrics(fdt_oldmiss, with_ci=False)
    breakdown["new_fromt_30_fusion"] = metrics(fdt[fdt["patient_id"].isin(new_fromt)], with_ci=False)
    breakdown["still_missing_51_fusion"] = metrics(fdt[fdt["patient_id"].isin(still_missing)], with_ci=False)
    for k, v in breakdown.items():
        print(f"breakdown {k}: {v}")

    # ---- direct_td (n=214) 功效 ----
    raw = pd.read_csv(PAIRS).set_index("patient_id")
    Xc = raw.loc[md["patient_id"], rc.CLIN_COLS]
    yc = md["label_T"].values.astype(float)
    fc = md["fold"].values.astype(int)
    p_clin, _ = rc.per_fold_lr(Xc.values, yc, fc, imp="median")
    clin_full = pd.DataFrame({"patient_id": md["patient_id"].values, "label_T": yc.astype(int),
                              "fold": fc, "prob": p_clin})
    clin_td = slice_ids(clin_full, td)
    t_td = slice_ids(full["t"], td)
    d_td = slice_ids(full["d"], td)
    f_t_td = tsm.fuse_zlogit({"clin": clin_td, "t": t_td}, md)
    f_dt_td = tsm.fuse_zlogit({"clin": clin_td, "d": d_td, "t": t_td}, md)
    # 旧 s2: clin + 旧 route2 d_conv
    d_ids = sorted(set(int(p) for p in d_old_all.index) & set(md["patient_id"]))
    dX = d_old_all.loc[d_ids].values
    dy = md_i.loc[d_ids, "label_T"].values.astype(float)
    dfl = md_i.loc[d_ids, "fold"].values.astype(int)
    p_dold, _ = rc.per_fold_lr(dX, dy, dfl, imp=None)
    d_old_oof = pd.DataFrame({"patient_id": d_ids, "label_T": dy.astype(int),
                              "fold": dfl, "prob": p_dold})
    d_old_td = slice_ids(d_old_oof, td)
    f_s2_td = tsm.fuse_zlogit({"clin": clin_td, "d": d_old_td}, md)
    td_res = {}
    for tag, f in (("clin_t", f_t_td), ("clin_d_t", f_dt_td), ("s2_clin_dconv", f_s2_td)):
        f["label_T"] = f["patient_id"].map(md_i["label_T"])
        f["fold"] = f["patient_id"].map(md_i["fold"])
        f = slice_ids(f, td)
        f.to_csv(os.path.join(OUT, f"td_{tag}.csv"), index=False)
        td_res[tag] = metrics(f)
        print(f"td {tag}: {td_res[tag]}")
    td_dl = {}
    td_dl["dt_vs_t"] = {"res": epm.delong_paired_test(f_dt_td, f_t_td)[0],
                        "n_paired": epm.delong_paired_test(f_dt_td, f_t_td)[1]}
    td_dl["dt_vs_s2"] = {"res": epm.delong_paired_test(f_dt_td, f_s2_td)[0],
                         "n_paired": epm.delong_paired_test(f_dt_td, f_s2_td)[1]}
    for k, v in td_dl.items():
        print(f"td DeLong {k}: {v['res']} n_paired={v['n_paired']}")

    out = {"direct_all": len(md), "direct_td": len(td), "direct_tcd": len(tcd),
           "singles": singles, "fusions_tcd": fusions, "delong_tcd": dl,
           "breakdown": breakdown, "td": td_res, "delong_td": td_dl}
    with open(JOUT, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    # ---- Markdown ----
    def line(name, m):
        ci = f" [{m['boot_95ci'][0]:.4f}, {m['boot_95ci'][1]:.4f}]" if m.get("boot_95ci") else ""
        fm = f" {m['fold_mean']:.4f} ± {m['fold_sd']:.4f}" if m.get("fold_mean") is not None else ""
        return f"| {name} | {m['n']} | {m['pooled_auc']:.4f}{ci} |{fm} |"
    md_rows = []
    md_rows.append("# exp_v2_round2_clean — 方案C 干净特征 (self→from_t, 清除 from_c) probe_dinov2_std (ddof=1)\n")
    md_rows.append("训练 direct_all=280, 主评估 direct_tcd=162, 功效验证 direct_td=214; 融合=可用模态 zlogit; d 掩码: self 139 + from_t 62 (全量), tcd 内 self 75 + from_t 30。\n")
    md_rows.append("## 1. 主评估 (direct_tcd n=162) 单路与多模态\n")
    md_rows.append("| 模型 | n | Pooled OOF AUC (95% CI) | Fold-Mean ± SD |")
    md_rows.append("|---|---|---|---|")
    for tag, m in (("single_d", singles["d"]), ("single_t", singles["t"]), ("single_c", singles["c"]),
                   ("clin+t", fusions["clin_t"]), ("clin+d+t (主模型)", fusions["clin_d_t"]),
                   ("clin+d+t+c (三平面对照)", fusions["clin_d_t_c"])):
        md_rows.append(line(tag, m))
    md_rows.append("\n## 2. 配对 DeLong (direct_tcd n=162)\n")
    md_rows.append("| 对比 | ΔAUC | z | p | n_paired |")
    md_rows.append("|---|---|---|---|---|")
    for k, v in (("① 干净 clin+d+t vs 旧 s4_clin_3plane.f1(0.7016)", dl["dt_vs_old_s4"]),
                 ("② 干净 clin+d+t vs clin+t (DWI 救回边际)", dl["dt_vs_t"]),
                 ("③ 干净 clin+d+t vs clin+d+t+c (剔 c 增益)", dl["dt_vs_dtc"])):
        r = v["res"]
        md_rows.append(f"| {k} | {r['diff']:+.4f} | {r['z']:.4f} | {r['p']:.4f} | {v['n_paired']} |")
    md_rows.append("\n## 3. from_t 救回实效拆解 (direct_tcd n=162)\n")
    md_rows.append("| 拆解 | n | d 单路 AUC | clin+d+t 融合 AUC |")
    md_rows.append("|---|---|---|---|")
    for src in ("self", "from_t"):
        md_rows.append(f"| mask_source={src} | {breakdown[src + '_d_single']['n']} | "
                       f"{breakdown[src + '_d_single']['pooled_auc']:.4f} | "
                       f"{breakdown[src + '_fusion']['pooled_auc']:.4f} |")
    md_rows.append(f"| 旧缺失 81 例整体 | {breakdown['old_missing_81_fusion']['n']} | — | {breakdown['old_missing_81_fusion']['pooled_auc']:.4f} |")
    md_rows.append(f"| 其中新救回 from_t 30 例 | {breakdown['new_fromt_30_fusion']['n']} | — | {breakdown['new_fromt_30_fusion']['pooled_auc']:.4f} |")
    md_rows.append(f"| 其中仍缺 DWI 51 例 | {breakdown['still_missing_51_fusion']['n']} | — | {breakdown['still_missing_51_fusion']['pooled_auc']:.4f} |")
    md_rows.append("\n> 旧基线对照: 旧 s4_clin_3plane.f1 全 162 例 pooled 0.7016; 旧缺失 81 例子集 0.6448。\n")
    md_rows.append("\n## 4. 扩大样本 direct_td (n=214) 功效验证\n")
    md_rows.append("| 模型 | n | Pooled OOF AUC (95% CI) | Fold-Mean ± SD |")
    md_rows.append("|---|---|---|---|")
    for tag, m in (("probe clin+t", td_res["clin_t"]), ("probe clin+d+t", td_res["clin_d_t"]),
                   ("旧 s2_clin_dconv", td_res["s2_clin_dconv"])):
        md_rows.append(line(tag, m))
    md_rows.append("\n| 对比 | ΔAUC | z | p | n_paired |")
    md_rows.append("|---|---|---|---|---|")
    for k, v in (("probe clin+d+t vs probe clin+t", td_dl["dt_vs_t"]),
                 ("probe clin+d+t vs 旧 s2_clin_dconv", td_dl["dt_vs_s2"])):
        r = v["res"]
        md_rows.append(f"| {k} | {r['diff']:+.4f} | {r['z']:.4f} | {r['p']:.4f} | {v['n_paired']} |")
    with open(MOUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_rows))
    print(f"saved -> {JOUT} / {MOUT}")


def torch_device_cpu():
    import torch
    return torch.device("cpu")


if __name__ == "__main__":
    main()
