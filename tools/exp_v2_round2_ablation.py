# -*- coding: utf-8 -*-
"""exp_v2_round2_ablation.py — m01211 快速消融 (不重提特征, 干净 slice_feats_v2/dinov2)。

评估口径: direct_tcd n=162 & direct_td n=214, ddof=1, label_T。
1. 停用噪声 from_t: d 分类器仅在 mask_source=='self' (direct_all 内 n=104) 上 5 折训练,
   可用模态 zlogit 融合 (self DWI 病例 clin+d_self+t, 无 self 病例退化 clin+t);
   对比 MeanMax 1536 在 n=162/n=214 的 Pooled OOF AUC 与 Fold-Mean±SD。
2. 3 种低维抗噪池化 (Mean 768 / AreaWeightedMean 768 / Top3-AreaMean 768),
   均配合训练折 std + 折内嵌套 3 折 CV L2-LR, 在 d_self 与 t 上, 汇报
   d_self(n=75)/t(n=155)/clin+d_self+t(n=162 & n=214)。
3. 只读核查 (不计算): 旧 d_conv/t_conv 特征提取 (route2_axial_mil.py / audit_tgr_task2c.py)
   均为每层 mask 质心固定 120mm 方框, 与 extract_slice_feats_v2.py 相同 —— 非紧致 ROI bbox。
落盘 logs/exp_v2_round2_ablation_T.{json,md}; OOF 存 outputs/exp_v2_round2_ablation/。
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import recompute_direct_tcd_singles as rc  # noqa: E402
import train_slice_mil as tsm             # noqa: E402
import eval_paper_metrics as epm          # noqa: E402

MASTER = "data/master_5fold_356.csv"
COHORT = "data/direct_cohort_ids.csv"
PAIRS = "data/hos_pairs_raw.csv"
DATA_ROOT = "features/slice_feats_v2"
BACKBONE = "dinov2"
CLIN_OOF_TCD = "outputs/oof_baseline_n162/singles/clin_clean.csv"
OUT = "outputs/exp_v2_round2_ablation"
JOUT = "logs/exp_v2_round2_ablation_T.json"
MOUT = "logs/exp_v2_round2_ablation_T.md"
PROBE_C_GRID = tsm.PROBE_C_GRID


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


def self_d_pids(md):
    """direct_all 内 mask_source=='self' 的 d 患者 (不含 from_t/remapped)。"""
    d = os.path.join(DATA_ROOT, BACKBONE, "d")
    on_disk = set()
    for fn in os.listdir(d):
        if fn.endswith(".npz"):
            pid = int(fn[:-4])
            z = np.load(os.path.join(d, fn))
            if str(z["mask_source"][0]) == "self":
                on_disk.add(pid)
    md_ids = set(md["patient_id"])
    return np.array(sorted(on_disk & md_ids), dtype=np.int64)


def pool_bags(bags, pids, pool):
    """池化 → (n, dim) 特征矩阵。pool ∈ {meanmax, mean, area_w, top3}。
    meanmax = concat(mean, max) → 1536 维 (round2 基线);
    mean = 切片均值 → 768 (去掉 Max 抗边缘切片);
    area_w = mask_area_ratio 加权均值 → 768 (抑制肿瘤上下极小截面);
    top3  = mask_area_ratio 最大前 3 层均值 → 768 (病灶最大截面)。"""
    Xs, ys = [], []
    for p in pids:
        f = bags[p][0].float()              # (K,768)
        K = f.shape[0]
        if pool == "meanmax":
            v = torch.cat([f.mean(dim=0), f.max(dim=0).values])
        elif pool == "mean":
            v = f.mean(dim=0)
        elif pool == "area_w":
            a = bags[p][2].float()          # (K,)
            s = a.sum().clamp(min=1e-9)
            v = (f * a.unsqueeze(-1)).sum(dim=0) / s
        elif pool == "top3":
            a = bags[p][2].float()
            k = min(3, K)
            idx = torch.topk(a, k).indices
            v = f[idx].mean(dim=0)
        else:
            raise ValueError(pool)
        Xs.append(v.cpu().numpy())
        ys.append(bags[p][3])
    return np.array(Xs), np.array(ys)


def probe_subset(md, pids, seq, pool, feat_norm="std", c_grid=PROBE_C_GRID, seed=0):
    """在 pids 子集 (如 d_self) 上做折内嵌套 3 折 CV 选 C 的 L2-LR 探针。
    5 折锁死 master 折列 (子集内 fold 分布), OOF 契约 patient_id,fold,label_T,prob。
    feat_norm='std': 池化后逐维折内 StandardScaler (fit 仅训练折)。"""
    device = torch.device("cpu")
    m = md.set_index("patient_id")
    have = tsm.collect_plane_pids(md, DATA_ROOT, BACKBONE, seq, set(pids))
    bags = tsm.preload_plane(md, DATA_ROOT, BACKBONE, seq, have, "label_T", "none", device)
    X, y = pool_bags(bags, have, pool)
    folds = m.loc[have, "fold"].values.astype(int)
    oof = np.full(len(have), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            Cs[k] = None
            continue
        Xtr, ytr = X[tr], y[tr]
        inner = folds[tr] % 3
        inner_auc = {}
        for c in c_grid:
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(np.unique(ytr[itr])) < 2:
                    continue
                sc = StandardScaler().fit(Xtr[itr]) if feat_norm == "std" else None
                Xi_tr = sc.transform(Xtr[itr]) if sc is not None else Xtr[itr]
                Xi_va = sc.transform(Xtr[iva]) if sc is not None else Xtr[iva]
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(Xi_tr, ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(Xi_va)[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(c_grid, key=lambda c: inner_auc[c])
        if feat_norm == "std":
            sc = StandardScaler().fit(Xtr)
            mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
            oof[va] = mm.predict_proba(sc.transform(X[va]))[:, 1]
        else:
            mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(Xtr, ytr)
            oof[va] = mm.predict_proba(X[va])[:, 1]
        Cs[int(k)] = float(best_c)
    rows = [{"patient_id": int(p), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp in zip(have, y, oof)]
    df = pd.DataFrame(rows)
    df["fold"] = df["patient_id"].map(m["fold"]).astype(int)
    return df[["patient_id", "fold", "label_T", "prob"]].sort_values("patient_id"), Cs


def clin_full_oof(md):
    """direct_all 全量临床 OOF (折内中位数填补, 与 baseline 同口径), 供 n=214 截取。"""
    raw = pd.read_csv(PAIRS).set_index("patient_id")
    Xc = raw.loc[md["patient_id"], rc.CLIN_COLS]
    yc = md["label_T"].values.astype(float)
    fc = md["fold"].values.astype(int)
    p, _ = rc.per_fold_lr(Xc.values, yc, fc, imp="median")
    return pd.DataFrame({"patient_id": md["patient_id"].values, "label_T": yc.astype(int),
                         "fold": fc, "prob": p})


def main():
    os.makedirs(OUT, exist_ok=True)
    master, md, td, tcd = load_cohorts()
    md_i = md.set_index("patient_id")
    self_pids = self_d_pids(md)
    n_self_dir = len(self_pids)
    n_self_tcd = len(set(self_pids) & tcd)
    n_self_td = len(set(self_pids) & td)
    print(f"direct_all={len(md)} td={len(td)} tcd={len(tcd)} "
          f"d_self direct={n_self_dir} tcd={n_self_tcd} td={n_self_td}")

    clin_tcd = pd.read_csv(CLIN_OOF_TCD)[["patient_id", "fold", "label_T", "prob"]]
    clin_all = clin_full_oof(md)

    pools = ["meanmax", "mean", "area_w", "top3"]
    res = {"d_self_counts": {"direct_all": int(n_self_dir), "tcd": int(n_self_tcd),
                             "td": int(n_self_td)}}
    oofs_save = {}

    for pool in pools:
        # d_self 探针 (仅 self 训练)
        d_df, d_cs = probe_subset(md, self_pids, "d", pool)
        d_df.to_csv(os.path.join(OUT, f"dself_{pool}_full.csv"), index=False)
        # t 探针 (全直集)
        t_all = np.sort(md["patient_id"].values.astype(np.int64))
        t_df, t_cs = probe_subset(md, t_all, "t", pool)
        t_df.to_csv(os.path.join(OUT, f"t_{pool}_full.csv"), index=False)

        d_tcd = slice_ids(d_df, tcd)
        t_tcd = slice_ids(t_df, tcd)
        d_td = slice_ids(d_df, td)
        t_td = slice_ids(t_df, td)

        # 融合: 可用模态 zlogit (clin + d_self + t; 无 self d 自动退化 clin+t)
        f162 = tsm.fuse_zlogit({"clin": clin_tcd, "d": d_tcd, "t": t_tcd}, md)
        f214 = tsm.fuse_zlogit({"clin": slice_ids(clin_all, td), "d": d_td, "t": t_td}, md)
        for f in (f162, f214):
            f["label_T"] = f["patient_id"].map(md_i["label_T"])
            f["fold"] = f["patient_id"].map(md_i["fold"])
        f162 = slice_ids(f162, tcd)
        f214 = slice_ids(f214, td)
        f162.to_csv(os.path.join(OUT, f"clin_dself_t_{pool}_162.csv"), index=False)
        f214.to_csv(os.path.join(OUT, f"clin_dself_t_{pool}_214.csv"), index=False)

        r = {"d_self_tcd": metrics(d_tcd, with_ci=False),
             "t_tcd": metrics(t_tcd, with_ci=False),
             "clin_dself_t_162": metrics(f162),
             "clin_dself_t_214": metrics(f214),
             "d_self_Cs": {str(k): v for k, v in (d_cs or {}).items()},
             "t_Cs": {str(k): v for k, v in (t_cs or {}).items()}}
        res[pool] = r
        print(f"[{pool}] d_self(tcd)={r['d_self_tcd']} t(tcd)={r['t_tcd']} "
              f"162={r['clin_dself_t_162']} 214={r['clin_dself_t_214']}")

    # 对照: 纯 clin+t (无 d) 于 162/214
    t_mm, _ = probe_subset(md, np.sort(md["patient_id"].values.astype(np.int64)), "t", "meanmax")
    f_t162 = tsm.fuse_zlogit({"clin": clin_tcd, "t": slice_ids(t_mm, tcd)}, md)
    f_t214 = tsm.fuse_zlogit({"clin": slice_ids(clin_all, td), "t": slice_ids(t_mm, td)}, md)
    for f in (f_t162, f_t214):
        f["label_T"] = f["patient_id"].map(md_i["label_T"])
        f["fold"] = f["patient_id"].map(md_i["fold"])
    res["clin_t_162"] = metrics(slice_ids(f_t162, tcd))
    res["clin_t_214"] = metrics(slice_ids(f_t214, td))
    print(f"[clin+t] 162={res['clin_t_162']} 214={res['clin_t_214']}")

    # DeLong: d_self+t vs clin+t (meanmax, n=162/214) — DWI 净边际
    dl = {}
    f162_dt = pd.read_csv(os.path.join(OUT, "clin_dself_t_meanmax_162.csv"))
    f162_t = pd.DataFrame({"patient_id": res and slice_ids(f_t162, tcd)["patient_id"]})
    f162_t = slice_ids(f_t162, tcd)[["patient_id", "fold", "label_T", "prob"]]
    dl["dt162_vs_t162"] = {"res": epm.delong_paired_test(f162_dt, f162_t)[0],
                           "n_paired": epm.delong_paired_test(f162_dt, f162_t)[1]}
    f214_dt = pd.read_csv(os.path.join(OUT, "clin_dself_t_meanmax_214.csv"))
    f214_t = slice_ids(f_t214, td)[["patient_id", "fold", "label_T", "prob"]]
    dl["dt214_vs_t214"] = {"res": epm.delong_paired_test(f214_dt, f214_t)[0],
                           "n_paired": epm.delong_paired_test(f214_dt, f214_t)[1]}
    res["delong"] = dl
    for k, v in dl.items():
        print(f"DeLong {k}: {v['res']} n_paired={v['n_paired']}")

    res["crop_audit"] = {
        "route2_axial_mil_t_conv": "每层 mask 质心裁固定 120×120mm 物理方框 (CROP_MM=120.0, 无变比例拉伸, 边界不足以中心外扩)",
        "audit_tgr_task2c_d_c_conv": "逐轴位层 mask 质心裁 120mm 物理正方形 (SIZE_MM=120.0)",
        "extract_slice_feats_v2_new": "每层 mask 质心 120mm 物理方框 (SIZE_MM=120.0, 边界不足 pad)",
        "conclusion": "旧 d_conv/t_conv 与新特征均用固定 120mm mask-质心大框, 非紧致肿瘤 bbox; "
                      "'旧版紧致框稀释' 假说不成立, 紧致 ROI bbox 仍是明天 4060 的可测改进点"}
    with open(JOUT, "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)

    # ---- Markdown ----
    md_rows = ["# exp_v2_round2_ablation — d_self 仅训练 + 低维抗噪池化 (dinov2, std, ddof=1)",
               "",
               f"d_self (direct_all) n={n_self_dir} | tcd n={n_self_tcd} | td n={n_self_td}; "
               "融合=可用模态 zlogit (self DWI 病例 clin+d_self+t, 无 self 退化 clin+t)。",
               "",
               "## 1. d_self 仅训练 vs 纯 clin+t (MeanMax 1536 基线)",
               "",
               "| 口径 | clin+d_self+t | clin+t (无 d) | Δ (DeLong) | p |",
               "|---|---|---|---|---|"]
    for tag, a, b in (("n=162", "clin_dself_t_162", "clin_t_162"),
                      ("n=214", "clin_dself_t_214", "clin_t_214")):
        ra, rb = res["meanmax"][a], res[b]
        nn = tag.split("=")[1]
        d = dl[f"dt{nn}_vs_t{nn}"]
        md_rows.append(f"| {tag} | {ra['pooled_auc']:.4f} [{ra['boot_95ci'][0]:.4f},{ra['boot_95ci'][1]:.4f}] "
                       f"({ra['fold_mean']:.4f}±{ra['fold_sd']:.4f}) | {rb['pooled_auc']:.4f} "
                       f"({rb['fold_mean']:.4f}±{rb['fold_sd']:.4f}) | {d['res']['diff']:+.4f} | {d['res']['p']:.4f} |")
    md_rows += ["",
                "## 2. 3 种低维抗噪池化 (训练折 std + 嵌套 CV L2-LR)",
                "",
                "| 池化 | 维数 | d_self 单路 (n=75) | t 单路 (n=155) | clin+d_self+t n=162 | clin+d_self+t n=214 |",
                "|---|---|---|---|---|---|"]
    for pool in ("meanmax", "mean", "area_w", "top3"):
        r = res[pool]
        md_rows.append(f"| {pool} | {'1536' if pool=='meanmax' else '768'} | "
                       f"{r['d_self_tcd']['pooled_auc']:.4f} (n={r['d_self_tcd']['n']}) | "
                       f"{r['t_tcd']['pooled_auc']:.4f} (n={r['t_tcd']['n']}) | "
                       f"{r['clin_dself_t_162']['pooled_auc']:.4f} "
                       f"[{r['clin_dself_t_162']['boot_95ci'][0]:.4f},{r['clin_dself_t_162']['boot_95ci'][1]:.4f}] "
                       f"({r['clin_dself_t_162']['fold_mean']:.4f}±{r['clin_dself_t_162']['fold_sd']:.4f}) | "
                       f"{r['clin_dself_t_214']['pooled_auc']:.4f} "
                       f"[{r['clin_dself_t_214']['boot_95ci'][0]:.4f},{r['clin_dself_t_214']['boot_95ci'][1]:.4f}] |")
    md_rows += ["",
                "## 3. 旧特征提取裁剪模式核查 (只读)",
                "",
                "- route2_axial_mil.py (t_conv): 每层 mask 质心裁固定 120×120mm 物理方框 (CROP_MM=120.0)。",
                "- audit_tgr_task2c.py (d/c_conv): 逐轴位层 mask 质心裁 120mm 物理正方形 (SIZE_MM=120.0)。",
                "- extract_slice_feats_v2.py (新): 每层 mask 质心 120mm 方框 (SIZE_MM=120.0, 边界 pad)。",
                "",
                "**结论**: 旧 d_conv/t_conv 与新特征均为固定 120mm mask-质心大框, 非紧致肿瘤 bbox; "
                "'旧版紧致框稀释' 假说不成立。紧致 ROI bbox (肿瘤边界面内框) 仍是明天 4060 端值得测试的改进点。"]
    with open(MOUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_rows))
    print(f"saved -> {JOUT} / {MOUT}")


if __name__ == "__main__":
    main()
