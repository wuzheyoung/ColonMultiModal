# -*- coding: utf-8 -*-
"""m05965 任务2+任务3: N 分期全量基准摸底 + std 归一化深度 MIL 对照 (4060 夜间挂机)。

全局硬约束:
- 训练集锁死 direct_all (n=280, 读 data/direct_cohort_ids.csv is_tgr==0), 5 折对齐
  data/master_5fold_356.csv 的 fold 列; 评估主口径 direct_tcd (n=162) 附报 direct_td (n=214)。
- Fold-SD 统一 ddof=1; 所有实验存逐样本 OOF 概率 CSV (patient_id,fold,label,prob);
  汇总 logs/exp_4060_overnight_summary.json + .md。

任务2: label_N 嵌套 CV 线性探针 (MeanMax + 训练折 std + L2-LR C∈{1e-4..1.0})
  - 视图: clin_clean / d / t / clin+d+t; 每视图在 direct_all 上 5 折 OOF, 评估取 tcd/td 子集
    pooled AUC + 95% CI (bootstrap 2000) + Fold-Mean ± SD (ddof=1)。
  - 2.5D 增益: 同流程跑 slice_feats_v2_25d 的 label_T/label_N 对比 2D。

任务3: label_T GatedAttentionMIL (std 归一化, hidden=32/dropout=0.6/wd=0.05/epochs=15/lr=3e-4)
  带与不带 --spatial-prior, 在 t/d/clin+d+t 上的 n=162 (direct_tcd) 结果。
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_slice_mil as tsm  # 复用 collect_plane_pids/preload_plane/oof_for_plane/fuse_*

MASTER = "data/master_5fold_356.csv"
DIRECT = "data/direct_cohort_ids.csv"
RAW = "data/hos_pairs_raw.csv"
CLIN_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
OUTDIR = "outputs/exp_4060_overnight"
SUMMARY_JSON = "logs/exp_4060_overnight_summary.json"
SUMMARY_MD = "logs/exp_4060_overnight_summary.md"
PROBE_C_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0)


# ---------------- 子集定义 ----------------
def load_subsets():
    master = pd.read_csv(MASTER)
    dc = pd.read_csv(DIRECT)
    direct_ids = set(dc[dc["is_tgr"] == 0]["pid"].astype(int))
    m = master.set_index("patient_id")
    # direct_all master 子集 (fold 列保留)
    md = master[master["patient_id"].isin(direct_ids)].copy()
    # 评估子集: 由特征存在性决定 (调用处传入), 这里仅返回 direct 集
    return master, md, direct_ids


def has_feat(pid, seq, root, backbone):
    p = os.path.join(root, backbone, seq, f"{int(pid)}.npz")
    return os.path.exists(p) and os.path.getsize(p) > 100


def eval_subset_ids(master, root, backbone, seqs):
    """direct 队列子集 (影像路径存在性, 非特征覆盖): tcd=direct∩(t,c,d 影像全有), td=direct∩(t,d 有)。"""
    raw = pd.read_csv(RAW).set_index("patient_id")
    m = master.set_index("patient_id")
    _, _, direct_ids = load_subsets()
    tcd, td = [], []
    for p in sorted(direct_ids):
        if p not in m.index:
            continue
        if p not in raw.index:
            continue
        def path_ok(col):
            v = raw.loc[p, col]
            return isinstance(v, str) and os.path.exists(v)
        if path_ok("mri_path_t") and path_ok("mri_path_c") and path_ok("mri_path_d"):
            tcd.append(p)
        if path_ok("mri_path_t") and path_ok("mri_path_d"):
            td.append(p)
    return np.array(sorted(tcd)), np.array(sorted(td))


# ---------------- 评估工具 ----------------
def pooled_auc_df(df, label_col="label"):
    ok = df[label_col].notna() & df["prob"].notna()
    y = df.loc[ok, label_col].astype(int)
    p = df.loc[ok, "prob"].astype(float)
    if ok.sum() < 10 or len(set(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def fold_mean_sd(df, label_col="label"):
    """Fold-Mean ± SD (ddof=1): 每折 pooled AUC 的均值与样本标准差。"""
    aucs = []
    for k in sorted(set(df["fold"])):
        a = pooled_auc_df(df[df["fold"] == k], label_col)
        if a is not None:
            aucs.append(a)
    aucs = np.array(aucs)
    if len(aucs) < 2:
        return None, None
    return float(aucs.mean()), float(aucs.std(ddof=1))


def auc_ci_boot(df, label_col="label", n_boot=2000, seed=0):
    """bootstrap 2000 次 pooled AUC 95% CI。"""
    ok = df[label_col].notna() & df["prob"].notna()
    y = df.loc[ok, label_col].astype(int).values
    p = df.loc[ok, "prob"].astype(float).values
    if len(y) < 10 or len(set(y)) < 2:
        return None, None
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(set(yy)) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(yy, pp)
    a = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
    return float(lo), float(hi)


# ---------------- 任务2: 线性探针 (含 clin) ----------------
def clin_oof(md, target="label_N"):
    """clin_clean 7 维 L2-LR, 折内 median impute + std, C 网格折内选 (嵌套), 5 折 OOF。"""
    raw = pd.read_csv(RAW).set_index("patient_id")
    X = raw[CLIN_COLS].replace([np.inf, -np.inf], np.nan).astype(float)
    rows = []
    for _, r in md.iterrows():
        pid = int(r["patient_id"])
        y = int(r[target])
        x = X.loc[pid].values if pid in X.index else np.full(len(CLIN_COLS), np.nan)
        rows.append({"patient_id": pid, "fold": int(r["fold"]), "label": y, "x": x})
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(int)
    Xm = np.stack(df["x"].values)
    y = df["label"].values
    folds = df["fold"].values
    oof = np.full(len(df), np.nan)
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(set(y[tr])) < 2:
            oof[va] = 0.5
            continue
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        ip = SimpleImputer(strategy="median").fit(Xm[tr])
        Xtr, Xva = ip.transform(Xm[tr]), ip.transform(Xm[va])
        inner = folds[tr] % 3
        inner_auc = {}
        for c in PROBE_C_GRID:
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(set(y[tr][itr])) < 2:
                    continue
                sc = StandardScaler().fit(Xtr[itr])
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(
                    sc.transform(Xtr[itr]), y[tr][itr])
                aucs.append(roc_auc_score(y[tr][iva], mm.predict_proba(sc.transform(Xtr[iva]))[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(PROBE_C_GRID, key=lambda c: inner_auc[c])
        sc = StandardScaler().fit(Xtr)
        mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), y[tr])
        oof[va] = mm.predict_proba(sc.transform(Xva))[:, 1]
    df["prob"] = oof
    return df[["patient_id", "fold", "label", "prob"]]


def plane_probe(md, root, backbone, seq, device, target="label_N", seed=0):
    """复用 tsm.linear_probe_plane 的 MeanMax+训练折std+嵌套选C, 改输出 label 列。"""
    df, cs = tsm.linear_probe_plane(md, root, backbone, seq, device, seed, target=target)
    if df is None:
        return None
    df = df.rename(columns={"label_T": "label"})
    return df[["patient_id", "fold", "label", "prob"]]


def fuse_clin_planes(clin_df, plane_dfs, md, mode="probmean"):
    """clin + d + t 融合: 支持 probmean (可用模态等权) / zlogit (折内训练 logit z-score)。"""
    oofs = {}
    if clin_df is not None:
        oofs["clin"] = clin_df.rename(columns={"label": "label_T"})
    for s, d in plane_dfs.items():
        if d is not None:
            oofs[s] = d.rename(columns={"label": "label_T"})
    if mode == "probmean":
        fd = tsm.fuse_probmean(oofs)
    else:
        fd = tsm.fuse_zlogit(oofs, md)
    m = md.set_index("patient_id")
    fd["label"] = fd["patient_id"].map(m["label_N"] if "label_N" in m.columns else m["label_T"])
    fd["fold"] = fd["patient_id"].map(m["fold"])
    return fd[["patient_id", "fold", "label", "prob"]]


# ---------------- 评估与落盘 ----------------
def evaluate_view(df, subset_ids, tag, label_col="label"):
    sub = df[df["patient_id"].isin(subset_ids)]
    if len(sub) < 10:
        return None
    auc = pooled_auc_df(sub, label_col)
    lo, hi = auc_ci_boot(sub, label_col)
    fm, fsd = fold_mean_sd(sub, label_col)
    return {"tag": tag, "n": int(len(sub)), "n_pos": int(sub[label_col].sum()),
            "pooled_auc": auc, "ci95": [lo, hi],
            "fold_mean": fm, "fold_sd": fsd}


def save_oof(df, name, label_col="label"):
    os.makedirs(OUTDIR, exist_ok=True)
    out = df[["patient_id", "fold", "label", "prob"]].copy()
    out.to_csv(os.path.join(OUTDIR, f"oof_{name}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task2", action="store_true")
    ap.add_argument("--task3", action="store_true")
    ap.add_argument("--both", action="store_true", help="默认: 全跑")
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--root2d", default="features/slice_feats_v2")
    ap.add_argument("--root25d", default="features/slice_feats_v2_25d")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    master, md, direct_ids = load_subsets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTDIR, exist_ok=True)
    summary = {"master": "direct_all n=280 5fold(master fold)", "backbone": args.backbone,
               "seed": args.seed, "tasks": {}}

    # 子集 (2D 特征存在性)
    tcd2, td2 = eval_subset_ids(md, args.root2d, args.backbone, ("t", "c", "d"))
    print(f"[subsets] direct_tcd={len(tcd2)} direct_td={len(td2)} (2D 特征)")
    summary["subsets_2d"] = {"direct_tcd": int(len(tcd2)), "direct_td": int(len(td2))}

    if args.task2 or args.both or not (args.task2 or args.task3):
        t0 = time.time()
        res2 = {}
        for target in ("label_N",):
            print(f"\n===== 任务2: {target} 线性探针 =====")
            # clin
            cdf = clin_oof(md, target)
            save_oof(cdf, f"clin_{target}")
            # d / t 平面
            pdfs = {}
            for seq in ("d", "t"):
                df = plane_probe(md, args.root2d, args.backbone, seq, device, target, args.seed)
                if df is not None:
                    save_oof(df, f"{seq}_{target}")
                pdfs[seq] = df
                print(f"  [{seq}] OOF n={None if df is None else len(df)}")
            # 融合
            fd = fuse_clin_planes(cdf, pdfs, md, "probmean")
            save_oof(fd, f"clin_d_t_{target}")
            # 评估 (tcd/td)
            views = {"clin_clean": cdf, "d": pdfs["d"], "t": pdfs["t"], "clin_d_t": fd}
            eval_tcd, eval_td = {}, {}
            for vname, df in views.items():
                if df is None:
                    continue
                r_tcd = evaluate_view(df, tcd2, vname)
                r_td = evaluate_view(df, td2, vname)
                eval_tcd[vname] = r_tcd
                eval_td[vname] = r_td
                print(f"  {vname}: tcd {r_tcd} | td {r_td}")
            res2[target] = {"direct_tcd": eval_tcd, "direct_td": eval_td}
            # 2.5D 增益: label_T 和 label_N (对照 2D 单层)
            for tgt in ("label_T", "label_N"):
                if not os.path.exists(args.root25d):
                    continue
                tcd25, td25 = eval_subset_ids(md, args.root25d, args.backbone, ("t", "d"))
                g25, g2d = {}, {}
                for seq in ("d", "t"):
                    df = plane_probe(md, args.root25d, args.backbone, seq, device, tgt, args.seed)
                    if df is None:
                        continue
                    save_oof(df, f"25d_{seq}_{tgt}")
                    g25[seq] = {"tcd": evaluate_view(df, tcd25, f"25d_{seq}"),
                                "td": evaluate_view(df, td25, f"25d_{seq}")}
                    print(f"  [25d {tgt}] {seq}: {g25[seq]}")
                    # 2D 对照
                    df2 = plane_probe(md, args.root2d, args.backbone, seq, device, tgt, args.seed)
                    if df2 is not None:
                        save_oof(df2, f"2d_{seq}_{tgt}")
                        g2d[seq] = {"tcd": evaluate_view(df2, tcd25, f"2d_{seq}"),
                                    "td": evaluate_view(df2, td25, f"2d_{seq}")}
                res2.setdefault("_25d_gain", {})[tgt] = {"25d": g25, "2d_baseline": g2d}
        summary["tasks"]["task2"] = res2
        summary["tasks"]["task2_time_s"] = round(time.time() - t0, 1)

    if args.task3 or args.both or not (args.task2 or args.task3):
        t0 = time.time()
        print(f"\n===== 任务3: label_T GatedAttentionMIL (std, n=162 tcd) =====")
        res3 = {}
        for sp in (False, True):
            key = "gated_sp" if sp else "gated"
            pdfs = {}
            for seq in ("d", "t"):
                df, curves, pt = tsm.oof_for_plane(
                    md, args.root2d, args.backbone, seq, "gated", 32, 0.25, 0.6, sp,
                    15, 3e-4, 0.05, args.seed, device, "exp_4060_mil",
                    target="label_T", feat_norm="std", bin_w=4, max_batch_size=16)
                if df is None:
                    continue
                df = df.rename(columns={"label_T": "label"})
                save_oof(df, f"{key}_{seq}")
                pdfs[seq] = df
                r = evaluate_view(df, tcd2, f"{key}_{seq}")
                print(f"  [{key} {seq}] tcd {r}")
            # clin + d + t 融合 (clin 用 label_T OOF)
            cdf_t = clin_oof(md, "label_T")
            fd = fuse_clin_planes(cdf_t, pdfs, md, "probmean")
            save_oof(fd, f"{key}_clin_d_t")
            r = evaluate_view(fd, tcd2, f"{key}_clin_d_t")
            print(f"  [{key} clin+d+t] tcd {r}")
            res3[key] = {"d": evaluate_view(pdfs.get("d"), tcd2, "d") if "d" in pdfs else None,
                         "t": evaluate_view(pdfs.get("t"), tcd2, "t") if "t" in pdfs else None,
                         "clin_d_t": r}
        summary["tasks"]["task3"] = res3
        summary["tasks"]["task3_time_s"] = round(time.time() - t0, 1)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # md
    lines = ["# exp_4060_overnight 夜间实验汇总 (2026-11-23)", "",
             f"训练集: direct_all n=280 (master fold 5 折) | 评估: direct_tcd n={len(tcd2)} / direct_td n={len(td2)}",
             f"backbone: {args.backbone} | seed: {args.seed}", ""]
    for tname, tbody in summary["tasks"].items():
        lines.append(f"## {tname}")
        lines.append("```json")
        lines.append(json.dumps(tbody, ensure_ascii=False, indent=1))
        lines.append("```")
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nDONE → {SUMMARY_JSON} + {SUMMARY_MD}")


if __name__ == "__main__":
    main()
