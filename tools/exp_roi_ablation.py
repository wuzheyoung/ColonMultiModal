# -*- coding: utf-8 -*-
"""exp_roi_ablation.py — ROI 消融对比 (m07302 指令): 三套特征目录 × 4 组合。

口径锁定:
- 训练集 direct_all (n=280, data/direct_cohort_ids.csv is_tgr==0), 5 折对齐 master fold;
- 评估主表 direct_td (n=214) 附报 direct_tcd (n=162) —— 影像路径存在性子集, 单独分表;
- d 序列统一限定 d_self (strictly self mask, 排除 from_t): 三套特征 d 目录的公共 pid
  集合 (manifest mask_source=='self' & has_valid==1 ∩ 三套 d npz 均存在), 确保每 fold
  有效样本与缺失分布 100% 一致;
- 三套特征目录:
  baseline_120mm_old = features/slice_feats_v2/dinov2
  clean_120mm       = features/slice_feats_v2_120mm_clean/dinov2
  tight60_m10       = features/slice_feats_v2_tight60_m10/dinov2
- 线性探针 probe_dinov2_std (MeanMax / AreaWeightedMean + 训练折 std + 嵌套 3 折选 C
  L2-LR C∈{1e-4..1.0}), 与 tsm.linear_probe_plane / exp_4060.clin_oof 同口径;
- 4 组合: t_only=t(AreaWeightedMean) | d_self_only=d(MeanMax) |
  img_td=d(MeanMax)+t(AWM) | full_multimodal=clin+d(MeanMax)+t(AWM);
  融合用 probmean (可用模态等权);
- 指标: Pooled OOF AUC + 2000 次分层 Bootstrap 95% CI + Fold-Mean±SD (ddof=1)
  + ECE + Brier; 配对 DeLong (img_td/full: clean vs baseline, tight vs clean, tight vs clin)。
产物: logs/exp_roi_ablation_metrics.json + logs/exp_roi_ablation_report.md
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_slice_mil as tsm

MASTER = "data/master_5fold_356.csv"
DIRECT = "data/direct_cohort_ids.csv"
RAW = "data/hos_pairs_raw.csv"
MANIFEST = "features/dwi_mask_remap_manifest.csv"
CLIN_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
PROBE_C_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
BACKBONE = "dinov2"
REFS = {
    "baseline_120mm_old": "features/slice_feats_v2/dinov2",
    "clean_120mm": "features/slice_feats_v2_120mm_clean/dinov2",
    "tight60_m10": "features/slice_feats_v2_tight60_m10/dinov2",
}
JSON_OUT = "logs/exp_roi_ablation_metrics.json"
MD_OUT = "logs/exp_roi_ablation_report.md"


def load_subsets():
    master = pd.read_csv(MASTER)
    dc = pd.read_csv(DIRECT)
    direct_ids = set(dc[dc["is_tgr"] == 0]["pid"].astype(int))
    raw = pd.read_csv(RAW).set_index("patient_id")
    m = master.set_index("patient_id")
    md = master[master["patient_id"].isin(direct_ids)].copy()
    tcd, td = [], []
    for p in sorted(direct_ids):
        if p not in m.index or p not in raw.index:
            continue
        def ok(col):
            v = raw.loc[p, col]
            return isinstance(v, str) and os.path.exists(v)
        if ok("mri_path_t") and ok("mri_path_c") and ok("mri_path_d"):
            tcd.append(p)
        if ok("mri_path_t") and ok("mri_path_d"):
            td.append(p)
    return master, md, np.array(sorted(tcd)), np.array(sorted(td))


def d_self_pids_common():
    """三套特征 d 目录的公共 self pid (manifest self & has_valid=1 & 三套 npz 均在)。"""
    man = pd.read_csv(MANIFEST)
    self_pids = set(man[(man["mask_source"] == "self") & (man["has_valid_mask"] == 1)]
                    ["pid"].astype(int))
    for ref in REFS.values():
        ddir = os.path.join(ref, "d")
        on_disk = {int(fn[:-4]) for fn in os.listdir(ddir) if fn.endswith(".npz")}
        self_pids &= on_disk
    return np.array(sorted(self_pids), dtype=np.int64)


def t_pids_common():
    """三套 t 目录公共 pid (应全 280)。"""
    out = None
    for ref in REFS.values():
        tdir = os.path.join(ref, "t")
        cur = {int(fn[:-4]) for fn in os.listdir(tdir) if fn.endswith(".npz")}
        out = cur if out is None else (out & cur)
    return np.array(sorted(out), dtype=np.int64)


def load_bags(pid_list, seq, ref, label_map, device):
    """返回 {pid: (feats(K,D)fp32, rel_z, area, label)} — 只取 pid_list。"""
    d = os.path.join(ref, seq)
    bags = {}
    for pid in pid_list:
        p = os.path.join(d, f"{int(pid)}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        bags[int(pid)] = (
            torch.from_numpy(z["feats"].astype(np.float32)).to(device),
            torch.from_numpy(z["rel_z"].astype(np.float32)).to(device),
            torch.from_numpy(z["mask_area_ratio"].astype(np.float32)).to(device),
            float(label_map[int(pid)]),
        )
    return bags


def pool_areaweighted(bags, pids):
    """AreaWeightedMean: sum(f_i * w_i)/sum(w_i), w=mask_area_ratio (物理面积成正比)。"""
    Xs, ys = [], []
    for p in pids:
        f, _, w, y = bags[p]
        wsum = w.sum().clamp(min=1e-9)
        Xs.append(((f * w.unsqueeze(-1)).sum(dim=0) / wsum).float().cpu().numpy())
        ys.append(y)
    return np.array(Xs), np.array(ys)


def pool_meanmax(bags, pids):
    """MeanMax: concat(mean, max) → (2D,)。"""
    return tsm.meanmax_pool_bags(bags, list(pids))


def probe_lr(X, y, folds, c_grid=PROBE_C_GRID):
    """折内嵌套 3 折选 C + 训练折 std 的 L2-LR, 返回 OOF prob (len(y),)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    oof = np.full(len(y), np.nan)
    folds = np.asarray(folds)
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(set(y[tr])) < 2:
            oof[va] = 0.5
            continue
        inner = folds[tr] % 3
        inner_auc = {}
        for c in c_grid:
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(set(y[tr][itr])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr][itr])
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(
                    sc.transform(X[tr][itr]), y[tr][itr])
                aucs.append(roc_auc_score(y[tr][iva], mm.predict_proba(sc.transform(X[tr][iva]))[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(c_grid, key=lambda c: inner_auc[c])
        sc = StandardScaler().fit(X[tr])
        mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(X[tr]), y[tr])
        oof[va] = mm.predict_proba(sc.transform(X[va]))[:, 1]
    return oof


def clin_oof(md, target="label_T"):
    """clin_clean 7 维 L2-LR 折内 median impute + std + 嵌套选 C (同 exp_4060.clin_oof)。"""
    raw = pd.read_csv(RAW).set_index("patient_id")
    X = raw[CLIN_COLS].replace([np.inf, -np.inf], np.nan).astype(float)
    rows = []
    for _, r in md.iterrows():
        pid = int(r["patient_id"])
        y = int(r[target])
        x = X.loc[pid].values if pid in X.index else np.full(len(CLIN_COLS), np.nan)
        rows.append({"patient_id": pid, "fold": int(r["fold"]), "label": y, "x": x})
    df = pd.DataFrame(rows)
    y = df["label"].values
    folds = df["fold"].values
    Xm = np.stack(df["x"].values)
    oof = np.full(len(df), np.nan)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(set(y[tr])) < 2:
            oof[va] = 0.5
            continue
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


# ---------------- 指标 ----------------
def pooled_auc(y, p):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(p)
    if ok.sum() < 10 or len(set(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], p[ok]))


def fold_mean_sd(y, p, folds):
    aucs = []
    for k in sorted(set(folds)):
        m = folds == k
        a = pooled_auc(y[m], p[m])
        if a is not None:
            aucs.append(a)
    aucs = np.array(aucs)
    if len(aucs) < 2:
        return None, None
    return float(aucs.mean()), float(aucs.std(ddof=1))


def stratified_boot_ci(y, p, n_boot=2000, seed=0):
    """2000 次分层 Bootstrap (按 label 比例重采样患者) pooled AUC 95% CI。"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    ok = ~np.isnan(p)
    yy, pp = y[ok], p[ok]
    if len(yy) < 10 or len(set(yy)) < 2:
        return None, None
    pos = np.where(yy == 1)[0]; neg = np.where(yy == 0)[0]
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        ip = rng.integers(0, len(pos), len(pos))
        in_ = rng.integers(0, len(neg), len(neg))
        idx = np.concatenate([pos[ip], neg[in_]])
        if len(set(yy[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(yy[idx], pp[idx])
    a = aucs[~np.isnan(aucs)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def ece_brier(y, p, n_bins=10):
    """ECE (等宽 10 箱) + Brier score。"""
    ok = ~np.isnan(p)
    y = np.asarray(y)[ok]; p = np.asarray(p)[ok]
    if len(y) == 0:
        return None, None
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        ece += (m.sum() / len(y)) * abs(acc - conf)
    return float(ece), brier


def delong_paired(y, pA, pB):
    """配对 DeLong 检验 (两模型同患者集): 返回 (aucA, aucB, z, p)。"""
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(pA) & ~np.isnan(pB)
    y = np.asarray(y)[ok].astype(int)
    a = np.asarray(pA)[ok]; b = np.asarray(pB)[ok]
    if len(y) < 10 or len(set(y)) < 2 or len(np.unique(y)) < 2:
        return None, None, None, None
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n1, n0 = len(pos), len(neg)
    def comp(x):
        M = np.zeros((n1, n0), dtype=np.float64)
        for i in range(n1):
            M[i] = (x[pos[i]] > x[neg]).astype(np.float64) + 0.5 * (x[pos[i]] == x[neg])
        return M
    MA, MB = comp(a), comp(b)
    aucA = float(roc_auc_score(y, a)); aucB = float(roc_auc_score(y, b))
    v10A = MA.mean(axis=1); v01A = MA.mean(axis=0)
    v10B = MB.mean(axis=1); v01B = MB.mean(axis=0)
    varA = v10A.var(ddof=1) / n1 + v01A.var(ddof=1) / n0
    varB = v10B.var(ddof=1) / n1 + v01B.var(ddof=1) / n0
    cov = np.cov(v10A, v10B, ddof=1)[0, 1] / n1 + np.cov(v01A, v01B, ddof=1)[0, 1] / n0
    var_diff = varA + varB - 2 * cov
    if var_diff <= 0:
        return aucA, aucB, None, None
    z = (aucA - aucB) / np.sqrt(var_diff)
    p = float(2 * (1 - norm.cdf(abs(z))))
    return aucA, aucB, float(z), p


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)
    master, md, tcd, td = load_subsets()
    m = master.set_index("patient_id")
    direct_ids = set(md["patient_id"].astype(int))
    print(f"direct_td={len(td)} direct_tcd={len(tcd)}")
    tP = np.array(sorted(set(t_pids_common()) & direct_ids))
    dP = np.array(sorted(set(d_self_pids_common()) & direct_ids))
    print(f"t_common∩direct={len(tP)} d_self_common∩direct={len(dP)}")

    # 三套特征每组合的 OOF 字典: {(ref, combo): df(patient_id,fold,label,prob)}
    oofs = {}
    label_map = {int(r["patient_id"]): int(r["label_T"]) for _, r in md.iterrows()}
    fold_map = {int(r["patient_id"]): int(r["fold"]) for _, r in md.iterrows()}

    for ref, root in REFS.items():
        t_bags = load_bags(tP, "t", root, label_map, device)
        d_bags = load_bags(dP, "d", root, label_map, device)
        # t_only
        Xt, yt = pool_areaweighted(t_bags, tP)
        pf = np.array([fold_map[p] for p in tP])
        ot = probe_lr(Xt, yt, pf)
        oofs[(ref, "t_only")] = pd.DataFrame(
            {"patient_id": tP, "fold": pf, "label": yt, "prob": ot})
        # d_self_only
        Xd, yd = pool_meanmax(d_bags, dP)
        pfd = np.array([fold_map[p] for p in dP])
        od = probe_lr(Xd, yd, pfd)
        oofs[(ref, "d_self_only")] = pd.DataFrame(
            {"patient_id": dP, "fold": pfd, "label": yd, "prob": od})
        # img_td: d(MeanMax) + t(AWM) concat → 单 L2-LR
        common = np.array(sorted(set(tP) & set(dP)))
        Xc = np.concatenate([pool_areaweighted(t_bags, common)[0],
                             pool_meanmax(d_bags, common)[0]], axis=1)
        yc = np.array([label_map[p] for p in common])
        pfc = np.array([fold_map[p] for p in common])
        oc = probe_lr(Xc, yc, pfc)
        oofs[(ref, "img_td")] = pd.DataFrame(
            {"patient_id": common, "fold": pfc, "label": yc, "prob": oc})
        print(f"[{ref}] t_only/d_self_only/img_td done "
              f"(n_t={len(tP)} n_d={len(dP)} n_td={len(common)})")
    # full_multimodal: clin + d(MeanMax) + t(AWM)
    clin_df = clin_oof(md, target="label_T")
    clin_map = {int(r["patient_id"]): float(r["prob"]) for _, r in clin_df.iterrows()}
    for ref in REFS:
        # 复用 img_td 的 common 患者, 融合 = probmean(clin, d, t)
        base = oofs[(ref, "img_td")]
        ps = base["patient_id"].values
        rows = []
        for p in ps:
            rows.append({"patient_id": p, "fold": int(fold_map[p]), "label": int(label_map[p]),
                         "prob": (float(clin_map.get(int(p), 0.5)) + float(base.loc[base["patient_id"] == p, "prob"].iloc[0])) / 2.0})
        oofs[(ref, "full_multimodal")] = pd.DataFrame(rows)
    print("full_multimodal done (probmean clin+d+t)")

    # 评估
    combos = ["t_only", "d_self_only", "img_td", "full_multimodal"]
    subsets = {"td": td, "tcd": tcd}
    res = {"n_subsets": {k: int(len(v)) for k, v in subsets.items()},
           "d_self_common_direct": int(len(dP)), "t_common_direct": int(len(tP)),
           "evaluation": {}}
    for combo in combos:
        res["evaluation"][combo] = {}
        for sname, ids in subsets.items():
            row = {}
            for ref in REFS:
                df = oofs[(ref, combo)]
                sub = df[df["patient_id"].isin(ids)]
                if len(sub) < 10:
                    row[ref] = None
                    continue
                y = sub["label"].values.astype(int)
                p = sub["prob"].values.astype(float)
                f = sub["fold"].values
                auc = pooled_auc(y, p)
                lo, hi = stratified_boot_ci(y, p, n_boot=2000, seed=args.seed)
                fm, fsd = fold_mean_sd(y, p, f)
                ece, br = ece_brier(y, p)
                row[ref] = {"n": int(len(sub)), "n_pos": int(y.sum()),
                            "pooled_auc": auc, "ci95": [lo, hi],
                            "fold_mean": fm, "fold_sd": fsd,
                            "ece": ece, "brier": br}
            res["evaluation"][combo][sname] = row
    # DeLong 配对 (img_td / full_multimodal, td 主表)
    delong = {}
    for combo in ["img_td", "full_multimodal"]:
        pairs = [("clean_120mm", "baseline_120mm_old"),
                 ("tight60_m10", "clean_120mm"),
                 ("tight60_m10", "clin")]
        row = {}
        for ra, rb in pairs:
            dfA = oofs[(ra, combo)]
            if rb == "clin":
                # tight vs clin: 用 tight img_td/full 的 prob 与 clin_map 同患者
                ids = dfA["patient_id"].values
                pA = dfA["prob"].values
                pB = np.array([clin_map.get(int(p), 0.5) for p in ids])
                y = dfA["label"].values
            else:
                dfB = oofs[(rb, combo)]
                ids = np.array(sorted(set(dfA["patient_id"].values) & set(dfB["patient_id"].values)))
                iA = {int(p): i for i, p in enumerate(dfA["patient_id"].values)}
                iB = {int(p): i for i, p in enumerate(dfB["patient_id"].values)}
                pA = np.array([dfA["prob"].values[iA[p]] for p in ids])
                pB = np.array([dfB["prob"].values[iB[p]] for p in ids])
                y = np.array([dfA.loc[dfA["patient_id"] == p, "label"].iloc[0] for p in ids])
            aucA, aucB, z, pval = delong_paired(y, pA, pB)
            row[f"{ra}_vs_{rb}"] = {"n": int(len(ids)), "aucA": aucA, "aucB": aucB,
                                    "z": z, "p": pval}
        delong[combo] = row
    res["delong_paired"] = delong

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    # ---- markdown ----
    L = []
    L.append("# ROI 消融对比 (m07302)")
    L.append("")
    L.append(f"- direct_td n={len(td)} / direct_tcd n={len(tcd)}; d 限定 d_self∩direct 公共 "
             f"{len(dP)} 例; t∩direct 公共 {len(tP)} 例")
    L.append("- 线性探针 probe_dinov2_std (MeanMax/AWM + 训练折 std + 嵌套 3 折选 C L2-LR)")
    L.append("")
    for combo in combos:
        L.append(f"## {combo}")
        L.append("")
        for sname in ("td", "tcd"):
            L.append(f"### {sname} (n={len(subsets[sname])})")
            L.append("")
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
            L.append("")
    L.append("## DeLong 配对检验 (td 主表)")
    L.append("")
    for combo in ["img_td", "full_multimodal"]:
        L.append(f"### {combo}")
        L.append("")
        L.append("| 对比 | n | AUC_A | AUC_B | z | p |")
        L.append("|---|---|---|---|---|---|")
        for k, v in delong[combo].items():
            pv = "–" if v["p"] is None else f"{v['p']:.4f}"
            L.append(f"| {k} | {v['n']} | {v['aucA']:.4f} | {v['aucB']:.4f} | "
                     f"{v['z'] if v['z'] is not None else '–'} | {pv} |")
        L.append("")
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("DONE ->", JSON_OUT, "|", MD_OUT)


if __name__ == "__main__":
    main()
