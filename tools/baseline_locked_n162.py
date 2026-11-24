# -*- coding: utf-8 -*-
"""baseline_locked_n162.py — 锁死旧特征 n=162 严格同口径基线主表 (任务 m00457-1)。

口径 (用户锁定, 严禁违背):
  - 唯一队列: data/direct_cohort_ids.csv  (direct_all=280 / ncrt=75);
  - data/master_5fold_356.csv 锁死 (355 例, fold 列不动);
  - 临床: data/hos_pairs_raw.csv (pid 372/395/402 共 7 处勘误已落盘, 直读不缓存);
  - 方法 = audit 切片法: 每路单路模型在"各自全部可用直接手术样本"上做 5 折交叉验证
    (clin_clean 折内 SimpleImputer(median); 影像分支无缺失 → nan_to_num 等价路径),
    评估阶段统一截取完全相同的 direct_tcd n=162 交集, 严禁任何跨折信息泄露;
  - 融合: audit fusion_variants 原版 (f1 = 折内训练 logit z-score 校准均值,
    f2 = rank 均值), 单路概率同样来自"全部直接样本训练 → n=162 切片"。

输出:
  - logs/baseline_locked_n162.json   完整结构化结果
  - logs/baseline_locked_n162.md     易读 Markdown 简表
  - logs/baseline_locked_n162.csv    主表 CSV
  - outputs/oof_baseline_n162/singles/{branch}.csv   逐样本 OOF (patient_id,fold,label_T,prob)
  - outputs/oof_baseline_n162/fusions/{step}.{f1,f2}.csv

用法 (Windows 安全, 无多进程):
  python tools/baseline_locked_n162.py
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 复用 recompute_direct_tcd_singles.py 的逐字一致实现 (per_fold_lr / 特征加载 / AUC)
spec = importlib.util.spec_from_file_location(
    "rc", os.path.join(ROOT, "tools", "recompute_direct_tcd_singles.py"))
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

COHORT_CSV = "data/direct_cohort_ids.csv"
OUT_JSON = "logs/baseline_locked_n162.json"
OUT_MD = "logs/baseline_locked_n162.md"
OUT_CSV = "logs/baseline_locked_n162.csv"
OOF_DIR = "outputs/oof_baseline_n162"

SINGLES = ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"]
STEPS = {
    "s1_singles": ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"],
    "s2_clin_dconv": ["clin_clean", "d_conv"],
    "s3_img_3plane": ["d_conv", "t_conv", "c_conv"],
    "s4_clin_3plane": ["clin_clean", "d_conv", "t_conv", "c_conv"],
    "s5_six": ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"],
}


def logit_clip(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fold_mean_auc_ddof1(y, p, folds):
    """Fold-Mean ± SD, 标准差用 ddof=1 (与 tools/eval_paper_metrics.py 完全一致)。

    m00622: 原 rc.fold_mean_auc 用 fa.std() (ddof=0), 与任务4 的 ddof=1 对不上;
    此处局部重实现, 保证 logs/baseline_locked_n162.* 与 logs/eval_paper_metrics.json
    两端 fold_sd 小数完全一致。
    """
    per = []
    for k in np.unique(folds):
        va = folds == k
        if len(np.unique(y[va])) < 2 or va.sum() < 2:
            continue
        per.append(roc_auc_score(y[va], p[va]))
    if not per:
        return None, None
    return round(float(np.mean(per)), 4), round(float(np.std(per, ddof=1)), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", default=OUT_JSON)
    args = ap.parse_args()

    master = pd.read_csv(rc.MASTER_CSV)
    raw = pd.read_csv(rc.PAIRS_RAW).set_index("patient_id")
    coh = pd.read_csv(COHORT_CSV)
    tgr = set(coh.loc[coh["is_tgr"] == 1, "pid"].astype(int))
    direct = master[~master["patient_id"].isin(tgr)].copy()
    d_ids = np.sort(direct["patient_id"].values.astype(np.int64))
    has_p = {s: set(raw.index[raw[f"mri_path_{s}"].notna()]) for s in ("t", "c", "d")}
    tcd = np.array(sorted(set(d_ids) & has_p["t"] & has_p["c"] & has_p["d"]))
    print(f"direct_all={len(d_ids)} ncrt={len(tgr)} direct_td="
          f"{len(set(d_ids) & has_p['t'] & has_p['d'])} direct_tcd={len(tcd)}")

    # ---------------- 特征 (与 audit/recompute 完全一致) ----------------
    rad_t = rc.rad_feats(rc.R3A_CSV, "body")
    rad_d = rc.rad_feats(rc.R3B["d"], "body")
    rad_c = rc.rad_feats(rc.R3B["c"], "body")
    conv_t = rc.conv_feats_dir(rc.R2T_DIR)
    conv_d, conv_c = rc.conv_feats_npy()
    clin_raw = raw[rc.CLIN_COLS].replace([np.inf, -np.inf], np.nan).astype(float)
    branches = {"t_rad": rad_t, "t_conv": conv_t, "d_rad": rad_d, "d_conv": conv_d,
                "c_rad": rad_c, "c_conv": conv_c, "clin_clean": clin_raw}
    m = master.set_index("patient_id")

    # ---------------- 任务1: 临床基线核查 (audit 同款, 直接队列) ----------------
    Xc = clin_raw.loc[d_ids].values.astype(float)
    yT = m.loc[d_ids, "label_T"].values.astype(int)
    fT = m.loc[d_ids, "fold"].values.astype(int)
    p_old, _ = rc.per_fold_lr(Xc, yT, fT, imp=None)
    p_clean, _ = rc.per_fold_lr(Xc, yT, fT, imp="median")
    fm_o, sd_o = fold_mean_auc_ddof1(yT, p_old, fT)
    fm_c, sd_c = fold_mean_auc_ddof1(yT, p_clean, fT)
    task1 = {"direct_n": int(len(d_ids)),
             "nan_to_num": {"pooled": rc.pooled_auc(yT, p_old), "fold_mean": fm_o, "fold_sd": sd_o},
             "imputer_median": {"pooled": rc.pooled_auc(yT, p_clean), "fold_mean": fm_c, "fold_sd": sd_c}}

    # ---------------- 单路: 全部直接样本训练 → n=162 切片 ----------------
    def branch_meta(name):
        if name == "clin_clean":
            return np.array(sorted(set(d_ids) & set(clin_raw.index))), clin_raw, "median"
        fdf = branches[name]
        return np.array(sorted(set(d_ids) & set(fdf.index))), fdf, None

    singles = {}
    oof_map = {}   # name -> (bid_all_sorted, p_all)
    for nm in SINGLES:
        ids_all, feats, imp = branch_meta(nm)
        if len(ids_all) < 8:
            print(f"  {nm}: n_train={len(ids_all)} <8, skip"); continue
        X = feats.loc[ids_all].values.astype(float)
        yl = m.loc[ids_all, "label_T"].values.astype(int)
        fl = m.loc[ids_all, "fold"].values.astype(int)
        p_all, det = rc.per_fold_lr(X, yl, fl, imp=imp)
        oof_map[nm] = (ids_all, p_all)
        # 切片到 n=162 交集
        mask = np.isin(ids_all, tcd)
        ids_e = ids_all[mask]
        p_e = p_all[mask]
        if len(ids_e) < 8:
            continue
        y_e = m.loc[ids_e, "label_T"].values.astype(int)
        f_e = m.loc[ids_e, "fold"].values.astype(int)
        fm, sd = fold_mean_auc_ddof1(y_e, p_e, f_e)
        singles[nm] = {"pooled": rc.pooled_auc(y_e, p_e),
                       "fold_mean": fm, "fold_sd": sd,
                       "n_train": int(len(ids_all)), "n_eval": int(len(ids_e)),
                       "T+_eval": int(y_e.sum())}
        print(f"  {nm:10s} train n={len(ids_all):3d} → eval n={len(ids_e):3d} "
              f"pooled={singles[nm]['pooled']} fold={fm}±{sd}")

    # ---------------- 融合 (audit fusion_variants 原版, 概率来自全直接训练 OOF) ----------------
    tcd_idx = {p: i for i, p in enumerate(tcd)}

    def fusion(step_names):
        """f1 z-logit 折内校准均值 / f2 rank 均值; 单路概率 = oof_map 中全直接训练的切片。"""
        n = len(tcd)
        yg = m.loc[tcd, "label_T"].values.astype(int)
        single = {}
        for nm in step_names:
            ids_all, p_all = oof_map[nm]
            bid = np.array([p for p in tcd if p in set(ids_all.tolist())])
            if len(bid) < 8:
                continue
            p = p_all[np.searchsorted(ids_all, bid)]
            single[nm] = (bid, p)
        F1 = np.full(n, np.nan); C1 = np.zeros(n)
        for nm, (bid, p) in single.items():
            floc = m.loc[bid, "fold"].values.astype(int)
            yloc = m.loc[bid, "label_T"].values.astype(int)
            Lva = logit_clip(p)
            cal = np.full(len(bid), np.nan)
            for k in np.unique(floc):
                va = floc == k
                tr = ~va
                if tr.sum() < 2 or len(np.unique(yloc[tr])) < 2:
                    cal[va] = Lva[va]
                    continue
                if nm == "clin_clean":
                    Xb = SimpleImputer(strategy="median").fit_transform(
                        clin_raw.loc[bid].values.astype(float))
                    ip = SimpleImputer(strategy="median").fit(Xb[tr])
                    Xb_tr = ip.transform(Xb[tr]); Xb_va = ip.transform(Xb[va])
                else:
                    Xb = branches[nm].loc[bid].values.astype(float)
                    Xb_tr, Xb_va = Xb[tr], Xb[va]
                sc = StandardScaler().fit(Xb_tr)
                mdl = LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs").fit(
                    sc.transform(Xb_tr), yloc[tr])
                Ptr = mdl.predict_proba(sc.transform(Xb_tr))[:, 1]
                Ltr = logit_clip(Ptr)
                m_tr, s_tr = Ltr.mean(), Ltr.std() + 1e-9
                cal[va] = (Lva[va] - m_tr) / s_tr
            idx = np.searchsorted(tcd, bid)
            for ii, i in enumerate(idx):
                if not np.isnan(cal[ii]):
                    F1[i] = cal[ii] if np.isnan(F1[i]) else F1[i] + cal[ii]
                    C1[i] += 1
        m1 = C1 > 0
        F1[m1] /= C1[m1]
        F2 = np.full(n, np.nan); C2 = np.zeros(n)
        for nm, (bid, p) in single.items():
            r = rankdata(p) / len(p)
            idx = np.searchsorted(tcd, bid)
            for ii, i in enumerate(idx):
                if not np.isnan(p[ii]):
                    F2[i] = r[ii] if np.isnan(F2[i]) else F2[i] + r[ii]
                    C2[i] += 1
        m2 = C2 > 0
        F2[m2] /= C2[m2]

        def auc_ignoring_nan(yv, pv, fv):
            ok = ~np.isnan(pv)
            if ok.sum() < 10 or len(np.unique(yv[ok])) < 2:
                return None, None, None
            return (rc.pooled_auc(yv[ok], pv[ok]),
                    *fold_mean_auc_ddof1(yv[ok], pv[ok], fv[ok]))

        fv = m.loc[tcd, "fold"].values.astype(int)
        a, fm, sd = auc_ignoring_nan(yg, F1, fv)
        r1 = {"pooled": a, "fold_mean": fm, "fold_sd": sd,
              "n_valid": int((~np.isnan(F1)).sum())}
        a, fm, sd = auc_ignoring_nan(yg, F2, fv)
        r2 = {"pooled": a, "fold_mean": fm, "fold_sd": sd,
              "n_valid": int((~np.isnan(F2)).sum())}
        return r1, r2, F1, F2

    fusions = {}
    for step, names in STEPS.items():
        r1, r2, F1, F2 = fusion(names)
        fusions[step] = {"f1": r1, "f2": r2}
        print(f"  {step:15s} f1={r1['pooled']} (n={r1['n_valid']})  f2={r2['pooled']} (n={r2['n_valid']})")

    # ---------------- 落盘 ----------------
    meta = {
        "cohort_source": COHORT_CSV,
        "direct_all": int(len(d_ids)), "ncrt": int(len(tgr)),
        "direct_td": int(len(set(d_ids) & has_p["t"] & has_p["d"])),
        "direct_tcd": int(len(tcd)),
        "master_rows": int(len(master)),
        "fold_counts": {int(k): int(v) for k, v in zip(*np.unique(m["fold"].values, return_counts=True))},
        "clin_source": "data/hos_pairs_raw.csv (372/395/402 erratum applied, read directly)",
        "feature_root": "features/logs",
        "method": "audit-slice: each single trained on ALL direct patients with that feature "
                  "(5-fold, fold-internal median impute for clin_clean / nan_to_num for images), "
                  "evaluated on the exact direct_tcd n=162 intersection; "
                  "fusions f1=fold-internal train-logit zscore-calibrated mean, f2=rank mean (audit verbatim).",
    }
    eval_set = {"n": int(len(tcd)),
                "T+": int(m.loc[tcd, "label_T"].sum()),
                "fold_counts": {int(k): int(v) for k, v in
                                zip(*np.unique(m.loc[tcd, "fold"].values, return_counts=True))}}
    res = {"meta": meta, "eval_set": eval_set, "task1_clin": task1,
           "singles": singles, "fusions": fusions}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)

    # 逐样本 OOF CSV
    os.makedirs(os.path.join(OOF_DIR, "singles"), exist_ok=True)
    os.makedirs(os.path.join(OOF_DIR, "fusions"), exist_ok=True)
    for nm in SINGLES:
        if nm not in oof_map:
            continue
        ids_all, p_all = oof_map[nm]
        mask = np.isin(ids_all, tcd)
        ids_e, p_e = ids_all[mask], p_all[mask]
        df = pd.DataFrame({"patient_id": ids_e,
                           "fold": m.loc[ids_e, "fold"].values,
                           "label_T": m.loc[ids_e, "label_T"].values,
                           "prob": p_e})
        df.to_csv(os.path.join(OOF_DIR, "singles", f"{nm}.csv"), index=False)
    for step in STEPS:
        for tag in ("f1", "f2"):
            df = pd.DataFrame({"patient_id": tcd,
                               "fold": m.loc[tcd, "fold"].values,
                               "label_T": m.loc[tcd, "label_T"].values,
                               "prob": np.nan})
            # 重新计算该 step 的融合分数 (与上面同一函数)
            r1, r2, F1, F2 = fusion(STEPS[step])
            df["prob"] = F1 if tag == "f1" else F2
            df.to_csv(os.path.join(OOF_DIR, "fusions", f"{step}.{tag}.csv"), index=False)

    # 易读表
    rows = []
    for nm in SINGLES:
        if nm not in singles:
            continue
        e = singles[nm]
        rows.append({"branch": nm, "n_train": e["n_train"], "n_eval": e["n_eval"],
                     "T+_eval": e["T+_eval"], "pooled": e["pooled"],
                     "fold_mean": e["fold_mean"], "fold_sd": e["fold_sd"]})
    for step in STEPS:
        rows.append({"branch": f"fusion_{step}_f1", "n_train": "-", "n_eval": fusions[step]["f1"]["n_valid"],
                     "T+_eval": eval_set["T+"], "pooled": fusions[step]["f1"]["pooled"],
                     "fold_mean": fusions[step]["f1"]["fold_mean"], "fold_sd": fusions[step]["f1"]["fold_sd"]})
        rows.append({"branch": f"fusion_{step}_f2", "n_train": "-", "n_eval": fusions[step]["f2"]["n_valid"],
                     "T+_eval": eval_set["T+"], "pooled": fusions[step]["f2"]["pooled"],
                     "fold_mean": fusions[step]["f2"]["fold_mean"], "fold_sd": fusions[step]["f2"]["fold_sd"]})
    tdf = pd.DataFrame(rows)
    tdf.to_csv(OUT_CSV, index=False, encoding="utf-8")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(f"# baseline_locked_n162 (direct_tcd n={eval_set['n']}, T+={eval_set['T+']})\n\n")
        f.write(f"- direct_all={meta['direct_all']} / ncrt={meta['ncrt']} / direct_td={meta['direct_td']}\n")
        f.write(f"- 方法: audit 切片法 — 每路单路在全部直接样本上 5 折训练, 评估统一截取 n={eval_set['n']} 交集\n\n")
        f.write("| branch | n_train | n_eval | T+_eval | pooled | fold_mean | fold_sd |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['branch']} | {r['n_train']} | {r['n_eval']} | {r['T+_eval']} "
                    f"| {r['pooled']} | {r['fold_mean']} | {r['fold_sd']} |\n")
        f.write("\n")
    print(f"\nDONE → {args.out_json} / {OUT_MD} / {OUT_CSV} / {OOF_DIR}")


if __name__ == "__main__":
    main()
