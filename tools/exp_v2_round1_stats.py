# -*- coding: utf-8 -*-
"""exp_v2_round1_stats.py — m00964 Step4: 汇总 round1 全部结果 + DWI 救回实效审计 + 配对 DeLong。

读取 outputs/exp_v2_round1/ 下 run 脚本产出的 tcd 评估 CSV (eval_{s}/fusion_{mode}/fusion_clin_{mode}),
输出 logs/exp_v2_round1_T.json + logs/exp_v2_round1_T.md:
  1) 全模型矩阵表 (3 骨干 × 探针{l2,std} / MIL{gated±prior,topk} × 平面 + 融合);
  2) DWI 救回实效审计 (最优骨干 d 单路 + 最优 clin+3plane): mask_source self/remapped 拆解,
     旧 d_conv 覆盖 81/81 拆解 (对照旧 s4.f1 子集 0.7523/0.6448);
  3) 最优多模态 vs 旧锁定 s4_clin_3plane.f1 (0.7016) 配对 DeLong + diff 95% CI (患者级 bootstrap)。
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eval_paper_metrics as epm

OUT = "outputs/exp_v2_round1"
OLD_S4 = "outputs/oof_baseline_n162/fusions/s4_clin_3plane.f1.csv"
OLD_D_PIDS = "features/logs/route2_cd_feats/d_pids.csv"
MASTER = "data/master_5fold_356.csv"
COHORT = "data/direct_cohort_ids.csv"
PAIRS = "data/hos_pairs_raw.csv"
DATA_ROOT = "features/slice_feats_v2"
N_BOOT = 2000


def get_tcd():
    master = pd.read_csv(MASTER)
    cohort = pd.read_csv(COHORT)
    pid_col = "patient_id" if "patient_id" in cohort.columns else "pid"
    tgr = set(cohort.loc[cohort["is_tgr"] == 1, pid_col])
    md = master[~master["patient_id"].isin(tgr)]
    raw = pd.read_csv(PAIRS).set_index("patient_id")
    dp = md.set_index("patient_id")
    def has(c):
        return raw["mri_path_" + c].notna().reindex(dp.index).fillna(False)
    return set(dp.index[has("t") & has("c") & has("d")])


def load_eval(path):
    df = pd.read_csv(path)
    df = df[["patient_id", "fold", "label_T", "prob"]].dropna(subset=["prob"])
    return df


def auc_sub(df, sub):
    d = df[df["patient_id"].isin(sub)]
    if len(d) < 8 or len(set(d["label_T"])) < 2:
        return {"n": len(d), "pooled_auc": None}
    return {"n": len(d), "pooled_auc": round(float(epm.pooled_auc(d)), 4)}


def boot_diff_ci(df1, df2, n=N_BOOT):
    common = sorted(set(df1["patient_id"]) & set(df2["patient_id"]))
    a = df1.set_index("patient_id").loc[common]
    b = df2.set_index("patient_id").loc[common]
    y = a["label_T"].values.astype(int)
    pa, pb = a["prob"].values.astype(float), b["prob"].values.astype(float)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n):
        sel = rng.choice(len(common), size=len(common), replace=True)
        if len(np.unique(y[sel])) < 2:
            continue
        d1 = epm.roc_auc_score(y[sel], pa[sel])
        d2 = epm.roc_auc_score(y[sel], pb[sel])
        diffs.append(d1 - d2)
    if len(diffs) < 100:
        return None
    qs = np.percentile(diffs, [2.5, 97.5])
    return [round(float(qs[0]), 4), round(float(qs[1]), 4)]


def main():
    tcd = get_tcd()
    old_d = set(pd.read_csv(OLD_D_PIDS).iloc[:, 0].astype(int))
    old_covered = sorted(tcd & old_d)
    old_missing = sorted(tcd - old_d)
    print(f"tcd={len(tcd)} old_covered={len(old_covered)} old_missing={len(old_missing)}")

    old_s4 = load_eval(OLD_S4)
    assert set(old_s4["patient_id"]) == tcd, "old s4 CSV 患者集 ≠ tcd!"
    old_s4_pooled = round(float(epm.pooled_auc(old_s4)), 4)
    old_s4_cov = auc_sub(old_s4, old_covered)
    old_s4_miss = auc_sub(old_s4, old_missing)
    print(f"old s4_clin_3plane.f1: pooled={old_s4_pooled} "
          f"cov81={old_s4_cov['pooled_auc']} (n={old_s4_cov['n']}) "
          f"miss81={old_s4_miss['pooled_auc']} (n={old_s4_miss['n']})")

    # 收集全部模型 (probe/mil) 的平面 + 融合
    models = {}
    for tag in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, tag)
        if not os.path.isdir(d):
            continue
        kind = "probe" if tag.startswith("probe") else "mil"
        m = {"kind": kind, "planes": {}, "fusions": {}}
        for s in ("t", "c", "d"):
            p = os.path.join(d, f"eval_{s}.csv")
            if os.path.exists(p):
                m["planes"][s] = eval_metrics_full(load_eval(p))
        for mode in ("zlogit", "clin_zlogit", "probmean", "clin_probmean"):
            p = os.path.join(d, f"fusion_{mode}.csv")
            if os.path.exists(p):
                m["fusions"][mode] = eval_metrics_full(load_eval(p))
        models[tag] = m

    # 最优骨干 (d 单路 pooled AUC 最大)
    d_scores = {tag: m["planes"]["d"]["pooled_auc"] for tag, m in models.items()
                if "d" in m["planes"] and m["planes"]["d"]["pooled_auc"]}
    best_bk_tag = max(d_scores, key=d_scores.get)
    best_bk = best_bk_tag.split("_")[1]  # convnext_tiny/dinov2/biomedclip
    print(f"\nbest-d backbone: {best_bk_tag} (d pooled={d_scores[best_bk_tag]})")

    # 最优多模态 (clin_zlogit pooled 最大)
    cz_scores = {tag: m["fusions"]["clin_zlogit"]["pooled_auc"] for tag, m in models.items()
                 if "clin_zlogit" in m["fusions"] and m["fusions"]["clin_zlogit"]["pooled_auc"]}
    best_fus_tag = max(cz_scores, key=cz_scores.get)
    print(f"best clin+3plane zlogit: {best_fus_tag} (pooled={cz_scores[best_fus_tag]})")

    # ---- DWI 救回实效审计 (最优骨干) ----
    audit = {}
    d_eval = load_eval(os.path.join(OUT, best_bk_tag, "eval_d.csv"))
    # mask_source (从 npz 读)
    src = {}
    for pid in set(d_eval["patient_id"]):
        p = os.path.join(DATA_ROOT, best_bk, "d", f"{pid}.npz")
        if os.path.exists(p):
            with np.load(p) as z:
                v = str(z["mask_source"][()]).strip("[]' ")
                src[pid] = "self" if v == "self" else "remapped"
    self_ids = {p for p in src if src[p] == "self"}
    remap_ids = {p for p in src if src[p] == "remapped"}
    audit["mask_source"] = {
        "self": auc_sub(d_eval, self_ids), "remapped": auc_sub(d_eval, remap_ids)}
    audit["old_coverage"] = {
        "old_covered": auc_sub(d_eval, old_covered),
        "old_missing": auc_sub(d_eval, old_missing)}
    # 旧缺失 81 例中实际有 d 特征的 (救回) 与仍无 d 的
    new_d_ids = set(d_eval["patient_id"])
    audit["rescue_detail"] = {
        "old_missing_with_d": len(set(old_missing) & new_d_ids),
        "old_missing_without_d": len(set(old_missing) - new_d_ids),
        "old_covered_with_d": len(set(old_covered) & new_d_ids)}

    # 最优 clin+3plane 融合的同一拆解
    best_fus_eval = load_eval(os.path.join(OUT, best_fus_tag, "fusion_clin_zlogit.csv"))
    audit["mask_source_fus"] = {
        "self": auc_sub(best_fus_eval, self_ids), "remapped": auc_sub(best_fus_eval, remap_ids)}
    audit["old_coverage_fus"] = {
        "old_covered": auc_sub(best_fus_eval, old_covered),
        "old_missing": auc_sub(best_fus_eval, old_missing)}
    audit["old_s4_ref"] = {"pooled": old_s4_pooled,
                           "old_covered": old_s4_cov, "old_missing": old_s4_miss}
    print("audit:", json.dumps(audit, ensure_ascii=False))

    # ---- 配对 DeLong: 最优多模态 vs 旧 s4 ----
    best_fus = load_eval(os.path.join(OUT, best_fus_tag, "fusion_clin_zlogit.csv"))
    dl, n_paired = epm.delong_paired_test(best_fus, old_s4)
    diff_ci = boot_diff_ci(best_fus, old_s4)
    delong = {**dl, "n_paired": n_paired, "diff_95ci": diff_ci,
              "old_s4_pooled": old_s4_pooled}
    print("DeLong:", json.dumps(delong, ensure_ascii=False))

    # ---- 汇总 JSON + MD ----
    out = {"eval_set": {"direct_tcd_n": len(tcd), "old_s4_pooled": old_s4_pooled},
           "models": models, "best_backbone": {"tag": best_bk_tag, "name": best_bk,
                                               "d_pooled": d_scores[best_bk_tag]},
           "best_fusion": {"tag": best_fus_tag, "pooled": cz_scores[best_fus_tag]},
           "dwi_rescue_audit": audit, "delong_vs_old_s4": delong}
    with open("logs/exp_v2_round1_T.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # MD
    L = []
    L.append("# exp_v2_round1_T — 第一轮真实特征对比 (label_T, 评估 direct_tcd n=162)\n")
    L.append(f"旧锁定基线 s4_clin_3plane.f1 pooled = **{old_s4_pooled}** "
             f"(old_covered 81: {old_s4_cov['pooled_auc']} / old_missing 81: {old_s4_miss['pooled_auc']})\n")
    L.append("\n## 全模型矩阵 (Pooled OOF AUC; n / Fold-Mean ± SD)\n")
    hdr = "| 模型 | d (n/AUC) | t | c | 3p zlogit | clin+3p zlogit |"
    L.append(hdr); L.append("|---|---|---|---|---|---|")
    for tag in sorted(models):
        m = models[tag]
        def cell(x):
            if not x or not x.get("pooled_auc"):
                return "—"
            return f"{x['pooled_auc']} (n={x['n']}, {x['fold_mean']}±{x['fold_sd']})"
        L.append(f"| {tag} | {cell(m['planes'].get('d'))} | {cell(m['planes'].get('t'))} | "
                 f"{cell(m['planes'].get('c'))} | {cell(m['fusions'].get('zlogit'))} | "
                 f"{cell(m['fusions'].get('clin_zlogit'))} |")
    L.append("\n## DWI 救回实效审计 (最优骨干 " + best_bk_tag + ")\n")
    a = audit
    L.append("| 拆解 | d 单路 | clin+3p zlogit | 旧 s4.f1 |")
    L.append("|---|---|---|---|")
    L.append(f"| mask_source=self | {_auccell(a['mask_source']['self'])} | "
             f"{_auccell(a['mask_source_fus']['self'])} | — |")
    L.append(f"| mask_source=remapped | {_auccell(a['mask_source']['remapped'])} | "
             f"{_auccell(a['mask_source_fus']['remapped'])} | — |")
    L.append(f"| 旧 d_conv 覆盖 81 例 | {_auccell(a['old_coverage']['old_covered'])} | "
             f"{_auccell(a['old_coverage_fus']['old_covered'])} | {_auccell(a['old_s4_ref']['old_covered'])} |")
    L.append(f"| 旧 d_conv 缺失 81 例 | {_auccell(a['old_coverage']['old_missing'])} | "
             f"{_auccell(a['old_coverage_fus']['old_missing'])} | {_auccell(a['old_s4_ref']['old_missing'])} |")
    L.append(f"\n救回明细: 旧缺失 81 例中 {a['rescue_detail']['old_missing_with_d']} 例现获得 d 特征 "
             f"({a['rescue_detail']['old_missing_without_d']} 例仍无 d); 旧覆盖 81 例中 "
             f"{a['rescue_detail']['old_covered_with_d']} 例有 d 特征。\n")
    L.append("\n## 配对 DeLong — 最优多模态 vs 旧 s4_clin_3plane.f1\n")
    L.append(f"- 最优多模态: `{best_fus_tag}` clin+3plane zlogit pooled = {cz_scores[best_fus_tag]}")
    L.append(f"- AUC 差值 = {delong['diff']} (95% CI {delong['diff_95ci']}), DeLong z={delong['z']}, "
             f"**p={delong['p']}**, n_paired={delong['n_paired']}\n")
    with open("logs/exp_v2_round1_T.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\nsaved logs/exp_v2_round1_T.json + .md")


def eval_metrics_full(df):
    return {"n": len(df), "n_pos": int(df["label_T"].sum()),
            "pooled_auc": round(float(epm.pooled_auc(df)), 4),
            "fold_mean": round(epm.fold_mean_sd(df)[0], 4) if epm.fold_mean_sd(df) else None,
            "fold_sd": round(epm.fold_mean_sd(df)[1], 4) if epm.fold_mean_sd(df) else None}


def _auccell(x):
    if not x or x.get("pooled_auc") is None:
        return "—"
    return f"{x['pooled_auc']} (n={x['n']})"


if __name__ == "__main__":
    main()
