# -*- coding: utf-8 -*-
"""
体检 T3 代理分层 (零训练): 用 Arm D OOF prob_N 做代理变量检验。
- 输入: p0_attr_D_binonly OOF (三 seed) + hos_pairs_raw (sex/pT/pN 原始值)
- 判据 (用户 m00153): 按 sex、label_T 分层算 N AUC, 并检查预测概率在 pN0/pN1/pN2 单调性。
  子群内 < 0.55 报警; 子群内 >= 0.59 且 pN0 < pN1 < pN2 单调递增判为通过。
- 输出: 分层 AUC 明细 + 判定, 写 README 并落盘 CSV
"""
import glob, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OOF_GLOB = os.path.join(ROOT, "results", "roidl_oof", "p0_attr_D_binonly_k*_s*.csv")
RAW_CLIN = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
OUT_DIR = os.path.join(ROOT, "tools", "phase2_healthcheck", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    clin = pd.read_csv(RAW_CLIN, encoding="utf-8-sig")
    clin = clin[["patient_id", "sex", "pT", "pN"]].copy()

    files = [f for f in sorted(glob.glob(OOF_GLOB)) if "_last5" not in f]
    seeds = sorted({os.path.basename(f).split("_s")[-1].replace(".csv", "")
                    for f in files})
    prob_cols = []
    for s in seeds:
        sfiles = [f for f in files if f.endswith(f"_s{s}.csv")]
        dfs = [pd.read_csv(f) for f in sfiles]
        df = pd.concat(dfs, ignore_index=True)
        df = df[["patient_id", "label_N", "prob_N"]].rename(columns={"prob_N": f"prob_N_s{s}"})
        if s == seeds[0]:
            merged = df
        else:
            merged = merged.merge(df, on=["patient_id", "label_N"], how="outer")
        prob_cols.append(f"prob_N_s{s}")
    # 三 seed 平均概率
    merged["prob_N_mean"] = merged[prob_cols].mean(axis=1)
    merged = merged.merge(clin, on="patient_id", how="left")
    merged["pT"] = pd.to_numeric(merged["pT"], errors="coerce")
    merged["pN"] = pd.to_numeric(merged["pN"], errors="coerce")
    # label_T 从 OOF 里没有, 用 pT 二值化
    merged["label_T"] = (merged["pT"] >= 3).astype(int)

    print("=== T3 代理分层 (Arm D OOF prob_N mean over seeds) ===")
    rows = []
    # 1) sex 分层
    for g, grp in merged.groupby("sex"):
        grp = grp.dropna(subset=["prob_N_mean"])
        if grp["label_N"].nunique() < 2 or len(grp) < 10:
            rows.append({"stratum": f"sex={g}", "n": len(grp), "n_pos": int(grp["label_N"].sum()),
                         "auc_n": np.nan})
            continue
        auc = roc_auc_score(grp["label_N"], grp["prob_N_mean"])
        rows.append({"stratum": f"sex={g}", "n": len(grp), "n_pos": int(grp["label_N"].sum()),
                     "auc_n": auc})
    # 2) label_T 分层
    for g, grp in merged.groupby("label_T"):
        grp = grp.dropna(subset=["prob_N_mean"])
        if grp["label_N"].nunique() < 2 or len(grp) < 10:
            rows.append({"stratum": f"label_T={g}", "n": len(grp), "n_pos": int(grp["label_N"].sum()),
                         "auc_n": np.nan})
            continue
        auc = roc_auc_score(grp["label_N"], grp["prob_N_mean"])
        rows.append({"stratum": f"label_T={g}", "n": len(grp), "n_pos": int(grp["label_N"].sum()),
                     "auc_n": auc})

    # 3) pN 单调性: prob_N_mean 在 pN0/pN1/pN2 上的均值
    mono = merged.dropna(subset=["prob_N_mean", "pN"])
    pn_means = mono.groupby("pN")["prob_N_mean"].agg(["mean", "count"])
    rows.append({"stratum": "pN 单调性", "n": len(mono),
                 "n_pos": int(mono["label_N"].sum()),
                 "auc_n": np.nan})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\n--- prob_N_mean 按 pN 亚级均值 ---")
    for pn, r in pn_means.iterrows():
        print(f"pN{pn}: {r['mean']:.4f} (n={int(r['count'])})")

    # 判定
    sub = res.dropna(subset=["auc_n"])
    alarm = sub[sub["auc_n"] < 0.55]
    passed_sub = sub[sub["auc_n"] >= 0.59]
    monotonic = (len(pn_means) >= 3 and
                 pn_means.loc[0, "mean"] < pn_means.loc[1, "mean"] < pn_means.loc[2, "mean"])
    if len(alarm) > 0:
        verdict = f"报警: 子群内 < 0.55 ({', '.join(alarm['stratum'])})"
    elif len(passed_sub) >= 2 and monotonic:
        verdict = "通过: 子群内 >= 0.59 且 pN0 < pN1 < pN2 单调递增"
    elif len(passed_sub) >= 1 and monotonic:
        verdict = f"部分通过: {len(passed_sub)} 个子群 >= 0.59 且 pN 单调递增"
    else:
        verdict = "边界情况: 未见 >= 0.59 子群或 pN 非单调"
    print(f"\n判定: {verdict}")
    res.to_csv(os.path.join(OUT_DIR, "t3_proxy_stratified.csv"), index=False)
    print(f"已落盘: {OUT_DIR}\\t3_proxy_stratified.csv")
    return verdict, res, pn_means


if __name__ == "__main__":
    main()
