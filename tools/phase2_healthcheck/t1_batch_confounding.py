# -*- coding: utf-8 -*-
"""
体检 T1 批次混杂 (零训练): 按导入批次拆 Arm D (binary_only) OOF 的批内 N AUC。
- 输入: results/roidl_oof/p0_attr_D_binonly_k{0-4}_s{0,42,7}.csv
- 批次: batch1 = patient_id<=200, batch2 = patient_id 201-408 (减200=原编号)
- 判据 (用户 m00153): 批内均值 < 0.56 报警 (混杂/伪影); 至少 2 个主要批次 >= 0.60 通过 (真信号)
- 输出: 逐 seed 批内 AUC + 三 seed 均值±std + 判定, 写 README 并落盘 CSV
"""
import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OOF_GLOB = os.path.join(ROOT, "results", "roidl_oof", "p0_attr_D_binonly_k*_s*.csv")
OUT_DIR = os.path.join(ROOT, "tools", "phase2_healthcheck", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def batch_of(pid):
    return "batch1" if pid <= 200 else ("batch2" if pid <= 408 else "batch3")


def main():
    files = sorted(glob.glob(OOF_GLOB))
    # 排除 _last5
    files = [f for f in files if "_last5" not in f]
    seeds = sorted({os.path.basename(f).split("_s")[-1].replace(".csv", "")
                    for f in files})
    rows = []
    per_seed = {}
    for s in seeds:
        sfiles = [f for f in files if f.endswith(f"_s{s}.csv")]
        dfs = [pd.read_csv(f) for f in sfiles]
        # 每折 OOF 患者应互斥, 合并即全队列
        df = pd.concat(dfs, ignore_index=True)
        df["batch"] = df["patient_id"].apply(batch_of)
        df = df[df["label_N"].isin([0, 1])]
        per_seed[s] = df
        for b, g in df.groupby("batch"):
            if g["label_N"].nunique() < 2:
                rows.append({"seed": s, "batch": b, "n": len(g), "n_pos": int(g["label_N"].sum()),
                             "auc_n": np.nan})
                continue
            auc = roc_auc_score(g["label_N"], g["prob_N"])
            rows.append({"seed": s, "batch": b, "n": len(g), "n_pos": int(g["label_N"].sum()),
                         "auc_n": auc})
    res = pd.DataFrame(rows)
    # 汇总: 每批 3 seed mean±std (全队列 auc 一并)
    summ = []
    for b in ["batch1", "batch2", "batch3"]:
        g = res[res["batch"] == b]
        if g.empty:
            continue
        aucs = g["auc_n"].dropna().values
        summ.append({"batch": b, "n": int(g["n"].max()), "auc_mean": aucs.mean() if len(aucs) else np.nan,
                     "auc_std": aucs.std(ddof=1) if len(aucs) > 1 else np.nan,
                     "seeds": len(aucs)})
    # 全队列 per seed
    full = []
    for s, df in per_seed.items():
        if df["label_N"].nunique() == 2:
            full.append({"seed": s, "n": len(df), "auc_full": roc_auc_score(df["label_N"], df["prob_N"])})
    full_df = pd.DataFrame(full)

    print("=== T1 批次混杂 (Arm D binary_only OOF, 零训练) ===")
    print(res.to_string(index=False))
    print("\n--- 汇总 (3 seed) ---")
    for r in summ:
        print(f"{r['batch']}: n={r['n']} 批内N AUC {r['auc_mean']:.3f}±{r['auc_std']:.3f} (seeds={r['seeds']})")
    print("\n--- 全队列 AUC ---")
    if not full_df.empty:
        print(f"全队列: {' / '.join(f'{s}:{a:.3f}' for s,a in zip(full_df['seed'], full_df['auc_full']))}")

    # 判定
    major = [r for r in summ if r["n"] >= 20]
    minor = [r for r in summ if r["n"] < 20]
    mean_auc = np.mean([r["auc_mean"] for r in summ]) if summ else np.nan
    passed = [r for r in major if r["auc_mean"] >= 0.60]
    verdict = "报警: 批内均值 < 0.56 (混杂/伪影)" if mean_auc < 0.56 else (
        "通过: 至少 2 个主要批次 >= 0.60 (真信号)" if len(passed) >= 2 else
        f"边界情况: 批内均值 {mean_auc:.3f}, 达标主要批次 {len(passed)} 个 (需至少2)")
    print(f"\n判定: {verdict}")
    print(f"提示: 批内样本量小的批次({[r['batch'] for r in minor]}) AUC 不稳定, 参考为主")

    # 落盘
    res.to_csv(os.path.join(OUT_DIR, "t1_batch_confounding_detail.csv"), index=False)
    summ_df = pd.DataFrame(summ)
    summ_df["verdict"] = verdict
    summ_df.to_csv(os.path.join(OUT_DIR, "t1_batch_confounding_summary.csv"), index=False)
    print(f"\n已落盘: {OUT_DIR}\\t1_batch_confounding_*.csv")
    return verdict, summ, full_df


if __name__ == "__main__":
    main()
