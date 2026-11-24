# -*- coding: utf-8 -*-
"""从已落盘 OOF CSV 重算 task2 (label_N 线性探针 + 2.5D 增益) 指标, 写回
logs/exp_4060_overnight_summary.json 的 tasks.task2 (原 task2 被 task3/task4 运行覆盖)。

用法: python tools/backfill_task2_json.py
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import exp_4060_overnight as e  # 复用 evaluate_view/pooled_auc_df/fold_mean_sd/auc_ci_boot/eval_subset_ids/load_subsets

OUTDIR = "outputs/exp_4060_overnight"
SUMMARY_JSON = "logs/exp_4060_overnight_summary.json"


def read_oof(name):
    p = os.path.join(OUTDIR, f"oof_{name}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def main():
    master, md, direct_ids = e.load_subsets()
    tcd2, td2 = e.eval_subset_ids(md, "features/slice_feats_v2", "dinov2", ("t", "c", "d"))
    tcd25, td25 = e.eval_subset_ids(md, "features/slice_feats_v2_25d", "dinov2", ("t", "d"))
    print(f"[subsets] tcd2={len(tcd2)} td2={len(td2)} tcd25={len(tcd25)} td25={len(td25)}")

    res2 = {}
    target = "label_N"
    views = {"clin_clean": read_oof(f"clin_{target}"),
             "d": read_oof(f"d_{target}"),
             "t": read_oof(f"t_{target}"),
             "clin_d_t": read_oof(f"clin_d_t_{target}")}
    eval_tcd, eval_td = {}, {}
    for vname, df in views.items():
        if df is None:
            continue
        eval_tcd[vname] = e.evaluate_view(df, tcd2, vname)
        eval_td[vname] = e.evaluate_view(df, td2, vname)
        print(f"  {vname}: tcd={eval_tcd[vname] and eval_tcd[vname]['pooled_auc']} "
              f"td={eval_td[vname] and eval_td[vname]['pooled_auc']}")
    res2[target] = {"direct_tcd": eval_tcd, "direct_td": eval_td}

    # 2.5D 增益: label_T / label_N (对照 2D 单层), 同首跑结构
    for tgt in ("label_T", "label_N"):
        g25, g2d = {}, {}
        for seq in ("d", "t"):
            df25 = read_oof(f"25d_{seq}_{tgt}")
            df2 = read_oof(f"2d_{seq}_{tgt}")
            if df25 is not None:
                g25[seq] = {"tcd": e.evaluate_view(df25, tcd25, f"25d_{seq}"),
                            "td": e.evaluate_view(df25, td25, f"25d_{seq}")}
            if df2 is not None:
                g2d[seq] = {"tcd": e.evaluate_view(df2, tcd25, f"2d_{seq}"),
                            "td": e.evaluate_view(df2, td25, f"2d_{seq}")}
            print(f"  [25d {tgt}] {seq}: 25d_tcd={g25[seq]['tcd'] and round(g25[seq]['tcd']['pooled_auc'],3)} "
                  f"2d_tcd={g2d[seq]['tcd'] and round(g2d[seq]['tcd']['pooled_auc'],3)}")
        res2.setdefault("_25d_gain", {})[tgt] = {"25d": g25, "2d_baseline": g2d}

    d = json.load(open(SUMMARY_JSON, encoding="utf-8"))
    d.setdefault("tasks", {})["task2"] = res2
    d["tasks"]["task2_time_s"] = None  # 重算无耗时
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"\nbackfilled task2 → {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
