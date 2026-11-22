# -*- coding: utf-8 -*-
"""阶段3 早间审计 (纯 CPU): 汇总决定性对照 + 分支产物, 输出 logs/audit_morning.json + README 追加。
内容: ①全局主表 last5 vs best 并列 (三臂3seed) ②T3-4 vs T1-2 ③batch1 vs batch2 批内 3-seed
     ④pN0/1/2 三级均值与两两 AUC ⑤train_loss 末期 (日志) ⑥单折耗时 (日志时间戳)。
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "logs")
OOF_DIR = os.path.join(ROOT, "results", "roidl_oof")
SEEDS = [42, 0, 7]


def load_oof(tag, seed, suffix=""):
    """合并 5 折 OOF csv -> DataFrame (patient_id,label_T,prob_T,label_N,prob_N)."""
    frames = []
    for k in range(5):
        p = os.path.join(OOF_DIR, f"{tag}_k{k}_s{seed}{suffix}.csv")
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p, encoding="utf-8-sig")
        df["fold"] = k
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def auc_by(df, group_col, prob_col="prob_N", label_col="label_N"):
    out = {}
    for g, sub in df.groupby(group_col):
        if sub[label_col].nunique() < 2 or len(sub) < 8:
            out[str(g)] = None
        else:
            out[str(g)] = float(roc_auc_score(sub[label_col], sub[prob_col]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="dc")
    args = ap.parse_args()
    pf = args.tag_prefix

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    arms = ["a1_sdfonly", "a2_img_sdf", "a3_img"]
    for arm in arms:
        per_seed = {}
        for s in SEEDS:
            tag = f"{pf}_{arm}_s{s}"
            d_best = load_oof(tag, s)
            d_l5 = load_oof(tag, s, "_last5")
            ent = {}
            for nm, d in (("best", d_best), ("last5", d_l5)):
                if d is None:
                    ent[nm] = None
                    continue
                ent[nm] = {
                    "n": len(d),
                    "global_N": (float(roc_auc_score(d["label_N"], d["prob_N"]))
                                 if d["label_N"].nunique() >= 2 else None),
                    "global_T": (float(roc_auc_score(d["label_T"], d["prob_T"]))
                                 if d["label_T"].nunique() >= 2 else None),
                    "T34_vs_T12": auc_by(d, "label_T"),
                }
            per_seed[str(s)] = ent
        report[arm] = per_seed

    # pN 三级 (OOF 不含 pN -> 用 hos_pairs_raw 补)
    raw = pd.read_csv(os.path.join(ROOT, "data", "hos_pairs_raw.csv"), encoding="utf-8-sig")
    raw["patient_id"] = raw["patient_id"].astype(int).astype(str).str.strip()
    for arm in arms:
        tag = f"{pf}_{arm}_s42"
        d = load_oof(tag, 42, "_last5")
        if d is None:
            continue
        d["patient_id"] = d["patient_id"].astype(int).astype(str).str.strip()
        m = d.merge(raw[["patient_id", "pN"]], on="patient_id", how="left")
        pn = m.groupby("pN")["prob_N"].mean().to_dict()
        pairs = {}
        for (a, b) in [(0, 1), (1, 2), (0, 2)]:
            sub = m[m["pN"].isin([a, b])]
            if sub["pN"].nunique() == 2 and len(sub) >= 8:
                pairs[f"{a}v{b}"] = float(roc_auc_score(sub["pN"].gt(a).astype(int), sub["prob_N"]))
        report.setdefault(arm, {})["pN_mean"] = pn
        report.setdefault(arm, {})["pN_pair_auc"] = pairs

    # 批内 (batch1<=200 / batch2>=201)
    raw_b = raw.copy()
    raw_b["batch"] = np.where(raw_b["patient_id"].astype(int) <= 200, "b1", "b2")
    for arm in arms:
        tag = f"{pf}_{arm}_s42"
        d = load_oof(tag, 42, "_last5")
        if d is None:
            continue
        d["patient_id"] = d["patient_id"].astype(int).astype(str).str.strip()
        m = d.merge(raw_b[["patient_id", "batch"]], on="patient_id", how="left")
        report.setdefault(arm, {})["batch_auc"] = auc_by(m, "batch")

    # train_loss 末期 (日志 tail) + 单折耗时
    perfs = {}
    for arm in arms:
        tag = f"{pf}_{arm}_s42"
        logs = sorted(glob.glob(os.path.join(ROOT, "logs", f"roidl_{tag}_s42_*.log")))
        loss_tail = []
        times = []
        for lp in logs:
            with open(lp, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            for ln in lines:
                mm = re.search(r"train_loss=([0-9.]+)", ln)
                if mm:
                    loss_tail.append(float(mm.group(1)))
            ts = [re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln) for ln in lines[:40]]
            ts = [t.group(1) for t in ts if t]
            te = [re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln) for ln in lines[-40:]]
            te = [t.group(1) for t in te if t]
            if ts and te:
                try:
                    d0 = time.mktime(time.strptime(ts[0], "%Y-%m-%d %H:%M:%S"))
                    d1 = time.mktime(time.strptime(te[-1], "%Y-%m-%d %H:%M:%S"))
                    times.append(round((d1 - d0) / 60, 1))
                except ValueError:
                    pass
        perfs[arm] = {"train_loss_last10": loss_tail[-10:], "fold_min": times}
    report["perf"] = perfs

    op = os.path.join(LOG_DIR, "audit_morning.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    with open(os.path.join(ROOT, "README.md"), "a", encoding="utf-8") as f:
        f.write(f"\n\n## 阶段3 早间审计 (自动接力 {time.strftime('%Y-%m-%d %H:%M')})\n")
        f.write(f"- 结果见 logs/audit_morning.json; 三臂 last5 全局 N AUC 汇总: ")
        f.write(", ".join(f"{a}={report.get(a, {}).get('s42', {}).get('last5', {}).get('global_N')}"
                         for a in arms))
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
