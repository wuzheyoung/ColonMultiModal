# -*- coding: utf-8 -*-
"""stage0_reproduce_baseline.py — 阶段0: 只读复现锁定基线 (10×5 重复分层 CV, TRIPOD 1b)

复用 exp_v2_validation_suite.py 的精确管线 (pool_branch/probe_generic/clin_cv/
calibrate_cv/fuse_zlogit_strategy), 锁定校准模式='bonly' (与 logs/exp_v2_validation_suite_T.json
一致)。只评估 direct_td / direct_tcd 两个 scope 的 clin_dt (clin_clean + d_self(MeanMax) + t(AWM)),
不运行 m1/m3/m4, 不触碰 20250428 时间分割集与 nCRT 队列 (铁律3)。

输出: logs/auto_staging_opt/stage0_baseline.json + progress_log.md 首节。
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

# 1650 端脚本 (exp_v2_final_T/exp_v2_final_calibrated) 硬编码 C:\Users\Chris\... 路径,
# 本机不存在 → import 前临时包装 os.chdir 跳过不存在的目录, 保持当前 cwd=项目根。
_orig_chdir = os.chdir

def _safe_chdir(p):
    if os.path.isdir(p):
        return _orig_chdir(p)
    # 目标目录不存在(1650 端路径) → 不切, 保持当前目录
    return None

os.chdir = _safe_chdir
try:
    import exp_v2_validation_suite as S  # noqa: E402
finally:
    os.chdir = _orig_chdir

LOCK_MODE = "bonly"  # 与既有 json locked_calibration_mode 一致
OUTDIR = "logs/auto_staging_opt"
os.makedirs(OUTDIR, exist_ok=True)

def run():
    # 精确复现 module2 的 clin_dt 部分 (仅 seed0-9, 两 scope)
    d_pids = S.all_direct
    t_have, t_X, t_y = S.pool_branch(d_pids, "t", "area_w")
    d_have, d_X, d_y = S.pool_branch(sorted(S.self_d), "d", "meanmax")
    t_map = {int(p): i for i, p in enumerate(t_have)}
    d_map = {int(p): i for i, p in enumerate(d_have)}
    pos_of_pid = {int(p): i for i, p in enumerate(d_pids)}
    y_all = S.md_i.loc[d_pids, "label_T"].values.astype(int)
    seeds = range(10)
    res = {}
    for scope_name, scope in (("direct_td", S.td), ("direct_tcd", S.tcd)):
        per_seed = {"clin": [], "clin_t": [], "clin_dt": []}
        fold_aucs = {"clin": [], "clin_t": [], "clin_dt": []}
        for seed in seeds:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_of = np.empty(len(d_pids), dtype=int)
            for k, (_, va) in enumerate(skf.split(d_pids, y_all)):
                fold_of[va] = k
            clin_df, _ = S.clin_cv(d_pids, fold_of)
            t_folds = np.array([fold_of[pos_of_pid[p]] for p in t_have])
            t_df, _ = S.probe_generic(t_have, t_X, t_y, t_folds)
            d_folds = np.array([fold_of[pos_of_pid[p]] for p in d_have])
            d_df, _ = S.probe_generic(d_have, d_X, d_y, d_folds)
            fuse_ct = S.F.fuse_zlogit_strategy({"clin": clin_df, "t": t_df}, strategy="B")
            fuse_cdt = S.F.fuse_zlogit_strategy({"clin": clin_df, "t": t_df, "d": d_df}, strategy="B")
            ct_cal, _ = S.calibrate_cv(fuse_ct, mode=LOCK_MODE)
            cdt_cal, _ = S.calibrate_cv(fuse_cdt, mode=LOCK_MODE)
            ct_s = ct_cal[ct_cal.patient_id.isin(scope)].reset_index(drop=True)
            cdt_s = cdt_cal[cdt_cal.patient_id.isin(scope)].reset_index(drop=True)
            cl_s = clin_df[clin_df.patient_id.isin(scope)].reset_index(drop=True)
            cl_s["prob"] = np.clip(cl_s["prob"], 0, 1)
            a_clin = float(roc_auc_score(cl_s["label_T"], cl_s["prob"]))
            a_ct = float(roc_auc_score(ct_s["label_T"], ct_s["prob"]))
            a_cdt = float(roc_auc_score(cdt_s["label_T"], cdt_s["prob"]))
            per_seed["clin"].append(round(a_clin, 4))
            per_seed["clin_t"].append(round(a_ct, 4))
            per_seed["clin_dt"].append(round(a_cdt, 4))
            for name, df in (("clin", cl_s), ("clin_t", ct_s), ("clin_dt", cdt_s)):
                for k in np.unique(df["fold"]):
                    g = df[df["fold"] == k]
                    if len(g) >= 8 and len(np.unique(g["label_T"])) == 2:
                        fold_aucs[name].append(round(float(roc_auc_score(g["label_T"], g["prob"])), 4))
        rep = {}
        for name in ("clin", "clin_t", "clin_dt"):
            a = np.array(per_seed[name])
            fa = np.array(fold_aucs[name])
            rep[name] = {
                "repeated_pooled_auc_mean": round(float(a.mean()), 4),
                "repeated_pooled_auc_sd_ddof1": round(float(a.std(ddof=1)), 4),
                "repeated_pooled_auc_min": round(float(a.min()), 4),
                "repeated_pooled_auc_max": round(float(a.max()), 4),
                "per_seed_aucs": [round(float(x), 4) for x in a],
                "fold50_mean": round(float(fa.mean()), 4) if len(fa) else None,
                "fold50_sd": round(float(fa.std(ddof=1)), 4) if len(fa) else None,
            }
        res[scope_name] = rep
    # 对照既有锁定 json
    old = json.load(open("logs/exp_v2_validation_suite_T.json", encoding="utf-8"))
    check = {}
    for scope_name in ("direct_td", "direct_tcd"):
        for name in ("clin", "clin_t", "clin_dt"):
            o = old["m2"][scope_name][name]["repeated_pooled_auc_mean"]
            n = res[scope_name][name]["repeated_pooled_auc_mean"]
            check[f"{scope_name}.{name}"] = {"reproduced": n, "locked_json": o,
                                             "match": abs(n - o) < 0.001}
    out = {"module": "stage0 reproduce baseline (10x5, bonly)", "lock_mode": LOCK_MODE,
           "reproduced": res, "vs_locked_json": check,
           "baseline_definition": "clin_clean + d_self(MeanMax) + t(AWM), features/slice_feats_v2/dinov2, strategy B + bonly calib"}
    with open(os.path.join(OUTDIR, "stage0_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # progress log 首节
    L = ["# auto_staging_opt 自主迭代实验进度日志", "",
         f"## 阶段0 (锁基线, {__import__('datetime').datetime.now().isoformat(timespec='seconds')})", ""]
    for scope_name in ("direct_td", "direct_tcd"):
        L.append(f"### {scope_name}")
        for name in ("clin", "clin_t", "clin_dt"):
            r = res[scope_name][name]
            L.append(f"- {name}: 10×5 Mean AUC {r['repeated_pooled_auc_mean']:.4f} ± "
                     f"{r['repeated_pooled_auc_sd_ddof1']:.4f} (min {r['repeated_pooled_auc_min']:.4f}, "
                     f"max {r['repeated_pooled_auc_max']:.4f}), fold50 {r['fold50_mean']}±{r['fold50_sd']}")
    L.append("")
    with open(os.path.join(OUTDIR, "progress_log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(json.dumps(check, ensure_ascii=False, indent=1))
    print("saved ->", os.path.join(OUTDIR, "stage0_baseline.json"))

if __name__ == "__main__":
    run()
