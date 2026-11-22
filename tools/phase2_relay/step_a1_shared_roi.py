# -*- coding: utf-8 -*-
"""路径A-A1: 修复多序列通道错位 —— mri_tc/mri_all 共用 mr_t 单一 bbox/SDF (--shared-roi)。
同 216 例 t∩c 子集 (data/hos_pairs_tc.csv), 3 seed 各跑:
  mri_tc_shared_sdf   (--mode mri_tc --sdf-channel --shared-roi)
  mri_all_shared_sdf  (--mode mri_all --sdf-channel --shared-roi)
对照: 阶段1 Arm2 (mri_t+SDF, 296例) 在 216 例上切片 (零重训)。
输出: logs/step_a1_shared_roi.json + README 追加小节。
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = r"D:\Development\miniconda\envs\Pytorch\python.exe"
LOG_DIR = os.path.join(ROOT, "logs")

BASE = [
    "tools/train_roi_dl.py", "--kfold", "5",
    "--kfold-csv", "data/master_5fold_356.csv",
    "--full-csv", "data/hos_pairs_tc.csv",
    "--epochs", "30", "--self-written", "--no-clinical",
    "--head-dropout", "0.5", "--roi-mask", "bbox",
]
RUNS = [
    ("a1_tc_shared_s{s}", ["--mode", "mri_tc", "--sdf-channel", "--shared-roi"]),
    ("a1_all_shared_s{s}", ["--mode", "mri_all", "--sdf-channel", "--shared-roi"]),
]
SEEDS = [42, 0, 7]
STEP_TIMEOUT = 2400


def log(m):
    print(time_str() + " " + str(m), flush=True)


def time_str():
    import time
    return time.strftime("[%H:%M:%S]")


def run(cmd, tag, timeout=STEP_TIMEOUT):
    log(f"[run:{tag}] {' '.join(cmd)[:140]}")
    t0 = time.time()
    try:
        r = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        dt = time.time() - t0
        ok = (r.returncode == 0)
        log(f"[run:{tag}] {'OK' if ok else 'EXIT=' + str(r.returncode)} ({dt/60:.1f}min)")
        if not ok:
            log((r.stdout or "")[-800:] + (r.stderr or "")[-300:])
        return ok
    except subprocess.TimeoutExpired:
        log(f"[run:{tag}] TIMEOUT -> KILL")
        return False


def aggregate(tag, seed):
    out = os.path.join(LOG_DIR, f"aggregate_roidl_{tag}_s{seed}.json")
    cmd = ["tools/aggregate_roidl_kfold.py", "--tag", tag, "--seed", str(seed), "--out", out]
    run(cmd, f"agg_{tag}_s{seed}", timeout=300)
    if os.path.exists(out):
        with open(out, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def pooled(agg, task):
    if not agg:
        return None
    return ((agg.get(f"task_{task}_last5") or {}).get("pooled") or {}).get("roc_auc")


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="dc")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    results = {}
    for tmpl, extra in RUNS:
        for s in SEEDS:
            tag = tmpl.format(s=s)
            cmd = BASE + extra + ["--seed", str(s), "--out-tag", tag]
            if not args.dry:
                run(cmd, tag)
            agg = aggregate(tag, s)
            results[tag] = {"last5_N": pooled(agg, "N"), "last5_T": pooled(agg, "T")}
            log(f"[agg] {tag}: N={results[tag]['last5_N']} T={results[tag]['last5_T']}")

    def arm(prefix):
        v = [results.get(f"{prefix}_s{s}", {}).get("last5_N") for s in SEEDS]
        v = [x for x in v if x is not None]
        return (sum(v) / len(v)) if v else None

    out = {
        "arm_tc_shared": arm("a1_tc_shared"),
        "arm_all_shared": arm("a1_all_shared"),
        "per_run": results,
        "note": "对照=阶段1 Arm2 (mri_t+SDF) 在 216 例子集切片 (见 phase1 聚合); "
                "若 tc_shared/all_shared 的 N AUC 明显高于 P2 0.6422 (t+c+SDF 未修复错位), 则错位修复有效",
    }
    op = os.path.join(LOG_DIR, "step_a1_shared_roi.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"done -> {op}")
    # README 追加
    readme = os.path.join(ROOT, "README.md")
    with open(readme, "a", encoding="utf-8") as f:
        f.write(f"\n\n## 路径A-A1: shared-roi 多序列通道错位修复 (自动接力 {time.strftime('%Y-%m-%d %H:%M')})\n")
        f.write(f"- mri_tc_shared_sdf N(last5,3seed均值)={out['arm_tc_shared']}; "
                f"mri_all_shared_sdf={out['arm_all_shared']}\n")
        f.write(f"- 对照: P2 (t+c+SDF 未修复) 0.6422; 若 shared 版本持平或更高且子群一致, 错位修复有效/中性\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
