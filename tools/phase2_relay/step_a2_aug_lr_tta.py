# -*- coding: utf-8 -*-
"""路径A-A2: 解剖保序弱增强 (--aug-lr 左右镜像+图像通道强度抖动, SDF 不碰) + 3 seed 训练。
口径: 决定性对照同款 mri_t+SDF (296例, bbox, epochs 30, no-clinical, self-written)。
输出: logs/step_a2_aug_lr.json + README 追加。
注: 2-view TTA 需推理侧支持 (train_roi_dl.py 无 TTA 参数), 本步先验证弱增强训练本身;
    TTA 可后续用 OOF 概率左右镜像平均实现 (待代码支持后补跑)。
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = r"D:\Development\miniconda\envs\Pytorch\python.exe"
LOG_DIR = os.path.join(ROOT, "logs")

BASE = [
    "tools/train_roi_dl.py", "--kfold", "5",
    "--kfold-csv", "data/master_5fold_356.csv",
    "--full-csv", "data/hos_pairs.csv",
    "--epochs", "30", "--self-written", "--no-clinical",
    "--head-dropout", "0.5", "--roi-mask", "bbox",
    "--mode", "mri_t", "--sdf-channel", "--aug-lr",
]
SEEDS = [42, 0, 7]
STEP_TIMEOUT = 2400


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="dc")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    results = {}
    for s in SEEDS:
        tag = f"a2_auglr_s{s}"
        cmd = BASE + ["--seed", str(s), "--out-tag", tag]
        if not args.dry:
            run(cmd, tag)
        agg = aggregate(tag, s)
        results[tag] = {"last5_N": pooled(agg, "N"), "last5_T": pooled(agg, "T")}
        log(f"[agg] {tag}: N={results[tag]['last5_N']} T={results[tag]['last5_T']}")

    v = [r["last5_N"] for r in results.values() if r["last5_N"] is not None]
    mean_n = (sum(v) / len(v)) if v else None
    out = {"aug_lr_N_mean": mean_n, "per_run": results,
           "note": "对照: 决定性对照 Arm2 (mri_t+SDF 无增强) 3seed 均值(见 phase1 决策 json)"}
    op = os.path.join(LOG_DIR, "step_a2_aug_lr.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"done -> {op}")
    with open(os.path.join(ROOT, "README.md"), "a", encoding="utf-8") as f:
        f.write(f"\n\n## 路径A-A2: 解剖保序弱增强 --aug-lr (自动接力 {time.strftime('%Y-%m-%d %H:%M')})\n")
        f.write(f"- mri_t+SDF+aug-lr N(last5,3seed均值)={mean_n}; 对照 Arm2 无增强均值(见决策 json)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
