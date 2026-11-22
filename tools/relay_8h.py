# -*- coding: utf-8 -*-
"""8 小时自动接力调度器 (ColonMultiModal 决定性对照 + 分支执行)。

设计:
- 阶段1 (0-2h): 9 次基准训练 (Arm1/2/3 x seed 42/0/7) 严格串行, 独立子进程, 每跑完 1 个 seed 退出由本脚本拉起下一个。
- 聚合: 调 tools/aggregate_roidl_kfold.py 输出各臂 3-seed last5 OOF 指标 (ROC/PR/F1@0.5/LOFO F1, T 与 N 并列)。
- 分流: Arm2_N - Arm1_N >= +0.025 -> 路径 A; < +0.025 (含持平/灰度拖后腿) -> 一律路径 B。
- 阶段2 (2-7.5h): 按分流执行路径 A 或 B 的步骤脚本 (tools/phase2_relay/step_*.py); 脚本缺失则记日志跳过, 不崩。
- 阶段3 (7.5-8h): 纯 CPU 早间审计 (tools/phase2_relay/audit_morning.py)。
- 硬截断: elapsed >= 7.5h 禁止再启动新训练, 直接进阶段3。
用法: python tools/relay_8h.py [--start-at HH:MM] [--tag-prefix dc] [--dry]
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = r"D:\Development\miniconda\envs\Pytorch\python.exe"
LOG_DIR = os.path.join(ROOT, "logs")
RELAY_DIR = os.path.join(ROOT, "tools", "phase2_relay")
STATE_FILE = os.path.join(LOG_DIR, "relay_8h_state.json")
LOG_FILE = os.path.join(LOG_DIR, "relay_8h.log")

# 统一基准命令骨架 (阶段1)
BASE_CMD = [
    "tools/train_roi_dl.py",
    "--kfold", "5",
    "--kfold-csv", "data/master_5fold_356.csv",
    "--full-csv", "data/hos_pairs.csv",
    "--mode", "mri_t",
    "--roi-mask", "bbox",
    "--epochs", "30",
    "--self-written",
    "--no-clinical",
    "--head-dropout", "0.5",
]
ARMS = [
    ("a1_sdfonly", ["--sdf-channel", "--sdf-only"]),
    ("a2_img_sdf", ["--sdf-channel"]),
    ("a3_img", []),
]
SEEDS = [42, 0, 7]
STEP_TIMEOUT = 1800  # 单次训练 watchdog: 30min (5折x30轮实测 ~13min, 余量 2x)
D = 0.025  # 分流阈值


def log(msg):
    line = time.strftime("[%H:%M:%S] ") + str(msg)
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_state(st):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_cmd(cmd, timeout=STEP_TIMEOUT, tag=""):
    """独立子进程运行; 超时 kill 并记录 (防硬件硬卡死). 返回 (ok, stdout_tail)."""
    log(f"[run:{tag}] {' '.join(cmd)[:160]}")
    t0 = time.time()
    try:
        r = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        dt = time.time() - t0
        tail = (r.stdout or "")[-1500:] + (r.stderr or "")[-500:]
        ok = (r.returncode == 0)
        if not ok:
            log(f"[run:{tag}] EXIT={r.returncode} ({dt/60:.1f}min)")
            # 落盘完整输出用于排错 (不再吞掉)
            tailf = os.path.join(LOG_DIR, f"run_{tag}.log")
            with open(tailf, "w", encoding="utf-8", errors="replace") as f:
                f.write((r.stdout or "") + "\n---STDERR---\n" + (r.stderr or ""))
            log(f"[run:{tag}] 详细输出 -> {tailf}")
        else:
            log(f"[run:{tag}] OK ({dt/60:.1f}min)")
        return ok, tail
    except subprocess.TimeoutExpired:
        log(f"[run:{tag}] TIMEOUT {timeout}s -> KILL (watchdog 触发, 防卡死)")
        return False, "TIMEOUT"


def aggregate(tag, seed):
    """调 aggregate_roidl_kfold.py, 返回 json 或 None."""
    out = os.path.join(LOG_DIR, f"aggregate_roidl_{tag}_s{seed}.json")
    cmd = ["tools/aggregate_roidl_kfold.py", "--tag", tag, "--seed", str(seed), "--out", out]
    ok, _ = run_cmd(cmd, timeout=300, tag=f"agg_{tag}_s{seed}")
    if ok and os.path.exists(out):
        with open(out, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def last5_auc(agg, task="N"):
    """取 3-seed 平均用的单 seed last5 pooled roc_auc."""
    if agg is None:
        return None
    s = agg.get(f"task_{task}_last5") or {}
    pooled = s.get("pooled") or {}
    return pooled.get("roc_auc")


def phase1(tag_prefix, dry):
    """9 次串行训练 + 聚合."""
    plan = []
    for arm_tag, extra in ARMS:
        for seed in SEEDS:
            tag = f"{tag_prefix}_{arm_tag}_s{seed}"
            plan.append((tag, BASE_CMD + extra + ["--seed", str(seed), "--out-tag", tag]))
    log(f"阶段1: 共 {len(plan)} 次串行训练")
    done = load_state().get("phase1_done", [])
    results = {}
    for tag, cmd in plan:
        if tag in done:
            log(f"[phase1] {tag} 已完成, 跳过")
        elif dry:
            log(f"[dry] 将运行: {tag}")
        else:
            ok, _ = run_cmd(cmd, tag=tag)
            if not ok:
                log(f"[phase1] {tag} 训练失败 -> 接力中止 (9 次训练是主裁判, 缺一不可). 详情: {os.path.join(LOG_DIR, f'run_{tag}.log')}")
                save_state({"phase1_done": done, "phase1_results": results,
                            "abort": f"train failed: {tag}", "abort_time": time.strftime("%H:%M:%S")})
                sys.exit(1)
            done.append(tag)
            save_state({"phase1_done": done, "phase1_results": results})
        agg = aggregate(tag, seed=int(tag.rsplit("_s", 1)[1]))
        results[tag] = {"last5_N": last5_auc(agg, "N"), "last5_T": last5_auc(agg, "T"),
                        "agg_file": os.path.join(LOG_DIR, f"aggregate_roidl_{tag}.json")}
    save_state({"phase1_done": done, "phase1_results": results})
    return results


def decide(results, tag_prefix):
    """按 Arm2_N - Arm1_N 无死角二分分流."""
    def arm(tag):
        v = [results.get(f"{tag_prefix}_{tag}_s{s}", {}).get("last5_N") for s in SEEDS]
        v = [x for x in v if x is not None]
        return (sum(v) / len(v)) if v else None

    a1, a2, a3 = arm("a1_sdfonly"), arm("a2_img_sdf"), arm("a3_img")
    log(f"分流依据 (3-seed 平均 last5 OOF N AUC): Arm1={a1} Arm2={a2} Arm3={a3}")
    if a1 is None or a2 is None:
        log("[warn] Arm1/Arm2 last5 N AUC 缺失, 无法分流 -> 强制走路径 B (保守)")
        path = "B"
    else:
        diff = a2 - a1
        path = "A" if diff >= D else "B"
        log(f"Arm2-Arm1 = {diff:+.4f} (阈值 {D}) -> 路径 {path}")
    return {"path": path, "arm1_N": a1, "arm2_N": a2, "arm3_N": a3, "diff": (a2 - a1) if (a1 is not None and a2 is not None) else None}


def run_steps(path, tag_prefix, deadline):
    """执行路径 A/B 步骤脚本; 脚本缺失记日志跳过."""
    steps = {
        "A": [
            ("a1_shared_roi", "step_a1_shared_roi.py", 5400),   # 6 次训练(2模式×3seed) ~80min
            ("a2_aug_lr_tta", "step_a2_aug_lr_tta.py", 3600),
            ("a3_mesorectum", "step_a3_mesorectum.py", 10800),  # 296例T2分割 2.1h + 验证 0.7h
        ],
        "B": [
            ("b1_occlusion", "step_b1_occlusion.py", 1800),
            ("b2_build_hr15", "step_b2_build_hr15.py", 5400),   # 296例 1.5mm³ 采样 ~0.8h
            ("b3_hr_train", "step_b3_hr_train.py", 7200),       # 高分辨三臂×3seed ~2h
            ("b4_hr_extend", "step_b4_hr_extend.py", 7200),
        ],
    }
    for name, script, tmo in steps[path]:
        if time.time() > deadline:
            log(f"[截断] 已到 7.5h 硬截断线, 不再启动新训练; 跳过 {name}")
            break
        sp = os.path.join(RELAY_DIR, script)
        if not os.path.exists(sp):
            log(f"[skip:{name}] 脚本未就绪: {sp} (夜间补写后可在次日手动补跑)")
            continue
        log(f"[step:{name}] 启动 {script}")
        run_cmd([sp, "--tag-prefix", tag_prefix], timeout=tmo, tag=name)


def phase3(tag_prefix):
    sp = os.path.join(RELAY_DIR, "audit_morning.py")
    if not os.path.exists(sp):
        log("[skip:audit] audit_morning.py 未就绪")
        return
    run_cmd([sp, "--tag-prefix", tag_prefix], timeout=3600, tag="audit_morning")


def wait_until(hhmm):
    """等到当日 HH:MM (简单格式 '01:30')."""
    now = time.localtime()
    target = time.strptime(hhmm, "%H:%M")
    t0 = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, target.tm_hour, target.tm_min, 0, 0, 0, -1))
    wait = t0 - time.time()
    if wait > 0:
        log(f"等待 {wait/60:.0f}min 至 {hhmm} 启动")
        time.sleep(wait)
    else:
        log(f"目标时间 {hhmm} 已过 ({-wait/60:.0f}min), 立即启动")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-at", default="", help="HH:MM 准点启动; 空=立即")
    ap.add_argument("--tag-prefix", default="dc")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = time.time()
    log(f"===== relay_8h 启动 (prefix={args.tag_prefix}) =====")
    if args.start_at:
        wait_until(args.start_at)
    t_start = time.time()

    results = phase1(args.tag_prefix, args.dry)
    decision = decide(results, args.tag_prefix)
    save_state({"phase1_done": list(results.keys()), "phase1_results": results, "decision": decision})

    deadline = t_start + 7.5 * 3600
    run_steps(decision["path"], args.tag_prefix, deadline)

    log("===== 进入阶段3: 早间审计 (纯 CPU) =====")
    phase3(args.tag_prefix)
    log(f"===== relay_8h 结束, 总耗时 {(time.time()-t_start)/3600:.2f}h =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
