# -*- coding: utf-8 -*-
"""整理阶段 1-3 (仅 .pyc/空日志删除、一次性脚本移动、archive 重复 PNG 移动)。
用法: python cleanup_stages123.py [--execute]
"""
import argparse
import csv
import glob
import hashlib
import os
import shutil
import subprocess
import time

ROOT = r"d:\PyFile\ColonMultiModal"
OUT = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "outputs", "cleanup_20261121")
TS = time.strftime("%Y%m%d_%H%M%S")
LOG = []


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line)
    LOG.append(line)


def git_status():
    try:
        return subprocess.run(["git", "status", "--short"], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    except Exception as e:
        return [f"git status failed: {e}"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage1():
    before = git_status()
    pyc = []
    for dp, _, fns in os.walk(ROOT):
        if ".git" in dp.split(os.sep):
            continue
        for n in fns:
            if n.lower().endswith((".pyc", ".pyo")):
                pyc.append(os.path.join(dp, n))
    zero = []
    for base in (os.path.join(ROOT, "logs"), os.path.join(ROOT, "results")):
        for dp, _, fns in os.walk(base):
            for n in fns:
                p = os.path.join(dp, n)
                try:
                    if os.path.getsize(p) == 0 and n.lower().endswith((".log", ".txt")):
                        zero.append(p)
                except OSError:
                    pass
    man = os.path.join(OUT, f"stage1_delete_manifest_{TS}.csv")
    os.makedirs(OUT, exist_ok=True)
    with open(man, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "kind"])
        for p in pyc:
            w.writerow([os.path.relpath(p, ROOT).replace("\\", "/"), os.path.getsize(p), "pyc"])
        for p in zero:
            w.writerow([os.path.relpath(p, ROOT).replace("\\", "/"), 0, "zero_log"])
    log(f"stage1 manifest={man} pyc={len(pyc)} zero_logs={len(zero)}")
    deleted = freed = 0
    errs = []
    for p in pyc + zero:
        try:
            s = os.path.getsize(p)
            os.remove(p)
            deleted += 1
            freed += s
        except Exception as e:
            errs.append((p, str(e)))
    removed_dirs = 0
    for dp, _, _ in sorted(os.walk(ROOT), key=lambda x: -len(x[0])):
        if os.path.basename(dp) == "__pycache__":
            try:
                if not os.listdir(dp):
                    os.rmdir(dp)
                    removed_dirs += 1
            except OSError:
                pass
    after = git_status()
    _write_git("stage1", before, after)
    log(f"stage1 deleted={deleted} freed={freed} pycache_dirs_removed={removed_dirs} errors={errs[:3]}")
    _flush("stage1")
    return {"deleted": deleted, "freed": freed, "pyc": len(pyc), "zero": len(zero), "dirs": removed_dirs}


def stage2():
    before = git_status()
    src = ["_wire_hos.py", "_wire_mri_lesion.py", "_smoke_mri_lesion.py", "_smoke_n_lesion.py",
           "README.md.gbk_backup"]
    dst = os.path.join(ROOT, "archive", "cleanup_20261121", TS)
    os.makedirs(dst, exist_ok=True)
    moved = []
    for n in src:
        p = os.path.join(ROOT, n)
        if os.path.exists(p):
            shutil.move(p, os.path.join(dst, n))
            moved.append((n, os.path.getsize(os.path.join(dst, n))))
    refs = []
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "cleanup_20261121")]
        for n in fns:
            if n.lower().endswith((".py", ".md", ".txt", ".json", ".ps1", ".bat")):
                p = os.path.join(dp, n)
                try:
                    lines = open(p, encoding="utf-8", errors="ignore").read().splitlines()
                except OSError:
                    continue
                for i, l in enumerate(lines, 1):
                    if any(s in l for s in src):
                        refs.append((os.path.relpath(p, ROOT).replace("\\", "/"), i, l.strip()[:160]))
    after = git_status()
    _write_git("stage2", before, after)
    log(f"stage2 moved={len(moved)} -> {dst}; refs={len(refs)}")
    for r in refs[:20]:
        log(f"  ref {r[0]}:{r[1]} {r[2]}")
    _flush("stage2")
    return {"moved": moved, "dst": dst, "refs": refs}


def stage3():
    before = git_status()
    # 重新按大小+哈希找 archive 内重复 PNG
    bysize = {}
    for dp, _, fns in os.walk(os.path.join(ROOT, "archive")):
        if "cleanup_20261121" in dp.split(os.sep):
            continue
        for n in fns:
            if n.lower().endswith((".png", ".jpg", ".jpeg")):
                p = os.path.join(dp, n)
                bysize.setdefault(os.path.getsize(p), []).append(p)
    groups = []
    for s, ps in bysize.items():
        if len(ps) < 2 or s == 0:
            continue
        h = {}
        for p in ps:
            h.setdefault(sha256(p), []).append(p)
        for _, same in h.items():
            if len(same) > 1:
                groups.append(same)
    base = os.path.join(ROOT, "archive", "cleanup_20261121", "duplicates")
    moved = []
    detail = []
    for g in groups:
        # 保留路径优先 plots_early (正式图目录), 其次路径更短
        keep = sorted(g, key=lambda p: (0 if "plots_early" in p.split(os.sep) else 1, len(p)))[0]
        for p in g:
            if p == keep:
                continue
            rel = os.path.relpath(p, os.path.join(ROOT, "archive"))
            dst = os.path.join(base, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sz = os.path.getsize(p)
            h = sha256(p)
            shutil.move(p, dst)
            moved.append((p, dst, sz))
            detail.append((os.path.relpath(keep, ROOT), os.path.relpath(p, ROOT), h, sz))
    after = git_status()
    _write_git("stage3", before, after)
    log(f"stage3 groups={len(groups)} moved={len(moved)} bytes={sum(x[2] for x in moved)}")
    for k, m, h, sz in detail:
        log(f"  keep={k} | move={m} | sha256={h[:16]} bytes={sz}")
    _flush("stage3")
    return {"groups": len(groups), "moved": moved, "detail": detail}


def _write_git(stage, before, after):
    p = os.path.join(OUT, f"{stage}_gitstatus_{TS}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("=== BEFORE ===\n" + "\n".join(before) + "\n=== AFTER ===\n" + "\n".join(after))
    bs, as_ = set(before), set(after)
    log(f"{stage} git status: before={len(before)} after={len(after)} "
        f"added={len(as_-bs)} removed={len(bs-as_)}; file={p}")


def _flush(stage):
    p = os.path.join(OUT, f"{stage}_{TS}.log")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    LOG.clear()
    print("LOG:", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--stage", type=int, default=0, help="1/2/3; 0=全部")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.execute:
        log("dry-run: 仅统计")
        pyc = glob.glob(os.path.join(ROOT, "**", "*.pyc"), recursive=True)
        zero = []
        for base in (os.path.join(ROOT, "logs"), os.path.join(ROOT, "results")):
            for dp, _, fns in os.walk(base):
                for n in fns:
                    p = os.path.join(dp, n)
                    if os.path.getsize(p) == 0 and n.lower().endswith((".log", ".txt")):
                        zero.append(p)
        log(f"pyc={len(pyc)} zero_logs={len(zero)}")
        return
    if args.stage in (0, 1):
        stage1()
    if args.stage in (0, 2):
        stage2()
    if args.stage in (0, 3):
        stage3()


if __name__ == "__main__":
    main()
