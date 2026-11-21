# -*- coding: utf-8 -*-
"""删除 checkpoints_hos_full / checkpoints_hos / archive / logs 下全部 .pth/.pt。
用法: python delete_ckpt_weights.py [--execute]
"""
import argparse
import csv
import glob
import json
import os
import re
import time

ROOT = r"d:\PyFile\ColonMultiModal"
TARGETS = ["checkpoints_hos_full", "checkpoints_hos", "archive", "logs"]
OUTDIR = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "outputs", "snapshot")
TS = time.strftime("%Y%m%d_%H%M%S")
LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    LOG.append(line)


def weights_under(base):
    out = []
    for dp, _, fns in os.walk(base):
        for n in fns:
            if n.lower().endswith((".pth", ".pt")):
                p = os.path.join(dp, n)
                st = os.stat(p)
                out.append({"path": p.replace("\\", "/"), "bytes": st.st_size,
                            "mtime_ns": st.st_mtime_ns, "dir": base})
    return sorted(out, key=lambda x: x["path"])


def nonweights_under(base):
    out = []
    for dp, _, fns in os.walk(base):
        for n in fns:
            if not n.lower().endswith((".pth", ".pt")):
                p = os.path.join(dp, n)
                st = os.stat(p)
                out.append({"path": p.replace("\\", "/"), "bytes": st.st_size,
                            "mtime_ns": st.st_mtime_ns})
    return sorted(out, key=lambda x: x["path"])


def load_runs():
    cands = sorted(glob.glob(os.path.join(OUTDIR, "run_summary_*.csv")))
    if not cands:
        return []
    p = max(cands, key=os.path.getmtime)
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def infer(basename, runs):
    stem = basename
    for ext in (".pth", ".pt"):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
    prefixes = ["best_model_n_", "best_model_", "last_model_n_", "last_model_",
                "model_n_", "model_", "best_", "last_"]
    tag = stem
    changed = True
    while changed:
        changed = False
        for pre in prefixes:
            if tag.startswith(pre):
                tag = tag[len(pre):]
                changed = True
    variants = [tag]
    for junk in ("roidl_", "attr_", "main_", "cal_"):
        variants += [v.replace(junk, "") for v in list(variants)]
    ck_k = re.search(r"_k(\d+)", tag)
    ck_s = re.search(r"_s(\d+)", tag)
    hits = []
    for r in runs:
        name = r["run_name"]
        r_k = re.search(r"_k(\d+)", name)
        r_s = re.search(r"_s(\d+)", name)
        if ck_k and r_k and ck_k.group(1) != r_k.group(1):
            continue
        if ck_s and r_s and ck_s.group(1) != r_s.group(1):
            continue
        if any(v and (v == r.get("tag") or v in name or name.endswith(v)) for v in variants):
            hits.append(name)
    hits = sorted(set(hits))
    if not hits:
        return "unknown", ""
    return ("unique" if len(hits) == 1 else f"multiple:{len(hits)}"), ";".join(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    log(f"start; execute={args.execute}")

    runs = load_runs()
    log(f"run_summary rows used for mapping={len(runs)}")

    allw, allnw = [], []
    for t in TARGETS:
        base = os.path.join(ROOT, t)
        if not os.path.isdir(base):
            log(f"target missing: {t}")
            continue
        w = weights_under(base)
        nw = nonweights_under(base)
        allw += w
        allnw += nw
        log(f"{t}: weights={len(w)} bytes={sum(x['bytes'] for x in w)} non_weights={len(nw)}")

    # 预检
    bad_scope = [w["path"] for w in allw
                 if not any(w["path"].lower().startswith(os.path.join(ROOT, t).replace("\\", "/").lower() + "/")
                            for t in TARGETS)]
    bad_ext = [w["path"] for w in allw if not w["path"].lower().endswith((".pth", ".pt"))]
    log(f"weights total={len(allw)} bytes={sum(w['bytes'] for w in allw)} "
        f"scope_violations={len(bad_scope)} ext_violations={len(bad_ext)}")
    if bad_scope or bad_ext:
        log("ABORT: 存在越界或非权重扩展名条目")
        _write_log(args.execute)
        raise SystemExit(2)

    # 待删清单
    man = os.path.join(OUTDIR, f"weights_ckpt_archive_logs_to_delete_{TS}.csv")
    unknown = 0
    with open(man, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "mtime", "top_dir", "match", "matched_runs"])
        for x in allw:
            status, hits = infer(os.path.basename(x["path"]), runs)
            if status == "unknown":
                unknown += 1
            w.writerow([os.path.relpath(x["path"], ROOT).replace("\\", "/"), x["bytes"],
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x["mtime_ns"] / 1e9)),
                        x["dir"], status, hits])
    log(f"manifest={man}; unknown_mapping={unknown}")

    # 删除前非权重状态
    pre = os.path.join(OUTDIR, f"nonweights_predelete_{TS}.csv")
    with open(pre, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "mtime_ns"])
        for x in allnw:
            w.writerow([os.path.relpath(x["path"], ROOT).replace("\\", "/"), x["bytes"], x["mtime_ns"]])
    log(f"nonweight pre-state={pre} rows={len(allnw)}")

    # pretrained 删除前状态
    pret = os.path.join(ROOT, "pretrained")
    pw = weights_under(pret)
    pstat = {"files": len(pw), "bytes": sum(x["bytes"] for x in pw),
             "mtimes": {os.path.basename(x["path"]): x["mtime_ns"] for x in pw}}
    log(f"pretrained before: files={pstat['files']} bytes={pstat['bytes']}")

    if not args.execute:
        log("dry-run done; no deletion")
        _write_log(args.execute)
        return

    deleted = fail = freed = 0
    errors = []
    for x in allw:
        try:
            os.remove(x["path"])
            deleted += 1
            freed += x["bytes"]
        except Exception as e:
            fail += 1
            errors.append((x["path"], f"{type(e).__name__}: {e}"))
    log(f"deleted={deleted} failed={fail} freed_bytes={freed}")
    if errors:
        log(f"errors sample={errors[:5]}")

    # 删除后校验
    remain = sum(len(weights_under(os.path.join(ROOT, t))) for t in TARGETS if os.path.isdir(os.path.join(ROOT, t)))
    post_nw = []
    for t in TARGETS:
        b = os.path.join(ROOT, t)
        if os.path.isdir(b):
            post_nw += nonweights_under(b)
    pre_map = {x["path"]: (x["bytes"], x["mtime_ns"]) for x in allnw}
    post_map = {x["path"]: (x["bytes"], x["mtime_ns"]) for x in post_nw}
    missing = [p for p in pre_map if p not in post_map]
    added = [p for p in post_map if p not in pre_map]
    changed = [p for p in pre_map if p in post_map and pre_map[p] != post_map[p]]
    log(f"after: remaining_weights={remain} non_weights={len(post_nw)} "
        f"missing={len(missing)} added={len(added)} changed={len(changed)}")

    pw2 = weights_under(pret)
    pstat2 = {"files": len(pw2), "bytes": sum(x["bytes"] for x in pw2),
              "mtimes": {os.path.basename(x["path"]): x["mtime_ns"] for x in pw2}}
    log(f"pretrained after: files={pstat2['files']} bytes={pstat2['bytes']} unchanged={pstat == pstat2}")

    for t in TARGETS:
        b = os.path.join(ROOT, t)
        if os.path.isdir(b):
            log(f"{t} after: weights={len(weights_under(b))} non_weights={len(nonweights_under(b))}")
    log(f"end; elapsed={time.time() - t0:.1f}s")
    _write_log(args.execute)


def _write_log(execute):
    p = os.path.join(OUTDIR, f"delete_ckpt_weights_{TS}{'_exec' if execute else '_dryrun'}.log")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print("LOG:", p)


if __name__ == "__main__":
    main()
