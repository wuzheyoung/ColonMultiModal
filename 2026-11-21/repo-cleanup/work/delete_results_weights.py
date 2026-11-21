# -*- coding: utf-8 -*-
"""按快照清单删除 results/ 下全部 .pth/.pt (仅删除, 不移动)。
用法: python delete_results_weights.py [--execute]
不加 --execute 只做预检与清单, 不删除。
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import time

ROOT = r"d:\PyFile\ColonMultiModal"
RESULTS = os.path.join(ROOT, "results")
SNAP_DIR = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "outputs", "snapshot")
TS = time.strftime("%Y%m%d_%H%M%S")
LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    LOG.append(line)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def latest_snapshot():
    cands = sorted(glob.glob(os.path.join(SNAP_DIR, "snapshot_*.json")))
    if not cands:
        raise SystemExit("no snapshot json")
    p = max(cands, key=os.path.getmtime)
    return p, json.load(open(p, encoding="utf-8"))


def load_sha_manifest(snap_json):
    # 与快照同时间戳的 sha256 csv
    ts = os.path.basename(snap_json).replace("snapshot_", "").replace(".json", "")
    p = os.path.join(SNAP_DIR, f"sha256_{ts}.csv")
    rows = []
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    return p, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="实际删除; 缺省仅预检")
    args = ap.parse_args()
    t0 = time.time()
    log(f"start; execute={args.execute}; ROOT={ROOT}")

    snapshot, snap = latest_snapshot()
    sha_path, sha_rows = load_sha_manifest(snapshot)
    log(f"snapshot={snapshot}; sha_manifest={sha_path} ({len(sha_rows)} rows)")

    expected = []
    for r in snap.get("runs", []):
        names = [x for x in (r.get("weight_names") or "").split(";") if x]
        for n in names:
            expected.append({"path": os.path.join(ROOT, r["run_dir"], n).replace("\\", "/"),
                             "run": r["run_name"],
                             "dir": os.path.join(ROOT, r["run_dir"])})
    log(f"snapshot weight entries={len(expected)}")

    disk = []
    for dp, _, fns in os.walk(RESULTS):
        for n in fns:
            if n.lower().endswith((".pth", ".pt")):
                disk.append(os.path.join(dp, n).replace("\\", "/"))
    disk = sorted(disk)
    exp_paths = sorted(x["path"] for x in expected)
    missing_on_disk = [p for p in exp_paths if not os.path.exists(p)]
    extra_on_disk = [p for p in disk if p not in set(exp_paths)]
    log(f"disk weights={len(disk)}; expected={len(exp_paths)}; "
        f"missing_on_disk={len(missing_on_disk)}; extra_on_disk={len(extra_on_disk)}")

    # 范围/扩展名校验
    bad_scope = [p for p in exp_paths if not p.lower().startswith(RESULTS.lower().replace("\\", "/") + "/")]
    bad_ext = [p for p in exp_paths if not p.lower().endswith((".pth", ".pt"))]
    log(f"scope_violations={len(bad_scope)} ext_violations={len(bad_ext)}")

    manifest_path = os.path.join(SNAP_DIR, f"weights_to_delete_{TS}.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "run", "exists"])
        total = 0
        for p in exp_paths:
            sz = os.path.getsize(p) if os.path.exists(p) else 0
            total += sz
            run = next((x["run"] for x in expected if x["path"] == p), "")
            w.writerow([p, sz, run, int(os.path.exists(p))])
    log(f"manifest={manifest_path}; total_bytes={total}")

    if missing_on_disk or extra_on_disk or bad_scope or bad_ext:
        log("ABORT: 清单与磁盘不一致或越界/扩展名异常, 未删除任何文件")
        log(f"missing samples={missing_on_disk[:5]}; extra samples={extra_on_disk[:5]}")
        _write_log(args.execute)
        raise SystemExit(2)

    if not args.execute:
        log("dry-run done; no file deleted")
        _write_log(args.execute)
        return

    deleted = fail = freed = 0
    errors = []
    for p in exp_paths:
        try:
            sz = os.path.getsize(p)
            os.remove(p)
            deleted += 1
            freed += sz
        except Exception as e:
            fail += 1
            errors.append((p, f"{type(e).__name__}: {e}"))
    log(f"deleted={deleted} failed={fail} freed_bytes={freed}")
    if errors:
        log(f"delete errors sample={errors[:5]}")

    # 删除后校验
    remain = []
    for dp, _, fns in os.walk(RESULTS):
        for n in fns:
            if n.lower().endswith((".pth", ".pt")):
                remain.append(os.path.join(dp, n))
    log(f"remaining weights under results={len(remain)}")

    nonweight = []
    for dp, _, fns in os.walk(RESULTS):
        for n in fns:
            if not n.lower().endswith((".pth", ".pt")):
                nonweight.append(os.path.join(dp, n))
    log(f"non-weight files under results={len(nonweight)}")

    mism = missing = 0
    checked = 0
    for r in sha_rows:
        p = os.path.join(ROOT, r["path"].replace("/", os.sep))
        if not os.path.exists(p):
            missing += 1
            if missing <= 5:
                log(f"hash-check missing: {r['path']}")
            continue
        if os.path.getsize(p) != int(r["bytes"]):
            mism += 1
            log(f"size mismatch: {r['path']}")
            continue
        if sha256(p) != r["sha256"]:
            mism += 1
            log(f"sha mismatch: {r['path']}")
            continue
        checked += 1
    log(f"hash-check: checked={checked} mismatch={mism} missing={missing} of {len(sha_rows)}")

    # 其他目录权重统计 (不删除)
    other = {}
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "results")]
        for n in fns:
            if n.lower().endswith((".pth", ".pt")):
                top = os.path.relpath(dp, ROOT).split(os.sep)[0]
                b = os.path.getsize(os.path.join(dp, n))
                o = other.setdefault(top, {"files": 0, "bytes": 0})
                o["files"] += 1
                o["bytes"] += b
    log(f"other weights (kept): {json.dumps(other, ensure_ascii=False)}")
    log(f"end; elapsed={time.time() - t0:.1f}s")
    _write_log(args.execute)


def _write_log(execute):
    p = os.path.join(SNAP_DIR, f"delete_weights_{TS}{'_exec' if execute else '_dryrun'}.log")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print("LOG:", p)


if __name__ == "__main__":
    main()
