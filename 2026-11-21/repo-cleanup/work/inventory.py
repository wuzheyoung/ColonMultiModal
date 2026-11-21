# -*- coding: utf-8 -*-
"""只读仓库盘点: 体积/文件数分布、最大项、重复副本、空目录、零字节、__pycache__。
不修改任何文件; 输出 JSON/文本到 2026-11-21/repo-cleanup/work/。
"""
import hashlib
import json
import os
import time

ROOT = r"d:\PyFile\ColonMultiModal"
OUT = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "work")
SKIP_WALK = {".git"}  # .git 单独统计, 不参与候选扫描


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def sha256(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    top = {}
    total_files = total_bytes = 0
    all_dirs = []
    empty_dirs = []
    zero_byte = []
    pyc_files = []
    pycache_dirs = []
    docs = []
    root_scripts = []
    tools = []

    for name in os.listdir(ROOT):
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            s = os.path.getsize(p)
            top[name] = {"type": "file", "bytes": s, "files": 1}
            total_files += 1
            total_bytes += s
            if name.lower().endswith(".py"):
                root_scripts.append({"path": name, "bytes": s})
        elif os.path.isdir(p):
            nb = nf = 0
            for dp, dns, fns in os.walk(p):
                if name in SKIP_WALK:
                    dns[:] = [d for d in dns if d != ".git"]
                all_dirs.append(rel(dp))
                if dp.endswith("__pycache__"):
                    pycache_dirs.append(rel(dp))
                if not dns and not fns:
                    empty_dirs.append((rel(dp), dp))
                for fn in fns:
                    fp = os.path.join(dp, fn)
                    try:
                        s = os.path.getsize(fp)
                    except OSError:
                        continue
                    nb += s
                    nf += 1
                    total_files += 1
                    total_bytes += s
                    low = fn.lower()
                    if low.endswith((".pyc", ".pyo")):
                        pyc_files.append((rel(fp), s))
                    if s == 0:
                        zero_byte.append((rel(fp), s))
                    if dp == os.path.join(ROOT, "docs") and low.endswith(".md"):
                        docs.append({"path": rel(fp), "bytes": s,
                                     "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(fp)))})
                    if dp == ROOT and low.endswith(".py"):
                        root_scripts.append({"path": rel(fp), "bytes": s})
                    if dp == os.path.join(ROOT, "tools") and low.endswith(".py"):
                        head = ""
                        try:
                            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                                lines = [next(f, "") for _ in range(6)]
                            head = " ".join(x.strip() for x in lines if x.strip())[:220]
                        except OSError:
                            pass
                        tools.append({"path": rel(fp), "bytes": s, "head": head})
            top[name] = {"type": "dir", "bytes": nb, "files": nf}

    # 去重候选: 仅扫描代码/文档目录 (排除 data/cache/checkpoints/results/logs/.git)
    dup_roots = ["archive", "tools", "docs", "utils", "config", "models", "data_loader"]
    groups = {}
    root_files = [os.path.join(ROOT, f) for f in os.listdir(ROOT)
                  if os.path.isfile(os.path.join(ROOT, f))]
    for fp in root_files:
        try:
            s = os.path.getsize(fp)
        except OSError:
            continue
        if s == 0:
            continue
        groups.setdefault((s, os.path.basename(fp).lower()), []).append(fp)
    for dirname in dup_roots:
        base = os.path.join(ROOT, dirname)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in ("__pycache__", ".git")]
            for fn in fns:
                fp = os.path.join(dp, fn)
                try:
                    s = os.path.getsize(fp)
                except OSError:
                    continue
                if s == 0:
                    continue
                groups.setdefault((s, fn.lower()), []).append(fp)
    duplicates = []
    for (s, fn), paths in groups.items():
        if len(paths) < 2:
            continue
        hashed = {}
        for fp in paths:
            try:
                h = sha256(fp)
            except OSError:
                continue
            hashed.setdefault(h, []).append(fp)
        for h, same in hashed.items():
            if len(same) > 1:
                duplicates.append({"bytes_each": s, "sha256": h[:16],
                                   "paths": [rel(x) for x in sorted(same)]})
    duplicates.sort(key=lambda x: -x["bytes_each"] * (len(x["paths"]) - 1))

    # 最大文件 / 最大目录 (顶层 + archive 子层)
    largest_files = []
    for dp, dns, fns in os.walk(ROOT):
        if ".git" in dp.split(os.sep):
            continue
        for fn in fns:
            fp = os.path.join(dp, fn)
            try:
                largest_files.append((os.path.getsize(fp), rel(fp)))
            except OSError:
                pass
    largest_files.sort(reverse=True)

    top_list = [{"name": k, **v, "pct": (v["bytes"] / total_bytes * 100) if total_bytes else 0}
                for k, v in sorted(top.items(), key=lambda kv: -kv[1]["bytes"])]

    report = {
        "root": ROOT,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totals": {"files": total_files, "bytes": total_bytes,
                   "gb": round(total_bytes / 1024 ** 3, 2)},
        "top_level": top_list,
        "largest_files": [{"bytes": s, "mb": round(s / 1024 ** 2, 1), "path": p} for s, p in largest_files[:15]],
        "duplicates": duplicates,
        "dup_waste_bytes": sum(d["bytes_each"] * (len(d["paths"]) - 1) for d in duplicates),
        "empty_dirs": [x[0] for x in empty_dirs],
        "zero_byte_files": [x[0] for x in zero_byte],
        "pycache": {"dirs": len(pycache_dirs), "files": len(pyc_files),
                    "bytes": sum(s for _, s in pyc_files)},
        "docs": sorted(docs, key=lambda x: -x["bytes"]),
        "root_py": sorted(root_scripts, key=lambda x: -x["bytes"]),
        "tools_py": sorted(tools, key=lambda x: -x["bytes"]),
        "pycache_dirs": sorted(pycache_dirs),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "inventory.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "empty_dirs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report["empty_dirs"]))
    with open(os.path.join(OUT, "zero_byte.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report["zero_byte_files"]))
    with open(os.path.join(OUT, "pyc_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"{p}\t{s}" for p, s in pyc_files))
    print(json.dumps({"totals": report["totals"],
                      "top5": report["top_level"][:5],
                      "dup_groups": len(duplicates), "dup_waste_mb": round(report["dup_waste_bytes"] / 1024 ** 2, 2),
                      "empty_dirs": len(empty_dirs), "zero_byte": len(zero_byte),
                      "pycache": report["pycache"]}, ensure_ascii=False, indent=2))
    print(f"elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
