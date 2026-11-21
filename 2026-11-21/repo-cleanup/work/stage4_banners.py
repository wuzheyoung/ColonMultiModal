# -*- coding: utf-8 -*-
"""阶段4-B: 四份历史文档顶部统一横幅 (仅改横幅, 正文不动; 不合并文档)。"""
import csv
import hashlib
import os
import re
import time

ROOT = r"d:\PyFile\ColonMultiModal"
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "outputs", "cleanup_20261121")
TS = time.strftime("%Y%m%d_%H%M%S")

BANNER = [
    "> **阶段快照：{date} / {phase}** — 本文记录该阶段状态，内容不代表当前状态。",
    "> 当前状态见 [README.md](../README.md) 与 [FIXES_2026-11-20.md](FIXES_2026-11-20.md)；"
    "数据规模与口径已变更，文中数字不可直接引用。",
    "> 2026-11-21 已删除全部实验权重（`checkpoints_*` / `archive` / `logs`），涉及 `.pth` 的命令需重跑生成。",
    "> 相关快照：[文件清单](FILE_INVENTORY.md) · [项目总览](PROJECT_OVERVIEW.md) · "
    "[进度汇报](PROGRESS_REPORT_HOS.md) · [优化路线图](OPTIMIZATION_ROADMAP.md)",
]

SPEC = {
    "FILE_INVENTORY.md": ("2026-09-29", "文件级索引"),
    "PROJECT_OVERVIEW.md": ("2026-09-29", "round4 归档后"),
    "PROGRESS_REPORT_HOS.md": ("2026-10-09", "154 样本阶段"),
    "OPTIMIZATION_ROADMAP.md": ("2026-10-09", "round6+ 规划"),
}


def body_hash(lines):
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def resolve_links(lines, path):
    ok, bad = [], []
    for i, l in enumerate(lines, 1):
        for m in re.finditer(r"\[[^\]]*\]\(([^)#]+)(#[^)]*)?\)", l):
            t = m.group(1).strip()
            if t.startswith(("http://", "https://", "mailto:")):
                continue
            p = os.path.normpath(os.path.join(os.path.dirname(path), t))
            (ok if os.path.exists(p) else bad).append((i, t))
    return ok, bad


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, (date, phase) in SPEC.items():
        p = os.path.join(DOCS, name)
        lines = open(p, encoding="utf-8").read().splitlines()
        title = lines[0]
        # 正文起点: 标题后第一个 `---` 或 `## `
        start = next(i for i in range(2, len(lines)) if lines[i].strip() == "---" or lines[i].startswith("## "))
        body = lines[start:]
        old_banner = lines[2:start]
        bh = body_hash(body)
        new = [title, ""] + [b.format(date=date, phase=phase) for b in BANNER] + [""] + body
        new_start = 1 + 1 + len(BANNER) + 1  # 0-based body start
        assert body_hash(new[new_start:]) == bh, name
        open(p, "w", encoding="utf-8").write("\n".join(new) + "\n")
        ok, bad = resolve_links(new, p)
        rows.append({
            "file": f"docs/{name}",
            "old_banner": f"3-{start}（{len(old_banner)} 行）",
            "new_banner": f"3-{3 + len(BANNER) - 1}（{len(BANNER)} 行）",
            "merged": "是（替换旧横幅）",
            "body_changes": 0,
            "body_sha_before": bh[:16],
            "body_sha_after": body_hash(new[new_start:])[:16],
            "links_ok": len(ok),
            "links_bad": ";".join(f"L{i}:{t}" for i, t in bad) or "无",
        })
    t = os.path.join(OUT, f"stage4_banner_table_{TS}.csv")
    with open(t, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(t)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
