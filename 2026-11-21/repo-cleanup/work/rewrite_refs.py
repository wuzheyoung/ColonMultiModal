# -*- coding: utf-8 -*-
"""改写本轮移动/删除产生的失效路径引用; 输出逐条对照表。"""
import csv
import os
import time

ROOT = r"d:\PyFile\ColonMultiModal"
OUT = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "outputs", "cleanup_20261121")
TS = time.strftime("%Y%m%d_%H%M%S")
MARK = "2026-11-21 权重清理"
rows = []


def edit(path, pairs, category):
    p = os.path.join(ROOT, path)
    lines = open(p, encoding="utf-8").read().splitlines()
    out = []
    changed = False
    for i, line in enumerate(lines, 1):
        new = line
        for old, rep in pairs:
            if old in new and rep not in new:
                new = new.replace(old, rep)
        if new != line:
            out.append(new)
            rows.append([path, i, line.strip()[:240], new.strip()[:240], category])
            changed = True
        else:
            out.append(line)
    if changed:
        open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    return changed


def insert_after(path, anchor, text, category):
    p = os.path.join(ROOT, path)
    lines = open(p, encoding="utf-8").read().splitlines()
    if any(MARK in x for x in lines):
        return False
    for i, line in enumerate(lines, 1):
        if anchor in line:
            lines[i:i] = text
            open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
            rows.append([path, i, line.strip()[:240], " + ".join(t.strip() for t in text)[:240], category])
            return True
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    # README
    insert_after("README.md", "实验可移回原名目录", [
        "",
        f"> **{MARK}**：`checkpoints_hos_full/`、`checkpoints_hos/`、`archive/`、`logs/` 下的实验",
        "> `.pth/.pt` 已全部删除（`pretrained/` 未动）；下方涉及 `.pth` 的命令不可直接执行，",
        "> 需重跑生成权重。旧指标已固化到 `2026-11-21/repo-cleanup/outputs/snapshot/`。",
    ], "删除")
    edit("README.md", [
        ("├── checkpoints_hos_full/   # 当前 full 分支 ckpt（best T/N + snapshot）",
         "├── checkpoints_hos_full/   # 空目录（权重已于 2026-11-21 删除，需重跑生成）"),
        ("├── results/round9/         # 每轮归档（run_NN_<tag>，ckpt+图+报告 8 件套；编号原子占位防撞号）",
         "├── results/round9/         # 每轮归档（run_NN_<tag>，报告/图保留；权重已于 2026-11-21 删除）"),
        ("├── checkpoints_early/       # 早期无后缀 ckpt（s0/s1/s7/s42）",
         "├── checkpoints_early/       # 早期 ckpt 目录（权重已于 2026-11-21 删除）"),
        ("├── backup_round5/           # round5 mri3 备份",
         "├── backup_round5/           # round5 mri3 备份（权重已删除）"),
    ], "删除")
    insert_after("README.md", "python tools/_ensemble_eval.py --ckpts", [
        "    > 该命令引用的权重已于 2026-11-21 删除，不可直接执行；需先重跑生成。",
    ], "删除")

    # PROJECT_OVERVIEW
    insert_after("docs/PROJECT_OVERVIEW.md", "已过时", [
        f"> **{MARK}**：本文涉及的实验 `.pth/.pt`（checkpoints_* / archive / logs）已于 2026-11-21 删除；",
        "> 目录树中的权重目录现为空，相关命令需重跑后才能执行。",
    ], "删除")
    edit("docs/PROJECT_OVERVIEW.md", [
        ("├── checkpoints_hos_full/  full 分支 best T/N + snapshot（多 seed）",
         "├── checkpoints_hos_full/  空（权重已于 2026-11-21 删除）"),
        ("│   ├── checkpoints_early/checkpoints_hos/           round5 及更早权重",
         "│   ├── checkpoints_early/checkpoints_hos/           round5 及更早权重（已删除）"),
    ], "删除")

    # PROGRESS
    edit("docs/PROGRESS_REPORT_HOS.md", [
        ("（权重已移入 `archive/checkpoints_early/checkpoints_hos/`，`hos_ct_test.csv` 不存在）。",
         "（权重已于 2026-11-21 删除，`hos_ct_test.csv` 不存在）。"),
    ], "删除")
    insert_after("docs/PROGRESS_REPORT_HOS.md", "python tools/_ensemble_eval.py --ckpts", [
        f"> 注（{MARK}）：该命令引用的权重已删除，不可直接执行；需先重跑生成。",
    ], "删除")

    # FILE_INVENTORY
    insert_after("docs/FILE_INVENTORY.md", "统计已过时", [
        f"> **{MARK}**：`checkpoints_hos/`、`archive/`、`logs/` 下的实验权重已于 2026-11-21 删除。",
    ], "删除")
    edit("docs/FILE_INVENTORY.md", [
        ("### checkpoints_hos/ — 权重", "### checkpoints_hos/ — 权重（已于 2026-11-21 删除）"),
        ("| `best_model_hos_s{42,0,1,7}.pth` | resnet10 T-best（162.8MB×4，round4） |",
         "| `best_model_hos_s{42,0,1,7}.pth` | resnet10 T-best（162.8MB×4，round4；已删除） |"),
        ("| `best_model_n_hos_s{42,0,1,7}.pth` | resnet10 N-best（round3 修复后新增） |",
         "| `best_model_n_hos_s{42,0,1,7}.pth` | resnet10 N-best（round3 修复后新增；已删除） |"),
    ], "删除")

    # FIXES top note
    insert_after("docs/FIXES_2026-11-20.md", "范围：`tools/` 全部脚本", [
        "",
        f"> **{MARK}**：本文提到的实验 `.pth/.pt` 已于 2026-11-21 删除；文中历史路径与命令",
        "> 仅为变更记录，不可直接执行，需重跑生成权重。",
    ], "删除")

    # Tools: 仅在默认读取已删权重的脚本顶部加说明
    tool_note = f"# {MARK}: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。"
    tools = ["ensemble_modalities.py", "crcfound_features.py", "radiomics_experiment.py",
             "roi_experiment.py", "visualize_cam.py", "probe_fold1.py", "_ensemble_eval.py",
             "ensemble_seeds.py", "eval_tta.py", "ensemble_snapshot.py"]
    for t in tools:
        p = os.path.join(ROOT, "tools", t)
        lines = open(p, encoding="utf-8").read().splitlines()
        if any(MARK in x for x in lines):
            continue
        # 插到首行编码声明之后
        idx = 1 if lines and lines[0].startswith("# -*-") else 0
        lines.insert(idx, tool_note)
        open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        rows.append([f"tools/{t}", idx + 1, "（新增说明）", tool_note, "删除"])

    # config
    p = os.path.join(ROOT, "config", "hos.py")
    lines = open(p, encoding="utf-8").read().splitlines()
    if not any(MARK in x for x in lines):
        for i, line in enumerate(lines):
            if "save_dir" in line and "checkpoints_hos" in line:
                lines.insert(i, f"    # {MARK}: 该目录权重已清空, 下次训练重新生成")
                rows.append(["config/hos.py", i + 1, line.strip(), "（新增说明）" + lines[i].strip(), "删除"])
                break
        open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    table = os.path.join(OUT, f"path_rewrite_table_{TS}.csv")
    with open(table, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "line", "original", "rewritten", "category"])
        w.writerows(rows)
    print(f"table={table} rows={len(rows)}")


if __name__ == "__main__":
    main()
