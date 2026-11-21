# -*- coding: utf-8 -*-
"""汇总 CLEANUP_LOG.md (只汇总已有事实)。"""
import os
import time

ROOT = r"d:\PyFile\ColonMultiModal"
OUT = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "outputs")
SNAP = os.path.join(OUT, "snapshot")
CLEAN = os.path.join(OUT, "cleanup_20261121")
WORK = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "work")

TIMELINE = [
    ("15:04:01", "仓库盘点（全仓体积/文件数、重复、空目录、零字节、pyc）", "37,796 文件", "144,622,323,978 B（134.69 GiB）",
     "生成 inventory.json / tables.md；发现 12 组重复、23 个 __pycache__、41 个零字节文件",
     "work/inventory.py；work/make_tables.py", "work/inventory.json（generated 字段）"),
    ("15:19:43–15:21:29", "results 文本/指标快照导出（共 4 次，权威版本 _152127）", "123 run / 228 fold；5 文件", "555,765 B",
     "JSON 严格合法；805 个非权重文件 SHA256 入库；失败 run 2（run_93 ring_c/t，保留空行）",
     "work/export_results_snapshot.py", "snapshot/export_20261121_152127.log"),
    ("15:52:49–15:52:53", "删除 results/ 全部权重（先 dry-run 后执行）", "127 个 .pth", "56,595,615,574 B（52.71 GiB）",
     "清单与磁盘 127/127 一致；805/805 非权重哈希通过；剩余权重 0",
     "work/delete_results_weights.py", "snapshot/delete_weights_20261121_155253_exec.log"),
    ("16:28:58–16:28:59", "删除 checkpoints_hos_full / checkpoints_hos / archive / logs 权重", "188 个 .pth",
     "30,231,267,468 B（28.15 GiB）", "非权重 784 个数量/字节/mtime 未变；pretrained 13 文件未变；146 个 ckpt 无法映射 run",
     "work/delete_ckpt_weights.py", "snapshot/delete_ckpt_weights_20261121_162858_exec.log"),
    ("16:29:27–16:29:29", "阶段 1：删除 .pyc 与零字节失败日志", "144 个（132 pyc + 12 日志）", "984,013 B",
     "清理空 __pycache__ 24 个；无错误", "work/cleanup_stages123.py", "cleanup_20261121/stage1_20261121_162927.log"),
    ("16:29:31", "阶段 2：根目录一次性脚本与 gbk 备份归档", "5 个", "135,953 B（移动，未释放）",
     "移至 archive/cleanup_20261121/20261121_162927/；无 import/运行引用，仅 FIXES 历史提及",
     "work/cleanup_stages123.py", "cleanup_20261121/stage2_20261121_162927.log"),
    ("16:29:31", "阶段 3：archive 内重复 PNG 去重", "11 组 / 11 个", "920,777 B（移动，未释放）",
     "保留 plots_early（及 TCGA fold1），移走 results_early（fold2）到 duplicates/；2 组重复 ckpt 已随权重删除跳过",
     "work/cleanup_stages123.py", "cleanup_20261121/stage3_20261121_162927.log"),
    ("16:30:43", "路径引用改写（README/docs/tools/config）", "26 条", "0 B",
     "26 条改写；残留 26 条全部标注故意保留；待改写 0；py_compile 通过",
     "work/rewrite_refs.py", "cleanup_20261121/rewrite_refs_20261121_163043.log"),
    ("16:33:48", "阶段 4：四份历史文档顶部统一横幅（方案 B，不合并）", "4 份文档", "0 B（净增约 1.5 KB）",
     "旧横幅合并为 4 行；正文 SHA256 4/4 未变；相对链接 25/25 有效",
     "work/stage4_banners.py", "cleanup_20261121/stage4_20261121_163348.log"),
]

CATS = [
    ("data", "30.402 GiB", "33,992"),
    ("pretrained", "16.215 GiB", "198"),
    ("cache_hos", "3.784 GiB", "544"),
    ("archive", "2.192 GiB", "224"),
    ("cache_crcfound", "1.133 GiB", "145"),
    ("plots_hos_full", "0.057 GiB", "65"),
    ("results", "0.026 GiB", "913"),
    ("logs", "0.011 GiB", "553"),
    ("2026-11-21（本轮产物）", "0.004 GiB", "111"),
    ("third_party / .git / docs / tools / 其余", "约 0.007 GiB", "608"),
]


def fsize(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def index_rows():
    rows = []
    files = []
    for base in (SNAP, CLEAN, OUT):
        if not os.path.isdir(base):
            continue
        for n in os.listdir(base):
            p = os.path.join(base, n)
            if os.path.isfile(p):
                files.append(p)
    files += [os.path.join(WORK, n) for n in ("inventory.json", "tables.md")]
    files = sorted(set(files), key=lambda p: (os.path.getmtime(p), p))
    for p in files:
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        st = os.stat(p)
        rows.append((time.strftime("%H:%M:%S", time.localtime(st.st_mtime)), rel, fsize(p)))
    return rows


def purpose(name):
    if name.startswith("snapshot_") and name.endswith(".json"):
        return "快照主文件（run/fold/失败/provenance）"
    if name.startswith("run_summary_"):
        return "run 级指标 CSV"
    if name.startswith("fold_detail_"):
        return "fold 级指标 CSV"
    if name.startswith("sha256_"):
        return "非权重文件 SHA256 清单"
    if name.startswith("export_"):
        return "快照导出日志"
    if name.startswith("weights_to_delete_"):
        return "results 权重删除清单"
    if name.startswith("delete_weights_"):
        return "results 权重删除日志"
    if name.startswith("weights_ckpt_archive_logs_to_delete_"):
        return "四目录权重删除清单"
    if name.startswith("delete_ckpt_weights_"):
        return "四目录权重删除日志"
    if name.startswith("nonweights_predelete_"):
        return "删除前非权重状态"
    if name.startswith("stage1_delete_manifest_"):
        return "阶段1 删除清单"
    if name.startswith("stage1_"):
        return "阶段1 日志/git 状态"
    if name.startswith("stage2_"):
        return "阶段2 日志/git 状态"
    if name.startswith("stage3_"):
        return "阶段3 日志/git 状态"
    if name.startswith("stage4_banner_table_"):
        return "阶段4 横幅对照表"
    if name.startswith("stage4_"):
        return "阶段4 日志"
    if name.startswith("path_rewrite_table_"):
        return "路径改写对照表"
    if name.startswith("residual_refs_"):
        return "残留引用分类表"
    if name.startswith("rewrite_refs_"):
        return "路径改写日志"
    if name.startswith("gitstatus_final_"):
        return "最终 git status"
    if name.endswith("repo-cleanup-plan.md"):
        return "仓库整理方案报告"
    if name.endswith("results-reusability-plan.md"):
        return "results 可复用性报告"
    if name == "inventory.json":
        return "盘点原始数据"
    if name == "tables.md":
        return "盘点表格"
    return "产物"


def main():
    rows = index_rows()
    L = []
    L.append("# 仓库整理全流程日志（2026-11-21）\n")
    L.append("> 本文只汇总已发生的动作与已有日志，不补写未执行步骤；时间取自日志/文件时间戳，个别条目为时间范围。\n")
    L.append("> 口径：体积按 GiB（1024³）；“释放”仅指删除，“移动”单独标注。\n")
    L.append("## 1. 时间线\n")
    L.append("| 时间 | 动作 | 文件数 | 字节 | 校验结果 | 执行脚本 | 原始日志 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in TIMELINE:
        L.append("| " + " | ".join(r) + " |")
    L.append("\n## 2. 体积汇总\n")
    L.append("| 项 | 数值 |")
    L.append("|---|---|")
    L.append("| 初始仓库 | 144,622,323,978 B（134.69 GiB）/ 37,796 文件 |")
    L.append("| 累计删除释放 | **86,827,867,055 B（80.86 GiB）** |")
    L.append("| 当前仓库 | **57,797,931,607 B（53.83 GiB）/ 37,413 文件** |")
    L.append("| 净减少 | 86,824,392,371 B（80.85 GiB）；与释放差 3,474,684 B 为本轮新增清单/快照/日志 |")
    L.append("| 移动未释放 | 1,056,730 B（阶段2 135,953 + 阶段3 920,777） |")
    L.append("\n### 当前分类体积（按顶层）\n")
    L.append("| 分类 | 体积 | 文件数 |")
    L.append("|---|---|---|")
    for c in CATS:
        L.append("| `" + c[0] + "` | " + c[1] + " | " + c[2] + " |")
    L.append("\n## 3. 校验记录汇总\n")
    L.append("| 校验 | 结果 |")
    L.append("|---|---|")
    L.append("| results 非权重 SHA256 | 805/805 通过；不一致 0；缺失 0 |")
    L.append("| results 删除后残留权重 | 0 |")
    L.append("| checkpoints/archive/logs 非权重 | 784 个，数量/字节/mtime 未变；缺失/新增/变化均 0 |")
    L.append("| pretrained 前后 | 13 文件 / 7,023,152,510 B（6.54 GiB），完全一致；顶层 4 个为 4.62 GiB，另有 9 个 TotalSegmentator 嵌套权重 |")
    L.append("| 快照 JSON | 严格合法（allow_nan=False）；123 run / 228 fold |")
    L.append("| 阶段4 正文 | 4/4 SHA256 一致，正文改动 0 |")
    L.append("| 文档链接 | 25/25 相对链接有效 |")
    L.append("\n## 4. 未执行与遗留\n")
    L.append("| 项 | 状态 |")
    L.append("|---|---|")
    L.append("| P0 正式训练（3 seed × 5 fold） | 本轮未启动；仓库内 P0 OOF 为此前产物 |")
    L.append("| Hos 数据内部处置 | 待单独批准（数据侧负责） |")
    L.append("| git 提交 | 未提交；仓库仅 36 个文件受跟踪，当前 81 行未提交状态 |")
    L.append("| 无法映射到 run 的 ckpt | 146 / 188（已随权重删除，manifest 留路径/时间） |")
    L.append("| 文档合并、archive 拍平、脚本同构合并 | 未做（阶段4 方案 B 仅加横幅） |")
    L.append("| 重复 PNG / 阶段2 文件 | 已移至 `archive/cleanup_20261121/`，未删除 |")
    L.append("\n## 5. 附录：日志与清单索引（按时间排序）\n")
    L.append(f"共 {len(rows)} 个文件。\n")
    L.append("| # | 时间 | 路径 | 字节 | 用途 |")
    L.append("|---|---|---|---|---|")
    for i, (t, p, s) in enumerate(rows, 1):
        L.append(f"| {i} | {t} | `{p}` | {s} | {purpose(os.path.basename(p))} |")
    body = "\n".join(L) + "\n"
    p = os.path.join(OUT, "CLEANUP_LOG.md")
    open(p, "w", encoding="utf-8").write(body)
    print(p, len(body.splitlines()), len(body.encode("utf-8")), len(rows))


if __name__ == "__main__":
    main()
