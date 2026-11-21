# -*- coding: utf-8 -*-
"""生成 results 可复用性报告 (只读 CSV -> Markdown)。"""
import os

import pandas as pd

ROOT = r"d:\PyFile\ColonMultiModal"
WORK = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "work")
OUT = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "outputs", "results-reusability-plan.md")

d = pd.read_csv(os.path.join(WORK, "results_runs.csv"), encoding="utf-8-sig")
d["num"] = d["run"].str.extract(r"run_(\d+)").astype(int)
d = d.sort_values(["num", "run"])

rows = []
for _, r in d.iterrows():
    met = str(r["metrics"])
    met = met if len(met) <= 110 else met[:107] + "..."
    rows.append(f"| `{r['run']}` | {r['mb']} | {r['created'][5:16]} | {r['rules']} | "
                f"{r['reusable']} | {r['action']} | {met} |")
run_table = "\n".join(rows)

refs = d[d["readme_ref"].notna() & (d["readme_ref"].astype(str) != "")]
ref_rows = [f"| `{r['run']}` | {r['readme_ref']} | {r['rules']} | {r['action']} |"
            for _, r in refs.iterrows()]

body = f"""# results/ 逐 run 可复用性判定与处置方案（只读盘点 · 未执行）

> 盘点时间：2026-11-21｜范围：`results/`（必要时对照 `checkpoints_*`、`logs/`）｜体积按 MiB/GiB（1024 进制）。
> 约束：未删除、未移动、未搬运、未训练；判定基于文件名、报告 JSON/文本、OOF patient_id 与创建时间。
> 明细：`work/results_runs.csv`（逐 run 可勾选）、`work/results_audit.json`、`work/results_audit.py`。

## 0. 结论

| 事实 | 数值 |
|---|---|
| run 总数 | **123** |
| `results/` 总体积 | **52.73 GiB**（56,622,474,854 B） |
| 可复用 run | **0 个 / 0 B** |
| 不可复用 run | **123 个 / 52.73 GiB** |
| 其中权重（.pth/.pt） | **52.71 GiB（99.95%）** |
| 其中图片（.png/.jpg） | **25.7 MiB（0.05%）** |
| 其中文本（json/txt/log/csv） | **2.65 MiB（0.005%）** |
| 处置：保留（含文档引用/新口径 OOF） | 68 runs / 23,012.4 MiB |
| 处置：移出权重、文本留仓 | 55 runs / 30,987.0 MiB（其中权重 30.25 GiB） |

**核心判断**：按“纯 MRI + 无临床 + 单次划分 + 不受数据侧改动影响”的严格合取条件，**没有任何 run 可复用**。最主要的一票否决是判定条目 5（全仓无任何报告含末 5 轮/LOFO 字段）与条目 1（旧主链路全带 7 维临床）。可释放体积几乎全部是权重；文本与图片合计仅 28.4 MiB，建议留仓。

## 1. 判定规则命中统计

| 条目 | 命中 run 数 | 说明 |
|---|---|---|
| 1 含临床 | 112 | 除 11 个 P0 run 外，旧主模型/2.5D/MIL/ROI-DL 均为 7 维临床 |
| 2 ring/body | 17 | 名称含 `ring`/`body`，或 ring 变体 |
| 3 5 折 CV | 64 | 名称 `_k0..k4` 或含 ≥2 折 OOF；P0 虽用 Master 表，但仍为 5 折，按“单次划分”硬条件计为不合格 |
| 4 多 seed | 14 | 名称 `_s0`/`_s1`/`_s7` |
| 5 指标字段缺失 | **123** | `grep -r "last5\\|calT_last5\\|oof_last5" results/` 命中 0；旧报告无 PR-AUC/末 5 轮，P0 只有 OOF/日志、无汇总报告 |
| 6 数据侧改动 | 112 | 旧 run 全部早于 2026-11-21 数据修正（382 剔除、391 CT、1/2/4/5/6 CT 置零） |

说明：
- P0 的 11 个 run（117–129）为 `--no-clinical`、纯 `mri_t` 或 `mri_t_ring`，OOF 覆盖 280 例且不含 pid 382；pid 391 与 1/2/4/5/6 的改动均为 **CT**，对纯 MRI 输入不构成影响，故条目 6 不计入 P0。
- 但 P0 仍触发条目 3（5 折，不是单次划分）与条目 5（无汇总报告；虽有 `_last5.csv`，但无同时含 AUC/PR-AUC/末 5 轮的 JSON）。
- 旧纯 MRI run（`run_07`/`run_10` 的 `smmri`）仍触发 1（7 维临床）、5、6（旧队列），不可复用。

## 2. 可复用 run 显式结论

**可复用 run：无（0 个）。** 最接近的是 P0 的纯 MRI 臂，但均不满足“单次划分 + 报告字段齐全”：

| run | 体积 | 为何接近 | 为何仍不可复用 |
|---|---|---|---|
| `run_121_p0_C_bbox` | 0.02 MiB | 纯 MRI、无临床、Master 折表、无 CT 依赖 | 5 折（条目 3）；无汇总报告（条目 5） |
| `run_118_p0_B_sdf_bbox` | 0.02 MiB | 同上 | 同上 |
| `run_123_p0_E_nonly` / `run_124_p0_F_tonly` | 各 0.02 MiB | 同上 | 同上；且单任务消融不作为主结论 |
| `run_07_hos_full_s42_smmri` / `run_10_hos_full_s7_smmri` | 556.1 MiB 各 | 纯 MRI 单次划分 | 条目 1（7 维临床）、5、6 |

> P0 指标口径提醒：T 的 AP 约 0.75–0.81，而 T 阳性率 0.714，AP 不能单独当作有效性证据；P0 的 F1 均值仅 0.36–0.50（A/C 臂）。

## 3. 体积汇总

| 类别 | 体积 | 占 results/ |
|---|---|---|
| 总体积 | 52.73 GiB | 100% |
| 可复用 | 0 B | 0% |
| 不可复用 | 52.73 GiB | 100% |
| └ 权重 | 52.71 GiB | 99.95% |
| └ 图片 | 25.7 MiB | 0.05% |
| └ 文本 | 2.65 MiB | 0.005% |

处置体积：

| 处置 | run 数 | 体积 | 说明 |
|---|---|---|---|
| 保留 | 68 | 23,012.4 MiB | README 引用的 run + 11 个 P0 run（新口径 OOF 证据） |
| 移出权重、文本留仓 | 55 | 30,987.0 MiB | 其中权重 30.25 GiB；文本 1.19 MiB、图片 14.8 MiB 建议留仓 |

## 4. 处置清单（123 run）

> 完整指标摘要、权重/图片/文本拆分、README 引用行见 `work/results_runs.csv`。`可复用=否` 不代表必须删；`保留` 是按“文档引用/新口径证据”给出的处置，不是复用判定。

| run | MiB | 创建 | 触发条目 | 可复用 | 处置 | 指标摘要 |
|---|---|---|---|---|---|---|
{run_table}

## 5. 被引用的 run 与引用破坏

| 事实 | 说明 |
|---|---|
| README 显式引用 run 编号 | 68 个（见下表） |
| `docs/FIXES_2026-11-20.md` | 仅 `run_93`（D1 并发撞号案例，L160） |
| 其他 docs | `PROGRESS_REPORT_HOS.md` / `PROJECT_OVERVIEW.md` / `FILE_INVENTORY.md` 无 `run_NN` 引用 |
| README 已引用但不存在的 run | `run_105`–`run_112`、`run_120` 在当前 `results/round9` 中不存在（README 行 1530–1566、1662），引用已失效 |
| 本次移出权重会破坏的引用 | 55 个移出 run **均未被 README 按编号引用**；仅 `run_93_hos_full_ring_c` / `_ring_t` 被 FIXES L160 引用（删除目录会破坏该案例叙述） |

> 处理建议：保留 run 目录与文本，仅移出 `.pth`；README 的 run 编号引用只指向 metrics/报告，不指向权重路径，因此移出权重不破坏 README；FIXES L160 需补一句“权重已移出，文本/OOF 保留”。

### README 引用映射（68 run）

| run | README 行 | 触发条目 | 处置 |
|---|---|---|---|
{chr(10).join(ref_rows)}

## 6. G: 盘搬运方案（先移后删，未执行）

### 6.1 目录结构建议

```
G:\\ColonMultiModal_archive\\
  results_20261121\\
    round9\\
      run_043_hos_full_s42\\
        weights\\best_model_hos_full_s42.pth
        weights\\best_model_n_hos_full_s42.pth
      ...
  manifests\\
    results_runs.csv
    sha256_manifest.csv
    move_log_20261121.txt
```

- 只搬权重，不搬文本/图片；本地 run 目录保留 `*.json`、`*.txt`、`*.csv`、`*.png`。
- 每个 run 的 `weights/` 下保持原文件名，便于按 manifest 反向恢复。
- `manifests/` 与仓库内 `archive/manifests/results_20261121/` 各存一份。

### 6.2 搬运命令（建议）

```powershell
# 1) 先 dry-run 列出将移动的文件（不写盘）
robocopy "d:\\PyFile\\ColonMultiModal\\results\\round9" "G:\\ColonMultiModal_archive\\results_20261121\\round9" *.pth /S /MOV /L /R:1 /W:1 /LOG:"G:\\ColonMultiModal_archive\\manifests\\dryrun.log"

# 2) 确认后执行移动（保留目录结构）
robocopy "d:\\PyFile\\ColonMultiModal\\results\\round9" "G:\\ColonMultiModal_archive\\results_20261121\\round9" *.pth /S /MOV /R:1 /W:1 /LOG:"G:\\ColonMultiModal_archive\\manifests\\move.log"

# 3) 仅对 55 个移出 run 执行；保留 run 不参与
```

### 6.3 校验方式

| 校验 | 方法 | 通过标准 |
|---|---|---|
| 文件数/字节一致 | 搬运前对 55 个 run 的 `.pth` 生成 `path,size,sha256`；搬运后对 G: 同名文件重算 | 每个文件 SHA256 相同，数量与总字节一致 |
| 本地无残留权重 | `Get-ChildItem results\\round9 -Recurse -Filter *.pth` 按 manifest 过滤 | 55 个 run 的权重 0 个残留；保留 run 不受影响 |
| 无脚本引用 | `git grep -n "\\.pth"` + `Select-String -Path tools\\*.py,*.py -Pattern "results[/\\\\]round9"` | 仅注释/文档命中；无代码按路径读取 |
| JSON/文本可读 | 对保留的 `*.json` 逐个 `json.load` | 全部可解析 |
| 回退演练 | 从 G: 选 1 个 run 拷回原路径并校验 SHA256 | 字节一致、脚本可继续读 |

## 7. manifest 方案

### 7.1 每条 run 抽取字段

| 字段 | 来源 | 缺失处理 |
|---|---|---|
| run 路径、名称、体积、创建时间 | 文件系统 | — |
| 命令 | 日志中含 `python`/`train_` 的首行；或 run 目录 `cmd.txt` | 标 `未记录`（旧 run 多数无 argv） |
| seed / fold | run 名正则 `_s(\\d+)`、`_k(\\d)`；报告 meta | 标 `未知` |
| 指标 JSON | `eval_report*.json` / `kfold_summary*.json` 原样复制 | 标 `无报告` |
| 报告文本 | `*.txt` / `*.log` 原样复制或截取指标段 | 标 `无文本` |
| OOF | 若存在 `*_k*_s*.csv`，复制 | 标 `无 OOF` |
| 触发条目 | 本报告规则 1–6 | — |
| 处置与删除后果 | 本报告 | — |

### 7.2 体积估计

| 项 | 体积 |
|---|---|
| 55 个移出 run 的现有文本 | 1.19 MiB |
| OOF（P0 保留，不删） | <0.1 MiB |
| manifest CSV + SHA256 清单（55 run / ~110 权重） | 约 0.1 MiB |
| **合计** | **约 1.4 MiB** |

### 7.3 建议永久保留在仓库的小体积文本

| 类型 | 理由 |
|---|---|
| 全部 `eval_report*.json` / `kfold_summary*.json` | 唯一的结构化指标证据，合计 <1 MiB |
| 全部 `*.txt` / `*.log` | 命令、异常、epoch 曲线；文本总量 2.65 MiB |
| 全部 OOF `*.csv` | stacking/校准/亚组分析唯一可复用输入；P0 全套 <0.1 MiB |
| run 级图片（25.7 MiB） | 体积极小，保留可避免“只剩数字无图” |

## 8. 风险与回退

| 风险 | 说明 | 回退 |
|---|---|---|
| git 未跟踪 | 仅 36 个文件受 git 跟踪；results 全部未跟踪 | 依赖 G: 副本 + SHA256 manifest，不依赖 git |
| 引用破坏 | README 68 个编号引用；FIXES L160 引用 run_93 | 保留文本/报告与 run 目录；仅移权重，引用不受影响；FIXES 补注 |
| 旧结果不可复现 | 数据侧改动后旧 run 无法原样重跑 | manifest 保留指标与命令；权重在 G: 可恢复推理 |
| 搬运中断 | `robocopy /MOV` 中途失败可能半移动 | 先 `/L` dry-run；逐 run 搬运并记录；失败按日志回拷 |
| 误删保留 run | 55 个 run 与 68 个保留 run 同处 `round9` | 用 CSV `action` 列生成白名单，禁止整目录删除 |

## 9. 分阶段执行顺序

| 阶段 | 动作 | run/文件 | 体积 | 风险 |
|---|---|---|---|---|
| 0 | 生成 manifest 与 SHA256 清单；G: 建目录 | 55 run / ~110 权重 | 0 | 无 |
| 1 | dry-run + 逐 run 搬权重到 G: | 55 run | 30.25 GiB 移动 | 低；可回拷 |
| 2 | 校验 SHA256/数量/脚本引用 | 55 run | 0 | 无 |
| 3 | 观察 7 天后删除本地权重 | 55 run | 释放 30.25 GiB | 中；需确认校验通过 |
| 4 | 文档补注（FIXES L160；README 缺失 run 说明） | 2 文件 | 0 | 无 |

**不执行项**：本轮不搬运、不删除、不改 README/FIXES；`results/` 文本与图片全部保留。
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(body)
print(OUT, len(body.splitlines()), "lines")
