---
name: colon-multimodal-dev
description: "ColonMultiModal 项目（CRC TNM 分期，PyTorch 3D 医学影像，RTX4060 8GB）开发中积累的全部问题→解决方案经验库。覆盖：文件读取与编码陷阱（GBK/UTF-8 混合、CRLF、nibabel 目录语义、缓存污染）、代码写入与编辑流程（edit 多行匹配失败、AST 验证、冒烟脚本模式）、训练运行与监控（AMP 溢出、WATCHDOG 卡死判定、BATCH_TIMEOUT_S）、kfold 逐折正确用法、TotalSegmentator/nnU-Net/MedicalNet 集成坑、排程与心跳纪律。当在本项目读取文件、修改代码、启动/巡检训练、遇到报错时，先查本 skill 对应章节。"
license: MIT
type: tool
tags: ["pytorch", "medical-imaging", "windows", "3d-cnn", "experience", "troubleshooting"]
---

# ColonMultiModal 开发经验库（问题 → 解决方案）

> 本 skill 沉淀自本项目长期开发的真实踩坑记录。**每个条目都是"当时发生了什么 → 为什么 → 怎么修/怎么避免"**。改代码、读文件、跑训练前先扫一遍相关章节。

## 0. 环境事实（先读这个，很多"问题"其实只是环境特性）

- **Python 解释器**: `D:\Anaconda\envs\Pytorch\python.exe`（torch 2.5.1+cu124，CUDA 可用）
- **GPU**: RTX 4060 8GB → 显存预算紧；大模型（medicalnet18+病灶分支）约 4.6GB
- **工作目录**: `D:\PyFile\ColonMultiModal`；所有训练必须在该目录下以 Pytorch env 启动
- **README.md 是 UTF-8+GBK 混合编码**：用 `Add-Content -Encoding Default`（GBK）追加，**避免 `±`/`³` 等非 GBK 字符**；用 Select-String 检索
- **训练命令模板**（后台 pwsh + 过滤噪声输出）:
  ```powershell
  $env:BATCH_TIMEOUT_S='120'
  & 'D:\Anaconda\envs\Pytorch\python.exe' train_hos_seeds.py ... 2>&1 |
    Where-Object { $_ -notmatch 'tensorflow|oneDNN|it/s|B/s|\d+%\|' } | Select-Object -Last 30
  ```
  后台运行用 `run_in_background: true`，返回 job id。

## 1. 文件读取：编码 / 结构 / 时序陷阱

- **GBK 输出乱码 ≠ 文件坏了**：PowerShell 控制台默认 GBK，读 UTF-8 文件会乱码；用 read 工具或 `Get-Content -Encoding UTF8`。
- **CSV 带 BOM**：`csv.DictReader` 第一列列名会带 `\ufeff` → 用 `open(..., encoding='utf-8-sig')`。
- **`-1` 标签污染**：分支 CSV 含 `-1` 占位标签，必须过滤后再喂 loader，否则权重/损失全错。
- **stale 报告污染**：`eval_report_full_*.json` 若已存在，`_load_previous_result` 会读旧结果 → 重新生成前**先删除对应 json**，或改用 log 回退解析。
- **TotalSegmentator 2.x 输出是"目录语义"**：`{pid}.nii.gz/colon.nii.gz`（每样本一个目录），不是单文件。读 mask 必须按目录结构；统计时 `nibabel` 读空文件报 `ImageFileError: Empty file` → 先查 `{pid}/colon.nii.gz` 是否存在。
- **缓存命中分支的变量**：loader `__getitem__` 缓存命中（`if pid in cache`）分支里，若新加的变量（如 `mri_lesion_has`）只在未命中分支初始化 → `UnboundLocalError`。缓存分支也要初始化或调同一加载函数。
- **DataLoader fetch 计时**：`for bidx, batch in enumerate(loader)` 只测索引时间（≈0s）；**必须 `it=iter(loader); t0=time.time(); batch=next(it)`** 才能测到真实 fetch（含 `__getitem__`+增强）。
- **冒烟脚本的 cfg**：直接 `from config.hos import HosConfig` 会拿到**模块**而非实例 → 用 `HosConfig()`；冒烟若不设 `cfg.backbone_name`（如 `"medicalnet18"`），build_backbone 默认 resnet18 会**触发 monai 联网下载 MedicalNet 权重而失败** → 冒烟脚本必须显式设 backbone_name 走本地 `pretrained/`。

## 2. 代码写入与编辑：正确姿势

- **CRLF 行尾文件**：`edit` 工具多行匹配会因 `\r\n` 失败 → 改用 python 脚本做 CRLF-aware 替换（读 bytes → replace → 写回，保留 `\r\n`）。
- **每次改完先 AST 验证**：`python -c "import ast; ast.parse(open('文件',encoding='utf-8').read()); print('AST-OK')"`。
- **插入位置破坏语法**：在函数体中间插入代码块时，缩进/位置错会静默破坏语法 → 插完立即 AST + `--help` 验证 CLI 参数可见。
- **CLI 三处调用要同步**：train_hos_seeds.py 的 main 有三个调用点（eval-only / kfold / 逐 seed）+ 三个 runner 签名（eval_existing/run_seed/run_kfold）+ `_apply_run_overrides`，**加新参数必须五处同步**（build_model getattr、runner 签名、调用、CLI、config 默认值）。
- **f-string 内反斜杠**：`f"路径\{seq}"` 在 Python 3.12 前非法（`SyntaxError: f-string expression part cannot include a backslash`）→ 用拼接或先取变量。
- **`python -c "..."` 引号地狱**：多行 python -c 带内部引号/反斜杠极易在 PowerShell 层出错 → 长逻辑写临时 `.py` 脚本文件再执行，别内联。
- **`torch.load` FutureWarning（weights_only）**：无害警告，可忽略；加载项目内自产权重安全。
- **垃圾临时文件**：`_wire_*.py`/`_smoke_*.py` 排障脚本完成后清理或移入 `tools/`。

## 3. 训练运行与监控

- **AMP fp16 梯度溢出 = "T 头不学习"**：症状 loss 降但 T acc 卡死。根因 fp16 下主损失梯度溢出。修复：**head 强制 FP32**（`torch.autocast(device_type="cuda", enabled=False)` + 显式 `.float()`）、GeM 在 FP32 计算后恢复 dtype、`GradScaler("cuda", init_scale=2.0**10)`（torch 2.5.1 无 max_scale 参数；growth_interval=2000 ≈ 55ep 后停止升 scale）。
- **GPU 硬卡死（环境问题）**：症状 GPU≈1% 但显存占用、CPU 满载、日志静止。**Watchdog 兜底**：`BATCH_TIMEOUT_S`（默认 120）daemon 线程检查 `_last_progress`，超时打印 `[WATCHDOG] 强制退出` 并 `os._exit(1)`；训练循环内 `t_fetch`/`_b_dt` 超时抛 RuntimeError。
- **卡死判定标准（别误杀）**：单次低 GPU%（epoch 间验证/保存）不是卡死；**日志 mtime ≥5min 静止 + GPU≈0% 才算**。训练中检查 = 日志尾行时间 + nvidia-smi。
- **单进程连续多任务易触发第二任务早期 GPU 硬卡死**（环境缺陷）→ **逐折分进程**：一次只跑一个 fold/任务，串行接力。
- **训练日志校验**：启动后 60s 内确认 `pretrain_msg`（如 "102/102"）+ 首个 `[Train] Epoch 001` 出现 + GPU 显存占用正确；否则立刻停。
- **巡检节奏**：不忙轮询；靠心跳 schedule（训练期 2100s / 暂停期 604800s）和到点提醒兜底；阶段完成→当场启动下一项或挂提醒，**不许停在汇报处等**。

## 4. kfold 逐折正确用法（重点，踩过大坑）

- **必须带 `--kfold-resume --kfold-max-folds 1`**：不带 `--kfold-resume` 每次运行都从 fold 0 开始，会**重复训练同一折**（5 次全写 `eval_report_full_k0.json`，白跑 2h）；不带 `--kfold-max-folds 1` 会尝试跑全部折（触发单进程多任务卡死）。
- **接力模式**：每折一个后台 job，同一条命令重复执行；resume 靠 `eval_report_full_k{fold}.json` 存在性跳折。
- **混合配置污染**：换配置重跑前**必须清空** `logs/eval_report_full_k0..k4.{json,txt}` + `kfold_summary_full.json`（单 seed 报告 `s*` 保留），否则 resume 跳过用旧配置结果。
- **折完成验证**：检查该折 `eval_report_full_k{fold}.json` 的 `meta.time`（应为本次运行时间）与 `meta.val_csv` 折号；汇总看 `kfold_summary_full.json` 的 aggregate mean/std。
- **判定门槛**（本项目惯例）：单 seed ≥+0.03 记疑似突破（不中断）；≥+0.05 需 5-fold 验证；5-fold 判定 N mean≥0.559 且 ≥3 折胜基线才算真突破。**如实判定，不达标就写"未达门槛"**。

## 5. 已知报错速查表

| 报错 | 根因 | 修复 |
|---|---|---|
| `NameError: build_eval_report not defined` | train_hos_seeds.py 缺 imports | 加 `from utils import setup_logger, calibrate_binary_threshold, build_eval_report, save_eval_report` |
| `ModuleNotFoundError: No module named 'utils.losses'` | compute_class_weights_n 在 **train_hos_seeds.py 本地定义**，不在 utils.losses | `from train_hos_seeds import compute_class_weights_n`，且**传 DataFrame 非 ndarray** |
| `could not broadcast input array from shape (0,0,0)` | 极小/边缘 mask 重采样后 `argwhere` 空 → `cen=nan→int` 巨大负数→空切片 | `idx.size==0 or z1<=z0` 时返回 `(None, False)`（零卷+存在标志，不丢样本） |
| `Invalid device type: 'cuda'`（TotalSegmentator） | TS 2.x 设备参数改版 | 用 `device='gpu'`（不是 cuda）；CLI choices 改 `["gpu","cpu"]` |
| `_PicklingError ... lambda`（nnU-Net Windows） | Windows spawn 无法 pickle lambda | 单进程逐样本流水线：`pool=None`，`preprocess_save_to_queue` 逐样本 yield |
| `AttributeError: 'Trainer' object has no attribute 'num_classes'` | 方法定义被误放进 `__init__`（缩进错误） | 把 `_watchdog_loop` 移到类方法位置；改完 AST + 冒烟 |
| `ImageFileError: Empty file` | mask 按目录语义读错（读到了目录而非文件） | 读 `{pid}/colon.nii.gz` |
| `KeyError: 'patient_id'` | CSV 列名不匹配（可能 BOM） | 打印列名核对；用 utf-8-sig |
| torch 报 `No module named 'huggingface_hub...'` | monai 尝试联网下权重 | 设 `cfg.backbone_name` 走本地 pretrained/，或断网 |
| 训练中途 `[exit code: 1]` 无 traceback | WATCHDOG 强杀（硬卡死） | 产物通常完整（每 epoch 存档），按卡死处置 |

## 6. 流程纪律（排程 / 心跳 / 汇报）

- **用户规则**：遇问题先自我判断能否修复，**修不了才停并问**；每步总结再推进；"疑似突破"只记录不中断。
- **心跳恢复**：用户喊"开始训练"的同一时刻，把心跳 schedule 从 604800s 切回 **2100s** 高频巡检（文案含"自动推进/接力"），并挂到点提醒。
- **归档 + README**：每轮训练完成 → `results/round9/run_NN_<tag>/`（ckpt/curves/cm/json/txt/png/log）→ README 追加（GBK、避免 ±/³）。
- **结果判定要诚实**：不达标写"未达门槛/证伪"，不夸大；可疑单 seed 提升标注"待 5-fold 验证"。
- **不要在阶段完成后停在汇报处**：汇报后要么立即启动下一项，要么挂"到点检查→推进"提醒；没有 job 在跑时唯一唤醒机制是心跳/提醒，低频心跳下会静默挂 8h。

## 7. 本 skill 自身

- 本文件是"历史问题→解决方案"的持续沉淀。**新增踩坑时**：追加到对应章节（或第 5 表新增一行），条目格式 = 症状 → 根因 → 修复 → 如何避免。保持每条一行可扫。
- 本 skill 由 agent 维护，改完无需冒烟（纯文档）。
