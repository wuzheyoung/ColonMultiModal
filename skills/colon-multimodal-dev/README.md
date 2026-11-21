# ColonMultiModal Agent — 经验库入口

本目录封装本项目全部历史问题与解决方案，供 agent 在任何会话中快速复用。

## 加载方式

- **DSH / Claude Code / Cline 等 agent**：按名称加载 skill
  - `skill colon-multimodal-dev`（本会话可用时）
  - 或直接读取：`skills/colon-multimodal-dev/SKILL.md`
- **规则引用**：在 agent 启动提示/规则文件中加一行：
  ```
  开始本项目的任何读取、写入、训练、排障工作前，先读 skills/colon-multimodal-dev/SKILL.md。
  ```

## 覆盖范围（章节速览）

| 章节 | 内容 |
|---|---|
| 0 环境事实 | 解释器/GPU/README 编码/训练命令模板 |
| 1 文件读取 | GBK/UTF-8、BOM、-1 标签、stale 报告、TS 目录语义、loader 缓存、DataLoader 计时、冒烟 cfg |
| 2 代码写入 | CRLF 与 edit、AST 验证、CLI 五处同步、f-string 反斜杠、python -c 引号 |
| 3 训练监控 | AMP 溢出、WATCHDOG、卡死判定、逐折分进程、巡检节奏 |
| 4 kfold | --kfold-resume 用法、清空重跑、折验证、判定门槛 |
| 5 报错速查表 | 10+ 条已知报错 → 根因 → 修复 |
| 6 流程纪律 | 心跳恢复、归档、诚实判定、不在汇报处停 |

## 维护规则

- 新增踩坑 → 追加到对应章节或速查表，格式：症状 → 根因 → 修复 → 如何避免
- 纯文档改动无需冒烟验证
