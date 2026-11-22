# -*- coding: utf-8 -*-
"""
最终汇总: audit_tgr_planes_fusion.json — 汇总 task1/task2a/task2c/task3a/task3b 全部结果
读取各任务 json, 合并成一个结构化报告 + README 追加简短结论。
用法: python tools/audit_tgr_planes_fusion.py [--readme]
"""
import os, sys, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
OUT_JSON = os.path.join(LOGS, "audit_tgr_planes_fusion.json")
README = os.path.join(ROOT, "README.md")


def load(name):
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        return {"_missing": name}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    report = {
        "task1_tgr_split": load("audit_tgr_task1.json"),
        "task2a_planes": {"_note": "平面审计结论在 README 章节 (mri_t=矢状位 PIR / mri_c/d=真轴位 LPS), 无独立 json, 见 README 第三阶段补充②"},
        "task2c_axial_probe": load("audit_tgr_task2c.json"),
        "task3a_radiomics_fixed": load("audit_tgr_task3a.json"),
        "task3b_fusion": load("audit_tgr_task3b.json"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"DONE → {OUT_JSON}")

    if "--readme" in sys.argv:
        missing = [k for k, v in report.items() if "_missing" in v]
        lines = [
            "",
            "### 第三阶段终极核验汇总 (audit_tgr_planes_fusion.json)",
            f"- 状态: task1/2a/2c/3b 完成" + (f"; 缺失: {', '.join(missing)}" if missing else "; task3a 结果见其章节"),
            "- 核心结论: ①真轴位(mri_c/d)冻结 ConvNeXt sex OOF 0.84-0.86 ≥0.80 → 特征管线与 ID 映射无误 ②T 分期有真实信号: 真轴位 direct 0.6532 / 三路晚融合 0.6547 ③N 分期在任何口径下均随机(0.39-0.53) ④TRG/nCRT 无信号 ⑤96³ 变比例 Resize 伪影与全结肠无肿瘤定位是 T 信号偏弱的主因",
            "",
        ]
        with open(README, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("README 已追加")


if __name__ == "__main__":
    main()
