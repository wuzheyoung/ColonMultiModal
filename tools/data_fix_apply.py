# -*- coding: utf-8 -*-
"""m05147 阶段2: 数据勘误安全落盘 (2026-11-23 用户"确认写入")

约束:
1. 病理 T/N 变更为 0 处 → master_5fold_356.csv 完全不修改 (5折/标签 100% 零变动)
2. 临床 7 维仅更新 pid372/395/402 共 7 处原始值 → data/hos_pairs_raw.csv
   - 防污染: 10 处 ca199 '<2' 保持解析为 2.0, 87 处 age 'XX岁' 保持为数值, 严禁误写 NaN
3. 新建 data/direct_cohort_ids.csv (仅 pid/is_tgr 两列, 锁死 direct_all=280/ncrt=75/direct_tcd=162)
4. 变更明细落盘 logs/data_fix_diff.json (无隐私字段) + README.md 简记
"""
import os, json, shutil, datetime
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
MASTER = os.path.join(ROOT, "data", "master_5fold_356.csv")
COHORT_OUT = os.path.join(ROOT, "data", "direct_cohort_ids.csv")
DIFF_OUT = os.path.join(ROOT, "logs", "data_fix_diff.json")
README = os.path.join(ROOT, "README.md")
BACKUP_DIR = os.path.join(ROOT, "data", "_csv_backup_20261123_1350")

# 7 处临床修正 (来自阶段1 diff, 用户已确认)
CLIN_FIXES = {
    372: {"ca199": 8.56, "cea": 1.25},
    395: {"prealbumin": 241.0, "hb": 139.0},
    402: {"albumin": 41.0, "prealbumin": 182.0, "hb": 148.0},
}

def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------- 0. 快照 master (只读校验, 绝不写入) ----------
    master = pd.read_csv(MASTER)
    master_snapshot = master.copy()
    n_master = len(master)
    assert n_master == 355, f"master rows {n_master} != 355"

    # ---------- 1. 备份 + 更新 raw 临床表 ----------
    raw = pd.read_csv(RAW)
    shutil.copy2(RAW, os.path.join(BACKUP_DIR, f"hos_pairs_raw_{ts}.csv"))
    before = raw.set_index("patient_id")

    # 防污染基线: 写入前记录
    ca199_eq2_before = int((raw["ca199"] == 2.0).sum())
    age_nonnull_before = int(raw["age"].notna().sum())
    age_isnum_before = int(pd.to_numeric(raw["age"], errors="coerce").notna().sum())

    changed = []
    for pid, fixes in CLIN_FIXES.items():
        mask = raw["patient_id"] == pid
        assert mask.sum() == 1, f"pid {pid} rows {mask.sum()}"
        for col, new in fixes.items():
            old = before.loc[pid, col]
            raw.loc[mask, col] = new
            changed.append({"pid": pid, "col": col,
                            "old": None if pd.isna(old) else float(old), "new": float(new)})

    raw.to_csv(RAW, index=False)

    # 防污染校验 (写入后)
    ca199_eq2_after = int((raw["ca199"] == 2.0).sum())
    age_nonnull_after = int(raw["age"].notna().sum())
    age_isnum_after = int(pd.to_numeric(raw["age"], errors="coerce").notna().sum())
    assert ca199_eq2_after == ca199_eq2_before == 10, f"ca199 <2 解析被破坏: {ca199_eq2_before}->{ca199_eq2_after}"
    assert age_isnum_after == age_isnum_before, "age 数值化被破坏"
    assert age_nonnull_after == age_nonnull_before, "age 非空计数变化"
    # 仅校验实际写入的 7 个单元格非空 (其余列 NaN 属正常缺失, 不在修正范围)
    raw_check = raw.set_index("patient_id")
    for pid, fixes in CLIN_FIXES.items():
        for col, new in fixes.items():
            v = raw_check.loc[pid, col]
            assert pd.notna(v) and float(v) == float(new), f"{pid}.{col} 写入校验失败: {v} vs {new}"

    # ---------- 2. 生成 direct_cohort_ids.csv ----------
    # tgr map (与阶段1 同构: 完整版 batch1 行序 + 9.29 batch2 编号+200, 病理TGR)
    from openpyxl import load_workbook
    tgr = {}
    wb_full = load_workbook(os.path.join(ROOT, "data", "Hos", "结直肠癌多模态患者数据收集表_完整版.xlsx"), data_only=True)
    for i, r in enumerate(list(wb_full.active.iter_rows(values_only=True))[1:], start=1):
        if r[14] is not None:
            try: tgr[i] = float(r[14])
            except Exception: pass
    wb_29 = load_workbook(os.path.join(ROOT, "data", "Hos", "结直肠癌多模态患者数据收集表9.29.xlsx"), data_only=True)
    for r in list(wb_29.active.iter_rows(values_only=True))[1:]:
        if r[0] is not None and r[8] is not None:
            try: tgr[int(r[0]) + 200] = float(r[8])
            except Exception: pass
    # 仅保留合法等级 1/2/3 (异常值已由用户修正清除)
    tgr = {p: v for p, v in tgr.items() if v in (1.0, 2.0, 3.0)}
    master_pids = set(master["patient_id"].astype(int))
    ncrt_pids = sorted(p for p in tgr if p in master_pids)
    direct_pids = sorted(master_pids - set(ncrt_pids))
    assert len(direct_pids) == 280 and len(ncrt_pids) == 75, f"cohort 计数 {len(direct_pids)}/{len(ncrt_pids)}"
    cohort = pd.DataFrame({
        "pid": direct_pids + ncrt_pids,
        "is_tgr": [0] * len(direct_pids) + [1] * len(ncrt_pids),
    }).sort_values("pid").reset_index(drop=True)
    cohort.to_csv(COHORT_OUT, index=False)

    # direct_tcd 计数 (master 内 direct ∩ t∩d 物理路径非空)
    raw_idx = raw.set_index("patient_id")
    direct_td = [p for p in direct_pids
                 if p in raw_idx.index and pd.notna(raw_idx.loc[p, "mri_path_t"]) and pd.notna(raw_idx.loc[p, "mri_path_d"])]
    direct_tcd = [p for p in direct_pids
                  if p in raw_idx.index and pd.notna(raw_idx.loc[p, "mri_path_t"]) and pd.notna(raw_idx.loc[p, "mri_path_c"]) and pd.notna(raw_idx.loc[p, "mri_path_d"])]
    assert len(direct_td) == 214 and len(direct_tcd) == 162, f"direct_td/tcd {len(direct_td)}/{len(direct_tcd)}"

    # ---------- 3. 落盘变更明细 (无隐私) ----------
    diff = {
        "applied_at": ts,
        "master_modified": False,
        "master_note": "病理T/N 与 master 100% 一致(0 变更), 未入组 batch1 pid(9/12/20/65/75/117/120) 暂不补入, master_5fold_356.csv 零变动",
        "tn_changes": 0,
        "clin_changes": changed,
        "clin_anticontamination_check": {
            "ca199_lt2_as_2.0": {"before": ca199_eq2_before, "after": ca199_eq2_after, "ok": ca199_eq2_before == ca199_eq2_after == 10},
            "age_xxsui_as_number": {"before_nonn": age_nonnull_before, "after_nonn": age_nonnull_after,
                                     "before_num": age_isnum_before, "after_num": age_isnum_after,
                                     "ok": age_isnum_before == age_isnum_after},
        },
        "tgr": {
            "new_map_n": len(tgr), "new_dist": {k: sum(1 for v in tgr.values() if v == float(k)) for k in ("1.0", "2.0", "3.0")},
            "old_dist": {"2.0": 40, "3.0": 22, "1.0": 10, "anomalies": ["14.92", "5/38", "41"]},
            "anomaly_cleanup_note": "清除 3 例错位 tgr 异常值(14.92/5-38/41) → direct_tcd 由 160 修正为 162",
        },
        "cohort": {
            "direct_all": len(direct_pids), "ncrt": len(ncrt_pids),
            "direct_td": len(direct_td), "direct_tcd": len(direct_tcd),
            "cohort_csv": "data/direct_cohort_ids.csv",
        },
    }
    with open(DIFF_OUT, "w", encoding="utf-8") as f:
        json.dump(diff, f, ensure_ascii=False, indent=1)

    # ---------- 4. README 简记 ----------
    section = "\n## 数据勘误 (2026-11-23)\n" \
              "- 依据 m05147: 用户手动修正 data/Hos 3 个 excel 后, 阶段1 只读 diff + 阶段2 安全落盘。\n" \
              "- 病理 T/N 与 master 100% 一致 (0 变更), master_5fold_356.csv 未改动 (5折/标签零变动)。\n" \
              "- 临床 7 维更新 7 处: pid372 ca199 8.56/cea 1.25, pid395 prealbumin 241.0/hb 139.0, pid402 albumin 41.0/prealbumin 182.0/hb 148.0 (hos_pairs_raw.csv)。\n" \
              "  - 防污染校验: 10 处 ca199 '<2' 保持解析 2.0, 87 处 age 'XX岁' 保持数值, 无 NaN 误写。\n" \
              "- 清除 3 例错位 tgr 异常值 (14.92 / '5-38' / 41) → direct_tcd 由 160 修正为 162。\n" \
              "- 新建 data/direct_cohort_ids.csv (仅 pid/is_tgr 两列): direct_all=280 / ncrt=75 / direct_td=214 / direct_tcd=162。\n" \
              "- 变更明细(无隐私): logs/data_fix_diff.json\n"
    with open(README, "a", encoding="utf-8") as f:
        f.write(section)

    print(json.dumps(diff, ensure_ascii=False, indent=1))
    print("OK: raw updated, cohort written, diff+README appended")

if __name__ == "__main__":
    main()
