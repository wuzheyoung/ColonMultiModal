# -*- coding: utf-8 -*-
"""阶段1 只读 diff 核验: 3 excel (用户 2026-11-23 修正) vs master/raw 旧值, 只读不改任何文件."""
import os, re, json, sys
import numpy as np, pandas as pd, openpyxl

HOS = os.path.join("data", "Hos")
X1 = os.path.join(HOS, "结直肠癌多模态患者数据收集表.xlsx")        # batch1 编号=pid
X2 = os.path.join(HOS, "结直肠癌多模态患者数据收集表9.29.xlsx")   # batch2 编号+200=pid
X3 = os.path.join(HOS, "结直肠癌多模态患者数据收集表_完整版.xlsx") # batch1 行序=pid
MASTER = "data/master_5fold_356.csv"
RAW = "data/hos_pairs_raw.csv"


def fnum(v):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return float("nan")
    if isinstance(v, str):
        s = v.strip()
        if s.lower().startswith("<"):
            try:
                return 2.0 if float(s[1:].strip()) == 2 else float("nan")
            except Exception:
                return float("nan")
        try:
            return float(s)
        except Exception:
            return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def parse_tnm(s):
    """'30x' -> (3.0, 0.0); '31x' -> (3.0, 1.0)"""
    if s is None:
        return float("nan"), float("nan")
    digits = re.sub(r"[^0-9]", "", str(s))
    if len(digits) >= 2:
        return float(digits[0]), float(digits[1])
    return float("nan"), float("nan")


def diff_entry(pid, col, old, new):
    return {"pid": int(pid), "col": col, "old": old, "new": new}


def cellnorm(v, is_sex=False):
    if is_sex:
        return 1.0 if v == "男" else (0.0 if v == "女" else float("nan"))
    return fnum(v)


def main():
    m = pd.read_csv(MASTER).set_index("patient_id")
    raw = pd.read_csv(RAW).set_index("patient_id")

    t1 = openpyxl.load_workbook(X1, data_only=True).worksheets[0]
    t2 = openpyxl.load_workbook(X2, data_only=True).worksheets[0]
    t3 = openpyxl.load_workbook(X3, data_only=True).worksheets[0]
    h1 = {str(t1.cell(1, c).value).strip(): c for c in range(1, t1.max_column + 1) if t1.cell(1, c).value is not None}
    h3 = {str(t3.cell(1, c).value).strip(): c for c in range(1, t3.max_column + 1) if t3.cell(1, c).value is not None}
    h2 = {str(t2.cell(1, c).value).strip(): c for c in range(1, t2.max_column + 1) if t2.cell(1, c).value is not None}
    clin_cols = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
    xl_clin = {"age": "年龄", "sex": "性别", "ca199": "ca199", "cea": "癌胚抗原",
               "albumin": "白蛋白", "prealbumin": "前白蛋白", "hb": "血红蛋白"}

    out = {"files": {}, "tgr_changes": [], "clin_changes": [], "tn_changes": [],
           "tgr_map_new": {}, "excel_hygiene": {}, "cohort_counts": {}}
    out["files"] = {
        "x1": os.path.basename(X1), "x1_mtime": os.path.getmtime(X1),
        "x2": os.path.basename(X2), "x2_mtime": os.path.getmtime(X2),
        "x3": os.path.basename(X3), "x3_mtime": os.path.getmtime(X3),
        "master_rows": len(m), "raw_rows": len(raw),
    }

    # ---------- batch1: 收集表 (编号=pid), 病理T/N 15/16, 病理M 17, 临床2-8 ----------
    for i in range(2, t1.max_row + 1):
        pidv = t1.cell(i, 1).value
        if pidv is None:
            continue
        pid = int(pidv)
        if pid not in m.index:
            continue
        # 病理T/N vs master
        pT = fnum(t1.cell(i, h1.get("病理T", 15)).value)
        pN = fnum(t1.cell(i, h1.get("病理N", 16)).value)
        for col, nv, ov in (("pT", pT, m.loc[pid, "pT"]), ("pN", pN, m.loc[pid, "pN"])):
            if not ((pd.isna(nv) and pd.isna(ov)) or (not pd.isna(nv) and not pd.isna(ov) and abs(nv - ov) < 1e-9)):
                out["tn_changes"].append(diff_entry(pid, col, float(ov) if not pd.isna(ov) else None,
                                                    float(nv) if not pd.isna(nv) else None))
        # 临床7维 vs raw
        if pid in raw.index:
            for col in clin_cols:
                xc = xl_clin[col]
                c = h1.get(xc)
                if c is None:
                    continue
                nv = cellnorm(t1.cell(i, c).value, is_sex=(col == "sex"))
                ov = raw.loc[pid, col]
                if not ((pd.isna(nv) and pd.isna(ov)) or (not pd.isna(nv) and not pd.isna(ov) and abs(nv - ov) < 1e-9)):
                    out["clin_changes"].append(diff_entry(pid, col, float(ov) if not pd.isna(ov) else None,
                                                          float(nv) if not pd.isna(nv) else None))

    # ---------- batch2: 9.29 (编号+200=pid), 病理学诊断(7列) parse_tnm, 病理TGR 9列, 临床10-14 ----------
    for i in range(2, t2.max_row + 1):
        nov = t2.cell(i, 1).value
        if nov is None:
            continue
        pid = int(nov) + 200
        if pid not in m.index:
            continue
        pT, pN = parse_tnm(t2.cell(i, h2.get("病理学诊断", 7)).value)
        for col, nv, ov in (("pT", pT, m.loc[pid, "pT"]), ("pN", pN, m.loc[pid, "pN"])):
            if not ((pd.isna(nv) and pd.isna(ov)) or (not pd.isna(nv) and not pd.isna(ov) and abs(nv - ov) < 1e-9)):
                out["tn_changes"].append(diff_entry(pid, col, float(ov) if not pd.isna(ov) else None,
                                                    float(nv) if not pd.isna(nv) else None))
        # 临床7维 vs raw (batch2 表头: 编号1/住院号2/姓名3/性别4/年龄5/影像信息6/病理学诊断7/淋巴结转移8/病理TGR9/白蛋白10/前白蛋白11/血红蛋白12/ca199 13/癌胚抗原14)
        b2_clin = {"age": 5, "sex": 4, "ca199": 13, "cea": 14, "albumin": 10, "prealbumin": 11, "hb": 12}
        if pid in raw.index:
            for col, c in b2_clin.items():
                nv = cellnorm(t2.cell(i, c).value, is_sex=(col == "sex"))
                ov = raw.loc[pid, col]
                if not ((pd.isna(nv) and pd.isna(ov)) or (not pd.isna(nv) and not pd.isna(ov) and abs(nv - ov) < 1e-9)):
                    out["clin_changes"].append(diff_entry(pid, col, float(ov) if not pd.isna(ov) else None,
                                                          float(nv) if not pd.isna(nv) else None))

    # ---------- tgr: batch1 完整版病理TGR(15列, 行序=pid), batch2 9.29病理TGR(9列, 编号+200) ----------
    for i in range(2, t3.max_row + 1):
        pid = i - 1
        v = t3.cell(i, h3.get("病理TGR", 15)).value
        if v is not None and str(v).strip() not in ("", "nan", "None"):
            try:
                out["tgr_map_new"][pid] = float(v)
            except Exception:
                out["tgr_map_new"][pid] = str(v)
    for i in range(2, t2.max_row + 1):
        nov = t2.cell(i, 1).value
        if nov is None:
            continue
        pid = int(nov) + 200
        v = t2.cell(i, h2.get("病理TGR", 9)).value
        if v is not None and str(v).strip() not in ("", "nan", "None"):
            try:
                out["tgr_map_new"][pid] = float(v)
            except Exception:
                out["tgr_map_new"][pid] = str(v)

    # 旧 tgr 分布 (audit_tgr_planes_fusion.json 或 audit_tgr_task1.json)
    old_dist = {}
    for fn in ("logs/audit_tgr_planes_fusion.json", "logs/audit_tgr_task1.json"):
        if os.path.exists(fn):
            try:
                j = json.load(open(fn, encoding="utf-8"))
                d = j.get("tgr_value_counts") or j.get("tgr_dist") or {}
                if d:
                    old_dist = {k: v for k, v in d.items()}
                    break
            except Exception:
                pass
    out["tgr_old_distribution"] = old_dist
    new_counts = {}
    for v in out["tgr_map_new"].values():
        k = str(v)
        new_counts[k] = new_counts.get(k, 0) + 1
    out["tgr_new_distribution"] = new_counts

    # ---------- 队列样本数 (master ∩ mri_path_t 子集 + 全表) ----------
    mri_t_pids = set()
    for fn in ("logs/audit_tgr_task1.json", "logs/audit_tgr_planes_fusion.json"):
        if os.path.exists(fn):
            try:
                j = json.load(open(fn, encoding="utf-8"))
                if "pids" in j:
                    mri_t_pids = set(j["pids"])
                    break
            except Exception:
                pass
    if not mri_t_pids:  # 回退: 有 mri_path_t 的 pid
        pr = pd.read_csv(RAW)
        mri_t_pids = set(pr.loc[pr["mri_path_t"].notna(), "patient_id"]) & set(m.index)

    def cohorts(subset):
        s = {}
        s["n_total"] = len(subset)
        s["n_direct"] = len([p for p in subset if p not in out["tgr_map_new"]])
        s["n_ncrt"] = len([p for p in subset if p in out["tgr_map_new"]])
        # direct_td = direct ∩ (t 与 d 均有 mask 可用)
        # direct_tcd = direct ∩ (t/c/d 均有)
        return s

    out["cohort_counts"]["mri_t_subset"] = cohorts(mri_t_pids)
    out["cohort_counts"]["all_master"] = cohorts(set(m.index))

    # ---------- Excel 脏格式体检 ----------
    hyg = {}
    for name, t, h, pidfn in (("x1", t1, h1, lambda i: int(t1.cell(i, 1).value)),
                              ("x2", t2, h2, lambda i: int(t2.cell(i, 1).value) + 200),
                              ("x3", t3, h3, lambda i: i - 1)):
        items = []
        for c in range(1, t.max_column + 1):
            vals = [t.cell(i, c).value for i in range(2, t.max_row + 1) if t.cell(i, 1).value is not None]
            types = {}
            for v in vals:
                if v is None:
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)):
                    types["numeric"] = types.get("numeric", 0) + 1
                elif isinstance(v, str):
                    if re.match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$", v.strip()):
                        types["numeric_str"] = types.get("numeric_str", 0) + 1
                    elif re.match(r"^\d{4}-\d{2}-\d{2}", v.strip()):
                        types["date_str"] = types.get("date_str", 0) + 1
                    elif re.match(r"^\d+\.\d+E[+-]\d+$", v.strip()):
                        types["sci_str"] = types.get("sci_str", 0) + 1
                    else:
                        types["text"] = types.get("text", 0) + 1
                else:
                    types[type(v).__name__] = types.get(type(v).__name__, 0) + 1
            num_vals = [float(v) for v in vals if isinstance(v, (int, float, np.integer, np.floating))]
            items.append({
                "col": h.get(c, c),
                "n_rows": len(vals),
                "n_null": sum(1 for v in vals if v is None),
                "types": types,
                "min": min(num_vals) if num_vals else None,
                "max": max(num_vals) if num_vals else None,
                "mean": float(np.mean(num_vals)) if num_vals else None,
            })
        hyg[name] = items
    out["excel_hygiene"] = hyg

    outfile = "logs/data_fix_diff.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"OK -> {outfile}")
    print(f"tn_changes={len(out['tn_changes'])} clin_changes={len(out['clin_changes'])} tgr_map={len(out['tgr_map_new'])}")
    print("clin_changes:", json.dumps(out["clin_changes"], ensure_ascii=False))
    print("tgr_new_dist:", json.dumps(out["tgr_new_distribution"]))
    print("tgr_old_dist:", json.dumps(out["tgr_old_distribution"]))
    print("cohort:", json.dumps(out["cohort_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
