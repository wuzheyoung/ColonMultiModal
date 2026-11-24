# -*- coding: utf-8 -*-
"""
audit_temporal_ncrt.py — 时间分布 / 时间切分 / ncrt 覆盖审计 (m01538 任务2, 只读)
================================================================================
为「时间外推 (TRIPOD 2b)」与「ncrt 域偏移测试」摸底:
  1) 检查时间分布 (direct_all n=280 与 direct_tcd n=162):
     - 检查时间字段来源: 2026-11-21/hos-data-inventory/work/dicom_meta.json
       (MRI 序列 DICOM StudyDate, 按 mri_path_{t,c,d} 文件名 pid_plane 联接,
       优先级 t → c → d, 缺省计缺失);
     - 按年份: 总例数 / label_T 阳性数与阳性率 / d_self 有效覆盖例数与覆盖率;
     - 时间排序 70/30 与 75/25 切分: 节点日期 + 训练/测试样本量、阳性率、
       测试集内 d_self 覆盖例数。
  2) ncrt (n=75) 放化疗后术前再分期队列覆盖审计:
     - label_T (ypT 二分类) 分布 + pT 细分 (ypT0/1/2/3/4);
     - clin_clean / slice_feats_v2 dinov2 t / d_self 有效覆盖例数。
输出: logs/audit_temporal_ncrt.{json,md} (+ 打印)
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

ROOT = r"C:\\Users\\Chris\\Projects\\ColonMultiModal"
os.chdir(ROOT)

MASTER = "data/master_5fold_356.csv"
PAIRS = "data/hos_pairs_raw.csv"
COHORT = "data/direct_cohort_ids.csv"
SUMMARY = "logs/dwi_mask_remap_summary.csv"
DICOM = "2026-11-21/hos-data-inventory/work/dicom_meta.json"
TNPZ = "features/slice_feats_v2/dinov2/t"
DNPZ = "features/slice_feats_v2/dinov2/d"
JOUT = "logs/audit_temporal_ncrt.json"
MOUT = "logs/audit_temporal_ncrt.md"

_PID_PLANE = re.compile(r"^(\d+)_([tcd])$")


def main():
    md = pd.read_csv(MASTER)
    co = pd.read_csv(COHORT)
    raw = pd.read_csv(PAIRS)
    sm = pd.read_csv(SUMMARY)

    tgr = set(co.loc[co.is_tgr == 1, "pid"])
    d_ids = set(md.loc[~md.patient_id.isin(tgr), "patient_id"])
    tcd = d_ids & set(raw.loc[raw.mri_path_t.notna(), "patient_id"]) \
                 & set(raw.loc[raw.mri_path_c.notna(), "patient_id"]) \
                 & set(raw.loc[raw.mri_path_d.notna(), "patient_id"])
    td = d_ids & set(raw.loc[raw.mri_path_t.notna(), "patient_id"]) \
                & set(raw.loc[raw.mri_path_d.notna(), "patient_id"])
    self_d = set(sm.loc[(sm.cohort_direct == 1) & (sm.mask_source == "self"), "pid"]) & d_ids

    # ---------- 检查时间: dicom StudyDate 联接 (优先级 t → c → d) ----------
    dj = json.load(open(DICOM, encoding="utf-8"))
    sdate = {}
    for x in dj["MRI"]:
        m = _PID_PLANE.match(x["dir"])
        if m:
            sdate[(int(m.group(1)), m.group(2))] = x["meta_first"].get("StudyDate")
    raw = raw.set_index("patient_id")

    def fname_pid(path):
        if not isinstance(path, str) or not path:
            return None
        b = os.path.basename(path).replace(".nii.gz", "")
        m = _PID_PLANE.match(b)
        return int(m.group(1)) if m else None

    def get_date(pid):
        for plane in ("t", "c", "d"):
            p = raw.loc[pid, "mri_path_" + plane] if pid in raw.index else None
            pp = fname_pid(p) if isinstance(p, str) else None
            sd = sdate.get((pp, plane)) if pp is not None else None
            if sd:
                return sd, plane
        return None, None

    def cohort_frame(pids, label="cohort"):
        rows = []
        for pid in sorted(pids):
            sd, plane = get_date(pid)
            rows.append({"patient_id": pid, "date": sd, "plane": plane})
        df = pd.DataFrame(rows)
        df["label_T"] = df.patient_id.map(
            md.set_index("patient_id")["label_T"].to_dict()).astype(int)
        df["d_self"] = df.patient_id.isin(self_d)
        return df

    f280 = cohort_frame(d_ids)
    f162 = cohort_frame(tcd)
    f214 = cohort_frame(td)

    def year_stats(df, name):
        y = df.loc[df.date.notna()].copy()
        y["year"] = y["date"].str[:4]
        out = {"cohort": name, "n": int(len(df)),
               "date_missing": int(df.date.isna().sum()),
               "date_range": [df.date.dropna().min(), df.date.dropna().max()]
               if df.date.notna().any() else None,
               "by_year": []}
        for yy in sorted(y["year"].unique()):
            g = y[y["year"] == yy]
            out["by_year"].append({
                "year": yy, "n": int(len(g)),
                "t_pos": int(g.label_T.sum()),
                "t_rate": round(float(g.label_T.mean()), 4),
                "d_self_n": int(g.d_self.sum()),
                "d_self_cov": round(float(g.d_self.mean()), 4)})
        return out

    def time_split(df, frac, name):
        y = df.loc[df.date.notna()].copy()
        y = y.sort_values("date").reset_index(drop=True)
        cut = int(round(frac * len(y)))
        tr, te = y.iloc[:cut], y.iloc[cut:]
        cut_date = te["date"].iloc[0] if len(te) else None
        return {
            "cohort": name, "frac": frac,
            "cutoff_date": cut_date,
            "n_total": len(y),
            "train_n": int(len(tr)), "train_t_rate": round(float(tr.label_T.mean()), 4),
            "test_n": int(len(te)), "test_t_rate": round(float(te.label_T.mean()), 4),
            "test_d_self_n": int(te.d_self.sum()),
            "test_d_self_cov": round(float(te.d_self.mean()), 4),
        }

    res = {
        "task": "audit_temporal_ncrt (m01538 任务2, 只读)",
        "date_source": "dicom_meta.json MRI StudyDate, 按 mri_path_{t,c,d} 文件名"
                       " pid_plane 联接, 优先级 t→c→d",
        "calibers": {"direct_all": len(d_ids), "direct_tcd": len(tcd),
                     "direct_td": len(td), "d_self_all": len(self_d)},
        "time_dist": [year_stats(f280, "direct_all"), year_stats(f162, "direct_tcd"),
                      year_stats(f214, "direct_td")],
        "time_split": [
            time_split(f280, 0.70, "direct_all"), time_split(f162, 0.70, "direct_tcd"),
            time_split(f214, 0.70, "direct_td"),
            time_split(f280, 0.75, "direct_all"), time_split(f162, 0.75, "direct_tcd"),
            time_split(f214, 0.75, "direct_td"),
        ],
    }

    # ---------- ncrt 覆盖审计 ----------
    ncrt_ids = set(co.loc[co.is_tgr == 1, "pid"]) & set(md.patient_id)
    ncrt = md.loc[md.patient_id.isin(ncrt_ids)].copy()
    # pT 细分: 用原始 pT 列 (数值 1-4; 若存在 0 则列出)
    pt_dist = ncrt.groupby("pT")["patient_id"].count().to_dict()
    # t / d 特征覆盖 (npz 文件名为 {pid}.npz)
    t_files = {int(os.path.splitext(f)[0]) for f in os.listdir(TNPZ) if f.endswith(".npz")}
    d_files = {int(os.path.splitext(f)[0]) for f in os.listdir(DNPZ) if f.endswith(".npz")}
    # clin_clean 覆盖: 临床 7 维至少一条非空 (raw 中有该患者行即视为有临床特征)
    clin_ok = set(raw.index) & ncrt_ids
    # d_self 覆盖: summary 中 mask_source=='self' (含 ncrt 患者, cohort_direct 可能为 0)
    ncrt_self = set(sm.loc[sm.mask_source == "self", "pid"]) & ncrt_ids
    ncrt_t = ncrt_ids & t_files
    ncrt_d = ncrt_ids & d_files
    res["ncrt"] = {
        "n": int(len(ncrt_ids)),
        "label_T": {str(k): int(v) for k, v in
                    ncrt.groupby("label_T")["patient_id"].count().to_dict().items()},
        "pT_dist": {str(k): int(v) for k, v in pt_dist.items()},
        "pT_known": int(ncrt.pT.notna().sum()),
        "t_rate": round(float(ncrt.label_T.mean()), 4),
        "cover_clin": int(len(clin_ok)),
        "cover_t_npz": int(len(ncrt_t)),
        "cover_d_npz": int(len(ncrt_d)),
        "cover_d_self": int(len(ncrt_self)),
    }

    with open(JOUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    lines = ["# audit_temporal_ncrt — 时间分布 / 切分 / ncrt 覆盖 (m01538 任务2)",
             f"- 检查时间来源: {res['date_source']}",
             f"- 口径: direct_all={len(d_ids)}, direct_tcd={len(tcd)}, direct_td={len(td)}, "
             f"d_self_all={len(self_d)}"]
    for ts in res["time_dist"]:
        lines.append(f"\n## {ts['cohort']} 时间分布 (n={ts['n']}, "
                     f"日期缺失={ts['date_missing']}, 范围={ts['date_range']})\n")
        lines.append("| year | n | T+ | T+率 | d_self n | d_self 覆盖率 |")
        lines.append("|---|---|---|---|---|---|")
        for r in ts["by_year"]:
            lines.append(f"| {r['year']} | {r['n']} | {r['t_pos']} | {r['t_rate']} | "
                         f"{r['d_self_n']} | {r['d_self_cov']} |")
    lines.append("\n## 时间切分模拟\n")
    lines.append("| cohort | frac | cutoff | train_n | train T+率 | test_n | "
                 "test T+率 | test d_self n | test d_self 覆盖率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in res["time_split"]:
        lines.append(f"| {r['cohort']} | {r['frac']} | {r['cutoff_date']} | "
                     f"{r['train_n']} | {r['train_t_rate']} | {r['test_n']} | "
                     f"{r['test_t_rate']} | {r['test_d_self_n']} | "
                     f"{r['test_d_self_cov']} |")
    n = res["ncrt"]
    lines.append(f"\n## ncrt (n={n['n']}) 覆盖审计\n")
    lines.append(f"- label_T 分布: {n['label_T']}, T+ 率 {n['t_rate']}")
    lines.append(f"- pT (ypT) 细分分布: {n['pT_dist']} (pT 已知 {n['pT_known']})")
    lines.append(f"- 覆盖: clin_clean {n['cover_clin']} / t npz {n['cover_t_npz']} / "
                 f"d npz {n['cover_d_npz']} / d_self {n['cover_d_self']}")
    with open(MOUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("DONE. JSON:", JOUT)
    print("MD:", MOUT)
    for ts in res["time_dist"]:
        print(ts["cohort"], "n=", ts["n"], "missing=", ts["date_missing"],
              "range=", ts["date_range"])
        for r in ts["by_year"]:
            print("  ", r)
    for r in res["time_split"]:
        print("split", r["cohort"], r["frac"], "cutoff", r["cutoff_date"],
              "tr", r["train_n"], r["train_t_rate"], "te", r["test_n"],
              r["test_t_rate"], "dself", r["test_d_self_n"])
    print("ncrt:", res["ncrt"])


if __name__ == "__main__":
    main()
