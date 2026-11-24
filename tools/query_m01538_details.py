# -*- coding: utf-8 -*-
"""m01716 四问: 校准表(Platt 单口径) / 日期缺失归因 / 时间切分明细 / ncrt 标签与 t∩d_self。
只读。复用 audit_temporal_ncrt 的日期联接逻辑。"""
import os, sys, json, re, glob
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\Users\Chris\Projects\ColonMultiModal")
sys.path.insert(0, "tools")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import eval_paper_metrics as epm
import exp_v2_final_calibrated as CAL

MASTER = "data/master_5fold_356.csv"
PAIRS = "data/hos_pairs_raw.csv"
COHORT = "data/direct_cohort_ids.csv"
SUMMARY = "logs/dwi_mask_remap_summary.csv"
DICOM = "2026-11-21/hos-data-inventory/work/dicom_meta.json"
OUT = "outputs/exp_v2_final_T"
_PID_PLANE = re.compile(r"^(\d+)_([tcd])$")

md = pd.read_csv(MASTER)
co = pd.read_csv(COHORT)
raw0 = pd.read_csv(PAIRS)
sm = pd.read_csv(SUMMARY)

tgr = set(co.loc[co.is_tgr == 1, "pid"])
d_ids = set(md.loc[~md.patient_id.isin(tgr), "patient_id"])
tcd = d_ids & set(raw0.loc[raw0.mri_path_t.notna(), "patient_id"]) \
             & set(raw0.loc[raw0.mri_path_c.notna(), "patient_id"]) \
             & set(raw0.loc[raw0.mri_path_d.notna(), "patient_id"])
td = d_ids & set(raw0.loc[raw0.mri_path_t.notna(), "patient_id"]) \
           & set(raw0.loc[raw0.mri_path_d.notna(), "patient_id"])
self_d = set(sm.loc[(sm.cohort_direct == 1) & (sm.mask_source == "self"), "pid"]) & d_ids

dj = json.load(open(DICOM, encoding="utf-8"))
sdate = {}
for x in dj["MRI"]:
    m = _PID_PLANE.match(x["dir"])
    if m:
        sdate[(int(m.group(1)), m.group(2))] = x["meta_first"].get("StudyDate")
raw = raw0.set_index("patient_id")

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

res = {}

# ---------- Q1 校准表: chosen, Platt 单口径 (5 折 a_k, b_k) ----------
def load(p):
    return pd.read_csv(os.path.join(OUT, p))
chosen = load("step1_clin_dself_t_2d_B.csv")
dc = CAL.platt_calibrate(chosen)
z = chosen["prob"].values.astype(float)
folds = chosen["fold"].values.astype(int)
y = chosen["label_T"].values.astype(int)
platt_params = {}
for k in range(5):
    tr = folds != k
    mm = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(
        z[tr].reshape(-1, 1), y[tr])
    platt_params[f"fold{k}"] = {
        "a_k_slope": float(mm.coef_[0][0]), "b_k_intercept": float(mm.intercept_[0]),
        "train_n": int(tr.sum()), "train_pos": int(y[tr].sum()),
        "train_neg": int((y[tr] == 0).sum())}
res["q1_platt"] = {
    "model": "clin+d_self+t_2d (chosen)",
    "pooled_auc_platt": float(epm.pooled_auc(dc)),
    "pooled_auc_sigmoid": float(epm.pooled_auc(CAL.F.to_prob_z(chosen))),
    "fold_mean_platt": float(epm.fold_mean_sd(dc)[0]),
    "fold_sd_platt": float(epm.fold_mean_sd(dc)[1]),
    "per_fold_platt_params": platt_params,
}

# ---------- Q2 日期缺失归因 ----------
have = {}
for pid in sorted(d_ids):
    sd, plane = get_date(pid)
    if sd:
        have[pid] = (sd, plane)
miss = d_ids - set(have)
# 缺失者: 有无 MRI 路径? 是否在 dicom_meta 有同名目录?
def any_path(pid):
    if pid not in raw.index:
        return False
    return any(isinstance(raw.loc[pid, "mri_path_" + p], str)
               for p in ("t", "c", "d"))
def path_planes(pid):
    out = []
    for p in ("t", "c", "d"):
        v = raw.loc[pid, "mri_path_" + p] if pid in raw.index else None
        out.append((p, isinstance(v, str) and v not in ("", "nan")))
    return out
miss_no_path = [p for p in miss if not any_path(p)]
miss_with_path_but_no_date = [p for p in miss if any_path(p)]
# 在 dicom_meta 中出现过对应 pid 的目录数
meta_pids = {int(x["dir"].split("_")[0]) for x in dj["MRI"]}
miss_meta_pid = [p for p in miss if p in meta_pids]
have_meta_pid = [p for p in have if p in meta_pids]
# 医院分布
hosp = md.set_index("patient_id")["hospital_id"].to_dict()
from collections import Counter
miss_hosp = Counter(hosp[p] for p in miss if p in hosp)
have_hosp = Counter(hosp[p] for p in have if p in hosp)
# 有日期者的年月跨度 + 年份分布
dates_have = sorted(v[0] for v in have.values())
years = Counter(v[0][:4] for v in have.values())
# 路径-平面 与 无路径 缺失构成
miss_path_breakdown = {}
for p, has in [("t", "c"), ("d", None)]:
    pass
n_path_any = sum(any_path(p) for p in miss)
miss_by_planes = Counter(tuple(pth for p_, pth in path_planes(p)) for p in miss)
res["q2_dates"] = {
    "n_direct": len(d_ids), "n_have": len(have), "n_missing": len(miss),
    "missing_no_any_mri_path": len(miss_no_path),
    "missing_with_mri_path_but_no_date": len(miss_with_path_but_no_date),
    "missing_meta_has_pid_dir": len(miss_meta_pid),
    "have_meta_has_pid_dir": len(have_meta_pid),
    "have_date_span_ym": [dates_have[0][:6], dates_have[-1][:6]],
    "have_date_span_full": [dates_have[0], dates_have[-1]],
    "have_by_year": dict(sorted(years.items())),
    "missing_by_hospital": dict(miss_hosp.most_common()),
    "have_by_hospital": dict(have_hosp.most_common()),
    "missing_path_plane_pattern": {str(k): v for k, v in miss_by_planes.most_common()},
}

# ---------- Q3 时间切分明细 (20250428) ----------
def split_detail(pids, cutoff="20250428"):
    rows = [{"patient_id": p, "date": have[p][0], "label_T": int(md.set_index("patient_id").loc[p, "label_T"]),
             "d_self": p in self_d} for p in pids if p in have]
    df = pd.DataFrame(rows)
    tr = df[df.date < cutoff]
    te = df[df.date >= cutoff]
    def stat(s):
        return {"n": int(len(s)), "t_pos": int(s.label_T.sum()),
                "t_neg": int((s.label_T == 0).sum()), "d_self": int(s.d_self.sum()),
                "t_rate": round(float(s.label_T.mean()), 4)}
    return {"cutoff": cutoff, "n_with_date": int(len(df)), "train": stat(tr), "test": stat(te)}
res["q3_split_20250428"] = {
    "direct_all": split_detail(sorted(d_ids)),
    "direct_tcd": split_detail(sorted(tcd)),
}

# ---------- Q4 ncrt 标签 + t∩d_self ----------
ncrt = set(co.loc[co.is_tgr == 1, "pid"]) & set(md.patient_id)
t_npz = {int(os.path.basename(p).split(".")[0])
         for p in glob.glob("features/slice_feats_v2/dinov2/t/*.npz")}
d_self_ncrt = set(sm.loc[sm.mask_source == "self", "pid"]) & ncrt
pair = (ncrt & t_npz) & d_self_ncrt
mi = md.set_index("patient_id")
res["q4_ncrt"] = {
    "n_ncrt": len(ncrt),
    "label_T": {"pos": int((mi.loc[list(ncrt), "label_T"] == 1).sum()),
                "neg": int((mi.loc[list(ncrt), "label_T"] == 0).sum())},
    "pT_dist": {str(k): int(v) for k, v in mi.loc[list(ncrt), "pT"].value_counts().sort_index().items()},
    "cover_t_npz": len(ncrt & t_npz), "cover_d_self": len(d_self_ncrt),
    "t_cap_d_self_n": len(pair),
    "t_cap_d_self_label_T": {"pos": int((mi.loc[list(pair), "label_T"] == 1).sum()),
                             "neg": int((mi.loc[list(pair), "label_T"] == 0).sum())},
}

with open("logs/query_m01538_details.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
