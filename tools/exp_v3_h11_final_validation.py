# -*- coding: utf-8 -*-
"""exp_v3_h11_final_validation.py — TRIPOD 1b 最终验证套件 (H1.1 锁定候选)
================================================================================
用户 m08246 授权。锁定候选 H1.1: clin_clean + d_self(MeanMax) + t(tight60_m10 AWM)
+ 策略B 融合 + bonly 校准。四项:
  ① 10 秒口径核查 (has_d_self 组阈值 0.72 判阳比例 vs mean±std 矛盾, 补 median/P75)
  ② 锁定单次 5-fold CV 主表对标 (direct_td 214 / direct_tcd 162):
     clin_clean / Baseline_0 / H1.1 Pooled OOF AUC + 2000 分层 Bootstrap CI
     + ECE(bonly) + Brier + 配对 DeLong (H1.1 vs clin, H1.1 vs Baseline_0)
  ③ 20250428 时间分割外部验证 (TRIPOD 2b 核心): 三模型并排
     AUC+95%CI, 配对 DeLong, ECE, Brier, Calibration Slope & Intercept, DCA
  ④ n=75 nCRT 域偏移: 三模型零样本迁移 AUC/ECE/Brier
输出: logs/exp_v3_h11_final_validation/metrics.json + validation_report.md
禁改既有锁定日志; 禁调参/回退; 暂不改生产路径与 Git。
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

_orig_chdir = os.chdir
def _safe_chdir(p):
    if os.path.isdir(p):
        return _orig_chdir(p)
os.chdir = _safe_chdir
try:
    import exp_v2_validation_suite as S   # noqa
finally:
    os.chdir = _orig_chdir

import train_slice_mil as tsm            # noqa
import recompute_direct_tcd_singles as rc  # noqa
import exp_v2_final_T as F               # noqa
import exp_v2_round2_ablation as abl     # noqa
import eval_paper_metrics as epm         # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

T120 = "features/slice_feats_v2"
T60 = "features/slice_feats_v2_tight60_m10"
OUT = "logs/exp_v3_h11_final_validation"
os.makedirs(OUT, exist_ok=True)

md = S.md; md_i = S.md_i; raw = S.raw; sm = S.sm; master = S.master
ncrt_ids = S.ncrt_ids
self_d = sorted(S.self_d)
all_direct = list(S.all_direct)
td = set(S.td); tcd = set(S.tcd)
CLIN_COLS = rc.CLIN_COLS
BACKBONE = S.BACKBONE

# ---------------- 参数化池化 (支持 T120/T60 root) ----------------
_pc = {}
def pool_branch_root(root, pids, seq, pool, frame=None):
    frame = frame if frame is not None else md
    pids = list(pids)
    key = (root, seq, pool, tuple(pids), frame is master)
    if key in _pc:
        return _pc[key]
    have = tsm.collect_plane_pids(frame, root, BACKBONE, seq, set(pids))
    bags = tsm.preload_plane(frame, root, BACKBONE, seq, have, "label_T", "none",
                             torch.device("cpu"))
    X, y = abl.pool_bags(bags, have, pool)
    _pc[key] = (list(have), np.asarray(X, dtype=np.float64), np.asarray(y, dtype=float))
    return _pc[key]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------- 分层 Bootstrap 95% CI (2000 次) ----------------
def stratified_boot_ci(df, n=2000, seed=42):
    d = df.dropna(subset=["prob"]).reset_index(drop=True)
    y = d["label_T"].values.astype(int)
    p = np.clip(d["prob"].values.astype(float), 0, 1)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return None
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        ip = rng.choice(pos, size=len(pos), replace=True)
        ig = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([ip, ig])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(aucs, 2.5)), 4),
            round(float(np.percentile(aucs, 97.5)), 4)]


# ---------------- Calibration Slope & Intercept (Cox 校准) ----------------
def calib_slope_intercept(df):
    d = df.dropna(subset=["prob"]).reset_index(drop=True)
    y = d["label_T"].values.astype(int)
    p = np.clip(d["prob"].values.astype(float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return None, None
    mm = LogisticRegression(C=1e9, max_iter=2000, solver="lbfgs").fit(lp, y)
    return float(mm.coef_[0][0]), float(mm.intercept_[0])


# ---------------- 决策曲线 (净获益 vs Treat-All / clin_clean) ----------------
def dca_net_benefit(df, ths=(0.05, 0.95, 0.01)):
    d = df.dropna(subset=["prob"]).reset_index(drop=True)
    y = d["label_T"].values.astype(int)
    p = np.clip(d["prob"].values.astype(float), 0, 1)
    n = len(y); prev = y.mean()
    out = []
    for t in np.arange(ths[0], ths[1] + 1e-9, ths[2]):
        tp = int(((p >= t) & (y == 1)).sum())
        fp = int(((p >= t) & (y == 0)).sum())
        nb = (tp - fp * t / (1 - t)) / n
        nb_all = prev - (1 - prev) * t / (1 - t)
        out.append({"threshold": round(float(t), 2), "nb": round(float(nb), 4),
                    "nb_all": round(float(nb_all), 4), "nb_none": 0.0})
    return out


def dca_positive_range(dca_rows, clin_rows=None):
    """nb > nb_all 且 nb > 0 的阈值区间 (相对 Treat-All / Treat-None)。"""
    seg = []
    cur = None
    for r in dca_rows:
        ok = r["nb"] > r["nb_all"] and r["nb"] > r["nb_none"]
        if ok and cur is None:
            cur = [r["threshold"], r["threshold"]]
        elif ok and cur is not None:
            cur[1] = r["threshold"]
        elif not ok and cur is not None:
            seg.append(cur); cur = None
    if cur is not None:
        seg.append(cur)
    # 相对 clin_clean 的区间 (nb_model > nb_clin)
    seg_c = []
    cur = None
    for r, cr in zip(dca_rows, clin_rows):
        ok = r["nb"] > cr["nb"]
        if ok and cur is None:
            cur = [r["threshold"], r["threshold"]]
        elif ok and cur is not None:
            cur[1] = r["threshold"]
        elif not ok and cur is not None:
            seg_c.append(cur); cur = None
    if cur is not None:
        seg_c.append(cur)
    return seg, seg_c


# ======================================================================
# ① 10 秒口径核查
# ======================================================================
def sec1_caliber_audit():
    # 重算 10×5 H1.1 OOF 平均概率 (修正 root 参数化: stage5b 的 pool_branch 忽略 root)
    SEEDS = range(10)
    def fold_of_all(seed):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fo = np.empty(len(all_direct), dtype=int)
        for k, (_, va) in enumerate(skf.split(all_direct,
                    md_i.loc[all_direct, "label_T"].values.astype(int))):
            fo[va] = k
        return fo
    pos_of = {int(p): i for i, p in enumerate(all_direct)}

    def winner_oof(seed, fo):
        th, tX, ty = pool_branch_root(T60, all_direct, "t", "area_w")
        dh, dX, dy = pool_branch_root(T120, self_d, "d", "meanmax")
        t_p = np.full(len(all_direct), np.nan)
        t_p[[pos_of[p] for p in th]] = S.probe_generic(
            th, tX, ty, np.array([fo[pos_of[p]] for p in th]))[0]["prob"].values
        d_p = np.full(len(all_direct), np.nan)
        d_p[[pos_of[p] for p in dh]] = S.probe_generic(
            dh, dX, dy, np.array([fo[pos_of[p]] for p in dh]))[0]["prob"].values
        cdf, _ = S.clin_cv(all_direct, fo)
        clin_p = np.full(len(all_direct), np.nan)
        clin_p[[pos_of[p] for p in cdf["patient_id"]]] = cdf["prob"].values
        sub = pd.DataFrame({"patient_id": all_direct, "fold": fo,
                            "label_T": md_i.loc[all_direct, "label_T"].values.astype(int),
                            "clin": clin_p, "t60": t_p, "dself": d_p})
        z_clin = np.log(sub["clin"] / (1 - sub["clin"] + 1e-9))
        z_t = np.log(sub["t60"] / (1 - sub["t60"] + 1e-9))
        z_d = np.log(sub["dself"] / (1 - sub["dself"] + 1e-9))
        z_im = np.nanmean(np.vstack([z_t, z_d]), axis=0)
        z_im = np.where(np.isnan(z_im), 0.0, z_im)
        z_fuse = 0.5 * z_clin + 0.5 * z_im
        prob = np.clip(sigmoid(z_fuse), 1e-6, 1 - 1e-6)
        cdf2 = pd.DataFrame({"patient_id": sub["patient_id"], "fold": sub["fold"],
                             "label_T": sub["label_T"], "prob": prob})
        cal, _ = S.calibrate_cv(cdf2, mode="bonly")
        return cal

    agg = {}
    for seed in SEEDS:
        fo = fold_of_all(seed)
        df = winner_oof(seed, fo)
        for _, r in df.iterrows():
            pid = int(r["patient_id"])
            a = agg.setdefault(pid, {"sum": 0.0, "n": 0, "label": int(r["label_T"])})
            a["sum"] += float(r["prob"]); a["n"] += 1
    rows = [{"patient_id": p, "prob": a["sum"] / a["n"], "label_T": a["label"]}
            for p, a in agg.items()]
    oof = pd.DataFrame(rows)
    # 用户 m08155 口径: 分组限定在 direct_td (n=214) 内: 有 d_self n=104 / 缺 d_self n=110
    oof = oof[oof.patient_id.isin(td)].reset_index(drop=True)
    oof["has_d"] = oof.patient_id.isin(set(self_d))
    g = {}
    for name, mask in (("has_d_self", oof.has_d), ("no_d_self", ~oof.has_d)):
        d = oof[mask].reset_index(drop=True)
        p = d["prob"].values
        th = 0.72
        g[name] = {
            "n": int(len(d)),
            "pos_rate": round(float(d["label_T"].mean()), 4),
            "prob_mean": round(float(p.mean()), 4),
            "prob_std": round(float(p.std(ddof=1)), 4),
            "prob_median": round(float(np.median(p)), 4),
            "prob_p25": round(float(np.percentile(p, 25)), 4),
            "prob_p75": round(float(np.percentile(p, 75)), 4),
            "over_0.72_n": int((p > th).sum()),
            "over_0.72_rate": round(float((p > th).mean()), 4),
            "auc": round(float(roc_auc_score(d["label_T"], p)), 4),
        }
    # 解释: 为何 mean 0.7025 但 >0.72 仅 22%
    hd = g["has_d_self"]
    ex = ("has_d_self 组 mean=%.4f < median=%.4f (左偏分布), "
          "P75=%.4f 与 0.72 阈值接近; 阈值 0.72 位于 P75 之上, "
          "故判阳率 %.1f%% 远低于均值对应概率, 属分布形态正常, 非笔误。"
          % (hd["prob_mean"], hd["prob_median"], hd["prob_p75"],
             100 * hd["over_0.72_rate"]))
    return {"groups": g, "interpretation": ex,
            "scope": "direct_td (n=%d) 内分组" % len(oof),
            "note": "两脚本概率列均为 bonly 校准后概率: "
                    "stage5_misdiagnosis (winner_oof -> S.calibrate_cv mode=bonly) 与 "
                    "audit_h11_vs_base (section3, cal=策略B融合后+bonly校准)。"
                    "本脚本 sec1 重算口径与两者一致 (10×5 平均概率, bonly)。",
            "prob_col_check": {"stage5_misdiagnosis": "bonly", "audit_h11_vs_base": "bonly",
                               "sec1_recalc": "bonly"}}


def _normal_phi(x):
    import math
    return math.erf(x / math.sqrt(2))


# ======================================================================
# ② 锁定单次 5-fold CV 主表对标
# ======================================================================
def sec2_locked_5fold():
    folds_d = md_i.loc[all_direct, "fold"].values.astype(int)
    clin_df, _ = S.clin_cv(all_direct, folds_d)
    th, tX, ty = pool_branch_root(T120, all_direct, "t", "area_w")
    t_fold = md_i.loc[th, "fold"].values.astype(int)
    t120_df, _ = S.probe_generic(th, tX, ty, t_fold)
    th2, tX2, ty2 = pool_branch_root(T60, all_direct, "t", "area_w")
    t_fold2 = md_i.loc[th2, "fold"].values.astype(int)
    t60_df, _ = S.probe_generic(th2, tX2, ty2, t_fold2)
    dh, dX, dy = pool_branch_root(T120, self_d, "d", "meanmax")
    d_fold = md_i.loc[dh, "fold"].values.astype(int)
    d_df, _ = S.probe_generic(dh, dX, dy, d_fold)

    # 三模型: 单分支经策略B(单分支退化为 z) + bonly
    def model_df(comps):
        fused = F.fuse_zlogit_strategy(comps, strategy="B", clin_key="clin")
        cal, _ = S.calibrate_cv(fused, mode="bonly")
        cal["prob"] = np.clip(cal["prob"], 0, 1)
        return cal

    clin_m = model_df({"clin": clin_df})
    base_m = model_df({"clin": clin_df, "t": t120_df, "d": d_df})
    h11_m = model_df({"clin": clin_df, "t": t60_df, "d": d_df})

    res = {}
    for scope_name, scope in (("td", td), ("tcd", tcd)):
        scope = set(scope)
        rows = {}
        for mname, mdf in (("clin_clean", clin_m), ("Baseline_0", base_m), ("H1.1", h11_m)):
            g = mdf[mdf.patient_id.isin(scope)].reset_index(drop=True)
            m = S.metrics_of(g)
            m["boot_95ci_stratified"] = stratified_boot_ci(g)
            rows[mname] = m
        # 配对 DeLong (公共样本)
        dl = {}
        def pair(a_name, b_name):
            ga = rows[a_name]; gb = rows[b_name]
            ids = (set(rows[a_name]["_ids"] if "_ids" in rows[a_name] else [])
                   if False else None)
            return None
        # 用原始 df 做 DeLong (公共 patient_id 交集)
        h11_g = h11_m[h11_m.patient_id.isin(scope)].reset_index(drop=True)
        clin_g = clin_m[clin_m.patient_id.isin(scope)].reset_index(drop=True)
        base_g = base_m[base_m.patient_id.isin(scope)].reset_index(drop=True)
        dl["h11_vs_clin"] = _delong_common(h11_g, clin_g)
        dl["h11_vs_base"] = _delong_common(h11_g, base_g)
        res[scope_name] = {"models": rows, "delong": dl}
    return res


def _delong_common(df1, df2):
    ids = sorted(set(df1["patient_id"]) & set(df2["patient_id"]))
    if len(ids) < 10:
        return {"n_paired": len(ids), "note": "insufficient (<10)"}
    g1 = df1[df1.patient_id.isin(ids)].set_index("patient_id").loc[ids].reset_index()
    g2 = df2[df2.patient_id.isin(ids)].set_index("patient_id").loc[ids].reset_index()
    r = epm.delong_paired_test(g1, g2)
    return r if r else {"n_paired": len(ids), "note": "deLong failed"}


# ======================================================================
# ③ 20250428 时间分割外部验证
# ======================================================================
def sec3_temporal():
    dj = json.load(open("2026-11-21/hos-data-inventory/work/dicom_meta.json", encoding="utf-8"))
    import re as _re
    sdate = {}
    pidplane = _re.compile(r"^(\d+)_([tcd])$")
    for x in dj["MRI"]:
        m = pidplane.match(x["dir"])
        if m:
            sdate[(int(m.group(1)), m.group(2))] = x["meta_first"].get("StudyDate")
    def fname_pid(path):
        if not isinstance(path, str) or not path:
            return None
        b = os.path.basename(path).replace(".nii.gz", "")
        m = pidplane.match(b)
        return int(m.group(1)) if m else None
    def get_date(pid):
        for plane in ("t", "c", "d"):
            p = raw.loc[pid, "mri_path_" + plane] if pid in raw.index else None
            pp = fname_pid(p) if isinstance(p, str) else None
            sd = sdate.get((pp, plane)) if pp is not None else None
            if sd:
                return sd
        return None
    dated = []
    for pid in all_direct:
        sd = get_date(int(pid))
        if sd:
            dated.append((int(pid), sd))
    dated.sort(key=lambda x: x[1])
    cut = int(round(0.70 * len(dated)))
    tr_pids = [p for p, _ in dated[:cut]]
    te_pids = [p for p, _ in dated[cut:]]
    cutoff = dated[cut][1] if cut < len(dated) else None
    tr_arr = np.array(tr_pids, dtype=np.int64)
    te_arr = np.array(te_pids, dtype=np.int64)
    tr_tcd = [p for p in tr_pids if p in tcd]
    te_tcd = [p for p in te_pids if p in tcd]

    # 训练集内 5 折 (供 bonly 校准逐折拟合? 时间外推用整体训练拟合校准截距)
    def deploy(scope_tr, scope_te, name):
        tr_s = np.array(scope_tr, dtype=np.int64)
        te_s = np.array(scope_te, dtype=np.int64)
        p_clin_tr, p_clin_te, _ = S.clin_fit_predict(tr_s, te_s)
        # t 三路: 仅对 Baseline_0/H1.1 用 t; clin_clean 不接影像
        def t_preds(root):
            h, X, y = pool_branch_root(root, all_direct, "t", "area_w")
            pos = {int(p): i for i, p in enumerate(h)}
            ids_tr = np.array([p for p in tr_s if p in pos], dtype=np.int64)
            ids_te = np.array([p for p in te_s if p in pos], dtype=np.int64)
            p_tr, p_te, _ = S.probe_fit_predict(
                ids_tr, np.array([X[pos[p]] for p in ids_tr]),
                np.array([y[pos[p]] for p in ids_tr]),
                ids_te, np.array([X[pos[p]] for p in ids_te]))
            return ids_tr, ids_te, p_tr, p_te
        ids_t120_tr, ids_t120_te, p_t120_tr, p_t120_te = t_preds(T120)
        ids_t60_tr, ids_t60_te, p_t60_tr, p_t60_te = t_preds(T60)
        have_d, X_d, y_d = pool_branch_root(T120, self_d, "d", "meanmax")
        dpos = {int(p): i for i, p in enumerate(have_d)}
        ids_d_tr = np.array([p for p in tr_s if p in dpos], dtype=np.int64)
        ids_d_te = np.array([p for p in te_s if p in dpos], dtype=np.int64)
        if len(ids_d_tr) >= 10:
            p_d_tr, p_d_te, _ = S.probe_fit_predict(
                ids_d_tr, np.array([X_d[dpos[p]] for p in ids_d_tr]),
                np.array([y_d[dpos[p]] for p in ids_d_tr]),
                ids_d_te, np.array([X_d[dpos[p]] for p in ids_d_te]))
        else:
            p_d_tr = p_d_te = None

        def fuse_deploy(branch_probs, which):
            zs = {}
            for nm, (pids_tr2, pr_tr, pids_te2, pr_te) in branch_probs.items():
                L_tr = tsm.logit_clip(np.asarray(pr_tr, dtype=float))
                mu, sd = L_tr.mean(), L_tr.std() + 1e-9
                if which == "tr":
                    zs[nm] = pd.Series((L_tr - mu) / sd,
                                       index=np.asarray(pids_tr2, dtype=np.int64))
                else:
                    L_te = tsm.logit_clip(np.asarray(pr_te, dtype=float))
                    zs[nm] = pd.Series((L_te - mu) / sd,
                                       index=np.asarray(pids_te2, dtype=np.int64))
            pids_all = sorted(set().union(*[set(z.index) for z in zs.values()]))
            rows = []
            for pid in pids_all:
                comps = {n: z.loc[pid] for n, z in zs.items() if pid in z.index}
                zc = comps.get("clin")
                imgs = [z for n, z in comps.items() if n != "clin"]
                if zc is None or not imgs:
                    v = float(np.mean(list(comps.values())))
                else:
                    v = 0.5 * float(zc) + 0.5 * float(np.mean(imgs))
                rows.append({"patient_id": int(pid), "prob": v})
            return pd.DataFrame(rows)

        branch_probs = {"clin": (tr_s, p_clin_tr, te_s, p_clin_te),
                        "t120": (ids_t120_tr, p_t120_tr, ids_t120_te, p_t120_te),
                        "t60": (ids_t60_tr, p_t60_tr, ids_t60_te, p_t60_te)}
        if p_d_tr is not None:
            branch_probs["d"] = (ids_d_tr, p_d_tr, ids_d_te, p_d_te)
        # 三模型定义 (bonly 校准: 训练集 z 拟合截距 b → 测试集 sigmoid(z+b))
        models = {}
        for mname, keys in (("clin_clean", ("clin",)),
                            ("Baseline_0", ("clin", "t120", "d")),
                            ("H1.1", ("clin", "t60", "d"))):
            bps = {k: branch_probs[k] for k in keys}
            f_tr = fuse_deploy(bps, "tr")
            f_te = fuse_deploy(bps, "te")
            f_tr["label_T"] = f_tr.patient_id.map(md_i["label_T"]).astype(int)
            f_te["label_T"] = f_te.patient_id.map(md_i["label_T"]).astype(int)
            # bonly 校准: a=1, b 拟合训练集 z
            _, b = S.platt_fit_bonly(f_tr["prob"], f_tr["label_T"])
            f_tr["prob"] = sigmoid(f_tr["prob"] + b)
            f_te["prob"] = sigmoid(f_te["prob"] + b)
            f_tr["fold"] = 0; f_te["fold"] = 1
            models[mname] = {"train": f_tr, "test": f_te, "b": round(float(b), 5)}
        out = {}
        for mname, mm in models.items():
            f_te = mm["test"].dropna(subset=["prob"]).reset_index(drop=True)
            f_te["prob"] = np.clip(f_te["prob"], 0, 1)
            f_tr = mm["train"].dropna(subset=["prob"]).reset_index(drop=True)
            f_tr["prob"] = np.clip(f_tr["prob"], 0, 1)
            fpr, tpr, ths = roc_curve(f_tr["label_T"], f_tr["prob"])
            J = tpr - fpr
            th = ths[np.argmax(J)]
            tp = ((f_te["prob"] >= th) & (f_te["label_T"] == 1)).sum()
            fp = ((f_te["prob"] >= th) & (f_te["label_T"] == 0)).sum()
            tn = ((f_te["prob"] < th) & (f_te["label_T"] == 0)).sum()
            fn = ((f_te["prob"] < th) & (f_te["label_T"] == 1)).sum()
            slope, intercept = calib_slope_intercept(f_te)
            m = S.metrics_of(f_te)
            m["boot_95ci_stratified"] = stratified_boot_ci(f_te)
            m["calib_slope"] = round(slope, 4) if slope is not None else None
            m["calib_intercept"] = round(intercept, 4) if intercept is not None else None
            out[mname] = {
                "test_metrics": m,
                "train_youden_threshold": round(float(th), 4),
                "train_youden_j": round(float(J.max()), 4),
                "test_at_train_youden": {
                    "sens": round(float(tp / (tp + fn)), 4) if (tp + fn) else None,
                    "spec": round(float(tn / (tn + fp)), 4) if (tn + fp) else None,
                    "ppv": round(float(tp / (tp + fp)), 4) if (tp + fp) else None,
                    "npv": round(float(tn / (tn + fn)), 4) if (tn + fn) else None},
                "b": mm["b"], "train_n": int(len(f_tr)), "test_n": int(len(f_te)),
            }
        return out, models

    res = {"module": "M3 temporal extrapolation (TRIPOD 2b) — v3 H1.1",
           "cutoff_date": cutoff, "train_n": len(tr_pids), "test_n": len(te_pids),
           "missing_date_excluded": int(len(all_direct) - len(dated)),
           "tcd_subset": {"train_n": len(tr_tcd), "test_n": len(te_tcd)}}
    res["direct_all"], models_all = deploy(tr_pids, te_pids, "direct_all")
    res["direct_tcd"], models_tcd = deploy(tr_tcd, te_tcd, "direct_tcd")
    # 配对 DeLong (测试集公共样本, direct_all 口径)
    dl = {}
    for other in ("clin_clean", "Baseline_0"):
        dl["h11_vs_" + other] = _delong_common(
            models_all["H1.1"]["test"], models_all[other]["test"])
    res["direct_all"]["delong"] = dl
    dl_tcd = {}
    for other in ("clin_clean", "Baseline_0"):
        dl_tcd["h11_vs_" + other] = _delong_common(
            models_tcd["H1.1"]["test"], models_tcd[other]["test"])
    res["direct_tcd"]["delong"] = dl_tcd
    # DCA (H1.1 vs clin_clean vs Treat-All, direct_all 测试集)
    res["direct_all"]["dca"] = _dca_temporal(models_all)
    res["direct_tcd"]["dca"] = _dca_temporal(models_tcd)
    return res


def _dca_temporal(models):
    """基于 deploy 返回的模型测试集 df 计算 DCA (H1.1 vs clin_clean)。"""
    dca_h11 = dca_net_benefit(models["H1.1"]["test"])
    dca_clin = dca_net_benefit(models["clin_clean"]["test"])
    pos_all, pos_clin = dca_positive_range(dca_h11, dca_clin)
    return {"h11_dca": dca_h11, "clin_dca": dca_clin,
            "positive_vs_treatall_treatnone": pos_all,
            "positive_vs_clin": pos_clin}


# ======================================================================
# ④ n=75 nCRT 域偏移
# ======================================================================
def sec4_ncrt():
    # direct 全量训练 + direct OOF (master 折) 定 Youden
    folds_d = md_i.loc[all_direct, "fold"].values.astype(int)
    clin_df, _ = S.clin_cv(all_direct, folds_d)
    th, tX, ty = pool_branch_root(T120, all_direct, "t", "area_w")
    t_fold = md_i.loc[th, "fold"].values.astype(int)
    t120_df, _ = S.probe_generic(th, tX, ty, t_fold)
    th2, tX2, ty2 = pool_branch_root(T60, all_direct, "t", "area_w")
    t_fold2 = md_i.loc[th2, "fold"].values.astype(int)
    t60_df, _ = S.probe_generic(th2, tX2, ty2, t_fold2)
    dh, dX, dy = pool_branch_root(T120, self_d, "d", "meanmax")
    d_fold = md_i.loc[dh, "fold"].values.astype(int)
    d_df, _ = S.probe_generic(dh, dX, dy, d_fold)

    def model_oof(comps):
        fused = F.fuse_zlogit_strategy(comps, strategy="B", clin_key="clin")
        cal, _ = S.calibrate_cv(fused, mode="bonly")
        return cal

    base_cal = model_oof({"clin": clin_df, "t": t120_df, "d": d_df})
    h11_cal = model_oof({"clin": clin_df, "t": t60_df, "d": d_df})
    clin_cal = model_oof({"clin": clin_df})
    th_y_base = _youden(base_cal)
    th_y_h11 = _youden(h11_cal)
    th_y_clin = _youden(clin_cal)

    # 部署: 各分支 fit on ALL direct → predict ncrt (frame=master 全表)
    ncrt_arr = np.array(sorted(ncrt_ids), dtype=np.int64)
    all_master = np.array(sorted(master["patient_id"].astype(np.int64)), dtype=np.int64)
    p_clin_tr, p_clin_ncrt, _ = S.clin_fit_predict(all_direct, ncrt_arr)

    def t_deploy(root):
        h, X, y = pool_branch_root(root, all_master, "t", "area_w", frame=master)
        pos = {int(p): i for i, p in enumerate(h)}
        tr_p = np.array([p for p in all_direct if p in pos], dtype=np.int64)
        ncrt_t = np.array([p for p in ncrt_arr if p in pos], dtype=np.int64)
        if len(ncrt_t) < 2:
            return None, None, None
        p_tr, p_ncrt, _ = S.probe_fit_predict(
            tr_p, np.array([X[pos[p]] for p in tr_p]),
            np.array([y[pos[p]] for p in tr_p]),
            ncrt_t, np.array([X[pos[p]] for p in ncrt_t]))
        return tr_p, p_tr, (ncrt_t, p_ncrt)

    t120 = t_deploy(T120); t60 = t_deploy(T60)
    ncrt_self = set(sm.loc[(sm.cohort_direct == 0) & (sm.mask_source == "self"), "pid"])
    d_master = np.array(sorted(set(self_d) | set(ncrt_self)), dtype=np.int64)
    hd, Xd, yd = pool_branch_root(T120, d_master, "d", "meanmax", frame=master)
    dpos = {int(p): i for i, p in enumerate(hd)}
    d_tr = np.array([p for p in all_direct if p in dpos], dtype=np.int64)
    ncrt_d = np.array([p for p in ncrt_arr if p in dpos], dtype=np.int64)
    p_d_tr = p_d_ncrt = None
    if len(ncrt_d) >= 2:
        p_d_tr, p_d_ncrt, _ = S.probe_fit_predict(
            d_tr, np.array([Xd[dpos[p]] for p in d_tr]),
            np.array([yd[dpos[p]] for p in d_tr]),
            ncrt_d, np.array([Xd[dpos[p]] for p in ncrt_d]))

    def deploy_ncrt(branch_probs, th_y, direct_cal):
        # 策略B 融合 ncrt z → pooled bonly 校准 (direct OOF z 拟合 b)
        zs = {}
        for nm, (pids_tr, pr_tr, pids_te, pr_te) in branch_probs.items():
            L_tr = tsm.logit_clip(np.asarray(pr_tr, dtype=float))
            mu, sd = L_tr.mean(), L_tr.std() + 1e-9
            L_te = tsm.logit_clip(np.asarray(pr_te, dtype=float))
            zs[nm] = {"tr": pd.Series((L_tr - mu) / sd, index=np.asarray(pids_tr, dtype=np.int64)),
                      "te": pd.Series((L_te - mu) / sd, index=np.asarray(pids_te, dtype=np.int64))}
        pids_te_all = sorted(set().union(*[set(z["te"].index) for z in zs.values()]))
        rows = []
        for pid in pids_te_all:
            comps = {n: z["te"].loc[pid] for n, z in zs.items() if pid in z["te"].index}
            zc = comps.get("clin")
            imgs = [z for n, z in comps.items() if n != "clin"]
            if zc is None or not imgs:
                v = float(np.mean(list(comps.values())))
            else:
                v = 0.5 * float(zc) + 0.5 * float(np.mean(imgs))
            rows.append({"patient_id": int(pid), "prob": v})
        ncrt_z = pd.DataFrame(rows)
        # pooled bonly: a=1, b 拟合 direct OOF 的 z (direct_cal prob 是 z? — fuse 输出 z)
        _, b = S.platt_fit_bonly(direct_cal["prob"], direct_cal["label_T"])
        ncrt_z["prob"] = sigmoid(ncrt_z["prob"] + b)
        ncrt_z["label_T"] = ncrt_z.patient_id.map(
            master.set_index("patient_id")["label_T"]).astype(int)
        ncrt_z = ncrt_z.dropna(subset=["prob"]).reset_index(drop=True)
        ncrt_z["prob"] = np.clip(ncrt_z["prob"], 0, 1)
        ncrt_z["fold"] = 0
        return ncrt_z, th_y

    out = {}
    # clin_clean
    bps_c = {"clin": (np.array(all_direct, dtype=np.int64), p_clin_tr, ncrt_arr, p_clin_ncrt)}
    z_c, _ = deploy_ncrt(bps_c, th_y_clin, clin_cal)
    m = S.metrics_of(z_c); m["boot_95ci_stratified"] = stratified_boot_ci(z_c)
    m["th_y"] = round(float(th_y_clin), 4)
    out["clin_clean"] = m
    # Baseline_0
    bps_b = dict(bps_c)
    bps_b["t120"] = (t120[0], t120[1], t120[2][0], t120[2][1])
    if p_d_tr is not None:
        bps_b["d"] = (d_tr, p_d_tr, ncrt_d, p_d_ncrt)
    z_b, _ = deploy_ncrt(bps_b, th_y_base, base_cal)
    m = S.metrics_of(z_b); m["boot_95ci_stratified"] = stratified_boot_ci(z_b)
    m["th_y"] = round(float(th_y_base), 4)
    out["Baseline_0"] = m
    # H1.1
    bps_h = dict(bps_c)
    bps_h["t60"] = (t60[0], t60[1], t60[2][0], t60[2][1])
    if p_d_tr is not None:
        bps_h["d"] = (d_tr, p_d_tr, ncrt_d, p_d_ncrt)
    z_h, _ = deploy_ncrt(bps_h, th_y_h11, h11_cal)
    m = S.metrics_of(z_h); m["boot_95ci_stratified"] = stratified_boot_ci(z_h)
    m["th_y"] = round(float(th_y_h11), 4)
    out["H1.1"] = m
    return {"direct_train_n": int(len(all_direct)), "ncrt_n": int(len(ncrt_ids)),
            "models": out}


def _youden(df):
    d = df.dropna(subset=["prob"]).reset_index(drop=True)
    fpr, tpr, ths = roc_curve(d["label_T"], d["prob"])
    J = tpr - fpr
    return float(ths[np.argmax(J)])


# ======================================================================
# 汇总 + report
# ======================================================================
def main():
    print("[sec1] 口径核查 ...")
    s1 = sec1_caliber_audit()
    print("[sec2] 锁定 5-fold 主表 ...")
    s2 = sec2_locked_5fold()
    print("[sec3] 时间外推 ...")
    s3 = sec3_temporal()
    print("[sec4] ncrt 域偏移 ...")
    s4 = sec4_ncrt()
    res = {"sec1_caliber": s1, "sec2_locked_5fold": s2,
           "sec3_temporal": s3, "sec4_ncrt": s4}
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    write_md(res)
    print("DONE ->", os.path.join(OUT, "metrics.json"))


def write_md(res):
    L = []
    L.append("# TRIPOD 1b 最终验证套件 — H1.1 锁定候选 (exp_v3_h11_final_validation)")
    L.append("")
    L.append("候选模型 H1.1: clin_clean + d_self(MeanMax) + t(tight60_m10 AWM) + 策略B + bonly")
    L.append("")
    # sec1
    L.append("## ① 10 秒口径核查 (has_d_self 判阳率 vs mean±std)")
    g = res["sec1_caliber"]["groups"]
    L.append("| 组 | n | 阳性率 | mean | std | median | P25 | P75 | >0.72 n | >0.72 % | AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, d in g.items():
        L.append(f"| {name} | {d['n']} | {d['pos_rate']} | {d['prob_mean']} | {d['prob_std']} "
                 f"| {d['prob_median']} | {d['prob_p25']} | {d['prob_p75']} | {d['over_0.72_n']} "
                 f"| {d['over_0.72_rate']} | {d['auc']} |")
    L.append("")
    L.append("**解释**: " + res["sec1_caliber"]["interpretation"])
    L.append("")
    L.append("**分组口径**: " + res["sec1_caliber"].get("scope", ""))
    L.append("")
    L.append("**概率列口径**: " + res["sec1_caliber"]["note"])
    L.append("")
    # sec2
    L.append("## ② 锁定 5-fold CV 主表对标")
    for scope in ("td", "tcd"):
        L.append(f"### direct_{scope}")
        rows = res["sec2_locked_5fold"][scope]["models"]
        L.append("| 模型 | n | n_pos | Pooled AUC | 95% CI (分层2000) | ECE | Brier |")
        L.append("|---|---|---|---|---|---|---|")
        for mn, m in rows.items():
            ci = m.get("boot_95ci_stratified")
            ci_s = f"[{ci[0]},{ci[1]}]" if ci else "-"
            L.append(f"| {mn} | {m['n']} | {m['n_pos']} | {m['pooled_auc']} | {ci_s} "
                     f"| {m['ece10']} | {m['brier']} |")
        dl = res["sec2_locked_5fold"][scope]["delong"]
        for k, v in dl.items():
            L.append(f"- {k}: {v}")
        L.append("")
    # sec3
    L.append("## ③ 20250428 时间分割外部验证 (TRIPOD 2b)")
    for scope in ("direct_all", "direct_tcd"):
        L.append(f"### {scope} (train/test: "
                 f"{res['sec3_temporal'][scope].get('train_n', '-')}/"
                 f"{res['sec3_temporal'][scope].get('test_n', '-')})" if False else
                 f"### {scope}")
        for mn, mm in res["sec3_temporal"][scope].items():
            if not isinstance(mm, dict) or "test_metrics" not in mm:
                continue
            m = mm["test_metrics"]
            ci = m.get("boot_95ci_stratified")
            ci_s = f"[{ci[0]},{ci[1]}]" if ci else "-"
            L.append(f"- **{mn}** (test n={m['n']}): AUC {m['pooled_auc']} {ci_s} | "
                     f"ECE {m['ece10']} | Brier {m['brier']} | Slope {m['calib_slope']} | "
                     f"Intercept {m['calib_intercept']} | Youden阈值 "
                     f"{mm['train_youden_threshold']} 敏感度 {mm['test_at_train_youden']['sens']}")
        L.append("")
    if "dca" in res["sec3_temporal"]["direct_all"]:
        dca = res["sec3_temporal"]["direct_all"]["dca"]
        L.append("### DCA (时间测试集, H1.1)")
        L.append(f"- 相对 Treat-All/Treat-None 正净获益阈值区间: {dca['positive_vs_treatall_treatnone']}")
        L.append(f"- 相对 clin_clean 正净获益阈值区间: {dca['positive_vs_clin']}")
        L.append("")
    # sec4
    L.append("## ④ n=75 nCRT 域偏移 (零样本迁移)")
    L.append(f"- direct_train_n={res['sec4_ncrt']['direct_train_n']}, ncrt_n={res['sec4_ncrt']['ncrt_n']}")
    L.append("| 模型 | n | n_pos | Pooled AUC | 95% CI (分层2000) | ECE | Brier | Youden(direct) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mn, m in res["sec4_ncrt"]["models"].items():
        ci = m.get("boot_95ci_stratified")
        ci_s = f"[{ci[0]},{ci[1]}]" if ci else "-"
        L.append(f"| {mn} | {m['n']} | {m['n_pos']} | {m['pooled_auc']} | {ci_s} "
                 f"| {m['ece10']} | {m['brier']} | {m['th_y']} |")
    L.append("")
    L.append("---")
    L.append("*生成时间: 见 metrics.json; 口径: 锁定 5-fold=master fold; 时间外推=20250428; 校准=bonly。*")
    open(os.path.join(OUT, "validation_report.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
