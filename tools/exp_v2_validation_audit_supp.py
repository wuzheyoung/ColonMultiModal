# -*- coding: utf-8 -*-
"""exp_v2_validation_audit_supp.py — m02069 快速明细核查与口径一致性诊断 (只读审计)

对 logs/exp_v2_validation_suite_T.{json,md} 与 outputs/exp_v2_validation_suite/ 做:
  1. M2 重复 CV 落差归因: 完整对照表 (tcd 162 / td 214 下 clin/t(AreaW)/d_self(MeanMax)/
     clin+t/clin+d_self+t 的 10×5 重复 Pooled OOF 均值±SD ddof=1 + min-max + 50 折 Fold-Mean±SD);
     固定 master_5fold 折上 M2 clin_cv 重训 vs 旧 clin_clean.csv 的逐样本 max|Δp| 与口径差异清单;
     融合换 M2 重训版 clin (master 折) + 折内 bonly 校准 → clin+d_self+t Pooled AUC (是否维持 0.7302)。
  2. M1 校准稳健补充 (tcd 162, 锁定 bonly 单口径): Calibration-in-the-large / Calibration Slope
     (LR y~logit(p_cal) 斜率 + 患者级 bootstrap 95% CI) / 5 等频分箱 ECE (对照 10 分箱)。
  3. M3 时间外推完整明细 (20250428 切分, direct_all test n=54 / direct_tcd 子测试集):
     clin_clean/clin+t/clin+d_self+t 的 AUC(95% BootCI)/Brier/ECE(5 分箱) +
     训练集 Youden 阈值迁移 Sens/Spec/PPV/NPV + 测试集 T+/T-/d_self 例数。
  4. M4 ncrt (n=75) 域偏移: clin_clean/clin+t/clin+d_self+t 的 AUC(95% CI)/Brier/预测均值
     (vs 真实 0.7733) + ypT2(n17) vs ypT3-4(n58) 预测概率中位数(IQR);
     t∩d_self 配对 n=34: t/d 单路 direct 内部 vs ncrt 的绝对 AUC + 95% BootCI。

严格只读: 不修改任何主表/OOF 文件; 结果落盘 logs/exp_v2_validation_audit_supp.json。
"""
import os, sys, json, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import exp_v2_validation_suite as S          # noqa: E402  (复用其全部 helper/常量)
import exp_v2_final_T as F                  # noqa: E402
import train_slice_mil as tsm               # noqa: E402
import eval_paper_metrics as epm            # noqa: E402

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

AOUT = "logs/exp_v2_validation_audit_supp.json"
OLD_CLIN = "outputs/oof_baseline_n162/singles/clin_clean.csv"


def clip01(p):
    return float(np.clip(p, 0.0, 1.0))


def scope_slice(df, scope):
    return df[df.patient_id.isin(scope)].reset_index(drop=True)


def ece_quantile(df, nbins=5):
    """等频 (quantile) 分箱 ECE。"""
    p = np.clip(df["prob"].values.astype(float), 0, 1)
    y = df["label_T"].values.astype(int)
    qs = np.quantile(p, np.linspace(0, 1, nbins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bins = np.digitize(p, qs[1:-1], right=False)
    ece = 0.0
    for b in range(nbins):
        m = bins == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return round(float(ece), 4)


def slope_and_ci(df, n=2000, seed=0):
    """Calibration slope: LR y ~ logit(p), 斜率 + 患者级 bootstrap 95% CI。"""
    p = np.clip(df["prob"].values.astype(float), 0, 1)
    y = df["label_T"].values.astype(int)
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9))

    def fit(i):
        try:
            mm = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(lp[i].reshape(-1, 1), y[i])
            return float(mm.coef_[0][0])
        except Exception:
            return np.nan

    mm0 = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(lp.reshape(-1, 1), y)
    slope0 = float(mm0.coef_[0][0])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n):
        i = rng.integers(0, len(df), size=len(df))
        boot.append(fit(i))
    boot = np.array(boot)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"slope": round(slope0, 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "n_boot_valid": int(len(boot))}


def per_fold_aucs(df):
    """scope 内每折 AUC (≥8 例且二类), 返回 (fold50 list, mean, sd ddof=1)。"""
    out = []
    for k in np.unique(df["fold"]):
        g = df[df["fold"] == k]
        if len(g) >= 8 and len(np.unique(g["label_T"])) == 2:
            out.append(round(float(roc_auc_score(g["label_T"], np.clip(g["prob"].values, 0, 1))), 4))
    out = np.array(out)
    if len(out) == 0:
        return [], None, None
    return out, round(float(out.mean()), 4), round(float(out.std(ddof=1)), 4)


# ============ 1. M2 完整对照表 (10×5 重跑, 含单路) ============
def m2_full(lock_mode="bonly"):
    d_pids = S.all_direct
    t_have, t_X, t_y = S.pool_branch(d_pids, "t", "area_w")
    d_have, d_X, d_y = S.pool_branch(sorted(S.self_d), "d", "meanmax")
    pos_of_pid = {int(p): i for i, p in enumerate(d_pids)}
    y_all = S.md_i.loc[d_pids, "label_T"].values.astype(int)
    scopes = {"direct_tcd": S.tcd, "direct_td": S.td}
    models = ("clin", "t", "d_self", "clin_t", "clin_dt")
    out = {}
    for scope_name, scope in scopes.items():
        per_seed = {m: [] for m in models}
        fold50 = {m: [] for m in models}
        win_clin, win_t, deltas_clin, deltas_t = 0, 0, [], []
        for seed in range(10):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_of = np.empty(len(d_pids), dtype=int)
            for k, (_, va) in enumerate(skf.split(d_pids, y_all)):
                fold_of[va] = k
            clin_df, _ = S.clin_cv(d_pids, fold_of)
            t_folds = np.array([fold_of[pos_of_pid[p]] for p in t_have])
            t_df, _ = S.probe_generic(t_have, t_X, t_y, t_folds)
            d_folds = np.array([fold_of[pos_of_pid[p]] for p in d_have])
            d_df, _ = S.probe_generic(d_have, d_X, d_y, d_folds)
            fuse_ct = F.fuse_zlogit_strategy({"clin": clin_df, "t": t_df}, strategy="B")
            fuse_cdt = F.fuse_zlogit_strategy({"clin": clin_df, "t": t_df, "d": d_df}, strategy="B")
            ct_cal, _ = S.calibrate_cv(fuse_ct, mode=lock_mode)
            cdt_cal, _ = S.calibrate_cv(fuse_cdt, mode=lock_mode)
            cl_s, t_s, d_s = scope_slice(clin_df, scope), scope_slice(t_df, scope), scope_slice(d_df, scope)
            ct_s, cdt_s = scope_slice(ct_cal, scope), scope_slice(cdt_cal, scope)
            dd = {"clin": cl_s, "t": t_s, "d_self": d_s, "clin_t": ct_s, "clin_dt": cdt_s}
            for m in models:
                p = np.clip(dd[m]["prob"].values, 0, 1)
                per_seed[m].append(round(float(roc_auc_score(dd[m]["label_T"], p)), 4))
                fa, _, _ = per_fold_aucs(dd[m])
                fold50[m].extend(fa)
            win_clin += int(per_seed["clin_dt"][-1] > per_seed["clin"][-1])
            win_t += int(per_seed["clin_dt"][-1] > per_seed["clin_t"][-1])
            deltas_clin.append(per_seed["clin_dt"][-1] - per_seed["clin"][-1])
            deltas_t.append(per_seed["clin_dt"][-1] - per_seed["clin_t"][-1])
        rep = {}
        for m in models:
            a = np.array(per_seed[m])
            fa = np.array(fold50[m])
            rep[m] = {
                "repeated_pooled_auc_mean": round(float(a.mean()), 4),
                "repeated_pooled_auc_sd_ddof1": round(float(a.std(ddof=1)), 4),
                "repeated_pooled_auc_min": round(float(a.min()), 4),
                "repeated_pooled_auc_max": round(float(a.max()), 4),
                "per_seed_aucs": per_seed[m],
                "fold50_mean": round(float(fa.mean()), 4) if len(fa) else None,
                "fold50_sd": round(float(fa.std(ddof=1)), 4) if len(fa) else None,
                "fold50_n": int(len(fa)),
            }
        rep["win_rate_best_vs_clin"] = f"{win_clin}/10"
        rep["win_rate_best_vs_clin_t"] = f"{win_t}/10"
        rep["mean_delta_vs_clin"] = round(float(np.mean(deltas_clin)), 4)
        rep["mean_delta_vs_clin_t"] = round(float(np.mean(deltas_t)), 4)
        out[scope_name] = rep
    return out


# ============ 1b. 口径一致性复核 (固定 master 折) ============
def caliber_check():
    folds_d = S.md_i.loc[S.all_direct, "fold"].values.astype(int)
    # M2 风格 clin_cv (折内嵌套3折CV 选 C) 跑在固定 master 折
    clin_m2, Cs_m2 = S.clin_cv(S.all_direct, folds_d)
    # 旧 pipeline 语义复刻: rc.per_fold_lr imp='median' (TRAIN-AUC 选 C) 在 master 折
    # 与 M2 clin_cv (折内嵌套3折CV 选 C) 对齐比对; 二者特征列/中位填补/StandardScaler/C 网格全同
    clin_old2, Cs_old = clin_cv_old_style(S.all_direct, folds_d)
    m2_tcd = scope_slice(clin_m2, S.tcd)
    old_tcd = scope_slice(clin_old2, S.tcd)
    auc_m2 = float(roc_auc_score(m2_tcd["label_T"], np.clip(m2_tcd["prob"].values, 0, 1)))
    auc_old = float(roc_auc_score(old_tcd["label_T"], np.clip(old_tcd["prob"].values, 0, 1)))
    # 与已落盘旧 clin_clean.csv 对比
    oldfile = pd.read_csv(OLD_CLIN)
    merged = m2_tcd.merge(oldfile[["patient_id", "prob"]], on="patient_id", suffixes=("_m2", "_old"))
    maxdiff = float(np.abs(merged["prob_m2"] - merged["prob_old"]).max())
    auc_old_file = float(roc_auc_score(oldfile["label_T"], np.clip(oldfile["prob"].values, 0, 1)))
    # 融合: M2 重训版 clin (master 折) + t/d (master 折) + 策略 B + bonly
    t_have, t_X, t_y = S.pool_branch(S.all_direct, "t", "area_w")
    t_fold = S.md_i.loc[t_have, "fold"].values.astype(int)
    t_oof, _ = S.probe_generic(t_have, t_X, t_y, t_fold)
    d_have, d_X, d_y = S.pool_branch(sorted(S.self_d), "d", "meanmax")
    d_fold = S.md_i.loc[d_have, "fold"].values.astype(int)
    d_oof, _ = S.probe_generic(d_have, d_X, d_y, d_fold)
    fused = F.fuse_zlogit_strategy({"clin": clin_m2, "t": t_oof, "d": d_oof}, strategy="B", clin_key="clin")
    cal, _ = S.calibrate_cv(fused, mode="bonly")
    cals = scope_slice(cal, S.tcd)
    auc_fus_m2clin = float(roc_auc_score(cals["label_T"], np.clip(cals["prob"].values, 0, 1)))
    # 旧 clin 版融合 (同样 master 折 + bonly) 作对照
    fused_old = F.fuse_zlogit_strategy({"clin": clin_old2, "t": t_oof, "d": d_oof}, strategy="B", clin_key="clin")
    cal_old, _ = S.calibrate_cv(fused_old, mode="bonly")
    cals_old = scope_slice(cal_old, S.tcd)
    auc_fus_oldclin = float(roc_auc_score(cals_old["label_T"], np.clip(cals_old["prob"].values, 0, 1)))
    return {
        "n_direct_train": int(len(S.all_direct)),
        "clin_cv_master_tcd_pooled_auc": round(auc_m2, 4),
        "old_pipeline_master_tcd_pooled_auc": round(auc_old, 4),
        "old_file_tcd_pooled_auc": round(auc_old_file, 4),
        "max_abs_prob_diff_m2_vs_oldfile": round(maxdiff, 4),
        "n_aligned": int(len(merged)),
        "C_m2_by_fold": {str(k): v for k, v in sorted(Cs_m2.items())},
        "C_old_by_fold": {str(k): v for k, v in sorted(Cs_old.items())},
        "fusion_bonly_tcd_pooled_auc_M2clin": round(auc_fus_m2clin, 4),
        "fusion_bonly_tcd_pooled_auc_oldclin": round(auc_fus_oldclin, 4),
        "caliber_diff_summary": ("唯一差异 = C 选择准则 (M2 clin_cv: 折内嵌套3折CV; 旧 pipeline: "
                                 "训练折 TRAIN-AUC); 特征列/中位填补/StandardScaler/C 网格/训练集(direct_all n=280) 全同"),
    }


def clin_cv_old_style(pids, folds):
    """旧 pipeline 语义复刻 (与 audit_fusion_debug/baseline 一致): 折内 median 填补 +
    StandardScaler + 每折 C 按训练折 TRAIN-AUC 选择 (C 网格 0.01/0.1/1/10)。"""
    from sklearn.impute import SimpleImputer
    pids = np.asarray(pids, dtype=np.int64)
    folds = np.asarray(folds, dtype=int)
    Xc = S.raw.loc[pids, S.rc.CLIN_COLS].values
    yc = S.md_i.loc[pids, "label_T"].values.astype(float)
    oof = np.full(len(pids), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(yc[tr])) < 2:
            oof[va] = 0.5
            Cs[int(k)] = None
            continue
        Xtr, ytr = Xc[tr], yc[tr]
        impm = SimpleImputer(strategy="median").fit(Xtr)
        Xtr_i = impm.transform(Xtr)
        sc = StandardScaler().fit(Xtr_i)
        Xtr_s = sc.transform(Xtr_i)
        best_c, best_auc = None, -1.0
        for c in S.c_grid_lr():
            mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(Xtr_s, ytr)
            a = float(roc_auc_score(ytr, mm.predict_proba(Xtr_s)[:, 1]))
            if a > best_auc:
                best_auc, best_c = a, c
        mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(Xtr_s, ytr)
        Xva_i = impm.transform(Xc[va])
        oof[va] = mm.predict_proba(sc.transform(Xva_i))[:, 1]
        Cs[int(k)] = float(best_c)
    rows = [{"patient_id": int(p), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp in zip(pids, yc, oof)]
    df = pd.DataFrame(rows)
    df["fold"] = df["patient_id"].map({int(p): int(f) for p, f in zip(pids, folds)})
    return df[["patient_id", "fold", "label_T", "prob"]].sort_values("patient_id").reset_index(drop=True), Cs


# ============ 2. M1 校准稳健补充 ============
def m1_supp():
    folds_d = S.md_i.loc[S.all_direct, "fold"].values.astype(int)
    clin_oof, _ = S.clin_cv(S.all_direct, folds_d)
    t_have, t_X, t_y = S.pool_branch(S.all_direct, "t", "area_w")
    t_fold = S.md_i.loc[t_have, "fold"].values.astype(int)
    t_oof, _ = S.probe_generic(t_have, t_X, t_y, t_fold)
    d_have, d_X, d_y = S.pool_branch(sorted(S.self_d), "d", "meanmax")
    d_fold = S.md_i.loc[d_have, "fold"].values.astype(int)
    d_oof, _ = S.probe_generic(d_have, d_X, d_y, d_fold)
    fused = F.fuse_zlogit_strategy({"clin": clin_oof, "t": t_oof, "d": d_oof}, strategy="B", clin_key="clin")
    cal_bonly, _ = S.calibrate_cv(fused, mode="bonly")
    lock = scope_slice(cal_bonly, S.tcd)
    lock["prob"] = np.clip(lock["prob"].values, 0, 1)
    return {
        "n": int(len(lock)),
        "n_pos": int(lock.label_T.sum()),
        "true_t_rate": round(float(lock.label_T.mean()), 4),
        "mean_pred": round(float(lock.prob.mean()), 4),
        "calibration_in_the_large_gap": round(float(lock.prob.mean() - lock.label_T.mean()), 4),
        "calibration_slope": slope_and_ci(lock),
        "ece_quantile_5": ece_quantile(lock, 5),
        "ece_quantile_10": ece_quantile(lock, 10),
        "ece_quantile_10_suite_json": 0.0797,
        "note": "校准列 = M1 锁定 bonly 单口径概率 (direct_all 训练折拟合校准层, tcd 162 评估)",
    }


# ============ 3. M3 时间外推明细 (复刻 module3 deploy, 补 ECE5/计数/CI) ============
def m3_deploy_all():
    dj = json.load(open(S.DICOM, encoding="utf-8"))
    sdate = {}
    pidplane = re.compile(r"^(\d+)_([tcd])$")
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
            p = S.raw.loc[pid, "mri_path_" + plane] if pid in S.raw.index else None
            pp = fname_pid(p) if isinstance(p, str) else None
            sd = sdate.get((pp, plane)) if pp is not None else None
            if sd:
                return sd
        return None

    dated = []
    for pid in S.all_direct:
        sd = get_date(int(pid))
        if sd:
            dated.append((int(pid), sd))
    dated.sort(key=lambda x: x[1])
    cut = int(round(0.70 * len(dated)))
    tr_pids, te_pids = [p for p, _ in dated[:cut]], [p for p, _ in dated[cut:]]
    tr_tcd, te_tcd = [p for p in tr_pids if p in S.tcd], [p for p in te_pids if p in S.tcd]

    def deploy(scope_tr, scope_te):
        tr_arr_s = np.array(scope_tr, dtype=np.int64)
        te_arr_s = np.array(scope_te, dtype=np.int64)
        p_clin_tr, p_clin_te, _ = S.clin_fit_predict(tr_arr_s, te_arr_s)
        have_t, X_t, y_t = S.pool_branch(S.all_direct, "t", "area_w")
        tpos = {int(p): i for i, p in enumerate(have_t)}
        ids_tr = np.array([p for p in tr_arr_s if p in tpos], dtype=np.int64)
        ids_te = np.array([p for p in te_arr_s if p in tpos], dtype=np.int64)
        p_t_tr, p_t_te, _ = S.probe_fit_predict(
            ids_tr, np.array([X_t[tpos[p]] for p in ids_tr]), np.array([y_t[tpos[p]] for p in ids_tr]),
            ids_te, np.array([X_t[tpos[p]] for p in ids_te]))
        have_d, X_d, y_d = S.pool_branch(sorted(S.self_d), "d", "meanmax")
        dpos = {int(p): i for i, p in enumerate(have_d)}
        ids_d_tr = np.array([p for p in tr_arr_s if p in dpos], dtype=np.int64)
        ids_d_te = np.array([p for p in te_arr_s if p in dpos], dtype=np.int64)
        if len(ids_d_tr) >= 10:
            p_d_tr, p_d_te, _ = S.probe_fit_predict(
                ids_d_tr, np.array([X_d[dpos[p]] for p in ids_d_tr]), np.array([y_d[dpos[p]] for p in ids_d_tr]),
                ids_d_te, np.array([X_d[dpos[p]] for p in ids_d_te]))
        else:
            p_d_tr, p_d_te = None, None
        branch_probs = {"clin": (tr_arr_s, p_clin_tr, te_arr_s, p_clin_te),
                        "t": (ids_tr, p_t_tr, ids_te, p_t_te)}
        if p_d_tr is not None:
            branch_probs["d"] = (ids_d_tr, p_d_tr, ids_d_te, p_d_te)

        def fuse_deploy(bps, which):
            zs = {}
            for nm, (pids_tr, pr_tr, pids_te, pr_te) in bps.items():
                L_tr = tsm.logit_clip(np.asarray(pr_tr, dtype=float))
                mu, sd = L_tr.mean(), L_tr.std() + 1e-9
                if which == "tr":
                    zs[nm] = pd.Series((L_tr - mu) / sd, index=np.asarray(pids_tr, dtype=np.int64))
                else:
                    L_te = tsm.logit_clip(np.asarray(pr_te, dtype=float))
                    zs[nm] = pd.Series((L_te - mu) / sd, index=np.asarray(pids_te, dtype=np.int64))
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

        models = {}
        for mname, bps in (("clin", {"clin": branch_probs["clin"]}),
                           ("clin_t", {"clin": branch_probs["clin"], "t": branch_probs["t"]}),
                           ("clin_dt", branch_probs)):
            f_tr = fuse_deploy(bps, "tr")
            f_te = fuse_deploy(bps, "te")
            f_tr["label_T"] = f_tr.patient_id.map(S.md_i["label_T"]).astype(int)
            f_te["label_T"] = f_te.patient_id.map(S.md_i["label_T"]).astype(int)
            a, b = S.platt_fit(f_tr["prob"], f_tr["label_T"])
            f_tr["prob"] = S.sigmoid(a * f_tr["prob"] + b)
            f_te["prob"] = S.sigmoid(a * f_te["prob"] + b)
            f_tr["fold"] = 0
            f_te["fold"] = 1
            models[mname] = {"train": f_tr, "test": f_te, "platt": (round(a, 5), round(b, 5))}
        out = {}
        for mname, mm in models.items():
            f_te = mm["test"].dropna(subset=["prob"]).reset_index(drop=True)
            f_te["prob"] = np.clip(f_te["prob"].values, 0, 1)
            f_tr = mm["train"].dropna(subset=["prob"]).reset_index(drop=True)
            f_tr["prob"] = np.clip(f_tr["prob"].values, 0, 1)
            fpr, tpr, ths = roc_curve(f_tr["label_T"], f_tr["prob"])
            J = tpr - fpr
            th = ths[np.argmax(J)]
            tp = ((f_te["prob"] >= th) & (f_te["label_T"] == 1)).sum()
            fp = ((f_te["prob"] >= th) & (f_te["label_T"] == 0)).sum()
            tn = ((f_te["prob"] < th) & (f_te["label_T"] == 0)).sum()
            fn = ((f_te["prob"] < th) & (f_te["label_T"] == 1)).sum()
            sens = tp / (tp + fn) if (tp + fn) else None
            spec = tn / (tn + fp) if (tn + fp) else None
            ppv = tp / (tp + fp) if (tp + fp) else None
            npv = tn / (tn + fn) if (tn + fn) else None
            mets = S.metrics_of(f_te)
            out[mname] = {
                "test_n": int(len(f_te)),
                "test_n_pos": int(f_te.label_T.sum()),
                "test_n_neg": int((1 - f_te.label_T).sum()),
                "test_d_self_n": int(len(ids_d_te)),
                "pooled_auc": mets["pooled_auc"],
                "boot_95ci": mets["boot_95ci"],
                "brier": mets["brier"],
                "ece_quantile_5": ece_quantile(f_te, 5),
                "ece_quantile_10": ece_quantile(f_te, 10),
                "train_youden_threshold": round(float(th), 4),
                "train_youden_j": round(float(J.max()), 4),
                "test_at_train_youden": {"sens": round(float(sens), 4) if sens is not None else None,
                                         "spec": round(float(spec), 4) if spec is not None else None,
                                         "ppv": round(float(ppv), 4) if ppv is not None else None,
                                         "npv": round(float(npv), 4) if npv is not None else None},
                "platt": list(mm["platt"]),
                "train_n": int(len(f_tr)),
            }
        return out

    return {"direct_all": deploy(tr_pids, te_pids),
            "direct_tcd": deploy(tr_tcd, te_tcd),
            "meta": {"date_source": "DICOM dicom_meta StudyDate",
                     "cutoff_date": dated[cut][1] if cut < len(dated) else None,
                     "dated_n": int(len(dated)),
                     "train_n": len(tr_pids), "test_n": len(te_pids),
                     "tcd_train_n": len(tr_tcd), "tcd_test_n": len(te_tcd),
                     "missing_date_excluded": int(len(S.all_direct) - len(dated))}}


# ============ 4. M4 ncrt 域偏移 (逐模型 + 配对 CI) ============
def m4_models():
    folds_d = S.md_i.loc[S.all_direct, "fold"].values.astype(int)
    clin_oof, _ = S.clin_cv(S.all_direct, folds_d)
    t_have, t_X, t_y = S.pool_branch(S.all_direct, "t", "area_w")
    t_fold = S.md_i.loc[t_have, "fold"].values.astype(int)
    t_oof, _ = S.probe_generic(t_have, t_X, t_y, t_fold)
    d_have, d_X, d_y = S.pool_branch(sorted(S.self_d), "d", "meanmax")
    d_fold = S.md_i.loc[d_have, "fold"].values.astype(int)
    d_oof, _ = S.probe_generic(d_have, d_X, d_y, d_fold)
    direct_oofs = {"clin": clin_oof, "t": t_oof, "d": d_oof}
    # 部署分支 (master 全表池化)
    ncrt_arr = np.array(sorted(S.ncrt_ids), dtype=np.int64)
    all_master = np.array(sorted(S.master["patient_id"].astype(np.int64)), dtype=np.int64)
    p_clin_tr_all, p_clin_ncrt, _ = S.clin_fit_predict(S.all_direct, ncrt_arr)
    have_tm, X_tm, y_tm = S.pool_branch(all_master, "t", "area_w", frame=S.master, use_cache=False)
    tpos = {int(p): i for i, p in enumerate(have_tm)}
    t_tr_pids = np.array([p for p in S.all_direct if p in tpos], dtype=np.int64)
    ncrt_t = np.array([p for p in ncrt_arr if p in tpos], dtype=np.int64)
    p_t_tr, p_t_ncrt, _ = S.probe_fit_predict(
        t_tr_pids, np.array([X_tm[tpos[p]] for p in t_tr_pids]),
        np.array([y_tm[tpos[p]] for p in t_tr_pids]),
        ncrt_t, np.array([X_tm[tpos[p]] for p in ncrt_t]))
    ncrt_self = set(S.sm.loc[(S.sm.cohort_direct == 0) & (S.sm.mask_source == "self"), "pid"])
    d_master_pids = np.array(sorted(set(S.self_d) | set(ncrt_self)), dtype=np.int64)
    have_dm, X_dm, y_dm = S.pool_branch(d_master_pids, "d", "meanmax", frame=S.master, use_cache=False)
    dpos = {int(p): i for i, p in enumerate(have_dm)}
    d_tr_pids = np.array([p for p in S.all_direct if p in dpos], dtype=np.int64)
    ncrt_d = np.array([p for p in ncrt_arr if p in dpos], dtype=np.int64)
    if len(ncrt_d) >= 2:
        p_d_tr, p_d_ncrt, _ = S.probe_fit_predict(
            d_tr_pids, np.array([X_dm[dpos[p]] for p in d_tr_pids]),
            np.array([y_dm[dpos[p]] for p in d_tr_pids]),
            ncrt_d, np.array([X_dm[dpos[p]] for p in ncrt_d]))
    else:
        p_d_tr, p_d_ncrt = None, None
    branch_tr = {"clin": (S.all_direct, p_clin_tr_all), "t": (t_tr_pids, p_t_tr), "d": (d_tr_pids, p_d_tr)}
    branch_te = {"clin": (ncrt_arr, p_clin_ncrt), "t": (ncrt_t, p_t_ncrt), "d": (ncrt_d, p_d_ncrt)}
    m_pT = S.master.set_index("patient_id")["pT"]

    def deploy_variant(vname, brs):
        fz = F.fuse_zlogit_strategy({n: direct_oofs[n] for n in brs}, strategy="B", clin_key="clin")
        a, b = S.platt_fit(fz["prob"], fz["label_T"])
        ztr, zte = {}, {}
        for n in brs:
            L_tr = tsm.logit_clip(np.asarray(branch_tr[n][1], dtype=float))
            mu, sd = L_tr.mean(), L_tr.std() + 1e-9
            ztr[n] = pd.Series((L_tr - mu) / sd, index=np.asarray(branch_tr[n][0], dtype=np.int64))
            L_te = tsm.logit_clip(np.asarray(branch_te[n][1], dtype=float))
            zte[n] = pd.Series((L_te - mu) / sd, index=np.asarray(branch_te[n][0], dtype=np.int64))
        pids_all = sorted(set().union(*[set(z.index) for z in zte.values()]))
        rows = []
        for pid in pids_all:
            comps = {n: z.loc[pid] for n, z in zte.items() if pid in z.index}
            zc = comps.get("clin")
            imgs = [z for n, z in comps.items() if n != "clin"]
            if zc is None or not imgs:
                v = float(np.mean(list(comps.values())))
            else:
                v = 0.5 * float(zc) + 0.5 * float(np.mean(imgs))
            rows.append({"patient_id": int(pid), "prob": v})
        df = pd.DataFrame(rows)
        df["prob"] = S.sigmoid(a * df["prob"].values + b)
        df["label_T"] = df.patient_id.map(S.master.set_index("patient_id")["label_T"]).astype(int)
        df["pT"] = df.patient_id.map(m_pT)
        df = df.dropna(subset=["prob"]).reset_index(drop=True)
        df["prob"] = np.clip(df["prob"].values, 0, 1)
        df["fold"] = 0
        mets = S.metrics_of(df)
        ypt2 = df[df["pT"] == 2]
        ypt34 = df[df["pT"] >= 3]
        out = {
            "n": int(len(df)), "n_pos": int(df.label_T.sum()),
            "pooled_auc": mets["pooled_auc"], "boot_95ci": mets["boot_95ci"],
            "brier": mets["brier"], "ece_quantile_5": ece_quantile(df, 5),
            "mean_pred": round(float(df.prob.mean()), 4),
            "true_t_rate": round(float(df.label_T.mean()), 4),
            "platt_deploy": [round(a, 5), round(b, 5)],
            "ypT2": {"n": int(len(ypt2)),
                     "median_prob": round(float(np.median(ypt2.prob)), 4) if len(ypt2) else None,
                     "iqr": [round(float(x), 4) for x in np.percentile(ypt2.prob, [25, 75])] if len(ypt2) else None},
            "ypT34": {"n": int(len(ypt34)),
                      "median_prob": round(float(np.median(ypt34.prob)), 4) if len(ypt34) else None,
                      "iqr": [round(float(x), 4) for x in np.percentile(ypt34.prob, [25, 75])] if len(ypt34) else None},
        }
        return out

    # 配对 n=34 (ncrt) 与 direct 内部配对 (92)
    pair_pids = set(ncrt_t) & set(ncrt_d)
    direct_pair = set(S.self_d) & set(t_tr_pids)
    t_single = pd.DataFrame({"patient_id": ncrt_t, "prob": np.clip(p_t_ncrt, 0, 1)})
    t_single["label_T"] = t_single.patient_id.map(S.master.set_index("patient_id")["label_T"]).astype(int)
    t_pair = t_single[t_single.patient_id.isin(pair_pids)].reset_index(drop=True)
    d_single = pd.DataFrame({"patient_id": ncrt_d, "prob": np.clip(p_d_ncrt, 0, 1)})
    d_single["label_T"] = d_single.patient_id.map(S.master.set_index("patient_id")["label_T"]).astype(int)
    d_pair = d_single[d_single.patient_id.isin(pair_pids)].reset_index(drop=True)
    t_oof_dp = t_oof[t_oof.patient_id.isin(direct_pair)].reset_index(drop=True)
    d_oof_dp = d_oof[d_oof.patient_id.isin(direct_pair)].reset_index(drop=True)

    def pair_stats(name, dfp):
        if dfp is None or len(dfp) < 2 or len(np.unique(dfp.label_T)) < 2:
            return {"n": int(len(dfp)) if dfp is not None else 0, "auc": None, "boot_95ci": None}
        p = np.clip(dfp.prob.values, 0, 1)
        auc = float(roc_auc_score(dfp.label_T, p))
        ci = epm.boot_ci(dfp)
        return {"n": int(len(dfp)), "auc": round(auc, 4),
                "boot_95ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}

    return {
        "variants": {v: deploy_variant(v, brs) for v, brs in
                     (("clin", ["clin"]), ("clin_t", ["clin", "t"]), ("clin_dt", ["clin", "t", "d"]))},
        "paired": {
            "ncrt_paired_n": int(len(pair_pids)),
            "direct_paired_n": int(len(direct_pair)),
            "t_direct": pair_stats("t_direct", t_oof_dp),
            "t_ncrt": pair_stats("t_ncrt", t_pair),
            "d_direct": pair_stats("d_direct", d_oof_dp),
            "d_ncrt": pair_stats("d_ncrt", d_pair),
            "t_attenuation_auc": round(float(roc_auc_score(t_oof_dp.label_T, np.clip(t_oof_dp.prob.values, 0, 1)) -
                                            roc_auc_score(t_pair.label_T, np.clip(t_pair.prob.values, 0, 1))), 4),
            "d_attenuation_auc": round(float(roc_auc_score(d_oof_dp.label_T, np.clip(d_oof_dp.prob.values, 0, 1)) -
                                            roc_auc_score(d_pair.label_T, np.clip(d_pair.prob.values, 0, 1))), 4),
            "note": "exploratory small-n; report absolute AUC + CI, no significance claims",
        },
    }


def main():
    res = {"task": "m02069 快速明细核查与口径一致性诊断 (只读)",
           "locked_calibration_mode": "bonly"}
    print("=== Part 1: M2 完整对照表 (10×5 重跑) ===")
    res["m2_full"] = m2_full()
    for scope in ("direct_tcd", "direct_td"):
        rep = res["m2_full"][scope]
        print(scope, {m: (rep[m]["repeated_pooled_auc_mean"], rep[m]["repeated_pooled_auc_sd_ddof1"])
                      for m in ("clin", "t", "d_self", "clin_t", "clin_dt")})
    print("=== Part 1b: 口径一致性 (固定 master 折) ===")
    res["caliber_check"] = caliber_check()
    print(res["caliber_check"])
    print("=== Part 2: M1 校准稳健补充 ===")
    res["m1_supp"] = m1_supp()
    print(res["m1_supp"])
    print("=== Part 3: M3 时间外推明细 ===")
    res["m3_deploy"] = m3_deploy_all()
    for scope in ("direct_all", "direct_tcd"):
        print(scope, {m: res["m3_deploy"][scope][m]["pooled_auc"] for m in ("clin", "clin_t", "clin_dt")})
    print("=== Part 4: M4 ncrt 逐模型 ===")
    res["m4_models"] = m4_models()
    print({v: res["m4_models"]["variants"][v]["pooled_auc"] for v in ("clin", "clin_t", "clin_dt")})
    print("paired:", res["m4_models"]["paired"])
    with open(AOUT, "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print("saved ->", AOUT)


if __name__ == "__main__":
    main()
