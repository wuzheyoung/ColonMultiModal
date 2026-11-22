# -*- coding: utf-8 -*-
"""m04813 四任务纯 CPU 排错审计与干净消融 (严禁 GPU/改数据/重新提取)。
任务1: clin nan_to_num 顺序核查 + SimpleImputer(median) 干净基线
任务2: d_conv N_direct=0.2862 逐折排查与归因
任务3: d_conv 全零 mask 回退逻辑 + 分群 T AUC
任务4: fusionA 暴跌原因 + 干净阶梯晚融合消融 (z-score logit 均值 / rank 均值)
产物: logs/audit_fusion_debug.json (+ --readme 追加 README)
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MASTER_CSV = "data/master_5fold_356.csv"
PAIRS_RAW = "data/hos_pairs_raw.csv"
X929 = "data/Hos/结直肠癌多模态患者数据收集表9.29.xlsx"
XFULL = "data/Hos/结直肠癌多模态患者数据收集表_完整版.xlsx"
R3A_CSV = "logs/route3a_radiomics_features.csv"
R3B = {"c": "logs/route3b_c_features.csv", "d": "logs/route3b_d_features.csv"}
R2T_DIR = "logs/route2_feats"
R2CD_DIR = "logs/route2_cd_feats"
CLIN_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
OUT_JSON = "logs/audit_fusion_debug.json"


def load_tgr_map():
    d929 = pd.read_excel(X929)
    dfull = pd.read_excel(XFULL)
    m = {}
    for _, r in d929.iterrows():
        if pd.notna(r["病理TGR"]):
            try:
                m[int(r["编号"]) + 200] = float(r["病理TGR"])
            except (ValueError, TypeError):
                pass
    for i, r in dfull.iterrows():
        if pd.notna(r["病理TGR"]):
            try:
                m[i + 1] = float(r["病理TGR"])
            except (ValueError, TypeError):
                pass
    return m


def rad_feats(csv, mtype="body"):
    df = pd.read_csv(csv)
    df = df[(df["mask_type"] == mtype) & (df["ok"] == 1)]
    feat = [c for c in df.columns if c not in
            ("patient_id", "mask_type", "ok", "label_T", "label_N", "fold")]
    return df.set_index("patient_id")[feat]


def conv_feats_dir(d):
    rows = []
    for fn in os.listdir(d):
        if not fn.endswith(".npy"):
            continue
        pid = int(fn.replace(".npy", ""))
        rows.append((pid, np.load(os.path.join(d, fn)).mean(axis=0)))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])


def conv_feats_npy():
    d_all = np.load(os.path.join(R2CD_DIR, "d.npy"))
    d_pids = pd.read_csv(os.path.join(R2CD_DIR, "d_pids.csv"))["patient_id"].values
    c_all = np.load(os.path.join(R2CD_DIR, "c.npy"))
    c_pids = pd.read_csv(os.path.join(R2CD_DIR, "c_pids.csv"))["patient_id"].values
    return (pd.DataFrame(d_all, index=d_pids), pd.DataFrame(c_all, index=c_pids))


def per_fold_lr(X, y, folds, C_grid=(0.01, 0.1, 1.0, 10.0), imp=None):
    """折内 imp(median 或 None=nan_to_num) → scaler → 折内选 C → 返回逐折明细+oof。
    返回 (oof, detail[])  detail: fold,n_tr,n_va,tr_pos,va_pos,va_p_mean,va_p_std,C,val_auc"""
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    detail = []
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 2 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            detail.append({"fold": int(k), "n_tr": int(tr.sum()), "n_va": int(va.sum()),
                           "skip": "degenerate"})
            continue
        if imp is not None:
            ip = SimpleImputer(strategy="median").fit(X[tr])
            Xtr = ip.transform(X[tr]); Xva = ip.transform(X[va])
        else:
            Xtr = np.nan_to_num(X[tr].astype(float), nan=0.0)
            Xva = np.nan_to_num(X[va].astype(float), nan=0.0)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
        best_c, best_a, best_m = C_grid[0], -1, None
        for c in C_grid:
            m = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(Xtr_s, y[tr])
            a = roc_auc_score(y[tr], m.predict_proba(Xtr_s)[:, 1])
            if a > best_a:
                best_a, best_c, best_m = a, c, m
        pv = best_m.predict_proba(Xva_s)[:, 1]
        oof[va] = pv
        try:
            va_auc = roc_auc_score(y[va], pv)
        except ValueError:
            va_auc = None
        detail.append({"fold": int(k), "n_tr": int(tr.sum()), "n_va": int(va.sum()),
                       "tr_pos": float(y[tr].mean()), "va_pos": float(y[va].mean()),
                       "va_p_mean": float(pv.mean()), "va_p_std": float(pv.std()),
                       "C": best_c, "val_auc": va_auc})
    return oof, detail


def pooled_auc(y, p):
    if p is None or np.isnan(p).any() or len(np.unique(y)) < 2:
        return None
    try:
        return round(float(roc_auc_score(y, p)), 4)
    except ValueError:
        return None


def fold_mean_auc(y, p, folds):
    fa = []
    for k in sorted(set(folds)):
        va = folds == k
        if va.sum() < 2 or len(np.unique(y[va])) < 2:
            continue
        try:
            fa.append(roc_auc_score(y[va], p[va]))
        except ValueError:
            pass
    fa = np.array(fa)
    if len(fa) == 0:
        return None, None
    return round(float(fa.mean()), 4), round(float(fa.std()), 4)


def zscore_pooled(y, p, folds):
    """折内 z-score 校准后 pooled AUC: 每折 val 概率按该折 val 的 mean/std 校准再合并"""
    pz = np.full_like(p, np.nan)
    for k in sorted(set(folds)):
        va = folds == k
        if va.sum() < 2 or np.isnan(p[va]).all():
            continue
        s = p[va]
        m_, sd_ = s.mean(), s.std() + 1e-9
        pz[va] = (s - m_) / sd_
    return pooled_auc(y, pz)


def logit_clip(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    master = pd.read_csv(MASTER_CSV)
    raw = pd.read_csv(PAIRS_RAW).set_index("patient_id")
    tgr = load_tgr_map()
    direct = master[~master["patient_id"].isin(tgr)].copy()
    d_ids = np.sort(direct["patient_id"].values.astype(np.int64))

    # ---- 分支特征 ----
    rad_t = rad_feats(R3A_CSV, "body")
    rad_d = rad_feats(R3B["d"], "body")
    rad_c = rad_feats(R3B["c"], "body")
    conv_t = conv_feats_dir(R2T_DIR)
    conv_d, conv_c = conv_feats_npy()
    clin_raw = raw[CLIN_COLS].replace([np.inf, -np.inf], np.nan).astype(float)
    branches = {"t_rad": rad_t, "t_conv": conv_t, "d_rad": rad_d, "d_conv": conv_d,
                "c_rad": rad_c, "c_conv": conv_c, "clin": clin_raw}
    res = {}

    # ============ 任务 1: clin 填补顺序核查 + 干净基线 ============
    t1 = {}
    # 上一轮脚本: audit_crossplane_fusion.py:124 clin 保留 NaN; oof_prob_lr:76
    # X = np.nan_to_num(X.astype(float), nan=0.0) 在 StandardScaler fit 之前, 折外常量变换无泄漏
    t1["order_prev"] = ("nan_to_num(0.0) 在 oof_prob_lr 内于 StandardScaler.fit 之前执行; "
                        "是全局固定变换(非折内拟合), 无信息泄漏但 0.0 是任意占位")
    miss = {}
    for c in CLIN_COLS:
        miss[c] = int(clin_raw.loc[d_ids, c].isna().sum())
    t1["direct_missing"] = miss
    t1["direct_n"] = len(d_ids)
    t1["missing_rate"] = {c: round(v / len(d_ids), 4) for c, v in miss.items()}
    # 重跑: 旧口径 vs 干净 SimpleImputer(median) 折内拟合
    Xc = clin_raw.loc[d_ids].values.astype(float)
    yT = master.set_index("patient_id").loc[d_ids, "label_T"].values.astype(int)
    folds = master.set_index("patient_id").loc[d_ids, "fold"].values.astype(int)
    p_old, _ = per_fold_lr(Xc, yT, folds, imp=None)
    p_clean, _ = per_fold_lr(Xc, yT, folds, imp="median")
    fm_o, sd_o = fold_mean_auc(yT, p_old, folds)
    fm_c, sd_c = fold_mean_auc(yT, p_clean, folds)
    t1["nan_to_num"] = {"pooled": pooled_auc(yT, p_old), "fold_mean": fm_o, "fold_sd": sd_o}
    t1["imputer_median"] = {"pooled": pooled_auc(yT, p_clean), "fold_mean": fm_c, "fold_sd": sd_c}
    res["task1_clin"] = t1
    print(f"[T1] clin 缺失 {miss}; 旧口径 pooled {t1['nan_to_num']['pooled']} "
          f"fold {fm_o}±{sd_o} | 干净 pooled {t1['imputer_median']['pooled']} fold {fm_c}±{sd_c}")

    # ============ 任务 2: d_conv N_direct = 0.2862 逐折 ============
    t2 = {}
    ids_d = np.array(sorted(set(d_ids) & set(conv_d.index)))
    yN = master.set_index("patient_id").loc[ids_d, "label_N"].values.astype(int)
    fN = master.set_index("patient_id").loc[ids_d, "fold"].values.astype(int)
    Xd = conv_d.loc[ids_d].values.astype(float)
    pN, detN = per_fold_lr(Xd, yN, fN)
    t2["n"] = len(ids_d); t2["pos"] = int(yN.sum()); t2["pos_rate"] = float(yN.mean())
    t2["per_fold"] = detN
    t2["pooled_raw"] = pooled_auc(yN, pN)
    fmN, sdN = fold_mean_auc(yN, pN, fN)
    t2["fold_mean"] = fmN; t2["fold_sd"] = sdN
    t2["pooled_zscore"] = zscore_pooled(yN, pN, fN)
    # 单折 AUC 列表 (含 nan=退化折)
    t2["fold_auc_list"] = [d.get("val_auc") for d in detN]
    res["task2_dconv_N_direct"] = t2
    print(f"[T2] d_conv N direct n={len(ids_d)} pos={int(yN.sum())} "
          f"pooled {t2['pooled_raw']} fold {fmN}±{sdN} zscore-pooled {t2['pooled_zscore']}")
    for d in detN:
        print(f"   fold{d['fold']}: n_tr={d.get('n_tr')} n_va={d.get('n_va')} "
              f"tr_pos={d.get('tr_pos')} va_pos={d.get('va_pos')} "
              f"va_p={d.get('va_p_mean')}±{d.get('va_p_std')} C={d.get('C')} auc={d.get('val_auc')}")

    # ============ 任务 3: d_conv 全零 mask 回退 + 分群 T AUC ============
    t3 = {}
    t3["fallback_logic"] = (
        "audit_tgr_task2c.py extract_case_feats: mask 文件不存在→None; 逐 z 层 mask.sum()==0 跳过; "
        "若所有层全零(138 例) feats 为空→返回 None→该例不进 d.npy/d_pids.csv, "
        "即 d_conv 特征仅覆盖 mask 非零病例; 全零病例在 fusion 中作为缺失模态(无 d 特征可预测)")
    # direct 组内: 有 d 特征 vs 无 d 特征(mask 全零回退) 拆分
    has_d = np.array([p in set(ids_d) for p in d_ids])
    sub_d = direct.iloc[np.where(has_d)[0]]
    sub_no = direct.iloc[np.where(~has_d)[0]]
    t3["has_d_n"] = int(has_d.sum()); t3["has_d_Tpos_rate"] = float(sub_d["label_T"].mean())
    t3["no_d_n"] = int((~has_d).sum()); t3["no_d_Tpos_rate"] = float(sub_no["label_T"].mean())
    # d_conv T AUC (有特征子集)
    yTd = master.set_index("patient_id").loc[ids_d, "label_T"].values.astype(int)
    pTd, detTd = per_fold_lr(Xd, yTd, fN)
    t3["dconv_T_has_d"] = {"pooled": pooled_auc(yTd, pTd)}
    fmT, sdT = fold_mean_auc(yTd, pTd, fN)
    t3["dconv_T_has_d"]["fold_mean"] = fmT; t3["dconv_T_has_d"]["fold_sd"] = sdT
    t3["dconv_T_has_d"]["per_fold"] = detTd
    res["task3_dconv_mask_split"] = t3
    print(f"[T3] has_d {int(has_d.sum())} (T+率 {float(sub_d['label_T'].mean()):.3f}) / "
          f"no_d {int((~has_d).sum())} (T+率 {float(sub_no['label_T'].mean()):.3f}); "
          f"d_conv T pooled {t3['dconv_T_has_d']['pooled']} fold {fmT}±{sdT}")

    # ============ 任务 4: fusionA 暴跌审计 + 干净阶梯消融 ============
    t4 = {"debug": {}, "ablation": {}}
    # 4.1 暴跌审计
    t4["debug"]["fusionA_branches"] = (
        "上一轮 fusionA img_names = t_rad, t_conv, d_rad, d_conv (代码 :162), 混入 d_rad(direct_all 单路 0.4271, "
        "direct_td 0.4750 均反向/弱) + clin; 缺失分支跳过(pa 初始 nan 仅累加可用) 未填 0.0; "
        "但各分支概率直接算术平均, 无折内尺度对齐 → 反信号分支拉低均值")
    t4["debug"]["fusionB_branches"] = (
        "上一轮 fusionB 用 t_rad/t_conv/d_rad/d_conv/clin 概率 + has 指示符折内 LR(C=0.1); "
        "同样混入 d_rad 反信号列")
    # 4.2 三集合定义 (真实路径交集: hos_pairs_raw mri_path_* notna)
    rp = raw.copy()
    has_p = {s: set(rp.index[rp[f"mri_path_{s}"].notna()]) for s in ("t", "c", "d")}
    scope_defs = {
        "direct_all": set(d_ids),
        "direct_td": set(d_ids) & has_p["t"] & has_p["d"],
        "direct_tcd": set(d_ids) & has_p["t"] & has_p["c"] & has_p["d"],
    }
    # 干净 clin (imputer median) OOF 概率 (direct_all 上)
    clin_clean_oof, _ = per_fold_lr(Xc, yT, folds, imp="median")
    branches["clin_clean"] = None  # 占位: 用上面算好的

    def oof_for_branch(name, ids):
        if name == "clin_clean":
            return clin_clean_oof[np.searchsorted(d_ids, ids)], ids
        fdf = branches[name]
        ids = np.array(sorted(set(ids) & set(fdf.index)))
        if len(ids) < 8:
            return None, None
        yl = master.set_index("patient_id").loc[ids, "label_T"].values.astype(int)
        fl = master.set_index("patient_id").loc[ids, "fold"].values.astype(int)
        X = fdf.loc[ids].values.astype(float)
        if name == "clin":
            p, _ = per_fold_lr(X, yl, fl, imp=None)
        else:
            p, _ = per_fold_lr(X, yl, fl, imp=None)
        return p, ids

    def fusion_variants(scope_name, branch_names):
        """方式1 z-score logit 均值 (按折内训练集 logit 统计量校准 val); 方式2 rank 均值;
        仅对样本实际可用分支等权平均。"""
        ids = np.array(sorted(scope_defs[scope_name]))
        y = master.set_index("patient_id").loc[ids, "label_T"].values.astype(int)
        fl = master.set_index("patient_id").loc[ids, "fold"].values.astype(int)
        n = len(ids)
        single = {}
        for nm in branch_names:
            p, bid = oof_for_branch(nm, ids)
            if p is not None:
                single[nm] = (bid, p)
        # 方式1: 每分支每折 train logit 统计量校准后等权平均
        F1 = np.full(n, np.nan); C1 = np.zeros(n)
        for nm, (bid, p) in single.items():
            floc = master.set_index("patient_id").loc[bid, "fold"].values.astype(int)
            yloc = master.set_index("patient_id").loc[bid, "label_T"].values.astype(int)
            Lva = logit_clip(p)
            cal = np.full(len(bid), np.nan)
            for k in np.unique(floc):
                va = floc == k
                tr = ~va
                if tr.sum() < 2 or len(np.unique(yloc[tr])) < 2:
                    cal[va] = Lva[va]
                    continue
                # 训练集 logit: 用同一模型在训练集上的概率 (clin_clean 折内 median 填补)
                if nm == "clin_clean":
                    Xb = SimpleImputer(strategy="median").fit_transform(
                        clin_raw.loc[bid].values.astype(float))
                    ip = SimpleImputer(strategy="median").fit(Xb[tr])
                    Xb_tr = ip.transform(Xb[tr]); Xb_va = ip.transform(Xb[va])
                else:
                    Xb = branches[nm].loc[bid].values.astype(float)
                    Xb_tr, Xb_va = Xb[tr], Xb[va]
                sc = StandardScaler().fit(Xb_tr)
                m = LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs").fit(sc.transform(Xb_tr), yloc[tr])
                Ptr = m.predict_proba(sc.transform(Xb_tr))[:, 1]
                Ltr = logit_clip(Ptr)
                m_tr, s_tr = Ltr.mean(), Ltr.std() + 1e-9
                cal[va] = (Lva[va] - m_tr) / s_tr
            idx = np.searchsorted(ids, bid)
            for ii, i in enumerate(idx):
                if not np.isnan(cal[ii]):
                    F1[i] = cal[ii] if np.isnan(F1[i]) else F1[i] + cal[ii]
                    C1[i] += 1
        m1 = C1 > 0
        F1[m1] /= C1[m1]
        # 方式2: rank 均值 (每分支 val 概率在分支内百分位秩)
        F2 = np.full(n, np.nan); C2 = np.zeros(n)
        for nm, (bid, p) in single.items():
            r = rankdata(p) / len(p)
            idx = np.searchsorted(ids, bid)
            for ii, i in enumerate(idx):
                if not np.isnan(p[ii]):
                    F2[i] = r[ii] if np.isnan(F2[i]) else F2[i] + r[ii]
                    C2[i] += 1
        m2 = C2 > 0
        F2[m2] /= C2[m2]
        out = {"n": int(n), "T+": int(y.sum())}
        for nm, (bid, p) in single.items():
            yl = master.set_index("patient_id").loc[bid, "label_T"].values.astype(int)
            out[nm] = {"pooled": pooled_auc(yl, p)}
            fm, sd = fold_mean_auc(yl, p, master.set_index("patient_id").loc[bid, "fold"].values.astype(int))
            out[nm]["fold_mean"] = fm; out[nm]["fold_sd"] = sd
        def auc_ignoring_nan(yv, pv, fv):
            ok = ~np.isnan(pv)
            if ok.sum() < 10 or len(np.unique(yv[ok])) < 2:
                return None, None, None
            a = pooled_auc(yv[ok], pv[ok])
            fm, sd = fold_mean_auc(yv[ok], pv[ok], fv[ok])
            return a, fm, sd
        out["f1_zlogit_avg"] = {}
        a, fm, sd = auc_ignoring_nan(y, F1, fl)
        out["f1_zlogit_avg"]["pooled"] = a
        out["f1_zlogit_avg"]["fold_mean"] = fm; out["f1_zlogit_avg"]["fold_sd"] = sd
        out["f1_zlogit_avg"]["n_valid"] = int((~np.isnan(F1)).sum())
        out["f2_rank_avg"] = {}
        a, fm, sd = auc_ignoring_nan(y, F2, fl)
        out["f2_rank_avg"]["pooled"] = a
        out["f2_rank_avg"]["fold_mean"] = fm; out["f2_rank_avg"]["fold_sd"] = sd
        out["f2_rank_avg"]["n_valid"] = int((~np.isnan(F2)).sum())
        return out

    # 阶梯消融 (剔除 d_rad, 用干净 clin_clean)
    steps = {
        "s1_singles": ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"],
        "s2_clin_dconv": ["clin_clean", "d_conv"],
        "s3_img_3plane": ["d_conv", "t_conv", "c_conv"],
        "s4_clin_3plane": ["clin_clean", "d_conv", "t_conv", "c_conv"],
        "s5_six": ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"],
    }
    for scope_name in ("direct_all", "direct_td", "direct_tcd"):
        t4["ablation"][scope_name] = {}
        for step, names in steps.items():
            t4["ablation"][scope_name][step] = fusion_variants(scope_name, names)
    res["task4_fusion_debug"] = t4
    print("[T4] 消融完成")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"DONE → {OUT_JSON}")


if __name__ == "__main__":
    main()
