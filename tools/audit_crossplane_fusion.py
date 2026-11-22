# -*- coding: utf-8 -*-
"""任务3: 直接手术队列"跨平面多序列 + 缺失容错" T 分期晚融合摸底。
- 评估全集: 全队列中所有无 tgr (直接手术) 且至少有任意 1 个可用模态的患者
- 输入分支 (单路 OOF 概率均在折内训练):
  1. mri_t (矢状位): 2D PyRadiomics (route3a body) + ConvNeXt MeanPool (logs/route2_feats)
  2. mri_d (真轴位): 2D PyRadiomics (route3b_d body) + ConvNeXt MeanPool (logs/route2_cd_feats/d.npy)
  3. mri_c (真轴位): 2D PyRadiomics (route3b_c body) + ConvNeXt MeanPool (logs/route2_cd_feats/c.npy)
  4. clin 7 维 L2 LR
- 融合: 方案A 动态可用模态均值/折内加权; 方案B 缺失指示符 Stacking
- 对比: 全部 direct (含缺失) vs t∩d 双平面共有子集
产物: logs/audit_crossplane_fusion.json
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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
OUT_JSON = "logs/audit_crossplane_fusion.json"


def load_tgr_map():
    import pandas as pd
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
    df = df[df["mask_type"] == mtype]
    df = df[df["ok"] == 1]
    feat = [c for c in df.columns if c not in
            ("patient_id", "mask_type", "ok", "label_T", "label_N", "fold")]
    return df.set_index("patient_id")[feat]


def conv_feats(d):
    rows = []
    for fn in os.listdir(d):
        if not fn.endswith(".npy"):
            continue
        pid = int(fn.replace(".npy", ""))
        rows.append((pid, np.load(os.path.join(d, fn)).mean(axis=0)))
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
    return df


def oof_prob_lr(X, y, folds, c_grid=(0.01, 0.1, 1.0, 10.0)):
    """折内选 C + StandardScaler, 返回 pooled OOF 概率 (含 nan=不可用折)"""
    X = np.nan_to_num(X.astype(float), nan=0.0)
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    for k in np.unique(folds):
        va, tr = folds == k, folds != k
        if tr.sum() < 2 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr_s, Xva_s = sc.transform(X[tr]), sc.transform(X[va])
        best_c, best_a = c_grid[0], -1
        for c in c_grid:
            m = LogisticRegression(C=c, max_iter=2000).fit(Xtr_s, y[tr])
            a = roc_auc_score(y[tr], m.predict_proba(Xtr_s)[:, 1])
            if a > best_a:
                best_a, best_c = a, c
        m = LogisticRegression(C=best_c, max_iter=2000).fit(Xtr_s, y[tr])
        oof[va] = m.predict_proba(Xva_s)[:, 1]
    return oof


def pooled_auc(y, p):
    if np.isnan(p).any() or len(np.unique(y)) < 2:
        return None
    try:
        return round(float(roc_auc_score(y, p)), 4)
    except ValueError:
        return None


def main():
    master = pd.read_csv(MASTER_CSV)
    raw = pd.read_csv(PAIRS_RAW).set_index("patient_id")
    tgr = load_tgr_map()
    direct = master[~master["patient_id"].isin(tgr)].copy()

    # 各分支特征矩阵 (index=patient_id)
    rad_t = rad_feats(R3A_CSV, "body")          # route3a: mri_t 矢状位 107 维
    rad_d = rad_feats(R3B["d"], "body")          # route3b_d: mri_d 真轴位 107 维
    rad_c = rad_feats(R3B["c"], "body")          # route3b_c: mri_c 真轴位 107 维
    conv_t = conv_feats(R2T_DIR)
    # conv_d/c: route2_cd_feats 整表 [n,768] + _pids.csv
    conv_d_all = np.load(os.path.join(R2CD_DIR, "d.npy"))
    conv_d_pids = pd.read_csv(os.path.join(R2CD_DIR, "d_pids.csv"))["patient_id"].values
    conv_d = pd.DataFrame(conv_d_all, index=conv_d_pids)
    conv_c_all = np.load(os.path.join(R2CD_DIR, "c.npy"))
    conv_c_pids = pd.read_csv(os.path.join(R2CD_DIR, "c_pids.csv"))["patient_id"].values
    conv_c = pd.DataFrame(conv_c_all, index=conv_c_pids)
    clin = raw[CLIN_COLS].replace([np.inf, -np.inf], np.nan).astype(float)

    # 分支定义: name -> (feat_df, kind)
    branches = {
        "t_rad": rad_t, "t_conv": conv_t,
        "d_rad": rad_d, "d_conv": conv_d,
        "c_rad": rad_c, "c_conv": conv_c,
        "clin": clin,
    }
    # 各分支可用性 (mask 交集已有, 直接按 index 在 direct 中)
    direct_ids = np.sort(direct["patient_id"].values.astype(np.int64))

    res = {}
    for scope_name, scope_ids in (("direct_all", direct_ids), ("direct_td", None)):
        if scope_name == "direct_td":
            td = set(conv_t.index) & set(conv_d.index)
            scope_ids = np.array(sorted(td))
        if len(scope_ids) < 20:
            continue
        yT = master.set_index("patient_id").loc[scope_ids, "label_T"].values.astype(int)
        folds = master.set_index("patient_id").loc[scope_ids, "fold"].values.astype(int)
        n = len(scope_ids)
        # 单路 OOF (分支只对 available 患者算)
        oof = {}
        avail = {}
        for name, fdf in branches.items():
            ids = np.array(sorted(set(scope_ids) & set(fdf.index)))
            if len(ids) < 20:
                continue
            X = fdf.loc[ids].values.astype(float)
            yloc = master.set_index("patient_id").loc[ids, "label_T"].values.astype(int)
            floc = master.set_index("patient_id").loc[ids, "fold"].values.astype(int)
            p = oof_prob_lr(X, yloc, floc)
            oof[name] = (ids, p)
            avail[name] = len(ids)
            print(f"[{scope_name}] {name}: n={len(ids)} pooled AUC {pooled_auc(yloc, p)}")
        # 方案A: 动态可用模态等权平均 (仅 t/d 平面两分支为"跨平面"主输入, 含 clin)
        # 用 t_rad+t_conv+d_rad+d_conv (图像分支) + clin; 每个样本取实际存在的
        img_names = [nm for nm in ("t_rad", "t_conv", "d_rad", "d_conv") if nm in oof]
        pa = np.full(n, np.nan)
        cnt = np.zeros(n)
        for nm in img_names:
            ids, p = oof[nm]
            idx = np.searchsorted(scope_ids, ids)
            for j, i in enumerate(idx):
                if not np.isnan(p[j]):
                    pa[i] = p[j] if np.isnan(pa[i]) else pa[i] + p[j]
                    cnt[i] += 1
        mask = cnt > 0
        pa[mask] /= cnt[mask]
        # 补 clin 平均 (若样本有 clin)
        if "clin" in oof:
            ids, p = oof["clin"]
            for j, i in enumerate(np.searchsorted(scope_ids, ids)):
                if not np.isnan(p[j]):
                    pa[i] = p[j] if np.isnan(pa[i]) else pa[i] + p[j]
                    cnt[i] += 1
            m2 = cnt > 0
            pa[m2] = pa[m2] / cnt[m2]
        res.setdefault(scope_name, {})["fusionA_avg_usable"] = pooled_auc(yT, pa)
        print(f"[{scope_name}] fusionA(可用均值) pooled AUC {pooled_auc(yT, pa)}")
        # 方案B: 缺失指示符 Stacking (t/d 两平面图像+clin, 缺失填折内先验+has 指示符)
        B_names = [nm for nm in ("t_rad", "t_conv", "d_rad", "d_conv", "clin") if nm in oof]
        Sb = np.full((n, len(B_names)), np.nan)
        Hb = np.zeros((n, len(B_names)))
        for j, nm in enumerate(B_names):
            ids, p = oof[nm]
            idx = np.searchsorted(scope_ids, ids)
            for ii, i in enumerate(idx):
                if not np.isnan(p[ii]):
                    Sb[i, j] = p[ii]
                    Hb[i, j] = 1
        # 缺失填折内先验: 每折训练集阳性率
        sb_filled = Sb.copy()
        for j in range(Sb.shape[1]):
            for k in np.unique(folds):
                va = folds == k
                miss = np.isnan(Sb[:, j]) & va
                if miss.sum() == 0:
                    continue
                prior = np.clip(yT[(folds != k) & ~np.isnan(Sb[:, j])].mean(), 0.05, 0.95) \
                    if np.isnan(Sb[:, j]).sum() < len(Sb) else yT[(folds != k)].mean()
                sb_filled[miss, j] = prior
        sb_filled = np.nan_to_num(sb_filled, nan=np.clip(yT.mean(), 0.05, 0.95))
        Xb = np.hstack([sb_filled, Hb])
        oof_b = np.full(n, np.nan)
        for k in np.unique(folds):
            va, tr = folds == k, folds != k
            if tr.sum() < 2 or va.sum() < 1 or len(np.unique(yT[tr])) < 2:
                oof_b[va] = 0.5
                continue
            m = LogisticRegression(C=0.1, max_iter=2000).fit(Xb[tr], yT[tr])
            oof_b[va] = m.predict_proba(Xb[va])[:, 1]
        res.setdefault(scope_name, {})["fusionB_stack_missing"] = pooled_auc(yT, oof_b)
        print(f"[{scope_name}] fusionB(缺失指示符stack) pooled AUC {pooled_auc(yT, oof_b)}")
        res.setdefault(scope_name, {})["avail"] = avail
        res.setdefault(scope_name, {})["n"] = int(n)
        res.setdefault(scope_name, {})["T+"] = int(yT.sum())

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"DONE → {OUT_JSON}")


if __name__ == "__main__":
    main()
