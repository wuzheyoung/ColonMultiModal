# -*- coding: utf-8 -*-
"""
任务3b: T 分期三路晚融合 (PyRadiomics + ConvNeXt + 临床7维)
- 输入特征:
  ① route1 PyRadiomics body 特征 (logs/route1_radiomics_features.csv, 107 列, body mask)
  ② route2-MRI ConvNeXt mean-pool 特征 (logs/route2_feats/{pid}.npy, 768 维)
  ③ 临床 7 维 (hos_pairs_raw.csv: age/sex/ca199/cea/albumin/prealbumin/hb 原始值)
- 融合: 等权平均 + 折内 LR Stacking
- 评估: master_5fold_356.csv 5 折框架, 5 折 Pooled OOF ROC-AUC (T 分期), 报均值±std
产物: logs/audit_tgr_task3b.json + stdout
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "data", "master_5fold_356.csv")
PAIRS_RAW = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
R1_CSV = os.path.join(ROOT, "logs", "route1_radiomics_features.csv")
R2_DIR = os.path.join(ROOT, "logs", "route2_feats")
OUT_JSON = os.path.join(ROOT, "logs", "audit_tgr_task3b.json")
CLIN_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]


def load_r1_feats():
    df = pd.read_csv(R1_CSV)
    df = df[df["mask_type"] == "body"]
    feat_cols = [c for c in df.columns if c not in ("patient_id", "mask_type", "ok", "label_T", "label_N", "fold")]
    df = df[df["ok"] == 1]
    return df.set_index("patient_id")[feat_cols], feat_cols


def load_r2_feats():
    rows = []
    for fn in os.listdir(R2_DIR):
        if not fn.endswith(".npy"):
            continue
        pid = int(fn.replace(".npy", ""))
        rows.append((pid, np.load(os.path.join(R2_DIR, fn)).mean(axis=0)))
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
    return df


def oof_lr(X, y, folds, C_grid=(0.01, 0.1, 1.0, 10.0)):
    """折内选 C + 标准化, 返回 pooled OOF 概率与逐折 AUC"""
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    per_fold = {}
    uniq = np.unique(folds)
    for k in uniq:
        va = folds == k
        tr = ~va
        if np.sum(tr) < 2 or np.sum(va) < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            per_fold[k] = {"n": int(np.sum(va)), "auc": None, "note": "skip"}
            continue
        best_c, best_auc = C_grid[0], -1.0
        for c in C_grid:
            m = LogisticRegression(C=c, max_iter=2000)
            m.fit(X[tr], y[tr])
            a = roc_auc_score(y[tr], m.predict_proba(X[tr])[:, 1])
            if a > best_auc:
                best_auc, best_c = a, c
        m = LogisticRegression(C=best_c, max_iter=2000)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        try:
            per_fold[k] = {"n": int(np.sum(va)), "auc": round(float(roc_auc_score(y[va], oof[va])), 4), "c": best_c}
        except ValueError:
            per_fold[k] = {"n": int(np.sum(va)), "auc": None, "c": best_c}
    return oof, per_fold


def main():
    master = pd.read_csv(MASTER)
    raw = pd.read_csv(PAIRS_RAW)
    r1, _ = load_r1_feats()
    r2 = load_r2_feats()
    raw_clin = raw.set_index("patient_id")[CLIN_COLS].replace([np.inf, -np.inf], np.nan)
    for c in CLIN_COLS:
        raw_clin[c] = pd.to_numeric(raw_clin[c], errors="coerce")

    pids = sorted(set(r1.index) & set(r2.index) & set(raw_clin.index) & set(master["patient_id"]))
    X1 = r1.loc[pids].values.astype(float)
    X2 = r2.loc[pids].values.astype(float)
    X3 = raw_clin.loc[pids].values.astype(float)
    X3 = np.nan_to_num(X3, nan=0.0)
    yT = master.set_index("patient_id").loc[pids, "label_T"].values.astype(int)
    folds = master.set_index("patient_id").loc[pids, "fold"].values.astype(int)
    n = len(pids)
    print(f"样本: n={n} (T+ {int(yT.sum())})")

    # 逐路 OOF
    roads = {"radiomics": X1, "convnext": X2, "clin": X3}
    oofs, res = {}, {}
    for name, X in roads.items():
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        o, pf = oof_lr(Xs, yT, folds)
        oofs[name] = o
        try:
            res[name] = {"pooled_auc": round(float(roc_auc_score(yT, o)), 4), "per_fold": pf}
        except ValueError:
            res[name] = {"pooled_auc": None, "per_fold": pf}
        print(f"  [{name}] pooled OOF T AUC {res[name]['pooled_auc']}")

    # 等权平均融合
    oof_avg = np.mean([oofs["radiomics"], oofs["convnext"], oofs["clin"]], axis=0)
    res["avg_equal"] = {"pooled_auc": round(float(roc_auc_score(yT, oof_avg)), 4)}
    print(f"  [avg_equal] pooled OOF T AUC {res['avg_equal']['pooled_auc']}")

    # 折内 LR Stacking (用三路 OOF 概率做特征)
    S = np.stack([oofs["radiomics"], oofs["convnext"], oofs["clin"]], axis=1)
    oof_stk = np.full(n, np.nan)
    per_fold_stk = {}
    for k in np.unique(folds):
        va = folds == k
        tr = ~va
        if np.sum(tr) < 2 or np.sum(va) < 1 or len(np.unique(yT[tr])) < 2:
            oof_stk[va] = 0.5
            continue
        m = LogisticRegression(C=1.0, max_iter=2000)
        m.fit(S[tr], yT[tr])
        oof_stk[va] = m.predict_proba(S[va])[:, 1]
        try:
            per_fold_stk[int(k)] = round(float(roc_auc_score(yT[va], oof_stk[va])), 4)
        except ValueError:
            per_fold_stk[int(k)] = None
    res["stacking"] = {"pooled_auc": round(float(roc_auc_score(yT, oof_stk)), 4), "per_fold": per_fold_stk}
    print(f"  [stacking] pooled OOF T AUC {res['stacking']['pooled_auc']}")

    res["sample"] = {"n": n, "T+": int(yT.sum())}
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(_clean(res), f, ensure_ascii=False, indent=2)
    print(f"DONE → {OUT_JSON}")


if __name__ == "__main__":
    main()
