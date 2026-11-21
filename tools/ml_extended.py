# -*- coding: utf-8 -*-
"""
round9-S8-ML 扩展: 在 radiomics 特征 (logs/radiomics_full_features.csv, 356×75)
上跑更广的经典 ML 家族, 对照 LASSO-LR/HistGB (项1 已跑), 补论文方法学广度。
模型: XGBoost / LightGBM / SVM-RBF / RandomForest / ExtraTrees / NB / KNN / Voting(软投票)
口径与项1 完全一致: train(284) 内 5 折 StratifiedKFold(seed42) CV + holdout val(72),
T/N 分别建模。报告 logs/ml_extended_full_results.{json,txt}。不占 GPU。
用法:
  python tools/ml_extended.py            # 全量
  python tools/ml_extended.py --smoke    # 小模型/小折冒烟
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEAT_CSV = "./logs/radiomics_full_features.csv"
RESULT_JSON = "./logs/ml_extended_full_results.json"
RESULT_TXT = "./logs/ml_extended_full_results.txt"
CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]


def feat_cols(df):
    skip = {"patient_id", "split", "label_T", "label_N"}
    return [c for c in df.columns if c not in skip]


def make_models(seed, smoke=False):
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  VotingClassifier)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    if smoke:
        return {"xgb": XGBClassifier(n_estimators=20, max_depth=2, n_jobs=-1,
                                     random_state=seed, eval_metric="logloss"),
                "lgbm": LGBMClassifier(n_estimators=20, max_depth=2, n_jobs=-1,
                                       random_state=seed, verbose=-1),
                "svm": SVC(kernel="rbf", C=1.0, probability=True, random_state=seed)}
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        random_state=seed, eval_metric="logloss")
    lgbm = LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                          random_state=seed, verbose=-1)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, n_jobs=-1, random_state=seed)
    et = ExtraTreesClassifier(n_estimators=300, max_depth=5, n_jobs=-1, random_state=seed)
    nb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=7)
    voting = VotingClassifier([("xgb", xgb), ("lgbm", lgbm), ("svm", svm),
                               ("rf", rf), ("et", et)], voting="soft", n_jobs=-1)
    return {"xgb": xgb, "lgbm": lgbm, "svm": svm, "rf": rf, "et": et,
            "nb": nb, "knn": knn, "voting": voting}


def run_model(df, label_col, feats, model, seed=42, n_splits=5):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    Xtr, ytr = train[feats].values, train[label_col].values
    Xva, yva = val[feats].values, val[label_col].values

    cv_f1, cv_auc = [], []
    k = max(2, min(n_splits, int(np.bincount(ytr).min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr_i, va_i in skf.split(Xtr, ytr):
        sc = StandardScaler().fit(Xtr[tr_i])
        m = model.__class__(**model.get_params()) if model.__class__.__name__ != "VotingClassifier" \
            else _clone_voting(model)
        m.fit(sc.transform(Xtr[tr_i]), ytr[tr_i])
        p = m.predict_proba(sc.transform(Xtr[va_i]))[:, 1]
        yv = ytr[va_i]
        cv_f1.append(f1_score(yv, (p >= 0.5).astype(int), average="macro"))
        if len(np.unique(yv)) > 1:
            cv_auc.append(roc_auc_score(yv, p))
    sc = StandardScaler().fit(Xtr)
    m = model.__class__(**model.get_params()) if model.__class__.__name__ != "VotingClassifier" \
        else _clone_voting(model)
    m.fit(sc.transform(Xtr), ytr)
    pv = m.predict_proba(sc.transform(Xva))[:, 1]
    return {
        "cv_macro_f1": float(np.mean(cv_f1)),
        "cv_f1_std": float(np.std(cv_f1)),
        "cv_auc": float(np.mean(cv_auc)) if cv_auc else float("nan"),
        "val_macro_f1": float(f1_score(yva, (pv >= 0.5).astype(int), average="macro")),
        "val_acc": float(accuracy_score(yva, (pv >= 0.5).astype(int))),
        "val_auc": float(roc_auc_score(yva, pv)) if len(np.unique(yva)) > 1 else float("nan"),
    }


def _clone_voting(v):
    from sklearn.ensemble import VotingClassifier
    return VotingClassifier(v.estimators, voting=v.voting, n_jobs=v.n_jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(FEAT_CSV):
        print(f"缺失特征文件: {FEAT_CSV} (先跑 tools/radiomics_experiment.py)")
        return
    df = pd.read_csv(FEAT_CSV)
    feats = feat_cols(df)
    print(f"[ML-x] 复用 {FEAT_CSV}: {len(df)} 样本 x {len(feats)} 特征")

    models = make_models(seed=42, smoke=args.smoke)
    results = {}
    for mname, model in models.items():
        for label in ("label_T", "label_N"):
            t0 = time.time()
            r = run_model(df, label, feats, model, seed=42)
            results[f"{mname}_{label[-1]}"] = r
            print(f"[ML-x] {mname} {label}: cvF1={r['cv_macro_f1']:.4f}±{r['cv_f1_std']:.4f} "
                  f"valF1={r['val_macro_f1']:.4f} valAUC={r['val_auc']:.4f} ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    lines = [f"ML 扩展结果 ({time.strftime('%Y-%m-%d %H:%M:%S')}, {len(df)}样本/{len(feats)}特征)"]
    lines.append(f"{'model':10s} {'task':4s} {'CV-F1':>12s} {'val-F1':>8s} {'val-acc':>8s} {'val-AUC':>8s}")
    for mname in models:
        for t in ("T", "N"):
            r = results[f"{mname}_{t}"]
            lines.append(f"{mname:10s} {t:4s} {r['cv_macro_f1']:10.4f}±{r['cv_f1_std']:.4f} "
                         f"{r['val_macro_f1']:8.4f} {r['val_acc']:8.4f} {r['val_auc']:8.4f}")
    txt = "\n".join(lines)
    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"[ML-x] 报告: {RESULT_JSON} / {RESULT_TXT}")


if __name__ == "__main__":
    main()
