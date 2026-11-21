# -*- coding: utf-8 -*-
"""
T2a/T2b 采集/形态标量逻辑回归 (零训练): 用物理标量预测 label_N, 5 折 (master 折表)。
- T2a 采集/缩放: fov_x/fov_y/fov_z, slice_thick, bbox_lx/ly/lz, clamp_frac
  判据: N AUC >= 0.58 报警 (CNN 只学采集视野); 0.47-0.54 (随机) 通过
- T2b 物理形态: vol_mm3, z_span_mm, ring_ratio (1mm³ 真实体积/纵轴跨度/环体比)
  判据: 在 T2a 通过前提下, N AUC >= 0.58 支持真实解剖形态信号
- 输入: tools/phase2_healthcheck/outputs/t2_scalars.csv + data/master_5fold_356.csv
- 输出: 逐折 AUC + mean±std + 判定, 写 README 并落盘
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCALARS = os.path.join(ROOT, "tools", "phase2_healthcheck", "outputs", "t2_scalars.csv")
MASTER = os.path.join(ROOT, "data", "master_5fold_356.csv")
OUT_DIR = os.path.join(ROOT, "tools", "phase2_healthcheck", "outputs")

T2A_FEATS = ["fov_x", "fov_y", "fov_z", "slice_thick", "bbox_lx", "bbox_ly", "bbox_lz", "clamp_frac"]
T2B_FEATS = ["vol_mm3", "z_span_mm", "ring_ratio"]


def run_lr(df, feats, tag):
    folds = sorted(df["fold"].unique())
    aucs = []
    detail = []
    for f in folds:
        tr = df[df["fold"] != f]
        va = df[df["fold"] == f]
        Xtr = tr[feats].values
        Xva = va[feats].values
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, C=0.1)
        clf.fit(sc.transform(Xtr), tr["label_N"])
        prob = clf.predict_proba(sc.transform(Xva))[:, 1]
        if va["label_N"].nunique() == 2:
            auc = roc_auc_score(va["label_N"], prob)
        else:
            auc = np.nan
        aucs.append(auc)
        detail.append({"fold": f, "n": len(va), "n_pos": int(va["label_N"].sum()), "auc": auc})
    aucs = np.array([a for a in aucs if not np.isnan(a)])
    mean, std = aucs.mean(), aucs.std(ddof=1) if len(aucs) > 1 else 0.0
    print(f"--- {tag} ---")
    for d in detail:
        print(f"  fold {d['fold']}: n={d['n']} pos={d['n_pos']} AUC={d['auc']:.4f}")
    print(f"  => N AUC {mean:.4f} ± {std:.4f}  (over {len(aucs)} folds)")
    return mean, std, pd.DataFrame(detail)


def main():
    sc = pd.read_csv(SCALARS, encoding="utf-8-sig")
    master = pd.read_csv(MASTER, encoding="utf-8-sig")
    df = sc.merge(master[["patient_id", "label_N", "fold"]], on="patient_id", how="inner")
    df = df.dropna(subset=T2A_FEATS + T2B_FEATS + ["label_N"])
    df = df[df["label_N"].isin([0, 1])]
    print(f"标量-主表 合并后样本: {len(df)} (fold 分布: {df.groupby('fold').size().to_dict()})")
    print(f"label_N 阳性率: {df['label_N'].mean():.3f}")

    m_a, s_a, d_a = run_lr(df, T2A_FEATS, "T2a 采集/缩放")
    m_b, s_b, d_b = run_lr(df, T2B_FEATS, "T2b 物理形态")

    # 判定
    v_a = "报警: T2a N AUC >= 0.58 (CNN 可能只学采集视野)" if m_a >= 0.58 else (
        "通过: T2a N AUC 0.47-0.54 随机区间" if 0.47 <= m_a <= 0.54 else
        f"边界情况: T2a N AUC {m_a:.3f} (判据: >=0.58 报警, 0.47-0.54 通过)")
    v_b = (f"支持真实解剖形态信号: T2b N AUC {m_b:.3f} >= 0.58 (前提 T2a 通过)" if (m_b >= 0.58 and m_a < 0.58)
           else f"T2b 未达 0.58 或 T2a 已报警: {m_b:.3f}")
    print(f"\n判定 T2a: {v_a}")
    print(f"判定 T2b: {v_b}")

    d_a.to_csv(os.path.join(OUT_DIR, "t2a_acquisition_lr.csv"), index=False)
    d_b.to_csv(os.path.join(OUT_DIR, "t2b_physics_lr.csv"), index=False)
    print(f"已落盘: {OUT_DIR}\\t2a_acquisition_lr.csv / t2b_physics_lr.csv")
    return v_a, v_b, m_a, s_a, m_b, s_b


if __name__ == "__main__":
    main()
