# -*- coding: utf-8 -*-
"""audit_h11_vs_base.py — 只读审计 (用户 m08155 三项要求):
1) H1.1 vs Baseline_0 在 td/tcd 完整指标: Pooled OOF AUC / 50 折胜率 / ECE(校准前=z→sigmoid, 校准后=bonly) / Brier
2) 核实 H1.2 归属与物理标量接入 (代码级结论, 输出说明文本)
3) direct_td 有/缺 d_self 两组: 真实阳性率 / H1.1 平均概率 mean±std / 分组 AUC / 影像 z 方差收缩效应

复用 stage1_hetero_views 的函数, 只读不修改任何生产文件/特征。
输出: logs/auto_staging_opt/audit_h11_vs_base.json
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "auto_staging_opt"))

_orig_chdir = os.chdir
def _safe_chdir(p):
    if os.path.isdir(p):
        return _orig_chdir(p)
os.chdir = _safe_chdir
try:
    import exp_v2_validation_suite as S   # noqa
    import exp_v2_final_calibrated as CALC  # noqa
finally:
    os.chdir = _orig_chdir

import train_slice_mil as tsm  # noqa
import exp_v2_round2_ablation as abl  # noqa
import recompute_direct_tcd_singles as rc  # noqa
import stage1_hetero_views as st1  # noqa
OUTDIR = "logs/auto_staging_opt"
SEEDS = list(range(10))
sigmoid = S.sigmoid


def run_pair(seed):
    """返回 baseline/h11 的 (fuse_z_df, cal_df)。d 分支两方案共用。"""
    fo = st1.fold_of_all(seed)
    # 公共: clin + d_self(meanmax, T120)
    clin = st1.clin_oof(st1.all_direct, fo)
    d_have, d_X, d_y = st1.pool_branch(st1.T120, "d", "meanmax", st1.d_self)
    df_ = np.array([fo[st1.pos_of[p]] for p in d_have])
    d_df = st1.probe_oof(d_have, d_X, d_y, df_)

    # baseline: t AWM T120
    t_have, t_X, t_y = st1.pool_branch(st1.T120, "t", "area_w", st1.all_direct)
    tf_ = np.array([fo[st1.pos_of[p]] for p in t_have])
    t_df = st1.probe_oof(t_have, t_X, t_y, tf_)
    fuse_b = S.F.fuse_zlogit_strategy({"clin": clin, "t": t_df, "d": d_df}, strategy="B")
    cal_b, _ = S.calibrate_cv(fuse_b, mode="bonly")

    # h11: t AWM T60
    t2h, t2X, t2y = st1.pool_branch(st1.T60, "t", "area_w", st1.all_direct)
    t2f = np.array([fo[st1.pos_of[p]] for p in t2h])
    t2_df = st1.probe_oof(t2h, t2X, t2y, t2f)
    fuse_h = S.F.fuse_zlogit_strategy({"clin": clin, "t": t2_df, "d": d_df}, strategy="B")
    cal_h, _ = S.calibrate_cv(fuse_h, mode="bonly")
    return fuse_b, cal_b, fuse_h, cal_h


def ece10(df):
    return CALC.ece10(df)


def scope_metrics(cal_df, fuse_df, scope_pids):
    g = cal_df[cal_df.patient_id.isin(scope_pids)].dropna(subset=["prob"]).copy()
    gz = fuse_df[fuse_df.patient_id.isin(scope_pids)].dropna(subset=["prob"]).copy()
    gz["prob_sig"] = sigmoid(gz["prob"].values.astype(float))
    y = g["label_T"].values
    p_cal = np.clip(g["prob"].values, 0, 1)
    p_pre = np.clip(gz["prob_sig"].values, 0, 1)
    auc = roc_auc_score(y, p_cal)
    # 50 折逐折 AUC (h11 vs base 按 (seed, fold) 对位)
    return {
        "pooled_auc_cal": round(float(auc), 4),
        "ece_pre": round(float(ece10(gz.drop(columns=["prob"]).rename(columns={"prob_sig": "prob"}))), 4),
        "ece_post": round(float(ece10(g)), 4),
        "brier_pre": round(float(brier_score_loss(y, p_pre)), 4),
        "brier_post": round(float(brier_score_loss(y, p_cal)), 4),
        "n": int(len(g)),
        "n_pos": int(y.sum()),
    }


def fold_aucs(df, scope_pids):
    g = df[df.patient_id.isin(scope_pids)].dropna(subset=["prob"]).copy()
    out = {}
    for (sd, k), gg in g.groupby(["seed", "fold"]):
        if len(np.unique(gg["label_T"])) < 2:
            out[(sd, k)] = None
        else:
            out[(sd, k)] = roc_auc_score(gg["label_T"], gg["prob"])
    return out


def main():
    results = {}
    # 收集 10 seeds 全部 df
    cal_b_all, cal_h_all, fuse_b_all, fuse_h_all = [], [], [], []
    fold_b, fold_h = {}, {}
    for sd in SEEDS:
        fuse_b, cal_b, fuse_h, cal_h = run_pair(sd)
        for d in (cal_b, cal_h, fuse_b, fuse_h):
            d["seed"] = sd
        cal_b_all.append(cal_b)
        cal_h_all.append(cal_h)
        fuse_b_all.append(fuse_b)
        fuse_h_all.append(fuse_h)
        fold_b.update({(sd, k): v for k, v in fold_aucs(cal_b, st1.td).items()})
        fold_h.update({(sd, k): v for k, v in fold_aucs(cal_h, st1.td).items()})
    cal_b = pd.concat(cal_b_all, ignore_index=True)
    cal_h = pd.concat(cal_h_all, ignore_index=True)
    fuse_b = pd.concat(fuse_b_all, ignore_index=True)
    fuse_h = pd.concat(fuse_h_all, ignore_index=True)

    # 1) H1.1 vs Baseline_0 完整指标
    comp = {}
    for name, cal, fuse in (("baseline_0", cal_b, fuse_b), ("h11", cal_h, fuse_h)):
        comp[name] = {
            "td": scope_metrics(cal, fuse, st1.td),
            "tcd": scope_metrics(cal, fuse, st1.tcd),
        }
    # 50 折胜率 (td): 逐 (seed,fold) h11 AUC > base AUC
    wins = 0
    ties = 0
    valid = 0
    for key in fold_b:
        a = fold_b[key]
        b = fold_h.get(key)
        if a is None or b is None:
            continue
        valid += 1
        if b > a:
            wins += 1
        elif b == a:
            ties += 1
    comp["td_fold_winrate_h11_vs_base"] = {
        "valid_folds": valid,
        "h11_wins": wins,
        "ties": ties,
        "h11_win_rate": round(wins / valid, 4) if valid else None,
    }
    results["section1_h11_vs_base"] = comp

    # 2) H1.2 归属核实 (代码级, 静态结论)
    results["section2_h12_attribution"] = {
        "h12_base_code": st1.make_oof_h12_base.__doc__ or "",
        "h12_base_uses_t120_for_t": True,   # pool_branch(T120,'t','area_w',...)
        "h12_base_uses_t60_for_scalars": True,  # phys_scalars(T60,...)
        "h12_base_fuse_channels": ["clin", "t", "d", "sc"],
        "scalar_feature_cols": ["max_area_mm2", "total_area_mm2", "z_span_slices", "area_concentration"],
        "scalar_preprocessing": "log1p(前3列) + z_span 不 log + fillna(median) + 折内 OOF L2-LR probe → sc 概率通道",
        "fuse_strategy_B_with_sc": "sc 作为独立影像类通道参与 mean(z_t,z_d,z_sc) 后 0.5*z_clin+0.5*mean(z_imgs)",
        "conclusion": "H1.2 AUC=0.6767(td) 是加在 Baseline_0(t=120mm) 上(而非 H1.1); 物理标量经独立 probe→zlogit 通道接入, 与 t/d 等权平均进影像侧, 不直接拼 z_clin",
    }

    # 3) d_self 分组统计 (H1.1 平均概率口径, 10 seeds 均值)
    dself = set(st1.d_self)
    td_all = list(st1.td)
    has_d = [p for p in td_all if p in dself]
    no_d = [p for p in td_all if p not in dself]
    grp = {}
    for gname, pids in (("has_d_self", has_d), ("no_d_self", no_d)):
        g = cal_h[cal_h.patient_id.isin(pids)]
        mean_prob = g.groupby("patient_id")["prob"].mean().reindex(pids)
        y = g.groupby("patient_id")["label_T"].first().reindex(pids).values.astype(int)
        prob_vals = mean_prob.values
        auc = None
        if len(np.unique(y)) >= 2:
            auc = float(roc_auc_score(y, np.clip(prob_vals, 0, 1)))
        # 影像 z 方差分解: 缺 d 组 z_im = z_t; 有 d 组 z_im = mean(z_t, z_d)
        zt = fuse_h[fuse_h.patient_id.isin(pids)].groupby("patient_id")["prob"].first().reindex(pids)
        z_im = zt.copy()
        if gname == "has_d_self":
            zd = fuse_h[fuse_h.patient_id.isin(pids)].groupby("patient_id")["prob"].first()
            # 需要 t/d 分列: 从 fuse 无法分离, 重算 z 分量
            z_im = None  # 占位, 下方单独算
        grp[gname] = {
            "n": int(len(pids)),
            "pos_rate": round(float(y.mean()), 4),
            "prob_mean": round(float(np.nanmean(prob_vals)), 4),
            "prob_std": round(float(np.nanstd(prob_vals)), 4),
            "auc_pooled": auc,
        }
    # 影像 z 方差收缩: 对每 seed 计算组内 z_t / z_d / z_im std, 取均值
    zvar = {"has_d": {"z_t": [], "z_d": [], "z_im": []}, "no_d": {"z_t": [], "z_im": []}}
    for sd in SEEDS:
        fuse_b, cal_b, fuse_h, cal_h = None, None, None, None
        # 复用 run_pair 输出 (重新跑一次代价高; 此处重建 t/d 分量)
    # ---- 单独: 逐 seed 计算 z 分量与 z_im std (z 口径同 fuse_zlogit_strategy: 折内 logit 标准化) ----
    def zstd(df):
        d = df.copy()
        p = np.clip(d["prob"].values.astype(float), 1e-6, 1 - 1e-6)
        L = np.log(p / (1 - p))
        out = np.full(len(d), np.nan)
        folds = d["fold"].values.astype(int)
        for k in np.unique(folds):
            va = folds == k
            tr = ~va
            if tr.sum() < 2:
                out[va] = L[va]
                continue
            mu, sd = L[tr].mean(), L[tr].std() + 1e-9
            out[va] = (L[va] - mu) / sd
        d["z"] = out
        return d.set_index("patient_id")["z"]

    zcomp = {"has_d_self": [], "no_d_self": []}
    for sd in SEEDS:
        fo = st1.fold_of_all(sd)
        d_have, d_X, d_y = st1.pool_branch(st1.T120, "d", "meanmax", st1.d_self)
        df_ = np.array([fo[st1.pos_of[p]] for p in d_have])
        d_df = st1.probe_oof(d_have, d_X, d_y, df_)
        t2h, t2X, t2y = st1.pool_branch(st1.T60, "t", "area_w", st1.all_direct)
        t2f = np.array([fo[st1.pos_of[p]] for p in t2h])
        t2_df = st1.probe_oof(t2h, t2X, t2y, t2f)
        zt = zstd(t2_df)
        zd = zstd(d_df)
        for gname, pids in (("has_d_self", has_d), ("no_d_self", no_d)):
            if gname == "has_d_self":
                zi = 0.5 * zt.reindex(pids) + 0.5 * zd.reindex(pids)
            else:
                zi = zt.reindex(pids)
            zcomp[gname].append({
                "z_t_std": round(float(zt.reindex(pids).std()), 4),
                "z_im_std": round(float(zi.std()), 4),
                "z_im_mean": round(float(zi.mean()), 4),
                "n": int(zi.notna().sum()),
            })
    for gname in zcomp:
        arr = zcomp[gname]
        zcomp[gname] = {
            "mean_z_t_std": round(float(np.mean([a["z_t_std"] for a in arr])), 4),
            "mean_z_im_std": round(float(np.mean([a["z_im_std"] for a in arr])), 4),
            "mean_z_im_mean": round(float(np.mean([a["z_im_mean"] for a in arr])), 4),
        }
    results["section3_dself_groups"] = {"has_d_self_n": len(has_d), "no_d_self_n": len(no_d),
                                        "groups": grp, "z_variance": zcomp}

    results["meta"] = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
                       "note": "cal=策略B融合+bonly校准; pre=z→sigmoid未校准; AUC 为 Pooled OOF; 折级胜率仅 td 口径",
                       "h12_note": "AUC=0.6767(td) 是 h12_base=Baseline_0(t=120mm)+物理标量(T60 npz 读取), 非 H1.1 上加标量"}
    with open(os.path.join(OUTDIR, "audit_h11_vs_base.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", os.path.join(OUTDIR, "audit_h11_vs_base.json"))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
