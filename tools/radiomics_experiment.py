# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""
round9-项1: 全体积影像组学 + 临床特征融合实验（对照 DL 基线 R5, 直击 N 短板）。

流程:
  1. 特征提取: 复用 data_loader 确定性管线(含磁盘缓存 cache_hos) 取 CT + MRI(t/c/d)
     的 96³ 卷, 每模态计算 一阶统计(11) + GLCM-3D(6) = 17 特征 × 4 模态 = 68 维,
     + 临床 7 维 = 75 维。T/N 标签 + seed42 划分(与 DL 完全一致)。
  2. ML 建模: train(284) 内 5 折 CV (StratifiedKFold seed42) + holdout val(72):
     M1 影像组学 only / M2 影像组学+临床 / M3 临床 only, 模型 = LASSO-LR(saga l1 CV)
     + HistGB, T/N 两个任务分别建模。
  3. DL 概率 Stacking: 用 R5 checkpoint (best_model_hos_full_s0.pth / _n 版) 对
     train/val 做 T/N 前向概率, 与 [M2 特征] 拼接 → LR stacking (T/N 分别)。
     注意: 该 ckpt 在 train 上训练过, train 端 DL 概率是 in-sample, CV 指标偏乐观。
  4. 报告: logs/radiomics_full_results.json/.txt (5 折 CV mean±std + holdout)。

用法:
  python tools/radiomics_experiment.py                 # 全流程(提取+ML+DL+stack)
  python tools/radiomics_experiment.py --skip-extract  # 复用 logs/radiomics_full_features.csv
  python tools/radiomics_experiment.py --smoke         # 4 样本快速冒烟
说明: 特征提取为确定性管线(无增强), 与 DL val 分支一致; 不占 GPU 长时训练。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import build_dataloaders
from data_loader.loader import _mri_seqs_deterministic, _SEQ_ORDER  # noqa: E402

CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
FEAT_CSV = "./logs/radiomics_full_features.csv"
RESULT_JSON = "./logs/radiomics_full_results.json"
RESULT_TXT = "./logs/radiomics_full_results.txt"

# ── 1. 特征提取 ──────────────────────────────────────────────────────────

def _load_ct_vol(ct_path, spatial=(96, 96, 96)):
    """读取 CT 确定性卷 [1,D,H,W]。优先磁盘缓存 cache_hos (与 DL loader 同缓存)。"""
    if not (isinstance(ct_path, str) and ct_path and os.path.exists(ct_path)):
        return None
    try:
        from data_loader.loader import _DetDiskCache, _det_cache_key, _ct_deterministic
        dc = _DetDiskCache("./cache_hos", spatial)
        key = _det_cache_key(ct_path, spatial)
        cached = dc.get_vol(key)
        if cached[1]:
            return np.ascontiguousarray(cached[0].numpy()).astype(np.float32)
        img = _ct_deterministic(spatial)(ct_path)
        arr = np.ascontiguousarray(img).astype(np.float32)
        dc.put_vol(key, torch.from_numpy(arr))
        return arr
    except Exception:
        return None


def _load_mri_vol(t, c, d, spatial=(96, 96, 96)):
    """读取 MRI 三序列卷 [3,D,H,W]。优先磁盘缓存 cache_hos。"""
    try:
        from data_loader.loader import (_DetDiskCache, _det_cache_key_mri,
                                        _mri_seqs_deterministic)
        dc = _DetDiskCache("./cache_hos", spatial)
        key = _det_cache_key_mri(t, c, d, spatial)
        cached = dc.get_vol(key)
        if cached[1]:
            return np.ascontiguousarray(cached[0].numpy()).astype(np.float32)
        vol = _mri_seqs_deterministic(t, c, d, [True, True, True], spatial)
        dc.put_vol(key, torch.from_numpy(vol))
        return vol
    except Exception:
        return None


def _first_order(vol):
    """一阶统计特征: mean/std/skew/kurtosis/energy/entropy/p10/p25/p50/p75/p90。"""
    x = vol[vol != 0]
    if x.size < 2:
        return [0.0] * 11
    mu, sd = x.mean(), x.std()
    feats = [mu, sd]
    feats.append(float(np.mean(((x - mu) / (sd + 1e-9)) ** 3)))       # skew
    feats.append(float(np.mean(((x - mu) / (sd + 1e-9)) ** 4)))       # kurtosis
    feats.append(float(np.sum(x ** 2)))                               # energy
    h, _ = np.histogram(x, bins=16, range=(x.min(), x.max() + 1e-9))
    p = h / max(h.sum(), 1)
    feats.append(float(-np.sum(p[p > 0] * np.log2(p[p > 0]))))        # entropy
    for q in (10, 25, 50, 75, 90):
        feats.append(float(np.percentile(x, q)))
    return feats


def _glcm3d(vol, levels=16, directions=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    """3D GLCM: 降采样 32³ + 灰度量化 levels 级 + 3 主方向, 返回
    [对比度, 能量, 熵, 同质性, ASM, 相关性] (方向平均)。"""
    from scipy.ndimage import zoom
    if vol.ndim == 4:
        vol = vol[0]                      # 去掉通道维 [1,D,H,W] -> [D,H,W]
    z = zoom(vol, (32 / vol.shape[0], 32 / vol.shape[1], 32 / vol.shape[2]), order=0)
    vmin, vmax = z.min(), z.max()
    if vmax - vmin < 1e-9:
        return [0.0] * 6
    q = np.floor((z - vmin) / (vmax - vmin + 1e-9) * levels).astype(np.int64)
    q = np.clip(q, 0, levels - 1)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    n_pairs = 0
    for d in directions:
        sl_a = tuple(slice(None, -n) if n > 0 else slice(None, None) for n in d)
        sl_b = tuple(slice(n, None) if n > 0 else slice(None, None) for n in d)
        a, b = q[sl_a], q[sl_b]
        idx = a.ravel() * levels + b.ravel()
        hist = np.bincount(idx, minlength=levels * levels).astype(np.float64)
        glcm += hist.reshape(levels, levels)
        n_pairs += a.size
    glcm /= max(n_pairs, 1)
    px = glcm.sum(1)
    py = glcm.sum(0)
    mx = np.dot(np.arange(levels), px)
    my = np.dot(np.arange(levels), py)
    iis = np.arange(levels)
    varx = np.dot((iis - mx) ** 2, px)
    vary = np.dot((iis - my) ** 2, py)
    diff = np.abs(iis[:, None] - iis[None, :])
    contrast = float(np.sum(glcm * diff ** 2))
    energy = float(np.sum(glcm ** 2))
    eps = 1e-12
    entropy = float(-np.sum(glcm[glcm > 0] * np.log2(glcm[glcm > 0] + eps)))
    homogeneity = float(np.sum(glcm / (1.0 + diff)))
    asm = energy
    corr_num = np.sum(glcm * (iis[:, None] - mx) * (iis[None, :] - my))
    corr = float(corr_num / (np.sqrt(varx * vary) + eps))
    return [contrast, energy, entropy, homogeneity, asm, corr]


def _modal_features(vol):
    """单模态 17 特征: 一阶(11) + GLCM(6)。"""
    if vol is None:
        return [0.0] * 17
    return _first_order(vol) + _glcm3d(vol)


def extract_features(train_csv, val_csv, out_csv, smoke=False):
    """提取全部样本特征表 (与 DL seed42 划分一致)。返回 DataFrame。"""
    dfs = []
    for split, csv_path in (("train", train_csv), ("val", val_csv)):
        df = pd.read_csv(csv_path)
        if smoke:
            df = df.head(2)
        rows = []
        n_fail_ct = n_fail_mri = 0
        for _, r in df.iterrows():
            ct_path = r.get("ct_path", "")
            ct_path = ct_path if isinstance(ct_path, str) else ""
            t_path = r.get("mri_path_t", r.get("mri_path", ""))
            t_path = t_path if isinstance(t_path, str) else ""
            c_path = r.get("mri_path_c", "")
            c_path = c_path if isinstance(c_path, str) else ""
            d_path = r.get("mri_path_d", "")
            d_path = d_path if isinstance(d_path, str) else ""
            ct_vol = _load_ct_vol(ct_path)
            mri_vol = _load_mri_vol(t_path, c_path, d_path)
            if ct_vol is None and ct_path:
                n_fail_ct += 1
            if mri_vol is None and (t_path or c_path or d_path):
                n_fail_mri += 1
            feats = _modal_features(ct_vol)
            if mri_vol is None:
                mri_feats = [0.0] * 17 * 3
            else:
                mri_feats = []
                for si, seq in enumerate(_SEQ_ORDER):
                    mri_feats += _modal_features(mri_vol[si])
            row = {"patient_id": str(r.get("patient_id", "")), "split": split,
                   "label_T": int(r["label_T"]), "label_N": int(r["label_N"])}
            for i, v in enumerate(feats):
                row[f"ct_{i}"] = v
            for si, seq in enumerate(_SEQ_ORDER):
                for i, v in enumerate(mri_feats[si * 17:(si + 1) * 17]):
                    row[f"mri_{seq}_{i}"] = v
            for c in CLINICAL_COLS:
                v = r.get(c, 0.0)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                row[c] = float(v)
            rows.append(row)
        dfs.append(pd.DataFrame(rows))
        print(f"[extract] {split}: {len(rows)} 样本完成; 读取失败 CT={n_fail_ct} MRI={n_fail_mri} "
              f"(失败样本特征按 0 填充)")
    if not dfs:
        raise RuntimeError("[extract] 空特征表: 检查 train/val csv")
    feats_df = pd.concat(dfs, ignore_index=True)
    feats_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[extract] 特征表已写入 {out_csv} ({len(feats_df)} 样本)")
    return feats_df


# ── 2. ML 建模 ──────────────────────────────────────────────────────────

def _feat_columns(df):
    return [c for c in df.columns if c not in
            {"patient_id", "split", "label_T", "label_N"}]


def run_ml(df, label_col, feats, n_splits=5, seed=42):
    """train 内 5 折 CV + holdout val。返回 dict(CV mean±std / val 指标)。"""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    Xtr, ytr = train[feats].values, train[label_col].values
    Xva, yva = val[feats].values, val[label_col].values

    def evals(y, p):
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        return {"auc": float(auc)}

    results = {}
    for name, make in (("lasso_lr",
                        lambda: LogisticRegressionCV(
                            penalty="l1", solver="saga", Cs=5, cv=3,
                            scoring="roc_auc", max_iter=3000, n_jobs=-1,
                            random_state=seed)),
                       ("histgb",
                        lambda: HistGradientBoostingClassifier(
                            max_iter=300, learning_rate=0.05,
                            max_depth=3, random_state=seed))):
        cv_scores, cv_aucs = [], []
        k = max(2, min(n_splits, int(np.bincount(ytr).min())))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(Xtr, ytr):
            sc = StandardScaler().fit(Xtr[tr_i])
            m = make()
            m.fit(sc.transform(Xtr[tr_i]), ytr[tr_i])
            p = m.predict_proba(sc.transform(Xtr[va_i]))[:, 1]
            yv = ytr[va_i]
            cv_scores.append(f1_score(yv, (p >= 0.5).astype(int),
                                      average="macro"))
            if len(np.unique(yv)) > 1:
                cv_aucs.append(roc_auc_score(yv, p))
        sc = StandardScaler().fit(Xtr)
        m = make()
        m.fit(sc.transform(Xtr), ytr)
        pv = m.predict_proba(sc.transform(Xva))[:, 1]
        results[name] = {
            "cv_macro_f1": float(np.mean(cv_scores)),
            "cv_f1_std": float(np.std(cv_scores)),
            "cv_auc": float(np.mean(cv_aucs)) if cv_aucs else float("nan"),
            "val_macro_f1": float(f1_score(yva, (pv >= 0.5).astype(int),
                                           average="macro")),
            "val_acc": float(accuracy_score(yva, (pv >= 0.5).astype(int))),
            "val_auc": evals(yva, pv)["auc"],
        }
    return results


# ── 3. DL 概率 Stacking ─────────────────────────────────────────────────

def _build_model(cfg):
    from models import CRCMultiModalNet
    return CRCMultiModalNet(
        backbone_name=cfg.backbone_name, fusion_mode=cfg.fusion_mode,
        num_clinical=cfg.num_clinical, num_classes=cfg.num_classes,
        num_tasks=cfg.num_tasks, swin_feature_size=cfg.swin_feature_size,
        num_fuse_stages=cfg.num_fuse_stages, img_size=cfg.spatial_size,
        use_self_attn=cfg.use_self_attn, self_attn_layers=cfg.self_attn_layers,
        self_attn_heads=cfg.self_attn_heads, use_imputer=cfg.use_imputer,
        imputer_hidden_dim=cfg.imputer_hidden_dim,
        head_dropout=getattr(cfg, "head_dropout", 0.3),
        pretrained_backbone=cfg.pretrained_backbone,
        mri_share_backbone=getattr(cfg, "mri_share_backbone", False),
        mri_sequences=getattr(cfg, "mri_sequences", ("t", "c", "d")),
        head_width=getattr(cfg, "head_width", 256),
        pool_mode=getattr(cfg, "pool_mode", "avg"),
        gem_p=getattr(cfg, "gem_p", 3.0),
        n_from_shallow=getattr(cfg, "n_from_shallow", False),
        n_shallow_idx=getattr(cfg, "n_shallow_idx", 3),
        n_head_width=getattr(cfg, "n_head_width", None),
        n_dropout=getattr(cfg, "n_dropout", None),
        n_modal=getattr(cfg, "n_modal", "both"),
        n_cascade=getattr(cfg, "n_cascade", False),
    )


def load_dl_probs(train_csv, val_csv, ckpt_t, ckpt_n):
    """用 R5 checkpoint 对 train/val 前向得到 T/N softmax 概率。
    复用 build_dataloaders + 内联 build_model; 手写前向(与 trainer.predict_proba 相同逻辑)。"""
    from config.hos import HosConfig

    cfg = HosConfig()
    cfg.train_csv, cfg.val_csv = train_csv, val_csv
    cfg.backbone_name = "medicalnet18"
    cfg.pool_mode = "gem"
    cfg.head_width = 128
    cfg.lr = 1e-4
    cfg.lr_n = 3e-4
    cfg.n_from_shallow = True   # R5: --n-shallow
    cfg.n_shallow_idx = 3
    cfg.class_weight_mode = "inverse"
    cfg.save_dir = "./checkpoints_hos_full"
    cfg.plot_dir = "./plots_hos_full"
    cfg.use_augmentation = False  # 概率提取用确定性卷
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _build_model(cfg)
    sd_t = torch.load(ckpt_t, map_location=cfg.device)
    if isinstance(sd_t, dict) and "state_dict" in sd_t:
        sd_t = sd_t["state_dict"]
    model.load_state_dict(sd_t)
    model.to(cfg.device).eval()

    out = {}
    for split, csv in (("train", train_csv), ("val", val_csv)):
        cfg.train_csv, cfg.val_csv = csv, csv
        _, loader = build_dataloaders(cfg, csv, csv)
        pts, pns, lts, lns = [], [], [], []
        ids = []
        with torch.no_grad():
            for batch in loader:
                ct = batch["ct"].to(cfg.device)
                mri = batch["mri"].to(cfg.device)
                clin = batch["clinical"].to(cfg.device)
                lbl = batch["label"].to(cfg.device)
                lbl_n = batch.get("label_secondary")
                ct_mask = batch["ct_available"].to(cfg.device).bool()
                mri_mask = batch["mri_available"].to(cfg.device).bool()
                mri_seq_mask = batch.get("mri_seq_available")
                if mri_seq_mask is not None:
                    mri_seq_mask = mri_seq_mask.to(cfg.device).bool()
                with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
                    o = model(ct, mri, clin, ct_mask=ct_mask, mri_mask=mri_mask,
                              mri_seq_mask=mri_seq_mask)
                pts.append(torch.softmax(o["main"], dim=1).float().cpu().numpy())
                lts.append(lbl.cpu().numpy())
                ids.extend(str(x) for x in batch.get("patient_id", []))
                if lbl_n is not None and "n" in o:
                    pns.append(torch.softmax(o["n"], dim=1).float().cpu().numpy())
                    lns.append(lbl_n.cpu().numpy())
        out[split] = {"pt": np.concatenate(pts), "lt": np.concatenate(lts), "ids": ids}
        if pns:
            out[split]["pn"] = np.concatenate(pns)
            out[split]["ln"] = np.concatenate(lns)
        print(f"[dl-probs] {split}: T{len(out[split]['lt'])} N{len(out[split].get('ln', []))}")
    return out


def subset_dl_probs(dl, ids):
    """按 patient_id 把 DL 概率过滤/重排到特征表子集 (ROI 等子集 stacking 必需)。"""
    out = {}
    for split, d in dl.items():
        pos = {str(pid): i for i, pid in enumerate(d.get("ids", []))}
        keep = [pos[str(pid)] for pid in ids if str(pid) in pos]
        if not keep:
            raise ValueError(f"[stacking] {split}: DL 概率与特征表 patient_id 无交集")
        cur = {"pt": d["pt"][keep], "lt": d["lt"][keep],
               "ids": [str(ids[i]) for i, pid in enumerate(ids) if str(pid) in pos]}
        if "pn" in d:
            cur["pn"] = d["pn"][keep]
            cur["ln"] = d["ln"][keep]
        out[split] = cur
    return out


def run_stacking(df, dl, label_col, base_feats, seed=42):
    """[ML 特征 + DL 概率] → LR stacking (5 折 CV + val)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    key = "lt" if label_col == "label_T" else "ln"
    pkey = "pt" if label_col == "label_T" else "pn"

    Xtr = np.hstack([train[base_feats].values,
                     dl["train"][pkey][:, 1:2]])
    ytr = train[label_col].values
    Xva = np.hstack([val[base_feats].values,
                     dl["val"][pkey][:, 1:2]])
    yva = val[label_col].values
    assert len(ytr) == dl["train"][key].shape[0], "DL prob 行数必须与特征一致"

    cv_f1, cv_auc = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_i, va_i in skf.split(Xtr, ytr):
        sc = StandardScaler().fit(Xtr[tr_i])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr[tr_i]), ytr[tr_i])
        p = m.predict_proba(sc.transform(Xtr[va_i]))[:, 1]
        yv = ytr[va_i]
        cv_f1.append(f1_score(yv, (p >= 0.5).astype(int), average="macro"))
        if len(np.unique(yv)) > 1:
            cv_auc.append(roc_auc_score(yv, p))
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    pv = m.predict_proba(sc.transform(Xva))[:, 1]
    return {
        "cv_macro_f1": float(np.mean(cv_f1)),
        "cv_f1_std": float(np.std(cv_f1)),
        "cv_auc": float(np.mean(cv_auc)) if cv_auc else float("nan"),
        "val_macro_f1": float(f1_score(yva, (pv >= 0.5).astype(int), average="macro")),
        "val_acc": float(accuracy_score(yva, (pv >= 0.5).astype(int))),
        "val_auc": float(roc_auc_score(yva, pv)) if len(np.unique(yva)) > 1 else float("nan"),
        "note": "DL 概率来自在 train 上训练的 ckpt (in-sample), CV 指标偏乐观; 跨方案比较只用 val 列",
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    train_csv = "./data/hos_full_train_s42.csv"
    val_csv = "./data/hos_full_val_s42.csv"
    t0 = time.time()

    if args.skip_extract and os.path.exists(FEAT_CSV):
        df = pd.read_csv(FEAT_CSV)
        print(f"[extract] 复用 {FEAT_CSV} ({len(df)} 样本)")
    else:
        df = extract_features(train_csv, val_csv, FEAT_CSV, smoke=args.smoke)

    if args.smoke:
        print(f"[smoke] 特征表 OK: {df.shape}, 列: {_feat_columns(df)[:6]} ... "
              f"{_feat_columns(df)[-3:]}")

    all_feats = _feat_columns(df)
    if args.smoke:
        print("[smoke] 退出 (真实数据上跑 ML + DL stacking)")
        return
    radio = [c for c in all_feats if c.startswith("ct_") or c.startswith("mri_")]
    radio_clin = all_feats  # 影像组学 + 临床 = 全部
    clin = [c for c in all_feats if c in CLINICAL_COLS]

    configs = {"M1_radiomics": radio, "M2_radiomics_clinical": radio_clin,
               "M3_clinical": clin}
    results = {}
    for name, feats in configs.items():
        for label in ("label_T", "label_N"):
            results[f"{name}_{label[-1]}"] = run_ml(df, label, feats)

    # DL stacking (M2 特征 + DL 概率)。注: s42 ckpt 已被 B1/B2 覆盖(级联头),
    # 用 s0 的 R5 配置 checkpoint(多 seed 共用 split_seed=42 划分, 与特征表一致)。
    dl = None
    if not args.smoke and torch.cuda.is_available():
        ckpt_t = "./checkpoints_hos_full/best_model_hos_full_s0.pth"
        ckpt_n = "./checkpoints_hos_full/best_model_n_hos_full_s0.pth"
        if os.path.exists(ckpt_t):
            dl = load_dl_probs(train_csv, val_csv, ckpt_t, ckpt_n)
            for label in ("label_T", "label_N"):
                results[f"M4_stack_{label[-1]}"] = run_stacking(df, dl, label, radio_clin)
    else:
        print("[dl] 跳过 DL stacking (smoke 或非 GPU)")

    # 报告
    summary = {}
    for k, v in results.items():
        summary[k] = {kk: round(vv, 4) if isinstance(vv, float) else vv
                      for kk, vv in v.items()}
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": summary},
                  f, ensure_ascii=False, indent=2)
    lines = [f"radiomics 实验报告 ({time.strftime('%Y-%m-%d %H:%M:%S')})",
             f"样本: train {len(df[df.split=='train'])} / val {len(df[df.split=='val'])}; "
             f"特征: {len(all_feats)} (影像组学 {len(radio)} + 临床 {len(clin)})",
             ""]
    for task in ("T", "N"):
        lines.append(f"── 任务 {task} ──")
        for k in ("M1_radiomics", "M2_radiomics_clinical", "M3_clinical"):
            kk = f"{k}_{task}"
            if kk in results and isinstance(results[kk], dict):
                for sub, r in results[kk].items():
                    lines.append(f"  {k}({sub}):<24 CV-F1 {r['cv_macro_f1']:.4f}±{r['cv_f1_std']:.4f} "
                                 f"AUC {r['cv_auc']:.4f} | val F1 {r['val_macro_f1']:.4f} "
                                 f"acc {r['val_acc']:.4f} AUC {r['val_auc']:.4f}")
        for k in ("M4_stack",):
            kk = f"{k}_{task}"
            if kk in results:
                r = results[kk]
                lines.append(f"  {k:<22} CV-F1 {r['cv_macro_f1']:.4f}±{r['cv_f1_std']:.4f} "
                             f"AUC {r['cv_auc']:.4f} | val F1 {r['val_macro_f1']:.4f} "
                             f"acc {r['val_acc']:.4f} AUC {r['val_auc']:.4f}")
    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[done] 总耗时 {time.time() - t0:.1f}s → {RESULT_JSON} / {RESULT_TXT}")


if __name__ == "__main__":
    main()
