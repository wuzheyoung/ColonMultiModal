# -*- coding: utf-8 -*-
"""train_slice_mil.py — 切片级 MIL 与跨平面融合训练框架 (m00524 任务3, m00622 打磨)。

输入契约 (与 4060 extract_slice_feats_v2.py 一致):
  features/slice_feats_v2/{backbone}/{seq}/{pid}.npz, seq ∈ {t,c,d}:
    feats (K,768) float16 | slice_idx (K,) int32 | rel_z (K,) float32
    mask_area_ratio (K,) float32 | mask_source (1,) str array

硬性约束:
  1. 读 float16 → 前向统一 float32 (MIL 特征级显存极小; 严禁 bf16);
  2. rel_z / mask_area_ratio 可选门控/位置先验 (--spatial-prior 拼接切片特征);
  3. 三种聚合对照: MeanMaxLinear / TopKPool / GatedAttentionMIL (Dropout+WD+LayerNorm);
     固定 epoch + 余弦退火 (严禁外层验证折挑 epoch/超参; 逐折曲线仅监测);
  4. 外层 5 折锁死 master 折列; 无跨折泄露断言;
  5. --target 支持 label_T / label_N; --feat-norm ∈ {none,l2,std} 适配特征尺度;
  6. --clin-oof <csv> 将临床 OOF 作为第 4 路加入融合, 一键输出 clin+3-plane 晚融合;
  7. 融合 (默认输出): probmean (可用模态等权均值) / zlogit (折内训练 logit z-score
     校准的可用模态均值, 支持缺模态, 与 audit s4.f1 同口径)。
     stack (折内 LR 堆叠) 默认关闭 (m00879 收尾): 元学习器训练特征 = 训练折患者的
     平面 OOF 概率, 而每个 OOF 由"排除该患者所在折"的平面模型产生 —— 对第 k 折而言
     这些模型见过第 k 折标签, 使 stack 训练特征间接携带第 k 折信息, 存在隐性跨折泄露;
     仅 --with-stack 显式开启 (论文对比若用 stack 需说明该偏差)。

速度 (m00622): 训练前一次性将各病例 .npz 预加载为 GPU float32 tensor (RAM 常驻),
同一 fold 内按切片数 K 分桶批量前向 (B,K,D), 消除逐袋 I/O 与逐袋 transfer。
"""
import argparse
import gc
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MASTER = "data/master_5fold_356.csv"
DEFAULT_DATA_ROOT = "features/slice_feats_v2"
D = 768          # 骨干输出维度 (convnext_tiny / dinov2 / biomedclip 均为 768)
SPATIAL_DIM = 2  # rel_z + mask_area_ratio
SEQS = ("t", "c", "d")


# ---------------- 聚合器 (全部支持 (B,K,D) 批量 + K-bin 补齐掩码) ----------------
class MeanMaxLinear(nn.Module):
    def __init__(self, h, dropout):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, x, mask):  # x: (B,K,h), mask: (B,K) float (1=真切片)
        m = mask.unsqueeze(-1)
        mean = (x * m).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        mx = x.masked_fill(m == 0, -float("inf")).max(dim=1).values
        return self.head(torch.cat([mean, mx], dim=-1))


class TopKPool(nn.Module):
    def __init__(self, h, k_frac=0.25, dropout=0.0):
        super().__init__()
        self.score = nn.Linear(h, 1)
        self.k_frac = k_frac
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, x, mask):  # x: (B,K,h), mask: (B,K)
        K = x.shape[1]
        n_true = int(mask.sum(dim=1).min().item())
        k = max(1, min(K, n_true, int(round(self.k_frac * K))))
        s = self.score(x).squeeze(-1).masked_fill(mask == 0, -float("inf"))  # (B,K)
        top = torch.topk(s, k, dim=1).indices                               # (B,k)
        sel = torch.gather(x, 1, top.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        return self.head(sel.mean(dim=1))


class GatedAttentionMIL(nn.Module):
    """ABMIL: tanh(Wx) ⊙ sigmoid(Ux) 门控注意力, 按切片 softmax → 袋向量 (补齐行掩码 -inf)。"""

    def __init__(self, h, dropout=0.4):
        super().__init__()
        self.att_a = nn.Linear(h, h)
        self.att_b = nn.Linear(h, h)
        self.att_v = nn.Linear(h, 1)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, x, mask):  # x: (B,K,h), mask: (B,K)
        a = torch.tanh(self.att_a(x)) * torch.sigmoid(self.att_b(x))
        w = torch.softmax(self.att_v(a).squeeze(-1).masked_fill(mask == 0, -float("inf")), dim=1)
        bag = (w.unsqueeze(-1) * x).sum(dim=1)
        return self.head(bag)


class SliceMIL(nn.Module):
    """切片投影 (可选空间先验拼接) → 聚合 → 线性头。全程 float32, 支持 (B,K,D)+掩码。"""

    def __init__(self, in_dim, agg="gated", h=256, k_frac=0.25, dropout=0.4, spatial_prior=False):
        super().__init__()
        self.spatial_prior = spatial_prior
        self.proj = nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.agg = {
            "meanmax": MeanMaxLinear(h, dropout),
            "topk": TopKPool(h, k_frac, dropout),
            "gated": GatedAttentionMIL(h, dropout),
        }[agg]

    def forward(self, feats, mask, rel_z=None, area=None):  # feats: (B,K,D), mask: (B,K)
        if self.spatial_prior:
            extra = torch.stack([rel_z, area], dim=-1).float()
            x = torch.cat([feats.float(), extra], dim=-1)
        else:
            x = feats.float()
        return self.agg(self.proj(x), mask).squeeze(-1)     # (B,)


# ---------------- 数据预加载 (一次进 RAM/GPU, 按 K 分桶批量) ----------------
def collect_plane_pids(master, data_root, backbone, seq, pids):
    """该平面在磁盘上实际存在的 .npz 患者集合 (∩ master pids), 返回排序 int64 数组。"""
    d = os.path.join(data_root, backbone, seq)
    on_disk = set()
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.endswith(".npz"):
                try:
                    on_disk.add(int(fn[:-4]))
                except ValueError:
                    continue
    return np.array(sorted(set(pids) & on_disk), dtype=np.int64)


def preload_plane(master, data_root, backbone, seq, have, target, feat_norm, device):
    """返回 {pid: (feats, rel_z, area, label)} — feats/rel_z/area 已驻留 GPU float32。

    feat_norm: 'l2' = 逐切片 L2 行归一 (无 fit);
               'std' = 折内 StandardScaler (在 oof_for_plane 中按训练折 fit)。
    """
    m = master.set_index("patient_id")
    d = os.path.join(data_root, backbone, seq)
    bags = {}
    for pid in have:
        p = os.path.join(d, f"{pid}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        feats = torch.from_numpy(z["feats"].astype(np.float32))       # (K,768)
        rel_z = torch.from_numpy(z["rel_z"].astype(np.float32))
        area = torch.from_numpy(z["mask_area_ratio"].astype(np.float32))
        if feat_norm == "l2":
            feats = feats / torch.clamp(feats.norm(dim=-1, keepdim=True), min=1e-6)
        label = float(m.loc[pid, target])
        bags[pid] = (feats.to(device), rel_z.to(device), area.to(device), label)
    return bags


def group_by_bin(bags, pids, bin_w=4):
    """同 fold 内按 K 分桶 (bin 宽 bin_w): {bin_id: (pids, Kmax_pad)}。

    桶内统一补齐到 (bin_id+1)*bin_w, 聚合器用掩码忽略补齐行 —— 将每 epoch 的
    batch 数从 ~33 (逐 K) 降到 ~8, 消除 Python 循环开销 (m00622 提速)。
    """
    groups = {}
    for pid in pids:
        if pid not in bags:
            continue
        K = bags[pid][0].shape[0]
        b = int(K - 1) // bin_w
        groups.setdefault(b, []).append(pid)
    return {b: (ps, (b + 1) * bin_w) for b, ps in groups.items()}


def make_batch(bags, pids, spatial_prior, Kpad):
    def pad(f, d):
        K = f.shape[0]
        if K < Kpad:
            return torch.cat([f, torch.zeros(Kpad - K, d, device=f.device)], dim=0)
        return f
    Df = bags[pids[0]][0].shape[-1]
    F = torch.stack([pad(bags[p][0], Df) for p in pids], dim=0)     # (B,Kpad,D)
    mask = torch.zeros(len(pids), Kpad, device=F.device)
    for i, p in enumerate(pids):
        mask[i, :bags[p][0].shape[0]] = 1.0
    y = torch.tensor([bags[p][3] for p in pids], device=F.device)
    if spatial_prior:
        def padv(f):
            K = f.shape[0]
            if K < Kpad:
                return torch.cat([f, torch.zeros(Kpad - K, device=f.device)], dim=0)
            return f
        R = torch.stack([padv(bags[p][1]) for p in pids], dim=0)
        A = torch.stack([padv(bags[p][2]) for p in pids], dim=0)
        return F, mask, R, A, y
    return F, mask, None, None, y


def make_std_bags(bags_raw, pids, tr_pids):
    """折内 StandardScaler: 均值/方差严格只在训练折患者的真实切片上计算。

    自检断言 (m00879 收尾): ① fit 切片数 == 训练折各患者 K 之和 —— 混入验证折
    会使切片数变多直接触发; ② fit 输入不含全零行 (padding 哨兵)。bags 内本就只存
    真实切片 (补齐零行只存在于 batch 内), 断言仍作为最小编码护栏保留。
    每折从原始特征重算, 不就地修改 bags_raw —— 旧版 apply_std_scale 就地覆盖会在
    折间累积二次标准化 (fold 1 在 fold 0 已缩放数据上再缩放), 属实际 bug, 已修。
    """
    tr_slices = torch.cat([bags_raw[p][0] for p in tr_pids], dim=0)  # (N,D)
    n_expected = sum(int(bags_raw[p][0].shape[0]) for p in tr_pids)
    assert tr_slices.shape[0] == n_expected, (
        f"std fit 切片数 {tr_slices.shape[0]} != 训练折真实切片数 {n_expected} "
        f"(混入验证折/padding?)")
    assert not bool(torch.any(torch.all(tr_slices == 0, dim=1)).item()), \
        "std fit 输入含全零行 (padding 哨兵混入)"
    mu = tr_slices.mean(dim=0, keepdim=True)
    sd = tr_slices.std(dim=0, unbiased=False, keepdim=True) + 1e-6
    out = {}
    for p in pids:
        f = bags_raw[p][0]
        out[p] = ((f - mu) / sd, bags_raw[p][1], bags_raw[p][2], bags_raw[p][3])
    return out


# ---------------- 训练 (固定 epoch + 余弦退火, 外层折仅监测) ----------------
def train_plane(bags, pid_train, pid_val, model, opt, sch, epochs,
                spatial_prior, device, seed, target, bin_w=4, max_batch_size=16):
    tr_ids = [p for p in pid_train if p in bags]
    va_ids = [p for p in pid_val if p in bags]
    tr_bins = group_by_bin(bags, tr_ids, bin_w)
    va_bins = group_by_bin(bags, va_ids, bin_w)
    crit = nn.BCEWithLogitsLoss()
    curves = []
    for ep in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(int(seed) + ep)
        ks = list(tr_bins)
        order = torch.randperm(len(ks), generator=g).tolist()
        tot = 0.0; nb = 0
        for i in order:
            b = ks[i]
            ps, kpad = tr_bins[b]
            # m00879 收尾: 大桶按 ≤max_batch_size 拆成多个梯度步, 保证每 epoch
            # 更新次数 ≥ ceil(训练袋数/max_batch_size), 避免真实数据大桶欠拟合。
            for j in range(0, len(ps), max_batch_size):
                psb = ps[j:j + max_batch_size]
                F, M, R, A, y = make_batch(bags, psb, spatial_prior, kpad)
                logit = model(F, M, R, A)
                loss = crit(logit, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(y); nb += len(y)
        sch.step()
        tr_loss = tot / max(1, nb)
        model.eval()
        with torch.no_grad():
            vl = []; va_y = []; va_p = []
            for b, (ps, kpad) in va_bins.items():
                F, M, R, A, y = make_batch(bags, ps, spatial_prior, kpad)
                logit = model(F, M, R, A)
                vl.append(crit(logit, y).item())
                va_y.extend(y.tolist())
                va_p.extend(torch.sigmoid(logit).tolist())
            va_auc = roc_auc_score(va_y, va_p) if len(set(va_y)) > 1 else None
            curves.append({"epoch": ep + 1, "train_loss": round(tr_loss, 4),
                           "val_loss": round(float(np.mean(vl)), 4) if vl else None,
                           "val_auc": round(float(va_auc), 4) if va_auc is not None else None})
    model.eval()
    out = []
    with torch.no_grad():
        for b, (ps, kpad) in va_bins.items():
            F, M, R, A, y = make_batch(bags, ps, spatial_prior, kpad)
            logit = model(F, M, R, A)
            psig = torch.sigmoid(logit).tolist()
            for p, pp, yy in zip(ps, psig, y.tolist()):
                out.append({"patient_id": p, "label_T": yy, "prob": float(pp)})
    return out, curves


def oof_for_plane(master, data_root, backbone, seq, agg, h, k_frac, dropout, spatial_prior,
                  epochs, lr, wd, seed, device, tag, target="label_T", feat_norm="none",
                  bin_w=4, max_batch_size=16):
    m = master.set_index("patient_id")
    pids = np.sort(master["patient_id"].values.astype(np.int64))
    have = collect_plane_pids(master, data_root, backbone, seq, set(pids))
    if len(have) < 12:
        return None, None
    t0 = time.time()
    bags_raw = preload_plane(master, data_root, backbone, seq, have, target, feat_norm, device)
    folds = m.loc[have, "fold"].values.astype(int)
    oof = []
    curves_all = []
    for k in sorted(set(folds)):
        tr = have[folds != k]
        va = have[folds == k]
        if len(tr) < 8 or len(va) < 1:
            continue
        # std: 每折从原始特征重算标准化 (fit 严格仅训练折真实切片, 见 make_std_bags)
        if feat_norm == "std":
            bags = make_std_bags(bags_raw, list(tr) + list(va), list(tr))
        else:
            bags = bags_raw
        torch.manual_seed(int(seed + k))
        model = SliceMIL(D + SPATIAL_DIM if spatial_prior else D, agg=agg, h=h,
                         k_frac=k_frac, dropout=dropout, spatial_prior=spatial_prior).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        rows, curves = train_plane(bags, tr, va, model, opt, sch, epochs,
                                   spatial_prior, device, seed + k, target, bin_w,
                                   max_batch_size)
        oof.extend(rows)
        for c in curves:
            c["fold"] = int(k)
            curves_all.append(c)
        del model, opt, sch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(oof)
    df["patient_id"] = df["patient_id"].astype(int)
    df["fold"] = df["patient_id"].map(m["fold"]).astype(int)
    df = df.sort_values("patient_id")
    assert set(df["patient_id"]).issubset(set(pids)), "OOF 患者越界!"
    return df, pd.DataFrame(curves_all), time.time() - t0


# ---------------- 融合 ----------------
def logit_clip(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fuse_probmean(oofs):
    """按患者可用模态等权均值 (缺模态自动跳过, 与 audit 可用模态均值同语义)。"""
    planes = {s: o for s, o in oofs.items() if o is not None}
    all_pids = sorted(set.union(*[set(o["patient_id"]) for o in planes.values()]))
    rows = []
    for pid in all_pids:
        ps = [planes[s].set_index("patient_id").loc[pid, "prob"] for s in planes
              if pid in planes[s].set_index("patient_id").index]
        rows.append({"patient_id": pid, "prob": float(np.mean(ps))})
    return pd.DataFrame(rows)


def fuse_zlogit(oofs, master):
    """可用模态 z-score 均值 (支持缺模态): 逐患者平均其可用平面的
    折内训练 logit z-score, 与 audit s4.f1 口径一致, 不再要求全平面完整。"""
    m = master.set_index("patient_id")
    planes = {s: o for s, o in oofs.items() if o is not None}
    all_pids = sorted(set.union(*[set(o["patient_id"]) for o in planes.values()]))
    n = len(all_pids)
    F = np.full(n, np.nan); C = np.zeros(n)
    idx = {p: i for i, p in enumerate(all_pids)}
    for s, o in planes.items():
        oi = o.set_index("patient_id")
        bids = [p for p in all_pids if p in oi.index]
        if len(bids) < 8:
            continue
        p = oi.loc[bids, "prob"].values.astype(float)
        floc = m.loc[bids, "fold"].values.astype(int)
        yloc = m.loc[bids, "label_T"].values.astype(int)
        Lva = logit_clip(p)
        cal = np.full(len(bids), np.nan)
        for k in np.unique(floc):
            va = floc == k; tr = ~va
            if tr.sum() < 2 or len(np.unique(yloc[tr])) < 2:
                cal[va] = Lva[va]
                continue
            m_tr, s_tr = Lva[tr].mean(), Lva[tr].std() + 1e-9
            cal[va] = (Lva[va] - m_tr) / s_tr
        for ii, b in enumerate(bids):
            i = idx[b]
            if not np.isnan(cal[ii]):
                F[i] = cal[ii] if np.isnan(F[i]) else F[i] + cal[ii]
                C[i] += 1
    m1 = C > 0
    F[m1] /= C[m1]
    return pd.DataFrame({"patient_id": all_pids,
                         "prob": np.where(m1, F, np.nan)})


def fuse_stack(oofs, master):
    """折内 LR 堆叠 (m00879 起默认关闭, 仅 --with-stack 显式开启)。

    泄露审计 (m00879): 对第 k 折, 元学习器 LR 在训练折患者的平面 OOF 概率上拟合,
    每个概率由"排除该患者所在折"的平面模型产生 —— 相对标签是嵌套的, 但**那些平面
    模型都见过第 k 折数据**, 使 LR 的训练特征间接携带第 k 折信息, 存在隐性跨折泄露
    (比 zlogit/probmean 乐观)。仅在全模态交集上拟合 (LR 需固定特征维度);
    缺模态患者交给 probmean/zlogit; 缺失指示器版 (audit fusionB) 不实现。
    """
    m = master.set_index("patient_id")
    planes = {s: o for s, o in oofs.items() if o is not None}
    common = sorted(set.intersection(*[set(o["patient_id"]) for o in planes.values()]))
    common = [p for p in common if p in m.index]
    n = len(common)
    y = m.loc[common, "label_T"].values.astype(int)
    fl = m.loc[common, "fold"].values.astype(int)
    X = np.column_stack([planes[s].set_index("patient_id").loc[common, "prob"].values
                         for s in planes])
    P = np.full(n, np.nan)
    for k in np.unique(fl):
        tr, va = fl != k, fl == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            P[va] = np.nan
            continue
        sc = StandardScaler().fit(X[tr])
        mdl = LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs").fit(sc.transform(X[tr]), y[tr])
        P[va] = mdl.predict_proba(sc.transform(X[va]))[:, 1]
    return pd.DataFrame({"patient_id": common, "prob": P})


# ---------------- 强正则线性探针 (MeanMax 池化 + 折内嵌套 3 折 CV 选 C L2-LR) ----------------
PROBE_C_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0)


def meanmax_pool_bags(bags, pids):
    """MeanMax 池化 (掩码无关: bags 只含真实切片): 每患者 (K,D) → (2D,)。"""
    Xs, ys = [], []
    for p in pids:
        f = bags[p][0].float().cpu().numpy()               # (K,D)
        Xs.append(np.concatenate([f.mean(axis=0), f.max(axis=0)]))
        ys.append(bags[p][3])
    return np.array(Xs), np.array(ys)


def linear_probe_plane(master, data_root, backbone, seq, device, seed,
                       target="label_T", c_grid=PROBE_C_GRID):
    """MeanMax 池化 + 折内嵌套 3 折 CV 选 C 的 L2-LogisticRegression 强正则线性探针。

    与 MIL 同一数据接口 (features/slice_feats_v2/.npz)、同一 5 折 (master 折列)、
    同一 OOF 输出契约 (patient_id,fold,label_T,prob)。选 C 仅用训练折 (fold%3 内层,
    内层均值 AUC), 全训练侧重拟合后预测验证折 —— 无跨折泄露。C 网格比 audit 更宽
    (1e-4..1.0, 含强正则端), 作为真实 slice_feats_v2 特征上对比深度 MIL 的强正则基线。
    """
    m = master.set_index("patient_id")
    pids = np.sort(master["patient_id"].values.astype(np.int64))
    have = collect_plane_pids(master, data_root, backbone, seq, set(pids))
    if len(have) < 12:
        return None, None
    bags = preload_plane(master, data_root, backbone, seq, have, target, "none", device)
    folds = m.loc[have, "fold"].values.astype(int)
    X, y = meanmax_pool_bags(bags, have)
    oof = np.full(len(have), np.nan)
    Cs = {}
    for k in sorted(set(folds)):
        tr, va = folds != k, folds == k
        if tr.sum() < 8 or va.sum() < 1 or len(np.unique(y[tr])) < 2:
            oof[va] = 0.5
            Cs[k] = None
            continue
        Xtr, ytr = X[tr], y[tr]
        inner = folds[tr] % 3
        inner_auc = {}
        for c in c_grid:
            aucs = []
            for g in range(3):
                itr, iva = inner != g, inner == g
                if iva.sum() < 2 or len(np.unique(ytr[itr])) < 2:
                    continue
                sc = StandardScaler().fit(Xtr[itr])
                mm = LogisticRegression(C=c, max_iter=3000, solver="lbfgs").fit(
                    sc.transform(Xtr[itr]), ytr[itr])
                aucs.append(roc_auc_score(ytr[iva], mm.predict_proba(sc.transform(Xtr[iva]))[:, 1]))
            inner_auc[c] = float(np.mean(aucs)) if aucs else -1.0
        best_c = max(c_grid, key=lambda c: inner_auc[c])
        sc = StandardScaler().fit(Xtr)
        mm = LogisticRegression(C=best_c, max_iter=3000, solver="lbfgs").fit(sc.transform(Xtr), ytr)
        oof[va] = mm.predict_proba(sc.transform(X[va]))[:, 1]
        Cs[int(k)] = None if best_c is None else float(best_c)
    rows = [{"patient_id": int(p), "label_T": int(yy), "prob": float(pp)}
            for p, yy, pp in zip(have, y, oof)]
    df = pd.DataFrame(rows)
    df["fold"] = df["patient_id"].map(m["fold"]).astype(int)
    df = df[["patient_id", "fold", "label_T", "prob"]].sort_values("patient_id")
    return df, Cs


def pooled_auc(df):
    y = df["label_T"].values.astype(int)
    p = df["prob"].values.astype(float)
    ok = ~np.isnan(p)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return None
    return round(float(roc_auc_score(y[ok], p[ok])), 4)


# ---------------- 合成冒烟数据 ----------------
def gen_smoke(root, backbone, master, n, seed=7):
    """生成符合 .npz 契约的合成数据: K 不等长 / 部分缺序列 / mask_source 混合。"""
    rng = np.random.default_rng(seed)
    m = master.set_index("patient_id")
    pids = np.sort(master["patient_id"].values.astype(np.int64))[:n]
    cover = {"t": 0.95, "c": 0.90, "d": 0.60}
    made = 0
    for pid in pids:
        y = float(m.loc[pid, "label_T"])
        for s in SEQS:
            if rng.random() > cover[s]:
                continue
            K = 1 if rng.random() < 0.02 else int(rng.integers(8, 40))
            feats = rng.standard_normal((K, D)).astype(np.float16)
            feats[:, :16] += (y - 0.5) * 1.4  # 类信号
            rel_z = rng.uniform(0, 1, K).astype(np.float32)
            area = rng.uniform(0.05, 0.6, K).astype(np.float32)
            src = np.array([rng.choice(["self", "remapped"])])
            out = os.path.join(root, backbone, s, f"{pid}.npz")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez_compressed(out, feats=feats, slice_idx=np.arange(K, dtype=np.int32),
                                rel_z=rel_z, mask_area_ratio=area, mask_source=src)
            made += 1
    return made


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--backbone", default="convnext_tiny", help="特征子目录名 (仅路径用)")
    ap.add_argument("--planes", default="t,c,d")
    ap.add_argument("--agg", choices=["meanmax", "topk", "gated"], default="gated")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--k-frac", type=float, default=0.25)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--spatial-prior", action="store_true", help="拼接 rel_z+mask_area_ratio")
    ap.add_argument("--feat-norm", choices=["none", "l2", "std"], default="none",
                    help="l2=逐切片行归一; std=折内 StandardScaler (DINOv2/BiomedCLIP 尺度适配)")
    ap.add_argument("--target", choices=["label_T", "label_N"], default="label_T")
    ap.add_argument("--clin-oof", default=None,
                    help="临床 OOF CSV (patient_id,fold,label_T,prob) → 作为第 4 路加入融合")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bin-w", type=int, default=4,
                    help="K 分桶宽 (补齐到桶上限; 掩码聚合精确, topk 的 k 量化到 0.25×桶上限)")
    ap.add_argument("--max-batch-size", type=int, default=16,
                    help="每 epoch 单个桶内拆分的最大 batch 袋数 (保证梯度更新次数, 防大桶欠拟合)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2, help="weight decay (强正则)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fusion", default="probmean,zlogit",
                    help="融合模式逗号列表 (默认不含 stack; stack 有隐性跨折泄露, 用 --with-stack 显式开启)")
    ap.add_argument("--with-stack", action="store_true",
                    help="显式开启 stack 融合 (元学习器训练特征由见过第 k 折的平面模型 OOF 产生, 间接泄露)")
    ap.add_argument("--linear-probe", action="store_true",
                    help="跳过深度 MIL, 直接跑 MeanMax 池化 + 折内嵌套 3 折 CV 选 C 的 "
                         "L2-LogisticRegression 强正则线性探针 (C∈{1e-4,1e-3,1e-2,0.1,1.0})")
    ap.add_argument("--tag", default="mil_run")
    ap.add_argument("--smoke-gen-root", default=None,
                    help="合成数据根目录 (冒烟): 生成后作为 --data-root 使用")
    ap.add_argument("--smoke-n", type=int, default=240)
    args = ap.parse_args()

    if args.smoke_gen_root:
        root = args.smoke_gen_root
        n_made = gen_smoke(root, args.backbone, pd.read_csv(MASTER), args.smoke_n)
        args.data_root = root
        print(f"[smoke] generated {n_made} npz under {root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = pd.read_csv(MASTER)
    planes = [s for s in args.planes.split(",") if s in SEQS]
    print(f"device={device} planes={planes} agg={args.agg} epochs={args.epochs} "
          f"spatial_prior={args.spatial_prior} feat_norm={args.feat_norm} "
          f"target={args.target} clin_oof={args.clin_oof} tag={args.tag}")

    t0 = time.time()
    peak = 0
    oofs = {}
    plane_times = {}
    probe_cs = {}
    for s in planes:
        if args.linear_probe:
            df, cs = linear_probe_plane(master, args.data_root, args.backbone, s, device,
                                        args.seed, args.target)
            probe_cs[s] = cs
        else:
            df, curves, pt = oof_for_plane(master, args.data_root, args.backbone, s, args.agg,
                                           args.h, args.k_frac, args.dropout, args.spatial_prior,
                                           args.epochs, args.lr, args.wd, args.seed, device, args.tag,
                                           args.target, args.feat_norm, args.bin_w, args.max_batch_size)
            plane_times[s] = round(pt, 1) if pt is not None else None
        if df is None:
            print(f"  [{s}] 无数据, 跳过")
            continue
        oofs[s] = df
        auc = pooled_auc(df)
        tag_desc = "probe" if args.linear_probe else f"({pt:.1f}s)"
        print(f"  [{s}] OOF n={len(df)} pooled_auc={auc} {tag_desc}")
        if not args.linear_probe and device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        os.makedirs(f"outputs/mil_{args.tag}", exist_ok=True)
        df.to_csv(f"outputs/mil_{args.tag}/oof_{s}.csv", index=False)
        if not args.linear_probe and curves is not None and len(curves):
            curves.to_csv(f"outputs/mil_{args.tag}/curves_{s}.csv", index=False)

    # 临床 OOF 作为第 4 路
    if args.clin_oof:
        cdf = pd.read_csv(args.clin_oof)
        cdf = cdf[["patient_id", "fold", "label_T", "prob"]].dropna(subset=["prob"])
        oofs["clin"] = cdf
        print(f"  [clin] OOF n={len(cdf)} (from {args.clin_oof})")

    # 融合 (默认仅 probmean,zlogit 两个无泄露模式; stack 需 --with-stack)
    fusion_modes = [m for m in args.fusion.split(",") if m]
    if args.with_stack and "stack" not in fusion_modes:
        fusion_modes.append("stack")
    fus = {}
    for mode in fusion_modes:
        if mode == "probmean":
            fd = fuse_probmean(oofs)
        elif mode == "zlogit":
            fd = fuse_zlogit(oofs, master)
        elif mode == "stack":
            fd = fuse_stack(oofs, master)
        else:
            print(f"  [fusion] 未知模式 {mode}, 跳过"); continue
        if fd is not None and len(fd):
            m = master.set_index("patient_id")
            fd["label_T"] = fd["patient_id"].map(m[args.target])
            fd["fold"] = fd["patient_id"].map(m["fold"])
            fd = fd[["patient_id", "fold", "label_T", "prob"]]
            auc = pooled_auc(fd)
            fus[mode] = auc
            fd.to_csv(f"outputs/mil_{args.tag}/fusion_{mode}.csv", index=False)
            print(f"  [fusion {mode}] n={len(fd)} pooled_auc={auc}")
            if mode == "stack" and args.smoke_gen_root:
                print("  [smoke format] fusion_stack.csv head:")
                print(fd.head(3).to_string(index=False))
        # clin 专属融合文件 (clin + 3-plane MIL 晚融合)
        if args.clin_oof and mode in ("probmean", "zlogit"):
            fd2 = fd[fd["patient_id"].isin(oofs["clin"]["patient_id"])]
            fd2.to_csv(f"outputs/mil_{args.tag}/fusion_clin_{mode}.csv", index=False)
            print(f"  [fusion clin+{mode}] n={len(fd2)} pooled_auc={pooled_auc(fd2)}")
    total = time.time() - t0
    summary = {"tag": args.tag, "agg": args.agg, "epochs": args.epochs,
               "spatial_prior": args.spatial_prior, "feat_norm": args.feat_norm,
               "target": args.target, "clin_oof": args.clin_oof, "device": str(device),
               "linear_probe": args.linear_probe, "with_stack": args.with_stack,
               "bin_w": args.bin_w, "max_batch_size": args.max_batch_size,
               "plane_times_s": plane_times,
               "probe_Cs": probe_cs or None,
               "planes_oof": {s: {"n": len(o), "pooled_auc": pooled_auc(o)} for s, o in oofs.items()},
               "fusions": fus, "total_s": round(total, 1),
               "peak_cuda_gb": round(peak, 3)}
    os.makedirs(f"outputs/mil_{args.tag}", exist_ok=True)
    with open(f"outputs/mil_{args.tag}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nDONE → outputs/mil_{args.tag}/  (total={round(total,1)}s peak_cuda={round(peak,3)}GB)")

    if args.smoke_gen_root:
        shutil.rmtree(args.smoke_gen_root, ignore_errors=True)
        shutil.rmtree(f"outputs/mil_{args.tag}", ignore_errors=True)
        print("[smoke] temp artifacts cleaned")


if __name__ == "__main__":
    main()
