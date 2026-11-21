# -*- coding: utf-8 -*-
"""
round9-2.5D 多平面训练脚本（不占用 3D 主流程）。
- 输入: CT 卷 [1,96,96,96] 与 MRI 三序列 [3,96,96,96]（复用 loader 的确定性磁盘缓存 cache_hos）
- 2.5D: 各模态取轴位中间 3 张连续切片 (z=46,47,48) → [3,96,96] 多通道
- 主干: 4× torchvision ResNet18(ImageNet 预训练, 去 fc), conv1 天然 3 通道匹配 3 切片
- 池化: GeM(p=3.0, FP32 计算) → 特征 concat(4×512=2048) + 临床7 → FC256 → T/N 双头
- 训练: R5 同口径 (lr 1e-4 / lr-n 3e-4 分离, T/N 各自逆频权重, 60ep, AMP, CosineAnnealing,
         best T/N 分别保存并以 best 权重做校准, 归档 results/round9/run_NN_2p5d_...)
用法:
  python tools/train_2p5d.py                 # 全量 seed42 60ep
  python tools/train_2p5d.py --epochs 2 --smoke  # CPU 冒烟
"""
import argparse, json, os, sys, time, shutil, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loader.loader import _DetDiskCache, _det_cache_key, _det_cache_key_mri  # noqa: E402
from utils import setup_logger, calibrate_binary_threshold  # noqa: E402

CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
SPATIAL = (96, 96, 96)
Z_SLICE = (46, 47, 48)  # 轴位中间 3 层
BRANCHES = ["ct", "mri_t", "mri_c", "mri_d"]


class GeM2D(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        # FP32 计算后还原 dtype（与 3D 版 GeM 一致）
        dt = x.dtype
        x = x.float().clamp(min=1e-6).pow(self.p)
        out = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p).flatten(1)
        return out.to(dt)


class ResNetGeM(nn.Module):
    """ResNet18 特征图部分 (conv1..layer4) + GeM 池化 → [B,512]（输入 96² → 特征图 3×3）。"""

    def __init__(self, base, p=3.0):
        super().__init__()
        self.features = nn.Sequential(*list(base.children())[:-2])  # 去 avgpool+fc
        self.gem = GeM2D(p)

    def forward(self, x):
        return self.gem(self.features(x))


class MultiBranch2D(nn.Module):
    def __init__(self, num_clinical=7, head_width=256, head_dropout=0.5,
                 freeze_prefixes=("conv1", "bn1", "relu", "maxpool", "layer1", "layer2")):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.ct_net = ResNetGeM(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))
        self.mri_nets = nn.ModuleDict(
            {s: ResNetGeM(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)) for s in ("t", "c", "d")})
        self.num_clinical = num_clinical
        in_feats = 4 * 512 + num_clinical
        self.fuse = nn.Sequential(
            nn.Linear(in_feats, head_width), nn.ReLU(inplace=True), nn.Dropout(head_dropout))
        self.head_t = nn.Linear(head_width, 2)
        self.head_n = nn.Linear(head_width, 2)
        self._freeze(freeze_prefixes)

    def _freeze(self, prefixes):
        n_frozen = 0
        for name, p in self.named_parameters():
            if any(("." + pr + ".") in name for pr in prefixes):
                p.requires_grad = False
                n_frozen += 1
        print(f"[2.5D] freeze params: {n_frozen}")

    def forward(self, ct, mri, clin, ct_mask, mri_mask):
        # ct: [B,1,96,96,96]  mri: [B,3,96,96,96] (t,c,d 序)  mask: [B] bool
        feats = []
        b = ct.size(0)
        ct_s = ct[:, 0, Z_SLICE[0]:Z_SLICE[2] + 1]          # [B,3,96,96]
        feats.append(self.ct_net(ct_s) * ct_mask.view(b, 1))
        for i, s in enumerate(("t", "c", "d")):
            ms = mri[:, i, Z_SLICE[0]:Z_SLICE[2] + 1]        # [B,3,96,96]
            m_avail = mri_mask[:, i] if mri_mask.dim() == 2 else mri_mask
            feats.append(self.mri_nets[s](ms) * m_avail.view(b, 1))
        x = torch.cat(feats + [clin], dim=1)                # [B, 4*512+7]
        h = self.fuse(x)
        return self.head_t(h), self.head_n(h)


class Data2D:
    """预加载全部确定性卷（内存）, 提供切片后的样本。"""

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.disk = _DetDiskCache(os.path.join(ROOT, "cache_hos"), SPATIAL)
        self._vols = []  # (ct [1,96,96,96] or None, mri [3,96,96,96] or None, ct_avail, mri_avail[3])
        self._clin = []
        self._lab_t = []
        self._lab_n = []
        miss_ct = miss_mri = 0
        for _, row in self.df.iterrows():
            ct_path = row.get("ct_path", "")
            ct_avail = isinstance(ct_path, str) and bool(ct_path) and os.path.exists(ct_path)
            t_path = row.get("mri_path_t", row.get("mri_path", ""))
            c_path = row.get("mri_path_c", "")
            d_path = row.get("mri_path_d", "")
            seq_avail = [
                bool(row.get("mri_available_t", True)) and isinstance(t_path, str) and os.path.exists(t_path),
                bool(row.get("mri_available_c", True)) and isinstance(c_path, str) and os.path.exists(c_path),
                bool(row.get("mri_available_d", True)) and isinstance(d_path, str) and os.path.exists(d_path),
            ]
            mri_avail = any(seq_avail)
            ct = None
            mri = None
            if ct_avail:
                ct, _ = self.disk.get(ct_path)
            if mri_avail:
                key = _det_cache_key_mri(t_path, c_path, d_path, SPATIAL)
                mri, _ = self.disk.get_vol(key)
            # 缓存缺失时不能用零卷冒充真实模态 → 跳过并计数 (避免静默喂零)
            if (ct_avail and ct is None) or (mri_avail and mri is None):
                if ct_avail and ct is None:
                    miss_ct += 1
                if mri_avail and mri is None:
                    miss_mri += 1
                continue
            self._vols.append((ct, mri, ct_avail, seq_avail))
            clin = []
            for c in CLINICAL_COLS:
                v = row.get(c, 0.0)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                clin.append(float(v))
            self._clin.append(clin)
            self._lab_t.append(int(row["label_T"]) if str(row.get("label_T", "")).strip() not in ("", "nan") else -1)
            self._lab_n.append(int(row["label_N"]) if str(row.get("label_N", "")).strip() not in ("", "nan") else -1)
        print(f"[2.5D] {os.path.basename(csv_path)} 载入 {len(self._vols)}/{len(self.df)} 样本 "
              f"(缓存缺失跳过: CT={miss_ct} MRI={miss_mri})")

    def __len__(self):
        return len(self._vols)

    def __getitem__(self, idx):
        ct, mri, ct_avail, seq_avail = self._vols[idx]
        ct_v = torch.zeros(1, *SPATIAL) if ct is None else ct
        mri_v = torch.zeros(3, *SPATIAL) if mri is None else mri
        return (ct_v, mri_v,
                torch.tensor(self._clin[idx], dtype=torch.float32),
                torch.tensor(ct_avail, dtype=torch.bool),
                torch.tensor(seq_avail, dtype=torch.bool),
                torch.tensor(self._lab_t[idx], dtype=torch.long),
                torch.tensor(self._lab_n[idx], dtype=torch.long))


def make_loader(ds, batch_size, shuffle, drop_last=True, smoke=False):
    from torch.utils.data import DataLoader
    if smoke:
        idxs = list(range(min(24, len(ds))))
        ds = torch.utils.data.Subset(ds, idxs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)


def _inv_w(labels, device):
    """按标签序列逆频计算类别权重 (仅用传入标签)。"""
    arr = np.array([int(l) for l in labels if int(l) >= 0], dtype=np.int64)
    if arr.size == 0:
        return torch.ones(2, dtype=torch.float32, device=device)
    bc = np.bincount(arr, minlength=2).astype(np.float64)
    bc[bc == 0] = 1.0
    return torch.tensor(arr.size / (2.0 * bc), dtype=torch.float32, device=device)


def _tn_loss(crit_t, crit_n, logit_t, logit_n, lt, ln, vt, vn):
    """T/N 各自剔除 label<0 的样本后再算 CE。"""
    loss = logit_t.float().sum() * 0.0
    if bool(vt.any()):
        loss = loss + crit_t(logit_t.float()[vt], lt[vt])
    if bool(vn.any()):
        loss = loss + crit_n(logit_n.float()[vn], ln[vn])
    return loss


def _val_probs(model, loader, device, task, ckpt_path):
    """加载指定 ckpt 重算 val 概率 (校准须对应 best 权重)。"""
    if not os.path.exists(ckpt_path):
        return None, None
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for ct, mri, clin, cm, mm, lt, ln in loader:
            ct, mri, clin, cm, mm = (x.to(device) for x in (ct, mri, clin, cm, mm))
            logit_t, logit_n = model(ct, mri, clin, cm, mm)
            logit, y = (logit_t, lt) if task == "t" else (logit_n, ln)
            v = y >= 0
            if v.any():
                ps.append(F.softmax(logit.float(), 1)[v].cpu())
                ys.append(y[v].cpu())
    if not ps:
        return None, None
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-n", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--head-width", type=int, default=256)
    ap.add_argument("--head-dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-tag", default="hos_full")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if (torch.cuda.is_available() and not args.smoke) else "cpu"
    logger = setup_logger(os.path.join(ROOT, "logs"), f"2p5d_{args.out_tag}_s{args.seed}")

    train_csv = os.path.join(ROOT, "data", f"{args.out_tag}_train_s{args.seed}.csv")
    val_csv = os.path.join(ROOT, "data", f"{args.out_tag}_val_s{args.seed}.csv")
    ds_tr = Data2D(train_csv)
    ds_va = Data2D(val_csv)
    tr = make_loader(ds_tr, args.batch_size, True, smoke=args.smoke)
    va = make_loader(ds_va, args.batch_size, False, drop_last=False, smoke=args.smoke)
    logger.info(f"[2.5D] train={len(ds_tr)} val={len(ds_va)} epochs={args.epochs} device={device} "
                f"lr={args.lr} lr_n={args.lr_n} head_width={args.head_width}")

    model = MultiBranch2D(num_clinical=len(CLINICAL_COLS), head_width=args.head_width,
                          head_dropout=args.head_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[2.5D] trainable {n_params/1e6:.1f}M / {all_params/1e6:.1f}M")

    # 类别权重: T/N 各自按本任务标签逆频 (旧版 N 误用 T 权重)
    if args.smoke:
        w_t = torch.ones(2, dtype=torch.float32, device=device)
        w_n = torch.ones(2, dtype=torch.float32, device=device)
    else:
        w_t = _inv_w(ds_tr._lab_t, device)
        w_n = _inv_w(ds_tr._lab_n, device)
    logger.info(f"[2.5D] weights T={w_t.tolist()} N={w_n.tolist()}")

    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "head_n" not in n], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "head_n" in n], "lr": args.lr_n},
    ], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"), init_scale=2.0 ** 10)
    crit_t = nn.CrossEntropyLoss(weight=w_t)
    crit_n = nn.CrossEntropyLoss(weight=w_n)

    best_t, best_n = (-1.0, -1), (-1.0, -1)
    curves = {"train_loss": [], "val_t": [], "val_n": [], "val_acc": []}
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, cnt = 0.0, 0
        for ct, mri, clin, cm, mm, lt, ln in tr:
            ct, mri, clin, cm, mm, lt, ln = (x.to(device) for x in (ct, mri, clin, cm, mm, lt, ln))
            vt, vn = lt >= 0, ln >= 0
            if not (vt.any() or vn.any()):
                continue
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logit_t, logit_n = model(ct, mri, clin, cm, mm)
                loss = _tn_loss(crit_t, crit_n, logit_t, logit_n, lt, ln, vt, vn)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * max(int(vt.sum().item()), 1)
            cnt += max(int(vt.sum().item()), 1)
        sched.step()
        # val
        model.eval()
        pt, pn, yt, yn = [], [], [], []
        with torch.no_grad():
            for ct, mri, clin, cm, mm, lt, ln in va:
                ct, mri, clin, cm, mm = (x.to(device) for x in (ct, mri, clin, cm, mm))
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    logit_t, logit_n = model(ct, mri, clin, cm, mm)
                vt, vn = lt >= 0, ln >= 0
                if vt.any():
                    pt.append(F.softmax(logit_t.float(), 1)[vt])
                    yt.append(lt[vt])
                if vn.any():
                    pn.append(F.softmax(logit_n.float(), 1)[vn])
                    yn.append(ln[vn])
        pt = torch.cat(pt) if pt else torch.zeros(0, 2)
        pn = torch.cat(pn) if pn else torch.zeros(0, 2)
        yt = torch.cat(yt) if yt else torch.zeros(0, dtype=torch.long)
        yn = torch.cat(yn) if yn else torch.zeros(0, dtype=torch.long)
        mt = macro_f1(pt, yt) if len(yt) else 0.0
        mn = macro_f1(pn, yn) if len(yn) else 0.0
        acc = (pt.argmax(1) == yt).float().mean().item() if len(yt) else 0.0
        curves["train_loss"].append(tot / max(cnt, 1))
        curves["val_t"].append(mt)
        curves["val_n"].append(mn)
        curves["val_acc"].append(acc)
        if mt > best_t[0]:
            best_t = (mt, ep)
            torch.save(model.state_dict(), os.path.join(ROOT, "checkpoints_hos_full", f"best_model_2p5d_{args.out_tag}_s{args.seed}.pth"))
        if mn > best_n[0]:
            best_n = (mn, ep)
            torch.save(model.state_dict(), os.path.join(ROOT, "checkpoints_hos_full", f"best_model_n_2p5d_{args.out_tag}_s{args.seed}.pth"))
        if ep % 10 == 0 or ep == 1 or ep == args.epochs or args.smoke:
            logger.info(f"[2.5D] Epoch {ep:03d} loss={curves['train_loss'][-1]:.4f} "
                        f"T={mt:.4f}(best {best_t[0]:.4f}@{best_t[1]}) N={mn:.4f}(best {best_n[0]:.4f}@{best_n[1]}) acc={acc:.4f}")
    logger.info(f"[2.5D] done. best T={best_t[0]:.4f}@{best_t[1]}  best N={best_n[0]:.4f}@{best_n[1]}")

    # 校准 + 报告
    if args.smoke:
        cal_t = None
        cal_n = None
    else:
        # 用 best 权重重算 val 概率 (旧版用末轮权重, 与落盘 ckpt 不一致)
        pt_b, yt_b = _val_probs(model, va, device, "t",
                                os.path.join(ROOT, "checkpoints_hos_full", f"best_model_2p5d_{args.out_tag}_s{args.seed}.pth"))
        pn_b, yn_b = _val_probs(model, va, device, "n",
                                os.path.join(ROOT, "checkpoints_hos_full", f"best_model_n_2p5d_{args.out_tag}_s{args.seed}.pth"))
        cal_t = calibrate_binary_threshold(pt_b, yt_b) if pt_b is not None else None
        cal_n = calibrate_binary_threshold(pn_b, yn_b) if pn_b is not None else None
    calT = (cal_t.get("best") if isinstance(cal_t, dict) and "best" in cal_t else cal_t) if cal_t else None
    calN = (cal_n.get("best") if isinstance(cal_n, dict) and "best" in cal_n else cal_n) if cal_n else None
    def _best_dict(c):
        if c is None:
            return None
        m = c.get("metrics", {}) if isinstance(c, dict) else {}
        return {"threshold": c.get("threshold", 0.5), "macro_f1": m.get("macro_f1"),
                "recall": m.get("per_class_recall", m.get("recall"))}
    calT = _best_dict(calT)
    calN = _best_dict(calN)
    report = {
        "meta": {"branch": args.out_tag, "run": "2p5d", "tag": args.out_tag, "seed": args.seed, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "params": {"backbone": "resnet18x4(2.5D ImageNet)", "lr": args.lr, "lr_n": args.lr_n,
                   "head_width": args.head_width, "pool": "gem3", "epochs": args.epochs, "z_slice": list(Z_SLICE)},
        "result": {
            "best_epoch": best_t[1], "best_macro_f1": best_t[0],
            "best_acc": curves["val_acc"][best_t[1] - 1] if best_t[1] > 0 else None,
            "best_epoch_n": best_n[1], "best_macro_f1_n": best_n[0],
            "calT": None if calT is None else {"threshold": calT.get("threshold", 0.5), "macro_f1": calT.get("macro_f1"),
                     "recall": calT.get("recall")},
            "calN": None if calN is None else {"threshold": calN.get("threshold", 0.5), "macro_f1": calN.get("macro_f1"),
                     "recall": calN.get("recall")},
        },
    }
    rp = os.path.join(ROOT, "logs", f"eval_report_2p5d_{args.out_tag}_s{args.seed}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"[2.5D] report saved: {rp}")
    if not args.smoke:
        archive(ROOT, args.out_tag, args.seed, report)


def macro_f1(prob, y):
    pred = prob.argmax(1)
    cm = torch.zeros(2, 2, dtype=torch.long)
    for a, b in zip(pred.tolist(), y.tolist()):
        if a < 2 and b < 2 and b >= 0:
            cm[a, b] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.array([cm[i, i] / max(cm[:, i].sum().item(), 1) for i in range(2)])
        prec = np.array([cm[i, i] / max(cm[i, :].sum().item(), 1) for i in range(2)])
        f1 = 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1)
    return float(np.nanmean(f1))


def archive(root, tag, seed, report):
    dst_base = None
    runs = sorted(glob.glob(os.path.join(root, "results", "round9", "run_*")))
    nxt = 1
    for r in runs:
        try:
            nxt = max(nxt, int(os.path.basename(r).split("_")[1]) + 1)
        except (IndexError, ValueError):
            pass
    dst = os.path.join(root, "results", "round9", f"run_{nxt:02d}_2p5d_{tag}_s{seed}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    while True:  # 原子占位, 避免并发训练撞号 (历史 run_93 重复即此原因)
        try:
            os.makedirs(dst)
            break
        except FileExistsError:
            nxt += 1
            dst = os.path.join(root, "results", "round9", f"run_{nxt:02d}_2p5d_{tag}_s{seed}")
    for f in glob.glob(os.path.join(root, "logs", f"eval_report_2p5d_{tag}_s{seed}.*")):
        shutil.copy2(f, dst)
    for f in glob.glob(os.path.join(root, "checkpoints_hos_full", f"*2p5d_{tag}_s{seed}.pth")):
        shutil.copy2(f, dst)
    for f in glob.glob(os.path.join(root, "logs", f"2p5d_{tag}_s{seed}_*.log")):
        shutil.copy2(f, dst)
    print(f"[2.5D] archived -> {dst}")


if __name__ == "__main__":
    main()
