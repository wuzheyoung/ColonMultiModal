# -*- coding: utf-8 -*-
"""
round9-C1 Patch-MIL 训练脚本（无 ROI 弱监督切块定位）。
- 动机: N 卡 0.51-0.58 = 全体积统计拿不到局部浸润信息 (CAM/radiomics/CRCFound 印证)。
        MIL (多实例学习) 无需病灶标注: 把卷切块, 模型自己学哪些块对 N 判别重要 = 隐式定位。
- 输入: CT 卷 [1,96,96,96] (N 依赖 CT, B3b/D1a 2-seed N 0.5792 证实; 8GB 显存考虑只用 CT)
- 切块: 无重叠 2x2x2 = 8 块 x 48^3 (用户确认无重叠; 8 块显存友好, 48^3 块不切碎病灶)
- 块编码器: MedicalNet resnet10 (medicalnet10 预训练 72/72, in_channels=1, 共享参数)
        输入 [B*8,1,48,48,48] 一次前向 → 每块 [512] (GAP)
- MIL 聚合: attention-MIL (per-block Linear(512→1) → softmax over 8 块) → 加权 [B,512]
- 头: concat(聚合特征, 临床7) → FC(head_width) → T/N 双头 (T/N 各自按本任务标签逆频加权)
- 训练: R5 同口径 (lr 1e-4 / lr-n 3e-4 分离, 60ep, AMP, CosineAnnealing, best T/N 分别保存,
        以 best 权重做阈值校准, 归档 results/round9/run_NN_mil_hos_full_s42)
用法:
  python tools/train_mil.py                      # 全量 seed42 60ep
  python tools/train_mil.py --epochs 1 --smoke   # CPU 冒烟
"""
import argparse, json, os, sys, time, shutil, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loader.loader import _DetDiskCache  # noqa: E402
from utils import setup_logger, calibrate_binary_threshold  # noqa: E402

CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
SPATIAL = (96, 96, 96)
BLOCKS_PER_DIM = 2  # 无重叠 2x2x2 = 8 块
BLOCK_SIZE = 48     # 96 / 2


class PatchMILNet(nn.Module):
    """共享 3D 块编码器 (medicalnet10) + attention-MIL 聚合 + 双头。"""

    def __init__(self, num_clinical=7, head_width=256, head_dropout=0.5,
                 freeze_prefixes=("conv1", "bn1", "relu", "maxpool", "layer1", "layer2")):
        super().__init__()
        from models.backbone import build_backbone
        enc = build_backbone("medicalnet10", pretrained=True, pretrained_dir=os.path.join(ROOT, "pretrained"))
        if enc.pretrain_msg:
            print(f"[MIL] {enc.pretrain_msg}")
        self.encoder = enc  # MedicalNetBackbone: forward → list of 5 stage feats
        self.num_clinical = num_clinical
        out_c = enc.out_channels  # 512
        self.att = nn.Linear(out_c, 1)          # attention-MIL 打分
        self.fuse = nn.Sequential(
            nn.Linear(out_c + num_clinical, head_width), nn.ReLU(inplace=True), nn.Dropout(head_dropout))
        self.head_t = nn.Linear(head_width, 2)
        self.head_n = nn.Linear(head_width, 2)
        self._freeze(freeze_prefixes)

    def _freeze(self, prefixes):
        n_frozen = 0
        for name, p in self.named_parameters():
            if any(("." + pr + ".") in name for pr in prefixes):
                p.requires_grad = False
                n_frozen += 1
        print(f"[MIL] freeze params: {n_frozen}")

    def _block_feats(self, ct):
        """ct: [B,1,96,96,96] → 切 8 块 → 共享编码器 → [B,8,512]"""
        B = ct.size(0)
        x = ct[:, 0]  # [B,96,96,96]
        # 无重叠切块: [B,2,48,2,48,2,48] → [B,8,48,48,48]
        x = x.view(B, BLOCKS_PER_DIM, BLOCK_SIZE, BLOCKS_PER_DIM, BLOCK_SIZE,
                   BLOCKS_PER_DIM, BLOCK_SIZE)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()  # [B,2,2,2,48,48,48]
        x = x.view(B * 8, 1, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        feats = self.encoder(x)[-1]                      # [B*8, 512, 1,1,1]
        feats = F.adaptive_avg_pool3d(feats, 1).flatten(1)  # [B*8, 512]
        return feats.view(B, 8, -1)

    def forward(self, ct, clin, ct_mask):
        """ct: [B,1,96,96,96]  clin: [B,7]  ct_mask: [B] bool → (logit_t, logit_n)"""
        b = ct.size(0)
        bf = self._block_feats(ct)                       # [B,8,512]
        att = self.att(bf).squeeze(-1)                    # [B,8]
        # 不可用样本 (无 CT): 各块 0 特征 → 统一权重 (不放大噪声)
        avail = ct_mask.float().view(b, 1)
        att = att + (1 - avail) * -1e9
        att_w = torch.softmax(att, dim=1)                 # [B,8]
        pooled = (bf * att_w.unsqueeze(-1)).sum(dim=1)    # [B,512]
        h = self.fuse(torch.cat([pooled, clin], dim=1))
        return self.head_t(h), self.head_n(h)


class DataMIL:
    """预加载 CT 确定性卷 (复用 cache_hos), 提供切块样本。"""

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.disk = _DetDiskCache(os.path.join(ROOT, "cache_hos"), SPATIAL)
        self._vols = []   # (ct [1,96,96,96] np or None, ct_avail)
        self._clin = []
        self._lab_t = []
        self._lab_n = []
        miss_ct = 0
        for _, row in self.df.iterrows():
            ct_path = row.get("ct_path", "")
            ct_avail = isinstance(ct_path, str) and bool(ct_path) and os.path.exists(ct_path)
            ct = None
            if ct_avail:
                ct, _ = self.disk.get(ct_path)
            if ct_avail and ct is None:  # 缓存缺失不能当零卷用
                miss_ct += 1
                continue
            self._vols.append((ct, ct_avail))
            clin = []
            for c in CLINICAL_COLS:
                v = row.get(c, 0.0)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                clin.append(float(v))
            self._clin.append(clin)
            self._lab_t.append(int(row["label_T"]) if str(row.get("label_T", "")).strip() not in ("", "nan") else -1)
            self._lab_n.append(int(row["label_N"]) if str(row.get("label_N", "")).strip() not in ("", "nan") else -1)
        print(f"[MIL] {os.path.basename(csv_path)} 载入 {len(self._vols)}/{len(self.df)} 样本 "
              f"(缓存缺失跳过: CT={miss_ct})")

    def __len__(self):
        return len(self._vols)

    def __getitem__(self, idx):
        ct, ct_avail = self._vols[idx]
        ct_v = torch.zeros(1, *SPATIAL) if ct is None else torch.as_tensor(np.ascontiguousarray(ct), dtype=torch.float32)
        return (ct_v,
                torch.tensor(self._clin[idx], dtype=torch.float32),
                torch.tensor(ct_avail, dtype=torch.bool),
                torch.tensor(self._lab_t[idx], dtype=torch.long),
                torch.tensor(self._lab_n[idx], dtype=torch.long))


def make_loader(ds, batch_size, shuffle, drop_last=True, smoke=False):
    from torch.utils.data import DataLoader
    if smoke:
        idxs = list(range(min(16, len(ds))))
        ds = torch.utils.data.Subset(ds, idxs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)


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
        for ct, clin, cm, lt, ln in loader:
            ct, clin, cm = (x.to(device) for x in (ct, clin, cm))
            logit_t, logit_n = model(ct, clin, cm)
            logit, y = (logit_t, lt) if task == "t" else (logit_n, ln)
            v = y >= 0
            if v.any():
                ps.append(F.softmax(logit.float(), 1)[v].cpu())
                ys.append(y[v].cpu())
    if not ps:
        return None, None
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def archive(root, tag, seed, report):
    runs = sorted(glob.glob(os.path.join(root, "results", "round9", "run_*")))
    nxt = 1
    for r in runs:
        try:
            nxt = max(nxt, int(os.path.basename(r).split("_")[1]) + 1)
        except (IndexError, ValueError):
            pass
    dst = os.path.join(root, "results", "round9", f"run_{nxt:02d}_mil_{tag}_s{seed}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    while True:  # 原子占位, 避免并发撞号
        try:
            os.makedirs(dst)
            break
        except FileExistsError:
            nxt += 1
            dst = os.path.join(root, "results", "round9", f"run_{nxt:02d}_mil_{tag}_s{seed}")
    for f in glob.glob(os.path.join(root, "logs", f"eval_report_mil_{tag}_s{seed}.*")):
        shutil.copy2(f, dst)
    for f in glob.glob(os.path.join(root, "checkpoints_hos_full", f"*mil_{tag}_s{seed}.pth")):
        shutil.copy2(f, dst)
    for f in glob.glob(os.path.join(root, "logs", f"mil_{tag}_s{seed}_*.log")):
        shutil.copy2(f, dst)
    print(f"[MIL] archived -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-n", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--head-width", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-tag", default="hos_full")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if (torch.cuda.is_available() and not args.smoke) else "cpu"
    logger = setup_logger(os.path.join(ROOT, "logs"), f"mil_{args.out_tag}_s{args.seed}")

    train_csv = os.path.join(ROOT, "data", f"{args.out_tag}_train_s{args.seed}.csv")
    val_csv = os.path.join(ROOT, "data", f"{args.out_tag}_val_s{args.seed}.csv")
    ds_tr = DataMIL(train_csv)
    ds_va = DataMIL(val_csv)
    tr = make_loader(ds_tr, args.batch_size, True, smoke=args.smoke)
    va = make_loader(ds_va, args.batch_size, False, drop_last=False, smoke=args.smoke)
    logger.info(f"[MIL] train={len(ds_tr)} val={len(ds_va)} epochs={args.epochs} device={device} "
                f"lr={args.lr} lr_n={args.lr_n} head_width={args.head_width} blocks=2x2x2(48^3)")

    model = PatchMILNet(num_clinical=len(CLINICAL_COLS), head_width=args.head_width).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[MIL] trainable {n_params/1e6:.1f}M / {all_params/1e6:.1f}M")

    if args.smoke:
        w_t = torch.ones(2, dtype=torch.float32, device=device)
        w_n = torch.ones(2, dtype=torch.float32, device=device)
    else:
        # T/N 各自按本任务标签逆频 (旧版 N 误用 T 权重)
        w_t = _inv_w(ds_tr._lab_t, device)
        w_n = _inv_w(ds_tr._lab_n, device)
    logger.info(f"[MIL] weights T={w_t.tolist()} N={w_n.tolist()}")

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
        for ct, clin, cm, lt, ln in tr:
            ct, clin, cm, lt, ln = (x.to(device) for x in (ct, clin, cm, lt, ln))
            vt, vn = lt >= 0, ln >= 0
            if not (vt.any() or vn.any()):
                continue
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logit_t, logit_n = model(ct, clin, cm)
                loss = _tn_loss(crit_t, crit_n, logit_t, logit_n, lt, ln, vt, vn)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * max(int(vt.sum().item()), 1)
            cnt += max(int(vt.sum().item()), 1)
        sched.step()
        model.eval()
        pt, pn, yt, yn = [], [], [], []
        with torch.no_grad():
            for ct, clin, cm, lt, ln in va:
                ct, clin, cm = (x.to(device) for x in (ct, clin, cm))
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    logit_t, logit_n = model(ct, clin, cm)
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
            torch.save(model.state_dict(), os.path.join(ROOT, "checkpoints_hos_full", f"best_model_mil_{args.out_tag}_s{args.seed}.pth"))
        if mn > best_n[0]:
            best_n = (mn, ep)
            torch.save(model.state_dict(), os.path.join(ROOT, "checkpoints_hos_full", f"best_model_n_mil_{args.out_tag}_s{args.seed}.pth"))
        if ep % 10 == 0 or ep == 1 or ep == args.epochs or args.smoke:
            logger.info(f"[MIL] Epoch {ep:03d} loss={curves['train_loss'][-1]:.4f} "
                        f"T={mt:.4f}(best {best_t[0]:.4f}@{best_t[1]}) N={mn:.4f}(best {best_n[0]:.4f}@{best_n[1]}) acc={acc:.4f}")
    logger.info(f"[MIL] done. best T={best_t[0]:.4f}@{best_t[1]}  best N={best_n[0]:.4f}@{best_n[1]}")

    if args.smoke:
        cal_t = None
        cal_n = None
    else:
        # 校准用 best 权重 (旧版用末轮权重, 与落盘 ckpt 不一致)
        pt_b, yt_b = _val_probs(model, va, device, "t",
                                os.path.join(ROOT, "checkpoints_hos_full", f"best_model_mil_{args.out_tag}_s{args.seed}.pth"))
        pn_b, yn_b = _val_probs(model, va, device, "n",
                                os.path.join(ROOT, "checkpoints_hos_full", f"best_model_n_mil_{args.out_tag}_s{args.seed}.pth"))
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
        "meta": {"branch": args.out_tag, "run": "mil", "tag": args.out_tag, "seed": args.seed, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        "params": {"backbone": "medicalnet10(shared block enc) + attn-MIL", "blocks": "2x2x2x48^3",
                   "lr": args.lr, "lr_n": args.lr_n, "head_width": args.head_width, "epochs": args.epochs},
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
    rp = os.path.join(ROOT, "logs", f"eval_report_mil_{args.out_tag}_s{args.seed}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"[MIL] report saved: {rp}")
    if not args.smoke:
        archive(ROOT, args.out_tag, args.seed, report)


if __name__ == "__main__":
    main()
