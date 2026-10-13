import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from models import FocalLoss
from utils import compute_metrics, plot_curves, plot_confusion_matrix


class Trainer:
    """训练与验证循环（支持模态 dropout + 辅助 loss + 重建 loss）。"""

    def __init__(self, model, train_loader, val_loader, cfg, logger=None,
                 class_weights=None, label_smoothing=0.0,
                 use_focal_loss=False, focal_gamma=2.0,
                 freeze_backbone_layers=None):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger

        # ── 结构减负: 冻结 backbone 低层, 只训练高层 + 融合/头 ──
        frozen = self._freeze_backbone(self.model, freeze_backbone_layers or [])
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._log(f"冻结 backbone 参数组: {frozen}  "
                  f"可训练参数: {trainable_params / 1e6:.1f}M / {total_params / 1e6:.1f}M")

        # T 主任务: 类别权重 + 标签平滑 (可用 Focal Loss 替代); N 次任务: 仅标签平滑
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        if class_weights is not None:
            class_weights = class_weights.to(cfg.device)
        if use_focal_loss:
            self.criterion_t = FocalLoss(
                gamma=focal_gamma, alpha=class_weights, label_smoothing=label_smoothing
            )
        else:
            self.criterion_t = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            )
        self.criterion_n = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion = self.criterion_t  # T 主任务别名
        # round6 B3: MixUp 数据增强 (仅作用于分类 loss: main/n/aux; rec 不受影响)
        self.mixup = getattr(cfg, "mixup", 0.0)
        self.mixup_alpha = getattr(cfg, "mixup_alpha", 0.2)
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params_list, lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)

        # 混合精度：仅在 CUDA 上启用
        self.use_amp = getattr(cfg, "use_amp", False) and str(cfg.device).startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.modality_dropout = getattr(cfg, "modality_dropout",
                                        {"ct_only": 0.15, "mri_only": 0.15, "both": 0.7})
        self.loss_weights = getattr(cfg, "loss_weights", {"aux": 0.3, "rec": 0.5})

        self.best_acc = 0.0
        self.best_macro_f1 = 0.0
        self.best_epoch = 0
        self.best_state_dict = None        # best 权重 (CPU), 供校准/评估使用
        # N 头独立 best: 按验证集 N 的 macro-F1 选择, 避免被 T 的 best epoch 绑架
        self.best_macro_f1_n = 0.0
        self.best_epoch_n = 0
        self.best_state_dict_n = None
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.plot_dir = getattr(cfg, "plot_dir", "./plots")
        self.num_classes = getattr(cfg, "num_classes", 2)
        self.history = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "val_macro_f1": [],
            "val_n_macro_f1": [],
            "train_main": [], "train_aux": [], "train_rec": [],
        }

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    @staticmethod
    def _freeze_backbone(model, layers):
        """冻结 CT/MRI backbone encoder 低层子模块。返回冻结参数组数。

        兼容 round5 三序列 MRI backbone (mri_backbones.{t,c,d}) 与
        round6 共享模式 (mri_shared_backbone)。MONAI ResNetFeatures 的
        激活层名为 'act' (配置里旧写法 'relu' 也兼容)。
        """
        if not layers:
            return 0
        layer_names = set(layers)
        if "relu" in layer_names:
            layer_names.discard("relu")
            layer_names.add("act")
        prefixes = []
        for layer in layer_names:
            prefixes.append(("ct_backbone", f"encoder.{layer}"))
            prefixes.append(("mri_shared_backbone", f"encoder.{layer}"))
            for seq in ("t", "c", "d"):
                prefixes.append((f"mri_backbones.{seq}", f"encoder.{layer}"))
        frozen = 0
        for name, param in model.named_parameters():
            for mod_prefix, layer_suffix in prefixes:
                if name == f"{mod_prefix}.{layer_suffix}" or \
                   name.startswith(f"{mod_prefix}.{layer_suffix}."):
                    param.requires_grad = False
                    frozen += 1
                    break
        return frozen

    def _sample_modality_masks(self, batch_size, device, ct_avail=None, mri_avail=None):
        """每个样本独立采样模态可用性；真实缺失的模态强制 False。"""
        p_ct = self.modality_dropout.get("ct_only", 0.15)
        p_mri = self.modality_dropout.get("mri_only", 0.15)
        ct_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        mri_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            r = random.random()
            if r < p_mri:
                mri_mask[i] = False
            elif r < p_mri + p_ct:
                ct_mask[i] = False
        if ct_avail is not None:
            ct_mask = ct_mask & ct_avail.to(device).bool()
        if mri_avail is not None:
            mri_mask = mri_mask & mri_avail.to(device).bool()
        return ct_mask, mri_mask

    @staticmethod
    def _reconstruction_loss(imputed_list, target_list, valid_mask):
        """对各 stage 池化向量计算 MSE + (1 - cosine)，仅在 valid_mask=True 的样本上累计。"""
        if not valid_mask.any():
            return torch.tensor(0.0, device=valid_mask.device)
        total = 0.0
        for imp, tgt in zip(imputed_list, target_list):
            imp_v = imp[valid_mask]
            tgt_v = tgt[valid_mask].detach()
            mse = F.mse_loss(imp_v, tgt_v)
            cos = 1.0 - F.cosine_similarity(imp_v, tgt_v, dim=1).mean()
            total = total + mse + cos
        return total / max(len(imputed_list), 1)

    @staticmethod
    def _mixup_ce(logits, target, lam, idx):
        """MixUp 软标签交叉熵 (vanilla，无类别权重/标签平滑)。

        logits: [B, C];  target: [B] 原始标签;  lam: [B] 混合系数;
        idx: [B] 置换 (target[idx] 为第二标签)。
        """
        lp = F.log_softmax(logits, dim=1)
        nll_a = F.nll_loss(lp, target, reduction="none")
        nll_b = F.nll_loss(lp, target[idx], reduction="none")
        return (lam * nll_a + (1 - lam) * nll_b).mean()

    def _compute_loss(self, out, label, ct_mask, mri_mask, label_secondary=None, mix=None):
        if mix is not None:
            mix_lam, mix_idx = mix
            # MixUp 分支: main/n/aux 全部用全 batch 软标签 CE, rec 不受影响
            main_loss = self._mixup_ce(out["main"], label, mix_lam, mix_idx)
            if label_secondary is not None and "n" in out:
                main_loss = main_loss + self._mixup_ce(
                    out["n"], label_secondary, mix_lam, mix_idx)
            aux_loss = self._mixup_ce(out["ct_aux"], label, mix_lam, mix_idx) \
                + self._mixup_ce(out["mri_aux"], label, mix_lam, mix_idx)
        else:
            main_loss = self.criterion_t(out["main"], label)
            if label_secondary is not None and "n" in out:
                # 多任务: 第二任务 (如病理N) 的 CE, 权重与主任务相同 (N 不加类别权重)
                main_loss = main_loss + self.criterion_n(out["n"], label_secondary)

            aux_loss = torch.tensor(0.0, device=label.device)
            if ct_mask.any():
                aux_loss = aux_loss + self.criterion_t(out["ct_aux"][ct_mask], label[ct_mask])
            if mri_mask.any():
                aux_loss = aux_loss + self.criterion_t(out["mri_aux"][mri_mask], label[mri_mask])

        rec_loss = torch.tensor(0.0, device=label.device)
        both = ct_mask & mri_mask
        if out.get("imputed_mri") is not None:
            rec_loss = rec_loss + self._reconstruction_loss(
                out["imputed_mri"], out["mri_pooled"], both
            )
        if out.get("imputed_ct") is not None:
            rec_loss = rec_loss + self._reconstruction_loss(
                out["imputed_ct"], out["ct_pooled"], both
            )

        total = main_loss \
            + self.loss_weights.get("aux", 0.3) * aux_loss \
            + self.loss_weights.get("rec", 0.5) * rec_loss
        return total, main_loss.detach(), aux_loss.detach(), rec_loss.detach()

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        sum_main = sum_aux = sum_rec = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            ct = batch["ct"].to(self.cfg.device)
            mri = batch["mri"].to(self.cfg.device)
            clinical = batch["clinical"].to(self.cfg.device)
            label = batch["label"].to(self.cfg.device)
            label_sec = batch.get("label_secondary")
            if label_sec is not None:
                label_sec = label_sec.to(self.cfg.device)
            B = label.size(0)

            ct_avail = batch.get("ct_available")
            mri_avail = batch.get("mri_available")
            ct_mask, mri_mask = self._sample_modality_masks(B, self.cfg.device,
                                                            ct_avail, mri_avail)
            mri_seq_avail = batch.get("mri_seq_available")
            if mri_seq_avail is not None:
                mri_seq_avail = mri_seq_avail.to(self.cfg.device).bool()

            # round6 B3: MixUp —— 以概率 mixup 对 batch 内样本做线性插值
            # (ct/mri/clinical 与 label 同步混合; 模态 mask 不混合)
            mix_lam = mix_idx = None
            if self.mixup > 0 and torch.rand(1).item() < self.mixup:
                mix_lam = torch.distributions.Beta(
                    self.mixup_alpha, self.mixup_alpha).sample((B,)).to(ct.device)
                mix_idx = torch.randperm(B, device=ct.device)
                lam_v = mix_lam.view(-1, *([1] * (ct.dim() - 1)))
                ct = lam_v * ct + (1 - lam_v) * ct[mix_idx]
                lam_v = mix_lam.view(-1, *([1] * (mri.dim() - 1)))
                mri = lam_v * mri + (1 - lam_v) * mri[mix_idx]
                lam_v = mix_lam.view(-1, 1)
                clinical = lam_v * clinical + (1 - lam_v) * clinical[mix_idx]

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask,
                                 mri_seq_mask=mri_seq_avail)
                loss, l_main, l_aux, l_rec = self._compute_loss(
                    out, label, ct_mask, mri_mask, label_sec,
                    mix=(mix_lam, mix_idx) if mix_lam is not None else None)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * B
            sum_main += l_main.item() * B
            sum_aux += l_aux.item() * B
            sum_rec += l_rec.item() * B
            correct += (out["main"].argmax(1) == label).sum().item()
            total += B
            pbar.set_postfix(loss=f"{total_loss / total:.4f}",
                             acc=f"{correct / total:.4f}")
        pbar.close()

        n = max(total, 1)
        avg_loss = total_loss / n
        acc = correct / n
        comp = {"main": sum_main / n, "aux": sum_aux / n, "rec": sum_rec / n}
        self._log(
            f"[Train] Epoch {epoch:03d}  loss={avg_loss:.4f}  acc={acc:.4f}  "
            f"main={comp['main']:.4f}  aux={comp['aux']:.4f}  rec={comp['rec']:.4f}"
        )
        return avg_loss, acc, comp

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss, correct, correct_n, total = 0.0, 0, 0, 0
        all_preds, all_labels = [], []
        all_n_preds, all_n_labels = [], []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [val]  ",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            ct = batch["ct"].to(self.cfg.device)
            mri = batch["mri"].to(self.cfg.device)
            clinical = batch["clinical"].to(self.cfg.device)
            label = batch["label"].to(self.cfg.device)
            label_sec = batch.get("label_secondary")
            if label_sec is not None:
                label_sec = label_sec.to(self.cfg.device)
            B = label.size(0)

            ct_mask = batch.get("ct_available")
            mri_mask = batch.get("mri_available")
            if ct_mask is not None:
                ct_mask = ct_mask.to(self.cfg.device).bool()
            if mri_mask is not None:
                mri_mask = mri_mask.to(self.cfg.device).bool()
            mri_seq_avail = batch.get("mri_seq_available")
            if mri_seq_avail is not None:
                mri_seq_avail = mri_seq_avail.to(self.cfg.device).bool()

            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask,
                                 mri_seq_mask=mri_seq_avail)
                loss = self.criterion(out["main"], label)

            preds = out["main"].argmax(1)
            total_loss += loss.item() * B
            correct += (preds == label).sum().item()
            if label_sec is not None and "n" in out:
                correct_n += (out["n"].argmax(1) == label_sec).sum().item()
                all_n_preds.extend(out["n"].argmax(1).cpu().tolist())
                all_n_labels.extend(label_sec.cpu().tolist())
            total += B
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
            pbar.set_postfix(loss=f"{total_loss / total:.4f}",
                             acc=f"{correct / total:.4f}")
        pbar.close()

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        n_acc = correct_n / max(total, 1) if correct_n > 0 or "n" in out else float("nan")
        self._log(f"[Val]   Epoch {epoch:03d}  loss={avg_loss:.4f}  acc={acc:.4f}  "
                  f"n_acc={n_acc:.4f}")
        return avg_loss, acc, all_preds, all_labels, n_acc, all_n_preds, all_n_labels

    @torch.no_grad()
    def predict_proba(self, loader=None, best_key="t"):
        """在 loader 上返回 T/N 的 softmax 概率与真值 (numpy)。用于阈值校准。

        best_key: "t" 用 T-best 权重 (best_state_dict), "n" 用 N-best 权重
        (best_state_dict_n)。校准 N 头时传 best_key="n", 使 N 结果与
        best_model_n.pth 一致, 不受 T 的 best epoch 绑架。
        """
        self.model.eval()
        loader = loader if loader is not None else self.val_loader
        best_state = None
        if best_key == "t" and getattr(self, "best_state_dict", None) is not None:
            best_state = self.best_state_dict
        elif best_key == "n" and getattr(self, "best_state_dict_n", None) is not None:
            best_state = self.best_state_dict_n
        saved_state = None
        if best_state is not None:
            saved_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(best_state)
        try:
            probs_t, labels_t, probs_n, labels_n = [], [], [], []
            for batch in loader:
                ct = batch["ct"].to(self.cfg.device)
                mri = batch["mri"].to(self.cfg.device)
                clinical = batch["clinical"].to(self.cfg.device)
                label = batch["label"].to(self.cfg.device)
                label_sec = batch.get("label_secondary")
                ct_mask = batch.get("ct_available")
                mri_mask = batch.get("mri_available")
                if ct_mask is not None:
                    ct_mask = ct_mask.to(self.cfg.device).bool()
                if mri_mask is not None:
                    mri_mask = mri_mask.to(self.cfg.device).bool()
                mri_seq_avail = batch.get("mri_seq_available")
                if mri_seq_avail is not None:
                    mri_seq_avail = mri_seq_avail.to(self.cfg.device).bool()
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    out = self.model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask,
                                     mri_seq_mask=mri_seq_avail)
                probs_t.append(F.softmax(out["main"], dim=1).float().cpu().numpy())
                labels_t.append(label.cpu().numpy())
                if label_sec is not None and "n" in out:
                    probs_n.append(F.softmax(out["n"], dim=1).float().cpu().numpy())
                    labels_n.append(label_sec.cpu().numpy())
            res = {
                "probs_t": np.concatenate(probs_t, axis=0),
                "labels_t": np.concatenate(labels_t),
            }
            if probs_n:
                res["probs_n"] = np.concatenate(probs_n, axis=0)
                res["labels_n"] = np.concatenate(labels_n)
        finally:
            if saved_state is not None:
                self.model.load_state_dict(saved_state)
        return res

    def fit(self, tag=""):
        self._log(f"Start training: {self.cfg.epochs} epochs, device={self.cfg.device}")
        last_preds, last_labels = [], []
        suffix = f"_{tag}" if tag else ""
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc, comp = self.train_one_epoch(epoch)
            val_loss, val_acc, preds, labels, n_acc, n_preds, n_labels = self.validate(epoch)
            self.scheduler.step()

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_main"].append(comp["main"])
            self.history["train_aux"].append(comp["aux"])
            self.history["train_rec"].append(comp["rec"])
            last_preds, last_labels = preds, labels

            # ── N 头独立 best: 按验证集 N 的 macro-F1 单独跟踪与保存,
            #    避免 N 被 T 的 best epoch 绑架 ──
            if n_preds:
                n_macro_f1 = compute_metrics(n_preds, n_labels, self.num_classes)["macro_f1"]
                self.history["val_n_macro_f1"].append(n_macro_f1)
                if n_macro_f1 > self.best_macro_f1_n:
                    self.best_macro_f1_n = n_macro_f1
                    self.best_epoch_n = epoch
                    path_n = os.path.join(self.cfg.save_dir, f"best_model_n{suffix}.pth")
                    torch.save(self.model.state_dict(), path_n)
                    self.best_state_dict_n = {k: v.detach().cpu().clone()
                                             for k, v in self.model.state_dict().items()}
                    self._log(f"  -> Saved best N model (n_macro_f1={n_macro_f1:.4f}, "
                              f"n_acc={n_acc:.4f}) @ epoch {epoch}")
            else:
                self.history["val_n_macro_f1"].append(float("nan"))

            if val_acc > self.best_acc:
                self.best_acc = val_acc
            val_macro_f1 = compute_metrics(preds, labels, self.num_classes)["macro_f1"]
            self.history["val_macro_f1"].append(val_macro_f1)
            if val_macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = val_macro_f1
                self.best_epoch = epoch
                path = os.path.join(self.cfg.save_dir, f"best_model{suffix}.pth")
                torch.save(self.model.state_dict(), path)
                self.best_state_dict = {k: v.detach().cpu().clone()
                                        for k, v in self.model.state_dict().items()}
                self._log(f"  -> Saved best model (macro_f1={val_macro_f1:.4f}, acc={val_acc:.4f})")

        n_str = (f"  N-best macro_f1={self.best_macro_f1_n:.4f} @ epoch {self.best_epoch_n}"
                 if self.best_epoch_n > 0 else "")
        self._log(f"Training done. Best val acc={self.best_acc:.4f}  "
                  f"macro_f1={self.best_macro_f1:.4f} @ epoch {self.best_epoch}{n_str}")

        # ── 可视化输出 ──
        curve_path = plot_curves(self.history, self.plot_dir, tag=tag)
        self._log(f"训练曲线已保存: {curve_path}")

        result = {"best_acc": self.best_acc, "best_macro_f1": self.best_macro_f1,
                  "best_epoch": self.best_epoch,
                  "best_macro_f1_n": self.best_macro_f1_n,
                  "best_epoch_n": self.best_epoch_n}
        if last_labels:
            metrics = compute_metrics(last_preds, last_labels, self.num_classes)
            self._log(f"最终验证指标: acc={metrics['accuracy']:.4f}  "
                      f"macro_f1={metrics['macro_f1']:.4f}  "
                      f"per_class_recall={metrics['per_class_recall']}")
            class_names = getattr(self.cfg, "class_names", None) \
                or [f"class{i}" for i in range(self.num_classes)]
            cm_path = plot_confusion_matrix(
                metrics["confusion_matrix"], class_names, self.plot_dir, tag=tag
            )
            self._log(f"混淆矩阵已保存: {cm_path}")
            result.update(metrics)
        return result
