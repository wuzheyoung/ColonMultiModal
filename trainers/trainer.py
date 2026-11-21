import os
import random
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:  # tqdm 仅用于进度条, 缺失时降级为空壳迭代器 (换解释器不再整体崩溃)
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class _NullTqdm:
        """无 tqdm 时的空壳: 原样迭代, set_postfix/close 静默。"""
        def __init__(self, iterable=None, *args, **kwargs):
            self._it = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self._it)

        def __len__(self):
            return len(self._it)

        def set_postfix(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def tqdm(iterable=None, *args, **kwargs):
        return _NullTqdm(iterable, *args, **kwargs)

from models import FocalLoss
from utils import compute_metrics, plot_curves, plot_confusion_matrix


class Trainer:
    """训练与验证循环（支持模态 dropout + 辅助 loss + 重建 loss）。"""

    def __init__(self, model, train_loader, val_loader, cfg, logger=None,
                 class_weights=None, label_smoothing=0.0,
                 use_focal_loss=False, focal_gamma=2.0,
                 freeze_backbone_layers=None, class_weights_n=None, lr_n=None,
                 ema_decay=None):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        # round8 熔断: 单 batch (含数据获取) 超过阈值秒即中止训练 + dump, 防偶发卡死空耗
        self.batch_timeout_s = float(os.environ.get("BATCH_TIMEOUT_S", "120"))
        # round8 硬看门狗: CUDA 硬卡死时主线程阻塞在 kernel 调用, 普通检查不生效,
        # daemon 线程监测「批完成时间戳」, 超时直接 os._exit(1) (任务快速失败, 由心跳捕获)
        self._last_progress = time.time()
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

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
        # round9: N 次任务类别权重 (N 短板优先) —— 同样 tensor 化 + 上设备
        if class_weights_n is not None and not isinstance(class_weights_n, torch.Tensor):
            class_weights_n = torch.tensor(class_weights_n, dtype=torch.float32)
        if class_weights_n is not None:
            class_weights_n = class_weights_n.to(cfg.device)
        if use_focal_loss:
            self.criterion_t = FocalLoss(
                gamma=focal_gamma, alpha=class_weights, label_smoothing=label_smoothing
            )
        else:
            self.criterion_t = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            )
        self.criterion_n = nn.CrossEntropyLoss(
            weight=class_weights_n, label_smoothing=label_smoothing)
        self.criterion = self.criterion_t  # T 主任务别名
        self.class_weights_t = class_weights  # round8: MixUp 分支也用 T 类别权重
        self.class_weights_n = class_weights_n  # MixUp 分支 N 头同口径
        # round6 B3: MixUp 数据增强 (仅作用于分类 loss: main/n/aux; rec 不受影响)
        self.mixup = getattr(cfg, "mixup", 0.0)
        self.mixup_alpha = getattr(cfg, "mixup_alpha", 0.2)
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        # round9 R4: T/N 分离学习率——N 头 (classifier_n.*) 用独立 lr_n,
        # 解决 R3 发现的「lr3e-4 偏好 N / lr1e-4 偏好 T」双任务冲突
        if lr_n is not None and abs(lr_n - cfg.lr) > 1e-9:
            named_trainable = [(n, p) for n, p in model.named_parameters()
                               if p.requires_grad]
            n_params = [p for n, p in named_trainable if "classifier_n" in n]
            other_params = [p for n, p in named_trainable if "classifier_n" not in n]
            self.optimizer = AdamW([
                {"params": other_params, "lr": cfg.lr,
                 "weight_decay": cfg.weight_decay},
                {"params": n_params, "lr": lr_n,
                 "weight_decay": cfg.weight_decay},
            ])
        else:
            self.optimizer = AdamW(trainable_params_list, lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
        self.snapshot_k = getattr(cfg, "snapshot_k", None)  # round9 S9-⑤: 每 k epoch 存快照
        self.warm_restarts = getattr(cfg, "warm_restarts", False)  # round9: 纯 warm restarts 不存快照
        # round9 S9-1: T/N 异方差不确定性加权 (Kendall 2018)
        # 必须在 scheduler 创建前 add_param_group, 否则 log_var 不进入调度参数组, lr 永不衰减
        self.uncertainty_weight = getattr(cfg, "uncertainty_weight", False)
        self._uncert_params = []
        if self.uncertainty_weight:
            self.log_var_t = nn.Parameter(torch.zeros(1))
            self.log_var_n = nn.Parameter(torch.zeros(1))
            self._uncert_params = [self.log_var_t, self.log_var_n]
            # 权重参数走主 lr, 不加 weight_decay (log 方差不应被 L2 正则)
            self.optimizer.add_param_group({"params": self._uncert_params,
                                            "lr": cfg.lr, "weight_decay": 0.0})
        if (self.snapshot_k and self.snapshot_k > 0) or self.warm_restarts:
            # Snapshot Ensemble: cosine warm restarts → 周期快照权重多样
            # warm_restarts=True: 只改 lr 调度不存快照 (省磁盘, round9 验证用)
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            T0 = self.snapshot_k if (self.snapshot_k and self.snapshot_k > 0) else 10
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=T0, T_mult=1)
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)

        # 混合精度：仅在 CUDA 上启用
        self.use_amp = getattr(cfg, "use_amp", False) and str(cfg.device).startswith("cuda")
        # round8 A-fix: init_scale=2^10 (默认 growth_interval=2000 步才翻倍,
        # 60 epoch ≈ 2160 步, 即整个训练 scale 基本保持 1024, 等效限幅),
        # 防止 fp16 反向中间梯度被 65536 级放大溢出
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp,
                                           init_scale=2.0 ** 10)

        self.modality_dropout = getattr(cfg, "modality_dropout",
                                        {"ct_only": 0.15, "mri_only": 0.15, "both": 0.7})
        self.loss_weights = getattr(cfg, "loss_weights", {"aux": 0.3, "rec": 0.5})
        # round9 C2: 一致性正则 —— 同一卷二次增强, MRI/CT 分支特征增强不变性
        # (小数据防过拟合; EMA 参数级已证伪, 特征级互补)
        self.consist_modal = getattr(cfg, "consist_modal", None)
        self.consist_weight = getattr(cfg, "consist_weight", 0.0)
        self.consist_pool_idx = int(getattr(cfg, "consist_pool_idx", -1))
        # round9 ④: PCGrad 梯度解耦 —— T/N 两任务梯度冲突时投影到彼此法平面,
        # 缓解"帮 T 伤 N / 帮 N 伤 T"的共享表征冲突 (Yu et al. 2020)
        self.pcgrad = getattr(cfg, "pcgrad", False)

        self.best_acc = 0.0
        self.best_macro_f1 = 0.0
        self.best_epoch = 0
        self.best_state_dict = None        # best 权重 (CPU), 供校准/评估使用
        # N 头独立 best: 按验证集 N 的 macro-F1 选择, 避免被 T 的 best epoch 绑架
        self.best_macro_f1_n = 0.0
        self.best_epoch_n = 0
        self.best_state_dict_n = None
        # round9 S3: EMA 影子权重 (参数指数滑动平均, 小数据抗噪)
        self.ema_decay = ema_decay
        self.ema_shadow = None
        self.ema_backup = None
        if ema_decay is not None and 0.0 < ema_decay < 1.0:
            self.ema_shadow = {n: p.detach().clone()
                               for n, p in model.named_parameters() if p.requires_grad}
            self._log(f"EMA 影子权重已启用: decay={ema_decay} "
                      f"({len(self.ema_shadow)} 个可训练参数)")
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

    def _watchdog_loop(self):
        # round8 硬看门狗: CUDA 硬卡死时主线程阻塞在 kernel 调用, 普通检查不生效,
        # daemon 线程监测「批完成时间戳」, 超时直接 os._exit(1) (任务快速失败, 由心跳捕获)
        while True:
            time.sleep(10)
            idle = time.time() - self._last_progress
            if idle > self.batch_timeout_s:
                print(f"[WATCHDOG] 训练无进展 {idle:.0f}s > "
                      f"{self.batch_timeout_s:.0f}s, 疑似 GPU 硬卡死, 强制退出", flush=True)
                os._exit(1)

    def _ema_update(self):
        """每个 batch 后更新 EMA 影子权重。"""
        if self.ema_shadow is None:
            return
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.ema_shadow:
                    self.ema_shadow[n].mul_(self.ema_decay).add_(
                        p.detach(), alpha=1.0 - self.ema_decay)

    def _ema_apply(self):
        """把影子权重临时写入模型 (验证/保存 best 用 EMA 权重)。"""
        if self.ema_shadow is None:
            return
        self.ema_backup = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.ema_shadow:
                    self.ema_backup[n] = p.detach().clone()
                    p.data.copy_(self.ema_shadow[n])

    def _ema_restore(self):
        """恢复训练权重。"""
        if self.ema_shadow is None or not self.ema_backup:
            return
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.ema_backup:
                    p.data.copy_(self.ema_backup[n])
        self.ema_backup = {}

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
        """每个样本独立采样模态可用性；真实缺失的模态强制 False。

        modality_dropout 三键语义（归一化后使用, 默认 0.15/0.15/0.7 与旧行为逐样本等价）:
          mri_only: 只丢 MRI；ct_only: 只丢 CT；both: 两模态都保留。
        旧实现只读前两键, --md-both 是静默空操作; 现在 both 真正参与采样。
        """
        p_ct = float(self.modality_dropout.get("ct_only", 0.15))
        p_mri = float(self.modality_dropout.get("mri_only", 0.15))
        p_both = float(self.modality_dropout.get("both", 1.0 - p_ct - p_mri))
        p_ct, p_mri, p_both = max(p_ct, 0.0), max(p_mri, 0.0), max(p_both, 0.0)
        tot = p_ct + p_mri + p_both
        if tot <= 0:
            p_mri, p_ct = 0.0, 0.0
        else:
            p_mri, p_ct = p_mri / tot, p_ct / tot
        ct_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        mri_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            r = random.random()
            if r < p_mri:            # 只丢 MRI
                mri_mask[i] = False
            elif r < p_mri + p_ct:   # 只丢 CT
                ct_mask[i] = False
            # 其余 → both (两模态都保留)
        if ct_avail is not None:
            ct_mask = ct_mask & ct_avail.to(device).bool()
        if mri_avail is not None:
            mri_mask = mri_mask & mri_avail.to(device).bool()
        return ct_mask, mri_mask

    def _apply_single_modal(self, ct_mask, mri_mask, B, device):
        """round9-CAM 后: --single-modal 消融的验证阶段强制掩码。

        训练阶段由 modality_dropout p=1.0 (见 _sample_modality_masks) 每样本强制丢弃;
        但 validate/predict_proba 原只按 batch 真实可用性 → 若不强制, 验证会悄悄
        回到双模态, 消融不干净。此处与训练对齐:
          single_modal="mri" → ct_mask 全 False (丢弃 CT, 留 MRI)
          single_modal="ct"  → mri_mask 全 False (丢弃 MRI, 留 CT)
        None 掩码按全 True 处理; 未设置 single_modal 时原样返回。
        """
        sm = getattr(self.cfg, "single_modal", None)
        if sm is None:
            return ct_mask, mri_mask
        ones = torch.ones(B, dtype=torch.bool, device=device)
        zeros = torch.zeros(B, dtype=torch.bool, device=device)
        if sm == "mri":
            return zeros, (mri_mask if mri_mask is not None else ones)
        if sm == "ct":
            return (ct_mask if ct_mask is not None else ones), zeros
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

    def _mixup_ce(self, logits, target, lam, idx, weight=None):
        """MixUp 软标签交叉熵（round8: 可选类别权重; weight=None 时与旧 vanilla 行为一致）。

        logits: [B, C];  target: [B] 原始标签;  lam: [B] 混合系数;
        idx: [B] 置换 (target[idx] 为第二标签);  weight: [C] 类别权重(可选)。
        """
        lp = F.log_softmax(logits, dim=1)
        nll_a = F.nll_loss(lp, target, reduction="none")
        nll_b = F.nll_loss(lp, target[idx], reduction="none")
        if weight is not None:
            # 加权软标签 CE: 两个标签各自取自己的类别权重 (旧版第二标签误用 weight[target])
            per = lam * weight[target] * nll_a + (1 - lam) * weight[target[idx]] * nll_b
        else:
            per = lam * nll_a + (1 - lam) * nll_b
        return per.mean()

    def _compute_loss(self, out, label, ct_mask, mri_mask, label_secondary=None, mix=None):
        if mix is not None:
            mix_lam, mix_idx = mix
            # MixUp 分支: main/n/aux 全部用全 batch 软标签 CE, rec 不受影响
            # round8: main/aux 带上 T 类别权重 (与 criterion_t 一致)
            main_loss = self._mixup_ce(out["main"], label, mix_lam, mix_idx,
                                       weight=self.class_weights_t)
            if label_secondary is not None and "n" in out:
                main_loss = main_loss + self._mixup_ce(
                    out["n"], label_secondary, mix_lam, mix_idx,
                    weight=self.class_weights_n)
            aux_loss = self._mixup_ce(out["ct_aux"], label, mix_lam, mix_idx,
                                      weight=self.class_weights_t) \
                + self._mixup_ce(out["mri_aux"], label, mix_lam, mix_idx,
                                 weight=self.class_weights_t)
        else:
            loss_t = self.criterion_t(out["main"], label)
            if label_secondary is not None and "n" in out:
                # 多任务: 第二任务 (如病理N) 的 CE, 权重与主任务相同 (N 不加类别权重)
                loss_n = self.criterion_n(out["n"], label_secondary)
            else:
                loss_n = None

            if self.uncertainty_weight and loss_n is not None:
                # round9 S9-1: Kendall 异方差不确定性加权 (标准式, 2026-11-20 修正):
                # L = 0.5·exp(-log_var_t)·L_t + 0.5·exp(-log_var_n)·L_n
                #     + 0.5·log_var_t + 0.5·log_var_n
                dev = loss_t.device
                w_t = 0.5 * torch.exp(-self.log_var_t.to(dev))
                w_n = 0.5 * torch.exp(-self.log_var_n.to(dev))
                main_loss = w_t * loss_t + w_n * loss_n \
                    + 0.5 * self.log_var_t.to(dev) + 0.5 * self.log_var_n.to(dev)
            else:
                main_loss = loss_t + (loss_n if loss_n is not None else 0.0)

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
        # PCGrad: 返回图连接的分离 T/N 损失 (main_loss 的两组成部分), 供梯度投影
        l_t = l_n = None
        if mix is None and loss_n is not None:
            if self.uncertainty_weight:
                dev = loss_t.device
                l_t = torch.exp(-self.log_var_t.to(dev)) * loss_t \
                    + 0.5 * self.log_var_t.to(dev)
                l_n = torch.exp(-self.log_var_n.to(dev)) * loss_n \
                    + 0.5 * self.log_var_n.to(dev)
            else:
                l_t = loss_t
                l_n = loss_n
        return total, main_loss.detach(), aux_loss.detach(), rec_loss.detach(), l_t, l_n

    def _apply_pcgrad(self, l_t, l_n):
        """PCGrad (Yu et al. 2020): 对每个共享参数, 若 T/N 两任务梯度内积<0,
        则把各自投影到对方的法平面 (g_i ← g_i − (g_i·g_j / |g_j|²) g_j), 再合并。
        已在 unscale 后调用 → 用 torch.autograd.grad 重算 T/N 各自梯度 (图由
        retain_graph=self.pcgrad 保留), 从 param.grad 中剥离 T/N 部分再投影合并。
        """
        params = [p for p in self.model.parameters()
                  if p.grad is not None and p.requires_grad]
        if not params:
            return
        g_t = torch.autograd.grad(l_t, params, retain_graph=True,
                                  allow_unused=True)
        g_n = torch.autograd.grad(l_n, params, retain_graph=False,
                                  allow_unused=True)
        eps = 1e-8
        for p, gt, gn in zip(params, g_t, g_n):
            g_rest = p.grad  # 总梯度 (unscale 后)
            gt = gt if gt is not None else torch.zeros_like(g_rest)
            gn = gn if gn is not None else torch.zeros_like(g_rest)
            gt = gt.float(); gn = gn.float(); g_rest = g_rest.float()
            g_rest = g_rest - gt - gn  # 剥离 T/N 后的其余梯度 (aux/rec/consist 等)
            dot = (gt * gn).sum()
            if dot < 0:
                gt = gt - (dot / (gn.norm().square() + eps)) * gn
                gn = gn - (dot / (gt.norm().square() + eps)) * gt
            p.grad = (g_rest + gt + gn).to(p.grad.dtype)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        sum_main = sum_aux = sum_rec = 0.0
        grad_norm_sum, grad_norm_cnt = 0.0, 0
        cls_correct = [0] * self.num_classes
        cls_total = [0] * self.num_classes

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]",
                    leave=False, dynamic_ncols=True)
        _last_t = time.time()
        for batch in pbar:
            t_fetch = time.time() - _last_t  # 含 dataloader 数据获取等待 (卡死热点)
            _b_start = time.time()
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

            # round9 F1: 病灶分支输入 (loader 提供 ct_lesion + lesion_available)
            ct_lesion = batch.get("ct_lesion")
            if ct_lesion is not None:
                ct_lesion = ct_lesion.to(self.cfg.device)
            lesion_avail = batch.get("lesion_available")
            if lesion_avail is not None:
                lesion_avail = lesion_avail.to(self.cfg.device).bool()
            # round9 MRI-t ROI: 结肠质心裁剪卷 + 可用标志
            mri_lesion = batch.get("mri_lesion")
            if mri_lesion is not None:
                mri_lesion = mri_lesion.to(self.cfg.device)
            mri_lesion_avail = batch.get("mri_lesion_available")
            if mri_lesion_avail is not None:
                mri_lesion_avail = mri_lesion_avail.to(self.cfg.device).bool()

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
                if ct_lesion is not None:
                    lam_v = mix_lam.view(-1, *([1] * (ct_lesion.dim() - 1)))
                    ct_lesion = lam_v * ct_lesion + (1 - lam_v) * ct_lesion[mix_idx]
                    lesion_avail = lesion_avail | lesion_avail[mix_idx] if lesion_avail is not None else None
                if mri_lesion is not None:
                    lam_v = mix_lam.view(-1, *([1] * (mri_lesion.dim() - 1)))
                    mri_lesion = lam_v * mri_lesion + (1 - lam_v) * mri_lesion[mix_idx]
                    mri_lesion_avail = mri_lesion_avail | mri_lesion_avail[mix_idx] if mri_lesion_avail is not None else None

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask,
                                 mri_seq_mask=mri_seq_avail, ct_lesion=ct_lesion,
                                 lesion_available=lesion_avail,
                                 mri_lesion=mri_lesion, mri_lesion_available=mri_lesion_avail)
                loss, l_main, l_aux, l_rec, l_t, l_n = self._compute_loss(
                    out, label, ct_mask, mri_mask, label_sec,
                    mix=(mix_lam, mix_idx) if mix_lam is not None else None)

            # round9 C2: 一致性正则 —— 用第二增强副本再前向, 比较真实池化特征
            # 双向 stop-grad (f1 vs f2.detach + f2 vs f1.detach), 只对可用样本
            if (self.consist_weight > 0 and self.consist_modal
                    and batch.get(f"{self.consist_modal}_second") is not None
                    and self.consist_modal in ("mri", "ct")):
                second = batch[f"{self.consist_modal}_second"].to(self.cfg.device)
                if mix_lam is not None:
                    lam_v = mix_lam.view(-1, *([1] * (second.dim() - 1)))
                    second = lam_v * second + (1 - lam_v) * second[mix_idx]
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    out2 = self.model(
                        second if self.consist_modal == "ct" else ct,
                        second if self.consist_modal == "mri" else mri,
                        clinical, ct_mask=ct_mask, mri_mask=mri_mask,
                        mri_seq_mask=mri_seq_avail, ct_lesion=ct_lesion,
                        lesion_available=lesion_avail)
                key = "ct_pooled" if self.consist_modal == "ct" else "mri_pooled"
                f1 = out[key][self.consist_pool_idx]
                f2 = out2[key][self.consist_pool_idx]
                m = mri_mask if self.consist_modal == "mri" else ct_mask
                if m.any():
                    consist_loss = torch.nn.functional.mse_loss(
                        f1[m], f2[m].detach()) + torch.nn.functional.mse_loss(
                        f2[m], f1[m].detach())
                    loss = loss + 0.5 * self.consist_weight * consist_loss
                    self._consist_loss_sum = getattr(self, "_consist_loss_sum", 0.0) + \
                        consist_loss.item() * B
                    self._consist_loss_cnt = getattr(self, "_consist_loss_cnt", 0) + int(m.sum())
            self.scaler.scale(loss).backward(retain_graph=self.pcgrad)
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            # round9 ④: PCGrad 梯度解耦 —— T/N 分离 backward + 冲突梯度投影
            # (在 unscale 后真实梯度上做, 再合并回 param.grad)
            if self.pcgrad and l_t is not None and l_n is not None \
                    and l_t.requires_grad and l_n.requires_grad:
                self._apply_pcgrad(l_t, l_n)
            # round8: 梯度 L2 范数 (unscale 后为真实梯度, 诊断「loss 不降」)
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.detach().norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            grad_norm_sum += grad_norm
            grad_norm_cnt += 1
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._ema_update()   # round9 S3: 每 batch 更新 EMA 影子权重

            total_loss += loss.item() * B
            sum_main += l_main.item() * B
            sum_aux += l_aux.item() * B
            sum_rec += l_rec.item() * B
            preds = out["main"].argmax(1)
            correct += (preds == label).sum().item()
            for c in range(self.num_classes):
                cls_total[c] += int((label == c).sum().item())
                cls_correct[c] += int(((preds == label) & (label == c)).sum().item())
            total += B
            pbar.set_postfix(loss=f"{total_loss / total:.4f}",
                             acc=f"{correct / total:.4f}")
            _b_dt = time.time() - _b_start
            if max(t_fetch, _b_dt) > self.batch_timeout_s:
                self._log(f"!!! 熔断: batch 获取 {t_fetch:.1f}s / 处理 {_b_dt:.1f}s "
                          f"> {self.batch_timeout_s:.0f}s, 中止训练")
                raise RuntimeError(
                    f"batch timeout: fetch={t_fetch:.1f}s process={_b_dt:.1f}s")
            self._last_progress = time.time()
            _last_t = time.time()
        pbar.close()

        n = max(total, 1)
        avg_loss = total_loss / n
        acc = correct / n
        comp = {"main": sum_main / n, "aux": sum_aux / n, "rec": sum_rec / n}
        lr_now = self.optimizer.param_groups[0]["lr"]
        gn = grad_norm_sum / max(grad_norm_cnt, 1)
        cls_acc = {c: (cls_correct[c] / cls_total[c] if cls_total[c] else float("nan"))
                   for c in range(self.num_classes)}
        self._log(
            f"[Train] Epoch {epoch:03d}  loss={avg_loss:.4f}  acc={acc:.4f}  "
            f"main={comp['main']:.4f}  aux={comp['aux']:.4f}  rec={comp['rec']:.4f}  "
            f"lr={lr_now:.2e}  grad_norm={gn:.3f}  cls_acc={cls_acc}"
            + (f"  consist={self._consist_loss_sum / max(self._consist_loss_cnt, 1):.4f}"
               if getattr(self, "_consist_loss_sum", 0.0) > 0 else "")
        )
        return avg_loss, acc, comp

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss, correct, correct_n, total = 0.0, 0, 0, 0
        all_preds, all_labels = [], []
        all_n_preds, all_n_labels = [], []
        has_n_out = False  # 空 val / 单任务时避免引用未定义的 out

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
            ct_mask, mri_mask = self._apply_single_modal(ct_mask, mri_mask, B,
                                                         self.cfg.device)
            mri_seq_avail = batch.get("mri_seq_available")
            if mri_seq_avail is not None:
                mri_seq_avail = mri_seq_avail.to(self.cfg.device).bool()

            # round9 F1: 病灶分支输入 (validate 阶段透传)
            ct_lesion = batch.get("ct_lesion")
            if ct_lesion is not None:
                ct_lesion = ct_lesion.to(self.cfg.device)
            lesion_avail = batch.get("lesion_available")
            if lesion_avail is not None:
                lesion_avail = lesion_avail.to(self.cfg.device).bool()
            mri_lesion = batch.get("mri_lesion")
            if mri_lesion is not None:
                mri_lesion = mri_lesion.to(self.cfg.device)
            mri_lesion_avail = batch.get("mri_lesion_available")
            if mri_lesion_avail is not None:
                mri_lesion_avail = mri_lesion_avail.to(self.cfg.device).bool()

            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                out = self.model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask,
                                 mri_seq_mask=mri_seq_avail, ct_lesion=ct_lesion,
                                 lesion_available=lesion_avail,
                                 mri_lesion=mri_lesion, mri_lesion_available=mri_lesion_avail)
                loss = self.criterion(out["main"], label)

            preds = out["main"].argmax(1)
            total_loss += loss.item() * B
            correct += (preds == label).sum().item()
            if label_sec is not None and "n" in out:
                has_n_out = True
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
        n_acc = correct_n / max(total, 1) if has_n_out else float("nan")
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
                ct_mask, mri_mask = self._apply_single_modal(
                    ct_mask, mri_mask, label.size(0), self.cfg.device)
                mri_seq_avail = batch.get("mri_seq_available")
                if mri_seq_avail is not None:
                    mri_seq_avail = mri_seq_avail.to(self.cfg.device).bool()
                # round9 F1: 病灶分支输入 (predict_proba 阶段透传)
                ct_lesion = batch.get("ct_lesion")
                if ct_lesion is not None:
                    ct_lesion = ct_lesion.to(self.cfg.device)
                lesion_avail = batch.get("lesion_available")
                if lesion_avail is not None:
                    lesion_avail = lesion_avail.to(self.cfg.device).bool()
                mri_lesion = batch.get("mri_lesion")
                if mri_lesion is not None:
                    mri_lesion = mri_lesion.to(self.cfg.device)
                mri_lesion_avail = batch.get("mri_lesion_available")
                if mri_lesion_avail is not None:
                    mri_lesion_avail = mri_lesion_avail.to(self.cfg.device).bool()
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    out = self.model(ct, mri, clinical, ct_mask=ct_mask, mri_mask=mri_mask,
                                     mri_seq_mask=mri_seq_avail, ct_lesion=ct_lesion,
                                     lesion_available=lesion_avail,
                                     mri_lesion=mri_lesion, mri_lesion_available=mri_lesion_avail)
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
            self._ema_apply()   # round9 S3: 验证 + 保存 best 用 EMA 权重
            val_loss, val_acc, preds, labels, n_acc, n_preds, n_labels = self.validate(epoch)
            self.scheduler.step()

            # round9 S9-⑤: 每 snapshot_k epoch 保存一次快照权重 (cosine restarts 周期末)
            # warm_restarts=True 时不存快照 (省磁盘)
            if self.snapshot_k and self.snapshot_k > 0 and epoch % self.snapshot_k == 0:
                snap_path = os.path.join(self.cfg.save_dir, f"snapshot{suffix}_ep{epoch}.pth")
                torch.save(self.model.state_dict(), snap_path)
                self._log(f"  -> Saved snapshot @ epoch {epoch}: {snap_path}")

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

            # round9 S3: best 已按 EMA 权重保存, 恢复训练权重进入下一 epoch
            self._ema_restore()

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
