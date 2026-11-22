# -*- coding: utf-8 -*-
"""
round9-A ROI 内 DL 特征 (病灶级) 训练脚本。
- 动机: 病灶级 ROI 手工特征 (roi_experiment) N 仍≈随机 (val F1 0.40) → 手工统计拿不到
        局部浸润信息 → 试深度特征: mask 中心裁剪 64³ 喂 3D ResNet。
- 输入: CT 确定性卷 [1,96,96,96] (cache_hos 缓存, 与 R5 相同预处理) + Task010 病灶 mask
        + 临床7: 从 data/hos_pairs_raw.csv 取原始值, **按折内训练集统计**做
        生理范围过滤 (age 0–120, ca199/cea ≥0, albumin 10–70, prealbumin 50–600, hb 30–220
        → 越界转 NaN/missing)
        → log1p(ca199/cea) → 1%–99% 截断 → 中位数补缺 → z-score → z±5, 并追加 7 个 missing 指示位
        (全队列 z-score 已弃用; --no-clinical 时临床位全 0; 不改 data/*.csv)
- mask 几何: 默认 --mask-geometry aligned —— mask 与缓存卷走同一条几何链
        (CT: Spacing(1mm)→窗→CropForeground(CT)→Resize nearest; MRI: 锚定 1mm 参考网格→
        同一 bbox→Resize nearest), 与影像同轴序同视野; legacy 保留旧行为(原始网格直接 nearest
        Resize 到 96³, 轴序 xyz vs 卷 zyx, 且无物理对齐) 仅供复现历史数字
- ring: aligned 模式下不再对 96³ 薄二值环做 nearest 缩放; 在 1mm 空间用
        distance_transform_edt 得到连续物理距离场(mm, 仅 bbox+带宽内精确) → 三线性缩放到 96³ → 阈值取环
        (环宽 = --ring-mm, 默认 2mm; 可选 5/10mm; 兼容旧 --ring-iter); legacy 仍用原网格二值膨胀
- SDF: --sdf-channel 为每个图像通道追加一路连续物理 SDF ([-1,1], ±15mm), 不二值化;
        在 96³ 上按 1mm bbox/96 的物理间距做 EDT (避免全幅 1mm 重采样再 EDT 的性能爆炸)
- 裁剪: --roi-mask centroid(默认, mask 质心)/bbox(mask bbox 中心)/masked(bbox 中心且只保留 mask 内体素)
        /binary_only(bbox 中心, 输入改为二值 mask 本身, 用于检验"形态捷径")
        → 64³ 立方体 (越界 clamp + 零填充); 无 mask 样本跳过
- 模型: medicalnet10 预训练 (72/72, in_channels=1) + GeM + FC(head_width) + T/N 双头
        (T/N 各自按本任务标签逆频加权, lr 分离 1e-4/3e-4, AMP, CosineAnnealing 60ep)
- 选模/校准 (方案 A): 每轮按 val ROC-AUC 保存 T/N best ckpt (单类折退回 macro-F1@0.5);
        kfold 阈值用 Leave-One-Fold-Out (其余折 OOF 搜阈值 → 当前折应用); 单 split 无 OOF 不校准
- 指标: best ckpt 的 ROC-AUC / PR-AUC / macro-F1@0.5、末 5 个 epoch 概率平均,
        并把逐折外折概率落盘 results/roidl_oof/ (best: {tag}_k{fold}_s{seed}.csv;
        末 5 轮均值: 同名 + _last5.csv), 供后续融合与 pN 亚组分析
- 产物命名: ckpt / 逐折报告 / kfold summary / OOF / 日志 均带 --out-tag; ckpt/报告/summary
        另带 _s{seed} (扩多 seed 不覆盖); OOF 已含 seed
- 成本埋点: 每 fold 记录 epoch_sec 列表/均值、data_wait_ratio (取批等待占比)、
        peak_gpu_mem_gb (无 CUDA 时为 None, 不报错), 写入单折报告与日志
- kfold: --kfold-csv data/master_5fold_356.csv 优先; 否则用固定全量表 data/{tag}_full.csv
        (不再拼接 _s{seed} 训练/验证表, 换 seed 不会找不到文件);
        不在主表内的患者显式跳过并计数打印 (n_not_in_master), 不静默纳入
- 单折标定: --kfold 5 --only-fold k 只训练该折, 其余折留空 (LOFO 阈值为 None)
- SDF 捷径: --sdf-only 把图像通道置零、仅保留 SDF (隐含 --sdf-channel; 仅 aligned;
        不可与 --roi-mask binary_only 同用)
- 报告: logs/eval_report_roidl_hos_full_s42.json + 归档 results/round9/run_NN_roidl_hos_full_s42
用法:
  python tools/train_roi_dl.py                    # 全量 seed42 60ep
  python tools/train_roi_dl.py --kfold 5 --kfold-csv data/master_5fold_356.csv
  python tools/train_roi_dl.py --kfold 5 --kfold-csv data/master_5fold_356.csv \
      --only-fold 0 --out-tag p0_cal --mode mri_t --roi-mask bbox --no-clinical
  python tools/train_roi_dl.py --epochs 1 --smoke # CPU 冒烟
"""
import argparse, json, os, sys, time, shutil, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loader.loader import (  # noqa: E402
    _DetDiskCache, _det_cache_key_mri, _mri_bbox, _mri_reference_grid, _zscore_nonzero)
from utils import setup_logger, calibrate_binary_threshold  # noqa: E402

CLINICAL_COLS = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
LOG_COLS = ("ca199", "cea")   # 右偏 → log1p
# D1 方案 C: 代码侧生理范围过滤 (不改 data/*.csv); 越界 → NaN → 点亮 missing 指示位
# age 0–120; ca199/cea 仅下界 0 (肿瘤标志物不可能为负, 同时避免 log1p 产生 NaN)
PHYS_RANGES = {"age": (0.0, 120.0), "ca199": (0.0, np.inf), "cea": (0.0, np.inf),
               "albumin": (10.0, 70.0), "prealbumin": (50.0, 600.0), "hb": (30.0, 220.0)}
SDF_CLIP_MM = 15.0  # 连续 SDF 截断/归一化尺度: 覆盖直肠系膜 5–15mm 范围
SPATIAL = (96, 96, 96)
ROI = 64          # 裁剪边长
HALF = ROI // 2
MASK_DIR = os.path.join(ROOT, "data", "masks_task010")
MASK_DIR_MR = {seq: os.path.join(ROOT, "data", f"masks_totalseg_mr_{seq}") for seq in ("t", "c", "d")}
_SEQ_ORDER = ("t", "c", "d")
RAW_CLIN_CSV = os.path.join(ROOT, "data", "hos_pairs_raw.csv")
OOF_DIR = os.path.join(ROOT, "results", "roidl_oof")


class SelfWrittenNet3D(nn.Module):
    """自写轻量 3D CNN (64³ ROI) —— 不依赖 MedicalNet 预训练权重。
    设计依据: agent 调研 (AECN slice-attention arXiv:2103.03931 ★★★ + 轻量 3D CNN 主题1):
    - stem conv3x3x3 stride1 → 3 个残差 stage (通道 32→64→128, stride2) → 特征 4³
    - 轴向 slice attention: 每轴向切片可学习重要性权重 (滤除 ROI 内正常组织/系膜)
    - GeM 池化 (p=3) → MLP → 双头 (T/N)
    参数 ~1.1M (medicalnet10 的 1/10), 从零训练小样本风险相对低。

    gam=True 时启用 H-CNN-ViT 降级版 (Local GAM, arXiv:2511.13869):
    - 双路径: CNN 局部路径 (主干) + 全局路径 (深度轴全局均值)
    - Local GAM 门控: g(z)=Wz+b → sigmoid → softmax 归一 → y = α_glob·z_glob + α_cnn·z_cnn
    (仅 ROI 分支内, 不引 ViT/不扩 3 序列分支, 规避 3 倍算力与小样本过拟合风险)
    """

    def __init__(self, in_channels=1, num_clinical=7, head_width=256, head_dropout=0.5, gam=False, fpn=False, aecn=False, clin_late=False, aux_pt=False, tn_hier=False):
        super().__init__()
        ch = [32, 64, 128]
        self.gam = gam
        self.fpn = fpn
        self.aecn = aecn
        self.clin_late = clin_late
        self.aux_pt = aux_pt
        self.tn_hier = tn_hier
        self.num_clinical = num_clinical
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, ch[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(ch[0]), nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        for i, c in enumerate(ch):
            s2 = 2 if i < len(ch) - 1 else 2  # 64³→32³→16³→8³
            self.stages.append(self._make_stage(ch[i - 1] if i else ch[0], c, s2))
        self.slice_att = nn.Sequential(  # 轴向 slice attention (深度轴 8 切片, 输入=每切片空间均值)
            nn.Linear(8, 8), nn.Sigmoid(),
        )
        if aecn:  # B2 AECN 扩展: x/y/z 三轴向 slice attention 相乘 (滤除各轴正常组织层)
            self.slice_att_x = nn.Sequential(nn.Linear(8, 8), nn.Sigmoid())
            self.slice_att_y = nn.Sequential(nn.Linear(8, 8), nn.Sigmoid())
        if gam:  # H-CNN-ViT 降级: 双路径 + Local GAM 门控
            self.glob_path = nn.Sequential(
                nn.Conv3d(ch[-1], ch[-1], kernel_size=1, bias=False),
                nn.BatchNorm3d(ch[-1]), nn.ReLU(inplace=True))
            self.gate = nn.Linear(ch[-1] * 2, ch[-1])  # g(z) = Wz+b (输入=cat(cnn,glob) 均值)
        if fpn:  # 3D FPN: 各 stage 输出 1x1 投影对齐通道, 上采样到 8³ concat (多尺度局部+全局)
            self.fpn_convs = nn.ModuleList(
                nn.Conv3d(c, ch[-1], kernel_size=1, bias=False) for c in ch)
        self.gem_p = 3.0
        fpn_in = len(ch) * ch[-1] if fpn else ch[-1]
        fuse_in = fpn_in if clin_late else fpn_in + num_clinical
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, head_width), nn.ReLU(inplace=True), nn.Dropout(head_dropout))
        head_in = head_width + (num_clinical if clin_late else 0)
        self.head_t = nn.Linear(head_in, 2)
        # tn_hier (B6): N 头以 T 头输出为条件 (层次建模, P(N|T) 先验注入)
        self.head_n = nn.Linear(head_in + (2 if tn_hier else 0), 2)
        if aux_pt:  # B5: pT 四类辅助头 (pT1/2/3/4 → 0/1/2/3), 与 T/N 联合训练
            self.head_pt = nn.Linear(head_in, 4)
        self.out_channels = ch[-1]

    def _make_stage(self, cin, cout, stride):
        return nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    def _gem(self, x):
        xf = x.float()
        return xf.clamp(min=1e-6).pow(self.gem_p).mean(dim=(2, 3, 4), keepdim=True).pow(1.0 / self.gem_p).to(x.dtype)

    def forward(self, ct, clin):
        """ct: [B,C,64,64,64]  clin: [B,7] → (logit_t, logit_n)"""
        h = self.stem(ct)
        feats = []
        for s in self.stages:
            h = s(h)
            feats.append(h)
        if self.fpn:  # 3D FPN: 各 stage 投影对齐 → 上采样到最小分辨率 (8³) → concat
            base = feats[-1].shape[2:]  # (8,8,8)
            up = []
            for i, f in enumerate(feats):
                f = self.fpn_convs[i](f)
                if f.shape[2:] != base:
                    f = F.interpolate(f, size=base, mode="trilinear", align_corners=False)
                up.append(f)
            h = torch.cat(up, dim=1)  # [B, 384, 8, 8, 8]
        sa = self.slice_att(h.mean(dim=(1, 3, 4)))   # [B,8] z 轴切片空间均值 → 轴向权重
        h = h * sa.view(h.shape[0], 1, 8, 1, 1)      # 加权切片 (z 轴 8)
        if self.aecn:  # B2: x/y 轴 slice attention 相乘 (三轴向联合滤层)
            sa_x = self.slice_att_x(h.mean(dim=(1, 2, 4)))   # [B,8] 保留 dim3 (x 轴)
            h = h * sa_x.view(h.shape[0], 1, 1, 8, 1)
            sa_y = self.slice_att_y(h.mean(dim=(1, 2, 3)))   # [B,8] 保留 dim4 (y 轴)
            h = h * sa_y.view(h.shape[0], 1, 1, 1, 8)
        if self.gam:  # H-CNN-ViT Local GAM: 双路径门控
            zg = self.glob_path(h)                    # 全局路径 [B,128,8,8,8]
            zc = h                                    # CNN 局部路径 (加权后)
            gv = self.gate(torch.cat([zc.mean(dim=(2,3,4)), zg.mean(dim=(2,3,4))], dim=1))  # [B,128]
            alpha = torch.sigmoid(gv).view(gv.shape[0], gv.shape[1], 1, 1, 1)  # 逐通道互补门控
            h = alpha * zc + (1 - alpha) * zg         # 门控融合
        f = self._gem(h).flatten(1)
        if self.clin_late:  # C4: 临床特征晚融合 (只进分类头, 不进 backbone 特征)
            h2 = self.fuse(f)
            h2 = torch.cat([h2, clin], dim=1)
        else:  # 早融合 (默认)
            h2 = self.fuse(torch.cat([f, clin], dim=1))
        logit_t = self.head_t(h2)
        if self.tn_hier:  # B6: N 头条件于 T 头 (stop-grad, 避免反向耦合)
            logit_n = self.head_n(torch.cat([h2, logit_t.detach()], dim=1))
        else:
            logit_n = self.head_n(h2)
        if self.aux_pt:
            return logit_t, logit_n, self.head_pt(h2)
        return logit_t, logit_n


class ROINet(nn.Module):
    """medicalnet10 + GeM + 双头 (单卷输入, 病灶 ROI 64³)。
    self_written=True 时改用 SelfWrittenNet3D (自写, 无预训练权重)。"""

    def __init__(self, num_clinical=7, head_width=256, head_dropout=0.5,
                 freeze_prefixes=("conv1", "bn1", "relu", "maxpool", "layer1", "layer2"),
                 in_channels=1, self_written=False, gam=False, fpn=False, aecn=False, clin_late=False, aux_pt=False, tn_hier=False):
        super().__init__()
        self.aux_pt = aux_pt
        self.tn_hier = tn_hier
        self.selfnet = None
        if self_written:
            self.encoder = None
            self.selfnet = SelfWrittenNet3D(in_channels=in_channels, num_clinical=num_clinical,
                                            head_width=head_width, head_dropout=head_dropout, gam=gam, fpn=fpn, aecn=aecn, clin_late=clin_late, aux_pt=aux_pt, tn_hier=tn_hier)
            print("[ROI] 自写轻量 3D CNN (无预训练权重)" + (" + H-CNN-ViT Local GAM 门控" if gam else "")
                  + (" + 3D FPN 多尺度" if fpn else "") + (" + AECN 三轴向 slice attention" if aecn else "")
                  + (" + 临床晚融合" if clin_late else ""))
            self.num_clinical = num_clinical
            self.gem = None
            self.fuse = None
            self.head_t = None
            self.head_n = None
            return
        from models.backbone import build_backbone
        enc = build_backbone("medicalnet10", in_channels=in_channels, pretrained=True,
                             pretrained_dir=os.path.join(ROOT, "pretrained"))
        if enc.pretrain_msg:
            print(f"[ROI] {enc.pretrain_msg}")
        self.encoder = enc
        self.num_clinical = num_clinical
        out_c = enc.out_channels  # 512
        self.gem = nn.AdaptiveAvgPool3d(1)  # 简用 avg pool (GeM 需重排, 此处判别力优先)
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
        print(f"[ROI] freeze params: {n_frozen}")

    def forward(self, ct, clin):
        """ct: [B,C,64,64,64] (C=1 或 mri_all 时 C=3)  clin: [B,7] → (logit_t, logit_n[, logit_pt])"""
        if self.selfnet is not None:
            return self.selfnet(ct, clin)
        feats = self.encoder(ct)[-1]                     # [B,512,2,2,2]
        f = self.gem(feats).flatten(1)                    # [B,512]
        h = self.fuse(torch.cat([f, clin], dim=1))
        logit_t = self.head_t(h)
        if self.tn_hier:
            logit_n = self.head_n(torch.cat([h, logit_t.detach()], dim=1))
        else:
            logit_n = self.head_n(h)
        if self.aux_pt:
            return logit_t, logit_n, self.head_pt(h)
        return logit_t, logit_n


RING_ITERATIONS = 2  # ring 膨胀层数 (A4: 调优 2→3/4)


_ALIGN_MEMO = {}
_SDF_MEMO = {}
_MASK_SHAPE = {}  # mask_path → 对齐前 1mm 网格形状 (供 SDF 物理间距)


def _spath(v):
    """CSV 单元格 → 安全路径字符串 (NaN/None → '')。"""
    return v.strip() if isinstance(v, str) else ""


def mask_path_of(pid, mask_kind):
    """mask_kind='task010' → data/masks_task010/{pid}.nii.gz (病灶)
    mask_kind='mr_t/c/d' → data/masks_totalseg_mr_{seq}/{pid}/colon.nii.gz (MRI 结肠)
    mask_kind='mr_t_ring' 等 → 同路径, 后续做二值膨胀取环"""
    ring = mask_kind.endswith("_ring")
    if mask_kind.startswith("mr_") and not ring:
        return os.path.join(MASK_DIR_MR[mask_kind[3:]], f"{pid}", "colon.nii.gz")
    if ring:
        return os.path.join(MASK_DIR_MR[mask_kind[3:-5]], f"{pid}", "colon.nii.gz")
    return os.path.join(MASK_DIR, f"{pid}.nii.gz")


def _seed_everything(seed, deterministic=False):
    """统一播种 random / numpy / torch(CPU+CUDA)。--aug 走 np.random, 缺 np 播种会漂移。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _sdf_from_mask96(m96, shape_1mm, spatial=SPATIAL):
    """在 96³ 二值 mask 上用**物理采样间距**算连续有符号 SDF, 归一化到 [-1,1]。

    不在整幅 1mm 网格上做两次 EDT: 1mm 盆腔网格可达 >300³, 实测 >60s/例且内存翻倍。
    这里用 shape_1mm / 96 得到每轴 mm 间距, 在 96³ 上 EDT (毫秒级), 距离仍以 mm 计;
    SDF 只用于网络输入, 精度受 96³ 网格限制, 但保留连续值与物理尺度。
    """
    from scipy.ndimage import distance_transform_edt
    m = np.asarray(m96) > 0
    if not m.any():
        return None
    vox = [float(shape_1mm[i]) / float(spatial[i]) for i in range(3)]
    sdf = (distance_transform_edt(~m, sampling=vox) - distance_transform_edt(m, sampling=vox)).astype(np.float32)
    return (np.clip(sdf, -SDF_CLIP_MM, SDF_CLIP_MM) / SDF_CLIP_MM).astype(np.float32)


def _apply_ring(m):
    """[legacy 专用] 原网格二值膨胀 RING_ITERATIONS 层后取环。
    aligned 路径改用 _ring_sdf: 96³ 上薄二值环最近邻缩放会整层消失, 不能再对二值环做 nearest。"""
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(m > 0, iterations=RING_ITERATIONS)
    return np.logical_and(dilated, ~(m > 0)).astype(np.float32)


def _ring_sdf(mask_1mm, ring_mm=None, spatial=SPATIAL):
    """1mm 各向同性空间 → 连续物理距离场(mm) → 三线性缩放 96³ → 阈值取环。

    为什么不在 96³ 上膨胀: 盆腔 bbox 常 150–250mm, 降到 96³ 后体素≈2mm,
    2mm 环宽不足 1 体素, 对薄二值环做 nearest 缩放会整层丢失 (或被放大成锯齿)。
    距离场是连续场, 三线性缩放后阈值仍以 mm 为单位, 环宽可保持物理意义。
    """
    from scipy.ndimage import distance_transform_edt
    m = np.asarray(mask_1mm) > 0
    if not m.any():
        return None
    w = float(RING_ITERATIONS if ring_mm is None else ring_mm)  # 1mm 空间: 1 层≈1mm
    dist = distance_transform_edt(~m).astype(np.float32)  # 0=mask 内, >0=到肠壁的 mm 距离
    if tuple(dist.shape) != tuple(spatial):
        t = torch.from_numpy(dist)[None, None]
        dist = F.interpolate(t, size=tuple(spatial), mode="trilinear",
                             align_corners=False)[0, 0].numpy()
    # 本体在 96³ 的占位 (nearest 缩放): 环必须严格在本体之外
    m96 = torch.from_numpy(m.astype(np.float32))[None, None]
    if tuple(m96.shape[2:]) != tuple(spatial):
        m96 = F.interpolate(m96, size=tuple(spatial), mode="nearest")
    body96 = m96[0, 0].numpy() > 0.5
    ring = ((dist > 0) & (dist <= w) & ~body96).astype(np.float32)
    if float(ring.sum()) <= 0:  # 极窄环被采样抹掉 → 至少保留紧邻本体的一层
        pos = dist[(dist > 0) & ~body96]
        if pos.size == 0:
            return None
        ring = ((dist > 0) & (dist <= float(pos.min()) + 1e-6) & ~body96).astype(np.float32)
    return ring


def load_mask_resized(pid, mask_kind="task010"):
    """[legacy] 原始网格直接 Resize 到 96³ 最近邻。
    注意: 轴序为原始 nii 的 (x,y,z), 而缓存卷为 (z,y,x), 且未做 1mm/bbox 对齐 —— 仅用于复现历史数字。"""
    import nibabel as nib
    from monai.transforms import Resize
    p = mask_path_of(pid, mask_kind)
    if not os.path.exists(p):
        return None
    try:
        m = nib.load(p).get_fdata()
    except Exception:
        return None
    if float((m > 0).sum()) <= 0:
        return None
    if mask_kind.endswith("_ring"):
        m = _apply_ring(m)
        if float(m.sum()) <= 0:
            return None
    m3 = Resize(spatial_size=SPATIAL, mode="nearest")(m[None].astype(np.float32))
    return np.asarray(m3[0])  # 统一返回 numpy, 与 aligned 路径一致


def _mask_ct_aligned(ct_path, mask_path, ring=False):
    """mask → 与 CT 缓存卷完全相同的 96³ 网格 (同轴序/同 bbox/同 1mm 网格)。
    变换链与 loader._ct_deterministic 对齐: Spacing(1mm) → HU 窗 → CropForeground(以 CT 为准)。
    ring=True: 在裁剪后的 1mm 网格上算距离场 (带内精确) → 三线性 96³ → 阈值取环。"""
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
                                  ScaleIntensityRanged, CropForegroundd, Resized)
    keys = ["ct", "mask"]
    tf = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="ct", a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="ct"),
    ] + ([] if ring else [Resized(keys=["mask"], spatial_size=SPATIAL, mode="nearest")]))
    d = tf({"ct": ct_path, "mask": mask_path})
    _MASK_SHAPE[mask_path] = tuple(int(x) for x in np.asarray(d["mask"]).shape[1:])
    if ring:
        return _ring_sdf(np.asarray(d["mask"][0]))
    return np.ascontiguousarray(np.asarray(d["mask"][0], dtype=np.float32))


def _mask_mri_aligned(seq_paths, mask_path, ring=False):
    """mask → 与 MRI 缓存卷通道相同的 96³ 网格。
    与 loader._mri_seqs_deterministic 对齐: 首个可用序列(优先 T2)为锚建 1mm 参考网格 →
    mask 最近邻重采样 → 用锚序列 nonzero bbox(pad=5) 裁剪 → Resize nearest。
    ring=True: bbox 内 1mm 距离场 (带内精确) → 三线性 96³ → 阈值取环。"""
    import SimpleITK as sitk
    from monai.transforms import Resize
    anchor = next((p for p in seq_paths if isinstance(p, str) and p and os.path.exists(p)), None)
    if anchor is None:
        return None
    ref = _mri_reference_grid(sitk.ReadImage(anchor))
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ref)
    rf.SetInterpolator(sitk.sitkNearestNeighbor)
    rf.SetDefaultPixelValue(0.0)
    m = sitk.GetArrayFromImage(rf.Execute(sitk.ReadImage(mask_path)))  # [D,H,W] 与卷同轴序
    a = sitk.GetArrayFromImage(ref)
    # 与 loader._mri_seqs_deterministic 完全同口径: 锚序列先 zscore 再取 nonzero bbox (pad=5)
    d0, d1, h0, h1, w0, w1 = _mri_bbox(_zscore_nonzero(a) != 0)
    _MASK_SHAPE[mask_path] = (int(d1 - d0), int(h1 - h0), int(w1 - w0))
    if ring:
        return _ring_sdf(m[d0:d1, h0:h1, w0:w1])
    m3 = Resize(spatial_size=SPATIAL, mode="nearest")(m[d0:d1, h0:h1, w0:w1][None].astype(np.float32))
    return np.ascontiguousarray(m3[0])


def load_mask_96(pid, mask_kind="task010", aligned=True, ct_path=None, seq_paths=None):
    """读 mask → 96³。aligned=True 时与影像缓存卷同网格 (默认); False 走 legacy 原始网格。"""
    if not aligned:
        return load_mask_resized(pid, mask_kind)
    p = mask_path_of(pid, mask_kind)
    if not os.path.exists(p):
        return None
    ring = mask_kind.endswith("_ring")
    key = (p, os.path.getmtime(p), "ct" if mask_kind == "task010" else tuple(seq_paths or ()),
           ("ring", RING_ITERATIONS) if ring else None)
    if key in _ALIGN_MEMO:
        return _ALIGN_MEMO[key]
    try:
        if mask_kind == "task010":
            if not ct_path or not os.path.exists(ct_path):
                return None
            m = _mask_ct_aligned(ct_path, p, ring=ring)
        else:
            m = _mask_mri_aligned(tuple(seq_paths or ()), p, ring=ring)
    except Exception as e:  # mask 与影像网格不兼容时显式告警, 不再静默
        print(f"[ROI][warn] mask 对齐失败 {p}: {e}")
        return None
    if m is None:
        return None
    if float((m > 0).sum()) <= 0:
        return None
    _ALIGN_MEMO[key] = m
    return m


def load_sdf_96(pid, mask_kind="task010", aligned=True, ct_path=None, seq_paths=None):
    """连续物理 SDF (96³, [-1,1], 截断尺度 15mm)。仅 aligned 模式支持。
    复用 load_mask_96 的 96³ 二值 mask + 1mm 网格形状, 在 96³ 上按物理采样间距做 EDT
    (避免在 >300³ 的 1mm CT 网格上重复 Spacingd+EDT, 实测 >60s/例)。"""
    if not aligned:
        return None
    base_kind = mask_kind[:-5] if mask_kind.endswith("_ring") else mask_kind  # SDF 由本体 mask 生成
    p = mask_path_of(pid, base_kind)
    if not os.path.exists(p):
        return None
    key = (p, os.path.getmtime(p), "ct" if base_kind == "task010" else tuple(seq_paths or ()))
    if key in _SDF_MEMO:
        return _SDF_MEMO[key]
    m96 = load_mask_96(pid, base_kind, aligned=True, ct_path=ct_path, seq_paths=seq_paths)
    shape = _MASK_SHAPE.get(p)
    if m96 is None or shape is None:
        return None
    s = _sdf_from_mask96(m96, shape)
    if s is None:
        return None
    _SDF_MEMO[key] = s
    return s


def _mask_center(mask, mode):
    idx = np.argwhere(mask > 0)
    if len(idx) == 0:
        return None
    if mode == "centroid":
        return idx.mean(axis=0).astype(int)
    lo, hi = idx.min(0), idx.max(0)
    return ((lo + hi) // 2).astype(int)


def crop_roi(ct, mask, mode="centroid"):
    """ct [96,96,96] + mask [96,96,96] → 64³ 裁剪 (clamp+pad0)。
    mode='centroid' 以 mask 质心为中心 (不屏蔽图像, 历史口径);
    mode='bbox' 以 mask bbox 中心为中心;
    mode='masked' 同 bbox 中心, 且把 mask 外体素置 0 (只保留 mask 内组织, ring 实验必需);
    mode='binary_only' 同 bbox 中心, 但输入换成 mask 本身 (非零区=1.0), 用于检验形态捷径。"""
    cen = _mask_center(mask, "centroid" if mode == "centroid" else "bbox")
    if cen is None:
        return None
    roi = np.zeros((ROI, ROI, ROI), dtype=np.float32)
    z0 = max(0, cen[0] - HALF); z1 = min(96, cen[0] + HALF)
    y0 = max(0, cen[1] - HALF); y1 = min(96, cen[1] + HALF)
    x0 = max(0, cen[2] - HALF); x1 = min(96, cen[2] + HALF)
    src = ct[z0:z1, y0:y1, x0:x1]
    dz = z1 - z0; dy = y1 - y0; dx = x1 - x0
    oz0 = max(0, HALF - cen[0]); oy0 = max(0, HALF - cen[1]); ox0 = max(0, HALF - cen[2])
    if mode == "masked":
        src = src * (mask[z0:z1, y0:y1, x0:x1] > 0)
    elif mode == "binary_only":
        src = (mask[z0:z1, y0:y1, x0:x1] > 0).astype(np.float32)
    roi[oz0:oz0 + dz, oy0:oy0 + dy, ox0:ox0 + dx] = src
    return roi


def _augment_roi(v):
    """ROI 随机增强 (v: [C,64,64,64] np.float32, 训练时应用, 与主模型 loader 增强思路对齐)。
    翻转/旋转/缩放/强度扰动 — 空间类增强共享同一随机数(对多通道卷保持通道一致)。"""
    import numpy as np
    v = np.asarray(v, dtype=np.float32)
    # 轴向翻转 (C,64,64,64 → 对 64 三轴)
    if np.random.rand() < 0.5:
        v = v[:, ::-1]
    if np.random.rand() < 0.5:
        v = v[:, :, ::-1]
    if np.random.rand() < 0.5:
        v = v[:, :, :, ::-1]
    # 90° 旋转 (xy 平面, 保持网格)
    if np.random.rand() < 0.5:
        k = np.random.randint(1, 4)
        v = np.rot90(v, k, axes=(1, 2))
    # 强度扰动: 增益 + 偏移
    v = v * float(np.random.uniform(0.9, 1.1)) + float(np.random.uniform(-0.1, 0.1))
    return np.ascontiguousarray(v, dtype=np.float32)


def _augment_roi_lr(v, n_img):
    """A2 解剖保序弱增强: 仅左右镜像 (x 轴) + 图像通道强度抖动, 禁碰 SDF 通道。
    v: [C,64,64,64]; 前 n_img 个通道为图像 (灰度), 其后为 SDF 通道。"""
    v = np.asarray(v, dtype=np.float32)
    if np.random.rand() < 0.5:
        v = v[:, :, :, ::-1]  # 左右镜像 (x 轴翻转, 解剖保序)
    if n_img > 0:
        v[:n_img] = v[:n_img] * float(np.random.uniform(0.9, 1.1)) + float(np.random.uniform(-0.1, 0.1))
    return np.ascontiguousarray(v, dtype=np.float32)


_RAW_CLIN_DF = None


def _raw_clin_index():
    """data/hos_pairs_raw.csv 的 patient_id → 原始临床值 (未标准化)。
    文件缺失时返回空表, 由 DataROI 退回读 split CSV 的同名列并告警。"""
    global _RAW_CLIN_DF
    if _RAW_CLIN_DF is None:
        if not os.path.exists(RAW_CLIN_CSV):
            print(f"[ROI][warn] 未找到 {RAW_CLIN_CSV}, 临床退回读 split CSV 列 (可能已是全队列 z-score)")
            _RAW_CLIN_DF = pd.DataFrame()
        else:
            df = pd.read_csv(RAW_CLIN_CSV, encoding="utf-8-sig")
            df["patient_id"] = df["patient_id"].astype(str).str.strip()
            _RAW_CLIN_DF = df.drop_duplicates("patient_id").set_index("patient_id")
    return _RAW_CLIN_DF


def _to_float(v):
    if v is None:
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan


class ClinPrep:
    """临床特征折内标准化 (禁止全队列统计)。

    顺序: 生理范围过滤 (age 0–120, ca199/cea ≥0, albumin 10–70, prealbumin 50–600, hb 30–220
    → 越界转 NaN) → log1p(右偏列 ca199/cea) → 1%/99% 分位截断 → 缺失补中位数 → z-score → z 截断 ±5
    → 追加 7 个 missing 指示位。统计量只能由调用方用**折内训练集**原始行 fit。
    生理过滤只作用于代码内存, 原始 data/*.csv 不变 (D1 方案 C)。
    """

    def __init__(self, raw, cols=CLINICAL_COLS, clip_q=(0.01, 0.99), add_missing=True, z_clip=5.0):
        X = self._phys_filter(np.asarray(raw, dtype=np.float64).reshape(-1, len(cols)), cols)
        self.cols = list(cols)
        self.add_missing = add_missing
        self.z_clip = z_clip
        n = len(self.cols)
        self.lo = np.zeros(n); self.hi = np.zeros(n); self.med = np.zeros(n)
        Y = X.copy()
        for j, c in enumerate(self.cols):
            col = Y[:, j]
            if c in LOG_COLS:
                col = np.log1p(np.clip(col, 0, None))
            v = col[np.isfinite(col)]
            if v.size >= 2:
                lo, hi = np.quantile(v, clip_q)
            elif v.size == 1:
                lo = hi = float(v[0])
            else:
                lo, hi = -np.inf, np.inf
            med = float(np.median(v)) if v.size else 0.0
            col = np.clip(col, lo, hi)
            col[~np.isfinite(col)] = med
            Y[:, j] = col
            self.lo[j], self.hi[j], self.med[j] = lo, hi, med
        self.mu = Y.mean(0)
        sd = Y.std(0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        self.sd = sd

    @staticmethod
    def _phys_filter(X, cols):
        """生理范围外的值视为录入错误 → NaN (随后被 missing 指示位标记 + 中位数填充)。"""
        X = X.copy()
        for j, c in enumerate(cols):
            r = PHYS_RANGES.get(c)
            if not r:
                continue
            col = X[:, j]
            bad = np.isfinite(col) & ((col < r[0]) | (col > r[1]))
            col[bad] = np.nan
            X[:, j] = col
        return X

    @property
    def dim(self):
        return len(self.cols) * (2 if self.add_missing else 1)

    def transform(self, raw):
        X = self._phys_filter(np.asarray(raw, dtype=np.float64).reshape(-1, len(self.cols)), self.cols)
        miss = ~np.isfinite(X)
        Y = X.copy()
        for j, c in enumerate(self.cols):
            col = Y[:, j]
            if c in LOG_COLS:
                col = np.log1p(np.clip(col, 0, None))
            col = np.clip(col, self.lo[j], self.hi[j])
            col[~np.isfinite(col)] = self.med[j]
            Y[:, j] = col
        Z = (Y - self.mu) / self.sd
        if self.z_clip:
            Z = np.clip(Z, -float(self.z_clip), float(self.z_clip))
        if self.add_missing:
            Z = np.concatenate([Z, miss.astype(np.float64)], axis=1)
        return Z.astype(np.float32)


class DataROI:
    """预加载卷 (CT 缓存或 MRI 序列缓存) + mask 质心裁剪, 只保留有非空 mask 的样本。

    mode='ct':   CT 卷 + Task010 病灶 mask
    mode='mri_t/c/d': MRI 序列卷 (缓存 [3,96,96,96] 取对应通道) + total_mr 同序列结肠 mask
    mode='mri_c_ct': MRI-c 结肠 ROI + CT 病灶 ROI 拼接 [2,64,64,64] (双模态 ROI, 自写网络用)
    mode='mri_tc':   MRI-t + MRI-c 结肠 ROI 拼接 [2,64,64,64] (A1, 自写网络 N 最强 0.6097)
    mode='mri_tc_ct': MRI-t + MRI-c 结肠 ROI + CT 病灶 ROI 拼接 [3,64,64,64] (CT 缺失→零卷+标志, ② n-modal ct 融合)
    """

    def __init__(self, csv_path, mode="ct", augment=False,
                 mask_geometry="aligned", roi_mask="centroid", cache_dir=None, sdf_channel=False,
                 sdf_only=False, shared_roi=False, aug_lr=False):
        self.mode = mode
        self.augment = augment
        self.aug_lr = bool(aug_lr)  # A2: 解剖保序弱增强 (仅左右镜像+图像通道强度抖动, 禁碰 SDF)
        self.mask_geometry = mask_geometry
        self.roi_mask = roi_mask
        self.sdf_channel = bool(sdf_channel or sdf_only)
        self.sdf_only = bool(sdf_only)  # 图像通道置零, 仅保留 SDF (形态捷径对照)
        self.shared_roi = bool(shared_roi)  # A1: 多序列共用 mr_t 单一 bbox 中心 + 单一 SDF (消除通道错位)
        if self.sdf_channel and mask_geometry != "aligned":
            raise ValueError("--sdf-channel/--sdf-only 仅支持 --mask-geometry aligned (legacy 网格无物理间距)")
        if self.sdf_only and roi_mask == "binary_only":
            raise ValueError("--sdf-only 不可与 --roi-mask binary_only 同用 (两者都是形态捷径, 叠加无解释)")
        self.df = pd.read_csv(csv_path)
        self.disk = _DetDiskCache(os.path.join(ROOT, cache_dir or "cache_hos"), SPATIAL)
        self._vols = []   # (roi [1,64,64,64] np or None)
        self._pids = []
        self._clin = []       # 标准化后的临床向量, 由 set_clin 覆盖
        self._clin_raw = []   # 原始值 (未标准化, NaN=缺失), 供折内 ClinPrep fit
        self._lab_t = []
        self._lab_n = []
        self._lab_pt = []
        self._skip = 0
        self._skip_reasons = {}
        self._raw_clin_missing = 0
        raw_idx = _raw_clin_index()
        seq = None
        if mode.startswith("mri_"):
            m = mode[4:]
            if m in ("all", "tc", "tc_ct", "c_ct", "t_ring", "c_ring"):
                seq = None
            else:
                seq = m
        if mode == "mri_all":
            seqs = _SEQ_ORDER  # t,c,d 三序列 ROI 拼接 [3,64,64,64]
        elif mode == "mri_tc":
            seqs = ("t", "c")  # t+c 两序列 ROI 拼接 [2,64,64,64] (样本充足验证自写网络)
        elif mode == "mri_tc_ct":
            seqs = ("t", "c")  # t+c 两序列 + CT 病灶可选第三通道 [3,64,64,64]
        elif mode == "mri_c_ct":
            seqs = None  # 特殊: CT 病灶 + MRI-c 结肠 双源
        elif mode == "mri_t_ring":
            seqs = ("t_ring",)  # ④ t 序列结肠 mask 膨胀 ring (肠周/系膜) 单通道
        elif mode == "mri_c_ring":
            seqs = ("c_ring",)
        elif mode == "mri_tc_ring":
            seqs = ("t_ring", "c_ring")  # ① ring t+c 两序列拼接 [2,64,64,64] (肠周/系膜双序列)
        elif mode == "mri_t_body":
            seqs = ("t", "t_ring")  # ② t 序列本体 + ring 双通道 [2,64,64,64] (病灶内+系膜, 同一 MRI 卷两 mask)
        elif mode == "mri_c_body":
            seqs = ("c", "c_ring")  # ② c 序列本体 + ring 双通道 [2,64,64,64]
        elif mode == "mri_all_ring":
            seqs = ("t_ring", "c_ring", "d_ring")  # ④ ring 三序列 t+c+d 拼接 [3,64,64,64]
        else:
            seqs = [seq] if seq else None
        mask_kinds = (["task010"] if mode == "ct"
                      else [f"mr_{s}" for s in seqs] + (["task010"] if mode == "mri_tc_ct" else []) if seqs
                      else ["task010", "mr_c"] if mode == "mri_c_ct"
                      else None)
        if self.shared_roi and mask_kinds:  # A1: 所有 mr_* 通道统一用 mr_t (bbox 中心/SDF 同源, 消除多序列通道错位)
            mask_kinds = ["mr_t" if k.startswith("mr_") else k for k in mask_kinds]
        # 图像通道数 (A2 弱增强强度抖动只碰图像通道, SDF 通道不碰)
        self._n_img = (1 if mode == "ct"
                       else 2 if mode == "mri_c_ct"
                       else (len(seqs) + (1 if mode == "mri_tc_ct" else 0)) if seqs else 1)
        self._ct_flag = []  # mri_tc_ct: CT 病灶通道是否真实(1.0/0.0), 拼接进临床特征
        for _, row in self.df.iterrows():
            pid = str(row.get("patient_id", "")).strip()
            ct_path = _spath(row.get("ct_path", ""))
            seq_paths = tuple(_spath(row.get(f"mri_path_{s}", "")) for s in _SEQ_ORDER)
            masks = {mk: load_mask_96(pid, mk, aligned=(self.mask_geometry == "aligned"),
                                      ct_path=ct_path, seq_paths=seq_paths) for mk in mask_kinds}
            sdfs = ({mk: load_sdf_96(pid, mk, aligned=(self.mask_geometry == "aligned"),
                                     ct_path=ct_path, seq_paths=seq_paths) for mk in mask_kinds}
                    if self.sdf_channel else {})
            if any(m is None for m in masks.values()):
                if mode == "mri_tc_ct" and all(masks.get(k) is not None for k in ("mr_t", "mr_c")):
                    pass  # ② CT 病灶 mask 缺失 → 零卷+标志, 保留样本 (仅 mr_t/mr_c 必需)
                else:
                    self._skipf("mask 缺失/空")
                    continue
            if mode == "ct":
                if not ct_path or not os.path.exists(ct_path):
                    self._skipf("无 CT")
                    continue
                vol, _ = self.disk.get(ct_path)
                if vol is None:
                    self._skipf("CT 缓存缺失")
                    continue
                src = vol[0] if vol.ndim == 4 else vol  # [96,96,96]
                roi = crop_roi(src, masks["task010"], mode=self.roi_mask)
                if roi is None:
                    self._skipf("ROI 裁剪失败")
                    continue
                roi_v = roi[None].astype(np.float32)  # [1,64,64,64]
                if self.sdf_channel:  # SDF 双通道: [原图, 连续 SDF]
                    s = sdfs.get("task010")
                    roi_s = crop_roi(s, masks["task010"], mode=self._sdf_mode()) if s is not None else None
                    if roi_s is None:
                        self._skipf("SDF 缺失")
                        continue
                    roi_v = np.concatenate([roi_v, roi_s[None].astype(np.float32)], axis=0)
                    if self.sdf_only:
                        roi_v[:1] = 0.0  # 仅 SDF: 原图通道置零
            elif mode == "mri_c_ct":  # 双源: CT 病灶 ROI + MRI-c 结肠 ROI
                if not ct_path or not os.path.exists(ct_path):
                    self._skipf("无 CT")
                    continue
                vol_ct, _ = self.disk.get(ct_path)
                if vol_ct is None:
                    self._skipf("CT 缓存缺失")
                    continue
                key = _det_cache_key_mri(*seq_paths, SPATIAL)
                vol_mr, hit = self.disk.get_vol(key)
                if vol_mr is None or not hit:
                    self._skipf("MRI 缓存缺失")
                    continue
                src_ct = vol_ct[0] if vol_ct.ndim == 4 else vol_ct  # [96,96,96]
                roi_ct = crop_roi(src_ct, masks["task010"], mode=self.roi_mask)
                roi_mr = crop_roi(vol_mr[_SEQ_ORDER.index("c")], masks["mr_c"], mode=self.roi_mask)
                if roi_ct is None or roi_mr is None:
                    self._skipf("ROI 裁剪失败")
                    continue
                roi_v = np.concatenate([roi_ct[None].astype(np.float32),
                                        roi_mr[None].astype(np.float32)], axis=0)  # [2,64,64,64]
                if self.sdf_channel:
                    s1 = sdfs.get("task010"); s2 = sdfs.get("mr_c")
                    r1 = crop_roi(s1, masks["task010"], mode=self._sdf_mode()) if s1 is not None else None
                    r2 = crop_roi(s2, masks["mr_c"], mode=self._sdf_mode()) if s2 is not None else None
                    if r1 is None or r2 is None:
                        self._skipf("SDF 缺失")
                        continue
                    roi_v = np.concatenate([roi_v, r1[None].astype(np.float32),
                                            r2[None].astype(np.float32)], axis=0)
                    if self.sdf_only:
                        roi_v[:2] = 0.0  # 仅 SDF: 两个图像通道置零
            else:  # mri_{seq} / mri_all / mri_tc / mri_tc_ct
                key = _det_cache_key_mri(*seq_paths, SPATIAL)
                vol, hit = self.disk.get_vol(key)
                if vol is None or not hit:
                    self._skipf("MRI 缓存缺失")
                    continue
                roi_vs = []
                sdf_vs = []
                ok = True
                for s, mk in zip(seqs, mask_kinds):
                    seq_base = s[:-5] if s.endswith("_ring") else s  # t_ring → t (通道同原始序列)
                    src = vol[_SEQ_ORDER.index(seq_base)]  # [96,96,96]
                    roi = crop_roi(src, masks[mk], mode=self.roi_mask)
                    if roi is None:
                        ok = False
                        break
                    roi_vs.append(roi[None].astype(np.float32))  # [1,64,64,64]
                    if self.sdf_channel:
                        sdf = sdfs.get(mk)
                        roi_s = crop_roi(sdf, masks[mk], mode=self._sdf_mode()) if sdf is not None else None
                        if roi_s is None:
                            ok = False
                            break
                        sdf_vs.append(roi_s[None].astype(np.float32))
                if not ok:
                    self._skipf("ROI 裁剪失败")
                    continue
                if mode == "mri_tc_ct":  # ② CT 病灶 ROI 可选第三通道 (零卷+标志保持样本量)
                    flag = 0.0
                    roi_ct_s = None
                    if ct_path and os.path.exists(ct_path):
                        vol_ct, _ = self.disk.get(ct_path)
                        if vol_ct is not None:
                            src_ct = vol_ct[0] if vol_ct.ndim == 4 else vol_ct
                            m_task = masks.get("task010")
                            if m_task is not None:
                                roi_ct = crop_roi(src_ct, m_task, mode=self.roi_mask)
                                if roi_ct is not None:
                                    roi_vs.append(roi_ct[None].astype(np.float32))
                                    flag = 1.0
                                    if self.sdf_channel:
                                        s_task = sdfs.get("task010")
                                        roi_ct_s = (crop_roi(s_task, m_task, mode=self._sdf_mode())
                                                    if s_task is not None else None)
                    if flag == 0.0:
                        roi_vs.append(np.zeros((1, 64, 64, 64), dtype=np.float32))
                    if self.sdf_channel:
                        sdf_vs.append(roi_ct_s[None].astype(np.float32) if roi_ct_s is not None
                                      else np.zeros((1, 64, 64, 64), dtype=np.float32))
                    self._ct_flag.append(flag)
                roi_v = np.concatenate(roi_vs + sdf_vs, axis=0)  # 图像通道在前, SDF 通道在后
                if self.sdf_only:
                    roi_v[:len(roi_vs)] = 0.0  # 仅 SDF: 所有图像通道置零 (CT 缺失的零卷也置零)
            self._vols.append(roi_v)
            self._pids.append(pid)
            raw_row = raw_idx.loc[pid] if (len(raw_idx) and pid in raw_idx.index) else None
            if raw_row is None:
                self._raw_clin_missing += 1
                raw_row = row  # 回退: 读 split CSV 同名列 (可能已是 z-score)
            self._clin_raw.append([_to_float(raw_row.get(c, np.nan)) for c in CLINICAL_COLS])
            self._clin.append(np.zeros(1, dtype=np.float32))  # 占位, 由 set_clin 覆盖
            self._lab_t.append(int(row["label_T"]) if str(row.get("label_T", "")).strip() not in ("", "nan") else -1)
            self._lab_n.append(int(row["label_N"]) if str(row.get("label_N", "")).strip() not in ("", "nan") else -1)
            # B5 pT 四类辅助: pT 原始值 1/2/3/4 → 0/1/2/3 (缺失/异常 → -1)
            pt_raw = str(row.get("pT", "")).strip()
            try:
                ptv = int(float(pt_raw))
                self._lab_pt.append(ptv - 1 if 1 <= ptv <= 4 else -1)
            except (ValueError, TypeError):
                self._lab_pt.append(-1)
        print(f"[ROI] {os.path.basename(csv_path)} mode={mode} 载入 {len(self._vols)} 样本, "
              f"跳过 {self._skip} {self._skip_reasons} | mask={self.mask_geometry} roi={self.roi_mask}"
              + (f" | 原始临床表缺 {self._raw_clin_missing} 例" if self._raw_clin_missing else ""))

    def _sdf_mode(self):
        """SDF 通道的裁剪中心: 与图像同窗 (masked/binary_only 时用 bbox, 不把场外置零)。"""
        return "bbox" if self.roi_mask in ("masked", "binary_only") else self.roi_mask

    def clin_raw_matrix(self):
        """[N,7] 原始临床值 (NaN=缺失), 用于折内 fit。"""
        return np.asarray(self._clin_raw, dtype=np.float64).reshape(-1, len(CLINICAL_COLS))

    def set_clin(self, arr, zero=False):
        """写入标准化后的临床矩阵 [N,d]; mri_tc_ct 追加 CT 病灶可用标志; zero=True 表示 --no-clinical。"""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or len(arr) != len(self._vols):
            raise ValueError(f"set_clin 维度不匹配: {arr.shape} vs N={len(self._vols)}")
        if zero:
            arr = np.zeros_like(arr)  # --no-clinical 只清临床值, CT 可用标志仍保留
        if self.mode == "mri_tc_ct":
            flag = np.asarray(self._ct_flag, dtype=np.float32).reshape(-1, 1)
            arr = np.concatenate([arr, flag], axis=1)
        self._clin = [r for r in arr]

    @property
    def clin_dim(self):
        return int(len(self._clin[0])) if self._clin else 0

    def _skipf(self, reason):
        self._skip += 1
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1

    def __len__(self):
        return len(self._vols)

    def __getitem__(self, idx):
        vol = self._vols[idx]
        if self.augment and vol is not None:
            if self.aug_lr:
                vol = _augment_roi_lr(vol, self._n_img)
            else:
                vol = _augment_roi(vol)
        return (torch.as_tensor(np.ascontiguousarray(vol), dtype=torch.float32),
                torch.tensor(self._clin[idx], dtype=torch.float32),
                torch.tensor(self._lab_t[idx], dtype=torch.long),
                torch.tensor(self._lab_n[idx], dtype=torch.long),
                torch.tensor(self._lab_pt[idx], dtype=torch.long))


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021). 两步: 扰动→恢复→更新。
    用法: base = AdamW(...); opt = SAM(base.param_groups, AdamW, rho=0.05, **adamw_kwargs)
    循环: loss.backward(); opt.first_step(zero_grad=True); 重算 loss2; loss2.backward(); opt.second_step(zero_grad=True)"""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (p.grad if not group["adaptive"] else p.grad * p).pow(2).mean().sqrt().add_(1e-12)
                p.add_(p.grad * scale / e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(self.state[p]["old_p"] - p.data)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared = torch.cat([p.grad.flatten() for group in self.param_groups for p in group["params"] if p.grad is not None])
        return shared.norm()

    def step(self, closure=None):
        raise NotImplementedError("SAM 需要显式调用 first_step / second_step")


def make_loader(ds, batch_size, shuffle, drop_last=True, smoke=False):
    from torch.utils.data import DataLoader
    if smoke:
        ds = torch.utils.data.Subset(ds, list(range(min(16, len(ds)))))
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


class _SubDS(torch.utils.data.Dataset):
    """按索引子集 (用于 kfold 折内划分)。clin 若给定则覆盖 base 的临床向量 (折内标准化结果)。"""
    def __init__(self, base, idxs, lab_t=None, lab_n=None, lab_pt=None, augment=False,
                 clin=None, pids=None, aug_lr=False):
        self.base, self.idxs = base, idxs
        self._lab_t = lab_t if lab_t is not None else []
        self._lab_n = lab_n if lab_n is not None else []
        self._lab_pt = lab_pt if lab_pt is not None else []
        self._clin = None if clin is None else np.asarray(clin, dtype=np.float32)
        self._pids = list(pids) if pids is not None else []
        self.augment = augment
        self.aug_lr = bool(aug_lr)
        self._n_img = getattr(base, "_n_img", 1)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        vol, clin, lt, ln, lpt = self.base[self.idxs[i]]
        if self.augment and vol is not None:
            if self.aug_lr:
                vol = _augment_roi_lr(vol, self._n_img)
            else:
                vol = _augment_roi(vol)
        c = clin if self._clin is None else torch.as_tensor(self._clin[i], dtype=torch.float32)
        return (torch.as_tensor(np.ascontiguousarray(vol), dtype=torch.float32),
                torch.as_tensor(c, dtype=torch.float32),
                torch.tensor(lt, dtype=torch.long),
                torch.tensor(ln, dtype=torch.long),
                torch.tensor(lpt, dtype=torch.long))


def archive(root, tag, seed):
    runs = sorted(glob.glob(os.path.join(root, "results", "round9", "run_*")))
    nxt = 1
    for r in runs:
        try:
            nxt = max(nxt, int(os.path.basename(r).split("_")[1]) + 1)
        except (IndexError, ValueError):
            pass
    dst = os.path.join(root, "results", "round9", f"run_{nxt:02d}_roidl_{tag}_s{seed}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    while True:  # 原子占位, 避免并发撞号
        try:
            os.makedirs(dst)
            break
        except FileExistsError:
            nxt += 1
            dst = os.path.join(root, "results", "round9", f"run_{nxt:02d}_roidl_{tag}_s{seed}")
    for f in glob.glob(os.path.join(root, "logs", f"eval_report_roidl_{tag}_s{seed}.*")):
        shutil.copy2(f, dst)
    for f in glob.glob(os.path.join(root, "checkpoints_hos_full", f"*roidl_{tag}_s{seed}.pth")):
        shutil.copy2(f, dst)
    for f in glob.glob(os.path.join(root, "logs", f"roidl_{tag}_s{seed}_*.log")):
        shutil.copy2(f, dst)
    print(f"[ROI] archived -> {dst}")


def _inv_weights(labels, n_cls, device="cpu"):
    """按给定标签序列的逆频率计算类别权重 (仅用传入的标签, kfold 时仅折内训练集)。"""
    arr = np.array([int(l) for l in labels if 0 <= int(l) < n_cls], dtype=np.int64)
    if arr.size == 0:
        return torch.ones(n_cls, dtype=torch.float32, device=device)
    bc = np.bincount(arr, minlength=n_cls).astype(np.float64)
    bc[bc == 0] = 1.0
    w = arr.size / (n_cls * bc)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _tn_loss(crit_t, crit_n, logit_t, logit_n, lt, ln, vt, vn):
    """T/N 双任务 CE, 逐任务剔除 label<0 的样本 (不再把 -1 直接喂给 CE)。"""
    loss = logit_t.float().sum() * 0.0
    if bool(vt.any()):
        loss = loss + crit_t(logit_t.float()[vt], lt[vt])
    if bool(vn.any()):
        loss = loss + crit_n(logit_n.float()[vn], ln[vn])
    return loss


def _bin_metrics(prob, y):
    """prob [N,2] / y [N] → macro_f1@0.5 + ROC-AUC + PR-AUC (单类时 AUC/PR 记 None)。"""
    if prob is None or y is None or len(y) == 0:
        return {"macro_f1": None, "roc_auc": None, "pr_auc": None, "n": 0, "n_pos": 0}
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    keep = y >= 0  # 未标注 (-1) 不参与任何指标
    prob, y = prob[keep], y[keep]
    if len(y) == 0:
        return {"macro_f1": None, "roc_auc": None, "pr_auc": None, "n": 0, "n_pos": 0}
    f1 = macro_f1(torch.as_tensor(prob, dtype=torch.float32), torch.as_tensor(y))
    p1 = prob[:, 1]
    if len(np.unique(y)) < 2:
        auc = pr = None
    else:
        auc = float(roc_auc_score(y, p1))
        pr = float(average_precision_score(y, p1))
    return {"macro_f1": float(f1), "roc_auc": auc, "pr_auc": pr,
            "n": int(len(y)), "n_pos": int((y == 1).sum())}


def _predict_ds(model, ds, device):
    """逐样本推理 (不打乱, 与 patient_id 顺序严格对齐), 返回 {'t'/'n': (prob[N,2], y[N])}。"""
    model.eval()
    pt, pn, lt, ln = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            ct, clin, yt, yn, _ = ds[i]
            ct = ct[None].to(device)
            clin = clin[None].to(device)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                out = model(ct, clin)
            pt.append(F.softmax(out[0].float(), 1)[0].cpu().numpy())
            pn.append(F.softmax(out[1].float(), 1)[0].cpu().numpy())
            lt.append(int(yt))
            ln.append(int(yn))
    return {"t": (np.asarray(pt), np.asarray(lt)), "n": (np.asarray(pn), np.asarray(ln))}


def _write_oof(path, pids, res):
    """逐折/逐 seed 外折概率落盘 (patient_id, label, prob), 供融合与 pN 亚组分析。
    res['t']/res['n'] 任一为 None 时 (--n-only/--t-only) 只写可用任务列。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = {"patient_id": pids}
    keep = np.ones(len(pids), dtype=bool)
    if res.get("t") is not None:
        _, yt = res["t"]
        keep &= np.asarray(yt) >= 0
    if res.get("n") is not None:
        _, yn = res["n"]
        keep &= np.asarray(yn) >= 0
    idx = np.where(keep)[0]
    cols["patient_id"] = [pids[i] for i in idx]
    if res.get("t") is not None:
        pt, yt = res["t"]
        cols["label_T"] = yt[idx]
        cols["prob_T"] = pt[idx, 1]
    if res.get("n") is not None:
        pn, yn = res["n"]
        cols["label_N"] = yn[idx]
        cols["prob_N"] = pn[idx, 1]
    pd.DataFrame(cols).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _threshold_metrics(prob, y, thr):
    """固定阈值下的 macro-F1 / 阳性召回 (用于 LOFO 外折评估)。"""
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    k = y >= 0
    p, y = prob[k, 1], y[k]
    if len(y) == 0:
        return {"threshold": float(thr), "macro_f1": None, "recall": None, "n": 0, "n_pos": 0}
    pred = (p >= float(thr)).astype(int)
    return {"threshold": float(thr),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y, pred, average="binary", zero_division=0)),
            "n": int(len(y)), "n_pos": int(y.sum())}


def lofo_calibrate(fold_probs):
    """Leave-One-Fold-Out 阈值: 第 i 折的阈值**只用其余折 OOF 概率**搜索, 再应用到第 i 折。

    fold_probs: list[(prob[N,2], y[N]) | None], 长度为折数; 返回同长度 list[dict|None]。
    保证当前折概率不参与自己的阈值选择 (零额外训练成本)。
    """
    out = []
    for i, cur in enumerate(fold_probs):
        if cur is None:
            out.append(None)
            continue
        others = [(j, x) for j, x in enumerate(fold_probs) if j != i and x is not None]
        if not others:
            out.append(None)
            continue
        po = np.concatenate([x[0] for _, x in others])
        yo = np.concatenate([x[1] for _, x in others])
        k = yo >= 0
        po, yo = po[k], yo[k]
        if len(np.unique(yo)) < 2:
            out.append(None)
            continue
        cal = calibrate_binary_threshold(po, yo)
        thr = (cal.get("best", cal) if isinstance(cal, dict) else {}) .get("threshold", 0.5)
        m = _threshold_metrics(cur[0], cur[1], thr)
        m["threshold_source_folds"] = [j for j, _ in others]
        m["threshold_source_n"] = int(len(yo))
        out.append(m)
    return out


def run_one(ds_tr, ds_va, args, device, logger, fold_label):
    """训练+验证一折/一 seed。
    返回 (best_t, best_n, calT, calN, extra): extra 含无阈值指标 / 末5轮均值 / OOF 路径。"""
    _seed_everything(args.seed)
    use_t = not getattr(args, "n_only", False)
    use_n = not getattr(args, "t_only", False)
    tr = make_loader(ds_tr, args.batch_size, True, smoke=args.smoke)
    va = make_loader(ds_va, args.batch_size, False, drop_last=False, smoke=args.smoke)
    logger.info(f"[ROI] {fold_label}: train={len(ds_tr)} val={len(ds_va)} epochs={args.epochs} device={device} "
                f"roi={ROI}^3 lr={args.lr} lr_n={args.lr_n} use_T={use_t} use_N={use_n}")

    in_ch = 3 if args.mode in ("mri_all", "mri_tc_ct", "mri_all_ring") else (2 if args.mode in ("mri_c_ct", "mri_tc", "mri_tc_ring", "mri_t_body", "mri_c_body") else 1)
    if getattr(args, "sdf_channel", False):  # 每个图像通道追加一路连续 SDF
        in_ch *= 2
    model = ROINet(num_clinical=getattr(args, "num_clinical", len(CLINICAL_COLS)),
                   head_width=args.head_width, in_channels=in_ch, head_dropout=args.head_dropout,
                   self_written=args.self_written, gam=args.gam, fpn=args.fpn, aecn=args.aecn, clin_late=args.clin_late,
                   aux_pt=args.aux_pt, tn_hier=args.tn_hier).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[ROI] trainable {n_params/1e6:.1f}M")

    if args.smoke:
        w_t = torch.ones(2, dtype=torch.float32, device=device)
        w_n = torch.ones(2, dtype=torch.float32, device=device)
    else:
        # T/N 各自按本任务标签逆频加权 (旧版 N 误用 T 权重, 已修正)
        w_t = _inv_weights(ds_tr._lab_t, 2, device)
        w_n = _inv_weights(ds_tr._lab_n, 2, device)
    logger.info(f"[ROI] class weights T={w_t.tolist()} N={w_n.tolist()}")

    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "head_n" not in n], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "head_n" in n], "lr": args.lr_n},
    ], weight_decay=5e-4)
    if args.sam:  # ⑥ A1: Sharpness-Aware Minimization (外层包装 AdamW, 禁用 AMP 简化)
        opt = SAM(opt.param_groups, torch.optim.AdamW, rho=args.sam_rho, lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs - args.warmup, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"), init_scale=2.0 ** 10)
    crit_t = nn.CrossEntropyLoss(weight=w_t)
    crit_n = nn.CrossEntropyLoss(weight=w_n)
    crit_pt = None
    if args.aux_pt:  # B5: pT 四类辅助损失 (逆频率权重)
        w_pt = torch.ones(4, dtype=torch.float32, device=device) if args.smoke else _inv_weights(ds_tr._lab_pt, 4, device)
        crit_pt = nn.CrossEntropyLoss(weight=w_pt)

    # 方案 A: 选模依据 = val ROC-AUC (单类时退回 macro-F1); F1@0.5 仍记录但不作为选择依据
    best_t = {"score": -1.0, "epoch": -1, "auc": None, "f1": 0.0}
    best_n = {"score": -1.0, "epoch": -1, "auc": None, "f1": 0.0}
    run_id = fold_label if fold_label.startswith("s") else f"{fold_label}_s{args.seed}"
    ckpt_t = os.path.join(ROOT, "checkpoints_hos_full", f"best_model_roidl_{args.out_tag}_{run_id}.pth")
    ckpt_n = os.path.join(ROOT, "checkpoints_hos_full", f"best_model_n_roidl_{args.out_tag}_{run_id}.pth")
    hist_t, hist_n = [], []  # 末 5 轮 val 概率 (概率平均即预测集成, 不需要权重平均)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epoch_secs, data_secs = [], []
    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        t_data = 0.0
        model.train()
        tot, cnt = 0.0, 0
        it = iter(tr)
        while True:
            t0 = time.time()
            try:
                ct, clin, lt, ln, lpt = next(it)
            except StopIteration:
                break
            t_data += time.time() - t0  # 数据加载/增强等待 (含 CPU→GPU 前的取批)
            ct, clin, lt, ln, lpt = (x.to(device) for x in (ct, clin, lt, ln, lpt))
            vt = (lt >= 0) & use_t
            vn = (ln >= 0) & use_n
            if not (vt.any() or vn.any()):
                continue
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                if args.aux_pt:
                    logit_t, logit_n, logit_pt = model(ct, clin)
                    loss = _tn_loss(crit_t, crit_n, logit_t, logit_n, lt, ln, vt, vn)
                    vpt = (lpt >= 0) & vt
                    if vpt.any():
                        loss = loss + crit_pt(logit_pt.float()[vpt], lpt[vpt])
                else:
                    logit_t, logit_n = model(ct, clin)
                    loss = _tn_loss(crit_t, crit_n, logit_t, logit_n, lt, ln, vt, vn)
            opt.zero_grad()
            if args.sam:  # ⑥ SAM 两步 (fp32 直连: 禁 AMP——SAM 需两次 backward, 与 GradScaler unscale 状态冲突)
                loss.backward()
                opt.first_step(zero_grad=True)
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    if args.aux_pt:
                        logit2_t, logit2_n, logit2_pt = model(ct.float(), clin.float())
                        loss2 = _tn_loss(crit_t, crit_n, logit2_t, logit2_n, lt, ln, vt, vn)
                        vpt2 = (lpt >= 0) & vt
                        if vpt2.any():
                            loss2 = loss2 + crit_pt(logit2_pt.float()[vpt2], lpt[vpt2])
                    else:
                        logit2_t, logit2_n = model(ct.float(), clin.float())
                        loss2 = _tn_loss(crit_t, crit_n, logit2_t, logit2_n, lt, ln, vt, vn)
                loss2.backward()
                opt.second_step(zero_grad=True)
            else:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            n_ok = int((vt & vn).sum().item()) or int(vt.sum().item()) or int(vn.sum().item()) or 1
            tot += loss.item() * n_ok
            cnt += n_ok
        if args.warmup > 0 and ep <= args.warmup:  # ⑦ A3: 线性 warmup (仅前 warmup 轮)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * ep / args.warmup
        else:
            sched.step()
        model.eval()
        pt, pn, yt, yn = [], [], [], []
        with torch.no_grad():
            for ct, clin, lt, ln, lpt in va:
                ct, clin = (x.to(device) for x in (ct, clin))
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    if args.aux_pt:
                        logit_t, logit_n, _ = model(ct, clin)
                    else:
                        logit_t, logit_n = model(ct, clin)
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
        mt = macro_f1(pt, yt) if (len(yt) and use_t) else 0.0
        mn = macro_f1(pn, yn) if (len(yn) and use_n) else 0.0
        # 方案 A: 以无阈值 AUC 选模; 单类 fold 无法算 AUC 时退回 F1@0.5
        met_t = _bin_metrics(pt.detach().cpu().numpy(), yt.detach().cpu().numpy()) if len(yt) else {}
        met_n = _bin_metrics(pn.detach().cpu().numpy(), yn.detach().cpu().numpy()) if len(yn) else {}
        at, an = met_t.get("roc_auc"), met_n.get("roc_auc")
        st = at if at is not None else mt
        sn = an if an is not None else mn
        if use_t:
            hist_t.append((pt.detach().cpu().numpy(), yt.detach().cpu().numpy()))
            hist_t = hist_t[-5:]
        if use_n:
            hist_n.append((pn.detach().cpu().numpy(), yn.detach().cpu().numpy()))
            hist_n = hist_n[-5:]
        if use_t and st > best_t["score"]:
            best_t = {"score": float(st), "epoch": ep, "auc": at, "f1": float(mt)}
            torch.save(model.state_dict(), ckpt_t)
        if use_n and sn > best_n["score"]:
            best_n = {"score": float(sn), "epoch": ep, "auc": an, "f1": float(mn)}
            torch.save(model.state_dict(), ckpt_n)
        tr_loss = (tot / cnt) if cnt > 0 else float("nan")
        logger.info(f"[ROI] {fold_label} Epoch {ep:03d} train_loss={tr_loss:.4f}")
        if ep % 10 == 0 or ep == 1 or ep == args.epochs or args.smoke:
            logger.info(f"[ROI] {fold_label} Epoch {ep:03d} T f1={mt:.4f} auc={at if at is not None else 'NA'} "
                        f"(best {best_t['score']:.4f}@{best_t['epoch']}) | N f1={mn:.4f} "
                        f"auc={an if an is not None else 'NA'} (best {best_n['score']:.4f}@{best_n['epoch']})")
        epoch_secs.append(time.time() - t_ep)
        data_secs.append(t_data)
    logger.info(f"[ROI] {fold_label} done. best T={best_t['score']:.4f}@{best_t['epoch']} "
                f"(auc={best_t['auc']}, f1={best_t['f1']:.4f}) | N={best_n['score']:.4f}@{best_n['epoch']} "
                f"(auc={best_n['auc']}, f1={best_n['f1']:.4f})")

    # 成本埋点: 每 epoch 墙钟 / 数据等待占比 / GPU 显存峰值 (无 CUDA 时 None, 不报错)
    peak_gpu_mem_gb = None
    if device == "cuda" and torch.cuda.is_available():
        peak_gpu_mem_gb = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3)
    epoch_total = float(np.sum(epoch_secs)) if epoch_secs else None
    perf = {
        "epoch_sec": [round(float(x), 3) for x in epoch_secs],
        "epoch_sec_mean": float(np.mean(epoch_secs)) if epoch_secs else None,
        "epoch_sec_total": epoch_total,
        "data_wait_ratio": (float(np.sum(data_secs) / epoch_total)
                            if epoch_total and epoch_total > 0 else None),
        "peak_gpu_mem_gb": peak_gpu_mem_gb,
        "batch_size": int(args.batch_size),
        "device": device,
    }
    logger.info(f"[ROI] {fold_label} perf: epoch_mean={perf['epoch_sec_mean']}s "
                f"data_wait_ratio={perf['data_wait_ratio']} peak_gpu_mem_gb={peak_gpu_mem_gb}")

    def _ckpt_probs(path):
        """用指定 ckpt 重算外折概率 (校准/指标必须与报告用的 best 权重一致, 不用末轮权重)。"""
        if not (path and os.path.exists(path)):
            return None
        model.load_state_dict(torch.load(path, map_location=device))
        return _predict_ds(model, ds_va, device)

    res_best = {"t": None, "n": None}
    if use_t:
        r = _ckpt_probs(ckpt_t)
        res_best["t"] = None if r is None else r["t"]
    if use_n:
        r = _ckpt_probs(ckpt_n)
        res_best["n"] = None if r is None else r["n"]

    # 方案 A: 不在本折 val 上选阈值。kfold 由 main 用 LOFO(其余折 OOF) 计算后回填;
    # 单 split 无其他折, calT/calN 保持 None (只报无阈值指标)。
    calT = calN = None

    # 末 5 轮概率平均 (无需权重平均): 每轮 val 概率同序, 直接逐样本平均
    res_last5 = {"t": None, "n": None}
    last5 = {}
    for key, hist in (("t", hist_t), ("n", hist_n)):
        if hist:
            res_last5[key] = (np.mean([h[0] for h in hist], axis=0), hist[-1][1])
            m = _bin_metrics(*res_last5[key])
            m["epochs"] = list(range(max(1, args.epochs - len(hist) + 1), args.epochs + 1))
            last5[key] = m
        else:
            last5[key] = None

    pids = list(getattr(ds_va, "_pids", []) or [str(i) for i in range(len(ds_va))])
    base = f"{args.out_tag}_{fold_label}" + ("" if fold_label.startswith("s") else f"_s{args.seed}")
    oof_path = None
    if res_best["t"] is not None or res_best["n"] is not None:
        oof_path = _write_oof(os.path.join(OOF_DIR, f"{base}.csv"), pids, res_best)
        logger.info(f"[ROI] OOF 概率已落盘: {oof_path}")
    oof_last5_path = None
    if res_last5["t"] is not None or res_last5["n"] is not None:
        oof_last5_path = _write_oof(os.path.join(OOF_DIR, f"{base}_last5.csv"), pids, res_last5)
        logger.info(f"[ROI] 末 5 轮均值 OOF 已落盘: {oof_last5_path}")

    extra = {
        "metrics_best": {"t": _bin_metrics(*res_best["t"]) if res_best["t"] is not None else None,
                         "n": _bin_metrics(*res_best["n"]) if res_best["n"] is not None else None},
        "last5": last5,
        "oof_path": oof_path,
        "oof_probs": res_best,  # 供 main 做 LOFO 阈值 (不写盘)
        "oof_last5_path": oof_last5_path,
        "oof_last5_probs": res_last5,  # 末 5 轮均值概率, 供 LOFO(last5) 与记录
        "perf": perf,
        "use_t": use_t, "use_n": use_n,
    }
    return best_t, best_n, calT, calN, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-n", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--head-width", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kfold", type=int, default=0,
                    help=">0 时跑 k 折交叉验证 (样本取自 --full-csv 或 data/{tag}_full.csv, 折划分取自 --kfold-csv), 0=单 seed train/val 划分")
    ap.add_argument("--kfold-csv", default="",
                    help="患者级折划分主表 (如 data/master_5fold_356.csv, 列 patient_id,fold); 缺列/缺样本时回退 StratifiedKFold")
    ap.add_argument("--only-fold", type=int, default=-1,
                    help="kfold 下只跑指定折号 (0-based, 用于单折标定; 其余折不训练, LOFO 阈值将为 None)")
    ap.add_argument("--full-csv", default="",
                    help="kfold 用的固定全量表 (默认 data/{out_tag}_full.csv); 不再拼接 _s{seed} 训练/验证表")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-tag", default="hos_full")
    ap.add_argument("--mode", choices=["ct", "mri_t", "mri_c", "mri_d", "mri_all", "mri_c_ct", "mri_tc", "mri_tc_ct", "mri_t_ring", "mri_c_ring", "mri_tc_ring", "mri_t_body", "mri_c_body", "mri_all_ring"], default="ct",
                    help="ct=CT病灶ROI(Task010 mask); mri_t/c/d=MRI序列结肠ROI(total_mr mask); mri_all=三序列拼接; mri_c_ct=MRI-c结肠ROI+CT病灶ROI双源拼接[2ch]; mri_tc=t+c两序列拼接[2ch]; mri_tc_ct=t+c+CT病灶ROI[3ch, CT缺失零卷+标志]; mri_t_ring/mri_c_ring=④结肠mask膨胀2层ring(肠周/系膜); mri_tc_ring=①ring t+c两序列拼接[2ch]; mri_t_body/mri_c_body=②本体+ring双通道[2ch]; mri_all_ring=④ring t+c+d三序列拼接[3ch]")
    ap.add_argument("--self-written", dest="self_written", action="store_true",
                    help="用自写轻量 3D CNN (无预训练权重, 含轴向 slice attention) 替代 medicalnet10")
    ap.add_argument("--ring-iter", type=int, default=2,
                    help="[兼容旧参数] aligned 模式下为环宽 mm; 已被 --ring-mm 覆盖")
    ap.add_argument("--ring-mm", type=float, default=None,
                    help="环宽 mm (常用 2/5/10; 默认 2=保留历史基线; 直肠系膜淋巴结常在 5–15mm, 2mm 在 96³ 下仅约 1 体素)")
    ap.add_argument("--sdf-channel", dest="sdf_channel", action="store_true",
                    help="输入改为 原图灰度 + 连续物理 SDF 双通道 (每图像通道配一路 SDF; 不做二值化; 仅 aligned)")
    ap.add_argument("--sdf-only", dest="sdf_only", action="store_true",
                    help="形态捷径对照: 图像通道置零, 仅保留 SDF 通道 (隐含 --sdf-channel; 仅 aligned; 不可与 --roi-mask binary_only 同用)")
    ap.add_argument("--head-dropout", type=float, default=0.5,
                    help="head MLP dropout (Day1⑥ 超参快筛)")
    ap.add_argument("--gam", dest="gam", action="store_true",
                    help="H-CNN-ViT 降级版 Local GAM 门控 (自写网络内双路径: CNN局部+全局+门控融合, arXiv:2511.13869)")
    ap.add_argument("--fpn", dest="fpn", action="store_true",
                    help="3D FPN 多尺度: 自写网络 3 个 stage 输出 1x1 投影对齐 + 上采样到 8³ concat (Day2 B1)")
    ap.add_argument("--aecn", dest="aecn", action="store_true",
                    help="AECN 三轴向 slice attention (x/y/z 各自切片加权相乘, 滤正常组织层, arXiv:2103.03931, Day2 B2)")
    ap.add_argument("--clin-late", dest="clin_late", action="store_true",                    help="临床特征晚融合: 只进分类头不进 backbone 特征 (C4, 默认早融合)")
    ap.add_argument("--no-clinical", dest="no_clinical", action="store_true",
                    help="消融: 临床向量全置 0 (仅保留影像; 维度不变, 便于与含临床实验严格对比)")
    ap.add_argument("--n-only", dest="n_only", action="store_true",
                    help="只监督 N 头 (T 头不参与 loss/不选 ckpt), 供 S1/S2 单任务对照")
    ap.add_argument("--t-only", dest="t_only", action="store_true",
                    help="只监督 T 头 (N 头不参与 loss/不选 ckpt), 供 S1/S2 单任务对照")
    ap.add_argument("--aug", dest="aug", action="store_true",
                    help="ROI 随机增强 (翻转/旋转90°/强度扰动, 仅训练集; Day1 ① 梯队-1)")
    ap.add_argument("--shared-roi", dest="shared_roi", action="store_true",
                    help="A1: 多序列 (mri_tc/mri_all 等) 所有 mr_* 通道共用 mr_t 单一 bbox 中心 + 单一 SDF (消除 t/c bbox 中心错位)")
    ap.add_argument("--aug-lr", dest="aug_lr", action="store_true",
                    help="A2: 解剖保序弱增强 = 仅左右镜像 (x 轴翻转) + 图像通道强度抖动; SDF 通道不碰 (与 --aug 互斥, 用此代替 --aug)")
    ap.add_argument("--aux-pt", dest="aux_pt", action="store_true",
                    help="pT 四类辅助头 (pT1/2/3/4 → 4类 CE 联合训练, 逆频率权重; Day1 ③ 梯队-1)")
    ap.add_argument("--tn-hier", dest="tn_hier", action="store_true",
                    help="T/N 层次建模 (B6): N 头以 T 头输出为条件 (stop-grad, P(N|T) 先验注入; Day1 ④ 梯队-1)")
    ap.add_argument("--sam", dest="sam", action="store_true",
                    help="⑥ A1: Sharpness-Aware Minimization (SAM, rho 默认 0.05, 禁 AMP; Day1 ⑥ 梯队-2)")
    ap.add_argument("--sam-rho", dest="sam_rho", type=float, default=0.05,
                    help="SAM 扰动半径 rho (默认 0.05)")
    ap.add_argument("--warmup", dest="warmup", type=int, default=0,
                    help="⑦ A3: 线性 warmup 轮数 (0=关闭; 典型 5, 配 --epochs 100)")
    ap.add_argument("--mask-geometry", choices=["aligned", "legacy"], default="aligned",
                    help="mask 与影像网格对齐: aligned=与缓存卷同几何同轴序(默认); legacy=原始网格直接 Resize(历史口径, 存在轴序/物理错位)")
    ap.add_argument("--roi-mask", choices=["centroid", "bbox", "masked", "binary_only"], default="centroid",
                    help="ROI 中心: centroid=mask 质心(历史口径, 不屏蔽图像); bbox=mask bbox 中心; masked=bbox 中心且只保留 mask 内体素; binary_only=bbox 中心且输入换成二值 mask 本身(形态捷径检验)")
    ap.add_argument("--cache-dir", default="cache_hos",
                    help="确定性预处理缓存目录 (默认 cache_hos; 传 cache_hos_n4 可复用 N4 缓存)")
    ap.add_argument("--mask-mr-root", default=None,
                    help="A3: MRI 结肠 mask 根目录覆盖 (默认 data/masks_totalseg_mr_{seq}; 传排斥版如 data/masks_totalseg_mr_t_excl)")
    args = ap.parse_args()
    if args.mask_mr_root:
        for seq in ("t", "c", "d"):
            MASK_DIR_MR[seq] = os.path.join(ROOT, args.mask_mr_root)
    if args.n_only and args.t_only:
        ap.error("--n-only 与 --t-only 互斥")
    if args.only_fold >= 0 and args.kfold <= 0:
        ap.error("--only-fold 需要同时指定 --kfold > 0 (单 seed 模式无折号)")
    if args.sdf_only:
        if args.mask_geometry != "aligned":
            ap.error("--sdf-only 仅支持 --mask-geometry aligned (legacy 无 SDF)")
        if args.roi_mask == "binary_only":
            ap.error("--sdf-only 与 --roi-mask binary_only 互斥 (两者都是形态捷径)")
        args.sdf_channel = True  # 隐含打开 SDF 通道
    global RING_ITERATIONS
    RING_ITERATIONS = int(round(args.ring_mm if args.ring_mm is not None else args.ring_iter))  # mm (aligned)

    _seed_everything(args.seed)
    device = "cuda" if (torch.cuda.is_available() and not args.smoke) else "cpu"
    logger = setup_logger(os.path.join(ROOT, "logs"), f"roidl_{args.out_tag}_s{args.seed}")

    train_csv = os.path.join(ROOT, "data", f"{args.out_tag}_train_s{args.seed}.csv")
    val_csv = os.path.join(ROOT, "data", f"{args.out_tag}_val_s{args.seed}.csv")
    raw_note = {"source": "data/hos_pairs_raw.csv", "fit": "fold_train_only",
                "phys_filter": {k: [None if not np.isfinite(v) else v for v in rng]
                                for k, rng in PHYS_RANGES.items()},
                "log1p": list(LOG_COLS), "clip": "1%-99%", "z_clip": "+-5",
                "impute": "train_median", "missing_indicator": True,
                "no_clinical": bool(args.no_clinical),
                "ring_mm": RING_ITERATIONS, "sdf_channel": bool(args.sdf_channel),
                "sdf_only": bool(args.sdf_only)}

    if args.kfold > 0:
        from sklearn.model_selection import StratifiedKFold
        cands = [args.full_csv,
                 os.path.join(ROOT, "data", f"{args.out_tag}_full.csv"),
                 os.path.join(ROOT, "data", "hos_branch_full.csv"),
                 os.path.join(ROOT, "data", "hos_pairs.csv")]
        full_csv = next((p for p in cands if p and os.path.exists(p)), "")
        if full_csv:
            # augment 由 _SubDS 统一控制, 底表不再增强 (否则训练折会双重增强)
            ds_all = DataROI(full_csv, mode=args.mode, augment=False, mask_geometry=args.mask_geometry,
                             roi_mask=args.roi_mask, cache_dir=args.cache_dir, sdf_channel=args.sdf_channel,
                             sdf_only=args.sdf_only, shared_roi=args.shared_roi, aug_lr=args.aug_lr)
        elif os.path.exists(train_csv) and os.path.exists(val_csv):
            logger.warning(f"[ROI] 未找到 {full_csv}, 回退拼接 {os.path.basename(train_csv)}+{os.path.basename(val_csv)}"
                           f" (仅对已存在的 seed 有效)")
            ds_all = DataROI(train_csv, mode=args.mode, augment=False, mask_geometry=args.mask_geometry,
                             roi_mask=args.roi_mask, cache_dir=args.cache_dir, sdf_channel=args.sdf_channel,
                             sdf_only=args.sdf_only, shared_roi=args.shared_roi, aug_lr=args.aug_lr)
            ds_b = DataROI(val_csv, mode=args.mode, augment=False, mask_geometry=args.mask_geometry,
                           roi_mask=args.roi_mask, cache_dir=args.cache_dir, sdf_channel=args.sdf_channel,
                           sdf_only=args.sdf_only, shared_roi=args.shared_roi, aug_lr=args.aug_lr)
            for attr in ("_vols", "_pids", "_clin", "_clin_raw", "_lab_t", "_lab_n", "_lab_pt", "_ct_flag"):
                getattr(ds_all, attr).extend(getattr(ds_b, attr))
        else:
            raise SystemExit(f"kfold 需要 {full_csv} 或已存在的 seed split (当前 seed={args.seed})")

        pids = ds_all._pids
        lab_t, lab_n, lab_pt = ds_all._lab_t, ds_all._lab_n, ds_all._lab_pt
        raw_all = ds_all.clin_raw_matrix()
        folds_idx = None
        not_in_master = []
        if args.kfold_csv and os.path.exists(args.kfold_csv):
            mt = pd.read_csv(args.kfold_csv, encoding="utf-8-sig")
            mt["patient_id"] = mt["patient_id"].astype(str).str.strip()
            fmap = dict(zip(mt["patient_id"], mt["fold"].astype(int)))
            if int(mt["fold"].max()) + 1 != args.kfold:
                logger.warning(f"[ROI] 主表折数={int(mt['fold'].max()) + 1} 与 --kfold={args.kfold} 不一致, "
                               f"按 --kfold 截断/补空折")
            missing = [p for p in pids if p not in fmap]
            not_in_master = list(missing)
            if missing:
                logger.warning(f"[ROI] 显式跳过 {len(missing)} 例 (不在主表, 不参与任何折): {missing}")
            keep = [i for i, p in enumerate(pids) if p in fmap]
            folds_idx = [([i for i in keep if int(fmap[pids[i]]) != k],
                          [i for i in keep if int(fmap[pids[i]]) == k]) for k in range(args.kfold)]
            logger.info(f"[ROI] 折划分来自 {args.kfold_csv} (n={len(keep)})")
        else:
            strat = [f"{int(lt)}_{int(ln)}" for lt, ln in zip(lab_t, lab_n)]
            skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
            folds_idx = [(tr.tolist(), va.tolist()) for tr, va in skf.split(np.zeros(len(pids)), strat)]
            logger.warning(f"[ROI] 未使用 --kfold-csv, 回退 StratifiedKFold(T×N, seed={args.seed}); "
                           f"跨子模态复用时请改用 data/master_5fold_356.csv")

        only = int(args.only_fold)
        if only >= len(folds_idx):
            raise SystemExit(f"--only-fold={only} 超出折数 {len(folds_idx)}")
        sel = [only] if only >= 0 else list(range(len(folds_idx)))
        if only >= 0:
            logger.warning(f"[ROI] --only-fold={only}: 仅训练/评估该折, 其余折不跑; "
                           f"因缺少其余折 OOF, LOFO 阈值将为 None (单折仅用于成本标定)")
        # slots 与折号严格对齐: 只跑单折时其余折留 None, 避免 LOFO 索引/来源错位
        oof_all = {"t": [None] * len(folds_idx), "n": [None] * len(folds_idx),
                   "t_last5": [None] * len(folds_idx), "n_last5": [None] * len(folds_idx)}
        folds = []
        for fi in sel:
            tr_list, va_list = folds_idx[fi]
            prep = ClinPrep(raw_all[tr_list])  # 折内训练集统计, 验证折不参与
            args.num_clinical = prep.dim + (1 if args.mode == "mri_tc_ct" else 0)
            z_tr, z_va = prep.transform(raw_all[tr_list]), prep.transform(raw_all[va_list])
            if args.no_clinical:
                z_tr, z_va = np.zeros_like(z_tr), np.zeros_like(z_va)
            if args.mode == "mri_tc_ct":  # CT 病灶可用标志始终保留 (与 DataROI.set_clin 同口径)
                flag = np.asarray(ds_all._ct_flag, dtype=np.float32)
                z_tr = np.concatenate([z_tr, flag[tr_list][:, None]], axis=1)
                z_va = np.concatenate([z_va, flag[va_list][:, None]], axis=1)
            # 折内标签必须切片: 旧版传全量 lab_*, 导致类别权重/辅助头统计混入验证折标签
            ds_tr_f = _SubDS(ds_all, tr_list,
                             lab_t=[lab_t[i] for i in tr_list], lab_n=[lab_n[i] for i in tr_list],
                             lab_pt=[lab_pt[i] for i in tr_list], augment=args.aug or args.aug_lr,
                             aug_lr=args.aug_lr,
                             clin=z_tr, pids=[pids[i] for i in tr_list])
            ds_va_f = _SubDS(ds_all, va_list,
                             lab_t=[lab_t[i] for i in va_list], lab_n=[lab_n[i] for i in va_list],
                             lab_pt=[lab_pt[i] for i in va_list], augment=False,
                             clin=z_va, pids=[pids[i] for i in va_list])
            bt, bn, calT, calN, extra = run_one(ds_tr_f, ds_va_f, args, device, logger, f"k{fi}")
            oof_all["t"][fi] = extra["oof_probs"]["t"]
            oof_all["n"][fi] = extra["oof_probs"]["n"]
            oof_all["t_last5"][fi] = extra["oof_last5_probs"]["t"]
            oof_all["n_last5"][fi] = extra["oof_last5_probs"]["n"]
            folds.append({"fold": fi,
                          "T_auc": bt["auc"], "T_f1": bt["f1"], "T_epoch": bt["epoch"],
                          "N_auc": bn["auc"], "N_f1": bn["f1"], "N_epoch": bn["epoch"],
                          "calT": calT, "calN": calN, "metrics_best": extra["metrics_best"],
                          "last5": extra["last5"], "oof_path": extra["oof_path"],
                          "oof_last5_path": extra["oof_last5_path"],
                          "perf": extra["perf"],
                          "n_train": len(tr_list), "n_val": len(va_list)})

        # 方案 A: 阈值来源 = 其余折 OOF (Leave-One-Fold-Out), 当前折不参与自己的阈值搜索
        use_t = not args.n_only
        use_n = not args.t_only
        calT_all = lofo_calibrate(oof_all["t"]) if use_t else [None] * len(folds_idx)
        calN_all = lofo_calibrate(oof_all["n"]) if use_n else [None] * len(folds_idx)
        calT_l5 = lofo_calibrate(oof_all["t_last5"]) if use_t else [None] * len(folds_idx)
        calN_l5 = lofo_calibrate(oof_all["n_last5"]) if use_n else [None] * len(folds_idx)
        for fd in folds:
            fi = fd["fold"]
            fd["calT"], fd["calN"] = calT_all[fi], calN_all[fi]
            fd["calT_last5"], fd["calN_last5"] = calT_l5[fi], calN_l5[fi]
            logger.info(f"[ROI] fold k{fi} LOFO calT={fd['calT']} calN={fd['calN']}")
            rp = os.path.join(ROOT, "logs", f"eval_report_roidl_{args.out_tag}_k{fi}_s{args.seed}.json")
            with open(rp, "w", encoding="utf-8") as f:
                json.dump({"meta": {"mode": args.mode, "kfold": args.kfold, "fold": fi,
                                    "only_fold": args.only_fold,
                                    "mask_geometry": args.mask_geometry, "roi_mask": args.roi_mask,
                                    "clinical": raw_note, "select": "val_roc_auc",
                                    "threshold": "LOFO(other folds OOF); only-fold 时无其余折 → None"},
                           "params": {"mode": args.mode, "epochs": args.epochs, "lr": args.lr, "lr_n": args.lr_n},
                           "result": {"best_epoch": fd["T_epoch"], "best_auc": fd["T_auc"],
                                      "best_macro_f1": fd["T_f1"],
                                      "best_epoch_n": fd["N_epoch"], "best_auc_n": fd["N_auc"],
                                      "best_macro_f1_n": fd["N_f1"],
                                      "calT": fd["calT"], "calN": fd["calN"],
                                      "calT_last5": fd["calT_last5"], "calN_last5": fd["calN_last5"],
                                      "metrics_best": fd["metrics_best"], "last5": fd["last5"],
                                      "oof_path": fd["oof_path"],
                                      "oof_last5_path": fd["oof_last5_path"],
                                      "perf": fd["perf"]}}, f, ensure_ascii=False, indent=2)
        import statistics

        def _ms(vals):
            vals = [float(v) for v in vals if v is not None]
            return (statistics.mean(vals), statistics.pstdev(vals)) if vals else (None, None)

        def _mean(key, task):
            vals = [fd["metrics_best"][task][key] for fd in folds
                    if fd["metrics_best"][task] and fd["metrics_best"][task][key] is not None]
            return statistics.mean(vals) if vals else None

        t_auc_m, t_auc_s = _ms([f["T_auc"] for f in folds])
        n_auc_m, n_auc_s = _ms([f["N_auc"] for f in folds])
        t_f1_m, t_f1_s = _ms([f["T_f1"] for f in folds])
        n_f1_m, n_f1_s = _ms([f["N_f1"] for f in folds])
        summ = {"mode": args.mode, "kfold": args.kfold, "only_fold": args.only_fold,
                "mask_geometry": args.mask_geometry,
                "roi_mask": args.roi_mask, "clinical": raw_note,
                "select": "val_roc_auc", "threshold": "LOFO(other folds OOF); only-fold 时 None",
                "T_auc_mean": t_auc_m, "T_auc_std": t_auc_s,
                "N_auc_mean": n_auc_m, "N_auc_std": n_auc_s,
                "T_mean": t_f1_m, "T_std": t_f1_s,
                "N_mean": n_f1_m, "N_std": n_f1_s,
                "T_cal_f1_mean": _ms([(f["calT"] or {}).get("macro_f1") for f in folds])[0],
                "N_cal_f1_mean": _ms([(f["calN"] or {}).get("macro_f1") for f in folds])[0],
                "T_pr_auc_mean": _mean("pr_auc", "t"), "N_pr_auc_mean": _mean("pr_auc", "n"),
                # 末 5 轮均值 (对冲早停偏置): 阈值 0.5 / 其自身 LOFO 阈值两套并列
                "T_last5_auc_mean": _ms([(f["last5"]["t"] or {}).get("roc_auc") for f in folds])[0],
                "T_last5_f1_mean": _ms([(f["last5"]["t"] or {}).get("macro_f1") for f in folds])[0],
                "T_last5_cal_f1_mean": _ms([(f["calT_last5"] or {}).get("macro_f1") for f in folds])[0],
                "N_last5_auc_mean": _ms([(f["last5"]["n"] or {}).get("roc_auc") for f in folds])[0],
                "N_last5_f1_mean": _ms([(f["last5"]["n"] or {}).get("macro_f1") for f in folds])[0],
                "N_last5_cal_f1_mean": _ms([(f["calN_last5"] or {}).get("macro_f1") for f in folds])[0],
                "perf_mean_epoch_sec": _ms([(f["perf"] or {}).get("epoch_sec_mean") for f in folds])[0],
                "perf_data_wait_ratio": _ms([(f["perf"] or {}).get("data_wait_ratio") for f in folds])[0],
                "perf_peak_gpu_mem_gb": _ms([(f["perf"] or {}).get("peak_gpu_mem_gb") for f in folds])[0],
                "n_not_in_master": len(not_in_master),
                "not_in_master_pids": not_in_master,
                "folds": folds}
        with open(os.path.join(ROOT, "logs", f"kfold_summary_roidl_{args.out_tag}_s{args.seed}.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)
        logger.info(f"[ROI] KFOLD done. T AUC={t_auc_m}+-{t_auc_s} F1={t_f1_m}+-{t_f1_s} | "
                    f"N AUC={n_auc_m}+-{n_auc_s} F1={n_f1_m}+-{n_f1_s} | "
                    f"calT={summ['T_cal_f1_mean']} calN={summ['N_cal_f1_mean']}")
        return

    for p in (train_csv, val_csv):
        if not os.path.exists(p):
            raise SystemExit(f"单 seed 模式需要 {p} (可用 --seed 指定已存在的划分, 或用 --kfold + --kfold-csv)")
    ds_tr = DataROI(train_csv, mode=args.mode, augment=args.aug or args.aug_lr,
                    mask_geometry=args.mask_geometry, roi_mask=args.roi_mask, cache_dir=args.cache_dir,
                    sdf_channel=args.sdf_channel, sdf_only=args.sdf_only,
                    shared_roi=args.shared_roi, aug_lr=args.aug_lr)
    ds_va = DataROI(val_csv, mode=args.mode, augment=False,
                    mask_geometry=args.mask_geometry, roi_mask=args.roi_mask, cache_dir=args.cache_dir,
                    sdf_channel=args.sdf_channel, sdf_only=args.sdf_only,
                    shared_roi=args.shared_roi, aug_lr=args.aug_lr)
    prep = ClinPrep(ds_tr.clin_raw_matrix())  # 只用训练集统计, 验证集不参与
    ds_tr.set_clin(prep.transform(ds_tr.clin_raw_matrix()), zero=args.no_clinical)
    ds_va.set_clin(prep.transform(ds_va.clin_raw_matrix()), zero=args.no_clinical)
    args.num_clinical = ds_tr.clin_dim
    logger.info(f"[ROI] 临床标准化: 训练集 fit (n={len(ds_tr)}), dim={args.num_clinical} "
                f"(7 值 + 7 missing 指示位{' +1 CT 标志' if args.mode == 'mri_tc_ct' else ''})")

    bt, bn, calT, calN, extra = run_one(ds_tr, ds_va, args, device, logger, f"s{args.seed}")
    report = {
        "meta": {"branch": args.out_tag, "run": "roidl", "tag": args.out_tag, "seed": args.seed, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "mask_geometry": args.mask_geometry, "roi_mask": args.roi_mask, "n_weight": "inverse(label_N)",
                 "clinical": raw_note, "select": "val_roc_auc (单类时退回 macro_f1@0.5)",
                 "threshold": "none (单 split 无 OOF; 方案 A 只报无阈值指标)"},
        "params": {"backbone": "self_written_3d_cnn" if args.self_written else "medicalnet10 + avgpool", "roi": f"{ROI}^3 ({args.roi_mask})", "mask": "Task010" if args.mode == "ct" else "total_mr colon",
                   "mode": args.mode, "lr": args.lr, "lr_n": args.lr_n, "head_width": args.head_width, "epochs": args.epochs,
                   "n_train": len(ds_tr), "n_val": len(ds_va)},
        "result": {
            "best_epoch": bt["epoch"], "best_auc": bt["auc"], "best_macro_f1": bt["f1"],
            "best_epoch_n": bn["epoch"], "best_auc_n": bn["auc"], "best_macro_f1_n": bn["f1"],
            "calT": calT, "calN": calN,
            "calT_last5": None, "calN_last5": None,
            "metrics_best": extra["metrics_best"], "last5": extra["last5"],
            "oof_path": extra["oof_path"], "oof_last5_path": extra["oof_last5_path"],
            "perf": extra["perf"],
        },
    }
    rp = os.path.join(ROOT, "logs", f"eval_report_roidl_{args.out_tag}_s{args.seed}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"[ROI] report saved: {rp}")
    if not args.smoke:
        archive(ROOT, args.out_tag, args.seed)


if __name__ == "__main__":
    main()
