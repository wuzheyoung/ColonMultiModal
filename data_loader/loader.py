"""
CMB-CRC 多模态数据加载。

读取配对 csv（每行一个样本）：
    patient_id, label, ct_path, mri_path, ct_available, mri_available, ...

特点:
  - CT / MRI 各自独立预处理管线（强度归一化方式不同）
  - 预处理拆分为【确定性部分】(读取/重采样/归一化/裁剪/Resize) 与
    【随机增强部分】(RandAffine/RandFlip)。确定性结果可缓存，训练时只在运行时
    做随机增强，避免每个 epoch 重复读盘和重采样（GPU 空等的主因）。
  - 缺失模态（path 为空 / available=False）填零张量，并置对应 available=False
  - 临床特征暂时跳过，用零向量占位（保持模型 forward 接口不变）
"""

import hashlib
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import SimpleITK as sitk

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Spacing,
    ScaleIntensityRange, NormalizeIntensity, CropForeground, Resize,
    RandAffine, RandFlip, RandZoom, RandShiftIntensity, RandScaleIntensity,
    RandGaussianNoise, RandCoarseDropout,
)

# MRI 三序列通道固定顺序（与 models/network.py 的 _SEQ_ORDER 保持一致）：
# loader 输出、缓存卷、mri_seq_available 均按此顺序；单序列子集只取对应下标切片。
_SEQ_ORDER = ("t", "c", "d")


def _ct_deterministic(spatial_size):
    # 先 HU 窗裁剪到软组织范围，CropForeground 去背景，最后前景 z-score
    # (与 MedicalNet 预训练权重的前景归一化对齐，提升迁移效果)
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRange(a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForeground(),
        Resize(spatial_size=spatial_size, mode="trilinear"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ])


def _mri_deterministic(spatial_size):
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        CropForeground(),
        Resize(spatial_size=spatial_size, mode="trilinear"),
    ])


# ── 三序列 MRI 确定性预处理（round5：T2/CE/DWI 整合）───────────────────────
# 关键: 同患者 t/c/d 三序列的 spacing/origin/direction/FOV 完全不同,
# 直接堆叠通道=解剖错位。策略: 以 T2(或任一可用序列) 为锚构建 1mm 各向同性
# 参考网格, 其余序列 resample 到该网格, 再用统一 bbox 裁剪 + Resize(96³)。
_MRI_SEQ_ORDER = ("t", "c", "d")
_MRI_REF_SPACING = (1.0, 1.0, 1.0)


def _mri_reference_grid(anchor):
    """以 anchor 图的物理范围构建 1mm 各向同性参考网格，返回该网格上的 anchor。"""
    rf = sitk.ResampleImageFilter()
    phys = np.array(anchor.GetSize()) * np.array(anchor.GetSpacing())
    new_size = np.ceil(phys / np.array(_MRI_REF_SPACING)).astype(int).tolist()
    rf.SetOutputSpacing(_MRI_REF_SPACING)
    rf.SetOutputDirection(anchor.GetDirection())
    rf.SetOutputOrigin(anchor.GetOrigin())
    rf.SetSize(new_size)
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(0.0)
    return rf.Execute(anchor)


def _resample_to_ref(img, ref):
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(ref)
    rf.SetInterpolator(sitk.sitkLinear)
    rf.SetDefaultPixelValue(0.0)
    return rf.Execute(img)


def _zscore_nonzero(arr):
    """按 nonzero 掩码 z-score（与 MONAI NormalizeIntensity(nonzero=True) 对齐）。"""
    arr = arr.astype(np.float32)
    mask = arr != 0
    if not mask.any():
        return arr
    mu = arr[mask].mean()
    sd = arr[mask].std()
    if not np.isfinite(sd) or sd == 0:
        return arr
    out = np.zeros_like(arr)
    out[mask] = (arr[mask] - mu) / sd
    return out


def _mri_bbox(mask, pad=5):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])
    d0, h0, w0 = idx.min(0)
    d1, h1, w1 = idx.max(0) + 1
    D, H, W = mask.shape
    d0, d1 = max(d0 - pad, 0), min(d1 + pad, D)
    h0, h1 = max(h0 - pad, 0), min(h1 + pad, H)
    w0, w1 = max(w0 - pad, 0), min(w1 + pad, W)
    return (d0, d1, h0, h1, w0, w1)


def _mri_seqs_deterministic(t_path, c_path, d_path, seq_avail, spatial_size):
    """
    三序列 MRI 确定性预处理（T2 锚定物理对齐）。
    seq_avail: [t_avail, c_avail, d_avail]（bool）
    返回 float32 [3, D, H, W]（通道序 t/c/d；缺失序列通道全 0）。
    """
    paths = (t_path, c_path, d_path)
    imgs = [sitk.ReadImage(p) if (isinstance(p, str) and p and os.path.exists(p)) else None
            for p in paths]

    anchor_idx = next((i for i in range(3) if imgs[i] is not None), None)
    if anchor_idx is None:
        return np.zeros((3,) + tuple(spatial_size), dtype=np.float32)

    ref = _mri_reference_grid(imgs[anchor_idx])
    aligned = []
    for img in imgs:
        if img is None:
            aligned.append(None)
        else:
            aligned.append(sitk.GetArrayFromImage(_resample_to_ref(img, ref)))  # [D,H,W]

    zs = [(_zscore_nonzero(a) if a is not None else None) for a in aligned]

    # 统一裁剪框：取任一可用序列的 nonzero 掩码 bbox
    mask_ref = None
    for z in zs:
        if z is not None:
            mask_ref = z != 0
            break
    bbox = _mri_bbox(mask_ref)

    resize = Resize(spatial_size=spatial_size, mode="trilinear")
    out = []
    for z in zs:
        if z is None:
            out.append(None)
        else:
            d0, d1, h0, h1, w0, w1 = bbox
            out.append(np.ascontiguousarray(resize(z[d0:d1, h0:h1, w0:w1][None])[0]))
            # 注意: z 是 [D,H,W]，bbox 索引按 (d,h,w)

    vols = []
    for i in range(3):
        vols.append(np.zeros(tuple(spatial_size), dtype=np.float32)
                    if out[i] is None else out[i].astype(np.float32))
    return np.stack(vols, axis=0)



def _random_augment():
    # 训练时随机增强（逐模态独立应用, 作用在确定性预处理输出上）。
    # 强化版: 几何(翻转/仿射/缩放) + 强度(偏移/缩放) + 噪声 + 局部擦除(cutout),
    # 针对 154 小样本过拟合/方差问题。旋转角度比旧版更温和(0.2→0.15 rad),
    # 仿射与翻转概率提高; 强度/噪声/cutout 均为低频温和扰动, 不破坏模态内容。
    return Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandAffine(prob=0.7, rotate_range=(0.15, 0.15, 0.15),
                   scale_range=(0.1, 0.1, 0.1),
                   shear_range=(0.05, 0.05, 0.05),
                   mode="bilinear", padding_mode="border"),
        RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1,
                 mode="trilinear", padding_mode="edge"),
        RandShiftIntensity(prob=0.5, offsets=0.15),
        RandScaleIntensity(prob=0.5, factors=0.2),
        RandGaussianNoise(prob=0.3, mean=0.0, std=0.02),
        RandCoarseDropout(prob=0.3, holes=3, spatial_size=12,
                          dropout_holes=True, fill_value=0.0),
    ])


# ── 确定性预处理持久化缓存 ──────────────────────────────────────────────
# 首轮把 (读盘/重采样/归一化/裁剪/Resize) 的确定性结果写为 .npy,
# 之后每次启动直接读盘, 跳过解压与重采样 (Hos 每样本约 25MB 压缩 NIfTI,
# 全量首轮预处理 ~8 分钟, 缓存后可忽略)。缓存输出与实时管线逐位一致,
# 不影响模型输入分布。
_CACHE_VERSION = "v2"


def _det_cache_key(path, spatial_size):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = "?"
    raw = f"{_CACHE_VERSION}|{os.path.abspath(path)}|{mtime}|{tuple(spatial_size)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _det_cache_key_mri(t_path, c_path, d_path, spatial_size):
    """三序列 MRI 缓存 key：含 t/c/d 三路径 + 各自 mtime（任一序列变化即失效）。"""
    parts = []
    for p in (t_path, c_path, d_path):
        if isinstance(p, str) and p:
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = "?"
            parts.append(f"{os.path.abspath(p)}|{mtime}")
        else:
            parts.append("NA")
    raw = f"{_CACHE_VERSION}|mri3|{';'.join(parts)}|{tuple(spatial_size)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class _DetDiskCache:
    """线程安全的 .npy 缓存: 读不到就计算并原子写盘。"""

    def __init__(self, cache_dir, spatial_size):
        self.cache_dir = os.path.abspath(cache_dir)
        self.spatial_size = tuple(spatial_size)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _npy_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.npy")

    def get(self, path):
        key = _det_cache_key(path, self.spatial_size)
        p = self._npy_path(key)
        if os.path.exists(p):
            return torch.from_numpy(np.load(p)), True
        return None, False

    def put(self, path, tensor):
        key = _det_cache_key(path, self.spatial_size)
        p = self._npy_path(key)
        if os.path.exists(p):
            return
        fd, tmp = tempfile.mkstemp(suffix=".npy.tmp", dir=self.cache_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, tensor.numpy())
            os.replace(tmp, p)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def get_vol(self, key):
        p = self._npy_path(key)
        if os.path.exists(p):
            return torch.from_numpy(np.load(p)), True
        return None, False

    def put_vol(self, key, tensor):
        p = self._npy_path(key)
        if os.path.exists(p):
            return
        fd, tmp = tempfile.mkstemp(suffix=".npy.tmp", dir=self.cache_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, tensor.numpy())
            os.replace(tmp, p)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


class CRCMultiModalDataset(Dataset):
    def __init__(self, csv_path, label_col="label", spatial_size=(96, 96, 96),
                 num_clinical=1, clinical_cols=None, multi_task=False,
                 secondary_label_col="label_N", is_train=True, cache=True,
                 disk_cache_dir=None, mri_seqs=("t", "c", "d"), ct_seq="auto",
                 use_aug=True, consist_modal=None, lesion_dir=None, lesion_size=64,
                 mri_lesion_dir=None, mri_lesion_size=64, mri_ring_dir=None):
        """consist_modal: 'mri'/'ct'/None —— round9 C2 一致性正则:
        训练时对该模态卷做第二次独立增强, 返回 {modal}_second 副本
        (主增强 vs 副本增强 的特征一致性约束, 增强不变性防过拟合)。
        lesion_dir: round9 F1 病灶融合分支——Task010 mask 目录 (data/masks_task010),
        质心中心裁剪 lesion_size³ 卷作为病灶分支输入 (无 mask 样本零卷 + 标志 False)。"""
        self.df = pd.read_csv(csv_path)
        self.label_col = label_col
        self.secondary_label_col = secondary_label_col if multi_task else None
        self.clinical_cols = list(clinical_cols or [])
        # 临床维度取配置与列数较大者（列缺失时用 0 兜底）
        self.num_clinical = max(num_clinical, len(self.clinical_cols))
        self.spatial_size = tuple(spatial_size)
        self.is_train = is_train
        self.ct_det = _ct_deterministic(self.spatial_size)
        # round5: MRI 改三序列对齐管线（_mri_seqs_deterministic），不再用单序列 _mri_deterministic
        self.mri_det = None
        self.augment = _random_augment() if (is_train and use_aug) else None  # round8: use_aug 开关
        self.consist_modal = consist_modal  # round9 C2: 一致性正则模态 (None/'mri'/'ct')
        # round9 F1: 病灶分支输入 (Task010 mask 质心 64³ 裁剪)
        self.lesion_dir = lesion_dir
        self.lesion_size = lesion_size
        self.mri_lesion_dir = mri_lesion_dir
        self.mri_lesion_size = mri_lesion_size
        self.mri_ring_dir = mri_ring_dir  # round9 Day1⑤: 若设置, mri_lesion 卷用 ring mask (肠周/系膜)
        self._lesion_cache = {}  # pid -> roi [1,lesion,lesion,lesion] np or None
        self._mri_lesion_cache = {}  # pid -> roi [1,L,L,L] np or None (round9 MRI-t ROI)
        self._mri_ring_cache = {}  # pid -> roi [1,L,L,L] np or None (Day1⑤ ring: 肠周/系膜)
        # 缓存确定性预处理结果（train / val 均可缓存；train 的随机增强在运行时另做）
        self._cache = {} if cache else None
        # 持久化磁盘缓存: 跨进程复用确定性预处理, 跳过重复解压/重采样
        self.disk = _DetDiskCache(disk_cache_dir, self.spatial_size) if disk_cache_dir else None
        # round7: MRI 序列子集（如 ("t",) 单序列分支）。缓存/全量卷按 _SEQ_ORDER 存 [t,c,d]，
        #          这里只记录子集下标，读取后做通道切片（磁盘缓存 key 不变，可复用全量缓存）。
        self.mri_seqs = tuple(mri_seqs)
        self._mri_idx = [_SEQ_ORDER.index(s) for s in self.mri_seqs]
        # round7: CT 序列选择（plain=平扫 / enhanced=增强 / auto=沿用原 ct_path 优先级）
        self.ct_seq = ct_seq

    def __len__(self):
        return len(self.df)

    def _zero_volume(self):
        return torch.zeros(1, *self.spatial_size, dtype=torch.float32)

    def _zero_mri(self):
        """MRI 全 0 张量 [3, D, H, W]（恒为全量三通道；子集由网络按 _SEQ_ORDER 索引取通道）。"""
        return torch.zeros(3, *self.spatial_size, dtype=torch.float32)

    def _pick_ct_path(self, row):
        """按 ct_seq 选择 CT 序列路径（plain / enhanced / auto）。"""
        key = {"plain": "ct_path_plain", "enhanced": "ct_path_enhanced",
               "auto": "ct_path"}.get(self.ct_seq, "ct_path")
        v = row.get(key, "")
        if not isinstance(v, str):
            return ""
        return v

    def _load_det(self, path, pipeline):
        img = pipeline(path)
        return torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32)

    def _label(self, row):
        # label 可能为空（如未补分期），空值统一记为 -1（训练时可据此过滤）
        val = row.get(self.label_col, "")
        if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
            return -1
        return int(val)

    def _secondary_label(self, row):
        if self.secondary_label_col is None:
            return -1
        val = row.get(self.secondary_label_col, "")
        if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).strip() == "":
            return -1
        return int(val)

    def _clinical_vec(self, row):
        vec = []
        for c in self.clinical_cols:
            v = row.get(c, 0.0)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0.0
            vec.append(float(v))
        if not vec:
            vec = [0.0] * self.num_clinical
        return torch.tensor(vec, dtype=torch.float32)

    def _load_mri_lesion_roi(self, pid, mri_vol, ring=False):
        """round9 MRI-t ROI: total_mr t 结肠 mask (masks_totalseg_mr_t/{pid}/colon.nii.gz)
        质心定位 → 从 MRI t 通道卷中心裁剪 lesion_size³。
        ring=True: 先二值膨胀 2 层取环 (肠周/系膜区域) 再质心定位 (Day1⑤ N 头 ring 嵌入)。
        mri_vol: np [3,96,96,96] 缓存卷, t 通道=索引 0 (_SEQ_ORDER=('t','c','d'))。
        返回 (roi [1,L,L,L] np.float32, has_mask bool); 无 mask/空 mask → (None, False)。"""
        if self.mri_lesion_dir is None:
            return None, False
        cache = self._mri_ring_cache if ring else self._mri_lesion_cache
        if pid in cache:
            roi = cache[pid]
            return roi, roi is not None
        import nibabel as nib
        from monai.transforms import Resize
        p = os.path.join(self.mri_lesion_dir, pid, "colon.nii.gz")
        roi = None
        has = False
        if os.path.exists(p):
            try:
                m = nib.load(p).get_fdata()
            except Exception:
                m = None
            if m is not None and float((m > 0).sum()) > 0:
                L = self.mri_lesion_size
                m3 = Resize(spatial_size=self.spatial_size, mode="nearest")(
                    m[None].astype(np.float32))[0]  # [96,96,96]
                if ring:
                    from scipy.ndimage import binary_dilation
                    d = binary_dilation(m3 > 0, iterations=2)
                    m3 = (d & ~(m3 > 0)).astype(np.float32)  # 膨胀取环
                idx = np.argwhere(m3 > 0)
                if idx.size == 0:
                    cache[pid] = None
                    return None, False
                cen = idx.mean(axis=0).astype(int)
                half = L // 2
                t = mri_vol[0] if mri_vol.ndim == 4 else mri_vol  # t 通道 [96,96,96]
                roi = np.zeros((L, L, L), dtype=np.float32)
                z0, z1 = max(0, cen[0] - half), min(96, cen[0] + half)
                y0, y1 = max(0, cen[1] - half), min(96, cen[1] + half)
                x0, x1 = max(0, cen[2] - half), min(96, cen[2] + half)
                if z1 <= z0 or y1 <= y0 or x1 <= x0:
                    cache[pid] = None
                    return None, False
                src = t[z0:z1, y0:y1, x0:x1]
                oz0 = max(0, half - cen[0]); oy0 = max(0, half - cen[1]); ox0 = max(0, half - cen[2])
                roi[oz0:oz0 + (z1 - z0), oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)] = src
                roi = roi[None]  # [1,L,L,L]
                has = True
        cache[pid] = roi
        return roi, has

    def _load_lesion_roi(self, pid, ct_vol):
        """Task010 mask 质心定位 → 从 CT 卷中心裁剪 lesion_size³ (含边界上下文)。
        ct_vol: np [96,96,96] 确定性 CT 卷 (或 [1,96,96,96], 自动 squeeze)。
        返回 (roi [1,L,L,L] np.float32, has_mask bool); 无 mask/空 mask → (None, False)。
        结果按 pid 内存缓存 (确定性, 与增强无关)。"""
        if self.lesion_dir is None:
            return None, False
        if pid in self._lesion_cache:
            roi = self._lesion_cache[pid]
            return roi, roi is not None
        import nibabel as nib
        from monai.transforms import Resize
        p = os.path.join(self.lesion_dir, f"{pid}.nii.gz")
        roi = None
        has = False
        if os.path.exists(p):
            try:
                m = nib.load(p).get_fdata()
            except Exception:
                m = None
            if m is not None and float((m > 0).sum()) > 0:
                L = self.lesion_size
                m3 = Resize(spatial_size=self.spatial_size, mode="nearest")(
                    m[None].astype(np.float32))[0]  # [96,96,96]
                idx = np.argwhere(m3 > 0)
                if idx.size == 0:          # 重采样后空 mask (极小/全背景) → 无病灶
                    self._lesion_cache[pid] = None
                    return None, False
                cen = idx.mean(axis=0).astype(int)
                half = L // 2
                ct = ct_vol[0] if ct_vol.ndim == 4 else ct_vol  # [96,96,96]
                roi = np.zeros((L, L, L), dtype=np.float32)
                z0, z1 = max(0, cen[0] - half), min(96, cen[0] + half)
                y0, y1 = max(0, cen[1] - half), min(96, cen[1] + half)
                x0, x1 = max(0, cen[2] - half), min(96, cen[2] + half)
                if z1 <= z0 or y1 <= y0 or x1 <= x0:   # 质心异常(越界/nan) → 无病灶
                    self._lesion_cache[pid] = None
                    return None, False
                src = ct[z0:z1, y0:y1, x0:x1]
                oz0 = max(0, half - cen[0]); oy0 = max(0, half - cen[1]); ox0 = max(0, half - cen[2])
                roi[oz0:oz0 + (z1 - z0), oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)] = src
                roi = roi[None]  # [1,L,L,L]
                has = True
        self._lesion_cache[pid] = roi
        return roi, has

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ct_path = self._pick_ct_path(row)
        ct_avail = isinstance(ct_path, str) and bool(ct_path) and os.path.exists(ct_path)
        # round5: MRI 三序列路径 + 逐序列可用性（兼容旧单列 mri_path/mri_available）
        t_path = row.get("mri_path_t", row.get("mri_path", ""))
        c_path = row.get("mri_path_c", "")
        d_path = row.get("mri_path_d", "")
        seq_avail = [
            bool(row.get("mri_available_t", True)) and isinstance(t_path, str) and os.path.exists(t_path),
            bool(row.get("mri_available_c", True)) and isinstance(c_path, str) and os.path.exists(c_path),
            bool(row.get("mri_available_d", True)) and isinstance(d_path, str) and os.path.exists(d_path),
        ]
        mri_avail = any(seq_avail)
        # round7: 序列子集可用性（顺序与网络 mri_sequences 对齐）
        # 注: mri 卷保持全量 [3, D, H, W] 输出（不切片），子集通道由网络按 _SEQ_ORDER 索引提取，
        #     避免 [3,D,H,W] 与 [B,n_seq,D,H,W] 维度语义混淆。
        sub_avail = [seq_avail[i] for i in self._mri_idx]
        mri_avail_out = any(sub_avail)

        # round9 F1: 病灶分支输入 —— Task010 mask 质心从 CT 卷裁剪 (无 mask 零卷)
        lesion_roi = None
        lesion_has = False
        # round9 MRI-t ROI: total_mr t 结肠 mask 质心裁剪 (t 通道 = 缓存卷索引 0)
        mri_lesion_roi = None
        mri_lesion_has = False
        # 确定性预处理（内存缓存优先, 其次磁盘缓存, 最后实时计算并落盘）
        if self._cache is not None and idx in self._cache:
            ct, mri = self._cache[idx]
            if self.mri_lesion_dir is not None and mri_avail and mri is not None:
                mri_lesion_roi, mri_lesion_has = self._load_mri_lesion_roi(
                    str(row.get("patient_id", "")), np.ascontiguousarray(mri.cpu().numpy()),
                    ring=(self.mri_ring_dir is not None))
        else:
            ct = self._zero_volume() if not ct_avail else None
            mri = self._zero_mri() if not mri_avail else None
            if ct_avail:
                if self.disk is not None:
                    cached, hit = self.disk.get(ct_path)
                    ct = cached if hit else None
                if ct is None:
                    ct = self._load_det(ct_path, self.ct_det)
                    if self.disk is not None:
                        self.disk.put(ct_path, ct)
            if ct_avail and ct is not None:
                lesion_roi, lesion_has = self._load_lesion_roi(str(row.get("patient_id", "")), ct)
            if mri_avail:
                mri_key = _det_cache_key_mri(t_path, c_path, d_path, self.spatial_size)
                if self.disk is not None:
                    cached, hit = self.disk.get_vol(mri_key)
                    mri = cached if hit else None
                if mri is None:
                    mri = torch.as_tensor(
                        np.ascontiguousarray(_mri_seqs_deterministic(
                            t_path, c_path, d_path, seq_avail, self.spatial_size)),
                        dtype=torch.float32,
                    )
                    if self.disk is not None:
                        self.disk.put_vol(mri_key, mri)
                # round7: 磁盘缓存即全量 [3,D,H,W] 卷；子集通道由网络按 _SEQ_ORDER 索引提取，无需切片
            # round9 MRI-t ROI: total_mr t 结肠 mask 质心裁剪 (t 通道 = 缓存卷索引 0)
            if self.mri_lesion_dir is not None and mri_avail and mri is not None:
                mri_lesion_roi, mri_lesion_has = self._load_mri_lesion_roi(
                    str(row.get("patient_id", "")), np.ascontiguousarray(mri.cpu().numpy()),
                    ring=(self.mri_ring_dir is not None))
            if self._cache is not None:
                self._cache[idx] = (ct, mri)

        # 随机增强（训练时每次现做，不缓存）
        mri_second = ct_second = None
        if self.augment is not None:
            if ct_avail:
                ct_pre = ct
                ct = torch.as_tensor(np.ascontiguousarray(self.augment(ct_pre)), dtype=torch.float32)
                if self.consist_modal == "ct":
                    ct_second = torch.as_tensor(
                        np.ascontiguousarray(self.augment(ct_pre)), dtype=torch.float32)
            elif self.consist_modal == "ct":
                ct_second = ct  # 不可用样本: 零卷副本 (trainer 端由 ct_mask 排除)
            if mri_avail_out:
                mri_pre = mri
                mri = torch.as_tensor(np.ascontiguousarray(self.augment(mri_pre)), dtype=torch.float32)
                if self.consist_modal == "mri":
                    mri_second = torch.as_tensor(
                        np.ascontiguousarray(self.augment(mri_pre)), dtype=torch.float32)
            elif self.consist_modal == "mri":
                mri_second = mri  # 不可用样本: 零卷副本

        # round9 F1: 病灶分支输出 (无 mask/无 CT → 零卷 [1,L,L,L] + 标志 False)
        if self.lesion_dir is not None:
            if lesion_has and lesion_roi is not None:
                ct_lesion = torch.as_tensor(np.ascontiguousarray(lesion_roi), dtype=torch.float32)
                lesion_available = True
            else:
                L = self.lesion_size
                ct_lesion = torch.zeros(1, L, L, L, dtype=torch.float32)
                lesion_available = False
        else:
            ct_lesion = None
            lesion_available = False

        # round9 MRI-t ROI 分支输出 (无 mask/无 MRI → 零卷 [1,L,L,L] + 标志 False)
        if self.mri_lesion_dir is not None:
            if mri_lesion_has and mri_lesion_roi is not None:
                mri_lesion = torch.as_tensor(np.ascontiguousarray(mri_lesion_roi), dtype=torch.float32)
                mri_lesion_available = True
            else:
                L = self.mri_lesion_size
                mri_lesion = torch.zeros(1, L, L, L, dtype=torch.float32)
                mri_lesion_available = False
        else:
            mri_lesion = None
            mri_lesion_available = False

        return {
            "ct": ct,
            "mri": mri,
            "clinical": self._clinical_vec(row),
            "label": torch.tensor(self._label(row), dtype=torch.long),
            "label_secondary": torch.tensor(self._secondary_label(row), dtype=torch.long),
            "ct_available": torch.tensor(ct_avail, dtype=torch.bool),
            "mri_available": torch.tensor(mri_avail_out, dtype=torch.bool),
            "mri_seq_available": torch.tensor(sub_avail, dtype=torch.bool),
            "patient_id": str(row.get("patient_id", idx)),
            "ct_second": ct if ct_second is None else ct_second,
            "mri_second": mri if mri_second is None else mri_second,
            "ct_lesion": ct_lesion,
            "lesion_available": torch.tensor(lesion_available, dtype=torch.bool),
            "mri_lesion": mri_lesion,
            "mri_lesion_available": torch.tensor(mri_lesion_available, dtype=torch.bool),
        }


def build_dataloaders(cfg, train_csv=None, val_csv=None):
    disk_cache_dir = getattr(cfg, "disk_cache_dir", None) or "./cache_hos"
    mri_seqs = tuple(getattr(cfg, "mri_sequences", ("t", "c", "d")))
    ct_seq = getattr(cfg, "ct_seq", "auto")
    use_aug = getattr(cfg, "use_augmentation", True)  # round8: 训练增强开关
    consist_modal = getattr(cfg, "consist_modal", None)  # round9 C2: 一致性正则模态
    lesion_dir = getattr(cfg, "lesion_dir", None)  # round9 F1: 病灶分支 mask 目录
    lesion_size = getattr(cfg, "lesion_size", 64)
    mri_lesion_dir = getattr(cfg, "mri_lesion_dir", None)  # round9 MRI-t ROI: total_mr t 结肠 mask
    mri_lesion_size = getattr(cfg, "mri_lesion_size", 64)
    mri_ring_dir = getattr(cfg, "mri_ring_dir", None)  # Day1⑤: ring mask 开关 (膨胀 2 层取环)
    train_ds = CRCMultiModalDataset(
        train_csv or cfg.train_csv, label_col=cfg.label_col, spatial_size=cfg.spatial_size,
        num_clinical=cfg.num_clinical, clinical_cols=getattr(cfg, "clinical_cols", None),
        multi_task=getattr(cfg, "multi_task", False),
        secondary_label_col=getattr(cfg, "secondary_label_col", "label_N"),
        is_train=True, disk_cache_dir=disk_cache_dir,
        mri_seqs=mri_seqs, ct_seq=ct_seq, use_aug=use_aug,
        consist_modal=consist_modal,
        lesion_dir=lesion_dir, lesion_size=lesion_size,
        mri_lesion_dir=mri_lesion_dir, mri_lesion_size=mri_lesion_size,
        mri_ring_dir=mri_ring_dir,
    )
    val_ds = CRCMultiModalDataset(
        val_csv or cfg.val_csv, label_col=cfg.label_col, spatial_size=cfg.spatial_size,
        num_clinical=cfg.num_clinical, clinical_cols=getattr(cfg, "clinical_cols", None),
        multi_task=getattr(cfg, "multi_task", False),
        secondary_label_col=getattr(cfg, "secondary_label_col", "label_N"),
        is_train=False, disk_cache_dir=disk_cache_dir,
        mri_seqs=mri_seqs, ct_seq=ct_seq,
        lesion_dir=lesion_dir, lesion_size=lesion_size,
        mri_lesion_dir=mri_lesion_dir, mri_lesion_size=mri_lesion_size,
        mri_ring_dir=mri_ring_dir,
    )
    num_workers = getattr(cfg, "num_workers", 0)
    pin = getattr(cfg, "pin_memory", False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader
