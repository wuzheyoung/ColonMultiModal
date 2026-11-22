# -*- coding: utf-8 -*-
"""笔记本环境一键自检（只读，不训练/不推理）。

用法（在迁移包根目录或 code/ 下均可）：
    python code/verify_laptop_env.py

检查项：
  1. Python 版本与核心库版本是否与 4060 对齐
  2. torch.cuda 是否可用、显卡型号/显存/算力是否满足 GTX 1650
  3. 能否正常读取 data/master_5fold_356.csv（编码/反序列化）
  4. 能否正常读取 features/ 下的 1 个 .npy 特征与 1 个 .csv 特征表
"""
import os
import sys
import glob
import platform

EXPECTED_PY = "3.10"
EXPECTED = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.14.1",
    "sklearn": "1.5.2",
    "matplotlib": "3.10.9",
    "seaborn": "0.13.2",
    "nibabel": "5.2.1",
    "SimpleITK": "2.5.8",
    "timm": "1.0.22",
    "tqdm": "4.70.1",
    "torch": "2.5.1+cu124",
    "torchvision": "0.20.1+cu124",
}

FAIL = []
WARN = []


def ok(msg):
    print("[PASS] " + msg)


def warn(msg):
    WARN.append(msg)
    print("[WARN] " + msg)


def fail(msg):
    FAIL.append(msg)
    print("[FAIL] " + msg)


def find_root():
    """向上查找包含 data/master_5fold_356.csv 的迁移包根目录。"""
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isfile(os.path.join(cur, "data", "master_5fold_356.csv")):
            return cur
        cur = os.path.dirname(cur)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_python():
    v = sys.version_info
    print("Python: %s (%s)" % (platform.python_version(), sys.executable))
    if "%d.%d" % (v.major, v.minor) == EXPECTED_PY:
        ok("Python minor 版本与 4060 对齐 (%s)" % EXPECTED_PY)
    else:
        fail("Python 版本为 %d.%d，4060 为 %s，请改用 3.10.x" % (v.major, v.minor, EXPECTED_PY))


def check_packages():
    import importlib
    for mod, want in EXPECTED.items():
        try:
            m = importlib.import_module(mod)
            got = getattr(m, "__version__", "unknown")
        except Exception as e:
            fail("无法导入 %s: %s" % (mod, e))
            continue
        if got == want or got.startswith(want.split("+")[0]):
            ok("%s==%s" % (mod, got))
        else:
            warn("%s 版本 %s，4060 为 %s" % (mod, got, want))


def check_cuda():
    try:
        import torch
    except Exception as e:
        fail("torch 导入失败: %s" % e)
        return
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() = False，请检查 NVIDIA 驱动 / CUDA wheel")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024 ** 3
    print("GPU: %s | sm_%d%d | VRAM %.2f GB | torch %s | cuda %s"
          % (name, cap[0], cap[1], vram, torch.__version__, torch.version.cuda))
    ok("CUDA 可用")
    if cap[0] < 7:
        fail("算力 sm_%d%d 过低，GTX 1650 应为 sm_75" % cap)
    elif cap[0] == 7 and cap[1] < 5:
        warn("算力 sm_%d%d 低于 GTX 1650 (sm_75)" % cap)
    else:
        ok("算力满足 GTX 1650 (sm_75) 及以上")
    if "1650" in name:
        ok("已识别 GTX 1650")
    else:
        warn("当前显卡不是 GTX 1650：%s（不影响自检）" % name)
    if props.major >= 8 and props.minor >= 0:
        warn("该卡支持 bf16；但项目代码统一走 float16 autocast，勿在 1650 上启用 bf16")
    if vram < 3.5:
        warn("显存 %.2f GB 低于预期 4GB" % vram)


def read_csv_robust(path):
    import pandas as pd
    last = None
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as e:
            last = e
    raise last


def check_tables(root):
    import pandas as pd
    master = os.path.join(root, "data", "master_5fold_356.csv")
    if not os.path.isfile(master):
        fail("缺少 %s" % master)
    else:
        try:
            df, enc = read_csv_robust(master)
            need = {"patient_id", "fold", "mri_t", "mri_c", "mri_d"}
            missing = need - set(df.columns)
            if missing:
                fail("master 表缺列: %s" % sorted(missing))
            else:
                ok("master_5fold_356.csv 读取成功: %d 行 x %d 列 (encoding=%s)"
                   % (df.shape[0], df.shape[1], enc))
        except Exception as e:
            fail("master_5fold_356.csv 读取失败: %s" % e)

    csvs = sorted(glob.glob(os.path.join(root, "features", "**", "*.csv"), recursive=True))
    if not csvs:
        warn("features/ 下未找到 csv 特征表（若仅同步 npy 特征可忽略）")
    else:
        try:
            df = pd.read_csv(csvs[0], encoding="utf-8")
            ok("特征表读取成功: %s (%d 行 x %d 列)"
               % (os.path.relpath(csvs[0], root), df.shape[0], df.shape[1]))
        except Exception as e:
            fail("特征表读取失败 %s: %s" % (csvs[0], e))

    npys = sorted(glob.glob(os.path.join(root, "features", "**", "*.npy"), recursive=True))
    if not npys:
        warn("features/ 下未找到 .npy 特征")
    else:
        try:
            import numpy as np
            arr = np.load(npys[0], allow_pickle=False)
            ok(".npy 特征读取成功: %s shape=%s dtype=%s"
               % (os.path.relpath(npys[0], root), arr.shape, arr.dtype))
        except Exception as e:
            fail(".npy 反序列化失败 %s: %s" % (npys[0], e))


def main():
    print("=" * 70)
    print(" ColonMultiModal 笔记本环境自检")
    print("=" * 70)
    root = find_root()
    print("迁移包根目录: %s" % root)
    check_python()
    check_packages()
    check_cuda()
    check_tables(root)
    print("-" * 70)
    if FAIL:
        print("结果: FAIL (%d 项失败, %d 项警告)" % (len(FAIL), len(WARN)))
        for m in FAIL:
            print("  - " + m)
        return 1
    print("结果: PASS (%d 项警告)" % len(WARN))
    for m in WARN:
        print("  - " + m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
