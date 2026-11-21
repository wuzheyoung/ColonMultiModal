# -*- coding: utf-8 -*-
"""
第二批 DICOM -> NIfTI 转换。

目录结构:
  data/Hos/第二批/{1-100,101-200,201-}/<目录名>/*.dcm
  <目录名> = {段内编号}{序列后缀}, 序列后缀语义(用户确认):
    t  = T2(矢状位)          -> 输出后缀 t
    d  = DWI(轴位)           -> 输出后缀 d
    +  = 增强 MRI (LAVA +C)  -> 输出后缀 c (对齐第一批增强 MRI)
    c  = 平扫 CT (DE_Abd_NC) -> 输出后缀 p (对齐第一批平扫 CT)
    c+ = 增强 CT (DE_Abd_A)  -> 输出后缀 z (对齐第一批增强 CT)

编号偏移(已用 DICOM 住院号与 xlsx 验证):
    1-100   段: 实际编号 = 目录编号
    101-200 段: 实际编号 = 目录编号 + 100
    201-    段: 实际编号 = 目录编号 + 200

异常处理:
    - 纯数字目录(仅 39) 视为 T2 (t)
    - 空目录(无 .dcm) 跳过
    - 缺 DICOM 头文件 用 glob 收集 *.dcm 直接读, 失败则记录

输出:
  data/Hos/_nifti/batch2/{实际编号}_{后缀}.nii.gz

用法:
  python utils/convert_batch2_to_nifti.py --dry-run     # 只打印计划
  python utils/convert_batch2_to_nifti.py               # 实际转换
"""
import os
import re
import sys
import glob
import argparse
import warnings

warnings.simplefilter("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import SimpleITK as sitk

# 关闭 ITK 警告输出到 stderr，避免 PowerShell 将其当作错误中断
sitk.ProcessObject.SetGlobalWarningDisplay(False)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B2 = os.path.join(BASE, "data", "Hos", "第二批")
OUT = os.path.join(BASE, "data", "Hos", "_nifti", "batch2")

SEG_OFFSET = {"1-100": 0, "101-200": 100, "201-": 200}
# 第二批后缀 -> 第一批语义后缀 (t/d 不变, + -> c, c -> p, c+ -> z)
SEQ_MAP = {"t": "t", "d": "d", "+": "c", "c": "p", "c+": "z"}
NAME_PAT = re.compile(r"^(\d+)(c\+|c|\+|d|t)$")


def is_valid_gzip(path):
    import gzip
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def convert_series(dcm_dir, out_path):
    """单个序列目录 -> NIfTI。返回 (ok, err)。"""
    reader = sitk.ImageSeriesReader()
    sids = reader.GetGDCMSeriesIDs(dcm_dir)
    if not sids:
        # 缺 DICM 头等: 手动收集 *.dcm 直接读
        files = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
        if not files:
            return False, "no_dcm"
        reader.SetFileNames(files)
    else:
        files = reader.GetGDCMSeriesFileNames(dcm_dir, sids[0])
        if not files:
            return False, "no_files"
        reader.SetFileNames(files)
    try:
        img = reader.Execute()
        sitk.WriteImage(img, out_path)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}:{e}"


def build_plan():
    """返回 [(seg, dirname, actual_id, src_seq, out_suffix, out_path)]"""
    plan = []
    for seg, off in SEG_OFFSET.items():
        segdir = os.path.join(B2, seg)
        if not os.path.isdir(segdir):
            continue
        for name in sorted(os.listdir(segdir)):
            d = os.path.join(segdir, name)
            if not os.path.isdir(d):
                continue
            # 空目录
            if not glob.glob(os.path.join(d, "*.dcm")):
                print(f"[skip-empty] {seg}/{name}")
                continue
            m = NAME_PAT.match(name)
            if m:
                aid = int(m.group(1)) + off
                src_seq = m.group(2)
            elif re.match(r"^\d+$", name):
                # 纯数字 -> 归为 T2
                aid = int(name) + off
                src_seq = "t"
                print(f"[special] {seg}/{name} -> 纯数字, 归为 t")
            else:
                print(f"[skip-unknown] {seg}/{name} 无法解析命名")
                continue
            out_suffix = SEQ_MAP[src_seq]
            out_path = os.path.join(OUT, f"{aid}_{out_suffix}.nii.gz")
            plan.append((seg, name, aid, src_seq, out_suffix, out_path))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    plan = build_plan()
    print(f"计划转换 {len(plan)} 个序列目录 -> {OUT}", flush=True)

    if args.dry_run:
        for seg, name, aid, src, suf, out in plan:
            print(f"  {seg}/{name:12s} -> 编号{aid:3d}  {src:3s}->{suf}  {os.path.basename(out)}")
        return

    done, fail, skip_existing = 0, [], 0
    for seg, name, aid, src, suf, out in plan:
        if os.path.exists(out) and is_valid_gzip(out):
            skip_existing += 1
            continue
        if os.path.exists(out):
            os.remove(out)  # 损坏残留
            print(f"[reconvert] {out}")
        d = os.path.join(B2, seg, name)
        ok, err = convert_series(d, out)
        if ok:
            done += 1
            print(f"[ok] {seg}/{name} -> {os.path.basename(out)}")
        else:
            fail.append((seg, name, err))
            print(f"[FAIL] {seg}/{name} : {err}")

    print("=" * 60)
    print(f"完成: 成功 {done}  跳过(已存在) {skip_existing}  失败 {len(fail)}")
    for seg, name, err in fail:
        print(f"  失败: {seg}/{name}  {err}")


if __name__ == "__main__":
    main()
