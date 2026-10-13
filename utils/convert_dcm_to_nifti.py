"""
将 TCGA DICOM 转换为 NIfTI (.nii.gz)。

逐患者遍历，按 SeriesInstanceUID 分组，每个 series 输出一个 nii.gz。
内置处理同一 series 内切片尺寸不一致的情况（TCGA 常见：主体积与定位片
或不同重建混在同一 UID）：按 (Rows, Cols) 再分组，各自转换。

输入:
  data/tcga/tcga_coad/<patient>/.../*.dcm
  data/tcga/tcga_read/<patient>/.../*.dcm
输出:
  data/tcga/tcga_coad/_nifti/<patient>/<patient>_<idx>_<uid8>.nii.gz
  data/tcga/tcga_read/_nifti/<patient>/...

依赖: SimpleITK, pydicom
用法:
  python utils/convert_dcm_to_nifti.py            # 转换全部队列
  python utils/convert_dcm_to_nifti.py --cohort tcga_coad
"""

import os
import glob
import argparse
import SimpleITK as sitk


def _size_split_convert(reader, dcm_dir, sid, out_dir, patient, idx):
    """处理同一 series 内尺寸不一致：按 (Rows, Cols) 分组分别写出。返回写出数。"""
    import pydicom
    files = reader.GetGDCMSeriesFileNames(dcm_dir, sid)
    groups = {}
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        groups.setdefault((int(ds.Rows), int(ds.Columns)), []).append(f)

    written = 0
    # 按组大小降序：主体积优先拿基础 idx
    for gi, (key, gfiles) in enumerate(
            sorted(groups.items(), key=lambda x: -len(x[1]))):
        def _z(fp):
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
            if "ImagePositionPatient" in ds:
                return float(ds.ImagePositionPatient[2])
            return int(getattr(ds, "InstanceNumber", 0))
        gfiles = sorted(gfiles, key=_z)
        reader.SetFileNames(gfiles)
        img = reader.Execute()
        suffix = f"_{key[0]}x{key[1]}" if len(groups) > 1 else ""
        name = f"{patient}_{idx:02d}{suffix}_{sid[-8:]}.nii.gz"
        sitk.WriteImage(img, os.path.join(out_dir, name))
        written += 1
    return written


def convert_patient(patient_dir, out_dir):
    reader = sitk.ImageSeriesReader()
    dcm_dirs = set()
    for root, _, files in os.walk(patient_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            dcm_dirs.add(root)

    idx, ok = 0, 0
    for d in sorted(dcm_dirs):
        for sid in reader.GetGDCMSeriesIDs(d):
            files = reader.GetGDCMSeriesFileNames(d, sid)
            if not files:
                continue
            idx += 1
            reader.SetFileNames(files)
            try:
                img = reader.Execute()
                name = f"{os.path.basename(patient_dir)}_{idx:02d}_{sid[-8:]}.nii.gz"
                sitk.WriteImage(img, os.path.join(out_dir, name))
                ok += 1
            except Exception:
                # 尺寸不一致等异常：按尺寸分组重试
                ok += _size_split_convert(reader, d, sid, out_dir,
                                          os.path.basename(patient_dir), idx)
    return ok


def convert_cohort(base, cohort):
    cohort_dir = os.path.join(base, "data", "tcga", cohort)
    if not os.path.isdir(cohort_dir):
        print(f"跳过（不存在）: {cohort_dir}")
        return
    for p in sorted(os.listdir(cohort_dir)):
        pdir = os.path.join(cohort_dir, p)
        if not os.path.isdir(pdir) or p.startswith("_"):
            continue
        out_dir = os.path.join(cohort_dir, "_nifti", p)
        os.makedirs(out_dir, exist_ok=True)
        n = convert_patient(pdir, out_dir)
        print(f"{cohort}/{p} -> {n} series")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None,
                    help="只转换指定队列 (tcga_coad / tcga_read)，默认全部")
    ap.add_argument("--base", default=None, help="项目根目录（默认自动推断）")
    args = ap.parse_args()

    base = args.base or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cohorts = [args.cohort] if args.cohort else ["tcga_coad", "tcga_read"]
    for c in cohorts:
        convert_cohort(base, c)
    print("DONE")


if __name__ == "__main__":
    main()
