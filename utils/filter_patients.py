"""
筛选 TCGA 患者文件夹：删除递归文件数不足阈值的文件夹（默认 < 10 个 dcm）。

用途:
  从 IDC/TCIA 下载的 TCGA 数据，个别患者只有零星几个文件（定位片或下载不全），
  不足以构成有效影像，训练前先剔除。

用法:
  python utils/filter_patients.py                 # 预览（dry-run，不删除）
  python utils/filter_patients.py --apply          # 实际删除
  python utils/filter_patients.py --min 10 --apply # 自定义阈值
  python utils/filter_patients.py --root data/tcga_new --apply  # 指定其他数据根

默认扫描:
  <root>/tcga_coad/*  和  <root>/tcga_read/*   (root 默认 data/tcga)
"""

import os
import glob
import shutil
import argparse

MIN_FILES = 10   # 患者文件夹内递归文件数下限（含此值保留）
EXT = ".dcm"     # 统计的文件扩展名


def _count_files(folder, ext=EXT):
    n = 0
    for _, _, files in os.walk(folder):
        n += sum(1 for f in files if f.lower().endswith(ext))
    return n


def filter_cohort(cohort_dir, min_files, apply):
    """扫描一个队列目录下的患者文件夹，返回 (保留列表, 待删列表)。"""
    keep, drop = [], []
    for pdir in sorted(glob.glob(os.path.join(cohort_dir, "*"))):
        if not os.path.isdir(pdir):
            continue
        # 跳过转换输出目录
        if os.path.basename(pdir).startswith("_"):
            continue
        n = _count_files(pdir)
        (keep if n >= min_files else drop).append((pdir, n))
    return keep, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=MIN_FILES,
                    help=f"最小文件数，低于此值删除（默认 {MIN_FILES}）")
    ap.add_argument("--apply", action="store_true",
                    help="实际执行删除；不加则仅预览")
    ap.add_argument("--base", default=None, help="项目根目录（默认自动推断）")
    ap.add_argument("--root", default=None,
                    help="数据根目录，含 tcga_coad/tcga_read（默认 <base>/data/tcga）")
    args = ap.parse_args()

    base = args.base or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = args.root if args.root else os.path.join(base, "data", "tcga")
    if not os.path.isabs(root):
        root = os.path.join(base, root)
    cohorts = [
        os.path.join(root, "tcga_coad"),
        os.path.join(root, "tcga_read"),
    ]

    total_drop = 0
    for cohort_dir in cohorts:
        if not os.path.isdir(cohort_dir):
            continue
        keep, drop = filter_cohort(cohort_dir, args.min, args.apply)
        print(f"\n[{os.path.basename(cohort_dir)}] 保留 {len(keep)} / 删除 {len(drop)}")
        for pdir, n in drop:
            print(f"  {'删除' if args.apply else '待删'}: {os.path.basename(pdir)}  ({n} 个 {EXT})")
            if args.apply:
                shutil.rmtree(pdir)
            total_drop += 1

    mode = "已删除" if args.apply else "预览(未删除，加 --apply 执行)"
    print(f"\n{mode}: 共 {total_drop} 个文件夹低于 {args.min} 个 {EXT}")


if __name__ == "__main__":
    main()
