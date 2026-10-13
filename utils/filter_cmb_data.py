"""
筛选 data/cmb 文件夹下的数据：
删除直接包含的 .dcm 文件个数 <= 10 的文件夹（不计入子文件夹中的 .dcm 文件）
"""

import os
import shutil
from pathlib import Path


def count_dcm_files(folder: Path) -> int:
    """统计文件夹中直接包含的 .dcm 文件数量（不递归子文件夹）"""
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".dcm")


def filter_cmb_data(cmb_root: str, threshold: int = 10, dry_run: bool = True):
    """
    遍历 cmb_root 下所有文件夹，删除直接包含 .dcm 文件数 <= threshold 的文件夹。

    Args:
        cmb_root: data/cmb 文件夹的路径
        threshold: .dcm 文件数量阈值，<= 该值的文件夹将被删除
        dry_run: 若为 True，仅打印将被删除的文件夹，不执行实际删除
    """
    cmb_path = Path(cmb_root)
    if not cmb_path.exists():
        print(f"路径不存在: {cmb_root}")
        return

    deleted_count = 0
    kept_count = 0

    for dirpath, dirnames, filenames in os.walk(cmb_path, topdown=False):
        current_dir = Path(dirpath)
        dcm_count = sum(1 for f in filenames if f.lower().endswith(".dcm"))

        if dcm_count == 0:
            continue

        if dcm_count <= threshold:
            if dry_run:
                print(f"[将删除] {current_dir} (含 {dcm_count} 个 .dcm 文件)")
            else:
                shutil.rmtree(current_dir)
                print(f"[已删除] {current_dir} (含 {dcm_count} 个 .dcm 文件)")
            deleted_count += 1
        else:
            kept_count += 1

    print(f"\n统计: 删除 {deleted_count} 个文件夹, 保留 {kept_count} 个文件夹")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    cmb_root = project_root / "data" / "cmb"

    print("=" * 60)
    print("CMB 数据筛选工具")
    print(f"目标路径: {cmb_root}")
    print(f"规则: 删除直接包含 .dcm 文件数 <= 10 的文件夹")
    print("=" * 60)

    print("\n[预览模式] 以下文件夹将被删除:\n")
    filter_cmb_data(str(cmb_root), threshold=10, dry_run=True)

    confirm = input("\n确认执行删除操作? (输入 yes 确认): ")
    if confirm.strip().lower() == "yes":
        filter_cmb_data(str(cmb_root), threshold=10, dry_run=False)
        print("\n操作完成。")
    else:
        print("\n已取消操作。")
