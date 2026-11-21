# -*- coding: utf-8 -*-
"""直接读取编号 97 的 DICOM 头 + 表 97 行，核对身份。只读。"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work")
import dicom_meta as dm  # noqa: E402

dm.WANT.update({
    (0x0010, 0x0030): "PatientBirthDate",
    (0x0010, 0x0040): "PatientSex",
    (0x0010, 0x1010): "PatientAge",
    (0x0018, 0x0015): "BodyPartExamined",
    (0x0008, 0x0081): "InstitutionName",
})

for d in ("97_p", "97_z"):
    files = sorted(glob.glob(rf"d:\PyFile\ColonMultiModal\data\Hos\CT\{d}\**\*.dcm", recursive=True))
    print(f"\n[{d}] dcm={len(files)}")
    for f in (files[0], files[len(files) // 2], files[-1]):
        m = dm.parse(f)
        print(f"  {os.path.basename(f)} -> ID={m.get('PatientID')} Name={m.get('PatientName')} "
              f"Sex={m.get('PatientSex')} Birth={m.get('PatientBirthDate')} Age={m.get('PatientAge')} "
              f"Date={m.get('StudyDate')} Mod={m.get('Modality')} Desc={m.get('SeriesDescription')} "
              f"BodyPart={m.get('BodyPartExamined')} Rows={m.get('Rows')}x{m.get('Columns')}")

for f in ("97_p.nii.gz", "97_z.nii.gz"):
    p = rf"d:\PyFile\ColonMultiModal\data\Hos\_nifti\CT\{f}"
    print(f"\n{f} exists={os.path.exists(p)} size={os.path.getsize(p) if os.path.exists(p) else '-'}")

df = pd.read_excel(r"c:\Users\wzy\.trae-cn\attachments\cc6db7db-c2b1-4a67-af45-b67546fc8f46_e67f5806-9202-4a5f-a975-aca895171bb4_副本结直肠癌多模态患者数据收集表.xlsx", dtype=str)
df.insert(0, "编号", range(1, len(df) + 1))
row = df[df["编号"] == 97]
print("\n[收集表 编号97]")
print(row[["编号", "姓名", "住院号", "病案号", "性别", "年龄", "手术日期", "影像信息", "影像学T", "影像学N", "影像学M", "病理T", "病理N", "病理M"]].to_string(index=False))
