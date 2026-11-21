# -*- coding: utf-8 -*-
"""住院号(PatientID) 校验工具：只读 DICOM 头 + 完整版收集表，供构建与校验脚本复用。"""
import os
import re
import struct

import pandas as pd

HOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Hos")
FULL_XLSX = os.path.join(HOS, "结直肠癌多模态患者数据收集表_完整版.xlsx")
OLD_XLSX = os.path.join(HOS, "结直肠癌多模态患者数据收集表.xlsx")

# 已核实的旧住院号（影像侧号码 -> 表内编号）
KNOWN_OLD_IDS = {"10900337": 97}

WANT_PID = (0x0010, 0x0020)
LONG_VR = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"}


def _read_elem(data, i, explicit, endian="<"):
    g, e = struct.unpack(endian + "HH", data[i:i + 4])
    if explicit:
        vr = data[i + 4:i + 6]
        if vr in LONG_VR:
            ln = struct.unpack(endian + "I", data[i + 8:i + 12])[0]
            return (g, e), i + 12, ln
        ln = struct.unpack(endian + "H", data[i + 6:i + 8])[0]
        return (g, e), i + 8, ln
    ln = struct.unpack(endian + "I", data[i + 4:i + 8])[0]
    return (g, e), i + 8, ln


def dicom_patient_id(path):
    """读取单个 DICOM 的 PatientID(住院号)，失败返回 ""。"""
    try:
        with open(path, "rb") as f:
            data = f.read(200000)
    except OSError:
        return ""
    i = 132 if data[128:132] == b"DICM" else 0
    ts = ""
    while i + 8 <= len(data):
        tag, vstart, ln = _read_elem(data, i, True)
        if ln == 0xFFFFFFFF:
            return ""
        if tag[0] != 0x0002:
            break
        if tag == (0x0002, 0x0010):
            ts = data[vstart:vstart + ln].decode("latin-1", "ignore").strip("\x00 ")
        i = vstart + ln
    explicit = ts != "1.2.840.10008.1.2"
    endian = ">" if ts == "1.2.840.10008.1.2.2" else "<"
    n = 0
    while i + 8 <= len(data) and n < 300:
        tag, vstart, ln = _read_elem(data, i, explicit, endian)
        if ln == 0xFFFFFFFF or vstart + ln > len(data):
            break
        if tag == WANT_PID:
            return data[vstart:vstart + ln].decode("latin-1", "ignore").strip("\x00 ")
        if tag == (0x7FE0, 0x0010):
            break
        i = vstart + ln
        n += 1
    return ""


def first_dicom(d):
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f.lower().endswith(".dcm"):
                p = os.path.join(root, f)
                if os.path.getsize(p) > 0:
                    return p
    return ""


def patient_id_of_dir(d):
    f = first_dicom(d)
    return dicom_patient_id(f) if f else ""


def norm(x):
    return str(x or "").strip().upper().replace(" ", "")


def load_table_identity():
    """返回 {编号: {"hospital_id","name","imaging"}}；优先完整版表。"""
    path = FULL_XLSX if os.path.exists(FULL_XLSX) else OLD_XLSX
    df = pd.read_excel(path, dtype=str)
    cols = set(df.columns)
    out = {}
    has_num = "编号" in cols
    for idx, (_, r) in enumerate(df.iterrows(), start=1):
        if has_num:
            try:
                num = int(float(str(r["编号"]).strip()))
            except (ValueError, TypeError):
                continue
        else:
            num = idx  # 完整版表无编号列，行序即编号（已逐行核验）
        out[num] = {
            "hospital_id": norm(r["住院号"]) if "住院号" in cols else "",
            "name": str(r.get("姓名", "")).strip() if "姓名" in cols else "",
            "imaging": str(r.get("影像信息", "")).strip() if "影像信息" in cols else "",
        }
    return out, path


def classify(ct_pid, mri_pid, hospital_id):
    """配对状态: ok / ok_old_id / typo_table / identity_conflict / unverified / no_table_id。"""
    c, m, h = norm(ct_pid), norm(mri_pid), norm(hospital_id)
    ids = {x for x in (c, m) if x}
    if not h:
        return "no_table_id"
    if not ids:
        return "no_image_id"
    if c and m and c != m:
        return "identity_conflict"
    vid = next(iter(ids))
    if vid == h:
        return "ok"
    if vid in KNOWN_OLD_IDS:
        return "ok_old_id"
    return "typo_table"
