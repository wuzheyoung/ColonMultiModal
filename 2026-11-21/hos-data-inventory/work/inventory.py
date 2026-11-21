# -*- coding: utf-8 -*-
"""只读盘点脚本：不移动/复制/删除任何文件，不写 data 目录。
输出：work/inventory.json + work/inventory_summary.txt
"""
import gzip
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

ROOT = r"d:\PyFile\ColonMultiModal\data\Hos"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"

MODS_CT = ("p", "z")
MODS_MR = ("c", "d", "t")


def nifti_header(path):
    """读取 nii.gz 前 352 字节，返回 dim/pixdim；异常返回 None。"""
    try:
        with gzip.open(path, "rb") as f:
            head = f.read(352)
        if len(head) < 348:
            return None
        sizeof_hdr = struct.unpack("<i", head[0:4])[0]
        endian = "<"
        if sizeof_hdr != 348:
            sizeof_hdr = struct.unpack(">i", head[0:4])[0]
            if sizeof_hdr != 348:
                return {"error": "bad_sizeof_hdr"}
            endian = ">"
        dim = struct.unpack(endian + "8h", head[40:56])
        pixdim = struct.unpack(endian + "8f", head[76:108])
        datatype = struct.unpack(endian + "h", head[70:72])[0]
        return {
            "dim": list(dim[1:dim[0] + 1]),
            "pixdim": [round(v, 4) for v in pixdim[:dim[0] + 1]],
            "datatype": datatype,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def scan_raw(side):
    base = os.path.join(ROOT, side)
    dirs = []
    loose = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            files = []
            exts = Counter()
            prefixes = Counter()
            for dp, _, fns in os.walk(p):
                for fn in fns:
                    files.append(os.path.join(dp, fn))
                    exts[os.path.splitext(fn)[1].lower()] += 1
                    if ".dcm" == os.path.splitext(fn)[1].lower():
                        m = re.match(r"^(ser\d+)", fn)
                        if m:
                            prefixes[m.group(1)] += 1
            dirs.append({
                "dir": name,
                "n_files": len(files),
                "ext": dict(exts),
                "ser_prefixes": dict(prefixes),
                "first_files": [os.path.basename(x) for x in files[:3]],
                "sizes": [os.path.getsize(x) for x in files[:1]],
            })
        else:
            loose.append({"name": name, "size": os.path.getsize(p)})
    return {"base": base, "dirs": dirs, "loose": loose}


def scan_nifti(sub):
    base = os.path.join(ROOT, "_nifti", sub)
    out = []
    for fn in sorted(os.listdir(base)):
        p = os.path.join(base, fn)
        if not os.path.isfile(p):
            continue
        m = re.match(r"^(.+?)_([a-z]+)\.nii\.gz$", fn)
        pid, mod = (m.group(1), m.group(2)) if m else (None, None)
        hdr = nifti_header(p)
        out.append({
            "file": fn, "pid": pid, "mod": mod, "size": os.path.getsize(p),
            "mtime": os.path.getmtime(p), "hdr": hdr,
        })
    return out


def dicom_tag_uid(path):
    """从 DICOM 原始字节抓 SeriesInstanceUID / SeriesDescription（explicit VR LE）。"""
    try:
        with open(path, "rb") as f:
            data = f.read(4096)
    except Exception:
        return {}
    res = {}
    for tag, key in ((b"(0020, 000e)", "SeriesInstanceUID"),
                     (b"(0008, 103e)", "SeriesDescription"),
                     (b"(0008, 0060)", "Modality")):
        i = data.find(tag)
        if i < 0:
            continue
        vr = data[i + 11:i + 13]
        if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
            ln = struct.unpack("<I", data[i + 13:i + 17])[0]
            val = data[i + 17:i + 17 + ln]
        else:
            ln = struct.unpack("<H", data[i + 13:i + 15])[0]
            val = data[i + 15:i + 15 + ln]
        res[key] = val.decode("latin-1", "ignore").strip("\x00 ")
    return res


def main():
    os.makedirs(WORK, exist_ok=True)
    rep = {}
    rep["raw_ct"] = scan_raw("CT")
    rep["raw_mr"] = scan_raw("MRI")
    rep["nifti_ct"] = scan_nifti("CT")
    rep["nifti_mr"] = scan_nifti("MRI")
    rep["nifti_b2"] = scan_nifti("batch2")

    # DICOM 元信息抽样：每个原始目录取首文件
    for key in ("raw_ct", "raw_mr"):
        for d in rep[key]["dirs"]:
            base = rep[key]["base"]
            dpath = os.path.join(base, d["dir"])
            firsts = [f for f in sorted(os.listdir(dpath))
                      if f.lower().endswith(".dcm")][:1]
            d["dicom_meta"] = dicom_tag_uid(os.path.join(dpath, firsts[0])) if firsts else {}

    # 患者集合
    rep["patient_sets"] = {
        "raw_ct_dirs": sorted({d["dir"].rsplit("_", 1)[0] for d in rep["raw_ct"]["dirs"]}),
        "raw_mr_dirs": sorted({d["dir"].rsplit("_", 1)[0] for d in rep["raw_mr"]["dirs"]}),
        "nifti_ct": sorted({x["pid"] for x in rep["nifti_ct"] if x["pid"]}),
        "nifti_mr": sorted({x["pid"] for x in rep["nifti_mr"] if x["pid"]}),
        "nifti_b2": sorted({x["pid"] for x in rep["nifti_b2"] if x["pid"]}),
    }

    with open(os.path.join(WORK, "inventory.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)

    ps = rep["patient_sets"]
    lines = []
    for k, v in ps.items():
        lines.append(f"{k}: patients={len(v)}")
    lines.append("")
    for k in ("raw_ct", "raw_mr"):
        lines.append(f"{k}: dirs={len(rep[k]['dirs'])} loose={len(rep[k]['loose'])} "
                     f"files={sum(d['n_files'] for d in rep[k]['dirs'])}")
    for k in ("nifti_ct", "nifti_mr", "nifti_b2"):
        lines.append(f"{k}: files={len(rep[k])} patients={len({x['pid'] for x in rep[k]})}")
    lines.append("")
    # 原始目录 vs nifti
    ct_raw = {d["dir"] for d in rep["raw_ct"]["dirs"]}
    ct_nii = {x["file"][:-7] for x in rep["nifti_ct"]}
    mr_raw = {d["dir"] for d in rep["raw_mr"]["dirs"]}
    mr_nii = {x["file"][:-7] for x in rep["nifti_mr"]}
    b2_nii = {x["file"][:-7] for x in rep["nifti_b2"]}
    lines.append(f"CT  raw_only={len(ct_raw - ct_nii)} nii_only={len(ct_nii - ct_raw)} both={len(ct_raw & ct_nii)}")
    lines.append("  raw_only: " + ",".join(sorted(ct_raw - ct_nii)))
    lines.append("  nii_only: " + ",".join(sorted(ct_nii - ct_raw)))
    lines.append(f"MRI raw_only={len(mr_raw - mr_nii)} nii_only={len(mr_nii - mr_raw)} both={len(mr_raw & mr_nii)}")
    lines.append("  raw_only: " + ",".join(sorted(mr_raw - mr_nii)))
    lines.append("  nii_only: " + ",".join(sorted(mr_nii - mr_raw)))
    lines.append(f"b2 overlap with nifti CT/MRI keys: {len(b2_nii & (ct_nii | mr_nii))}")
    with open(os.path.join(WORK, "inventory_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
