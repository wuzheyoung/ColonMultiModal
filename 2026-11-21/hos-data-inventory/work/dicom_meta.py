# -*- coding: utf-8 -*-
"""最小 DICOM 头解析（支持 explicit/implicit VR little endian），只读。"""
import json
import os
import struct

ROOT = r"d:\PyFile\ColonMultiModal\data\Hos"
WORK = r"d:\PyFile\ColonMultiModal\2026-11-21\hos-data-inventory\work"

WANT = {
    (0x0008, 0x0020): "StudyDate",
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x103E): "SeriesDescription",
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0018, 0x0050): "SliceThickness",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0020, 0x0011): "SeriesNumber",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
}
LONG_VR = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"}


def read_elem(data, i, explicit, endian="<"):
    g, e = struct.unpack(endian + "HH", data[i:i + 4])
    if explicit:
        vr = data[i + 4:i + 6]
        if vr in LONG_VR:
            ln = struct.unpack(endian + "I", data[i + 8:i + 12])[0]
            return (g, e), vr, i + 12, ln
        ln = struct.unpack(endian + "H", data[i + 6:i + 8])[0]
        return (g, e), vr, i + 8, ln
    ln = struct.unpack(endian + "I", data[i + 4:i + 8])[0]
    return (g, e), b"", i + 8, ln


def parse(path, limit=200):
    out = {}
    try:
        with open(path, "rb") as f:
            data = f.read(500000)
    except Exception:
        return out
    i = 132 if data[128:132] == b"DICM" else 0
    # 先解析 meta group（总是 explicit VR little endian）
    ts = ""
    while i + 8 <= len(data):
        tag, vr, vstart, ln = read_elem(data, i, True, "<")
        if ln == 0xFFFFFFFF:
            return out
        val = data[vstart:vstart + ln]
        if tag[0] != 0x0002:
            break
        if tag == (0x0002, 0x0010):
            ts = val.decode("latin-1", "ignore").strip("\x00 ")
        i = vstart + ln
    explicit = ts != "1.2.840.10008.1.2"
    endian = ">" if ts == "1.2.840.10008.1.2.2" else "<"
    out["_ts"] = ts
    n = 0
    while i + 8 <= len(data) and n < limit:
        tag, vr, vstart, ln = read_elem(data, i, explicit, endian)
        if ln == 0xFFFFFFFF or vstart + ln > len(data):
            break
        if tag == (0x7FE0, 0x0010):
            break
        if tag in WANT:
            out[WANT[tag]] = data[vstart:vstart + ln].decode("latin-1", "ignore").strip("\x00 ")
        i = vstart + ln
        n += 1
    return out


def scan(side):
    base = os.path.join(ROOT, side)
    res = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        dcms = []
        for dp, _, fns in os.walk(d):
            for fn in fns:
                if fn.lower().endswith(".dcm"):
                    dcms.append(os.path.join(dp, fn))
        dcms.sort()
        if not dcms:
            res.append({"dir": name, "n_dcm": 0})
            continue
        first = parse(dcms[0])
        last = parse(dcms[-1]) if len(dcms) > 1 else first
        res.append({
            "dir": name, "n_dcm": len(dcms),
            "subdirs": sorted({os.path.basename(os.path.dirname(p)) for p in dcms}),
            "meta_first": first,
            "same_series": first.get("SeriesInstanceUID") == last.get("SeriesInstanceUID"),
        })
    return res


if __name__ == "__main__":
    out = {"CT": scan("CT"), "MRI": scan("MRI")}
    with open(os.path.join(WORK, "dicom_meta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    lines = []
    for side in ("CT", "MRI"):
        lines.append(f"=== {side}")
        for r in out[side]:
            m = r.get("meta_first", {})
            lines.append(
                f"{r['dir']:<28} n={r['n_dcm']:<5} mod={m.get('Modality','?'):<3} "
                f"desc={m.get('SeriesDescription','?')[:24]:<24} pid={m.get('PatientID','?'):<14} "
                f"name={m.get('PatientName','?')[:10]:<10} date={m.get('StudyDate','?')} "
                f"ser#{m.get('SeriesNumber','?'):<4} rows={m.get('Rows','?')} "
                f"oneSeries={r.get('same_series')} ts={m.get('_ts','?')}")
    with open(os.path.join(WORK, "dicom_meta.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("done", {k: len(v) for k, v in out.items()})
