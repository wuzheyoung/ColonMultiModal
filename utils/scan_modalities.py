"""
扫描 CMB-CRC 影像元数据，为每个患者挑选最佳 CT / MRI series，生成配对 CSV。

数据来源:
  - digest xlsx: 含 Modality / SeriesDescription / ImageCount 等（每 series 一行）
  - metadata.csv: 含 SeriesInstanceUID → S5cmdManifestPath（series 文件夹绝对路径）
  - 分期对应.xlsx: 患者 ID → stage（无表头，两列）

输出:
  - data/cmb_series_index.csv: 每 series 一行（人工检查用）
  - data/cmb_pairs.csv:        每患者一行（dataloader 直接读）
      列: patient_id, stage, label, ct_path, mri_path,
          ct_available, mri_available, n_ct_slices, n_mri_slices,
          ct_desc, mri_desc
"""

import os
import pandas as pd


# ── series 选取启发式 ──────────────────────────────────────────────
# CT: 腹盆腔静脉期轴位优先；排除定位/重建/肺窗/矢状冠状等
CT_EXCLUDE = ["SAG", "COR", "SCOUT", "TOPOGRAM", "MIP", "LUNG",
              "SCANOGRAM", "RECON", "LOC", "TENSOR"]
# MRI: 轴位 T2 优先（直肠癌分期主力序列）；排除弥散/减影/定位/矢状冠状
MR_EXCLUDE = ["SAG", "COR", "DIFF", "ADC", "LOC", "SUB", "MIP", "TRACEW"]


def _desc(row):
    return str(row.get("SeriesDescription", "")).upper()


def _body(row):
    return str(row.get("BodyPartExamined", "")).upper()


def _score_ct(row):
    """CT 优先级打分，越大越优先；返回 None 表示排除。"""
    d = _desc(row)
    if any(x in d for x in CT_EXCLUDE):
        return None
    if int(row.get("ImageCount", 0)) < 10:
        return None
    body = _body(row)
    score = 0
    if "VENOUS" in d or "VEN" in d:
        score += 100
    elif "ARTERIAL" in d or "ART" in d:
        score += 60
    if "AXIAL" in d or "AX" in d:
        score += 30
    if "ABD" in body or "PELVIS" in body:
        score += 20
    return score


def _score_mr(row):
    """MRI 优先级打分，越大越优先；返回 None 表示排除。"""
    d = _desc(row)
    if any(x in d for x in MR_EXCLUDE):
        return None
    if int(row.get("ImageCount", 0)) < 10:
        return None
    score = 0
    if "T2" in d:
        score += 100
    elif "T1" in d:
        score += 50
    if "AX" in d:
        score += 30
    if "PELVIS" in d or "RECTUM" in d:
        score += 20
    return score


def _pick_best(rows, score_fn):
    """在候选 series 中按 (score, ImageCount) 取最优，返回该行或 None。"""
    best, best_key = None, None
    for _, r in rows.iterrows():
        s = score_fn(r)
        if s is None:
            continue
        key = (s, int(r.get("ImageCount", 0)))
        if best_key is None or key > best_key:
            best, best_key = r, key
    return best


def build_pairs(digest_path, metadata_csv, stage_xlsx, out_dir):
    df = pd.read_excel(digest_path, sheet_name="Metadata")
    mdf = pd.read_csv(metadata_csv)
    path_map = dict(zip(mdf["SeriesInstanceUID"], mdf["S5cmdManifestPath"]))

    stage_df = pd.read_excel(stage_xlsx, sheet_name=0, header=None)
    stage_df.columns = ["patient_id", "stage"]
    stage_map = dict(zip(stage_df["patient_id"], stage_df["stage"]))

    # stage → 连续 label（本数据集只有 3/4 两类）
    unique_stages = sorted(set(stage_map.values()))
    label_map = {s: i for i, s in enumerate(unique_stages)}

    # ── series 索引（仅含目标患者）──
    sub = df[df["PatientID"].isin(stage_map.keys())].copy()
    sub["folder_path"] = sub["SeriesInstanceUID"].map(path_map)
    index_cols = ["PatientID", "Modality", "SeriesDescription", "BodyPartExamined",
                  "ImageCount", "SeriesInstanceUID", "folder_path"]
    series_index = sub[index_cols].reset_index(drop=True)

    # ── 每患者配对 ──
    pairs = []
    for pid, stage in stage_map.items():
        pdf = sub[sub["PatientID"] == pid]
        ct_rows = pdf[pdf["Modality"] == "CT"]
        mr_rows = pdf[pdf["Modality"] == "MR"]

        best_ct = _pick_best(ct_rows, _score_ct) if len(ct_rows) else None
        best_mr = _pick_best(mr_rows, _score_mr) if len(mr_rows) else None

        ct_path = path_map.get(best_ct["SeriesInstanceUID"], "") if best_ct is not None else ""
        mri_path = path_map.get(best_mr["SeriesInstanceUID"], "") if best_mr is not None else ""

        pairs.append({
            "patient_id": pid,
            "stage": stage,
            "label": label_map[stage],
            "ct_path": ct_path if isinstance(ct_path, str) else "",
            "mri_path": mri_path if isinstance(mri_path, str) else "",
            "ct_available": bool(ct_path),
            "mri_available": bool(mri_path),
            "n_ct_slices": int(best_ct["ImageCount"]) if best_ct is not None else 0,
            "n_mri_slices": int(best_mr["ImageCount"]) if best_mr is not None else 0,
            "ct_desc": best_ct["SeriesDescription"] if best_ct is not None else "",
            "mri_desc": best_mr["SeriesDescription"] if best_mr is not None else "",
        })

    pairs_df = pd.DataFrame(pairs)

    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, "cmb_series_index.csv")
    pairs_path = os.path.join(out_dir, "cmb_pairs.csv")
    series_index.to_csv(idx_path, index=False, encoding="utf-8-sig")
    pairs_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    print(f"label 映射: {label_map}")
    print(f"series 索引 -> {idx_path}  ({len(series_index)} 行)")
    print(f"患者配对   -> {pairs_path}  ({len(pairs_df)} 行)")
    print(pairs_df[["patient_id", "stage", "label", "ct_available", "mri_available",
                    "n_ct_slices", "n_mri_slices"]].to_string(index=False))
    return pairs_df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_dir = os.path.join(base, "data", "cmb", "metadata")
    build_pairs(
        digest_path=os.path.join(meta_dir, "CMB-CRC_v09_20251010-nbia-digest-6.xlsx"),
        metadata_csv=os.path.join(meta_dir, "metadata.csv"),
        stage_xlsx=os.path.join(meta_dir, "分期对应.xlsx"),
        out_dir=os.path.join(base, "data"),
    )
