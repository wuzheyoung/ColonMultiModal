# -*- coding: utf-8 -*-
"""生成 features/dwi_mask_remap_summary.csv (1650 审计 P2 兜底核实落盘)。"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RAW = "data/hos_pairs_raw.csv"


def main():
    m = pd.read_csv("features/dwi_mask_remap_manifest.csv")
    m["is_direct"] = (m["cohort_direct"] == 1)
    raw = pd.read_csv(RAW).set_index("patient_id")

    def cnt(g):
        d = g["mask_source"].value_counts().to_dict()
        return {k: int(v) for k, v in d.items()}

    def has_tcd(pid):
        try:
            r = raw.loc[pid]
        except KeyError:
            return False
        return all(isinstance(r[c], str) and os.path.exists(r[c])
                   for c in ("mri_path_t", "mri_path_c", "mri_path_d"))

    allc = cnt(m)
    da = cnt(m[m["is_direct"]])
    tcd_mask = m[m["is_direct"]].apply(lambda r: has_tcd(int(r["pid"])), axis=1)
    tcdc = cnt(m[m["is_direct"]][tcd_mask.values])

    with open("features/dwi_mask_remap_summary.csv", "w", encoding="utf-8") as f:
        f.write("scope,self,from_c,from_t,center_fallback,total\n")
        for name, d in (("all", allc), ("direct_all", da), ("direct_tcd", tcdc)):
            f.write(f"{name},{d.get('self', 0)},{d.get('from_c', 0)},{d.get('from_t', 0)},"
                    f"{d.get('center_fallback', 0)},{sum(d.values())}\n")
        f.write("#center_fallback=0 -> discard, no 120mm center-crop needed\n")
    print(open("features/dwi_mask_remap_summary.csv", encoding="utf-8").read())


if __name__ == "__main__":
    main()
