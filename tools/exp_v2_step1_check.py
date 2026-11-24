# -*- coding: utf-8 -*-
"""Step1: local slice_feats_v2 integrity + K distribution + mask_source audit (m00964)."""
import os, sys, json, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FEAT = os.path.join("features", "slice_feats_v2")

def main():
    npzs = sorted(glob.glob(os.path.join(FEAT, "*", "*", "*.npz")))
    print(f"total npz: {len(npzs)}")
    bad = []          # unreadable / non-finite
    no_feats = []     # missing feats key
    k_all = []        # (backbone, seq, pid, K)
    src_all = {}      # (seq) -> Counter(mask_source)
    keys_seen = set()
    for p in npzs:
        bk, seq, fn = p.split(os.sep)[-3], p.split(os.sep)[-2], os.path.basename(p)
        pid = int(fn.replace(".npz", ""))
        try:
            with np.load(p) as z:
                keys_seen |= set(z.files)
                f = z["feats"]
                if not np.isfinite(f).all():
                    bad.append((bk, seq, pid, "nonfinite"))
                k_all.append((bk, seq, pid, int(f.shape[0])))
                src = z["mask_source"][()] if "mask_source" in z.files else None
                src_all.setdefault(seq, {})
                key = str(src) if src is not None else "<missing>"
                src_all[seq][key] = src_all[seq].get(key, 0) + 1
        except Exception as e:
            bad.append((bk, seq, pid, f"unreadable: {e}"))
    print("npz keys:", sorted(keys_seen))
    print("bad (unreadable/nonfinite):", len(bad))
    for b in bad[:10]:
        print("  ", b)

    # K distribution per dinov2/{t,c,d}
    df = pd.DataFrame(k_all, columns=["backbone", "seq", "pid", "K"])
    print("\nK stats (min/median/p95/max) per backbone/seq:")
    out_k = {}
    for (bk, seq), g in df.groupby(["backbone", "seq"]):
        K = np.sort(g["K"].values)
        st = {"n": int(len(K)), "min": int(K.min()), "median": float(np.median(K)),
              "p95": float(np.percentile(K, 95)), "max": int(K.max())}
        out_k[f"{bk}/{seq}"] = st
        print(f"  {bk}/{seq}: n={st['n']} min={st['min']} median={st['median']:.1f} p95={st['p95']:.1f} max={st['max']}")

    # mask_source distribution for all d
    print("\nmask_source distribution (all cases, per seq):")
    for seq in sorted(src_all):
        print(f"  {seq}: {src_all[seq]}")

    # direct_tcd n=162 d mask_source
    master = pd.read_csv("data/master_5fold_356.csv")
    cohort = pd.read_csv("data/direct_cohort_ids.csv")
    pid_col = "patient_id" if "patient_id" in cohort.columns else "pid"
    tgr = set(cohort.loc[cohort["is_tgr"] == 1, pid_col])
    raw = pd.read_csv("data/hos_pairs_raw.csv").set_index("patient_id")
    direct = master[~master["patient_id"].isin(tgr)]
    dp = direct.set_index("patient_id")
    has = lambda c: raw["mri_path_" + c].notna().reindex(dp.index).fillna(False)
    has_t = set(dp.index[has("t")]); has_c = set(dp.index[has("c")]); has_d = set(dp.index[has("d")])
    tcd = direct["patient_id"].isin(has_t & has_c & has_d)
    tcd_ids = set(direct.loc[tcd, "patient_id"])
    print(f"\ndirect_all={len(direct)}  direct_tcd(n=162 check)={len(tcd_ids)}")
    d_df = df[(df["seq"] == "d") & (df["pid"].isin(tcd_ids))]
    # re-join mask_source from npz values
    src_tcd = {}
    for p in npzs:
        bk, seq, fn = p.split(os.sep)[-3], p.split(os.sep)[-2], os.path.basename(p)
        if seq != "d":
            continue
        pid = int(fn.replace(".npz", ""))
        if pid not in tcd_ids:
            continue
        with np.load(p) as z:
            s = str(z["mask_source"][()])
        src_tcd[s] = src_tcd.get(s, 0) + 1
    print(f"direct_tcd d mask_source ({sum(src_tcd.values())} cases): {src_tcd}")
    # how many tcd d cases have remapped vs self, and old-d_conv coverage (route2_cd d_pids)
    old_d = pd.read_csv("features/logs/route2_cd_feats/d_pids.csv").iloc[:, 0]
    old_d = set(old_d.astype(int))
    inter = tcd_ids & old_d
    print(f"tcd d cases also in old d_conv: {len(inter)} / missing from old: {len(tcd_ids - old_d)}")

    out = {"total_npz": len(npzs), "bad": len(bad), "keys": sorted(keys_seen),
           "k_stats": out_k, "mask_source_all": src_all,
           "direct_tcd_n": len(tcd_ids),
           "mask_source_tcd_d": src_tcd,
           "tcd_d_in_old_dconv": len(inter), "tcd_d_missing_old": len(tcd_ids - old_d)}
    with open("logs/exp_v2_step1_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nsaved logs/exp_v2_step1_check.json")

if __name__ == "__main__":
    main()
