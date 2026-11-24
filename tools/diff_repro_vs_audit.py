# -*- coding: utf-8 -*-
"""diff_repro_vs_audit.py — 逐字段对比 recompute (tgr版) 与 audit_fusion_debug.json 的复现段。"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

MINE = r"C:\Users\Chris\Projects\ColonMultiModal\logs\recompute_direct_tcd_singles_tgr.json"
AUDIT = r"C:\Users\Chris\Projects\ColonMultiModal\logs\audit_fusion_debug.json"

mine = json.load(open(MINE, encoding="utf-8"))
audit = json.load(open(AUDIT, encoding="utf-8"))

print("=== task1_clin mine vs audit ===")
print("mine :", json.dumps(mine["task1_clin"], ensure_ascii=False))
print("audit:", json.dumps(audit["task1_clin"], ensure_ascii=False))
print()

ma = mine["task4_fusion_debug"]["ablation"]
au = audit["task4_fusion_debug"]["ablation"]
for sc in ["direct_all", "direct_td", "direct_tcd"]:
    for st in ["s1_singles"]:
        m, a = ma[sc][st], au[sc][st]
        print("=== %s.%s: mine n=%s T+=%s | audit n=%s T+=%s ==="
              % (sc, st, m["n"], m["T+"], a["n"], a["T+"]))
        for k in ["clin_clean", "d_conv", "t_conv", "c_conv", "t_rad", "c_rad"]:
            mm, aa = m.get(k, {}), a.get(k, {})
            print("  %-10s mine %s/%s±%s n=%s | audit %s/%s±%s n=%s"
                  % (k, mm.get("pooled"), mm.get("fold_mean"), mm.get("fold_sd"),
                     mm.get("n"), aa.get("pooled"), aa.get("fold_mean"),
                     aa.get("fold_sd"), aa.get("n")))
print()
print("=== strict (my tgr list) 六路 ===")
for k, e in mine["strict_n160_singles"].items():
    print("  %-10s %s / %s±%s  n=%s" % (k, e["pooled"], e["fold_mean"], e["fold_sd"], e["n"]))
