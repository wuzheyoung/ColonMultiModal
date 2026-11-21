# -*- coding: utf-8 -*-
"""results/ 逐 run 可复用性盘点 (只读; 不训练/不读影像/不调 GPU)。
输出: results_audit.json + results_runs.csv (work/ 目录)
"""
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = r"d:\PyFile\ColonMultiModal"
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "work")
DATA_CASES = ["382", "391", "1", "2", "4", "5", "6"]


def dsize(p):
    b = f = 0
    ck = pg = tx = 0
    for dp, _, fns in os.walk(p):
        for n in fns:
            fp = os.path.join(dp, n)
            try:
                s = os.path.getsize(fp)
            except OSError:
                continue
            b += s
            f += 1
            low = n.lower()
            if low.endswith((".pth", ".pt")):
                ck += s
            elif low.endswith((".png", ".jpg", ".jpeg")):
                pg += s
            else:
                tx += s
    return b, f, ck, pg, tx


def metrics_from_json(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    if "tasks" in d:
        out = []
        for t in ("T", "N"):
            x = d["tasks"].get(t, {})
            out.append(f"{t} AUC={x.get('auc_roc')} AP={x.get('auc_pr')} F1={x.get('macro_f1')}")
        return " | ".join(out)
    r = d.get("result", {})
    if "best_auc" in r:
        return (f"T AUC={r.get('best_auc')} F1={r.get('best_macro_f1')} | "
                f"N AUC={r.get('best_auc_n')} F1={r.get('best_macro_f1_n')}")
    if "best_macro_f1" in r:
        return f"T F1={r.get('best_macro_f1')} | N F1={r.get('best_macro_f1_n')} (无 AUC)"
    return None


def metrics_from_oof(run_dir):
    csvs = sorted(glob.glob(os.path.join(run_dir, "*_k*_s*.csv")))
    csvs = [c for c in csvs if not c.endswith("_last5.csv")]
    if not csvs:
        return None, 0, set()
    rows, pids = [], set()
    for c in csvs:
        try:
            d = pd.read_csv(c, encoding="utf-8-sig")
        except Exception:
            continue
        if not {"patient_id", "label_T", "prob_T"}.issubset(d.columns):
            continue
        pids |= set(d["patient_id"].astype(str))
        y = d["label_T"].to_numpy()
        p = d["prob_T"].to_numpy()
        if len(np.unique(y)) > 1:
            rows.append((roc_auc_score(y, p), average_precision_score(y, p),
                         f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)))
    if not rows:
        return None, len(csvs), pids
    a = np.array(rows)
    return (f"T AUC={a[:,0].mean():.3f} AP={a[:,1].mean():.3f} F1={a[:,2].mean():.3f} "
            f"(5折均值, 逐折见 OOF)"), len(csvs), pids


def main():
    refs = {}
    for i, l in enumerate(open(os.path.join(ROOT, "README.md"), encoding="utf-8", errors="ignore").read().splitlines(), 1):
        for m in re.finditer(r"run[_-]?(\d{1,3})", l):
            refs.setdefault(int(m.group(1)), []).append(i)

    runs = sorted([p for p in glob.glob(os.path.join(RESULTS, "**", "run_*"), recursive=True) if os.path.isdir(p)])
    records = []
    for p in runs:
        name = os.path.basename(p)
        num = int(re.search(r"run_(\d+)", name).group(1))
        b, f, ck, pg, tx = dsize(p)
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.stat(p).st_ctime))
        js = sorted(glob.glob(os.path.join(p, "*report*.json"))) + sorted(glob.glob(os.path.join(p, "*.json")))
        met = None
        for j in js:
            met = metrics_from_json(j)
            if met:
                break
        folds = 0
        pids = set()
        if not met:
            met, folds, pids = metrics_from_oof(p)
        else:
            csvs = glob.glob(os.path.join(p, "*_k*_s*.csv"))
            folds = len([c for c in csvs if not c.endswith("_last5.csv")])
            for c in csvs[:3]:
                try:
                    pids |= set(pd.read_csv(c, encoding="utf-8-sig")["patient_id"].astype(str))
                except Exception:
                    pass
        is_p0 = "p0_" in name
        is_kfold = bool(re.search(r"_k[0-4]($|_)", name)) or folds >= 2
        is_ring = ("ring" in name) or ("body" in name)
        is_multi_seed = bool(re.search(r"_s(0|1|7)($|_)", name))
        is_pure_mri = ("smmri" in name) or (("mri_t" in name) and ("tc" not in name))
        clinical = not is_p0  # 旧主模型/ROI-DL 均带 7 维临床; P0 显式 --no-clinical
        data_changed = (not is_p0)  # 旧 run 全部早于数据侧修正 (382/391/1/2/4/5/6)
        rule5 = True  # 无任何 JSON 含 last5/calT_last5 字段 (全仓 grep 为 0)
        rules = []
        if clinical:
            rules.append("1")
        if is_ring:
            rules.append("2")
        if is_kfold:
            rules.append("3")
        if is_multi_seed:
            rules.append("4")
        if rule5:
            rules.append("5")
        if data_changed:
            rules.append("6")
        reusable = (is_pure_mri and not clinical and not is_kfold and not is_multi_seed
                    and not is_ring and not rule5 and not data_changed)
        referenced = bool(refs.get(num))
        if is_p0:
            action = "保留"
        elif referenced:
            action = "保留"
        else:
            action = "移出权重(文本留仓)"
        loss = []
        if ck:
            loss.append(f"权重 {ck/1024**2:.0f} MB")
        if folds:
            loss.append(f"OOF {folds} 折")
        if pg:
            loss.append(f"图 {pg/1024**2:.1f} MB")
        if referenced:
            loss.append(f"README 引用行 {refs.get(num)}")
        records.append({
            "run": name, "path": os.path.relpath(p, ROOT).replace("\\", "/"),
            "mb": round(b / 1024 ** 2, 1), "files": f, "created": created,
            "weights_mb": round(ck / 1024 ** 2, 1), "images_mb": round(pg / 1024 ** 2, 1),
            "text_mb": round(tx / 1024 ** 2, 2), "oof_folds": folds,
            "metrics": met or "—",
            "reusable": "是" if reusable else "否",
            "rules": ",".join(rules) if rules else "—",
            "action": action,
            "loss_if_deleted": "; ".join(loss) if loss else "仅报告文本",
            "readme_ref": ",".join(map(str, refs.get(num, []))),
            "pids_hit": ",".join(sorted(pids & set(DATA_CASES))),
            "pure_mri": is_pure_mri, "p0": is_p0,
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT, "results_runs.csv"), index=False, encoding="utf-8-sig")
    total = int(df["mb"].sum() * 1024 ** 2)
    reus = int(df.loc[df["reusable"] == "是", "mb"].sum() * 1024 ** 2)
    summary = {
        "n_runs": len(df), "results_bytes": total, "results_gib": round(total / 1024 ** 3, 2),
        "reusable_bytes": reus, "nonreusable_bytes": total - reus,
        "weights_bytes": int(df["weights_mb"].sum() * 1024 ** 2),
        "images_bytes": int(df["images_mb"].sum() * 1024 ** 2),
        "text_bytes": int(df["text_mb"].sum() * 1024 ** 2),
        "actions": df["action"].value_counts().to_dict(),
        "rules_count": {r: int(df["rules"].str.contains(r).sum()) for r in "123456"},
        "referenced_runs": sorted(df.loc[df["readme_ref"] != "", "run"].tolist()),
    }
    json.dump({"summary": summary, "runs": records}, open(os.path.join(OUT, "results_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(df[["run", "mb", "created", "rules", "action"]].to_string(index=False))


if __name__ == "__main__":
    main()
