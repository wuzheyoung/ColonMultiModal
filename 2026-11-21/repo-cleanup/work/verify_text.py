# -*- coding: utf-8 -*-
"""逐 run 核验文本/数字产物是否能替代权重 (只读)。"""
import glob
import json
import os
import re

import pandas as pd

ROOT = r"d:\PyFile\ColonMultiModal"
RESULTS = os.path.join(ROOT, "results")
WORK = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "work")
OOF_DIR = os.path.join(RESULTS, "roidl_oof")
P0_TAGS = {
    117: "p0_main_A_ring5m_masked", 118: "p0_main_B_sdf", 121: "p0_main_C_bbox",
    122: "p0_attr_D_binonly", 123: "p0_attr_E_nonly", 124: "p0_attr_F_tonly",
    125: "p0_attr_G_Tclin", 126: "p0_main_A_ring5m_masked", 127: "p0_main_A_ring5m_masked",
    128: "p0_attr_D_binonly", 129: "p0_attr_D_binonly",
}


def walk_files(p):
    out = []
    for dp, _, fns in os.walk(p):
        for n in fns:
            out.append(os.path.join(dp, n))
    return out


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def parse(run_dir):
    files = walk_files(run_dir)
    js = [f for f in files if f.lower().endswith(".json")]
    logs = [f for f in files if f.lower().endswith((".log", ".txt")) and "eval_report" not in f]
    txts = [f for f in files if f.lower().endswith(".txt")]
    csvs = [f for f in files if f.lower().endswith(".csv")]
    weights = [f for f in files if f.lower().endswith((".pth", ".pt"))]
    eval_js = [f for f in js if "eval_report" in os.path.basename(f).lower()]
    ksum_js = [f for f in js if "kfold_summary" in os.path.basename(f).lower()]
    cmd = [f for f in files if re.search(r"(^|[\\/])(cmd|command|args|config)", os.path.basename(f), re.I)]
    parsed, bad = [], 0
    for f in js:
        try:
            parsed.append((f, json.load(open(f, encoding="utf-8"))))
        except Exception:
            bad += 1
    cov = {"T_acc": False, "T_f1": False, "T_auc": False, "T_pr": False,
           "N_acc": False, "N_f1": False, "N_auc": False, "N_pr": False,
           "threshold": False, "cm": False, "per_fold": False, "params": False}
    for _, d in parsed:
        tasks = d.get("tasks", {}) if isinstance(d, dict) else {}
        for t in ("T", "N"):
            x = tasks.get(t, {}) or {}
            cov[f"{t}_acc"] |= num(x.get("accuracy"))
            cov[f"{t}_f1"] |= num(x.get("macro_f1"))
            cov[f"{t}_auc"] |= num(x.get("auc_roc"))
            cov[f"{t}_pr"] |= num(x.get("auc_pr"))
            if isinstance(x.get("calibrated"), dict):
                cov["threshold"] |= num(x["calibrated"].get("threshold"))
            cov["cm"] |= isinstance(x.get("confusion_matrix"), list)
        r = d.get("result", {}) if isinstance(d, dict) else {}
        # 非 tasks 结构 (2.5D/MIL/ROI-DL): result 里的 best_acc/best_macro_f1 也算数字
        cov["T_acc"] |= num(r.get("best_acc"))
        cov["T_f1"] |= num(r.get("best_macro_f1"))
        cov["T_auc"] |= num(r.get("best_auc"))
        cov["T_pr"] |= num(r.get("best_auc_pr") or r.get("best_pr_auc"))
        cov["N_f1"] |= num(r.get("best_macro_f1_n"))
        cov["N_auc"] |= num(r.get("best_auc_n"))
        cov["N_pr"] |= num(r.get("best_auc_pr_n"))
        cov["threshold"] |= isinstance(r.get("calT"), dict) and num(r.get("calT", {}).get("threshold"))
        cov["per_fold"] |= isinstance(d.get("folds"), list)
        cov["params"] |= isinstance(d.get("params"), dict)
    cov["cm"] = cov["cm"] and cov["threshold"]
    cov["per_fold"] = cov["per_fold"] or len(eval_js) >= 2
    logtxt = ""
    for f in logs + txts:
        try:
            logtxt += open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            pass
    cov["log_auc"] = bool(re.search(r"(?i)\bauc\b", logtxt))
    cov["log_pr"] = bool(re.search(r"(?i)(pr[-_ ]?auc|auc_pr|ap=)", logtxt))
    cov["log_f1"] = bool(re.search(r"(?i)(macro_f1|f1[=: ])", logtxt))
    cov["log_acc"] = bool(re.search(r"(?i)(accuracy|acc[=: ])", logtxt))
    return files, weights, eval_js, ksum_js, logs, txts, csvs, cmd, parsed, bad, cov


def main():
    runs = sorted([p for p in glob.glob(os.path.join(RESULTS, "**", "run_*"), recursive=True) if os.path.isdir(p)])
    recs = []
    for p in runs:
        name = os.path.basename(p)
        m = re.search(r"run_(\d+)", name)
        numid = int(m.group(1))
        files, weights, eval_js, ksum_js, logs, txts, csvs, cmd, parsed, bad, cov = parse(p)
        oof = [c for c in csvs if c.endswith("_last5.csv") or re.search(r"_k\d+_s\d+\.csv$", c)]
        if not oof and numid in P0_TAGS:
            oof = sorted(glob.glob(os.path.join(OOF_DIR, P0_TAGS[numid] + "_k*_s*.csv")))
        structured = any(cov[k] for k in ("T_f1", "T_auc", "N_f1", "N_auc")) or bool(oof)
        any_num = structured or any(re.search(r"(?i)auc|f1", open(f, encoding="utf-8", errors="ignore").read())
                                    for f in logs + txts)
        wb = sum(os.path.getsize(f) for f in weights)
        recs.append({
            "run": name, "path": os.path.relpath(p, ROOT).replace("\\", "/"),
            "weight_n": len(weights), "weight_mb": round(wb / 1024 ** 2, 1),
            "weight_files": ";".join(os.path.basename(f) for f in weights[:3]),
            "eval_json": len(eval_js), "kfold_summary": len(ksum_js),
            "log": len(logs), "oof_csv": len(oof), "cmd_config": len(cmd),
            "json_parse_fail": bad,
            **{k: int(v) for k, v in cov.items()},
            "structured_numbers": int(structured), "any_numbers": int(any_num),
            "irreplaceable": int(len(weights) > 0 and not any_num),
        })
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(WORK, "run_text_audit.csv"), index=False, encoding="utf-8-sig")

    # README 数字风险: 按行提到的指标类型, 检查同编号 run 的 JSON/日志是否支撑
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8", errors="ignore").read().splitlines()
    cap = {}
    for _, r in df.iterrows():
        n = int(re.search(r"run_(\d+)", r["run"]).group(1))
        c = cap.setdefault(n, {k: False for k in ("auc", "pr", "f1", "acc")})
        c["auc"] |= bool(r["T_auc"] or r["N_auc"] or r["log_auc"])
        c["pr"] |= bool(r["T_pr"] or r["N_pr"] or r["log_pr"])
        c["f1"] |= bool(r["T_f1"] or r["N_f1"] or r["log_f1"])
        c["acc"] |= bool(r["T_acc"] or r["N_acc"] or r["log_acc"])
    risk = []
    for i, line in enumerate(readme, 1):
        mm = re.search(r"run[_-]?(\d{1,3})", line)
        if not mm:
            continue
        n = int(mm.group(1))
        vals = re.findall(r"\b0\.\d{3,4}\b|\b\d{2,3}\.\d%", line)
        if not vals:
            continue
        need = []
        if re.search(r"(?i)auc|pr[-_ ]?auc", line):
            need.append("auc")
        if re.search(r"(?i)pr[-_ ]?auc", line):
            need.append("pr")
        if re.search(r"(?i)f1", line):
            need.append("f1")
        if re.search(r"(?i)acc|准确率", line):
            need.append("acc")
        c = cap.get(n, {})
        missing = [k for k in need if not c.get(k)]
        if missing:
            risk.append({"line": i, "run": f"run_{n}", "numbers": ",".join(vals[:6]),
                         "missing_metric": ",".join(missing),
                         "text": line.strip()[:180]})
    pd.DataFrame(risk).to_csv(os.path.join(WORK, "readme_numbers_at_risk.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "n_runs": len(df),
        "with_weights": int((df.weight_n > 0).sum()),
        "without_weights": int((df.weight_n == 0).sum()),
        "weight_gib": round(df.weight_mb.sum() / 1024, 2),
        "irreplaceable_n": int(df.irreplaceable.sum()),
        "irreplaceable_mib": round(df.loc[df.irreplaceable == 1, "weight_mb"].sum(), 1),
        "irreplaceable_runs": df.loc[df.irreplaceable == 1, "run"].tolist(),
        "structured_n": int(df.structured_numbers.sum()),
        "no_structured": int((df.structured_numbers == 0).sum()),
        "any_numbers": int(df.any_numbers.sum()),
        "eval_json_runs": int((df.eval_json > 0).sum()),
        "log_runs": int((df.log > 0).sum()),
        "oof_runs": int((df.oof_csv > 0).sum()),
        "cmd_runs": int((df.cmd_config > 0).sum()),
        "readme_risk_lines": len(risk),
        "coverage": {k: int(df[k].sum()) for k in
                     ("T_acc", "T_f1", "T_auc", "T_pr", "N_acc", "N_f1", "N_auc", "N_pr",
                      "threshold", "cm", "per_fold", "params")},
    }
    json.dump({"summary": summary, "runs": recs, "readme_risk": risk},
              open(os.path.join(WORK, "run_text_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(df[["run", "weight_n", "weight_mb", "structured_numbers", "any_numbers",
              "irreplaceable"]].to_string(index=False))
    print("README risk rows:", len(risk))


if __name__ == "__main__":
    main()
