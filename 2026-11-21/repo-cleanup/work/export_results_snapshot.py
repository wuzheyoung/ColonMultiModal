# -*- coding: utf-8 -*-
"""全量导出 results/ 数值快照 (只读文本/元数据; 不读影像/不加载模型/不训练)。

输出到 2026-11-21/repo-cleanup/outputs/snapshot/:
  run_summary_<ts>.csv / fold_detail_<ts>.csv / snapshot_<ts>.json
  sha256_<ts>.csv / export_<ts>.log
幂等: 同名存在时自动追加 _1/_2 后缀, 不覆盖。
"""
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)

ROOT = r"d:\PyFile\ColonMultiModal"
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
OUTDIR = os.path.join(ROOT, "2026-11-21", "repo-cleanup", "outputs", "snapshot")
TS = time.strftime("%Y%m%d_%H%M%S")

LOG = []
FAIL = {"json_parse_error": [], "no_numeric_source": [], "no_metrics": [], "oof_parse_error": []}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    LOG.append(line)


def unique_path(base):
    p = base
    i = 1
    while os.path.exists(p):
        root, ext = os.path.splitext(base)
        p = f"{root}_{i}{ext}"
        i += 1
    return p


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and (math.isnan(o) or math.isinf(o))) else float(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fnum(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def metrics_from_probs(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    keep = np.isfinite(p) & (y >= 0)
    y, p = y[keep], p[keep]
    if len(y) == 0:
        return {}
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    one = len(np.unique(y)) < 2
    return {"acc": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "auc": None if one else float(roc_auc_score(y, p)),
            "pr_auc": None if one else float(average_precision_score(y, p)),
            "threshold": 0.5, "cm": cm, "n": int(len(y))}


def extract_json(run_dir, files):
    js = [f for f in files if f.lower().endswith(".json")]
    eval_js = [f for f in js if "eval_report" in os.path.basename(f).lower()]
    parsed, bad = [], []
    for f in js:
        try:
            parsed.append((f, json.load(open(f, encoding="utf-8"))))
        except Exception as e:
            bad.append((f, f"{type(e).__name__}: {e}"))
    return eval_js, js, parsed, bad


def main():
    log(f"start export; ROOT={ROOT}; script={os.path.abspath(__file__)}")
    t0 = time.time()
    run_dirs = sorted([p for p in glob.glob(os.path.join(RESULTS, "**", "run_*"), recursive=True)
                       if os.path.isdir(p)])
    log(f"discovered {len(run_dirs)} run dirs under results/")

    run_rows, fold_rows, manifest, failures = [], [], [], []
    src_bytes = src_files = 0
    for rd in run_dirs:
        name = os.path.basename(rd)
        m = re.search(r"run_(\d+)", name)
        run_no = int(m.group(1)) if m else None
        rel = os.path.relpath(rd, ROOT).replace("\\", "/")
        files = []
        for dp, _, fns in os.walk(rd):
            for n in fns:
                files.append(os.path.join(dp, n))
        wfiles = [f for f in files if f.lower().endswith((".pth", ".pt"))]
        txtfiles = [f for f in files if not f.lower().endswith((".pth", ".pt"))]
        wbytes = sum(os.path.getsize(f) for f in wfiles)
        src_bytes += sum(os.path.getsize(f) for f in files)
        src_files += len(files)
        eval_js, all_js, parsed, bad = extract_json(rd, files)
        for f, err in bad:
            FAIL["json_parse_error"].append((rel, f, err))
        logs = [f for f in files if f.lower().endswith((".log", ".txt")) and "eval_report" not in os.path.basename(f)]
        oofs = [f for f in files if f.lower().endswith(".csv")]
        # P0: OOF 在 results/roidl_oof, 按 tag 关联
        p0tag = None
        for tag in sorted(set(re.sub(r"_k\d+_s\d+(_last5)?\.csv$", "",
                                     os.path.basename(x)) for x in glob.glob(os.path.join(RESULTS, "roidl_oof", "*.csv")))):
            key = tag.replace("p0_main_", "").replace("p0_attr_", "").replace("p0_cal_", "")
            if name.startswith(f"run_{run_no}_p0_") and key.split("_")[0].lower() in name.lower():
                p0tag = tag
                break
        if name.startswith(f"run_{run_no}_p0_") and p0tag:
            fold_of = re.findall(r"_p0_([A-G])", name)
            letter = fold_of[0] if fold_of else ""
            cand = {"A": "p0_main_A_ring5m_masked", "B": "p0_main_B_sdf", "C": "p0_main_C_bbox",
                    "D": "p0_attr_D_binonly", "E": "p0_attr_E_nonly", "F": "p0_attr_F_tonly",
                    "G": "p0_attr_G_Tclin"}.get(letter)
            if cand:
                sm = re.search(r"_s(\d+)", name)
                seed = int(sm.group(1)) if sm else 42
                oofs = sorted(glob.glob(os.path.join(RESULTS, "roidl_oof", f"{cand}_k*_s{seed}.csv")))
                oofs += sorted(glob.glob(os.path.join(RESULTS, "roidl_oof", f"{cand}_k*_s{seed}_last5.csv")))
        text_log = ""
        for f in logs:
            try:
                text_log += open(f, encoding="utf-8", errors="ignore").read()
            except OSError:
                pass
        command = None
        for line in text_log.splitlines():
            if re.search(r"(?i)(python .*\.py|命令行|command:|args:)", line):
                command = line.strip()[:400]
                break

        # 主指标: tasks (主模型) > result (2.5D/MIL/ROI) > OOF
        T = {k: None for k in ("acc", "f1", "auc", "pr_auc", "threshold", "cm")}
        N = dict(T)
        best_epoch = None
        train_time = None
        data_csv = None
        params = {}
        meta = {}
        cov = set()
        for f, d in parsed:
            meta = d.get("meta", meta) if isinstance(d, dict) else meta
            params = d.get("params", params) if isinstance(d, dict) else params
            tasks = d.get("tasks", {}) if isinstance(d, dict) else {}
            for t, box in (("T", T), ("N", N)):
                x = tasks.get(t) or {}
                box["acc"] = fnum(x.get("accuracy")) or box["acc"]
                box["f1"] = fnum(x.get("macro_f1")) or box["f1"]
                box["auc"] = fnum(x.get("auc_roc")) or box["auc"]
                box["pr_auc"] = fnum(x.get("auc_pr")) or box["pr_auc"]
                if isinstance(x.get("calibrated"), dict):
                    box["threshold"] = fnum(x["calibrated"].get("threshold")) or box["threshold"]
                if isinstance(x.get("confusion_matrix"), list):
                    box["cm"] = x["confusion_matrix"]
            r = d.get("result", {}) if isinstance(d, dict) else {}
            if r:
                T["acc"] = fnum(r.get("best_acc")) or T["acc"]
                T["f1"] = fnum(r.get("best_macro_f1")) or T["f1"]
                T["auc"] = fnum(r.get("best_auc")) or T["auc"]
                N["f1"] = fnum(r.get("best_macro_f1_n")) or N["f1"]
                N["auc"] = fnum(r.get("best_auc_n")) or N["auc"]
                if isinstance(r.get("calT"), dict):
                    T["threshold"] = fnum(r["calT"].get("threshold")) or T["threshold"]
                if isinstance(r.get("calN"), dict):
                    N["threshold"] = fnum(r["calN"].get("threshold")) or N["threshold"]
                best_epoch = r.get("best_epoch", best_epoch)
            if isinstance(d, dict) and isinstance(d.get("data"), dict):
                data_csv = d["data"].get("val_csv") or d["data"].get("train_csv") or data_csv
        if not data_csv:
            cm = re.search(r"(data/[A-Za-z0-9_./-]+\.csv)", text_log)
            data_csv = cm.group(1) if cm else None
        if not train_time:
            train_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.stat(rd).st_ctime))

        # fold 级: 目录内多份 eval_report (每份一折), 或 OOF CSV 每折
        fold_sources = []
        for f, d in parsed:
            if not isinstance(d, dict):
                continue
            if "tasks" in d or "result" in d:
                fm = re.search(r"_k(\d+)", os.path.basename(f)) or re.search(r"_f(\d+)", os.path.basename(f))
                if fm:
                    fold_sources.append((int(fm.group(1)), f))
        best_oof = [f for f in oofs if not f.endswith("_last5.csv")]
        if fold_sources:
            for fold, f in fold_sources:
                d = dict(parsed)[f]
                tasks = d.get("tasks", {})
                xt, xn = tasks.get("T", {}), tasks.get("N", {})
                fold_rows.append({"run": name, "round": rel.split("/")[1] if "/" in rel else "",
                                  "fold": fold, "source_kind": "eval_report",
                                  "source_file": os.path.relpath(f, ROOT).replace("\\", "/"),
                                  "threshold": (xt.get("calibrated") or {}).get("threshold"),
                                  "cm": json.dumps(xt.get("confusion_matrix"), ensure_ascii=False),
                                  "best_epoch": d.get("result", {}).get("best_epoch"),
                                  "T_acc": xt.get("accuracy"), "T_f1": xt.get("macro_f1"),
                                  "T_auc": xt.get("auc_roc"), "T_pr_auc": xt.get("auc_pr"),
                                  "N_acc": xn.get("accuracy"), "N_f1": xn.get("macro_f1"),
                                  "N_auc": xn.get("auc_roc"), "N_pr_auc": xn.get("auc_pr")})
        else:
            best_oof = [f for f in oofs if not f.endswith("_last5.csv")]
            last_oof = [f for f in oofs if f.endswith("_last5.csv")]
            for kind, group in (("oof_best", best_oof), ("oof_last5", last_oof)):
                for f in sorted(group):
                    try:
                        d = pd.read_csv(f, encoding="utf-8-sig")
                        has_t = {"label_T", "prob_T"}.issubset(d.columns)
                        has_n = {"label_N", "prob_N"}.issubset(d.columns)
                        if not has_t and not has_n:
                            continue
                        mt = metrics_from_probs(d["label_T"], d["prob_T"]) if has_t else {}
                        mn = metrics_from_probs(d["label_N"], d["prob_N"]) if has_n else {}
                        fm = re.search(r"_k(\d+)_s\d+", os.path.basename(f))
                        fold_rows.append({"run": name, "round": rel.split("/")[1] if "/" in rel else "",
                                          "fold": int(fm.group(1)) if fm else None, "source_kind": kind,
                                          "source_file": os.path.relpath(f, ROOT).replace("\\", "/"),
                                          "threshold": mt.get("threshold"),
                                          "cm": json.dumps(mt.get("cm"), ensure_ascii=False),
                                          "best_epoch": None,
                                          "T_acc": mt.get("acc"), "T_f1": mt.get("f1"),
                                          "T_auc": mt.get("auc"), "T_pr_auc": mt.get("pr_auc"),
                                          "N_acc": mn.get("acc"), "N_f1": mn.get("f1"),
                                          "N_auc": mn.get("auc"), "N_pr_auc": mn.get("pr_auc")})
                    except Exception as e:
                        FAIL["oof_parse_error"].append((rel, f, f"{type(e).__name__}: {e}"))
            if not oofs and not any(T.values()) and not any(N.values()):
                FAIL["no_numeric_source"].append((rel, "无 eval_report/OOF", ""))
        # run 级指标缺失但存在 OOF: 用 best OOF 拼接后汇总 (P0 等; T/N 独立处理)
        if best_oof:
            try:
                ds = [pd.read_csv(f, encoding="utf-8-sig") for f in best_oof]
                if not any(v is not None for v in T.values()):
                    td = [d for d in ds if {"label_T", "prob_T"}.issubset(d.columns)]
                    if td:
                        td = pd.concat([d[["label_T", "prob_T"]] for d in td])
                        mt = metrics_from_probs(td["label_T"], td["prob_T"])
                        T.update({k: mt.get(k) for k in ("acc", "f1", "auc", "pr_auc", "threshold", "cm")})
                if not any(v is not None for v in N.values()):
                    nd = [d for d in ds if {"label_N", "prob_N"}.issubset(d.columns)]
                    if nd:
                        nd = pd.concat([d[["label_N", "prob_N"]] for d in nd])
                        mn = metrics_from_probs(nd["label_N"], nd["prob_N"])
                        N.update({k: mn.get(k) for k in ("acc", "f1", "auc", "pr_auc", "threshold", "cm")})
            except Exception as e:
                FAIL["oof_parse_error"].append((rel, "pooled OOF", f"{type(e).__name__}: {e}"))
        if not any(v is not None for v in T.values()) and not any(v is not None for v in N.values()):
            FAIL["no_metrics"].append((rel, "JSON 解析成功但无 T/N 指标", ""))
        has_last5 = any(f.endswith("_last5.csv") for f in oofs)
        has_auc = T["auc"] is not None or N["auc"] is not None
        metric_version = "v2_auc_pr_last5" if (has_last5 and has_auc) else ("v1_auc_pr" if has_auc else "v0_partial")
        run_rows.append({
            "run_dir": rel, "run_name": name, "run_no": run_no,
            "round": rel.split("/")[1] if "/" in rel else "",
            "tag": (meta.get("tag") or meta.get("branch") or re.sub(r"^run_\d+_", "", name)),
            "seed": (meta.get("seed") if isinstance(meta, dict) else None) or
                    (int(re.search(r"_s(\d+)", name).group(1)) if re.search(r"_s(\d+)", name) else None),
            "fold": (int(re.search(r"_k(\d+)", name).group(1)) if re.search(r"_k(\d+)", name) else None),
            "command": command, "config_source": (os.path.relpath(all_js[0], ROOT).replace("\\", "/") if all_js else None),
            "has_clinical": int(bool(params.get("num_clinical", 0))) if isinstance(params, dict) else 0,
            "is_ring": int(("ring" in name.lower()) or ("body" in name.lower()) or ("ring" in str(params).lower())),
            "is_kfold": int(bool(re.search(r"_k\d", name)) or len(best_oof := [f for f in oofs if not f.endswith("_last5.csv")]) >= 2),
            "metric_version": metric_version,
            "T_acc": T["acc"], "T_f1": T["f1"], "T_auc": T["auc"], "T_pr_auc": T["pr_auc"],
            "T_threshold": T["threshold"], "T_cm": json.dumps(clean(T["cm"]), ensure_ascii=False) if T["cm"] else None,
            "N_acc": N["acc"], "N_f1": N["f1"], "N_auc": N["auc"], "N_pr_auc": N["pr_auc"],
            "N_threshold": N["threshold"], "N_cm": json.dumps(clean(N["cm"]), ensure_ascii=False) if N["cm"] else None,
            "best_epoch": best_epoch, "train_time": (meta.get("time") if isinstance(meta, dict) else None),
            "weight_n": len(wfiles), "weight_mb": round(wbytes / 1024 ** 2, 1),
            "weight_names": ";".join(os.path.basename(f) for f in wfiles),
            "text_artifacts": ";".join(os.path.relpath(f, rd).replace("\\", "/") for f in txtfiles),
            "data_csv": data_csv, "json_files": len(all_js), "json_parse_fail": len(bad),
            "oof_files": len(oofs),
        })
        for f in txtfiles:
            try:
                manifest.append({"run": name, "path": os.path.relpath(f, ROOT).replace("\\", "/"),
                                 "bytes": os.path.getsize(f), "sha256": sha256(f)})
            except OSError as e:
                log(f"sha256 fail {f}: {e}")

    os.makedirs(OUTDIR, exist_ok=True)
    run_csv = unique_path(os.path.join(OUTDIR, f"run_summary_{TS}.csv"))
    fold_csv = unique_path(os.path.join(OUTDIR, f"fold_detail_{TS}.csv"))
    snap_json = unique_path(os.path.join(OUTDIR, f"snapshot_{TS}.json"))
    sha_csv = unique_path(os.path.join(OUTDIR, f"sha256_{TS}.csv"))
    log_path = unique_path(os.path.join(OUTDIR, f"export_{TS}.log"))

    pd.DataFrame(run_rows).to_csv(run_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(sha_csv, index=False, encoding="utf-8-sig")

    # logs 目录总量 (仅元数据)
    lb = lf = 0
    for dp, _, fns in os.walk(LOGS):
        for n in fns:
            try:
                lb += os.path.getsize(os.path.join(dp, n))
                lf += 1
            except OSError:
                pass

    fail_counts = {k: len(v) for k, v in FAIL.items()}
    snap = clean({
        "provenance": {"script": os.path.abspath(__file__), "executed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "root": ROOT, "results_dir": RESULTS, "logs_dir": LOGS,
                       "results_runs": len(run_dirs), "results_files": src_files,
                       "results_bytes": src_bytes, "logs_files": lf, "logs_bytes": lb,
                       "hash_scope": "全部非权重文件; .pth/.pt 仅记录数量/大小",
                       "elapsed_sec": round(time.time() - t0, 1)},
        "counts": {"run_rows": len(run_rows), "fold_rows": len(fold_rows),
                   "manifest_rows": len(manifest), "failures": fail_counts},
        "failures": {k: [{"run": a, "file": b, "reason": c} for a, b, c in v[:50]] for k, v in FAIL.items()},
        "runs": run_rows, "folds": fold_rows,
    })
    with open(snap_json, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2, allow_nan=False)

    log(f"run_rows={len(run_rows)} fold_rows={len(fold_rows)} manifest_rows={len(manifest)}")
    log(f"failures={fail_counts}; samples={ {k: v[:3] for k, v in FAIL.items()} }")
    log(f"outputs: run={os.path.basename(run_csv)} fold={os.path.basename(fold_csv)} "
        f"json={os.path.basename(snap_json)} sha={os.path.basename(sha_csv)}")
    log(f"elapsed={time.time() - t0:.1f}s")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print(json.dumps({"run_rows": len(run_rows), "fold_rows": len(fold_rows),
                      "failures": fail_counts, "log": log_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
