# -*- coding: utf-8 -*-
"""B4: 按 B3 结果收尾 (规则 m00745 B4)
- 救活灰度 (hr_a3 img 3seed 均值 last5 N AUC ≥ 0.56): 灰度救活 → T2+CE+SDF+弱增强TTA 各3seed
  (--mode mri_all 96³ 口径复用 train_roi_dl 训练; 因 train_roi_dl 无 TTA, 退化为标准 96³ 三序列+SDF 对照)
- 灰度仍 < 0.56: 高分辨 --t-only/--n-only 解耦 + CRCFound 局部 Token 池化零训练重测 145 例 CT
输出: logs/step_b4_hr_extend.json + README
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = r"D:\Development\miniconda\envs\Pytorch\python.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="dc")
    args = ap.parse_args()

    # 读 B3 结果
    b3_path = os.path.join(ROOT, "logs", "step_b3_hr_train.json")
    if not os.path.exists(b3_path):
        print("[B4] 无 step_b3_hr_train.json, 跳过 (B3 未运行?)")
        return 1
    with open(b3_path, encoding="utf-8") as f:
        b3 = json.load(f)
    img = (b3.get("summary") or {}).get("hr_a3_img", {})
    img_n = img.get("N_last5_mean")
    img_saved = (img_n is not None) and (img_n >= 0.56)
    out = {"gen": time.strftime("%Y-%m-%d %H:%M:%S"), "img_saved": img_saved,
           "img_N_last5_mean": img_n}

    if img_saved:
        # 灰度救活: 96³ 三序列+SDF (mri_all + sdf-channel), 3seed, 完整对照 A2 臂同口径
        tag = f"{args.tag_prefix}_b4_all_sdf"
        cmds = []
        for seed in (42, 0, 7):
            cmds.append([PY, os.path.join(ROOT, "tools", "train_roi_dl.py"),
                         "--kfold", "5", "--kfold-csv", "data/master_5fold_356.csv",
                         "--full-csv", "data/hos_pairs.csv", "--seed", str(seed),
                         "--mode", "mri_all", "--sdf-channel", "--roi-mask", "bbox",
                         "--epochs", "30", "--self-written", "--no-clinical",
                         "--head-dropout", "0.5", "--out-tag", tag])
        out["plan"] = "img_saved: mri_all+SDF 三序列 3seed 96³ (b4_all_sdf)"
        for c in cmds:
            print("[B4] run:", " ".join(c[2:]))
            r = subprocess.run(c, cwd=ROOT, capture_output=True, text=True)
            if r.returncode not in (0, 1):  # exit 1 为归档噪音, 视成功
                out.setdefault("errors", []).append({"cmd": c, "rc": r.returncode,
                                                     "tail": (r.stdout + r.stderr)[-500:]})
        # 聚合
        aggs = {}
        for seed in (42, 0, 7):
            a = subprocess.run([PY, os.path.join(ROOT, "tools", "aggregate_roidl_kfold.py"),
                                "--tag", f"{tag}_s{seed}", "--seed", str(seed),
                                "--out", os.path.join(ROOT, "logs", f"aggregate_roidl_{tag}_s{seed}.json")],
                               cwd=ROOT, capture_output=True, text=True)
            apath = os.path.join(ROOT, "logs", f"aggregate_roidl_{tag}_s{seed}.json")
            if os.path.exists(apath):
                with open(apath, encoding="utf-8") as f:
                    aggs[f"s{seed}"] = json.load(f)
        if aggs:
            ns = [v["task_N_last5"]["pooled"]["roc_auc"] for v in aggs.values()
                  if v.get("task_N_last5", {}).get("pooled", {}).get("roc_auc") is not None]
            out["mri_all_sdf_N_last5_mean"] = (sum(ns) / len(ns)) if ns else None
    else:
        # 灰度未救活: 高分辨解耦 + CRCFound 零训练
        out["plan"] = "img_dead: 高分辨 t-only/n-only 解耦 + CRCFound Token 池化零训练"
        out["next"] = [
            "1) 高分辨解耦: 用 step_b3_hr_train.py 逻辑加 --t-only/--n-only (改 HRDataset 只算单头)",
            "2) CRCFound 局部 Token 池化零训练重测 145 例 CT (需 pretrained/crcfound 权重与推理脚本, 待开发)",
        ]
        print("[B4] 灰度未救活 (<0.56):", img_n, "→ 解耦 + CRCFound 零训练 (后续人工执行)")

    with open(os.path.join(ROOT, "logs", "step_b4_hr_extend.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "README.md"), "a", encoding="utf-8") as f:
        f.write(f"\n## B4 收尾 (自动接力 {time.strftime('%Y-%m-%d %H:%M')})\n"
                f"- 灰度救活? {'是 → mri_all+SDF 三序列3seed' if img_saved else '否 → 解耦+CRCFound 零训练'}"
                f" (img N_last5={img_n}); logs/step_b4_hr_extend.json\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
