# -*- coding: utf-8 -*-
"""解析最新 eval_report_full_* (json+txt) 并输出标准化的 T/N 全指标汇报表。

用法 (Pytorch 环境):
  python tools/report_run.py [--path logs/eval_report_full_s42.json]
"""
import argparse
import glob
import json
import os
import re


def _latest_json():
    files = sorted(glob.glob("logs/eval_report_full_*.json"), key=os.path.getmtime)
    return files[-1] if files else None


def _latest_txt():
    files = sorted(glob.glob("logs/eval_report_full_*.txt"), key=os.path.getmtime)
    return files[-1] if files else None


def _parse_txt(path):
    """txt 报告: 提取 T/N 校准段 (实际格式见 [4.0]/[4.1]:
    "阈值校准: 阈值=0.5  macro_f1=0.5960  accuracy=0.6528  per_class_recall=[...]")。"""
    if not path or not os.path.isfile(path):
        return {}
    text = open(path, encoding="utf-8").read()
    out = {}
    blocks = re.findall(
        r"阈值校准:\s*阈值=([0-9.]+)\s+macro_f1=([0-9.]+)\s+accuracy=([0-9.]+)", text)
    if blocks:
        thr, f1, acc = blocks[0]
        out.update({"cal_t_threshold": float(thr), "cal_t_macro_f1": float(f1),
                    "cal_t_acc": float(acc)})
    if len(blocks) > 1:
        thr, f1, acc = blocks[1]
        out.update({"cal_n_threshold": float(thr), "cal_n_macro_f1": float(f1),
                    "cal_n_acc": float(acc)})
    # 兼容旧命名的 key=value 写法
    for key in ("best_threshold_t", "best_threshold_n"):
        m = re.search(rf"{key}\s*[:=]\s*([0-9.]+)", text)
        if m:
            out[key] = float(m.group(1))
    recalls = re.findall(r"(?:per_class_recall|recall)\s*=\s*\[([^\]]*)\]", text)
    if recalls:
        out["recall_t"] = recalls[0]
        if len(recalls) > 1:
            out["recall_n"] = recalls[1]
    return out


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    args = ap.parse_args()
    jp = args.path or _latest_json()
    if not jp:
        print("无 eval_report_full json")
        return
    d = json.load(open(jp, encoding="utf-8"))
    res = d.get("result", {})
    meta = d.get("meta", {})
    params = d.get("params", {})
    print("=" * 62)
    print(f"运行: {meta.get('branch')} | run={meta.get('run')} | {meta.get('time')}")
    print(f"配置: backbone={params.get('backbone_name')} "
          f"pool={params.get('pool_mode')} head_width={params.get('head_width')} "
          f"lr={params.get('lr')} epochs={params.get('epochs')} "
          f"mixup={params.get('mixup')} use_aug={params.get('use_augmentation')}")
    print(f"数据: train={d.get('data', {}).get('train_n')} "
          f"val={d.get('data', {}).get('val_n')}")
    print("-" * 62)
    print(f"T 任务: best_epoch={res.get('best_epoch')} "
          f"macro_f1={_fmt(res.get('best_macro_f1'))} acc={_fmt(res.get('best_acc'))}")
    print(f"N 任务: best_epoch={res.get('best_epoch_n')} "
          f"macro_f1={_fmt(res.get('best_macro_f1_n'))} acc={_fmt(res.get('best_acc_n'))}")
    txt = _parse_txt(args.path.replace('.json', '.txt') if args.path else _latest_txt())
    if txt:
        print(f"校准T: thr={_fmt(txt.get('cal_t_threshold'), 2)} "
              f"macro_f1={_fmt(txt.get('cal_t_macro_f1'))} acc={_fmt(txt.get('cal_t_acc'))}")
        print(f"校准N: thr={_fmt(txt.get('cal_n_threshold'), 2)} "
              f"macro_f1={_fmt(txt.get('cal_n_macro_f1'))} acc={_fmt(txt.get('cal_n_acc'))}")
    else:
        print("校准段: — (txt 缺失或格式不匹配)")
    print("=" * 62)


if __name__ == "__main__":
    main()
