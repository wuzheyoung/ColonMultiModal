# tools/gen_4060_md.py — 从 exp_4060_overnight_summary.json 生成汇总 .md
# 用法: python tools/gen_4060_md.py
# 支持嵌套: task3 = {"gated": {"d": res, "t": res, "clin_d_t": res}, "gated_sp": {...}}
#          task2 = {"label_N": {"direct_tcd": {...}, "direct_td": {...}}, "_25d_gain": {...}}
import json

SUM = "logs/exp_4060_overnight_summary.json"
OUT = "logs/exp_4060_overnight_summary.md"


def fmt(res):
    if res is None:
        return "n/a (样本不足)"
    return (f"n={res['n']} (pos {res['n_pos']}) AUC {res['pooled_auc']:.3f} "
            f"[{res['ci95'][0]:.3f},{res['ci95'][1]:.3f}] fold {res['fold_mean']:.3f}±{res['fold_sd']:.3f}")


def is_res(x):
    return isinstance(x, dict) and "pooled_auc" in x and "n" in x


def emit_view(lines, label, res):
    lines.append(f"- **{label}**: {fmt(res)}")


def render_res_block(lines, label, obj):
    """obj 是 {view: res} 或直接 res。"""
    if is_res(obj):
        emit_view(lines, label, obj)
    else:
        for view in ("tcd", "td"):
            if isinstance(obj, dict) and view in obj:
                emit_view(lines, f"{label}/{view}", obj[view])


def render_task(lines, name, body):
    lines.append(f"## {name}")
    lines.append("")
    if isinstance(body, dict):
        # 任务2 视图嵌套: label_N → direct_tcd/direct_td 或 _25d_gain
        for k, v in body.items():
            if k == "_25d_gain":
                lines.append("### 2.5D vs 2D 增益")
                lines.append("")
                for tgt, gg in v.items():
                    for seq, g25 in gg.get("25d", {}).items():
                        b = gg.get("2d_baseline", {}).get(seq, {})
                        a25 = g25.get("tcd")
                        a2 = b.get("tcd")
                        lines.append(f"- **{tgt} {seq}**: 2.5D tcd {fmt(a25)} | 2D tcd {fmt(a2)}")
                lines.append("")
            elif is_res(v) or (isinstance(v, dict) and any(x in v for x in ("tcd", "td"))):
                render_res_block(lines, k, v)
            elif isinstance(v, dict):
                # 子分组 (label_N → direct_tcd/direct_td; task3 → gated/gated_sp)
                for subk, subv in v.items():
                    if is_res(subv):
                        emit_view(lines, f"{k}/{subk}", subv)
                    elif isinstance(subv, dict) and any(is_res(x) for x in subv.values()):
                        # {view: res} 任意键映射 (如 direct_tcd → {clin_clean, d, t, clin_d_t})
                        for vk, vr in subv.items():
                            if is_res(vr):
                                emit_view(lines, f"{k}/{subk}/{vk}", vr)
                    else:
                        render_res_block(lines, f"{k}/{subk}", subv)
                lines.append("")
    # time_s / params 附加
    if isinstance(body, dict):
        if "time_s" in body:
            lines.append(f"- 耗时 {body['time_s']}s")
        if "params" in body:
            lines.append(f"- 参数量 {body['params']/1e6:.1f}M")
        lines.append("")


def main():
    d = json.load(open(SUM, encoding="utf-8"))
    tasks = d.get("tasks", d)
    lines = ["# exp_4060_overnight 夜间实验汇总 (2026-11-23)",
             "",
             "训练集: direct_all n=280 (master fold 5 折) | 评估: direct_tcd n=162 / direct_td n=214",
             ""]
    for k, v in tasks.items():
        if k.endswith("_time_s") or k == "task2_time_s":
            continue
        render_task(lines, k, v)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
