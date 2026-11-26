# final_autonomous_report — 4060 端 6 阶段自主迭代 (10×5 重复 CV)

生成时间: 2026-11-26T01:31:00

## 0. 基线 (阶段0)
- 锁定基线数字澄清: **direct_td n=214 基线 = 0.6763±0.0302**; 用户指令所称 0.6935±0.0357 实为 direct_tcd n=162 口径 (两口径均精确复现 match<0.001)。
- Baseline_0 (direct_td): clin_clean + d_self(MeanMax) + t(120mm AWM), 策略B + bonly 校准 → **0.6763±0.0302**

## 1. 阶段1 异构视野/物理尺度/浸润聚合
| 方案 | td 10×5 | tcd 10×5 | 判定 |
|---|---|---|---|
| Baseline_0 | 0.6763±0.0302 | 0.6935±0.0357 | 基准 |
| **H1.1 t→tight60_m10 AWM** | **0.7003±0.0298** | 0.7099±0.0323 | ✅ 采纳 (+0.024) |
| H1.2 baseline+物理标量 | 0.6767±0.0266 | 0.7081±0.0282 | ❌ 无增益 |
| H1.2 H1.1+物理标量 | 0.6910±0.0272 | 0.7141±0.0276 | ❌ 反而-0.009 |
| H1.3 t60 Top3 | 0.6953±0.0351 | 0.7004±0.0343 | ❌ 略逊 AWM |

## 2. 阶段2 DWI from_t 纳回
| 方案 | td | 判定 |
|---|---|---|
| b0 d_self | 0.6763±0.0302 | 基准 |
| b0 +from_t | 0.6627±0.0316 | ❌ -0.014 |
| b0 +from_t+is_d_self | 0.6611±0.031 | ❌ -0.015 |
| H1.1 d_self | 0.7003±0.0298 | 基准 |
| H1.1 +from_t | 0.6902±0.0202 | ❌ -0.010 |
| H1.1 +from_t+is_d_self | 0.6898±0.022 | ❌ -0.011 |
→ **阶段2 拒绝**: from_t 借壳掩码纳回全面劣化, is_d_self 指示符无增益。

## 3. 阶段3 DINOv2 Patch Token 瘤内/瘤周双流
| 方案 | td | tcd | 判定 |
|---|---|---|---|
| H3.1 t(intra) | 0.6849±0.0364 | 0.698 | ❌ 低于 H1.1 |
| **H3.2 t(intra+peri)** | 0.6985±0.0272 | 0.7076±0.0313 | ≈H1.1 持平 |
| ref t(cls) | 0.6526±0.0332 | 0.6587 | 参考 |
| H3.3 d(intra) | 0.6526 | 0.6587 | ❌ d 无增益 |
→ **阶段3 拒绝**: patch token 双流未超越 tight60 CLS AWM (H1.1)。

## 4. 阶段4 放射组学 (3D 形态 + 瘤周环 3/5mm)
| 方案 | td | 判定 |
|---|---|---|
| rad_only(intra) | 0.5576±0.0261 | ❌ 弱 |
| clin+rad(rim_3mm) | 0.6494±0.0116 | 弱于 H1.1 |
| H4.2 clin+rad(rim_3mm)+t60+dself | 0.6816±0.0225 | ❌ 低于 H1.1 0.7003 |
| H4.2 clin+rad(rim_5mm)+t60+dself | 0.6804±0.0212 | ❌ 低于 H1.1 |
→ **阶段4 拒绝**: 放射组学在三模态融合无增益, 甚至拖累。

## 5. 阶段5 分模态 Stacking + 贪心准入
| 元模型 | td | 判定 |
|---|---|---|
| stack l2 (4模态) | 0.627±0.0735 | ❌ 大幅劣化 |
| stack enet (4模态) | 0.6595±0.0634 | ❌ |
| stack svm+platt (4模态) | 0.6611±0.0599 | ❌ |
| stack l2 (clin,t60,dself) | 0.6472±0.0667 | ❌ |
| stack l2 (t60,dself) | 0.6017±0.0816 | ❌ |
→ **阶段5 拒绝**: 概率级 Stacking 全部劣化 (小样本过拟合+概率相关性), 策略B 简单融合最优。贪心准入: 无方案满足 +0.004/胜率60%/ECE≤0.065 → **Winner_Model = H1.1**。

## 6. 错例诊断 (Winner_Model, td Youden 阈值)
- AUC 0.716, 阈值 0.72, 混淆矩阵 {'TP': 62, 'FP': 6, 'FN': 86, 'TN': 60}
- 假阴 (FN, n=86): max_area 中位/均值/n = (1977.8, 1918.1, 78), DWI 缺失率 0.442, 临床中位 {'age': 62.0, 'sex': 1.0, 'ca199': 9.3, 'cea': 3.8, 'albumin': 39.0, 'prealbumin': 188.0, 'hb': 127.0}
- 假阳 (FP, n=6): max_area 中位/均值/n = (2023.3, 1691.7, 6), DWI 缺失率 0.667, 临床中位 {'age': 72.0, 'sex': 1.0, 'ca199': 18.79, 'cea': 8.48, 'albumin': 36.5, 'prealbumin': 198.5, 'hb': 130.0}
- 真阳 (TP, n=62): max_area (2344.1, 2217.9, 60), DWI 缺失率 0.661
- 真阴 (TN, n=60): max_area (1813.9, 1757.7, 57), DWI 缺失率 0.45

## 7. Winner_Model 精确配置
```
模态:  clin_clean (7维, 折内 median impute + std + 嵌套选C L2-LR)
       t:    features/slice_feats_v2_tight60_m10/dinov2/t, AreaWeightedMean 池化 (mask_area_mm2 权重)
       d:    features/slice_feats_v2/dinov2/d, MeanMax 池化 (仅 d_self 104 例)
融合:  策略B = 0.5*z_clin + 0.5*mean(z_t, z_d) (fuse_zlogit_strategy)
校准:  bonly (platt_fit_bonly 逐折)
评估:  10×5 重复分层 CV (StratifiedKFold 5折 shuffle seed0-9), Pooled OOF AUC
结果:  direct_td 0.7003±0.0302 | direct_tcd 0.7099±0.0323
```

## 8. 结论
- 唯一明确增益: t 序列裁剪视野 120mm→tight60_m10 (bbox 中点+对称补黑, 边长=max(60,span+20)), +0.024 (0.6763→0.7003)。
- 物理尺度标量、DWI from_t 纳回、Patch Token 双流、放射组学、Stacking 均无增益或劣化。
- 保留 H1.1 为 Winner_Model, 未触碰 20250428 外部验证集与 nCRT 队列 (铁律3)。
