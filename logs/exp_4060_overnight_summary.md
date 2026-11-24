# exp_4060_overnight 夜间实验汇总 (2026-11-23)

训练集: direct_all n=280 (master fold 5 折) | 评估: direct_tcd n=162 / direct_td n=214

## task3

- **gated/d**: n=105 (pos 70) AUC 0.551 [0.429,0.666] fold 0.601±0.077
- **gated/t**: n=155 (pos 105) AUC 0.596 [0.498,0.688] fold 0.630±0.040
- **gated/clin_d_t**: n=162 (pos 77) AUC 0.543 [0.451,0.631] fold 0.543±0.101

- **gated_sp/d**: n=105 (pos 70) AUC 0.541 [0.421,0.663] fold 0.591±0.079
- **gated_sp/t**: n=155 (pos 105) AUC 0.604 [0.505,0.699] fold 0.635±0.051
- **gated_sp/clin_d_t**: n=162 (pos 77) AUC 0.540 [0.448,0.629] fold 0.547±0.093


## task4_e2e_t

- **res/tcd**: n=155 (pos 105) AUC 0.434 [0.335,0.535] fold 0.531±0.085
- **res/td**: n=155 (pos 105) AUC 0.434 [0.335,0.535] fold 0.531±0.085
- 耗时 12145.0s
- 参数量 87.0M

## task4_e2e_d

- **res/tcd**: n=105 (pos 70) AUC 0.599 [0.479,0.712] fold 0.637±0.120
- **res/td**: n=154 (pos 108) AUC 0.552 [0.452,0.651] fold 0.601±0.099
- 耗时 13059.1s
- 参数量 87.0M

## task2

- **label_N/direct_tcd/clin_clean**: n=162 (pos 77) AUC 0.483 [0.390,0.570] fold 0.493±0.037
- **label_N/direct_tcd/d**: n=105 (pos 48) AUC 0.467 [0.355,0.583] fold 0.482±0.103
- **label_N/direct_tcd/t**: n=155 (pos 72) AUC 0.527 [0.436,0.618] fold 0.541±0.081
- **label_N/direct_tcd/clin_d_t**: n=162 (pos 77) AUC 0.512 [0.424,0.604] fold 0.524±0.085
- **label_N/direct_td/clin_clean**: n=214 (pos 98) AUC 0.479 [0.398,0.558] fold 0.496±0.036
- **label_N/direct_td/d**: n=154 (pos 69) AUC 0.428 [0.339,0.525] fold 0.445±0.094
- **label_N/direct_td/t**: n=201 (pos 91) AUC 0.502 [0.427,0.583] fold 0.510±0.079
- **label_N/direct_td/clin_d_t**: n=214 (pos 98) AUC 0.476 [0.399,0.556] fold 0.485±0.059

### 2.5D vs 2D 增益

- **label_T d**: 2.5D tcd n=105 (pos 70) AUC 0.640 [0.525,0.754] fold 0.638±0.117 | 2D tcd n=105 (pos 70) AUC 0.629 [0.514,0.744] fold 0.618±0.088
- **label_T t**: 2.5D tcd n=155 (pos 105) AUC 0.630 [0.529,0.721] fold 0.640±0.053 | 2D tcd n=155 (pos 105) AUC 0.629 [0.530,0.722] fold 0.641±0.099
- **label_N d**: 2.5D tcd n=105 (pos 48) AUC 0.490 [0.386,0.607] fold 0.489±0.175 | 2D tcd n=105 (pos 48) AUC 0.467 [0.355,0.583] fold 0.482±0.103
- **label_N t**: 2.5D tcd n=155 (pos 72) AUC 0.571 [0.477,0.657] fold 0.560±0.077 | 2D tcd n=155 (pos 72) AUC 0.527 [0.436,0.618] fold 0.541±0.081

