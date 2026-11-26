# auto_staging_opt 自主迭代实验进度日志

## 阶段0 (锁基线, 2026-11-26T00:38:34)

### direct_td
- clin: 10×5 Mean AUC 0.6270 ± 0.0185 (min 0.6087, max 0.6651), fold50 0.6407±0.085
- clin_t: 10×5 Mean AUC 0.6620 ± 0.0282 (min 0.6164, max 0.7110), fold50 0.6912±0.0714
- clin_dt: 10×5 Mean AUC 0.6763 ± 0.0302 (min 0.6395, max 0.7278), fold50 0.7046±0.0768
### direct_tcd
- clin: 10×5 Mean AUC 0.6300 ± 0.0181 (min 0.6056, max 0.6660), fold50 0.6363±0.0915
- clin_t: 10×5 Mean AUC 0.6782 ± 0.0274 (min 0.6420, max 0.7201), fold50 0.7±0.08
- clin_dt: 10×5 Mean AUC 0.6935 ± 0.0357 (min 0.6435, max 0.7409), fold50 0.7134±0.0887
- stage1 baseline_0 (t120awm+dself+clin, td): td 0.6763±0.0302 | tcd 0.6935±0.0357
- stage1 H1.1 (t60awm+dself+clin): td 0.7003±0.0298 | tcd 0.7099±0.0323
- stage1 H1.2 baseline+phys_scalars(t60): td 0.6767±0.0266 | tcd 0.7081±0.0282
- stage1 H1.2 H1.1+phys_scalars(t60): td 0.691±0.0272 | tcd 0.7141±0.0276
- stage1 H1.3 t60 Top3+dself+clin: td 0.6953±0.0351 | tcd 0.7004±0.0343
- stage2 b0 t120 dself+clin+t: td 0.6763±0.0302 | tcd 0.6935±0.0357
- stage2 b0 t120 dall(self+ft)+clin+t: td 0.6627±0.0316 | tcd 0.6813±0.0384
- stage2 b0 t120 dall+is_d_self ind+clin+t: td 0.6611±0.031 | tcd 0.6802±0.0388
- stage2 h11 t60 dself+clin+t: td 0.7003±0.0298 | tcd 0.7099±0.0323
- stage2 h11 t60 dall(self+ft)+clin+t: td 0.6902±0.0202 | tcd 0.6998±0.0286
- stage2 h11 t60 dall+is_d_self ind+clin+t: td 0.6898±0.022 | tcd 0.699±0.0287
- stage3 H3.1 t(intra_patch)+dself+clin: td 0.6849±0.0364 | tcd 0.698±0.037
- stage3 H3.2 t(intra+peri)+dself+clin: td 0.6985±0.0272 | tcd 0.7076±0.0313
- stage3 ref t(cls)+dself+clin: td 0.6526±0.0332 | tcd 0.6587±0.0268
- stage3 H3.3 t(cls)+d(intra)+clin: td 0.6526±0.0332 | tcd 0.6587±0.0268
- stage3 H3.3 t(intra+peri)+d(intra)+clin: td 0.6985±0.0272 | tcd 0.7076±0.0313
- stage4 rad_only(intra): td 0.5576±0.0261 | tcd 0.5876±0.0328
- stage4 clin+rad(intra): td 0.615±0.0213 | tcd 0.6436±0.0269
- stage4 rad_only(rim_3mm): td 0.5911±0.0215 | tcd 0.6243±0.0263
- stage4 clin+rad(rim_3mm): td 0.6494±0.0116 | tcd 0.6785±0.0158
- stage4 rad_only(rim_5mm): td 0.5997±0.0231 | tcd 0.6356±0.026
- stage4 clin+rad(rim_5mm): td 0.6482±0.0139 | tcd 0.6786±0.017
- stage4 H4.2 clin+rad(intra)+t60+dself: td 0.6676±0.0265 | tcd 0.7005±0.0258
- stage4 H4.2 clin+rad(rim_3mm)+t60+dself: td 0.6816±0.0225 | tcd 0.7125±0.0278
- stage4 H4.2 clin+rad(rim_5mm)+t60+dself: td 0.6804±0.0212 | tcd 0.7137±0.0269
- stage5 stack l2 (clin,t60,dself,rad3): td 0.627±0.0735 | tcd 0.6425±0.0838
- stage5 stack enet (clin,t60,dself,rad3): td 0.6595±0.0634 | tcd 0.6753±0.0779
- stage5 stack svm (clin,t60,dself,rad3): td 0.6611±0.0599 | tcd 0.6761±0.0726
- stage5 stack l2 (clin,t60,dself): td 0.6472±0.0667 | tcd 0.6732±0.0736
- stage5 stack l2 (t60,dself): td 0.6017±0.0816 | tcd 0.643±0.0992
