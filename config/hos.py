from .default import Config


# round7 分支实验定义: 分支名 → 覆盖的 cfg 字段。
# - MRI 分支强制 ct_only=1.0 (CT 全走 imputer 伪 CT), mri_sequences 取对应子集
# - CT 分支强制 mri_only=1.0 (MRI 全走 imputer 伪 MRI), ct_seq 取平扫/增强
# - full: 仅切换 csv, 保留默认模态 dropout 与三序列
HOS_BRANCHES = {
    "mri_t":   {"mri_sequences": ("t",),
                "modality_dropout": {"ct_only": 1.0, "mri_only": 0.0, "both": 0.0}},
    "mri_c":   {"mri_sequences": ("c",),
                "modality_dropout": {"ct_only": 1.0, "mri_only": 0.0, "both": 0.0}},
    "mri_d":   {"mri_sequences": ("d",),
                "modality_dropout": {"ct_only": 1.0, "mri_only": 0.0, "both": 0.0}},
    "mri_tdc": {"mri_sequences": ("t", "c", "d"),
                "modality_dropout": {"ct_only": 1.0, "mri_only": 0.0, "both": 0.0}},
    "ct_p":    {"ct_seq": "plain",
                "modality_dropout": {"ct_only": 0.0, "mri_only": 1.0, "both": 0.0},
                "mri_sequences": ("t",)},
    "ct_z":    {"ct_seq": "enhanced",
                "modality_dropout": {"ct_only": 0.0, "mri_only": 1.0, "both": 0.0},
                "mri_sequences": ("t",)},
    "full":    {},
}


class HosConfig(Config):
    """data/Hos 多模态 (CT+MRI+临床+病理T/N) 训练配置。"""
    # ── 数据 ──
    train_csv = "./data/hos_pairs.csv"
    val_csv = "./data/hos_pairs.csv"
    label_col = "label_T"                 # 主任务: 病理T 二分类 (pT1-2=0, pT3-4=1)
    secondary_label_col = "label_N"       # 次任务: 病理N 二分类 (pN0=0, pN+=1)
    multi_task = True
    clinical_cols = ["age", "sex", "ca199", "cea", "albumin", "prealbumin", "hb"]
    num_clinical = 7
    spatial_size = (96, 96, 96)

    # ── 模型 ──
    backbone_name = "resnet18"
    pretrained_backbone = True
    # ── round8 A2: MedicalNet 3D 医学预训练 (resnet{10,18}_23dataset.pth 放 pretrained/) ──
    # backbone_name 可选 "medicalnet10"/"medicalnet18"（23 数据集 3D 医学预训练,
    # 替代 ImageNet-2D→3D 广播权重）; pretrained_backbone=True 时自动从 pretrained_dir 加载
    pretrained_dir = "./pretrained"
    # 多尺度融合: 融合 resnet 最后 2 个 stage (12³ + 6³), 保留局部(壁厚/浸润)与全局信息
    # 注: num_fuse_stages=3 会引入 24³ 层, cross-attn token 数过大, 故取 2
    fusion_mode = "multi"
    num_fuse_stages = 2
    num_classes = 2
    num_tasks = 2                         # 双头: T + N
    use_self_attn = True
    self_attn_layers = 1
    self_attn_heads = 4
    use_imputer = True
    imputer_hidden_dim = 256

    # ── round7 分支训练参数 ──
    # MRI 序列子集（单序列分支只建对应 backbone；全量训练保持 t/c/d 三序列）
    mri_sequences = ("t", "c", "d")
    # CT 序列选择: "auto"(原优先级 z>p>特殊名) / "plain"(平扫) / "enhanced"(增强)
    ct_seq = "auto"

    # ── 缺失模态 ──
    # Hos: batch3 后 CT/MRI 覆盖大幅提升, 缺失模态仍走 imputer 补全
    # 三键语义 (归一化后采样): ct_only=只丢 CT / mri_only=只丢 MRI / both=两模态都保留
    modality_dropout = {"ct_only": 0.20, "mri_only": 0.15, "both": 0.65}

    # ── 多任务 loss 权重 ──
    # 旧值 {"aux": 0.3, "rec": 0.5}: 重建/辅助损失梯度占比高; 4 组对照(seed42)
    #   已验证此为最优: best macro-F1 0.6043, 而 aux/rec=0.2/0.3 -> 0.5241, 0.1/0.1 -> 0.5241
    # 结论: 重建正则权重不可削减, 维持旧值
    loss_weights = {"aux": 0.3, "rec": 0.5}

    # ── 正则化 / 类不平衡 (针对 T 多数类塌缩) ──
    # class_weight_mode: "inverse"=满逆频率 / "sqrt"=√逆频率(温和) / "none"=不加权
    class_weight_mode = "inverse"   # T/N 两个任务都按各自标签的满逆频率加权 (compute_class_weights_n 同模式)
    use_focal_loss = False          # True 时 T 主任务改用 Focal Loss (alpha=类别权重)
    focal_gamma = 2.0
    label_smoothing = 0.1
    head_dropout = 0.5
    # ── round8 A3/A4: 池化方式与主分类头宽度 ──
    pool_mode = "avg"          # "avg"=全局平均池化 / "gem"=GeM 广义均值池化(细粒度特征友好)
    gem_p = 3.0                # GeM 幂指数 (pool_mode="gem" 时生效)
    head_width = 256           # 主分类头(T/N)隐层宽度, 小样本可减窄至 128 降低过拟合

    # ── 结构减负: 冻结 backbone 低层 (降低可训练参数量, 缓解小样本过拟合) ──
    # 作用于 CT/MRI 两个 backbone 的 encoder 子模块; 置空列表 = 不冻结
    # 注: resnet18-3D 低层参数少, 建议至少冻结到 layer3 才有明显减负效果
    freeze_backbone_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"]

    # ── 评估: 在验证集上扫描概率阈值校准 T/N 决策边界 ──
    # 已知未修 (2026-11-20): 阈值在 val 上选取再报告 val 指标 → 存在乐观偏差, 按用户要求保留现状
    calibrate_thresholds = True

    # ── 复现性: True 时启用 cudnn.deterministic/关闭 benchmark (CLI --deterministic) ──
    deterministic = False

    # ── round9 F1: 病灶融合分支 (Task010 mask 质心裁剪) ──
    lesion_dir = "./data/masks_task010"   # None=关闭病灶分支; mask 缺失样本走零卷+标志
    lesion_size = 64                      # 质心中心裁剪边长 (96³ 卷内取 64³, 含边界上下文)
    lesion_backbone = "medicalnet10"      # 病灶分支 backbone (轻量, 避免 8GB 显存超限)
    lesion_head_width = 256               # 病灶嵌入宽度
    # round9 MRI-t ROI: MRI-t 结肠质心裁剪卷 (total_mr mask 目录)
    mri_lesion_dir = "./data/masks_totalseg_mr_t"
    mri_lesion_size = 64
    mri_lesion_backbone = "medicalnet10"
    mri_lesion_head_width = 256

    # ── round8: 训练随机增强开关 (False=关闭, 对照「增强过强导致 T 学不动」假设) ──
    use_augmentation = True

    # ── 训练 ──
    batch_size = 8
    lr = 1e-4
    weight_decay = 5e-4
    epochs = 60
    num_workers = 0
    pin_memory = True
    use_amp = True
    device = "cuda"
    seed = 42
    # 2026-11-21 权重清理: 该目录权重已清空, 下次训练重新生成
    save_dir = "./checkpoints_hos"
    plot_dir = "./plots_hos"
    class_names = ["pT1-2", "pT3-4"]      # 混淆矩阵坐标 (T 二分类)
    n_class_names = ["pN0", "pN+"]        # N 任务独立类名 (避免报告误用 T 的 pT 标签)
    optimizer_name = "AdamW"              # 优化器 (写入评估报告参数段)
    scheduler_name = "CosineAnnealingLR"  # 学习率调度 (写入评估报告参数段)
