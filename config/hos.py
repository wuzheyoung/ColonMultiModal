from .default import Config


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

    # ── 缺失模态 ──
    # Hos: 39 例双模态 / 24 例仅 CT / 92 例仅 MRI，缺失模态走 imputer 补全
    modality_dropout = {"ct_only": 0.20, "mri_only": 0.15, "both": 0.65}

    # ── 多任务 loss 权重 ──
    # 旧值 {"aux": 0.3, "rec": 0.5}: 重建/辅助损失梯度占比高; 4 组对照(seed42)
    #   已验证此为最优: best macro-F1 0.6043, 而 aux/rec=0.2/0.3 -> 0.5241, 0.1/0.1 -> 0.5241
    # 结论: 重建正则权重不可削减, 维持旧值
    loss_weights = {"aux": 0.3, "rec": 0.5}

    # ── 正则化 / 类不平衡 (针对 T 多数类塌缩) ──
    # class_weight_mode: "inverse"=满逆频率 / "sqrt"=√逆频率(温和) / "none"=不加权
    class_weight_mode = "inverse"   # T 主任务按满逆频率加权 (N 分布均衡, 不加权)
    use_focal_loss = False          # True 时 T 主任务改用 Focal Loss (alpha=类别权重)
    focal_gamma = 2.0
    label_smoothing = 0.1
    head_dropout = 0.5

    # ── 结构减负: 冻结 backbone 低层 (降低可训练参数量, 缓解小样本过拟合) ──
    # 作用于 CT/MRI 两个 backbone 的 encoder 子模块; 置空列表 = 不冻结
    # 注: resnet18-3D 低层参数少, 建议至少冻结到 layer3 才有明显减负效果
    freeze_backbone_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"]

    # ── 评估: 在验证集上扫描概率阈值校准 T/N 决策边界 ──
    calibrate_thresholds = True

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
    save_dir = "./checkpoints_hos"
    plot_dir = "./plots_hos"
    class_names = ["pT1-2", "pT3-4"]      # 混淆矩阵坐标 (T 二分类)
