class Config:
    # ── 数据 ──
    # TCGA-COAD/READ CT 数据集：由 build_tcga_index.py + split_tcga.py 生成
    # 分层划分 train 18 / val 5（仅含有 AJCC 分期的 23 例）
    train_csv = "./data/tcga_train.csv"
    val_csv = "./data/tcga_val.csv"
    clinical_cols = []          # 暂时跳过临床特征
    label_col = "label_bin"     # 二分类：早期(I/II)->0, 晚期(III/IV)->1
    spatial_size = (96, 96, 96)

    # ── 模型 ──
    backbone_name = "resnet18"  # "resnet18" / "resnet50" / "swin"
    pretrained_backbone = True  # 加载 MedicalNet(Med3D) 预训练权重做迁移学习
    fusion_mode = "single"      # "single" / "multi"
    num_classes = 2             # 二分类：早期 vs 晚期
    num_clinical = 1            # 临床占位维度（零向量），保持 forward 接口
    swin_feature_size = 48
    num_fuse_stages = 3

    # ── 模态内 self-attention（方案 B）──
    use_self_attn = True
    self_attn_layers = 1
    self_attn_heads = 4

    # ── 缺失模态处理（Imputer + mask token + 模态 dropout）──
    use_imputer = True
    imputer_hidden_dim = 256
    # TCGA 无真实 MRI，强制走 CT-only 路径看单模态 CT 效果
    modality_dropout = {"ct_only": 1.0, "mri_only": 0.0, "both": 0.0}

    # ── 多任务 loss 权重（辅助分类 / 重建对齐）──
    loss_weights = {"aux": 0.3, "rec": 0.5}

    # ── 训练 ──
    batch_size = 8              # 数据已全内存缓存，加大 batch 提升 GPU 利用率
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 50
    num_workers = 0            # 小数据集 + 内存缓存：单进程让缓存跨 epoch 生效最快
    pin_memory = True          # CPU->GPU 传输提速
    use_amp = True             # 混合精度 (autocast + GradScaler)，4060 上提速省显存
    device = "cuda"
    seed = 42
    save_dir = "./checkpoints"
    plot_dir = "./plots"        # 训练曲线 / 混淆矩阵输出目录
    class_names = ["early(I/II)", "late(III/IV)"]  # 混淆矩阵坐标标签 (二分类)
