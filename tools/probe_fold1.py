# -*- coding: utf-8 -*-
# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""定位 5-fold 卡死源: 对 Fold 1 train+val 逐样本 GPU 前向, 报告每个样本耗时。
watchdog: 单样本前向 >60s → 打印 SUSPECT patient_id 并 os._exit(1)。
用法: D:\\Development\\miniconda\\envs\\Pytorch\\python.exe tools/probe_fold1.py
"""
import os, sys, time, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
from config.hos import HosConfig
from train_hos_seeds import build_model
from data_loader.loader import build_dataloaders

cfg = HosConfig()
cfg.backbone_name = "medicalnet18"
cfg.pool_mode = "gem"
cfg.head_width = 128
cfg.n_from_shallow = True
cfg.n_shallow_idx = 3
cfg.pretrained_backbone = True
cfg.use_augmentation = False
cfg.batch_size = 1

model = build_model(cfg).eval().cuda()
ckpt = torch.load("checkpoints_hos_full/best_model_hos_full_s0.pth", map_location="cpu")
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]
model.load_state_dict(ckpt, strict=False)

current = {"pid": None, "t0": time.time()}

def watchdog():
    while True:
        time.sleep(65)
        if current["pid"] is not None and time.time() - current["t0"] > 60:
            print(f"SUSPECT patient_id={current['pid']} forward >60s, killing", flush=True)
            os._exit(1)

threading.Thread(target=watchdog, daemon=True).start()

def run_loader(name, loader):
    with torch.no_grad():
        for i, batch in enumerate(loader):
            pid = batch["patient_id"][0] if hasattr(batch["patient_id"], "__getitem__") else batch["patient_id"]
            current["pid"] = pid
            current["t0"] = time.time()
            t0 = time.time()
            # T 主任务 logits (reuse trainer-style forward)
            outs = model(
                ct=batch["ct"].cuda(), mri=batch["mri"].cuda(),
                clinical=batch["clinical"].cuda(), ct_mask=batch["ct_available"].cuda(),
                mri_mask=batch["mri_available"].cuda(),
                mri_seq_mask=batch["mri_seq_available"].cuda(),
            )
            dt = time.time() - t0
            if dt > 5:
                print(f"[SLOW] {name} #{i} pid={pid} forward={dt:.2f}s", flush=True)
            else:
                print(f"[OK]   {name} #{i} pid={pid} forward={dt:.2f}s", flush=True)

train_loader, val_loader = build_dataloaders(cfg, train_csv="data/hos_full_kfold_f1_train.csv",
                                             val_csv="data/hos_full_kfold_f1_val.csv")
print("=== Fold1 TRAIN (285) ===", flush=True)
run_loader("train", train_loader)
print("=== Fold1 VAL (71) ===", flush=True)
run_loader("val", val_loader)
print("PROBE DONE — no suspect sample", flush=True)
