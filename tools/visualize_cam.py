# 2026-11-21 权重清理: 本脚本默认引用的实验 ckpt 已删除, 需先重跑生成; 暂不可直接执行。
"""Grad-CAM 热图可视化: 验证模型注意力是否聚焦肿瘤区域 (round9-CAM)。

用法:
  python tools/visualize_cam.py [--ckpt checkpoints_hos_full/best_model_hos_full_s42.pth]
      [--n-samples 6] [--out-dir plots_hos_full/cam] [--device cuda]

对 val 集前 N 个样本, 用 T 主任务 logits 对 CT/MRI 各自 layer3 特征图做
Grad-CAM, 热图上采样回 96^3 并叠加原图中间轴位切片, 保存 PNG。
"""
import argparse, os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.hos import HosConfig
from train_hos_seeds import build_model, build_dataloaders


def _hook_outputs(module, inp, out):
    _hook_outputs.storage.append(out.detach())


def _hook_grads(module, grad_in, grad_out):
    _hook_grads.storage.append(grad_out[0].detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="checkpoints_hos_full/best_model_hos_full_s42.pth")
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--out-dir", type=str, default="plots_hos_full/cam")
    ap.add_argument("--save-npy", type=str, default=None,
                    help="目录: 额外保存 3D CAM 数组 cam_ct_XXX.npy / cam_mri_XXX.npy")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--branch", type=str, default="full")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="用 data/hos_{branch}_val_s{split}.csv 的验证集样本 (旧版用全队列 csv, 会含训练病人)")
    args = ap.parse_args()

    cfg = HosConfig()
    cfg.backbone_name = "medicalnet18"
    cfg.pool_mode = "gem"
    cfg.head_width = 128
    cfg.n_from_shallow = True
    cam_csv = f"./data/hos_{args.branch}_val_s{args.split_seed}.csv"
    if not os.path.exists(cam_csv):
        raise SystemExit(f"[CAM] 缺少验证集 csv: {cam_csv}")
    cfg.train_csv = cam_csv
    cfg.val_csv = cam_csv
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"[CAM] 缺少 ckpt: {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_npy:
        os.makedirs(args.save_npy, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device}, ckpt={args.ckpt}")

    _, val_loader = build_dataloaders(cfg, cfg.train_csv, cfg.val_csv)
    model = build_model(cfg).to(device)
    # 剔除 N 头权重: B1/B2 变体 checkpoint 的 N 头维度可能不同, CAM 只分析 T 头
    sd = torch.load(args.ckpt, map_location=device)
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier_n")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={missing} unexpected={unexpected}")
    model.eval()

    # Grad-CAM 目标: CT / MRI(序列 t) 的 layer3 特征图
    ct_hook_mod = model.ct_backbone.encoder.layer3
    mri_t_mod = (model.mri_shared_backbone if getattr(model, "mri_shared_backbone", None)
                 else model.mri_backbones[model.mri_sequences[0]])
    mri_hook_mod = mri_t_mod.encoder.layer3
    ct_hook = ct_hook_mod.register_forward_hook(_hook_outputs)
    ct_ghook = ct_hook_mod.register_full_backward_hook(_hook_grads)
    mri_hook = mri_hook_mod.register_forward_hook(_hook_outputs)
    mri_ghook = mri_hook_mod.register_full_backward_hook(_hook_grads)

    def gradcam(feat, grad):
        cam = (grad * feat).mean(dim=1).clamp(min=0)          # [B,...]
        cam = cam / (cam.flatten(1).max(dim=1).values.view(-1, *([1] * (cam.dim() - 1))) + 1e-8)
        return F.interpolate(cam.unsqueeze(1), size=(96, 96, 96),
                             mode="trilinear", align_corners=False)[:, 0]  # [B,96,96,96]

    cmap = plt.get_cmap("jet")
    n_done = 0
    for batch in val_loader:
        ct = batch["ct"].to(device)
        mri = batch["mri"].to(device)
        clinical = batch["clinical"].to(device)
        for i in range(ct.shape[0]):
            _hook_outputs.storage = []
            _hook_grads.storage = []
            ct_i, mri_i, clin_i = ct[i:i + 1], mri[i:i + 1], clinical[i:i + 1]
            ct_i.requires_grad_(True)
            mri_i.requires_grad_(True)
            out = model(ct_i, mri_i, clin_i)
            main = out["main"][0]
            cls = int(main.argmax().item())
            model.zero_grad()
            main[cls].backward()

            ct_feat = _hook_outputs.storage[0]   # 0=ct 1=mri
            mri_feat = _hook_outputs.storage[1]
            ct_grad = _hook_grads.storage[0]
            mri_grad = _hook_grads.storage[1]
            cam_ct = gradcam(ct_feat, ct_grad)[0].cpu().numpy()
            cam_mri = gradcam(mri_feat, mri_grad)[0].cpu().numpy()
            if args.save_npy:
                np.save(os.path.join(args.save_npy, f"cam_ct_{n_done:03d}.npy"), cam_ct)
                np.save(os.path.join(args.save_npy, f"cam_mri_{n_done:03d}.npy"), cam_mri)

            slice_idx = 48
            ct_slice = ct_i[0, 0, slice_idx].detach().cpu().numpy()
            mri_slice = mri_i[0, 0, slice_idx].detach().cpu().numpy()

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for a, (img, cam, name) in enumerate(
                    [(ct_slice, cam_ct, "CT"), (mri_slice, cam_mri, "MRI-t")]):
                a_img = img / (img.max() + 1e-8)
                a_cam = cmap(cam[slice_idx])[..., :3]
                a_cam = (a_cam * 255).astype(np.uint8)
                a_img = (a_img * 255).astype(np.uint8)
                overlay = (0.5 * a_img[..., None] + 0.5 * a_cam).astype(np.uint8)
                ax[a].imshow(overlay)
                ax[a].set_title(f"{name} CAM (T->cls{cls})")
                ax[a].axis("off")
            fig.suptitle(f"val sample {n_done} — 热图是否聚焦病灶区域?")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"cam_{n_done:03d}.png"), dpi=110)
            plt.close(fig)
            n_done += 1
            print(f"sample {n_done}: pred_T_cls={cls}")
            if n_done >= args.n_samples:
                ct_hook.remove(); ct_ghook.remove()
                mri_hook.remove(); mri_ghook.remove()
                print(f"=== 完成: {n_done} 张热图 -> {args.out_dir} ===")
                return


if __name__ == "__main__":
    main()
