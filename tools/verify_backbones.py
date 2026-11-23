# -*- coding: utf-8 -*-
"""阶段1-1: 三大冻结骨干离线加载核验 (convnext_tiny / dinov2_vitb14_reg4 / biomedclip_vitb16)。
打印键匹配状态、输入归一化参数、输出维度; 喂 224x224 张量验证 float16/32 特征无 NaN/Inf。
全离线: HF_HUB_OFFLINE=1, TF_ENABLE_ONEDNN_OPTS=0。单进程 GPU 串行。
"""
import os, json
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import torch
import timm
from safetensors.torch import load_file

ROOT = "D:/PyFile/ColonMultiModal"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def feats_ok(f):
    return bool(torch.isfinite(f).all()) and f.dtype in (torch.float16, torch.float32)

def run_convnext():
    m = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, num_classes=0)
    m = m.to(DEV).eval()
    cfg = m.default_cfg
    n_params = sum(p.numel() for p in m.parameters())
    t = torch.randn(1, 3, 224, 224).to(DEV)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        f = m(t)
    print(f"[convnext_tiny] dim={f.shape[-1]} params={n_params} cfg.mean={cfg.get('mean')} cfg.std={cfg.get('std')} "
          f"feats_ok={feats_ok(f)} dtype={f.dtype}")
    return {"model": "convnext_tiny", "dim": int(f.shape[-1]), "params": n_params,
            "mean": cfg.get("mean"), "std": cfg.get("std"), "feats_ok": bool(feats_ok(f)),
            "dtype": str(f.dtype)}

def run_dinov2():
    sd = load_file(os.path.join(ROOT, "pretrained", "dinov2_vitb14_reg4", "model.safetensors"))
    keys = list(sd.keys())
    print(f"[dinov2] safetensors keys={len(keys)} first={keys[:3]} last={keys[-2:]}")
    m = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False,
                          num_classes=0, dynamic_img_size=True)
    miss, unex = m.load_state_dict(sd, strict=False)
    print(f"[dinov2] missing={len(miss)} unexpected={len(unex)}")
    if unex:
        print("  unexpected sample:", unex[:5])
    m = m.to(DEV).eval()
    n_params = sum(p.numel() for p in m.parameters())
    cfg = m.default_cfg
    t = torch.randn(1, 3, 224, 224).to(DEV)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        f = m(t)
    print(f"[dinov2] dim={f.shape[-1]} params={n_params} cfg.mean={cfg.get('mean')} cfg.std={cfg.get('std')} "
          f"feats_ok={feats_ok(f)} dtype={f.dtype}")
    return {"model": "dinov2_vitb14_reg4", "dim": int(f.shape[-1]), "params": n_params,
            "missing_keys": len(miss), "unexpected_keys": len(unex),
            "mean": cfg.get("mean"), "std": cfg.get("std"), "feats_ok": bool(feats_ok(f)),
            "dtype": str(f.dtype)}

def run_biomedclip():
    cfg_path = os.path.join(ROOT, "pretrained", "biomedclip_vitb16", "open_clip_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    pp = cfg["preprocess_cfg"]
    sd = torch.load(os.path.join(ROOT, "pretrained", "biomedclip_vitb16", "open_clip_pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    keys = list(sd.keys())
    trunk = [k for k in keys if k.startswith("visual.trunk.")]
    proj = [k for k in keys if k.startswith("visual.head.")]
    print(f"[biomedclip] total keys={len(keys)} trunk={len(trunk)} head={len(proj)}")
    print("  trunk sample:", trunk[:2], " head sample:", proj[:2])
    # 映射 visual.trunk.xxx -> xxx (timm vit_base_patch16_224)
    trunk_sd = {k[len("visual.trunk."):]: v for k, v in sd.items() if k.startswith("visual.trunk.")}
    # 处理 timm 键 (pos_embed 无 cls 前缀差异 → timm vit_base_patch16 的 pos_embed 含 cls token, 直接匹配)
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    miss, unex = m.load_state_dict(trunk_sd, strict=False)
    print(f"[biomedclip] missing={len(miss)} unexpected={len(unex)}")
    if unex:
        print("  unexpected sample:", unex[:5])
    if miss:
        print("  missing sample:", miss[:5])
    # 顺手保存 visual.head.proj 为 npy
    head_dir = os.path.join(ROOT, "features", "slice_feats_v2")
    os.makedirs(head_dir, exist_ok=True)
    hp = os.path.join(head_dir, "biomedclip_head_proj.npy")
    if proj:
        w = sd[proj[0]].float().numpy()
        np.save(hp, w)
        print(f"[biomedclip] visual.head.proj saved → {hp} shape={w.shape}")
    m = m.to(DEV).eval()
    n_params = sum(p.numel() for p in m.parameters())
    t = torch.randn(1, 3, 224, 224).to(DEV)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        f = m(t)
    print(f"[biomedclip] dim={f.shape[-1]} params={n_params} mean={pp['mean']} std={pp['std']} "
          f"feats_ok={feats_ok(f)} dtype={f.dtype}")
    return {"model": "biomedclip_vitb16", "dim": int(f.shape[-1]), "params": n_params,
            "missing_keys": len(miss), "unexpected_keys": len(unex),
            "mean": pp["mean"], "std": pp["std"], "feats_ok": bool(feats_ok(f)),
            "dtype": str(f.dtype), "head_proj_saved": bool(proj)}

def main():
    print("device:", DEV)
    out = {}
    out["convnext_tiny"] = run_convnext()
    out["dinov2_vitb14_reg4"] = run_dinov2()
    out["biomedclip_vitb16"] = run_biomedclip()
    with open(os.path.join(ROOT, "logs", "verify_backbones.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("DONE → logs/verify_backbones.json")

if __name__ == "__main__":
    main()
