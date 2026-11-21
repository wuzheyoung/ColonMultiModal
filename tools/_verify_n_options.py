"""round9 B 系列维度验证: n_modal / n_cascade / n_shallow_idx 组合前向 (CPU, 不训练)。"""
import sys, time
sys.path.insert(0, ".")
import torch
from models.network import CRCMultiModalNet

torch.manual_seed(0)

def check(tag, **kw):
    m = CRCMultiModalNet(
        backbone_name="medicalnet18", num_clinical=7, num_classes=2, num_tasks=2,
        head_width=128, pool_mode="gem", pretrained_backbone=False,
        n_from_shallow=True, **kw)
    m.eval()
    ct = torch.randn(2, 1, 96, 96, 96)
    mri = torch.randn(2, 3, 96, 96, 96)
    clin = torch.randn(2, 7)
    t0 = time.time()
    with torch.no_grad():
        out = m(ct, mri, clin)
    n_shape = out["n"].shape
    assert n_shape == (2, 2), f"{tag}: n shape {n_shape} != (2,2)"
    print(f"[OK] {tag}: main={tuple(out['main'].shape)} n={tuple(n_shape)}  "
          f"({time.time()-t0:.1f}s)")

check("n_modal=both idx=3 (R5 默认)", n_modal="both", n_shallow_idx=3, n_cascade=False)
check("n_modal=ct  idx=2        ", n_modal="ct",   n_shallow_idx=2, n_cascade=False)
check("n_modal=mri idx=2        ", n_modal="mri",  n_shallow_idx=2, n_cascade=False)
check("n_cascade=True both idx=3", n_modal="both", n_shallow_idx=3, n_cascade=True)
print("=== 全部维度验证通过 ===")
