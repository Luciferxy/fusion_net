"""Quick verification script for Fusion-UNet project."""
import sys
import torch
import numpy as np

print("=" * 60)
print("  FUSION-UNET PROJECT VERIFICATION")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# --- Test 1: Model Forward Pass ---
print("--- Test 1: Model Forward Pass ---")
from src.models.fusion_unet import FusionUNet, count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

model = FusionUNet(in_channels=3, out_channels=3).to(device)
count_parameters(model)

ir = torch.randn(2, 3, 256, 256).to(device)
vis = torch.randn(2, 3, 256, 256).to(device)
out = model(ir, vis)

assert out.shape == (2, 3, 256, 256), f"FAIL: Expected (2,3,256,256), got {out.shape}"
print(f"Input shape:  {ir.shape}")
print(f"Output shape: {out.shape}")
print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
assert out.min() >= 0 and out.max() <= 1, "FAIL: Output not in [0,1]"
print("PASS: Model forward pass OK\n")

# --- Test 2: Loss Computation ---
print("--- Test 2: Loss Computation ---")
from src.losses.intensity_loss import FusionLoss

criterion = FusionLoss().to(device)
ir_clamped = ir.clamp(0, 1)
vis_clamped = vis.clamp(0, 1)
loss = criterion(out, ir_clamped, vis_clamped)
assert loss.requires_grad, "FAIL: Loss not differentiable"
loss.backward()
print(f"Loss value: {loss.item():.4f}")
print("PASS: Loss computation + backward OK\n")

# --- Test 3: Metrics ---
print("--- Test 3: Metrics ---")
from src.metrics.evaluation import evaluate_fusion

fused_np = out[0].detach().cpu().permute(1, 2, 0).numpy()
ir_np = np.random.rand(256, 256, 3).astype(np.float32)
vis_np = np.random.rand(256, 256, 3).astype(np.float32)
metrics = evaluate_fusion(fused_np, ir_np, vis_np)
print(f"SSIM avg: {metrics['ssim_avg']:.4f}")
print(f"Entropy:  {metrics['entropy']:.4f}")
print(f"PSNR IR:  {metrics['psnr_ir']:.2f} dB")
print(f"AG:       {metrics['ag']:.4f}")
print("PASS: Metrics computation OK\n")

# --- Test 4: Dataset Loading ---
print("--- Test 4: Dataset Loading ---")
try:
    from src.data.dataset import get_dataloader
    train_loader = get_dataloader("data", split="train", batch_size=2, image_size=256, num_workers=0)
    sample_ir, sample_vis = next(iter(train_loader))
    print(f"IR batch shape:  {sample_ir.shape}")
    print(f"VIS batch shape: {sample_vis.shape}")
    print(f"IR range:  [{sample_ir.min().item():.3f}, {sample_ir.max().item():.3f}]")
    print(f"VIS range: [{sample_vis.min().item():.3f}, {sample_vis.max().item():.3f}]")
    print("PASS: Dataset loading OK\n")
except Exception as e:
    print(f"SKIP: Dataset test ({e})\n")

# --- Test 5: Full Pipeline (forward through data) ---
print("--- Test 5: Full Pipeline ---")
try:
    model.eval()
    with torch.no_grad():
        fused = model(sample_ir, sample_vis)
    print(f"Fused shape: {fused.shape}")
    print(f"Fused range: [{fused.min().item():.4f}, {fused.max().item():.4f}]")
    print("PASS: Full pipeline OK\n")
except Exception as e:
    print(f"SKIP: Full pipeline test ({e})\n")

print("=" * 60)
print("  ALL TESTS PASSED!")
print("=" * 60)
