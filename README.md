# 🔥 Fusion-UNet: IR-VIS Image Fusion

> A dual-encoder U-Net architecture for fusing Infrared (IR) and Visible (VIS) images using attention-based feature fusion.

---

## 🏗️ Architecture

```
┌──────────┐    ┌──────────┐
│ IR Image │    │VIS Image │
└────┬─────┘    └────┬─────┘
     │               │
┌────▼─────┐   ┌────▼─────┐
│IR Encoder│   │VIS Encoder│    ← Separate weights (no sharing)
│  5 levels│   │  5 levels │
└────┬─────┘   └────┬─────┘
     │               │
     └───────┬───────┘
             │
   ┌─────────▼─────────┐
   │   Fusion Blocks    │       ← Channel + Spatial Attention
   │  (at each level)   │
   └─────────┬─────────┘
             │
   ┌─────────▼─────────┐
   │     Decoder        │       ← Skip connections from fused features
   │    4 levels        │
   └─────────┬─────────┘
             │
   ┌─────────▼─────────┐
   │  Output (Sigmoid)  │       ← Fused RGB image ∈ [0, 1]
   └───────────────────┘
```

**Channel progression:** 3 → 64 → 128 → 256 → 512 → 1024 (encoder) → 512 → 256 → 128 → 64 → 3 (decoder)

---

## 📁 Project Structure

```
FUSION NET/
├── config/
│   ├── train_config.yaml       # Training hyperparameters
│   └── model_config.yaml       # Architecture configuration
├── data/
│   ├── train/ir/ & vis/        # Training pairs
│   ├── val/ir/ & vis/          # Validation pairs
│   └── test/ir/ & vis/         # Test pairs
├── src/
│   ├── models/
│   │   ├── encoder.py          # Dual encoder (ConvBlock, Encoder)
│   │   ├── decoder.py          # Decoder with skip connections
│   │   ├── fusion_blocks.py    # Attention-based feature fusion
│   │   └── fusion_unet.py      # Complete FusionUNet model
│   ├── data/
│   │   ├── dataset.py          # Paired IR-VIS dataset loader
│   │   └── transforms.py       # Data augmentation pipeline
│   ├── losses/
│   │   ├── ssim_loss.py        # SSIM loss (differentiable)
│   │   └── intensity_loss.py   # L1 + gradient + combined loss
│   ├── metrics/
│   │   ├── ssim.py             # SSIM evaluation metric
│   │   ├── entropy.py          # Shannon entropy
│   │   └── evaluation.py       # PSNR, MI, AG, full evaluation suite
│   ├── utils/
│   │   ├── visualization.py    # Comparison grids, training curves
│   │   └── checkpoint.py       # Model save/load utilities
│   ├── train.py                # Training script
│   ├── test.py                 # Evaluation script
│   └── inference.py            # Single/batch inference
├── checkpoints/                # Saved model weights
├── outputs/                    # Inference results
├── experiments/logs/           # TensorBoard logs
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place paired IR and VIS images in the `data/` directory:
```
data/train/ir/image_001.png    ↔    data/train/vis/image_001.png
data/train/ir/image_002.png    ↔    data/train/vis/image_002.png
...
```

### 3. Train
```bash
python src/train.py --config config/train_config.yaml
```

### 4. Evaluate
```bash
python src/test.py --checkpoint checkpoints/best_model.pth
```

### 5. Inference
```bash
# Single pair
python src/inference.py --ir path/to/ir.png --vis path/to/vis.png --output fused.png

# Batch
python src/inference.py --ir_dir data/test/ir --vis_dir data/test/vis --output_dir outputs/
```

### 6. Monitor Training
```bash
tensorboard --logdir experiments/logs
```

---

## 📊 Loss Function

| Component | Formula | Weight |
|-----------|---------|--------|
| **Intensity** | λ₁·‖F-IR‖₁ + λ₂·‖F-VIS‖₁ | λ₁=λ₂=0.5 |
| **SSIM** | 0.5·(1-SSIM(F,IR)) + 0.5·(1-SSIM(F,VIS)) | α = 1.0 |
| **Gradient** | ‖∇F - max(∇IR, ∇VIS)‖₁ | β = 0.5 |

**Total:** L = L_intensity + α·L_ssim + β·L_gradient

---

## 📈 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| SSIM | Structural Similarity | > 0.85 |
| Entropy | Information richness | > 6.5 |
| PSNR | Peak Signal-to-Noise Ratio | Higher = better |
| MI | Mutual Information | Higher = better |
| AG | Average Gradient (edges) | Higher = better |

---

## ⚙️ Training Configuration

| Parameter | Default |
|-----------|---------|
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Scheduler | Cosine Annealing |
| Batch Size | 2 |
| Epochs | 200 |
| Mixed Precision | ✅ FP16 |
| Gradient Clipping | 1.0 |
| Early Stopping | 20 epochs |

---

## 📜 References

- Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- Zhang et al. (2020) — Image Fusion with Deep Learning: A Survey
- MSRS Dataset — Multi-Spectral Road Scenarios
