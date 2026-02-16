# Research Paper Basis: Fusion-UNet (CMF-Net)

## Title Idea
**"CMF-Net: A Dual-Stream, Multi-Scale Attention-Guided Network for Infrared and Visible Image Fusion"**

## 1. Abstract Summary
This paper proposes a novel deep learning framework for fusing infrared (IR) and visible (VIS) images. The core challenge in multi-modal fusion is preserving the thermal radiation information from IR images while maintaining the rich texture details from VIS images. We address this by introducing a **Dual-Encoder U-Net architecture** equipped with a **Multi-Scale Attention Fusion Module**. Unlike traditional methods that fuse at a single layer, our approach performs feature fusion at five distinct scales, leveraging both **Channel** and **Spatial Attention mechanisms** to adaptively weight the importance of each modality. Extensive experiments on the **MSRS dataset** demonstrate the method's superiority in visual quality and objective metrics.

---

## 2. Novelty & Contributions
The key innovative contributions of this project that can be highlighted in your paper:

### A. Dual-Stream Feature Extraction (DSFE)
Instead of concatenating the IR and VIS images at the input (Early Fusion), which forces the network to learn shared features prematurely, we employ **two separate encoders** with non-shared weights.
*   **Novelty:** This allows the network to extract modality-specific features independently—learning thermal signatures in one stream and texture edges in the other—before fusion occurs.

### B. Multi-Scale Attention-Guided Fusion
We introduce a lightweight but effective **Fusion Block** at every encoder level (skip connection), consisting of:
1.  **Channel Attention (CA):** Adaptively recalibrates channel-wise feature responses. It answers *"What content is important?"* (e.g., enhancing thermal targets).
2.  **Spatial Attention (SA):** Focuses on informative regions. It answers *"Where is the important content?"* (e.g., focusing on pedestrians/vehicles).
*   **Novelty:** Applying this dual-attention mechanism densely across all 5 scales (from 64 to 1024 channels) ensures both high-level semantic information and low-level fine details are preserved.

### C. Hybrid Structure-Texture Loss
We designed a comprehensive loss function that goes beyond simple pixel differences:
*   $$ L_{total} = L_{intensity} + \alpha L_{SSIM} + \beta L_{gradient} $$
*   **Novelty:** The inclusion of a dedicated **Gradient Loss** (Sobel-based) specifically forces the network to retain sharp edges from the visible image, while the **SSIM Loss** ensures global structural consistency.

---

## 3. Methodology

### 3.1. Network Architecture
The proposed model is based on the U-Net architecture but modified for multi-modal input.
*   **Encoders:** Two parallel encoders (IR and VIS), each with 5 stages. Each stage consists of a double `Conv3x3 + BN + ReLU` block followed by MaxPool downsampling.
    *   *Channels:* 64 → 128 → 256 → 512 → 1024.
*   **Fusion Layer:** At each stage $i$, features $F_{IR}^i$ and $F_{VIS}^i$ are concatenated and passed through the Fusion Block to generate $F_{Fused}^i$.
*   **Decoder:** A single decoder path that upsamples features and concatenates them with the corresponding fused features ($F_{Fused}^i$) from the skip connections.

### 3.2. Fusion Strategy Details
The fusion block operates as:
$$ F_{concat} = Concat([F_{IR}, F_{VIS}]) $$
$$ F_{reduced} = Conv1x1(F_{concat}) $$
$$ F_{channel} = F_{reduced} \otimes \sigma(MLP(AvgPool(F_{reduced})) + MLP(MaxPool(F_{reduced}))) $$
$$ F_{final} = F_{channel} \otimes \sigma(Conv7x7(Concat([AvgPool(F_{channel}), MaxPool(F_{channel})]))) $$
*(Note: This mathematical formulation corresponds to the Channel/Spatial attention implemented in `src/models/fusion_blocks.py`.)*

---

## 4. Experimental Setup

### 4.1. Dataset
*   **Dataset:** MSRS (Multi-Spectral Road Scenarios).
*   **Training Set:** 983 image pairs.
*   **Validation Set:** 100 image pairs.
*   **Test Set:** 361 image pairs.
*   **Preprocessing:** Images resized to 256×256 and normalized to [0, 1]. Data augmentation includes random horizontal flips and rotations (±10°).

### 4.2. Implementation Details
*   **Framework:** PyTorch.
*   **Optimizer:** Adam ($lr=1e-4$).
*   **Scheduler:** Cosine Annealing LR.
*   **Loss Weights:** Intensity (1.0), SSIM (5.0), Gradient (10.0).
*   **Hardware:** Trained on NVIDIA GPU.

---

## 5. Metrics for Evaluation
To quantitatively validate the method, we report:
*   **SSIM (Structural Similarity Index):** Measures structural consistency.
*   **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction quality.
*   **Entropy (EN):** Measures information content.
*   **Mutual Information (MI):** Measures information transfer from source images.
*   **Average Gradient (AG):** Measures image sharpness/clarity.
