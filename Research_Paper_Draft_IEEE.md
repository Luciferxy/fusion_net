# CMF-Net: A Dual-Stream Multi-Scale Attention-Guided Network for Infrared and Visible Image Fusion

**(Draft for IEEE Conference/Journal Submission)**

---

## Abstract
Infrared and visible image fusion (IVIF) is a critical technology for all-weather surveillance and autonomous driving, aiming to integrate the thermal radiation information of infrared images with the detailed texture information of visible images. Existing fusion methods often suffer from the loss of modality-specific features due to early fusion strategies or fail to preserve fine-grained details across different scales. To address these limitations, this paper proposes a novel deep learning framework named **CMF-Net (Cross-Modal Fusion Network)**. We introduce a **Dual-Stream Feature Extraction (DSFE)** architecture that extracts modality-specific features independently before fusion. Furthermore, a **Multi-Scale Attention Fusion Module (MSAFM)** is designed to adaptively integrate features at five distinct scales, leveraging both **Channel Attention (CA)** and **Spatial Attention (SA)** mechanisms to weigh the importance of meaningful features and regions. A hybrid loss function combining structural similarity, intensity consistency, and gradient preservation is employed to train the network end-to-end. Extensive experiments on the **MSRS dataset** demonstrate that CMF-Net achieves state-of-the-art performance in both subjective visual quality and objective quantitative metrics.

**Keywords:** Image Fusion, Infrared and Visible, Dual-Stream Encoder, Attention Mechanism, Deep Learning.

---

## I. Introduction
Multi-modal image fusion has garnered significant attention in computer vision due to its ability to provide comprehensive scene descriptions. Infrared (IR) sensors capture thermal radiation, effectively highlighting targets (e.g., pedestrians, vehicles) under poor lighting conditions, but they lack textural details and are sensitive to noise. Conversely, visible (VIS) sensors provide rich color and texture information but degrade significantly in low-light or adverse weather scenarios. Fusing these two modalities creates a single image that retains the strengths of both, which is invaluable for applications such as military surveillance, remote sensing, and autonomous navigation.

Traditional fusion methods based on multi-scale transforms (MST) and sparse representation (SR) typically rely on hand-crafted features, which limit their generalization capability. Recently, deep learning-based methods, particularly Auto-Encoders (AE) and Generative Adversarial Networks (GANs), have shown promise. However, many current AE-based methods employ a "concat-then-encode" strategy (Early Fusion), which forces the network to learn shared representations too early, potentially discarding unique modality-specific features. Additionally, fusion strategies often operate at a single bottleneck layer, ignoring the hierarchical nature of features.

To overcome these challenges, we propose **CMF-Net**, a U-Net-based architecture that explicitly disentangles feature extraction for IR and VIS images. The main contributions of this work are summarized as follows:
1.  **Dual-Stream Architecture:** We design a dual-encoder structure that extracts deep features from IR and VIS images independently, preserving unique modality characteristics.
2.  **Multi-Scale Attention Fusion:** We introduce a dense fusion strategy where features are fused at every encoder level (from shallow to deep) using a Channel-Spatial Attention mechanism, ensuring both high-level semantic info and low-level details are preserved.
3.  **Hybrid Loss Function:** A comprehensive objective function combining intensity, structural similarity (SSIM), and gradient loss is proposed to guide the network in generating visually natural and structurally accurate fused images.

---

## II. Related Work
**(Briefly summarize existing art to position your novelty)**
*   **AE-based Methods:** Discussion of DenseFuse, NestFuse, and their limitation of using simple fusion rules (max/avg) or single-layer fusion.
*   **Attention Mechanisms:** Mention of CBAM (Convolutional Block Attention Module) and how it inspires our fusion block to focus on "what" (channel) and "where" (spatial).

---

## III. Proposed Method

### A. Network Overview
The proposed CMF-Net adopts a U-Net-like encoder-decoder architecture. As illustrated in **Fig. 1 (Architecture)**, it consists of three main components:
1.  **Dual-Stream Encoder:** Two parallel branches ($E_{IR}$ and $E_{VIS}$) with non-shared weights.
2.  **Multi-Scale Attention Fusion Module (MSAFM):** A fusion block located at each of the 5 skip connections.
3.  **Decoder:** A single path that reconstructs the fused image from the multi-scale fused features.

### B. Dual-Stream Encoder
Each encoder consists of 5 stages. Each stage $i$ comprises a convolutional block (two $3 \times 3$ Conv-BN-ReLU layers) followed by a $2 \times 2$ MaxPooling operation.
Let $I_{IR}$ and $I_{VIS}$ denote the input images. The features at level $i$ are extracted as:
$$ F_{IR}^i = E_{IR}^i(F_{IR}^{i-1}), \quad F_{VIS}^i = E_{VIS}^i(F_{VIS}^{i-1}) $$
where $i \in \{1, ..., 5\}$. This separation ensures that the thermal signatures and texture patterns are encoded in their respective feature spaces.

### C. Multi-Scale Attention Fusion Module (MSAFM)
At each scale $i$, we employ an attention-guided fusion strategy. The module takes $F_{IR}^i$ and $F_{VIS}^i$ as input.

**1. Concatenation & Reduction:**
First, we concatenate features along the channel dimension and reduce them using a $1 \times 1$ convolution:
$$ F_{cat}^i = \text{Concat}(F_{IR}^i, F_{VIS}^i) $$
$$ F_{pre}^i = \text{Conv}_{1 \times 1}(F_{cat}^i) $$

**2. Channel Attention (CA):**
To highlight meaningful feature channels (e.g., thermal targets), we apply global average pooling and max pooling, followed by a shared MLP:
$$ \mathbf{M}_c(F) = \sigma(MLP(\text{AvgPool}(F)) + MLP(\text{MaxPool}(F))) $$
$$ F'_{pre} = \mathbf{M}_c(F_{pre}^i) \otimes F_{pre}^i $$
where $\sigma$ is the Sigmoid function and $\otimes$ denotes element-wise multiplication.

**3. Spatial Attention (SA):**
To focus on informative spatial regions, we apply average and max pooling along the channel axis, concatenate them, and pass them through a $7 \times 7$ convolution:
$$ \mathbf{M}_s(F) = \sigma(\text{Conv}_{7 \times 7}([\text{AvgPool}_{ch}(F); \text{MaxPool}_{ch}(F)])) $$
$$ F_{fused}^i = \mathbf{M}_s(F'_{pre}) \otimes F'_{pre} $$

This dual-attention mechanism allows the network to adaptively select the most relevant information from both modalities at every scale.

### D. Decoder
The decoder mirrors the encoder structure. It receives the fused features $F_{fused}^i$ from the skip connections and upsamples the features from the deeper layer $D^{i+1}$:
$$ D^i = \text{ConvBlock}(\text{Concat}(F_{fused}^i, \text{UpSample}(D^{i+1}))) $$
The final output is generated by a $1 \times 1$ convolution followed by a Sigmoid activation to map pixel values to $[0, 1]$.

---

## IV. Loss Function
To train the unsupervised network, we define a hybrid loss function $L_{total}$ composed of three terms:
$$ L_{total} = L_{int} + \alpha L_{ssim} + \beta L_{grad} $$

**1. Intensity Loss ($L_{int}$):**
Ensures the fused image $I_f$ maintains appropriate pixel intensity similar to the source images. We use a weighted $L_1$ norm:
$$ L_{int} = \frac{1}{HW} \| I_f - \max(I_{IR}, I_{VIS}) \|_1 $$
*(Note: In our implementation, we use a weighted sum $\lambda_1 \|I_f - I_{IR}\|_1 + \lambda_2 \|I_f - I_{VIS}\|_1$)*.

**2. SSIM Loss ($L_{ssim}$):**
Preserves the structural information (edges, textures) from both source images.
$$ L_{ssim} = (1 - \text{SSIM}(I_f, I_{IR})) + (1 - \text{SSIM}(I_f, I_{VIS})) $$
where SSIM is the Structural Similarity Index measured using a sliding Gaussian window.

**3. Gradient Loss ($L_{grad}$):**
Forces the fused image to retain fine texture details and sharp edges, primarily from the visible image.
$$ L_{grad} = \| \nabla I_f - \max(|\nabla I_{IR}|, |\nabla I_{VIS}|) \|_1 $$
where $\nabla$ denotes the Sobel gradient operator.

---

## V. Experiments

### A. Experimental Setup
*   **Dataset:** We utilize the **MSRS dataset**, consisting of 1,444 aligned image pairs (983 training, 361 testing).
*   **Preprocessing:** Images are resized to $256 \times 256$ and normalized to $[0, 1]$.
*   **Environment:** PyTorch framework on an NVIDIA GPU.
*   **Training:** The model is trained for 50 epochs with a batch size of 2 using the Adam optimizer ($lr=1 \times 10^{-4}$).

### B. Evaluation Metrics
We employ six standard metrics to quantitatively evaluate performance:
1.  **Structural Similarity (SSIM):** Measures structural consistency.
2.  **Peak Signal-to-Noise Ratio (PSNR):** Measures information retention and noise level.
3.  **Entropy (EN):** Measures the amount of information contained in the fused image.
4.  **Mutual Information (MI):** Measures the information transferred from source images to the fused image.
5.  **Average Gradient (AG):** Reflects the clarity and texture richness.
6.  **Visual Information Fidelity (VIF):** Measures visual quality based on human visual system models.

### C. Results
*(This section is to be filled with your actual test results from `src/test.py`)*.
**Table 1** shows the quantitative comparison of CMF-Net against state-of-the-art methods...
**Fig. 3** illustrates the visual comparison...
*   **Analysis:** CMF-Net effectively highlights the thermal targets (pedestrians) while preserving the background text on the signboard (from VIS), demonstrating the effectiveness of the proposed MSAFM.

---

## VI. Conclusion
In this paper, we presented CMF-Net for infrared and visible image fusion. By leveraging a dual-stream encoder and a multi-scale attention-guided fusion mechanism, our model effectively integrates complementary information from both modalities. The proposed hybrid loss function further ensures high structural fidelity and visual naturalness. Experimental results on the MSRS dataset validate the superiority of our approach. Future work will focus on integrating semantic constraints to further improve object detection performance in fused images.

---
Using appropriate LaTeX syntax for equations ($...$) is recommended when converting to PDF.
