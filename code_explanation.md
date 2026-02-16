# Fusion Net Codebase Explanation

This document provides a detailed explanation of the **Fusion Net** codebase, which implements a deep learning model for fusing Infrared (IR) and Visible (VIS) images.

## 1. Project Overview
The goal of this project is to merge the complementary information from IR and VIS images into a single high-quality fused image. It uses a **Dual-Encoder U-Net** architecture (Fusion-UNet) to extract features from both modalities and fuse them using attention mechanisms.

## 2. Directory Structure
- **`src/`**: Contains all the source code for the project.
  - **`models/`**: Defines the neural network architecture.
  - **`data/`**: Handles data loading and preprocessing.
  - **`losses/`**: Custom loss functions (e.g., FusionLoss).
  - **`metrics/`**: Evaluation metrics like SSIM, Entropy.
  - **`utils/`**: Helper functions for visualization and checkpoints.
- **`config/`**: Configuration files (YAML) for training and debugging.
- **`data/`**: Directory where training and validation datasets should be stored.
- **`experiments/`**: Stores training logs (TensorBoard) and outputs.
- **`checkpoints/`**: Stores saved model weights (`.pth` files).

## 3. Key Components Detail

### A. Model Architecture (`src/models/fusion_unet.py`)
The core of the project is the `FusionUNet` class.
- **Dual Encoders**: It has two separate encoders, `ir_encoder` and `vis_encoder`. These processing streams do *not* share weights, allowing them to learn modality-specific features.
- **Fusion Blocks**: At each level of the encoder, features from IR and VIS are combined using `FusionBlock` modules. These blocks likely use channel and spatial attention to select the most important features from each modality.
- **Decoder**: The fused features are then passed to a `Decoder`, which reconstructs the final fused image. It uses skip connections from the fusion blocks to preserve spatial details.
- **Output**: The final layer uses a Sigmoid activation to produce an image with pixel values in the range `[0, 1]`.

### B. Data Loading (`src/data/dataset.py`)
- **`FusionDataset`**: This class is responsible for loading image pairs.
  - It expects two subdirectories: `ir/` and `vis/`.
  - It sorts files by filename to ensure that the $i$-th IR image corresponds to the $i$-th VIS image.
  - It converts images to PyTorch tensors and normalizes them to `[0, 1]`.
  - It supports data augmentation via the `transform` argument.
- **`get_dataloader`**: A helper function that creates PyTorch `DataLoader` instances for training, validation, and testing. It handles batching, shuffling, and parallel loading (`num_workers`).

### C. Training Logic (`src/train.py`)
This script handles the training process.
- **`load_config`**: Reads hyperparameters from `config/train_config.yaml`.
- **`train_one_epoch`**: The core loop for a single epoch.
  - It iterates over the training data.
  - Uses `Mixed Precision` (via `torch.cuda.amp`) for faster training and lower memory usage if enabled in the config.
  - Calculates loss using `FusionLoss` (a combination of Intensity Loss, SSIM Loss, and Gradient Loss).
  - Updates model weights using the `Adam` optimizer.
  - Logs training progress to TensorBoard.
- **`validate`**: Evaluates the model on the validation set after every epoch. It computes metrics like SSIM and Entropy to monitor performance.
- **Checkpointing**: Saves the model weights periodically and keeps the "best" model based on the SSIM score.

### D. Inference (`src/inference.py`)
This script is used to use the trained model on new images.
- **Modes**:
  - **Single Pair**: Fuse a specific pair of IR and VIS images.
  - **Batch Mode**: Fuse an entire directory of IR/VIS pairs.
- **Process**:
  1. Loads the model and checkpoint.
  2. Preprocesses the input images (resize, normalize).
  3. Runs the model inference.
  4. Postprocesses the output (denormalize, save as image).

### E. Configuration (`config/train_config.yaml`)
This file controls all aspects of training without changing the code.
- **`data`**: Paths to datasets, batch size, image size.
- **`training`**: Learning rate, epochs, optimizer settings.
- **`loss`**: Weights for different loss components ($\lambda_{ir}, \lambda_{vis}, \alpha_{ssim}, \beta_{grad}$).
- **`logging`**: How often to log to TensorBoard and save sample images.

## 4. Deep Dive: Architecture & Training

### A. Dual-Encoder Architecture
The model uses a **Siamese-like** architecture but with unshared weights:
1.  **Modality-Specific Encoders**: Since IR and VIS images have different statistical properties (IR measures thermal radiation, VIS measures reflected light), sharing weights would be suboptimal. Separate encoders allow the model to learn features specific to each modality.
2.  **Feature Extraction**: Each encoder extracts features at multiple scales ($64 \to 128 \to 256 \to 512 \to 1024$ channels).

### B. Fusion Mechanism (`FusionBlock`)
At every scale of the encoder, features from the IR path and VIS path are combined. The simple concatenation is refined by two attention mechanisms:
1.  **Channel Attention**:
    - Based on "Squeeze-and-Excitation" networks.
    - It computes a weight vector (size $C$) that emphasizes important feature channels and suppresses less useful ones.
    - *Why?* Some channels might be noisy or irrelevant in one modality.
2.  **Spatial Attention**:
    - It computes a spatial map (size $H \times W$) indicating "where" the information is most important.
    - It uses max-pooling and average-pooling across channels, followed by a convolution.
    - *Why?* To focus on salient regions (e.g., a pedestrian in IR, a sign in VIS).

### C. Training Objective (Loss Function)
The model is trained without ground truth fused images (unsupervised fusion). The goal is to keep the "best" parts of both inputs. The loss function has three components:

$$L_{total} = L_{intensity} + \alpha L_{SSIM} + \beta L_{gradient}$$

1.  **Intensity Loss (`IntensityLoss`)**:
    - Ensures the fused image has similar pixel intensity to the source images.
    - $L_{int} = \lambda_{ir}||I_{f} - I_{ir}||_1 + \lambda_{vis}||I_{f} - I_{vis}||_1$
    - It uses weights ($\lambda$) to control trade-off (default: 0.5 each).

2.  **SSIM Loss (`SSIMLoss`)**:
    - **Structural Similarity Index Measure (SSIM)** is a perceptual metric.
    - It ensures the fused image preserves the *structure* (textures, objects) of source images, rather than just pixel values.
    - We maximize SSIM (or minimize $1 - SSIM$) between fused image and both sources.

3.  **Gradient Loss (`GradientLoss`)**:
    - Computed using Sobel operators.
    - It forces the fused image's gradients to match the **maximum** gradient of the two sources.
    - *Goal:* If an edge is sharp in IR but blurry in VIS, the fused image should have the sharp edge.

### D. Optimization
- **Optimizer**: Adam (Adaptive Moment Estimation) is used for stable convergence.
- **Scheduler**: Cosine Annealing decay reduces the learning rate over time to fine-tune the model.
- **Mixed Precision**: Uses FP16 (half-precision) during training to reduce memory usage and speed up computation on modern GPUs (Tensor Cores).

## 5. How to Run

### Training
To start training the model with default settings:
```bash
python src/train.py --config config/train_config.yaml
```

To resume training from a checkpoint:
```bash
python src/train.py --config config/train_config.yaml --resume checkpoints/checkpoint_epoch_010.pth
```

### Inference
To fuse a single pair of images:
```bash
python src/inference.py --ir data/test/ir/001.png --vis data/test/vis/001.png --output result.png
```

To fuse all images in a folder:
```bash
python src/inference.py --ir_dir data/test/ir --vis_dir data/test/vis --output_dir outputs/
```

### Testing
To evaluate the model and generate metrics:
```bash
python src/test.py --checkpoint checkpoints/best_model.pth --data_dir data/test
```
