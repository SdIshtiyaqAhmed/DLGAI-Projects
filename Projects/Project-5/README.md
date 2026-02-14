# Consistency Model for MNIST Dataset

A complete PyTorch implementation of a **Consistency Model** trained on the **MNIST handwritten digits dataset**.
This project demonstrates **efficient generative modeling** by learning a direct mapping from noisy images to clean images, enabling **single-step image generation** without iterative denoising.

---

## 🚀 Key Features

* **Consistency-Based Generative Modeling**:

  * Learns output consistency across multiple noise levels.
  * Enables fast, one-step sampling from noise to image space.
* **UNet-Based Noise Prediction Network**:

  * Encoder–decoder architecture with skip connections.
  * Designed to map noisy inputs directly to clean images.
* **Stable and Efficient Training**:

  * Combines consistency loss with reconstruction loss.
  * Prevents trivial constant-output solutions.
* **End-to-End Training & Generation**:

  * Optimized for MNIST-scale grayscale images.
  * Generates digit-like samples in a single forward pass.

---

## 🏗️ Architecture Overview

The model follows a **Consistency Model architecture** built around a **UNet-based noise-prediction network**.

### 1. **UNet-Based Noise Prediction Network**

* Input: Noisy grayscale image (28 × 28).
* Architecture:

  * Downsampling path with residual blocks.
  * Bottleneck layer for high-level feature representation.
  * Upsampling path with skip connections to preserve spatial details.
* Output:

  * Predicted clean image corresponding to the noisy input.

The UNet learns to denoise images across a continuous range of noise scales.

---

## 📊 Results & Performance

The consistency model was trained for **30 epochs** on the MNIST dataset.

| Metric                      | Observation                          |
| --------------------------- | ------------------------------------ |
| **Training Stability**      | Stable convergence                   |
| **Mode Collapse**           | Not observed                         |
| **Consistency Enforcement** | Successfully maintained              |
| **Sampling Speed**          | Very fast (single-step generation)   |
| **Visual Quality**          | Smooth but recognizable digit shapes |

### Image Generation

Images are generated using **one forward pass** from random Gaussian noise.
The resulting samples preserve overall digit structure while appearing smoother than diffusion-based outputs, which is an expected trade-off for single-step generation.

---

## 🛠️ Setup & Usage

### Prerequisites

* Python 3.8+
* PyTorch & Torchvision
* tqdm, matplotlib

### Running the Project

1. Open the Jupyter Notebook: `Project-5.ipynb`.
2. Enable GPU acceleration (recommended).
3. Run all cells to:

   * Train the consistency model.
   * Monitor training loss across epochs.
   * Generate MNIST digit samples using one-step sampling.
   * View the final deliverables report.

---

## 📁 Project Structure

```text
DLGAI/
└── Project-5/
    ├── Project-5.ipynb    # Main implementation (UNet, Loss, Training)
    ├── Project-5.pdf      # Project Report
    └── README.md          # Project Documentation
```

---

## 📌 Notes & Limitations

* Generated images appear smoother compared to diffusion models due to single-step sampling.
* Mean Squared Error–based objectives encourage averaged solutions, leading to reduced sharpness.
* Image quality can be improved using:

  * Larger UNet architectures
  * EMA teacher models
  * Iterative or hybrid sampling strategies