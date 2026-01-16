# Unconditional Generative Adversarial Network (GAN) on MNIST

A complete PyTorch implementation of an **Unconditional Generative Adversarial Network (GAN)** designed to generate realistic 28×28 MNIST handwritten digits. This project demonstrates adversarial learning through a Generator–Discriminator framework and addresses common GAN training stability challenges.

## 🚀 Key Features

- **Adversarial Learning Framework**:
  - Generator learns to map random noise vectors to image space.
  - Discriminator learns to distinguish real images from generated samples.
- **Stabilized GAN Training**:
  - Binary Cross-Entropy with Logits for numerical stability.
  - Label smoothing to prevent discriminator overconfidence.
  - Balanced learning rates for stable convergence.
- **End-to-End Image Generation**:
  - Fully connected architecture tailored for MNIST.
  - Produces visually recognizable handwritten digits.

## 🏗️ Architecture Overview

The model follows the standard GAN framework consisting of two competing neural networks:

1. **Generator**:
   - Input: 100-dimensional latent noise vector sampled from a standard normal distribution.
   - Fully connected layers with Batch Normalization and ReLU activation.
   - Output layer with Tanh activation producing normalized 28×28 images.

2. **Discriminator**:
   - Input: Flattened 28×28 image.
   - Fully connected layers with LeakyReLU activation.
   - Outputs raw logits to improve numerical stability during training.

## 📊 Results & Performance

The GAN was trained for 30 epochs on the MNIST dataset, achieving stable convergence and high-quality generation.

| Metric | Value |
| :--- | :--- |
| **Final Generator Loss** | ~1.0–2.5 |
| **Training Stability** | Stable after applying regularization |
| **Mode Collapse** | Mitigated using label smoothing |

### Sample Generation
The generator produces a $4 \times 4$ grid of handwritten digit samples by transforming random noise vectors into realistic MNIST-style images.

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.8+
- PyTorch & Torchvision
- tqdm, matplotlib

### Running the Project
1. Open the provided Jupyter Notebook: `TaskSet-1/Project-2.ipynb`.
2. Ensure a GPU is available for faster training (recommended).
3. Run all cells to:
   - Train the Generator and Discriminator.
   - Monitor adversarial losses during training.
   - Generate and visualize a $4 \times 4$ grid of digit samples.

## 📁 Project Structure

```text
DLGAI/
└── Project-2/
    ├── Project-2.ipynb    # Main implementation (Model, Training, Generation)
    ├── Project-2.pdf      # Project Report
    └── README.md          # Project Documentation
```

## 📜 Acknowledgments
Based on the foundational GAN paper "Generative Adversarial Networks" by Goodfellow et al.