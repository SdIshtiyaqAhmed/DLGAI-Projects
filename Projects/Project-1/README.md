# High-Fidelity Denoising Diffusion Probabilistic Model (DDPM)

A complete PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) designed to generate high-quality 28x28 MNIST digits. This project features an advanced U-Net architecture with self-attention and residual blocks to achieve sharp, realistic generation results.

## 🚀 Key Features

- **Advanced U-Net Architecture**: 
  - **Residual Blocks**: Uses GELU activations and time-feature integration.
  - **Self-Attention**: Multi-head attention layer at the bottleneck (resolution 7x7) for global dependency mapping.
  - **Time MLP**: Deep MLP for continuous timestep embeddings.
- **Optimized Diffusion Process**:
  - Linear beta schedule ($\beta_1 = 10^{-4}$ to $\beta_T = 0.02$).
  - 1,000 diffusion timesteps for fine-grained reverse process.
- **Robust Evaluation**: Integrated CNN classifier to measure generation confidence and class diversity.

## 🏗️ Architecture Overview

The model follows the standard DDPM framework but enhances the reverse process (denoising) using a sophisticated U-Net:

1. **Down-sampling**: Residual blocks followed by Max Pooling.
2. **Bottleneck**: Residual block integrated with Multi-head Self-Attention.
3. **Up-sampling**: ConvTranspose2d layers combined with skip connections and residual blocks.
4. **Time Embedding**: Timestep $t$ is embedded via a sinusoidal-like MLP and injected into every residual block.

## 📊 Results & Performance

The model was trained for 30 epochs on an MNIST dataset, achieving the following metrics:

| Metric | Value |
| :--- | :--- |
| **Final Training MSE** | ~0.023 |
| **Mean Classifier Confidence** | ~0.85+ |
| **Class Diversity** | 9/10 or 10/10 classes per batch |

### Sample Generation
The generation process starts from pure Gaussian noise $\mathcal{N}(0, 1)$ and iteratively denoises over 1,000 steps to produce sharp MNIST-style digits.

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.8+
- PyTorch & Torchvision
- tqdm, matplotlib

### Running the Project
1. Open the provided Jupyter Notebook: `TaskSet-1/Project-1.ipynb`.
2. Ensure a GPU (e.g., NVIDIA T4) is available for optimal training performance.
3. Run all cells to:
   - Train the robust CNN evaluator.
   - Train the High-Fidelity DDPM.
   - Generate and visualize a $4 \times 4$ grid of realistic digits.

## 📁 Project Structure

```text
DLGAI/
├── TaskSet-1/
│   ├── Project-1.ipynb    # Main implementation (Model, Training, Evaluation)
│   └── Project-1.pdf      # Project Report
└── README.md              # Project Documentation
```

## 📜 Acknowledgments
Based on the foundational DDPM paper *"Denoising Diffusion Probabilistic Models"* by Ho et al.
