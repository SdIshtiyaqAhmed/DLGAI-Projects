# DLGAI-Projects

**Deep Learning and Generative AI Projects**

A collection of PyTorch implementations for modern **generative models** trained on the MNIST dataset.  
This repository explores both **diffusion-based** and **adversarial** generative modeling paradigms, with complete implementations, training pipelines, evaluations, and reports.

---

## 📌 Projects Overview

### 📍 Project-1: High-Fidelity Denoising Diffusion Probabilistic Model (DDPM)

A complete PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** designed to generate high-quality 28×28 MNIST handwritten digits.

**Key Features**
- Advanced U-Net architecture with residual blocks and multi-head self-attention
- Time-step embeddings injected into all residual blocks
- Linear beta noise schedule with 1,000 diffusion timesteps
- Robust evaluation using a trained CNN classifier

**Results**
- Final training MSE ≈ **0.023**
- Mean classifier confidence ≈ **0.85+**
- High class diversity in generated samples

---

### 📍 Project-2: Unconditional Generative Adversarial Network (GAN) on MNIST

A PyTorch implementation of an **Unconditional Generative Adversarial Network (GAN)** trained on the MNIST dataset using adversarial learning between a Generator and a Discriminator.

**Key Features**
- Fully connected Generator and Discriminator architectures
- Stable training using **BCEWithLogitsLoss**
- Label smoothing to prevent discriminator overconfidence
- Balanced learning rates for stable adversarial convergence

**Results**
- Stable training over 30 epochs
- Visually recognizable handwritten digit samples
- No significant mode collapse observed

---

## 🚀 Setup & Requirements

### Prerequisites
- Python 3.8+
- PyTorch
- Torchvision
- tqdm
- matplotlib
- Jupyter Notebook (optional)

Install dependencies:
```bash
pip install torch torchvision tqdm matplotlib
````

---

## ▶️ Running the Projects

Each project is implemented in a Jupyter Notebook.

### Steps

1. Navigate to the desired project directory:

   ```bash
   cd DLGAI-Projects/Project-1
   ```

   or

   ```bash
   cd DLGAI-Projects/Project-2
   ```

2. Open the notebook:

   ```bash
   jupyter notebook Project-1.ipynb
   ```

3. Run all cells to:

   * Train the model
   * Generate sample images
   * View evaluation metrics and deliverables

> GPU acceleration is recommended for faster training.

---

## 📁 Repository Structure

```text
DLGAI-Projects/
├── Project-1/
│   ├── Project-1.ipynb    # High-Fidelity DDPM implementation
│   ├── Project-1.pdf     # Project Report
│   └── README.md         # Project documentation
├── Project-2/
│   ├── Project-2.ipynb   # Unconditional GAN implementation
│   ├── Project-2.pdf    # Project Report
│   └── README.md        # Project documentation
├── README.md             # Repository overview
└── LICENSE               # MIT License
```

---

## 🧪 Outputs

Each project generates a **4×4 grid of MNIST digit samples** at the end of training, visualized directly in the notebooks and included in the project reports.

---

## 📜 Acknowledgments

This repository is developed for educational and research purposes and is based on foundational works in generative modeling:

* *Denoising Diffusion Probabilistic Models* — Ho et al.
* *Generative Adversarial Networks* — Goodfellow et al.

---

## 📄 License

This repository is licensed under the **MIT License**.
