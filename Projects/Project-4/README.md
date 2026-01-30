# Text-to-Image Generation Using Conditional GAN (cGAN)

A complete PyTorch implementation of a **Text-to-Image Generation system** using a **Conditional Generative Adversarial Network (cGAN)**.
This project demonstrates **multimodal generative modeling** by integrating **Natural Language Processing (NLP)** and **Computer Vision**, enabling image generation conditioned on textual descriptions.

---

## 🚀 Key Features

* **Multimodal Generative Modeling**:

  * Combines text embeddings and random noise for conditional image synthesis.
  * Enables controlled image generation using natural language prompts.
* **Pre-trained Language Model Integration**:

  * Uses **BERT-base (uncased)** as a text encoder.
  * Text encoder is frozen for stable and efficient training on limited hardware.
* **Conditional GAN Framework**:

  * Generator and Discriminator jointly condition on text embeddings.
  * Adversarial training ensures realistic and diverse image outputs.
* **End-to-End Training & Generation**:

  * Fully connected architecture optimized for MNIST-scale images.
  * Supports generation of multiple image samples per text prompt.

---

## 🏗️ Architecture Overview

The model follows a **Conditional GAN (cGAN)** architecture consisting of three main components.

### 1. **Text Encoder (BERT)**

* Input: Natural language description (e.g., `"digit three"`).
* Model: **BERT-base (uncased)**.
* Output: 768-dimensional semantic embedding.
* Strategy:

  * Pre-trained weights are frozen to reduce computational cost.
  * Uses the pooled `[CLS]` token representation.

---

### 2. **Generator**

* Input:

  * Random noise vector sampled from a standard normal distribution.
  * Text embedding from the BERT encoder.
* Architecture:

  * Fully connected layers with **Batch Normalization**.
  * **ReLU** activations.
  * **Tanh** output layer.
* Output:

  * Flattened image vector reshaped to **28 × 28** grayscale image.

The Generator learns to synthesize images consistent with the provided text description.

---

### 3. **Discriminator**

* Input:

  * Image (real or generated).
  * Corresponding text embedding.
* Architecture:

  * Fully connected layers.
  * **LeakyReLU** activations.
* Output:

  * Single logit indicating whether the image-text pair is real or fake.

The Discriminator jointly evaluates **image realism** and **text-image alignment**.

---

## 📊 Results & Performance

The conditional GAN was trained for **30 epochs** on the MNIST dataset with text conditioning.

| Metric                 | Observation                    |
| ---------------------- | ------------------------------ |
| **Training Stability** | Stable adversarial convergence |
| **Mode Collapse**      | Not observed                   |
| **Text Conditioning**  | Successfully enforced          |
| **Image Diversity**    | High across noise samples      |
| **Visual Quality**     | Comparable to MLP-based GANs   |

### Image Generation

Images generated from prompts such as `"digit zero"`, `"digit five"`, and `"digit nine"` are visually recognizable and exhibit diversity across different noise samples, demonstrating effective text-to-image conditioning.

---

## 🛠️ Setup & Usage

### Prerequisites

* Python 3.8+
* PyTorch & Torchvision
* HuggingFace Transformers
* tqdm, matplotlib

### Running the Project

1. Open the Jupyter Notebook: `Project-4.ipynb`.
2. Enable GPU acceleration (recommended for faster training).
3. Run all cells to:

   * Train the Conditional GAN.
   * Monitor Generator and Discriminator losses.
   * Generate text-conditioned image samples.
   * View the final deliverables report.

---

## 📁 Project Structure

```text
DLGAI/
└── Project-4/
    ├── Project-4.ipynb    # Main implementation (Text Encoder, GAN, Training)
    ├── Project-4.pdf      # Project Report
    └── README.md          # Project Documentation
```

---

## 📌 Notes & Limitations

* The use of a fully connected GAN limits spatial coherence compared to convolutional architectures.
* Image sharpness can be improved by adopting DCGAN-style convolutional generators.
* Text conditioning is demonstrated on MNIST; richer datasets would further enhance semantic alignment.

---

## 📜 Acknowledgments

* **Goodfellow et al. (2014)** — *Generative Adversarial Networks*
* **Devlin et al. (2019)** — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*