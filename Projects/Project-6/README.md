# Masked Language Modeling (MLM) with DistilBERT

A PyTorch implementation of **Masked Language Modeling (MLM)** using a pre-trained **DistilBERT** model.
This project demonstrates **self-supervised language modeling**, where randomly masked tokens are predicted using contextual information learned from surrounding words.

---

## 🚀 Key Features

* **Self-Supervised Language Modeling**:

  * Learns contextual word representations via masked token prediction.
  * Dynamically masks tokens during training.
* **Pre-trained Transformer Integration**:

  * Uses **DistilBERT-base (uncased)**.
  * Fine-tunes a pre-trained model for the MLM task.
* **Dynamic Masking Strategy**:

  * Implements `DataCollatorForLanguageModeling`.
  * Masks 15% of tokens per training batch.
* **End-to-End Training & Inference**:

  * Trainer API used for efficient fine-tuning.
  * Includes inference pipeline for real-time masked word prediction.

---

## 🏗️ Architecture Overview

The model follows a standard **Transformer-based Masked Language Modeling pipeline**.

### 1. **Tokenizer**

* Model: `DistilBertTokenizerFast`
* Converts raw text into token IDs.
* Applies:

  * Padding (`max_length`)
  * Truncation
  * Maximum sequence length = 128

---

### 2. **Masked Data Collator**

* Implements dynamic masking:

  * `mlm=True`
  * `mlm_probability=0.15`
* Randomly masks 15% of tokens per batch.
* Ensures training is stochastic and robust.

---

### 3. **DistilBERT Model**

* Model: `distilbert-base-uncased`
* Loaded using `DistilBertForMaskedLM`
* Pre-trained on large English corpora
* Fine-tuned on WikiText-2 dataset

DistilBERT provides:

* Reduced size compared to BERT
* Faster training
* Lower memory usage
* Strong contextual representation

---

## 📊 Results & Performance

The model was trained for **2 epochs** on the WikiText-2 dataset.

| Metric                 | Observation                     |
| ---------------------- | ------------------------------- |
| **Training Stability** | Stable convergence              |
| **Validation Loss**    | ~1.97 (after 2 epochs)          |
| **Dynamic Masking**    | Successfully applied            |
| **Inference Quality**  | Accurate contextual predictions |

---

### Example Inference

Input:

```
The capital of India is [MASK].
```

Predicted outputs:

* delhi
* mumbai
* chennai
* hyderabad
* kolkata

This demonstrates the model’s ability to understand contextual word relationships.

---

## 🛠️ Setup & Usage

### Prerequisites

* Python 3.8+
* PyTorch
* HuggingFace Transformers
* HuggingFace Datasets
* accelerate

Install dependencies:

```bash
pip install -U transformers datasets accelerate
```

---

### Running the Project

1. Open the notebook: `Project-6.ipynb`
2. Enable GPU acceleration (recommended).
3. Run all cells to:

   * Load and preprocess dataset
   * Apply dynamic masking
   * Fine-tune DistilBERT
   * Evaluate model performance
   * Run inference pipeline
   * View deliverables report

---

## 📁 Project Structure

```text
DLGAI/
└── Project-6/
    ├── Project-6.ipynb    # Main implementation (MLM Training & Inference)
    ├── Project-6.pdf      # Project Report
    └── README.md          # Project Documentation
```

---

## 📌 Notes & Limitations

* Trained for 2 epochs due to computational constraints.
* Longer training improves performance and lowers perplexity.
* WikiText-2 is a moderate-size dataset; larger corpora improve language understanding.
* DistilBERT is lighter than BERT, trading some accuracy for efficiency.