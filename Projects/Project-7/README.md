# Self-Supervised Learning for Named Entity Recognition (NER) using DistilBERT

A PyTorch implementation of **Self-Supervised Learning for Named Entity Recognition (NER)** using a Transformer-based architecture (**DistilBERT**).

This project demonstrates a modern **two-stage training pipeline**:

1. **Self-Supervised Pretraining (Masked Language Modeling - MLM)** on unlabeled text.
2. **Supervised Fine-Tuning** for Named Entity Recognition using labeled data.

The approach improves NER performance by leveraging contextual representations learned from large-scale unlabeled corpora before task-specific fine-tuning.

---

## 🚀 Key Features

- **Two-Stage Learning Framework**
  - Stage 1: Self-supervised pretraining (MLM)
  - Stage 2: Supervised fine-tuning (NER)

- **Transformer-Based Architecture**
  - Uses `distilbert-base-uncased`
  - Lightweight and efficient compared to BERT

- **Dynamic Masking Strategy**
  - 15% random token masking during MLM training
  - Enables contextual representation learning

- **Token-Level Classification**
  - Fine-tuned for sequence labeling
  - Supports entity categories: **PER, ORG, LOC, MISC**

- **Robust Evaluation**
  - F1-score (SeqEval)
  - Precision & Recall
  - Full classification report

- **Deployment Ready**
  - HuggingFace inference pipeline
  - Saved model artifacts for API integration

---

## 🏗 Architecture Overview

The project follows a **Self-Supervised → Supervised Transformer Pipeline**:

```

Unlabeled Text (WikiText-2)
↓
Masked Language Modeling (MLM)
↓
Pretrained DistilBERT Encoder
↓
Add Token Classification Head
↓
Fine-Tune on CoNLL-2003
↓
Named Entity Recognition

```

---

# 1️⃣ Self-Supervised Phase (MLM)

### Dataset
- **WikiText-2 (Raw Version)**
- Large unlabeled English corpus

### Objective
Predict masked tokens using contextual information.

### Components

**Tokenizer**
- `DistilBertTokenizerFast`
- Max sequence length = 128
- Padding + truncation applied

**Data Collator**
- `DataCollatorForLanguageModeling`
- `mlm=True`
- `mlm_probability=0.15`

**Model**
- `DistilBertForMaskedLM`
- Pretrained `distilbert-base-uncased`
- Fine-tuned on WikiText-2

---

# 2️⃣ Supervised Phase (NER Fine-Tuning)

### Dataset
- **CoNLL-2003**
- Standard benchmark dataset for NER

### Task
Token-level classification (Sequence Labeling)

### Components

**Label Alignment**
- Handles subword tokenization
- Ignores padding tokens using `-100`

**Model**
- `DistilBertForTokenClassification`
- Loads pretrained encoder from MLM phase
- Adds token classification head

### Entity Categories

| Label | Description |
|-------|------------|
| PER | Person |
| ORG | Organization |
| LOC | Location |
| MISC | Miscellaneous |

---

## 📊 Results & Performance

### Training Configuration

| Phase | Epochs |
|-------|--------|
| MLM Pretraining | 1 |
| NER Fine-Tuning | 2 |

### Final Evaluation (Validation Set)

| Metric | Score |
|--------|--------|
| **F1 Score** | **0.93** |
| Precision | 0.93 |
| Recall | 0.94 |
| Validation Loss | 0.058 |

### Per-Entity Performance

| Entity | F1 Score |
|--------|----------|
| PER | 0.97 |
| LOC | 0.95 |
| ORG | 0.90 |
| MISC | 0.83 |

The model demonstrates strong contextual understanding and accurate entity recognition.

---

## 🔍 Sample Inference

**Input:**

```

Barack Obama was born in Hawaii and worked at Microsoft.

````

**Output:**

- Barack Obama → PER  
- Hawaii → LOC  
- Microsoft → ORG  

This confirms the model’s ability to identify named entities from contextual cues.

---

## 🔬 Learning Strategy Explained

### Why Self-Supervised Pretraining?

NER datasets are relatively small.  
Pretraining with MLM:

- Improves contextual representations
- Reduces overfitting
- Enhances generalization
- Boosts downstream F1 performance

This follows the standard paradigm used in:

- BERT
- RoBERTa
- DistilBERT
- Modern NLP production systems

---

## 🛠 Setup & Usage

### Prerequisites

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- accelerate
- seqeval

### Install Dependencies

```bash
pip install -U transformers datasets accelerate seqeval
````

---

## ▶️ Running the Project

1. Open the Colab notebook (`Project.ipynb`)
2. Enable GPU acceleration (Recommended)
3. Run all cells to:

   * Load WikiText-2 dataset
   * Perform MLM pretraining
   * Load CoNLL-2003 dataset
   * Fine-tune for NER
   * Evaluate with F1 score
   * Generate classification report
   * Run inference pipeline
   * Save trained model

---

## 📁 Project Structure

```
DLGAI/
└── SelfSupervised-NER/
    ├── Project.ipynb        # Complete implementation (MLM + NER)
    ├── final_ner_model/     # Saved trained model
    ├── Project-Report.pdf   # Detailed report
    └── README.md            # Documentation
```

---

## 📌 Notes & Limitations

* MLM trained for 1 epoch (can be increased for better performance)
* NER fine-tuned for 2 epochs (more epochs may improve F1)
* WikiText-2 is moderate-size; larger corpora improve results
* DistilBERT trades slight accuracy for efficiency
* GPU strongly recommended

---

## 📈 Future Improvements

* Increase MLM pretraining epochs
* Use larger corpora (Wikipedia, OpenWebText)
* Experiment with larger models (BERT-base, RoBERTa)
* Add contrastive self-supervised objectives
* Deploy using FastAPI for real-time inference

---

## 📚 References

* Devlin et al., 2018 — *BERT: Pre-training of Deep Bidirectional Transformers*
* Sanh et al., 2019 — *DistilBERT*
* CoNLL-2003 Shared Task Dataset
* HuggingFace Transformers Documentation

---

## 🎯 Conclusion

This project successfully demonstrates:

✔ Self-supervised representation learning
✔ Transformer-based transfer learning
✔ Token-level NER classification
✔ End-to-end training and evaluation
✔ Deployment-ready pipeline

The achieved F1 score (~93%) confirms the effectiveness of combining self-supervised pretraining with supervised fine-tuning.