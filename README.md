<h1 align="center">
Transformer for German-to-English Machine Translation
</h1>

<p align="left">
PyTorch implementation of <i>Attention Is All You Need</i> (Vaswani et al., NeurIPS 2017) for German-to-English Neural Machine Translation on the Multi30k dataset.
</p>

<p align="left">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Dataset](https://img.shields.io/badge/Dataset-Multi30k-orange)
![BLEU](https://img.shields.io/badge/Test_BLEU-40.03-brightgreen)
![Architecture](https://img.shields.io/badge/Architecture-Transformer-purple)

</p>

---
# 🔗 Links

- **GitHub Repository:**  
  https://github.com/nabojwal/da6401_assignment_3

- **Weights & Biases Report:**  
  https://api.wandb.ai/links/nabojwal_dl1/d81hu3wu

---

# 📘 Overview

This project implements the Transformer architecture proposed in *Attention Is All You Need* for German-to-English Neural Machine Translation using the Multi30k dataset.

The implementation includes:
- Multi-Head Self-Attention
- Sinusoidal Positional Encoding
- Encoder-Decoder Attention
- Noam Learning Rate Scheduler
- Label Smoothing
- Greedy Autoregressive Decoding
- BLEU Evaluation using SacreBLEU
- Attention Head Visualization  

---

# 📂 Project Structure

```text
da6401_assignment_3/
├── model.py
├── dataset.py
├── lr_scheduler.py
├── train.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/nabojwal/da6401_assignment_3.git
cd da6401_assignment_3
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Install spaCy Language Models

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

# 🏋️ Training

Run the complete training pipeline:

```bash
python train.py
```

---

# 🔍 Inference

```python
from model import Transformer

model = Transformer()
model.eval()

sentence = "ein mann spielt gitarre"

translation = model.infer(sentence)

print(translation)
```

### Example Output

```text
a man plays guitar.
```

---

# ⚙️ Model Configuration

| Hyperparameter | Value |
|---|---|
| d_model | `256` |
| Layers (N) | `4` |
| Attention Heads | `8` |
| d_ff | `1024` |
| Dropout | `0.1` |
| Batch Size | `128` |
| Warmup Steps | `4000` |
| Label Smoothing | `0.1` |
| Training Epochs | `25` |

---

# 📈 Final Performance

| Metric | Score |
|---|---|
| Validation BLEU | **39.86** |
| Test BLEU | **40.03** |

---

# 📝 Config. Note

Although the assignment references the original Transformer-Base architecture, the original paper also explored several smaller Transformer variants. Since this project uses the much smaller Multi30k dataset instead of the large-scale WMT benchmark, I used a lighter configuration (`N=4`, `d_model=256`, `d_ff=1024`, `h=8`, `d_k=32`) to reduce overfitting and improve computational efficiency.

Despite the reduced parameter count, the model preserved all core Transformer components, successfully passed the provided autograder test cases, learned meaningful attention patterns, and achieved strong BLEU scores.

---

# 📚 Reference

> Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
