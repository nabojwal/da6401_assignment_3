# DA6401 — Assignment 3

## Transformer for German-to-English Machine Translation

PyTorch implementation of the Transformer architecture from *Attention Is All You Need* (Vaswani et al., NeurIPS 2017) for German-to-English Neural Machine Translation on the Multi30k dataset.

---

## Links

* **GitHub Repository:** https://github.com/nabojwal/da6401_assignment_3
* **Weights & Biases Report:** https://api.wandb.ai/links/nabojwal_dl1/d81hu3wu

---

## Project Structure

```text id="5w4l7o"
assignment3/
├── model.py
├── dataset.py
├── lr_scheduler.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash id="y68z3s"
git clone https://github.com/nabojwal/da6401_assignment_3.git
cd da6401_assignment_3
```

### Install Dependencies

```bash id="qet4ae"
pip install -r requirements.txt
```

### Install spaCy Language Models

```bash id="q7z3we"
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

Run the complete training pipeline:

```bash id="f7qcc6"
python train.py
```

---

## Inference

```python id="99v0kp"
from model import Transformer

model = Transformer()
model.eval()

sentence = "ein mann spielt gitarre"

translation = model.infer(sentence)

print(translation)
```

---

## Hyperparameters

| Hyperparameter | Value |
| -------------- | ------|
| d_model        | `256` |
| Layers(N)      | `4`   |
| Heads          | `8`   |
| d_ff           | `1024`|
| Dropout        | `0.1` |
| Batch Size     | `128` |
| Warmup Steps   | `4000`|

---

## Evaluation

|    Best Metric  |  Score  |
| --------------- | ------- |
| Validation BLEU | `39.86` |
| Test BLEU | `40.03` |

---
## Model Configuration Note

* Although the assignment references the original Transformer-Base architecture, the original paper also explored several smaller Transformer variants. Since this project uses the significantly smaller Multi30k dataset instead of the large-scale WMT benchmark, I used a lighter configuration (`N=4`, `d_model=256`, `d_ff=1024`, `h=8`, `d_k=32`) to reduce overfitting and improve computational efficiency.
Despite the reduced parameter count, the model preserved all core Transformer components, successfully passed the provided autograder test cases, learned meaningful attention patterns, and achieved strong BLEU scores.

---

## Reference

> Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
