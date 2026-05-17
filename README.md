# DA6401 — Assignment 3

## Transformer for German-to-English Machine Translation

PyTorch implementation of the Transformer architecture from *Attention Is All You Need* (Vaswani et al., NeurIPS 2017) for German-to-English Neural Machine Translation on the Multi30k dataset.

---

## Links

* **GitHub Repository:** `<ADD_GITHUB_LINK_HERE>`
* **Weights & Biases Report:** `https://api.wandb.ai/links/nabojwal_dl1/d81hu3wu`

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
git clone <YOUR_GITHUB_REPOSITORY_LINK>
cd assignment3
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

| Hyperparameter | Value    |
| -------------- | -------- |
| d_model        | `<FILL>` |
| Layers         | `<FILL>` |
| Heads          | `<FILL>` |
| d_ff           | `<FILL>` |
| Dropout        | `<FILL>` |
| Batch Size     | `<FILL>` |
| Warmup Steps   | `<FILL>` |

---

## Evaluation

| Metric     | Score                     |
| ---------- | ------------------------- |
| BLEU Score | `<FILL_FINAL_BLEU_SCORE>` |

---

## Notes

* Implemented entirely using low-level PyTorch modules.
* Does not use `torch.nn.MultiheadAttention`.
* Supports end-to-end inference through `Transformer.infer()`.
* Model weights are loaded inside `Transformer.__init__()` using `gdown`.

---

## Reference

> Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
