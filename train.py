"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol,         │
  │                end_symbol, device)  → torch.Tensor [1, out_len]     │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask



def detokenize(text: str) -> str:
    """
    Detokenize a space-joined token string into clean English for BLEU.
    Args:
        text : Space-joined token string.

    Returns:
        Clean English string ready for SacreBLEU.
    """
    import re

    text = re.sub(r" ([.,!?:;])", r"\1", text)

    text = re.sub(r"(\w) (')([a-zA-Z])", r"\1\2\3", text)  # "it 's" → "it's"
    text = re.sub(r"(\w)(') ([a-zA-Z])", r"\1\2\3", text)  # "it' s" → "it's" (rare)

    text = re.sub(r"\( ", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r"\[ ", "[", text)
    text = re.sub(r" \]", "]", text)

    text = re.sub(r" - ", "-", text)

    text = re.sub(r"  +", " ", text)

    return text.strip()


class LabelSmoothingLoss(nn.Module):
    """
    Args:
        vocab_size (int)  : Output vocabulary size.
        pad_idx    (int)  : <pad> index — always receives zero probability.
        smoothing  (float): ε (default 0.1).
    """


    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        # KLDivLoss(reduction='sum') → we normalise manually by non-pad count
        self.criterion  = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [N, vocab_size]  raw model output  (N = batch * tgt_len)
            target : [N]              gold token indices

        Returns:
            Scalar mean loss over non-PAD positions.
        """
        V = self.vocab_size
        N = target.size(0)

        log_probs = torch.log_softmax(logits, dim=-1)   # [N, V]

        smooth_val  = self.smoothing / max(V - 2, 1)   # exclude true + pad
        smooth_dist = torch.full((N, V), smooth_val, device=logits.device)

        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        smooth_dist[:, self.pad_idx] = 0.0

        pad_mask = (target == self.pad_idx)
        smooth_dist[pad_mask] = 0.0

        loss = self.criterion(log_probs, smooth_dist)
        non_pad = (~pad_mask).sum().float().clamp(min=1.0)
        return loss / non_pad

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over non-PAD tokens (float).
    """
    model.train() if is_train else model.eval()
    PAD_IDX = 1

    total_loss   = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} {'Train' if is_train else 'Val '}")
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing: feed gold tokens as input, predict next token
            tgt_input  = tgt[:, :-1]   # [B, T-1]
            tgt_output = tgt[:, 1:]    # [B, T-1]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask)  # [B, T-1, V]

            B, T, V = logits.size()
            loss = loss_fn(logits.reshape(B * T, V), tgt_output.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            non_pad = (tgt_output != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_tokens, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Args:
        model        : Trained Transformer.
        src          : [1, src_len] source token indices.
        src_mask     : [1, 1, 1, src_len].
        max_len      : Maximum tokens to generate (including BOS).
        start_symbol : <sos> index.
        end_symbol   : <eos> index.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : [1, out_len]
    """
    PAD_IDX = 1
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask   = make_tgt_mask(ys, pad_idx=PAD_IDX)
            logits     = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys         = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Args:
        model           : Trained Transformer (eval mode).
        test_dataloader : Yields (src, tgt) token-index tensors.
        tgt_vocab       : idx→token dict, OR token→idx dict, OR torchtext Vocab.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus BLEU × 100.
    """
    try:
        import evaluate as hf_evaluate
        bleu_metric = hf_evaluate.load("sacrebleu")
        use_sacrebleu = True
    except Exception:
        
        try:
            import sacrebleu as sb
            use_sacrebleu = False  
        except ImportError:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            use_sacrebleu = None  

    
    if isinstance(tgt_vocab, dict):
        first_key = next(iter(tgt_vocab))
        if isinstance(first_key, int):
            itos = tgt_vocab
        else:
            itos = {v: k for k, v in tgt_vocab.items()}
    else:
        itos = {}
        for idx in range(len(tgt_vocab)):
            try:
                itos[idx] = tgt_vocab.lookup_token(idx)
            except Exception:
                itos[idx] = tgt_vocab.itos[idx]

    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    special = {PAD_IDX, SOS_IDX, EOS_IDX}

    model.eval()
    predictions = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                s_src = src[i].unsqueeze(0)
                s_tgt = tgt[i].unsqueeze(0)

                src_mask = make_src_mask(s_src, pad_idx=PAD_IDX)
                pred_ids = greedy_decode(
                    model, s_src, src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )

                pred_str = detokenize(" ".join(
                    itos.get(idx, "<unk>")
                    for idx in pred_ids[0].tolist()
                    if idx not in special
                ))
                ref_str  = detokenize(" ".join(
                    itos.get(idx, "<unk>")
                    for idx in s_tgt[0].tolist()
                    if idx not in special
                ))
                predictions.append(pred_str)
                references.append(ref_str)

    
    if use_sacrebleu is True:
        
        result = bleu_metric.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        return float(result["score"])          # already 0–100

    elif use_sacrebleu is False:
        
        import sacrebleu as sb
        bleu = sb.corpus_bleu(predictions, [references])
        return bleu.score                      # 0–100

    else:
        
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        hyps = [p.split() for p in predictions]
        refs = [[r.split()] for r in references]
        score = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
        return score * 100.0



def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Args:
        model     : Transformer instance.
        optimizer : Optimizer.
        scheduler : NoamScheduler.
        epoch     : Current epoch number.
        path      : Output file path.
    """
    model_config = {
        "src_vocab_size": model.src_embedding.num_embeddings,
        "tgt_vocab_size": model.tgt_embedding.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
        "dropout":        model.encoder.layers[0].dropout.p,
    }

    save_dict = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config":         model_config,
    }

    if hasattr(model, "src_vocab") and model.src_vocab is not None:
        save_dict["src_vocab"] = model.src_vocab
    if hasattr(model, "tgt_vocab") and model.tgt_vocab is not None:
        save_dict["tgt_vocab"] = model.tgt_vocab

    torch.save(save_dict, path)
    print(f"[checkpoint] saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Args:
        path      : Checkpoint file path.
        model     : Transformer with matching architecture.
        optimizer : Pass None to skip restore.
        scheduler : Pass None to skip restore.

    Returns:
        epoch 
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "src_vocab" in checkpoint:
        model.src_vocab = checkpoint["src_vocab"]
        model.src_itos  = {v: k for k, v in model.src_vocab.items()}
    if "tgt_vocab" in checkpoint:
        model.tgt_vocab = checkpoint["tgt_vocab"]
        model.tgt_itos  = {v: k for k, v in model.tgt_vocab.items()}

    epoch = checkpoint.get("epoch", 0)
    print(f"[checkpoint] loaded ← {path}  (epoch {epoch})")
    return epoch

def run_training_experiment() -> None:
   
    import wandb
    from dataset import build_dataloaders, PAD_IDX
    from lr_scheduler import NoamScheduler


    config = {
        "d_model":         256,
        "N":               4,
        "num_heads":       8,
        "d_ff":            1024,
        "dropout":         0.1,
        "batch_size":      128,
        "num_epochs":      25,
        "warmup_steps":    4000,
        "label_smoothing": 0.1,
        "min_freq":        2,
        "max_len":         100,
        "checkpoint_path": "best_model.pt",
    
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    wandb.init(project="da6401-a3", config=config)

    # Dataset 
    (
        train_loader, val_loader, test_loader,
        src_vocab, tgt_vocab,
        src_itos, tgt_itos,
    ) = build_dataloaders(
        batch_size=config["batch_size"],
        min_freq=config["min_freq"],
    )

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"Vocab sizes — src: {src_vocab_size}, tgt: {tgt_vocab_size}")

    # Model 
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        checkpoint_path=None,
        gdrive_file_id=None,
    ).to(device)

    # Attach vocab 
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
    model.src_itos  = src_itos
    model.tgt_itos  = tgt_itos

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # Optimiser, scheduler, loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"]
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=PAD_IDX,
        smoothing=config["label_smoothing"],
    )

    # Training loop 
    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={current_lr:.2e}"
        )
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                path=config["checkpoint_path"],
            )
            print(f"  ↳ Best checkpoint updated (val_loss={val_loss:.4f})")

    # Test BLEU 
    load_checkpoint(config["checkpoint_path"], model)
    model.eval()

    bleu = evaluate_bleu(model, test_loader, tgt_itos, device=device, max_len=config["max_len"])
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
