"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Transformer for Machine Translation"

SIGNATURES:
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q       : Query,  shape (..., seq_q, d_k)
        K       : Key,    shape (..., seq_k, d_k)
        V       : Value,  shape (..., seq_k, d_v)
        mask    : Bool mask broadcastable to (..., seq_q, seq_k).
                  True → masked out.
        dropout : Optional nn.Dropout on attention weights.

    Returns:
        output : (..., seq_q, d_v)
        attn_w : (..., seq_q, seq_k)  
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    attn_w = F.softmax(scores, dim=-1)

    if dropout is not None:          
        attn_w = dropout(attn_w)

    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for encoder.

    Args:
        src     : [B, src_len]
        pad_idx : <pad> token index

    Returns:
        Bool mask [B, 1, 1, src_len]; True → PAD (masked out).
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)   # [B,1,1,src_len]


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask for decoder.
    Args:
        tgt     : [B, tgt_len]
        pad_idx : <pad> token index

    Returns:
        Bool mask [B, 1, tgt_len, tgt_len]; True → masked out.
    """
    B, T  = tgt.size()
    device = tgt.device

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)   
    pad_mask = pad_mask.expand(B, 1, T, T)                  

    causal = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0)   

    return pad_mask | causal   # [B,1,T,T]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(p=dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None  

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, d_model] → [B, heads, S, d_k]"""
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)

        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        # Dropout applied inside attention function  
        attn_out, attn_w = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )
        self.last_attn_weights = attn_w.detach()   
        # Concat heads: [B, seq_q, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, -1, self.d_model)
        return self.W_o(attn_out)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # [L,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )   # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)   # even dims → sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims  → cos

        self.register_buffer('pe', pe.unsqueeze(0))    # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW₁+b₁)W₂+b₂"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """x → [Self-Attn → Add&Norm] → [FFN → Add&Norm]"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """[B, src_len, d_model] → [B, src_len, d_model]"""
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """x → [Masked Self-Attn → A&N] → [Cross-Attn → A&N] → [FFN → A&N]"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """Stack of N EncoderLayers with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N DecoderLayers with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Args:
        src_vocab_size  : Source vocab size.  None → determined from checkpoint.
        tgt_vocab_size  : Target vocab size.  None → determined from checkpoint.
        d_model         : Model dimensionality (default 256).
        N               : Encoder/decoder depth (default 4).
        num_heads       : Attention heads (default 8).
        d_ff            : FFN inner dim (default 1024).
        dropout         : Dropout probability (default 0.1).
        checkpoint_path : Local filename for checkpoint (default 'transformer_de_en.pt').
        gdrive_file_id  : Google Drive file-id for gdown; None to skip download.
        pad_idx         : <pad> index (default 1).
        sos_idx         : <sos> index (default 2).
        eos_idx         : <eos> index (default 3).
        max_len         : Max tokens generated per infer() call (default 100).
    """

    def __init__(
        self,
        src_vocab_size:  Optional[int] = None,
        tgt_vocab_size:  Optional[int] = None,
        d_model:         int   = 256,
        N:               int   = 4,
        num_heads:       int   = 8,
        d_ff:            int   = 1024,
        dropout:         float = 0.1,
        checkpoint_path: str   = "best_model.pt",
        gdrive_file_id:  Optional[str] = "14mqQWESnajBJcjxdhd5j0pQ8bWkLQ-4Z",
        pad_idx:         int   = 1,
        sos_idx:         int   = 2,
        eos_idx:         int   = 3,
        max_len:         int   = 100,
    ) -> None:
        super().__init__()

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.d_model = d_model   

        self._spacy_de = _load_spacy("de_core_news_sm")
        self._spacy_en = _load_spacy("en_core_web_sm")

        self.src_vocab: Optional[dict] = None
        self.tgt_vocab: Optional[dict] = None
        self.src_itos:  Optional[dict] = None
        self.tgt_itos:  Optional[dict] = None

        checkpoint = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            
            if gdrive_file_id:
                gdown.download(
                    id=gdrive_file_id,
                    output=checkpoint_path,
                    quiet=False
                )
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location="cpu"
                    )
                    
        
        if checkpoint is not None:
            
            if "src_vocab" in checkpoint:
                self.src_vocab = checkpoint["src_vocab"]
                self.src_itos  = {v: k for k, v in self.src_vocab.items()}
            if "tgt_vocab" in checkpoint:
                self.tgt_vocab = checkpoint["tgt_vocab"]
                self.tgt_itos  = {v: k for k, v in self.tgt_vocab.items()}

            cfg = checkpoint.get("model_config", {})
            
            d_model         = cfg.get("d_model",    d_model)
            N               = cfg.get("N",           N)
            num_heads       = cfg.get("num_heads",   num_heads)
            d_ff            = cfg.get("d_ff",        d_ff)
            dropout         = cfg.get("dropout",     dropout)
            
            src_vocab_size  = cfg.get(
                "src_vocab_size",
                len(self.src_vocab) if self.src_vocab else src_vocab_size,
            )
            tgt_vocab_size  = cfg.get(
                "tgt_vocab_size",
                len(self.tgt_vocab) if self.tgt_vocab else tgt_vocab_size,
            )
            self.d_model = d_model  

        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError(
                "src_vocab_size and tgt_vocab_size must be provided either "
                "explicitly or via a checkpoint. "
                "Pass them as arguments or supply a valid gdrive_file_id."
            )

    
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pe        = PositionalEncoding(d_model, dropout)
        self.tgt_pe        = PositionalEncoding(d_model, dropout)

        self.output_proj        = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output_proj.weight = self.tgt_embedding.weight

        if checkpoint is not None and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            
            self._init_parameters()

    def _init_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]

    def _numericalize(self, tokens, vocab: dict):
        unk = vocab.get("<unk>", 0)
        return [vocab.get(t, unk) for t in tokens]

    def _src_embed(self, src: torch.Tensor) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        return self.src_pe(x)

    def _tgt_embed(self, tgt: torch.Tensor) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        return self.tgt_pe(x)


    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(self._src_embed(src), src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        dec_out = self.decoder(self._tgt_embed(tgt), memory, src_mask, tgt_mask)
        return self.output_proj(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Transformer.infer() requires loaded vocabularies. "
                "Ensure the checkpoint contains 'src_vocab' and 'tgt_vocab' keys, "
                "or call build_vocab() and attach the dicts to model.src_vocab / "
                "model.tgt_vocab before inference."
            )

        self.eval()
        device = next(self.parameters()).device

        tokens  = self._tokenize_de(src_sentence)
        ids     = (
            [self.sos_idx]
            + self._numericalize(tokens, self.src_vocab)
            + [self.eos_idx]
        )
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,S]

        with torch.no_grad():
            src_mask = make_src_mask(src, self.pad_idx)
            memory   = self.encode(src, src_mask)

            ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

            for _ in range(self.max_len):
                tgt_mask   = make_tgt_mask(ys, self.pad_idx)
                logits     = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok   = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
                ys         = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == self.eos_idx:
                    break

        special = {self.sos_idx, self.eos_idx, self.pad_idx}
        words   = [
            self.tgt_itos.get(idx, "<unk>")
            for idx in ys[0].tolist()
            if idx not in special
        ]
        # Detokenize
        import re
        text = " ".join(words)
        text = re.sub(r" ([.,!?:;])",    r"\1",   text)   # punctuation
        text = re.sub(r"(\w) (')([a-zA-Z])", r"\1\2\3", text)  # "it 's" → "it's"
        text = re.sub(r"(\w)(') ([a-zA-Z])", r"\1\2\3", text)  # "it' s" → "it's"
        text = re.sub(r"\( ",  "(",  text)
        text = re.sub(r" \)",  ")",  text)
        text = re.sub(r" - ",  "-",  text)
        text = re.sub(r"  +",  " ",  text)
        return text.strip()


# ══════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════
def _load_spacy(model_name: str):
    """
    Load spaCy tokenizer safely.
    """
    import spacy

    try:
        return spacy.load(model_name)

    except Exception:
        if model_name.startswith("de"):
            return spacy.blank("de")
        elif model_name.startswith("en"):
            return spacy.blank("en")
        else:
            return spacy.blank("xx")
