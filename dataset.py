"""
dataset.py — Multi30k Dataset Loading & Preprocessing
DA6401 Assignment 3: "A Transformer for Machine Translation"
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Special token constants 
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

# Module-level spaCy cache 
_SPACY_CACHE: Dict[str, object] = {}


def _get_spacy(model_name: str):
    
    if model_name not in _SPACY_CACHE:
        import spacy
        try:
            _SPACY_CACHE[model_name] = spacy.load(model_name)
        except OSError:
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", model_name], check=True
            )
            _SPACY_CACHE[model_name] = spacy.load(model_name)
    return _SPACY_CACHE[model_name]



class Multi30kDataset(Dataset):
    def __init__(self, split: str = "train"):
        
        from datasets import load_dataset

        self.split = split

        self._spacy_de = _get_spacy("de_core_news_sm")
        self._spacy_en = _get_spacy("en_core_web_sm")

        dataset = load_dataset("bentrevett/multi30k")
        self._raw = dataset[split]

        self.src_vocab: Dict[str, int] = {}
        self.tgt_vocab: Dict[str, int] = {}
        self.src_itos:  Dict[int, str] = {}
        self.tgt_itos:  Dict[int, str] = {}
        self._data: List[Tuple[List[int], List[int]]] = []


    def tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self._spacy_en.tokenizer(text)]

    

    def build_vocab(
        self,
        min_freq: int = 2,
        train_dataset: Optional["Multi30kDataset"] = None,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Build src (de) and tgt (en) vocabularies.

        Args:
            min_freq      : Minimum token frequency to include.
            train_dataset : Training split instance to count from; if None,
                            uses ``self`` (appropriate when self.split=='train').

        Returns:
            src_vocab, tgt_vocab 
        """
        source = train_dataset if train_dataset is not None else self

        de_counter: Counter = Counter()
        en_counter: Counter = Counter()

        for example in source._raw:
            de_counter.update(source.tokenize_de(example["de"]))
            en_counter.update(source.tokenize_en(example["en"]))

        def _build(counter: Counter) -> Dict[str, int]:
            vocab: Dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
            for token, freq in sorted(counter.items()):   # sorted → deterministic
                if freq >= min_freq and token not in vocab:
                    vocab[token] = len(vocab)
            return vocab

        self.src_vocab = _build(de_counter)
        self.tgt_vocab = _build(en_counter)
        self.src_itos  = {v: k for k, v in self.src_vocab.items()}
        self.tgt_itos  = {v: k for k, v in self.tgt_vocab.items()}
        return self.src_vocab, self.tgt_vocab

    

    def process_data(self) -> None:
       
        if not self.src_vocab or not self.tgt_vocab:
            raise RuntimeError("Call build_vocab() before process_data().")

        self._data = []
        for example in self._raw:
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])

            src_ids = (
                [SOS_IDX]
                + [self.src_vocab.get(t, UNK_IDX) for t in src_tokens]
                + [EOS_IDX]
            )
            tgt_ids = (
                [SOS_IDX]
                + [self.tgt_vocab.get(t, UNK_IDX) for t in tgt_tokens]
                + [EOS_IDX]
            )
            self._data.append((src_ids, tgt_ids))

    

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids, tgt_ids = self._data[idx]
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def build_dataloaders(
    batch_size: int = 128,
    min_freq:   int = 2,
) -> Tuple:
    train_ds = Multi30kDataset(split="train")
    val_ds   = Multi30kDataset(split="validation")
    test_ds  = Multi30kDataset(split="test")

    src_vocab, tgt_vocab = train_ds.build_vocab(min_freq=min_freq)

    for ds in (val_ds, test_ds):
        ds.src_vocab = src_vocab
        ds.tgt_vocab = tgt_vocab
        ds.src_itos  = train_ds.src_itos
        ds.tgt_itos  = train_ds.tgt_itos

    for ds in (train_ds, val_ds, test_ds):
        ds.process_data()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn,
    )

    return (
        train_loader, val_loader, test_loader,
        src_vocab, tgt_vocab,
        train_ds.src_itos, train_ds.tgt_itos,
    )
