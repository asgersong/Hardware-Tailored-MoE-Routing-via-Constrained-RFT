import os
from typing import List, Optional, Tuple

# Avoid tokenizers fork warnings in multiprocess contexts.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import Dataset, DataLoader


class PackedDataset(Dataset):
    def __init__(self, tokens: List[torch.Tensor], block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        full = self.tokens[idx]
        return full[:-1], full[1:]


def build_dataloader(
    tokens: List[torch.Tensor],
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    ds = PackedDataset(tokens, block_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


def load_tinystories_tokens(
    tokenizer_name: str = "gpt2",
    split: str = "train",
    block_size: int = 256,
    num_samples: Optional[int] = None,
) -> Tuple[List[torch.Tensor], int]:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Please install datasets and transformers to load TinyStories.") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = block_size
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("roneneldan/TinyStories", split=split)
    if num_samples:
        dataset = dataset.select(range(num_samples))

    tokens: List[torch.Tensor] = []
    for record in dataset:
        text = record["text"]
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=block_size + 1)["input_ids"][0]
        if ids.numel() < block_size + 1:
            pad = torch.full((block_size + 1 - ids.numel(),), tokenizer.eos_token_id, dtype=torch.long)
            ids = torch.cat([ids, pad], dim=0)
        tokens.append(ids[: block_size + 1])
    return tokens, tokenizer.vocab_size
