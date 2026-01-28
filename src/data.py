import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import torch


class Vocab:
    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        if "<unk>" not in self.stoi:
            raise ValueError("<unk> token required in vocab")

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.stoi["<unk>"]) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            itos = json.load(f)
        return cls(itos)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.rstrip("\n"))


def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [normalize(line) for line in f]


def build_vocab(lines: List[str], max_size: int = 2000, min_freq: int = 1) -> Vocab:
    counter = Counter()
    for line in lines:
        counter.update(line)
    # keep most common characters within constraints
    kept = [ch for ch, freq in counter.most_common() if freq >= min_freq][: max_size - 1]
    itos = kept + ["<unk>"]
    return Vocab(itos)


def make_training_batches(lines: List[str], vocab: Vocab, seq_len: int, batch_size: int):
    inputs = []
    targets = []
    for line in lines:
        ids = vocab.encode(line)
        if len(ids) < 2:
            continue
        for i in range(1, len(ids)):
            start = max(0, i - seq_len)
            seq = ids[start:i]
            target = ids[i]
            inputs.append(seq)
            targets.append(target)
    if not inputs:
        raise ValueError("No training pairs created; check training data")

    # pad sequences on the left for simplicity
    padded_inputs = []
    unk_idx = vocab.stoi.get("<unk>", 0)
    for seq in inputs:
        pad_len = seq_len - len(seq)
        if pad_len > 0:
            seq = [unk_idx] * pad_len + seq
        else:
            seq = seq[-seq_len:]
        padded_inputs.append(seq)

    x = torch.tensor(padded_inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
