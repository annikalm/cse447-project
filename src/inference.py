import random
from pathlib import Path
from typing import List

import torch

from data import Vocab, load_lines


def topk_from_logits(logits, vocab: Vocab, k: int = 3) -> str:
    values, indices = torch.topk(logits, k)
    # ensure unique characters in case of duplicates
    seen = set()
    chars: List[str] = []
    for idx in indices.tolist():
        ch = vocab.itos[idx]
        if ch in seen:
            continue
        seen.add(ch)
        chars.append(ch)
        if len(chars) == k:
            break
    # fallback if not enough unique chars
    while len(chars) < k:
        chars.append(random.choice(vocab.itos))
    return "".join(chars)


def predict_batch(model, vocab: Vocab, contexts: List[str], device, k: int = 3) -> List[str]:
    model.eval()
    encoded = [vocab.encode(c) for c in contexts]
    if not encoded:
        return []
    max_len = max(len(seq) for seq in encoded) or 1
    unk_idx = vocab.stoi.get("<unk>", 0)
    padded = []
    for seq in encoded:
        pad_len = max_len - len(seq)
        padded.append([unk_idx] * pad_len + seq)
    x = torch.tensor(padded, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
        last_logits = logits[:, -1, :]
    preds = [topk_from_logits(logit, vocab, k=k) for logit in last_logits]
    return preds


def predict_file(model, vocab: Vocab, input_path: str, output_path: str, batch_size: int, device):
    lines = load_lines(input_path)
    out_lines: List[str] = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        try:
            preds = predict_batch(model, vocab, batch, device=device, k=3)
        except Exception:
            # keep running if something odd happens in a batch
            preds = ["???" for _ in batch]
        out_lines.extend(preds)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in out_lines:
            f.write(p + "\n")
