import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.optim import Adam

from data import Vocab


def train_model(model: nn.Module, dataloader, device, epochs: int = 3, lr: float = 1e-3):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            # use last timestep logits
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, y)
            loss.backward()
            optimizer.step()


def save_checkpoint(work_dir: str, model: nn.Module, vocab: Vocab, config: Dict):
    path = Path(work_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    vocab.save(path / "vocab.json")
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
