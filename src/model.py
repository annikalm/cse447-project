import torch
from torch import nn

class CharGRU(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.gru(emb, hidden)
        logits = self.fc(out)
        return logits, hidden
    