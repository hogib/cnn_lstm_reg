"""CNN-LSTM over one day: a strided 1-D CNN turns the day into a sequence of tokens, a bidirectional LSTM
reads the sequence, and the mean of its outputs gives one number, log1p(days to the event).

  raw       2 x 432000 (5 Hz)  -> CNN, total stride 300 -> 1440 tokens (one per minute)
  features  C x 432 (200 s)    -> CNN, total stride 2   ->  216 tokens (one per 400 s)
"""
import torch
from torch import nn


def block(c_in, c_out, kernel, stride):
    return nn.Sequential(nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
                         nn.GroupNorm(8, c_out), nn.GELU())


class CnnLstm(nn.Module):
    def __init__(self, kind, in_channels, width=64, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        if kind == "raw":      # 5 Hz -> 1 Hz -> 0.25 Hz -> one per 20 s -> one per 60 s
            self.cnn = nn.Sequential(block(in_channels, 16, 15, 5), block(16, 32, 9, 4), block(32, width, 9, 5),
                                     block(width, width, 7, 3))
        else:
            self.cnn = nn.Sequential(block(in_channels, width, 5, 1), block(width, width, 5, 2))
        self.lstm = nn.LSTM(width, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        tokens = self.cnn(x).transpose(1, 2)          # (batch, tokens, width)
        out, _ = self.lstm(tokens)
        return self.head(out.mean(dim=1)).squeeze(-1)
