"""One day -> one number, log1p(days to the event): a tokenizer turns the day into a sequence of tokens, a
sequence model reads them, and the mean of its outputs goes through a linear layer.

Tokens (the same count for both tokenizers):
  raw       C x 432000 (5 Hz)  -> 1440 tokens, one per minute (300 samples)
  features  C x 432 (200 s)    ->  216 tokens, one per 400 s (2 windows)

tokenizer
  cnn    strided 1-D convolutions (raw: 5 Hz -> 1 Hz -> 0.25 Hz -> one per 20 s -> one per 60 s)
  patch  no CNN: each token is a linear function of its own stretch of the input

arch
  lstm         bidirectional LSTM, mean over all tokens
  transformer  pre-norm encoder with learned time-of-day positions; tokens whose stretch of the day has no
               data at all are left out of the attention and of the mean

The last input channel is the 0/1 data mask (data.py).
"""
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = {"raw": 1440, "features": 216, "pca": 216}
SPAN = {"raw": 300, "features": 2, "pca": 2}      # input samples per token


def block(c_in, c_out, kernel, stride):
    return nn.Sequential(nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
                         nn.GroupNorm(8, c_out), nn.GELU())


class DayModel(nn.Module):
    def __init__(self, kind, in_channels, arch="lstm", tokenizer="cnn", width=64, hidden=64, layers=None,
                 heads=4, feedforward=128, dropout=0.1):
        super().__init__()
        self.kind, self.arch = kind, arch
        if tokenizer == "patch":
            self.tokenizer = nn.Conv1d(in_channels, width, SPAN[kind], stride=SPAN[kind])
        elif kind == "raw":
            self.tokenizer = nn.Sequential(block(in_channels, 16, 15, 5), block(16, 32, 9, 4), block(32, width, 9, 5),
                                           block(width, width, 7, 3))
        else:
            self.tokenizer = nn.Sequential(block(in_channels, width, 5, 1), block(width, width, 5, 2))
        if arch == "lstm":
            layers = 1 if layers is None else layers
            self.lstm = nn.LSTM(width, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                dropout=dropout if layers > 1 else 0.0)
            out = 2 * hidden
        else:
            layers = 3 if layers is None else layers
            self.position = nn.Parameter(torch.zeros(1, TOKENS[kind], width))
            nn.init.trunc_normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(width, heads, feedforward, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
            out = width
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out, 1))

    def forward(self, x):
        tokens = self.tokenizer(x).transpose(1, 2)                    # (batch, tokens, width)
        if self.arch == "lstm":
            out, _ = self.lstm(tokens)
            return self.head(out.mean(dim=1)).squeeze(-1)
        valid = F.adaptive_avg_pool1d(x[:, -1:], tokens.shape[1]).squeeze(1) > 0      # token has some data
        valid = valid | ~valid.any(dim=1, keepdim=True)               # a day without any data: use every token
        out = self.encoder(tokens + self.position, src_key_padding_mask=~valid)
        weight = valid.unsqueeze(-1).to(out.dtype)
        return self.head((out * weight).sum(dim=1) / weight.sum(dim=1)).squeeze(-1)
