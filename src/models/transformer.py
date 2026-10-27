"""A small encoder-decoder Transformer for machine translation."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float, max_length: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_length).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dimension, 2) *
                                (-math.log(10000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.encoding[:, :x.size(1)])


class TranslationTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 encoder_layers: int = 4, decoder_layers: int = 4,
                 ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.source_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.target_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.source_position = PositionalEncoding(d_model, dropout)
        self.target_position = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.projection = nn.Linear(d_model, vocab_size)

    def encode(self, source):
        embedded = self.source_position(self.source_embedding(source) * math.sqrt(self.d_model))
        return self.transformer.encoder(embedded, src_key_padding_mask=source.eq(0))

    def decode(self, target, memory, source):
        embedded = self.target_position(self.target_embedding(target) * math.sqrt(self.d_model))
        causal = torch.triu(torch.ones(target.size(1), target.size(1),
                                      dtype=torch.bool, device=target.device), diagonal=1)
        hidden = self.transformer.decoder(
            embedded, memory, tgt_mask=causal,
            tgt_key_padding_mask=target.eq(0),
            memory_key_padding_mask=source.eq(0),
        )
        return self.projection(hidden)

    def forward(self, source, target_input):
        return self.decode(target_input, self.encode(source), source)


def encode_text(tokenizer, text: str, limit: int, source: bool):
    tokens = tokenizer.encode(text)[:limit - 2]
    return ([2] + tokens + [3]) if source else ([2] + tokens + [3])


@torch.inference_mode()
def translate(model, tokenizer, source: str, max_source_tokens: int,
              max_new_tokens: int, temperature: float | None = None):
    device = next(model.parameters()).device
    ids = encode_text(tokenizer, source, max_source_tokens, True)
    source_ids = torch.tensor([ids], device=device)
    memory = model.encode(source_ids)
    target = torch.tensor([[2]], device=device)
    ended = False
    for _ in range(max_new_tokens):
        logits = model.decode(target, memory, source_ids)[:, -1, :]
        if temperature is None:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
        target = torch.cat((target, next_id), dim=1)
        if next_id.item() == 3:
            ended = True
            break
    output_ids = [x for x in target[0].tolist()[1:] if x not in (0, 2, 3)]
    return tokenizer.decode(output_ids), not ended

