import torch
import torch.nn as nn
import math

class PymTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_layers,
        sequence_length,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.register_buffer(
            "pos_enc",
            self._make_sinusoidal(sequence_length, hidden_dim),
            persistent=False
        )

        bool_mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)

        self.register_buffer(
            "causal_mask",
            bool_mask,
            persistent=False
        )

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_dim)

        self.output = nn.Linear(
            hidden_dim,
            vocab_size,
            bias=False
        )

        self.output.weight = self.embedding.weight

    def _make_sinusoidal(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tokens,pad_token_id=256):
        _, seq_len = tokens.shape

        key_padding_mask = (tokens == pad_token_id)
        

        x = self.embedding(tokens)

        x = x + self.pos_enc[:seq_len].unsqueeze(0)

        causal_mask = self.causal_mask[:seq_len, :seq_len]

        for layer in self.layers:
            x = layer(x, causal_mask,key_padding_mask)

        x = self.final_ln(x)

        logits = self.output(x)

        return logits, None

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, causal_mask,key_padding_mask):

        attn_input = self.ln1(x)

        attn_out, _ = self.attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            attn_mask=causal_mask,
            need_weights=False,
            key_padding_mask=key_padding_mask
        )

        x = x + attn_out

        ffn_input = self.ln2(x)

        ffn_out = self.ffn(ffn_input)

        x = x + ffn_out

        return x
