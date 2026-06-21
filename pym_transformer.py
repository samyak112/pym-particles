import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, causal_mask=None, key_padding_mask=None, layer_past=None):
        B, L_q, D = x.shape

        q = self.q_proj(x).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Save current state for the next generation step
        current_layer_past = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal_mask is not None and L_q > 1:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L_q, D)

        return self.o_proj(out), current_layer_past


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.attn = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=4
        )

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, causal_mask, key_padding_mask, layer_past=None):
        attn_input = self.ln1(x)

        attn_out, current_layer_past = self.attn(
            x=attn_input,
            causal_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            layer_past=layer_past
        )

        x = x + attn_out
        ffn_input = self.ln2(x)
        ffn_out = self.ffn(ffn_input)
        x = x + ffn_out

        return x, current_layer_past


class PymTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, sequence_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.register_buffer(
            "pos_enc",
            self._make_sinusoidal(sequence_length, hidden_dim),
            persistent=False
        )

        # True means masked out (ignore)
        bool_mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", bool_mask, persistent=False)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def _make_sinusoidal(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tokens, pad_token_id=256, past_key_values=None, past_key_padding_mask=None):
        B, seq_len = tokens.shape

        if past_key_values is not None:
            step = past_key_values[0][0].size(2) 
        else:
            step = 0

        current_padding_mask = (tokens == pad_token_id)
        if past_key_padding_mask is not None:
            current_padding_mask = torch.cat([past_key_padding_mask, current_padding_mask], dim=1)

        x = self.embedding(tokens)
        
        x = x + self.pos_enc[step : step + seq_len].unsqueeze(0)

        if seq_len > 1:
            causal_mask = self.causal_mask[:seq_len, :seq_len]
        else:
            # During decoding, a 1-token query doesn't need a causal mask against past keys
            causal_mask = None 

        current_key_values = []
        
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            x, layer_cache = layer(
                x, 
                causal_mask=causal_mask, 
                key_padding_mask=current_padding_mask,
                layer_past=layer_past
            )
            current_key_values.append(layer_cache)

        x = self.final_ln(x)
        logits = self.output(x)

        return logits, current_key_values, current_padding_mask