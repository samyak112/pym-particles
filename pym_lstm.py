import torch
import torch.nn as nn
import math

class PymLSTM(nn.Module):
    def __init__(self, vocab_size=8192, hidden_dim=200, num_layers=2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # sinusoidal position encoding for overlap tokens only
        self.register_buffer('pos_encoding', 
            self._make_sinusoidal(100_000, hidden_dim))  # 100k max windows
        
        # 2 layer LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # output projection to vocabulary
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def _make_sinusoidal(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * 
            (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, tokens, hidden=None, overlap_pos=None):
        # get token embeddings
        x = self.embedding(tokens)  # [batch, seq_len, hidden_dim]
        
        # add sinusoidal only to overlap token (first token of each window)
        if overlap_pos is not None:
            x[:, 0, :] = x[:, 0, :] + self.pos_encoding[overlap_pos]
        
        # run through LSTM
        out, hidden = self.lstm(x, hidden)
        
        # project to vocabulary
        logits = self.output(out)  # [batch, seq_len, vocab_size]
        
        return logits, hidden
    
    def init_hidden(self, batch_size=1):
        # zeros at start of each chunk
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        )