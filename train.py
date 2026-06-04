import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


# ── model ──────────────────────────────────────────────────────────────────────

class PymLSTM(nn.Module):
    def __init__(self, vocab_size=8192, hidden_dim=200, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm      = nn.LSTM(hidden_dim, hidden_dim,
                                 num_layers=num_layers, batch_first=True)
        self.output    = nn.Linear(hidden_dim, vocab_size)

        # sinusoidal table — one row per window, 100k windows is more than enough
        self.register_buffer('pos_enc', self._make_sinusoidal(100_000, hidden_dim))

    def _make_sinusoidal(self, max_len, dim):
        pe       = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tokens, hidden, win_indices):
        """
        tokens:      [B, seq_len]  — token ids
        hidden:      tuple of (h, c) each [num_layers, B, hidden_dim]
        win_indices: [B]           — window index for each item in batch
                                     used to look up sinusoidal for overlap token
        """
        x = self.embedding(tokens)                      # [B, seq_len, hidden_dim]

        # add sinusoidal only to position 0 (overlap / anchor token) per window
        for i, w_idx in enumerate(win_indices):
            left_id  = 2 * w_idx.item()
            right_id = left_id + 1

            x[i, 0, :]  = x[i, 0, :]  + self.pos_enc[left_id]
            x[i, -1, :] = x[i, -1, :] + self.pos_enc[right_id]

        out, hidden = self.lstm(x, hidden)
        logits      = self.output(out)                  # [B, seq_len, vocab_size]
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


# ── tokenizer ──────────────────────────────────────────────────────────────────

def train_tokenizer(chunk_path, vocab_size=8192):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer   = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,             # include everything so UNK never appears
        special_tokens=["[UNK]"]
    )
    tokenizer.train(files=[chunk_path], trainer=trainer)
    return tokenizer


# ── window builder ─────────────────────────────────────────────────────────────

def precompute_windows(token_ids, window_size=2048):
    """
    Builds all windows upfront as tensors so training just indexes into them.

    overlap rule: last token of window N == first token of window N+1
    so step size is window_size - 1 not window_size

    window 0: input tokens[0    : 2048],  target tokens[1    : 2049]
    window 1: input tokens[2047 : 4095],  target tokens[2048 : 4096]
    window 2: input tokens[4094 : 6142],  target tokens[4095 : 6143]
    """
    inputs      = []
    targets     = []
    win_indices = []

    start   = 0
    win_idx = 0

    while start + window_size < len(token_ids):
        inputs.append(token_ids[start     : start + window_size])
        targets.append(token_ids[start + 1 : start + window_size + 1])
        win_indices.append(win_idx)

        start   += window_size - 1       # -1 creates the overlap
        win_idx += 1

    inputs      = torch.tensor(inputs,      dtype=torch.long)   # [W, 2048]
    targets     = torch.tensor(targets,     dtype=torch.long)   # [W, 2048]
    win_indices = torch.tensor(win_indices, dtype=torch.long)   # [W]

    return inputs, targets, win_indices


# ── training ───────────────────────────────────────────────────────────────────

def train(chunk_path, epochs=20, lr=1e-3, window_size=2048,
          vocab_size=8192, batch_size=32):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device       : {device}")

    # ── tokenizer ──
    print("training tokenizer...")
    tokenizer = train_tokenizer(chunk_path, vocab_size)

    with open(chunk_path, 'r', encoding='utf-8') as f:
        text = f.read()

    token_ids = tokenizer.encode(text).ids

    unk_id    = tokenizer.token_to_id("[UNK]")
    unk_count = token_ids.count(unk_id)
    print(f"total tokens : {len(token_ids):,}")
    print(f"unk tokens   : {unk_count:,}")
    assert unk_count == 0, "UNK tokens found — check min_frequency"

    # ── windows ──
    print("precomputing windows...")
    inputs, targets, win_indices = precompute_windows(token_ids, window_size)
    num_windows = len(inputs)
    ram_mb      = (inputs.nbytes + targets.nbytes) / 1024 / 1024
    print(f"total windows: {num_windows:,}")
    print(f"tensor RAM   : {ram_mb:.1f} MB")

    # ── model ──
    model     = PymLSTM(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters   : {n_params:,}  ({n_params * 4 / 1024 / 1024:.1f} MB float32)")

    # ── loop ──
    for epoch in range(epochs):
        model.train()
        total_loss  = 0.0
        total_steps = 0

        # shuffle every epoch — important for overfitting quality
        perm = torch.randperm(num_windows)

        for batch_start in range(0, num_windows, batch_size):
            idx = perm[batch_start : batch_start + batch_size]

            inp_b = inputs[idx].to(device)       # [B, 2048]
            tgt_b = targets[idx].to(device)      # [B, 2048]
            win_b = win_indices[idx]              # [B]  stays on cpu for .item()

            B      = inp_b.size(0)
            hidden = model.init_hidden(B, device)

            optimizer.zero_grad()

            logits, _ = model(inp_b, hidden, win_b)   # [B, 2048, vocab_size]

            loss = criterion(
                logits.view(-1, vocab_size),
                tgt_b.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_steps += 1

        scheduler.step()

        avg_loss       = total_loss / total_steps
        bits_per_token = avg_loss / math.log(2)
        current_lr     = scheduler.get_last_lr()[0]
        print(f"epoch {epoch + 1:3d}/{epochs}"
              f"  loss {avg_loss:.4f}"
              f"  {bits_per_token:.3f} bits/token"
              f"  lr {current_lr:.2e}")

    # ── save ──
    model_path     = chunk_path + '.model.pt'
    tok_path       = chunk_path + '.tokenizer.json'
    torch.save(model.state_dict(), model_path)
    tokenizer.save(tok_path)
    print(f"saved model     → {model_path}")
    print(f"saved tokenizer → {tok_path}")

    return model, tokenizer


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train('test.txt', epochs=20)