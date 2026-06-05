import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import pickle

import os

# ── model ──────────────────────────────────────────────────────────────────────

class PymLSTM(nn.Module):
    def __init__(self, vocab_size=8192, hidden_dim=512, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm      = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output    = nn.Linear(hidden_dim, vocab_size)

        # sinusoidal table — one row per window, 100k windows is more than enough
        # self.register_buffer('pos_enc', self._make_sinusoidal(100_000, hidden_dim),persistent=False)

    def _make_sinusoidal(self, max_len, dim):
        pe       = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, tokens, win_indices,hidden=None):
        x = self.embedding(tokens)                      # [B, seq_len, hidden_dim]

        # This is needed in rare cases where the first word in two different sequences is same
        # pos = self.pos_enc[win_indices]
        # x[:, 0, :] = x[:, 0, :] + pos

        out, hidden = self.lstm(x, hidden)
        logits      = self.output(out)                  
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device
        )


# ── tokenizer ──────────────────────────────────────────────────────────────────

def train_tokenizer(chunk_path, vocab_size):
    tokenizer_path = f"tokenizer_{vocab_size}.json"

    if os.path.exists(tokenizer_path):
        print('done')
        return Tokenizer.from_file(tokenizer_path)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["[UNK]"]
    )

    tokenizer.train(files=[chunk_path], trainer=trainer)
    tokenizer.save(tokenizer_path)

    return tokenizer


# ── window builder ─────────────────────────────────────────────────────────────

def precompute_windows(token_ids, window_size=512, device='cuda'):
    # 1. Convert raw list to a 1D tensor and send it directly to the GPU
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    
    step = window_size - 1
    
    # 2. Unfold creates the overlapping windows instantly.
    # We slice tokens_tensor[:-1] for inputs, and tokens_tensor[1:] for targets
    # so they perfectly align for next-token prediction.
    inputs  = tokens_tensor[:-1].unfold(0, window_size, step)
    targets = tokens_tensor[1:].unfold(0, window_size, step)
    
    # 3. Generate the window indices directly on the GPU
    num_windows = inputs.size(0)
    win_indices = torch.arange(num_windows, dtype=torch.long, device=device)
    
    return inputs, targets, win_indices


import numpy as np

def get_byte_ids(chunk_path):
    cache_path = f"{chunk_path}.bytes.npy"

    # Load from fast numpy cache if it exists
    if os.path.exists(cache_path):
        print("loading byte cache...")
        return np.load(cache_path)

    print("reading raw bytes...")
    # Read the file strictly as raw bytes ('rb')
    with open(chunk_path, "rb") as f:
        raw_bytes = f.read()

    # Convert directly to uint8 integers (0 to 255)
    byte_ids = np.frombuffer(raw_bytes, dtype=np.uint8)
    
    np.save(cache_path, byte_ids)
    
    print(f"Loaded {len(byte_ids):,} bytes.")
    return byte_ids

def get_token_ids(chunk_path, tokenizer):
    cache_path = f"{chunk_path}.tokens.pkl"

    if os.path.exists(cache_path):
        print("loading token cache")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with open(chunk_path, "r", encoding="utf-8") as f:
        text = f.read()

    encoded = tokenizer.encode(text)
    token_ids = encoded.ids

    with open(cache_path, "wb") as f:
        pickle.dump(token_ids, f)

    print(f"Reduction ratio: {len(text) / len(encoded.ids):.2f}x")

    return token_ids


# ── training ───────────────────────────────────────────────────────────────────

model_name = 'pym_particles.pt'

def train(chunk_path, epochs=20, lr=1e-3, window_size=512,
          vocab_size=256, batch_size=64):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device       : {device}")

    # ── tokenizer ──
    print("training tokenizer...")
    # tokenizer = train_tokenizer(chunk_path, vocab_size)

    token_ids = get_byte_ids(chunk_path=chunk_path)

    # ── windows ──
    print("precomputing windows...")
    inputs, targets, win_indices = precompute_windows(token_ids, window_size)
    num_windows = len(inputs)
    ram_mb      = (inputs.nbytes + targets.nbytes) / 1024 / 1024
    print(f"total windows: {num_windows:,}")
    print(f"tensor RAM   : {ram_mb:.1f} MB")

    # ── model ──
    model     = PymLSTM(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,fused=True)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=60),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    ],
    milestones=[60]
)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters   : {n_params:,}  ({n_params * 4 / 512 / 512:.1f} MB float32)")

    # ── loop ──

    import time

    # 1. Record the start time

    inputs = inputs.to(device)
    targets = targets.to(device)
    win_indices = win_indices.to(device)

    print('starting training')

    scaler = torch.cuda.amp.GradScaler()


    for epoch in range(epochs):
        start_time = time.perf_counter()

        model.train()
        total_loss  = 0.0
        total_steps = 0

        # shuffle every epoch — important for overfitting quality
        perm = torch.randperm(num_windows, device=device)

        for batch_start in range(0, num_windows, batch_size):
            idx = perm[batch_start : batch_start + batch_size]

            inp_b = inputs[idx]
            tgt_b = targets[idx]
            win_b = win_indices[idx]

            B      = inp_b.size(0)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):


                logits, _ = model(inp_b, win_b)   # [B, window_size, vocab_size]
                # 1. Drop the final time step from the sequence dimension (dim 1)
                logits_trimmed = logits[:, :-1, :].reshape(-1, vocab_size)
                targets_trimmed = tgt_b[:, :-1].reshape(-1)

                # 2. Compute loss. The gradient will automatically only backpropagate
                # through the steps that were kept.
                loss = criterion(logits_trimmed, targets_trimmed)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()


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
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        print(f"Total Execution Time: {elapsed_time:.4f} seconds")


        # ── save ──
        model_path     = 'models/' + model_name
        tok_path       = 'models/' + str(epoch) + '.tokenizer.json'
        torch.save(model.state_dict(), model_path)
        # tokenizer.save(tok_path)
        print(f"saved model     → {model_path}")
        # print(f"saved tokenizer → {tok_path}")

    return model

if __name__ == '__main__':
    train('test.txt', epochs=100)