
import math
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
# ── model ──────────────────────────────────────────────────────────────────────
from tokenizers.pre_tokenizers import Whitespace


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

    def forward(self, x, causal_mask):

        attn_input = self.ln1(x)

        attn_out, _ = self.attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            attn_mask=causal_mask,
            is_causal=True,
            need_weights=False
        )

        x = x + attn_out

        ffn_input = self.ln2(x)

        ffn_out = self.ffn(ffn_input)

        x = x + ffn_out

        return x

class PymTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        hidden_dim=256,
        num_layers=2,
        max_pos=512
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.register_buffer(
            "pos_enc",
            self._make_sinusoidal(max_pos, hidden_dim),
            persistent=False
        )

        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(max_pos),
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

    def _make_sinusoidal(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tokens, win_indices):
        _, seq_len = tokens.shape

        x = self.embedding(tokens)

        x = x + self.pos_enc[:seq_len].unsqueeze(0)

        causal_mask = self.causal_mask[:seq_len, :seq_len]

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.final_ln(x)

        logits = self.output(x)

        return logits, None


# ── window builder ─────────────────────────────────────────────────────────────

def train_tokenizer(chunk_path, vocab_size):
    tokenizer_path = f"tokenizer_{vocab_size}.json"

    if os.path.exists(tokenizer_path):
        print('done')
        return Tokenizer.from_file(tokenizer_path)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["[UNK]"]
    )

    tokenizer.train(files=[chunk_path], trainer=trainer)
    tokenizer.save(tokenizer_path)

    return tokenizer


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

def precompute_windows(token_ids, window_size=512, device='cuda'):
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    
    step = window_size - 1
    
    inputs  = tokens_tensor[:-1].unfold(0, window_size, step)
    targets = tokens_tensor[1:].unfold(0, window_size, step)
    
    num_windows = inputs.size(0)
    
    # win_indices is kept for compatibility with your loop, 
    # but the model no longer strictly requires it for positional slicing
    win_indices = torch.arange(num_windows, dtype=torch.long, device=device) * step
    
    return inputs, targets, win_indices

def get_byte_ids(chunk_path):
    cache_path = f"{chunk_path}.bytes.npy"

    if os.path.exists(cache_path):
        print("loading byte cache...")
        return np.load(cache_path)

    print("reading raw bytes...")
    with open(chunk_path, "rb") as f:
        raw_bytes = f.read()

    byte_ids = np.frombuffer(raw_bytes, dtype=np.uint8)
    np.save(cache_path, byte_ids)
    
    print(f"Loaded {len(byte_ids):,} bytes.")
    return byte_ids

# ── training ───────────────────────────────────────────────────────────────────

model_name = 'pym_particles.pt'

def train(chunk_path, epochs=20, lr=1e-4, window_size=256, vocab_size=256, batch_size=128):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device       : {device}")

    # ── data ──
    # tokenizer = train_tokenizer(chunk_path=chunk_path,vocab_size=vocab_size)
    token_ids = get_byte_ids(chunk_path=chunk_path)

    print("precomputing windows...")
    inputs, targets, win_indices = precompute_windows(token_ids, window_size, device)
    num_windows = len(inputs)
    ram_mb      = (inputs.nbytes + targets.nbytes) / 1024 / 1024
    print(f"total windows: {num_windows:,}")
    print(f"tensor VRAM  : {ram_mb:.1f} MB")

    # ── model ──
    model = PymTransformer(
        vocab_size=vocab_size, 
        max_pos=window_size  # Cap the sine table at exactly window_size
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    scaler = torch.cuda.amp.GradScaler() 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters   : {n_params:,}  ({n_params * 4 / 1024 / 1024:.1f} MB float32)")

    # ── loop ──
    print('starting training')

    for epoch in range(epochs):
        start_time = time.perf_counter()

        model.train()
        total_loss  = 0.0
        total_steps = 0

        perm = torch.randperm(num_windows, device=device)

        for batch_start in range(0, num_windows, batch_size):
            idx = perm[batch_start : batch_start + batch_size]

            inp_b = inputs[idx]
            tgt_b = targets[idx]
            win_b = win_indices[idx]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(inp_b, win_b)
                
                loss = criterion(
                    logits.reshape(-1, vocab_size), 
                    tgt_b.reshape(-1)
                )

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
        elapsed_time   = time.perf_counter() - start_time
        
        log_line = (f"epoch {epoch + 1:3d}/{epochs}"
              f"  loss {avg_loss:.4f}"
              f"  {bits_per_token:.3f} bits/token"
              f"  lr {current_lr:.2e}"
              f"  time {elapsed_time:.2f}s")
        
        print(log_line)
        

        with open("training.log", "a") as f:
            f.write(log_line + "\n")

        # ── save ──
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', model_name)
        torch.save(model.state_dict(), model_path)

    return model

if __name__ == '__main__':
    train('test.txt', epochs=100)