import math
import torch
import torch.nn as nn
from tokenizer import get_byte_ids
from data_processing import generate_windows
from pym_transformer import PymTransformer
import time
import json

log_path = "loss_log.json"

def train(chunk_path,model_path,device,size, epochs=20, lr=5e-3, window_size=256,
          vocab_size=258, batch_size=64,hidden_dims=128,layers=2,):
    
    with open(log_path, "a") as f:
        f.write(json.dumps({'file_name':chunk_path,'hidden_dims':hidden_dims,'layers':layers, 'lr':lr}) + "\n")

    token_ids = get_byte_ids(chunk_path=chunk_path,size_mb=size)
    windows = generate_windows(token_ids, window_size,device=device)
    num_windows = len(windows)

    steps_per_epoch = math.ceil(num_windows / batch_size)
    total_steps = epochs * steps_per_epoch

    model     = PymTransformer(vocab_size=vocab_size,hidden_dim=hidden_dims,num_layers=layers,sequence_length=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,fused=True)
    scaler = torch.cuda.amp.GradScaler()
    warmup_steps = 500

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
    optimizer, factor=1.0, total_iters=total_steps - warmup_steps
)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, constant_scheduler],
    milestones=[warmup_steps]
)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')


    print('starting training')

    for epoch in range(epochs):
        local_start_time = time.perf_counter()

        model.train()
        total_loss = torch.tensor(0.0, device=device)
        running_steps = 0

        perm = torch.randperm(num_windows, device=device)
        shuffled_windows = windows[perm] # Do the heavy random read ONCE

        for batch_start in range(0, num_windows, batch_size):
            inp_b = shuffled_windows[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16):

                logits, _, _ = model(inp_b)
                logit_start_idx = window_size//2 - 1

                logits_trimmed = logits[:, logit_start_idx : -1, :].reshape(-1, vocab_size)

                target_start_idx = logit_start_idx + 1
                targets_trimmed = inp_b[:, target_start_idx:].reshape(-1)

                loss = criterion(logits_trimmed, targets_trimmed)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)

            scaler.step(optimizer)
            scaler.update()
            
            total_loss.add_(loss.detach())
            running_steps += 1

            scheduler.step()


        avg_loss = (total_loss / running_steps).item()
        bits_per_token = avg_loss / math.log(2)
        current_lr     = scheduler.get_last_lr()[0]
        
        local_end_time = time.perf_counter()
        elapsed_time = local_end_time - local_start_time
        log_entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "bits_per_token": bits_per_token,
            "lr": current_lr,
            "time_sec": elapsed_time
        }

        print(log_entry)

    
        if avg_loss < best_loss:
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            best_loss = avg_loss

            torch.save(
                {k: v.half() for k, v in model.state_dict().items()},
                model_path
            )

            print(f"saved model     → {model_path}")
            print(f"new best loss   → {best_loss:.4f}")

    return model

