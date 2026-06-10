import math
import torch
import torch.nn as nn
from tokenizer import get_byte_ids
from data_processing import generate_windows
from pym_transformer import PymTransformer
import time
import json

model_name = 'pym_particles.pt'
log_path = "loss_log.json"



def train(chunk_path, epochs=20, lr=1e-4, window_size=256,
          vocab_size=258, batch_size=64):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device       : {device}")

    token_ids = get_byte_ids(chunk_path=chunk_path)
    windows = generate_windows(token_ids, window_size,device=device)
    num_windows = len(windows)

    steps_per_epoch = math.ceil(num_windows / batch_size)
    total_steps = epochs * steps_per_epoch

    model     = PymTransformer(vocab_size=vocab_size,hidden_dim=256,num_layers=2,sequence_length=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,fused=True)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)
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

                t0 = time.perf_counter()
                logits, _ = model(inp_b)
                torch.cuda.synchronize()
                print("forward", time.perf_counter() - t0)

                # 1. Define the index of the token making the first prediction you care about.
                # The token '4' is at index 3.
                logit_start_idx = window_size//2 - 1

                # 2. Slice logits: start from the token making the prediction, and drop 
                # the final sequence step (since the last token has no target to predict).
                # This gives us the logits at indices [3, 4, 5, 6]
                logits_trimmed = logits[:, logit_start_idx : -1, :].reshape(-1, vocab_size)

                # 3. Slice targets: the target is always one step ahead of the logit.
                # This gives us the targets [5, 6, 7, 8] at indices [4, 5, 6, 7]
                target_start_idx = logit_start_idx + 1
                targets_trimmed = inp_b[:, target_start_idx:].reshape(-1)

                # print('this is the input logit',logits_trimmed)
                # print(targets_trimmed)

                # 4. Compute loss
                loss = criterion(logits_trimmed, targets_trimmed)
                # print('this is the loss',loss)
                # print('batch',batch_start/batch_size)

            t0 = time.perf_counter()
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            print("backward", time.perf_counter() - t0)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)

            t0 = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            print("step", time.perf_counter() - t0)


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

            model_path = 'models/' + model_name

            torch.save(
                {k: v.half() for k, v in model.state_dict().items()},
                model_path
            )

            print(f"saved model     → {model_path}")
            print(f"new best loss   → {best_loss:.4f}")

    return model

train('test.txt', epochs=100)