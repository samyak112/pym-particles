import torch
import torch.nn as nn
from train import PymLSTM  # import your model class
import os
model_path = 'models/0.model.pt'  # path to your saved model

model = PymLSTM(vocab_size=1024).cpu().float()
model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
model.eval()

quantized = torch.quantization.quantize_dynamic(
    model,
    {nn.RNN, nn.Linear},
    dtype=torch.qint8
)

out_path = model_path.replace('.pt', '.q8.pt')
torch.save(quantized.state_dict(), out_path)

raw_mb = os.path.getsize(model_path) / 1024 / 1024
q8_mb  = os.path.getsize(out_path)  / 1024 / 1024
print(f"original:   {raw_mb:.1f} MB")
print(f"quantized:  {q8_mb:.1f} MB")
print(f"reduction:  {raw_mb/q8_mb:.2f}x")