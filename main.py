from pym_lstm import PymLSTM

model = PymLSTM()
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
print(f"Model size: {total * 4 / 1024 / 1024:.2f} MB (float32)")
print(f"Model size: {total / 1024 / 1024:.2f} MB (int8)")