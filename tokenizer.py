from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


file_name =  'test.txt'

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(
    vocab_size=8192,
    min_frequency=1,
    special_tokens=["[UNK]"]
)

tokenizer.train(files=[file_name], trainer=trainer)

with open(file_name, 'rb') as f:
    data = f.read()

text = data.decode('utf-8')
encoded = tokenizer.encode(text)
print(f"Original chars: {len(text):,}")
print(f"Encoded tokens: {len(encoded.ids):,}")
print(f"Reduction ratio: {len(text) / len(encoded.ids):.2f}x")