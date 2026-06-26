from train import train
from compressor import compress
from decompressor import decompress
from pathlib import Path
import torch
from pym_transformer import PymTransformer
from data_processing import verify

MODEL_PATH      = 'models/pym_particles.pt'
VOCAB_SIZE      = 258
WINDOW_SIZE = 256
STRIDE = 128
HIDDEN_DIMS = 128
BATCH_SIZE = 64
LAYERS = 2
NUM_CHUNKS = 100
SCALE = 1_000_000

SIZE = None

input_path = input("Enter file path: ").strip()

INPUT_FILE = Path(input_path)

if not INPUT_FILE.exists():
    print("No .txt file found")

else:
    
    INPUT_FILE = str(INPUT_FILE)
    COMPRESSED_FILE = f"{INPUT_FILE}.pym"
    RECONSTRUCTED = f"{INPUT_FILE}_reconstructed.txt"
    SEED_FILE       = f'{INPUT_FILE}.bin'

    path = Path(MODEL_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extraction Device: {device}")

    if not path.exists():
        train(
            chunk_path=INPUT_FILE,
            device=device,
            model_path=str(MODEL_PATH), 
            epochs=10,
            lr=5e-3,
            window_size=WINDOW_SIZE,
            vocab_size=VOCAB_SIZE,
            batch_size=BATCH_SIZE,
            hidden_dims=HIDDEN_DIMS,
            layers=LAYERS,
            size=SIZE
            )
        
    model = PymTransformer(
            vocab_size=VOCAB_SIZE,
            hidden_dim=HIDDEN_DIMS,
            num_layers=LAYERS,
            sequence_length=WINDOW_SIZE
        ).to(device)
    
    model = torch.compile(model)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()



    path = Path(COMPRESSED_FILE)

    if not path.exists():
        compress(size=SIZE,
                input_file=INPUT_FILE,
                compressed_file=str(COMPRESSED_FILE),
                seed_path=SEED_FILE,
                window_size=WINDOW_SIZE,
                model=model,
                num_chunks=NUM_CHUNKS
                )

    decompress(
        model=model,
        compressed_path=COMPRESSED_FILE,
        stride=STRIDE,
        output_path=RECONSTRUCTED,
        seed_path=SEED_FILE,
        window_size=WINDOW_SIZE,
        num_chunks=NUM_CHUNKS,
        device=device
    )


    verify(INPUT_FILE, RECONSTRUCTED, size=SIZE)