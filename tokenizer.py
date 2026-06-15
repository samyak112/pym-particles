from pathlib import Path
import numpy as np

def get_byte_ids(chunk_path: str | Path, size=None) -> np.ndarray:
    chunk_path = Path(chunk_path)
    cache_path = chunk_path.with_suffix(chunk_path.suffix + ".bytes.npy")

    if cache_path.exists():
        print("loaded byte cache...")
        byte_ids = np.load(cache_path)
    else:
        print("reading raw bytes...")
        byte_ids = np.frombuffer(chunk_path.read_bytes(), dtype=np.uint8).copy()

        np.save(cache_path, byte_ids)
        print(f"loaded {len(byte_ids):,} bytes.")

    if size is not None:
        n_bytes = int(size * 1024 * 1024)
        byte_ids = byte_ids[:n_bytes]

    print(len(set(byte_ids)))

    return byte_ids