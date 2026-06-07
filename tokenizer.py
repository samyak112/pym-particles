from pathlib import Path
import numpy as np

def get_byte_ids(chunk_path: str | Path) -> np.ndarray:
    chunk_path = Path(chunk_path)
    cache_path = chunk_path.with_suffix(chunk_path.suffix + ".bytes.npy")

    if cache_path.exists():
        print("loaded byte cache...")
        return np.load(cache_path)

    print("reading raw bytes...")
    byte_ids = np.frombuffer(chunk_path.read_bytes(), dtype=np.uint8).copy()

    np.save(cache_path, byte_ids)
    print(f"loaded {len(byte_ids):,} bytes.")

    return byte_ids