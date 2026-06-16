from pathlib import Path
import numpy as np

def get_byte_ids(
    chunk_path: str | Path,
    size_mb: float | None = None
) -> np.ndarray:
    chunk_path = Path(chunk_path)

    cache_suffix = (
        f".{size_mb}mb.bytes.npy"
        if size_mb is not None
        else ".bytes.npy"
    )
    cache_path = chunk_path.with_suffix(chunk_path.suffix + cache_suffix)

    if cache_path.exists():
        print("loaded byte cache...")
        return np.load(cache_path)

    print("reading raw bytes...")

    raw_bytes = chunk_path.read_bytes()

    if size_mb is not None:
        max_bytes = int(size_mb * 1024 * 1024)
        raw_bytes = raw_bytes[:max_bytes]

    byte_ids = np.frombuffer(raw_bytes, dtype=np.uint8).copy()

    np.save(cache_path, byte_ids)
    print(f"loaded {len(byte_ids):,} bytes.")

    return byte_ids