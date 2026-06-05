INPUT_FILE = "enwik9"
OUTPUT_FILE = "slice_100mb.txt"

SIZE_MB = 100
BYTES_TO_COPY = SIZE_MB * 1024 * 1024

with open(INPUT_FILE, "rb") as src:
    data = src.read(BYTES_TO_COPY)

with open(OUTPUT_FILE, "wb") as dst:
    dst.write(data)

print(f"Saved first {SIZE_MB} MB to {OUTPUT_FILE}")