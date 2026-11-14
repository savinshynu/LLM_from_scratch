import os
from datasets import load_dataset

N = 1000 # Download 1000 books
out_dir = "gutenberg_en"
os.makedirs(out_dir, exist_ok=True)

ds = load_dataset("manu/project_gutenberg", split="en", streaming=True)

for i, book in enumerate(ds):
    if i >= N:
        break
    with open(f"{out_dir}/{book['id']}.txt", "w", encoding="utf-8") as f:
        f.write(book["text"])
