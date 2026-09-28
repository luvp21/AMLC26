"""Battle-test src/features/image.py's CLIP embedder against real ABO product
images — confirms URL fetching, caching, batching, and the transformers CLIP
API all actually work together before the sprint.
"""
import sys

sys.path.insert(0, "../..")

from src.features.image import embed_images_clip

with open("sample_image_urls.txt") as f:
    urls = [line.strip() for line in f if line.strip()]

print(f"embedding {len(urls)} real ABO product images...")
embeddings = embed_images_clip(urls, cache_dir="image_cache", batch_size=8)
print(f"embeddings shape: {embeddings.shape}")
print(f"embedding norm range: {(embeddings ** 2).sum(axis=1) ** 0.5}")
