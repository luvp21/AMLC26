"""Image featurization: CLIP embeddings (2025 winners' approach) plus an OCR
fallback (2024's baseline approach, useful for sanity-checking VLM outputs
even if you fine-tune a VLM as the main model).
"""
from __future__ import annotations

import io
import os

import numpy as np


def _load_image(path_or_url: str, cache_dir: str | None = None):
    from PIL import Image

    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests

        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, path_or_url.rsplit("/", 1)[-1])
            if os.path.exists(cache_path):
                return Image.open(cache_path).convert("RGB")
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if cache_path:
            img.save(cache_path)
        return img
    return Image.open(path_or_url).convert("RGB")


def embed_images_clip(
    image_paths_or_urls: list[str],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    cache_dir: str | None = "data/image_cache",
    device: str | None = None,
) -> np.ndarray:
    """CLIP image embeddings. Pair with `text.embed_texts` (ideally CLIP's own
    text tower, for a shared embedding space) for multimodal fusion — this is
    what the 2025 top-8 team did with CLIP + DistilBERT.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    all_embeds = []
    for i in range(0, len(image_paths_or_urls), batch_size):
        batch_paths = image_paths_or_urls[i : i + batch_size]
        images = [_load_image(p, cache_dir) for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        all_embeds.append(feats.cpu().numpy())
    return np.concatenate(all_embeds, axis=0)


def ocr_extract_text(image_path_or_url: str, cache_dir: str | None = "data/image_cache") -> str:
    """EasyOCR fallback for reading printed values off packaging (weight,
    voltage, dimensions) when you want a rule-based cross-check against a
    fine-tuned VLM's output, or as the day-1 baseline before fine-tuning.
    """
    import easyocr  # lazy import — install with: pip install easyocr
    import numpy as np

    reader = _get_ocr_reader()
    img = _load_image(image_path_or_url, cache_dir)
    result = reader.readtext(np.array(img), detail=0)
    return " ".join(result)


_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader
