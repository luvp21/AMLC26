"""Parse a real ABO listings shard into a flat DataFrame — proxy data for
battle-testing src/ against real messy multilingual catalog JSON before the
actual 2026 dataset drops. Scratch script, not part of the reusable src/ set.
"""
import gzip
import json

import pandas as pd


def pick_text(field, prefer_lang="en_US"):
    if not field:
        return None
    for item in field:
        if item.get("language_tag") == prefer_lang:
            return item["value"]
    return field[0].get("value")


def pick_number(entry):
    """item_weight is a LIST of {"normalized_value": {"value": 1.2, "unit": "pounds"}, ...};
    item_dimensions.<axis> is a single dict with the same normalized_value shape (no list
    wrapper) — real ABO schema quirk, handle both.
    """
    if not entry:
        return None
    if isinstance(entry, list):
        entry = entry[0]
    norm = entry.get("normalized_value")
    if norm and "value" in norm:
        return norm["value"]
    return entry.get("value")


records = []
with gzip.open("listings/metadata/listings_0.json.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        weight = pick_number(d.get("item_weight"))
        dims = d.get("item_dimensions") or {}
        height = pick_number(dims.get("height")) if isinstance(dims, dict) else None
        length = pick_number(dims.get("length")) if isinstance(dims, dict) else None
        name = pick_text(d.get("item_name"))
        bullets = d.get("bullet_point") or []
        bullet_text = " ".join(pick_text([b], b.get("language_tag")) or "" for b in bullets)
        keywords = d.get("item_keywords") or []
        keyword_text = " ".join(pick_text([k], k.get("language_tag")) or "" for k in keywords)
        product_type = (d.get("product_type") or [{}])[0].get("value")

        records.append({
            "item_id": d.get("item_id"),
            "item_name": name,
            "bullet_text": bullet_text,
            "keyword_text": keyword_text,
            "product_type": product_type,
            "main_image_id": d.get("main_image_id"),
            "item_weight": weight,
            "item_height": height,
            "item_length": length,
        })

df = pd.DataFrame(records)
print(f"parsed {len(df):,} listings from shard 0")
print(f"non-null item_name: {df['item_name'].notna().sum():,}")
print(f"non-null item_weight: {df['item_weight'].notna().sum():,}")
print(f"non-null item_height: {df['item_height'].notna().sum():,}")
print(df["product_type"].value_counts().head(10))
df.to_parquet("abo_shard0.parquet")
print("saved abo_shard0.parquet")
