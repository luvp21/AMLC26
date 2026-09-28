"""Generic data loading helpers — swap in the real column names the moment
the 2026 dataset schema is published; everything else in src/ stays as-is.
"""
from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"loaded {path}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(df.isna().mean().sort_values(ascending=False).head(10))
    return df


def make_holdout_split(
    df: pd.DataFrame, target_col: str, test_size: float = 0.15, seed: int = 42, stratify_col: str | None = None
):
    stratify = df[stratify_col] if stratify_col else None
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    print(f"train: {len(train_df):,} / val: {len(val_df):,}")
    return train_df, val_df


def build_vlm_sft_dataset(
    df: pd.DataFrame,
    image_col: str,
    instruction_template: str,
    output_col: str,
    out_path: str = "data/vlm_sft.json",
) -> str:
    """Build a LLaMA-Factory-style SFT dataset (list of {images, messages})
    for VLM fine-tuning, in case the announced task is image-centric like 2024.
    `instruction_template` may reference other df columns, e.g.
        "What is the {entity_name} of this product?"
    """
    import json

    records = []
    for _, row in df.iterrows():
        instruction = instruction_template.format(**row.to_dict())
        records.append({
            "images": [row[image_col]],
            "messages": [
                {"role": "user", "content": f"<image>{instruction}"},
                {"role": "assistant", "content": str(row[output_col])},
            ],
        })
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"wrote {len(records):,} VLM SFT examples to {out_path}")
    return out_path
