"""Diagnose exactly why full-scale blocking recall is 83.1% instead of the
~90% seen on the 300k-row sample — using the already-saved candidates.pkl
checkpoint, so this doesn't require re-running the ~4 hour candidate
generation step. Prompted by leaderboard scores of 0.96-0.98 (see
docs/ENTITY_RESOLUTION_APPROACH.md): at 83.1% recall, even a perfect
classifier caps out around F_0.5=0.961, so several teams scoring above that
implies their blocking recall is far higher than ours — this is a blocking
problem, not (only) a classifier problem.

For each missed true match, categorizes the cause:
  - "purged": s1 and the true match share a token, but it was dropped from
    the index (max_doc_freq/max_postings cap) — fix: relax the cap.
  - "ranked_out": the true match shares a token that's IN the index, and
    would appear in an unrestricted candidate pool, but didn't survive the
    top_k=30 cut — fix: rank by real similarity instead of raw token count,
    or raise top_k.
  - "zero_overlap": s1 and the true match share no token at all, even before
    purging — token-based blocking fundamentally can't find this one; needs
    a different signal (character n-grams, phonetic matching, embeddings).
"""
from __future__ import annotations

import pickle
import random

import pandas as pd

from src.entity_resolution.block import candidates_for_record
from src.entity_resolution.pipeline import (
    flatten_source_indexes,
    add_normalized_columns_parallel,
    build_source_indexes,
)
from src.metrics import parse_id_list

TRAIN_DIR = "data_set/student_resource/dataset/train"
CHECKPOINT_DIR = "runs/checkpoints"
SAMPLE_MISSES = 200
SEED = 42


def main():
    print("Loading candidates checkpoint...")
    with open(f"{CHECKPOINT_DIR}/candidates.pkl", "rb") as f:
        candidates = pickle.load(f)

    print("Loading + normalizing full training data (fast stages only, no candidate regen)...")
    gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    add_normalized_columns_parallel(s1)
    add_normalized_columns_parallel(s2)
    add_normalized_columns_parallel(s3)

    print("Rebuilding indexes (need the purge sets, not just final candidates)...")
    s2_indexes = build_source_indexes(s2)
    s3_indexes = build_source_indexes(s3)
    all_indexes, all_weights = flatten_source_indexes((s2_indexes, s3_indexes))

    s1_lookup = s1.set_index("entity_id")
    s2_lookup = s2.set_index("entity_id")
    s3_lookup = s3.set_index("entity_id")

    true_matches = {row.source1_entity_id: parse_id_list(row.matched_entity_ids) for row in gt.itertuples()}

    print("Finding missed true matches...")
    misses = []
    for s1_id, true_set in true_matches.items():
        if not true_set:
            continue
        cand_set = set(candidates.get(s1_id, []))
        for missed_id in true_set - cand_set:
            misses.append((s1_id, missed_id))
    print(f"  total misses: {len(misses):,}")

    random.seed(SEED)
    sample = random.sample(misses, min(SAMPLE_MISSES, len(misses)))

    counts = {"purged": 0, "ranked_out": 0, "zero_overlap": 0, "id_not_found": 0}
    examples = {"purged": [], "ranked_out": [], "zero_overlap": []}

    for s1_id, missed_id in sample:
        if s1_id not in s1_lookup.index:
            counts["id_not_found"] += 1
            continue
        cand_lookup = s2_lookup if missed_id.startswith("S2-") else s3_lookup
        if missed_id not in cand_lookup.index:
            counts["id_not_found"] += 1
            continue

        s1_row = s1_lookup.loc[s1_id]
        m_row = cand_lookup.loc[missed_id]
        s1_tokens = set(s1_row.name_tokens) | set(s1_row.addr_tokens)
        m_tokens = set(m_row.name_tokens) | set(m_row.addr_tokens)
        shared_raw = s1_tokens & m_tokens

        if not shared_raw:
            counts["zero_overlap"] += 1
            if len(examples["zero_overlap"]) < 5:
                examples["zero_overlap"].append((s1_id, missed_id, s1_row.business_name, m_row.business_name))
            continue

        # Is the shared token actually present in the (purged) index?
        query_tokens = list(s1_tokens)
        unrestricted = candidates_for_record(
            query_tokens, *all_indexes, weights=all_weights, top_k=10_000_000, min_shared_tokens=1
        )
        if missed_id in unrestricted:
            counts["ranked_out"] += 1
            rank = unrestricted.index(missed_id)
            if len(examples["ranked_out"]) < 5:
                examples["ranked_out"].append((s1_id, missed_id, rank, len(unrestricted)))
        else:
            counts["purged"] += 1
            if len(examples["purged"]) < 5:
                examples["purged"].append((s1_id, missed_id, shared_raw, s1_row.business_name, m_row.business_name))

    print(f"\n=== Root cause breakdown (sample of {len(sample)} misses) ===")
    for cause, n in counts.items():
        print(f"  {cause}: {n} ({n / len(sample):.1%})")

    print("\n--- 'zero_overlap' examples (blocking fundamentally can't find these) ---")
    for s1_id, m_id, s1_name, m_name in examples["zero_overlap"]:
        print(f"  {s1_id} {s1_name!r}  vs  {m_id} {m_name!r}")

    print("\n--- 'ranked_out' examples (present but lost to top_k cut) ---")
    for s1_id, m_id, rank, pool_size in examples["ranked_out"]:
        print(f"  {s1_id} vs {m_id}: rank {rank} of {pool_size} unrestricted candidates (top_k=30 cutoff)")

    print("\n--- 'purged' examples (shared token dropped from index) ---")
    for s1_id, m_id, shared, s1_name, m_name in examples["purged"]:
        print(f"  {s1_id} {s1_name!r} vs {m_id} {m_name!r}: shared tokens {shared} — all purged from index")


if __name__ == "__main__":
    main()
