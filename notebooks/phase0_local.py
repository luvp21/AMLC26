"""Phase 0 diagnostics that need only the raw data files (no cached model
artifacts): items 1-5 and 9, plus the predicted-match-rate half of item 7
(from the already-submitted output/matching_results.tsv). Items 6, 7
(probability histogram) and 8 need PARAM Shavak's cached features and
candidates — see phase0_remote.py.

Run from repo root: PYTHONPATH=. python3 notebooks/phase0_local.py
"""
from __future__ import annotations

import os

import pandas as pd

from src.metrics import parse_id_list

ROOT = "data_set/student_resource/dataset"
OUT = "runs/phase0_local.txt"

# Explicit Unicode script ranges (Python's re has no \p{Script}).
INDIC = "ऀ-ൿ"  # Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
OTHER_NON_LATIN = (
    "Ͱ-ϿЀ-ӿ֐-׿؀-ۿ฀-๿"  # Greek Cyrillic Hebrew Arabic Thai
    "぀-ヿ一-鿿가-힯"  # Kana, CJK, Hangul
)
PATTERNS = {
    "indic": f"[{INDIC}]",
    "other_non_latin": f"[{OTHER_NON_LATIN}]",
    "any_non_ascii": r"[^\x00-\x7f]",  # context only: includes accented Latin (French)
}

lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    lines.append(msg)


def read(split: str, source: int, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        f"{ROOT}/{split}/{split}_source{source}.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=cols
    )


def match_bin(n: int) -> str:
    if n <= 3:
        return str(n)
    return "4-10" if n <= 10 else ">10"


def main() -> None:
    os.makedirs("runs", exist_ok=True)

    gt = pd.read_csv(f"{ROOT}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1_country = read("train", 1, ["entity_id", "country"]).set_index("entity_id").country
    gt["country"] = gt.source1_entity_id.map(s1_country)
    gt["n_matches"] = gt.matched_entity_ids.map(lambda s: len(parse_id_list(s)))

    log("=== Item 1: true singleton rate (train) ===")
    log(f"overall: {(gt.n_matches == 0).mean():.4f}  ({(gt.n_matches == 0).sum():,} / {len(gt):,})")
    for c, g in gt.groupby("country"):
        log(f"  {c}: {(g.n_matches == 0).mean():.4f}  ({(g.n_matches == 0).sum():,} / {len(g):,})")

    log("\n=== Item 2: match-count distribution per entity (train) ===")
    order = ["0", "1", "2", "3", "4-10", ">10"]
    dist = pd.crosstab(gt.n_matches.map(match_bin), gt.country, margins=True, normalize="columns")
    log(dist.reindex(order).round(4).to_string())

    pairs = gt[["source1_entity_id", "country", "matched_entity_ids"]].copy()
    pairs["cand"] = pairs.matched_entity_ids.map(lambda s: sorted(parse_id_list(s)))
    pairs = pairs.explode("cand").dropna(subset=["cand"])[["source1_entity_id", "country", "cand"]]
    log(f"\ntotal true (S1, S2/S3) pairs: {len(pairs):,}")

    log("\n=== Item 3: S2/S3 IDs in more than one S1 match list (train) ===")
    multiplicity = pairs.cand.value_counts()
    n_multi = int((multiplicity > 1).sum())
    log(f"distinct matched S2/S3 IDs: {len(multiplicity):,}")
    log(f"IDs appearing in >1 S1 list: {n_multi:,} ({n_multi / len(multiplicity):.4%}), max lists per ID: {multiplicity.max()}")
    log(multiplicity.value_counts().sort_index().head(10).to_string())

    log("\n=== Item 4: matches crossing country labels (train) ===")
    cand_country = pd.concat(
        [read("train", 2, ["entity_id", "country"]), read("train", 3, ["entity_id", "country"])]
    ).set_index("entity_id").country
    pairs["cand_country"] = pairs.cand.map(cand_country)
    missing = pairs.cand_country.isna().sum()
    cross = pairs[pairs.cand_country.notna() & (pairs.country != pairs.cand_country)]
    log(f"pairs with candidate ID not found in S2/S3: {missing:,}")
    log(f"cross-country pairs: {len(cross):,} ({len(cross) / len(pairs):.4%})")
    if len(cross):
        log(cross.groupby(["country", "cand_country"]).size().to_string())
    del cand_country

    log("\n=== Item 5: pool sizes per source and country ===")
    for split in ("train", "test"):
        for source in (1, 2, 3):
            counts = read(split, source, ["country"]).country.value_counts()
            share = (counts / counts.sum()).round(4)
            log(f"{split} source{source}: total {counts.sum():,}  " + ", ".join(
                f"{c}={n:,} ({share[c]:.1%})" for c, n in counts.items()))

    log("\n=== Item 7 (partial): test predicted match rate per country (submitted run) ===")
    if os.path.exists("output/matching_results.tsv"):
        test_s1 = read("test", 1, ["entity_id", "country"])
        sub = pd.read_csv("output/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
        m = test_s1.merge(sub, left_on="entity_id", right_on="source1_entity_id", how="left")
        m["k"] = m.matched_entity_ids.fillna("").map(lambda s: len(parse_id_list(s)))
        for c, g in m.groupby("country"):
            log(f"  {c}: predicted-any={(g.k > 0).mean():.4f}  avg predicted per entity={g.k.mean():.3f}")
        true_rate = gt.groupby("country").n_matches.apply(lambda s: (s > 0).mean())
        log("  (train TRUE match rate for comparison: " + ", ".join(f"{c}={v:.4f}" for c, v in true_rate.items()) + ")")
    else:
        log("  output/matching_results.tsv not found locally — skipped")

    log("\n=== Item 9: share of records with non-Latin script (raw text, before anyascii) ===")
    rows = []
    for split in ("train", "test"):
        for source in (1, 2, 3):
            df = read(split, source, ["business_name", "business_address", "country"])
            for field in ("business_name", "business_address"):
                flags = {k: df[field].str.contains(p, regex=True) for k, p in PATTERNS.items()}
                for country in sorted(df.country.unique()):
                    mask = df.country == country
                    rows.append({
                        "split": split, "source": source, "field": field.replace("business_", ""),
                        "country": country, "n": int(mask.sum()),
                        **{k: round(float(v[mask].mean()), 4) for k, v in flags.items()},
                    })
            del df
    log(pd.DataFrame(rows).to_string(index=False))

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
