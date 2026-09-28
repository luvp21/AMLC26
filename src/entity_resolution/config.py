"""Single source of truth for experiment settings. Every pipeline change is a
field here so any experiment can be switched off and every cached artifact is
keyed by the hash of the settings it depends on.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SplitConfig:
    scale: str = "subsample"  # "subsample" (fast iteration) | "full" (end-of-phase rerun)
    n_train: int = 300_000
    n_tune: int = 50_000  # threshold / early stopping / calibration — never the reported fold
    n_val: int = 100_000
    seed: int = 42


@dataclass(frozen=True)
class NormalizationConfig:
    """All flags off reproduces the baseline tokens exactly (Phase 2 steps)."""
    version: str = "baseline"
    keep_digits: bool = False  # digit tokens of any length + single letters next to digits
    country_abbrev: bool = False  # {country: table} + generic; French table and street-position rules
    stopwords: bool = False  # generic English + landmark words; French articles for France
    landmarks: bool = False  # "near/opp/behind X" -> addr_landmark, out of addr_tokens
    legal_bag: bool = False  # legal-form tokens among the last 3 name tokens, any order
    extract_fields: bool = False  # postal, postal_all, house_nums, name_core (stored; no effect on 13 features)


@dataclass(frozen=True)
class FeatureConfig:
    text: str = "raw"  # fuzzy ratios on: "raw" strings (baseline) | "casefold" raw.lower() | "normalized" joined tokens


@dataclass(frozen=True)
class BlockingConfig:
    method: str = "baseline_cached"  # runs/checkpoints/{candidates,test_candidates}.pkl, dict raw-count top 30
    top_k: int = 30


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    is_unbalance: bool = True
    seed: int = 42


@dataclass(frozen=True)
class DecisionConfig:
    thresholds: tuple[float, ...] = tuple(round(0.10 + 0.05 * i, 2) for i in range(18))


@dataclass(frozen=True)
class Config:
    split: SplitConfig = field(default_factory=SplitConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


def config_hash(*parts) -> str:
    """Short stable hash of one or more config sections (or plain values)."""
    payload = json.dumps([asdict(p) if hasattr(p, "__dataclass_fields__") else p for p in parts], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]
