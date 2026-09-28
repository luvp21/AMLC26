"""v2 paths and constants. Every stage reads/writes under OUT_DIR/<tag>/ where
tag is train, dev or test; learned artifacts (key vocab, routing, vectorizer
settings, pruner, models) live in OUT_DIR/model/."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("V2_DATA_ROOT", "data_set/student_resource/dataset")
OUT_DIR = os.environ.get("V2_OUT_DIR", "runs/v2")
MODEL_DIR = f"{OUT_DIR}/model"
WORKERS = int(os.environ.get("V2_WORKERS", min(32, mp.cpu_count())))
SEED = 42

# hash split of S1 entity ids (md5: stable across machines and library versions)
HOLDOUT_PCT, TUNE_PCT = 10, 5
XFIT_K = 3  # cross-fit groups over training-fold S1 entities (matcher and pruner)
XFIT_SALT = "xfit|"
MIN_FREE_DISK_GB = int(os.environ.get("V2_MIN_FREE_GB", "30"))  # stages abort below this

# partition keys and routing
KEY_MIN_COUNT = 50  # an address part must appear this often in its source/country to be a key
ROUTE_MIN_MATCHES = 30  # query keys with fewer training matches search the whole country
ROUTE_COVERAGE = 0.997
# label-free routing (countries without training data; blocking --unsup-routing)
UNSUP_PURITY = 0.98  # a place part maps to a key when one key holds this share of its co-occurrences
UNSUP_MIN_SUPPORT = 20
UNSUP_MAX_SELF_MISS = 0.002  # routing off for a country whose S1 fail the self-check more often
UNSUP_MAX_KEYS = 2  # a query whose parts point to more keys searches the whole country

# blocking
TFIDF = dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_df=0.05, dtype=np.float32)
VOCAB_SAMPLE = 1_000_000
TOPK = {"n": 5, "n_empty_addr": 30, "b": 10, "a": 10, "k": 5}
TFIDF_OVERRIDES = {"k": {"ngram_range": (3, 4)}}  # skeleton index: char 3-4 grams
# blocking indexes (M1: n,b; M2 adds a = address only, k = skeleton for non-Latin queries)
CHANNELS = [c for c in os.environ.get("V2_CHANNELS", "n,b,a,k").split(",") if c]
SCORE_FLOOR = 0.05
QUERY_CHUNK = 200_000

# pruner
PRUNE_MAX_RECALL_LOSS = 0.001
PRUNE_TRAIN_ROWS = 20_000_000  # all positives + sampled negatives; cutoff is still set on unsampled tune rows

S_WEIGHTS = {"US": 0.383, "India": 0.468, "LOCO": 0.150}


def tag_dir(tag: str) -> str:
    path = f"{OUT_DIR}/{tag}"
    os.makedirs(path, exist_ok=True)
    return path


def model_dir() -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    return MODEL_DIR


def md5_mod(values, mod: int, salt: str = "") -> np.ndarray:
    return np.fromiter((int(hashlib.md5((salt + v).encode()).hexdigest(), 16) % mod for v in values),
                       dtype=np.int64, count=len(values))


def fold_of(entity_ids) -> np.ndarray:
    """holdout (0-9) / tune (10-14) / train (15-99) from md5(entity_id) % 100."""
    h = md5_mod(list(entity_ids), 100)
    return np.where(h < HOLDOUT_PCT, "holdout", np.where(h < HOLDOUT_PCT + TUNE_PCT, "tune", "train"))


def xgroup_of(entity_ids) -> np.ndarray:
    """Cross-fit group 0..XFIT_K-1 from a second, salted md5."""
    return md5_mod(list(entity_ids), XFIT_K, XFIT_SALT).astype(np.int8)


def resources() -> str:
    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0]) / 2**20
    return f"disk free ~ {free_gb:.0f} GB; RAM available {mem.get('MemAvailable', 0):.0f} / {mem.get('MemTotal', 0):.0f} GB"


def stage_start(log, name: str) -> float:
    """Log resources; abort if free disk on ~ is below MIN_FREE_DISK_GB."""
    log(f"[{name}] start {time.strftime('%F %T')}; {resources()}; workers {WORKERS}")
    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
    if free_gb < MIN_FREE_DISK_GB:
        raise SystemExit(f"[{name}] aborting: only {free_gb:.0f} GB free on ~ (< {MIN_FREE_DISK_GB} GB)")
    return time.time()


def stage_end(log, name: str, t0: float) -> None:
    log(f"[{name}] done in {time.time() - t0:.0f}s; {resources()}")


def log_to(path: str):
    """print + append to a log file."""
    def log(msg: str = "") -> None:
        print(msg, flush=True)
        with open(path, "a") as f:
            f.write(str(msg) + "\n")
    return log
