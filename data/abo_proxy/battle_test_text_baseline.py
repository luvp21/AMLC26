"""Battle-test src/features/text.py + src/models/baseline_gbm.py +
src/metrics.py + src/postprocess.py against real ABO catalog data — proxy
for the 2026 problem statement's text-regression shape (à la 2023's product
length or 2025's price task). Target: item_weight (pounds).
"""
import sys

sys.path.insert(0, "../..")  # repo root, so `import src...` resolves

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.text import TfidfFeaturizer, basic_text_stats
from src.metrics import mape_score, smape
from src.models.baseline_gbm import train_gbm_cv
from src.models.ensemble import find_blend_weights
from src.postprocess import clip_to_train_range, inverse_log1p, log1p_transform

df = pd.read_parquet("abo_shard0.parquet")
df = df[df["item_weight"].notna()].copy()
df["text"] = (
    df["item_name"].fillna("") + " " + df["bullet_text"].fillna("") + " " + df["keyword_text"].fillna("")
)
# real data has outlier weights (e.g. furniture mixed with phone cases) — drop the top 1% so
# the proxy test isn't dominated by units-mismatch noise, same triage you'd do on real EDA
df = df[df["item_weight"] < df["item_weight"].quantile(0.99)]
print(f"rows after cleaning: {len(df):,}")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tfidf = TfidfFeaturizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(train_df["text"].tolist()).toarray()
X_test_tfidf = tfidf.transform(test_df["text"].tolist()).toarray()
X_train_stats = basic_text_stats(train_df["text"].tolist())
X_test_stats = basic_text_stats(test_df["text"].tolist())

X_train = np.hstack([X_train_tfidf, X_train_stats]).astype(np.float32)
X_test = np.hstack([X_test_tfidf, X_test_stats]).astype(np.float32)
print(f"feature matrix: {X_train.shape}")

y_train_raw = train_df["item_weight"].values
y_test_raw = test_df["item_weight"].values
y_train_log = log1p_transform(y_train_raw)

results = train_gbm_cv(X_train, y_train_log, X_test, task="regression", n_splits=5)

for name in results:
    oof_pred = inverse_log1p(results[name]["oof"])
    oof_pred = clip_to_train_range(oof_pred, y_train_raw)
    test_pred = inverse_log1p(results[name]["test"])
    test_pred = clip_to_train_range(test_pred, y_train_raw)
    print(f"\n{name}: OOF SMAPE={smape(y_train_raw, oof_pred):.2f}  "
          f"OOF score={mape_score(y_train_raw, oof_pred):.2f}  "
          f"TEST SMAPE={smape(y_test_raw, test_pred):.2f}  "
          f"TEST score={mape_score(y_test_raw, test_pred):.2f}")

oof_dict = {name: inverse_log1p(results[name]["oof"]) for name in results}
weights = find_blend_weights(oof_dict, y_train_raw, lambda yt, yp: smape(yt, yp), minimize_metric=True)
print(f"\nblend weights (SMAPE-optimal): {weights}")
blend_test = sum(w * inverse_log1p(results[name]["test"]) for name, w in weights.items())
blend_test = clip_to_train_range(blend_test, y_train_raw)
print(f"blended TEST SMAPE={smape(y_test_raw, blend_test):.2f}  score={mape_score(y_test_raw, blend_test):.2f}")
