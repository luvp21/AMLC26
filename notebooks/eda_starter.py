# %% [markdown]
# EDA starter — run this first thing on Sept 25 once the real train.csv lands.
# Open in VS Code / Jupyter with `# %%` cell support, or `jupytext` to a notebook.

# %%
import pandas as pd

from src.data.loaders import load_csv

train = load_csv("data/train.csv")
test = load_csv("data/test.csv")
train.head()

# %% [markdown]
# ## Target distribution
# Check skew immediately — decide log-transform + clipping strategy
# (see src/postprocess.py) before writing a single model.

# %%
TARGET_COL = "price"  # <- rename once the real target column is known
train[TARGET_COL].describe()

# %%
train[TARGET_COL].hist(bins=80)

# %% [markdown]
# ## Text field quick look
# Confirm which columns are free text vs categorical vs already-numeric.

# %%
train.dtypes

# %% [markdown]
# ## Missingness
# Already printed by load_csv, but re-check per column that matters.

# %%
train.isna().sum().sort_values(ascending=False).head(15)

# %% [markdown]
# ## Day-1 checklist
# - [ ] confirm target column + scoring metric, implement it in src/metrics.py
# - [ ] confirm modality: text-only / image-only / text+image
# - [ ] run src.models.baseline_gbm on TF-IDF + basic_text_stats features
# - [ ] get *a* submission in within the first few hours
