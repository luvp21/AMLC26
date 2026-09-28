# Pre-submit checklist

You get **5 submissions/day, 3 days** (see `docs/CHALLENGE_RULES.md`) — don't
burn one on a formatting mistake. Run through this before every submit, not
just the first one.

## Format

- [ ] Column names match the sample submission **exactly** (case, spacing, order if it matters).
- [ ] Row count == test set row count. No dropped/duplicated rows.
- [ ] ID column values match the test set 1:1 — no reordering, no reindexing drift
      (a `merge`/`join` on ID is safer than assuming row order is preserved).
- [ ] File format matches what's asked (CSV vs JSON, delimiter, header row present/absent).
- [ ] No `NaN`/`inf` in the prediction column — check with `df.isna().sum()` and
      `np.isfinite(df[col]).all()` before writing the file.

## Sanity on the predictions themselves

- [ ] Predictions are in the target's native scale, not log-space (check you called
      `inverse_log1p` / undid whatever transform `src/postprocess.py` applied).
- [ ] Range looks like the training target's range — `clip_to_train_range` applied,
      no wild outliers or negative values where the target can't be negative.
- [ ] For a classification/entity task: labels match the exact vocabulary/unit strings
      expected (see `src/metrics.normalize_unit_value` for the unit-normalization the
      2024-style scorer expects) — a correct value with the wrong unit string scores zero.
- [ ] Spot-check 10 random rows by eye against the input text/image — do they look plausible?

## Versioning

- [ ] Save every submitted CSV with a timestamp/experiment name (e.g.
      `submissions/2026-09-25_gbm-tfidf-v1.csv`) — shortlisting is based on submitted
      solutions and you may need to reproduce or explain any of them later.
- [ ] Note in a `submissions/LOG.md` (one line per submission) what changed vs the
      previous one and its public-leaderboard score, so you're not guessing which
      submission to fall back to under time pressure.

## Before the final submission of the sprint

- [ ] Confirm it's your best **public leaderboard** score, but also sanity-check it's not
      overfit to the public split — prefer the one with the best OOF/CV score if the two disagree.
- [ ] Both required artefacts attached/ready: the 1–2 page approach doc
      (`docs/APPROACH_TEMPLATE.md`) and commented source code.
