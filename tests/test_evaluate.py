import numpy as np
import pandas as pd
import pytest

from src.entity_resolution.config import SplitConfig
from src.entity_resolution.evaluate import (
    blocking_report,
    error_decomposition,
    evaluate,
    f05,
    make_splits,
    sweep_threshold,
)


def test_f05_problem_statement_example():
    # predict 3, 2 correct, 2 true -> 0.714
    assert f05([2], [3], [2])[0] == pytest.approx(0.7142857, abs=1e-6)


@pytest.mark.parametrize("T,K,TP,expected", [
    (0, 0, 0, 1.0),  # true singleton, empty prediction
    (0, 2, 0, 0.0),  # true singleton, any prediction
    (3, 0, 0, 0.0),  # has matches, empty prediction
    (3, 2, 0, 0.0),  # has matches, all predictions wrong
    (2, 2, 2, 1.0),  # perfect
])
def test_f05_edge_cases(T, K, TP, expected):
    assert f05([T], [K], [TP])[0] == pytest.approx(expected)


def _toy():
    entities = pd.DataFrame({"s1_id": ["a", "b", "c"], "country": ["US", "US", "India"], "T": [2, 0, 1]})
    pairs = pd.DataFrame({
        "s1_id": ["a", "a", "a", "b", "c"],
        "candidate_id": ["x1", "x2", "x3", "y1", "z1"],
        "rank": [0, 1, 2, 0, 0],
        "label": [True, False, False, False, False],  # a has 1 of 2 matches retrieved; c's match not retrieved
    })
    return entities, pairs


def test_evaluate_and_sweep():
    entities, pairs = _toy()
    p = np.array([0.9, 0.8, 0.1, 0.7, 0.2])
    report = evaluate(entities, pairs, p >= 0.75)
    # a: K=2 TP=1 T=2 -> 1.25/(0.5+2)=0.5 ; b: singleton, K=0 -> 1 ; c: K=0, T=1 -> 0
    assert report.loc["ALL", "macro_f05"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert report.loc["ALL", "singleton_acc"] == 1.0
    best_t, best_f, _ = sweep_threshold(entities, pairs, p, [0.5, 0.75, 0.85])
    # at 0.85: a: K=1 TP=1 -> 1.25/(0.5+1)=0.833 ; b: 1 ; c: 0 -> 0.611
    assert best_t == 0.85 and best_f == pytest.approx((1.25 / 1.5 + 1) / 3)


def test_error_decomposition_is_additive():
    entities, pairs = _toy()
    pred = np.array([True, True, False, True, False])  # also a false merge on singleton b
    d = error_decomposition(entities, pairs, pred)
    parts = d[["blocking_loss", "singleton_false_merge", "fp_on_matched", "missed_retrieved"]].sum(axis=1)
    assert np.allclose(1 - d["actual"], parts)
    assert d.loc["ALL", "singleton_false_merge"] == pytest.approx(1 / 3)


def test_blocking_report_recall_at_k():
    entities, pairs = _toy()
    rep = blocking_report(entities, pairs, ks=(1, 3))
    assert rep.loc[1, ("pair_recall", "ALL")] == pytest.approx(1 / 3)  # 1 retrieved of 3 true pairs
    assert rep.loc[3, ("entity_all_recall", "ALL")] == 0.0  # no matched entity has all matches


def test_make_splits_disjoint_sized_and_stratified():
    rng = np.random.default_rng(0)
    s1 = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(2000)],
                       "country": rng.choice(["US", "India"], size=2000, p=[0.6, 0.4])})
    s = make_splits(s1, SplitConfig(n_train=600, n_tune=100, n_val=200))
    assert (len(s["train"]), len(s["tune"]), len(s["val"])) == (600, 100, 200)
    assert not (set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["tune"]) or set(s["tune"]) & set(s["val"]))
    share_us = s1.set_index("entity_id").country.loc[s["val"]].eq("US").mean()
    assert abs(share_us - s1.country.eq("US").mean()) < 0.02


def test_record_exclusive_keeps_each_records_best_pair():
    from src.entity_resolution.harness_io import record_exclusive

    cands = ["S2-a", "S2-a", "S2-b", "S2-a", "S2-c", "S2-c"]
    p = [0.9, 0.95, 0.4, 0.2, 0.7, 0.7]
    assert record_exclusive(cands, p).tolist() == [False, True, True, False, True, False]


def test_s_components_and_bootstrap():
    import pandas as pd

    from src.entity_resolution.decision import paired_bootstrap, s_components

    ref = pd.DataFrame({"s1_id": list("abcdef"), "country": ["US"] * 3 + ["India"] * 3,
                        "f_val": [1.0, 0.5, 0.0, 1.0, 0.0, 0.5], "f_loco": [0.5] * 6})
    comp = s_components(ref)
    assert comp["F_US"] == 0.5 and comp["F_India"] == 0.5 and comp["LOCO_avg"] == 0.5
    assert abs(comp["S"] - (0.383 + 0.468 + 0.150) * 0.5) < 1e-12
    new = ref.assign(f_val=ref.f_val + 0.1)
    out = paired_bootstrap(new, ref, n_boot=200)
    assert abs(out.loc["S", "delta"] - (0.383 + 0.468) * 0.1) < 1e-12
    assert out.loc["S", "significant"] and abs(out.loc["S", "ci_low"] - out.loc["S", "delta"]) < 1e-9


def test_score_with_oof_never_scores_a_row_with_its_own_model():
    import numpy as np

    from src.entity_resolution.decision import score_with_oof

    ids = np.array(["a", "a", "b", "b", "c", "c", "d"])
    train = np.array([True, True, True, True, False, False, False])
    fitted = []

    class M:
        def __init__(self, mask):
            self.mask = mask

        def predict_proba(self, X):
            assert not (self.mask & np.isin(np.arange(7), X)).any(), "row scored by a model trained on it"
            return np.c_[np.zeros(len(X)), np.ones(len(X))]

    def fit(mask):
        fitted.append(mask.copy())
        return M(mask)

    p = score_with_oof(np.arange(7), np.zeros(7), ids, train, fit, seed=0)
    assert len(fitted) == 3 and (p == 1).all()
