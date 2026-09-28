import numpy as np

from src.v2.blocking import union
from src.v2.normalize import normalize_address, normalize_name, skeleton
from src.v2.prune import training_rows


def test_name_legal_forms_aliases_and_honorifics():
    r = normalize_name("M/s Sai Tech Limited Private")
    assert r["name_core"] == "sai tech" and r["name_legal_class"] == "private_limited"
    r = normalize_name("Joe's Diner L.L.C. dba Joe & Sons")
    assert r["name"] == "joe s diner llc" and r["name_alias"] == "joe and sons" and r["name_legal_class"] == "llc"
    assert normalize_name("हरि सिस्टम्स")["non_latin"] and not normalize_name("Café Été")["non_latin"]


def test_skeleton_rules():
    assert skeleton("philip shah") == "flp sh"
    assert skeleton("krishna management") == skeleton("krshna managment")


def test_address_numbers_and_france_rules():
    r = normalize_address("Rougemont, NC, 506 Medford Oakley Road", "US")
    assert r["house_num"] == "506" and r["postcode"] == ""
    r = normalize_address("Shop 15, Goregaon East Near Dindoshi Bus Depot, Mumbai, 400063", "India")
    assert (r["house_num"], r["postcode"], r["addr_landmark"]) == ("15", "400063", "dindoshi bus depot")
    r = normalize_address("5 B R. du Belvedere, 33950 Lege Cap Ferret Cedex", "France")
    assert r["addr"].startswith("5bis rue du belvedere") and r["postcode"] == "33950" and "cedex" not in r["addr"]
    assert "rue" not in normalize_address("5 B R. du Belvedere", "US")["addr"]


def test_union_merges_channels_and_marks_missing():
    out = {"n": [(np.array([1, 1]), np.array([7, 8]), np.array([0.9, 0.5], np.float32), np.array([0, 1], np.int16))],
           "b": [(np.array([1]), np.array([8]), np.array([0.8], np.float32), np.array([0], np.int16))]}
    df = union(out)
    assert df[["q", "s"]].values.tolist() == [[1, 7], [1, 8]]
    assert df.b_rank.tolist() == [99, 0] and df.b_score.tolist()[0] == 0


def test_pruner_training_rows_keep_all_positives():
    y = np.array([True, False, False, False, True, False])
    rows = training_rows(np.ones(6, bool), y, cap=3, seed=0)
    assert rows[y].all() and rows.sum() == 3


def test_cross_fit_never_scores_a_training_row_with_a_model_that_saw_it():
    from src.v2.crossfit import cross_fit

    n = 12
    X = np.arange(n)
    train = np.array([True] * 9 + [False] * 3)
    groups = np.array([0, 1, 2] * 3 + [0, 0, 0])

    class M:
        def __init__(self, rows):
            self.rows = set(np.flatnonzero(rows))

        def predict_proba(self, Xs):
            assert not self.rows & set(Xs.tolist()), "row scored by a model trained on it"
            return np.c_[np.zeros(len(Xs)), np.full(len(Xs), 0.5)]

    p, models = cross_fit(X, train, groups, 3, lambda rows, g: M(rows), log=lambda *_: None)
    assert len(models) == 3 and np.allclose(p, 0.5)
    for g, m in enumerate(models):
        assert not any(groups[i] == g for i in m.rows)


def test_folds_and_groups_use_md5_and_are_stable():
    import hashlib

    from src.v2.config import fold_of, md5_mod, xgroup_of

    ids = ["S1-714132312", "S1-106407869", "S1-1", "S1-2"]
    assert md5_mod(ids, 100).tolist() == [int(hashlib.md5(i.encode()).hexdigest(), 16) % 100 for i in ids]
    assert md5_mod(ids, 100).tolist() == [44, 63, 57, 97]
    assert fold_of(["x"] * 0).tolist() == []
    buckets = md5_mod([f"S1-{i}" for i in range(20000)], 100)
    folds = fold_of([f"S1-{i}" for i in range(20000)])
    assert ((folds == "holdout") == (buckets < 10)).all() and ((folds == "tune") == ((buckets >= 10) & (buckets < 15))).all()
    assert xgroup_of(ids).tolist() == [1, 1, 2, 0]


def test_test_countries_reuse_train_vocabulary_and_new_labels_fit_their_own():
    from src.v2.blocking import fit_vectorizer, vectorizer_for

    words = ["".join(chr(97 + (i * 7 + j * 3) % 26) for j in range(6)) for i in range(200)]
    us = fit_vectorizer([f"{w} street" for w in words], 42)
    saved = {"US": {"n": us}}
    vec, source = vectorizer_for("US", "n", saved, ["totally different text"] * 30)
    assert vec is us and source == "train vocabulary"
    vec, source = vectorizer_for("France", "n", saved, [f"rue {w[::-1]}" for w in words])
    assert vec is not us and "own text" in source


def test_pair_codes_are_int32_and_round_trip_large_ids():
    ids = np.array([f"S2-{i}" for i in range(5)] + ["S1-a", "S1-b"], dtype=object)
    big = 12_600_000
    out = {"n": [(np.array([big, 3]), np.array([5, 6]), np.array([0.9, 0.4], np.float32), np.array([0, 0], np.int16))],
           "b": []}
    df = union(out)
    assert df.q.dtype == np.int32 and df.s.dtype == np.int32 and df.n_score.dtype == np.float32
    assert sorted(zip(df.q.tolist(), df.s.tolist())) == [(3, 6), (big, 5)]
    small = df[df.q < len(ids)]
    assert ids[small.q.to_numpy()].tolist() == ["S2-3"] and ids[small.s.to_numpy()].tolist() == ["S1-b"]


def test_expected_f_picks_k_and_empty():
    from src.v2.decode import expected_f_pred

    s = np.array([1, 1, 1, 2, 2])
    p = np.array([0.95, 0.9, 0.1, 0.05, 0.02])
    keep = np.ones(5, bool)
    pred = expected_f_pred(s, p, keep, alpha=1.0, m=0.0)
    assert pred.tolist() == [True, True, False, False, False]
    # a huge alpha makes the empty prediction win everywhere
    assert not expected_f_pred(s, p, keep, alpha=1e6, m=0.0).any()


def test_france_leading_legal_forms_boxes_and_zones():
    from src.v2.normalize import normalize_record

    r = normalize_record("SARL Pessac Sportive", "15 Bis Avenue Phenix, BP 123, 33600 Pessac Cedex", "France")
    assert (r["name_core"], r["name_legal_class"]) == ("pessac sportive", "sarl")
    assert r["postcode"] == "33600" and "123" not in r["addr"] and "cedex" not in r["addr"]
    r = normalize_record("Ste Martin Distribution", "ZI des Pins, CS 40012, 44000 Nantes", "France")
    assert r["name_core"] == "martin distribution" and r["addr"].startswith("zone industrielle des pins")
    assert r["postcode"] == "44000"
    r = normalize_record("Ets Dupont SCI", "ZAC du Parc, TSA 1234, Lyon", "France")
    assert r["name_core"] == "dupont" and "zone" in r["addr"] and "1234" not in r["addr"]
    assert normalize_record("SARL", "", "France")["name_core"] == "sarl"  # never emptied
    # the address table keeps st/ste as saint/sainte
    assert "sainte" in normalize_record("x", "3 Rue Ste Catherine, Bordeaux", "France")["addr"]


def test_france_rules_do_not_touch_other_labels():
    from src.v2.normalize import normalize_record

    for country in ("US", "India", "Germany"):
        r = normalize_record("Ste Martin Sci Labs", "ZI Pins, BP 12, Pune", country)
        assert r["name_core"] == "ste martin sci labs"
        assert "zi" in r["addr"].split() and "bp" in r["addr"].split()


def test_per_country_cutoffs_and_default_for_unseen_labels():
    from src.v2.prune import apply_cutoffs

    p = np.array([0.05, 0.05, 0.05, 0.001])
    c = np.array(["US", "India", "France", "France"])
    keep = apply_cutoffs(p, c, {"US": 0.1, "India": 0.01}, default=0.01)
    assert keep.tolist() == [False, True, True, False]


def test_france_house_numbers_never_swallow_street_types():
    from src.v2.normalize import normalize_address as a

    assert a("73 Q. DE PALUDATE, BORDEAUX", "France")["house_num"] == "73"
    assert a("77 B AV PHENIX", "France")["house_num"] == a("77 bis Avenue Phenix", "France")["house_num"] == "77bis"
    assert a("N° 19 BD JULES SIMON", "France")["house_num"] == "19"
    assert a("12a rue X", "France")["house_num"] == "12a"
    assert a("73 Q. DE PALUDATE", "US")["house_num"] == "73q"  # other labels unchanged


def test_france_tail_cie_and_ei():
    from src.v2.normalize import normalize_record

    assert normalize_record("Ets Socio E.I.", "", "France")["name_core"] == "socio"
    assert normalize_record("SAS Locranirdes et Cie", "", "France")["name_core"] == "locranirdes et"


def test_second_pass_context_and_sibling_support():
    import pandas as pd

    from src.v2.sibling import SECOND_PASS_NAMES, _second_best, second_pass_features

    assert _second_best(np.array([1, 1, 1, 2]), np.array([0.2, 0.9, 0.5, 0.7])).tolist() == [0.5, 0.5, 0.5, 0.0]
    # S1 0 has three candidates: q=10 and q=11 are near-duplicates, q=12 is unrelated
    q, s, p = np.array([10, 11, 12]), np.array([0, 0, 0]), np.array([0.9, 0.3, 0.2])
    rec = pd.DataFrame({"name_core": [""] * 10 + ["acme tools", "acme tool", "zeta foods"],
                        "addr": [""] * 10 + ["5 main st", "5 main st", "9 oak rd"],
                        "house_num": [""] * 10 + ["5", "5", "9"], "postcode": [""] * 13, "source": [1] * 10 + [2, 3, 2]})
    f = pd.DataFrame(second_pass_features(q, s, p, rec, workers=1), columns=SECOND_PASS_NAMES)
    assert f.sib_support_name[1] > 0.8 and f.sib_support_name[2] < 0.3  # q=11 is backed by the strong q=10
    assert f.sib_house[1] == np.float32(0.9) and f.sib_strong_addr[1] == 1 and f.s_p_count[0] == 1
    assert f.q_is_argmax.tolist() == [1, 1, 1] and f.p1.tolist() == [np.float32(0.9), np.float32(0.3), np.float32(0.2)]


def pytest_log1p(x):
    return float(np.log1p(x))


def test_house_relation_operators():
    from src.v2.house import REL, pair_features, relation

    assert relation("452", "0452") == REL["equal"]
    assert relation("78c", "a78c") == REL["letters_only"]
    assert relation("2905", "905") == REL["digit_dropped_or_added"]
    assert relation("1704", "1705") == REL["near_1_2"]
    assert relation("5425", "1425") == REL["digit_substituted"]
    assert relation("421", "412") == REL["permuted"]
    assert relation("12", "") == REL["missing"] and relation("100", "57") == REL["different"]
    # a new unit number prepended; the S1 number still appears later in the query address
    rel, edit, in_other, other_in_self, overlap, gap, gap_log, small, same_len = pair_features("3568g", "8 004 2", "", "004", "8 2", "")
    assert rel == REL["different"] and in_other == 1.0 and other_in_self == 0.0 and overlap > 0
    # next-door neighbour vs digit typo
    assert pair_features("827", "", "", "836", "", "")[5:] == (9, pytest_log1p(9), 1.0, 1.0)
    assert pair_features("2905", "", "", "905", "", "")[7] == 0.0  # large gap, not next door
    assert pair_features("", "", "", "12", "", "")[5] == -1.0


def test_novel_extra_words_are_new_words_not_variants():
    from src.v2.normalize import normalize_name as n
    from src.v2.novel import novel_words

    def novel(a, b):
        return novel_words(n(a)["name_core"], n(b)["name"])[0]

    assert novel("Judo Comite France SAS", "Judo Comite SAS") == ["france"]
    assert novel("Centre Hospitalier De La Arts International", "Centre Hospitalier de la Arts") == ["international"]
    assert novel("Acme Holdings", "Acme Tools") == ["holdings"]
    assert novel("Pessac Sportive SAS", "Pessac Groupe SARL") == ["sportive"]
    assert novel("Sai Tech Limitet", "Sai Tech Private Limited") == []  # misspelled legal word
    assert novel("Hari Sistms", "Hari Systems") == []  # typo
    assert novel("Mr Sharma Traders", "Sharma Traders") == []  # too short


def test_name_crowd_features():
    import pandas as pd

    from src.v2.crowd import CROWD_FEATURES, build

    # S1 rids 0,1,2: 0 and 1 share the name "acme"; query 3 (empty address) sees all three
    rec = pd.DataFrame({"source": [1, 1, 1, 2], "country": ["US"] * 4, "name_core": ["acme", "acme", "zeta", "acme"],
                        "addr": ["5 main st", "7 main st", "5 main st", ""],
                        "addr_parts": ["5 main st", "7 main st", "5 main st", ""]})
    pairs = pd.DataFrame({"q": [3, 3, 3], "s": [0, 1, 2], "prune_p": [0.9, 0.6, 0.2]})
    f = build(pairs, rec)
    assert f.name_dup_country.tolist() == [2, 2, 1]
    assert f.name_dup_cand.tolist() == [2, 2, 1]
    assert f.crowd_rank.tolist() == [1, 2, 1]
    assert np.allclose(f.crowd_margin.tolist(), [0.3, -0.3, 1.0])
    assert f.q_core_equal.tolist() == [1, 1, 0] and list(f.columns[2:]) == CROWD_FEATURES
    assert f.addr_dup_country.tolist() == [2, 1, 2] and f.street_dup_country.tolist() == [3, 3, 3]
    assert f.q_name_dup_country.tolist() == [2, 2, 2] and f.q_addr_dup_country.tolist() == [0, 0, 0]


def test_em_prior_recovers_base_rate_under_label_shift():
    from src.v2.prior import em_prior

    rng = np.random.default_rng(0)
    # calibrated scores for a train prior of 0.4: positives ~Beta(5,2), negatives ~Beta(2,5)
    def sample(n, pi):
        y = rng.random(n) < pi
        s = np.where(y, rng.beta(5, 2, n), rng.beta(2, 5, n))
        return s, y
    s_tr, y_tr = sample(200000, 0.4)
    # calibrate on train: p = P(y | s) via binning
    bins = np.linspace(0, 1, 51)
    idx = np.clip(np.digitize(s_tr, bins) - 1, 0, 49)
    cal = np.array([y_tr[idx == b].mean() if (idx == b).any() else 0.0 for b in range(50)])
    s_te, _ = sample(200000, 0.1)  # test: many more negatives
    p_te = cal[np.clip(np.digitize(s_te, bins) - 1, 0, 49)]
    est = em_prior(p_te, pi_tr=y_tr.mean())
    assert abs(est - 0.1) < 0.02
    # same mix as train -> no shift
    p_same = cal[np.clip(np.digitize(sample(200000, 0.4)[0], bins) - 1, 0, 49)]
    assert abs(em_prior(p_same, pi_tr=y_tr.mean()) - 0.4) < 0.02
