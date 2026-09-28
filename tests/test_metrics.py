import numpy as np

from src.metrics import (
    entity_resolution_f_score,
    exact_match_f1,
    mape_score,
    normalize_unit_value,
    parse_id_list,
    smape,
)


def test_smape_zero_for_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    assert smape(y, y) == 0.0


def test_smape_known_value():
    # |10-20| / ((10+20)/2) * 100 = 66.67
    assert abs(smape(np.array([10.0]), np.array([20.0])) - 66.666) < 0.01


def test_mape_score_perfect_prediction_is_100():
    y = np.array([5.0, 10.0, 15.0])
    assert abs(mape_score(y, y) - 100.0) < 1e-6


def test_mape_score_never_negative():
    y_true = np.array([1.0])
    y_pred = np.array([1000.0])
    assert mape_score(y_true, y_pred) == 0.0


def test_normalize_unit_value_parses_known_units():
    assert normalize_unit_value("121 volt") == (121.0, "volt")
    assert normalize_unit_value("2.5kg") == (2.5, "kilogram")
    assert normalize_unit_value("garbage") is None


def test_exact_match_f1_all_correct():
    y_true = ["121 volt", "2.5 kg"]
    y_pred = ["121 volt", "2.5 kilogram"]
    assert exact_match_f1(y_true, y_pred) == 1.0


def test_exact_match_f1_all_wrong():
    y_true = ["121 volt"]
    y_pred = ["50 watt"]
    assert exact_match_f1(y_true, y_pred) == 0.0


def test_parse_id_list_handles_empty_and_nan():
    assert parse_id_list("") == set()
    assert parse_id_list(None) == set()
    assert parse_id_list(float("nan")) == set()
    assert parse_id_list("S2-00047,S2-00193 ") == {"S2-00047", "S2-00193"}


def test_entity_f_score_matches_spec_worked_example():
    # From the problem statement: pred={S2-00047,S2-00193,S3-00812},
    # true={S2-00047,S3-00812} -> precision=2/3, recall=1.0, F_0.5=0.714
    predictions = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    ground_truth = {"S1-00001": {"S2-00047", "S3-00812"}}
    score = entity_resolution_f_score(predictions, ground_truth)
    assert abs(score - 0.714) < 0.001


def test_entity_f_score_singleton_correct_prediction_is_perfect():
    predictions = {"S1-00003": set()}
    ground_truth = {"S1-00003": set()}
    assert entity_resolution_f_score(predictions, ground_truth) == 1.0


def test_entity_f_score_false_merge_on_singleton_is_zero():
    predictions = {"S1-00003": {"S2-00001"}}
    ground_truth = {"S1-00003": set()}
    assert entity_resolution_f_score(predictions, ground_truth) == 0.0


def test_entity_f_score_missed_singleton_prediction_defaults_to_empty():
    # entity missing from predictions entirely -> treated as empty prediction
    ground_truth = {"S1-00003": set()}
    assert entity_resolution_f_score({}, ground_truth) == 1.0


def test_entity_f_score_macro_averages_across_entities():
    predictions = {"S1-1": {"S2-1"}, "S1-2": set()}
    ground_truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    assert entity_resolution_f_score(predictions, ground_truth) == 1.0
