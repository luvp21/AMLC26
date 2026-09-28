import numpy as np

from src.postprocess import (
    bias_correct_min,
    clip_to_train_range,
    inverse_log1p,
    log1p_transform,
    snap_to_nearest_training_value,
)


def test_log1p_roundtrip():
    y = np.array([0.0, 1.0, 99.0, 1000.0])
    assert np.allclose(inverse_log1p(log1p_transform(y)), y, atol=1e-6)


def test_clip_to_train_range():
    train = np.array([10.0, 20.0, 30.0])
    preds = np.array([5.0, 25.0, 50.0])
    clipped = clip_to_train_range(preds, train)
    assert clipped.tolist() == [10.0, 25.0, 30.0]


def test_snap_to_nearest_training_value():
    train = np.array([10.0, 20.0, 30.0])
    preds = np.array([11.0, 24.0])
    snapped = snap_to_nearest_training_value(preds, train)
    assert snapped.tolist() == [10.0, 20.0]


def test_bias_correct_min():
    a = np.array([10.0, 5.0])
    b = np.array([8.0, 7.0])
    assert bias_correct_min(a, b).tolist() == [8.0, 5.0]
