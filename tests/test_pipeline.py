"""Correctness checks for the forecasting pipeline.

These guard the properties that would silently invalidate the reported
results rather than exercising every branch: no leakage across the
train/validation/test boundary, metrics that match their definitions, a
one-step protocol that never reads the future, and saved predictions whose
stored metrics agree with a recomputation from the stored series.

    python3 -m pytest tests/ -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.baselines import metrics, persistence, seasonal_naive
from src.prepare_sequences import ScalerParams, make_xy_windows

PRED_DIR = _ROOT / "data" / "processed" / "predictions"
SEQ_DIR = _ROOT / "data" / "processed" / "sequences"
SQUARES = [5161, 5059, 5259, 4159, 4556]
MODEL_STEMS = ["persistence", "seasonal_naive", "sarima", "lstm", "cnn"]

TEST_START = pd.Timestamp("2013-12-16 00:00", tz="Europe/Rome")
TEST_END = pd.Timestamp("2013-12-22 23:50", tz="Europe/Rome")
STEPS_PER_WEEK = 1008


def _synthetic_series(n: int = 600, freq: str = "10min") -> pd.Series:
    idx = pd.date_range("2013-11-01", periods=n, freq=freq, tz="Europe/Rome")
    t = np.arange(n, dtype=np.float64)
    values = 100.0 + 50.0 * np.sin(2 * np.pi * t / 144.0) + t * 0.01
    return pd.Series(values, index=idx, name="internet")


# --------------------------------------------------------------- metrics


def test_metrics_match_their_definitions():
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([110.0, 180.0, 55.0])
    m = metrics(y_true, y_pred)

    assert m["mae"] == pytest.approx((10 + 20 + 5) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400 + 25) / 3))
    assert m["mape"] == pytest.approx(100 * (0.10 + 0.10 + 0.10) / 3)


def test_metrics_are_zero_for_a_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    m = metrics(y, y)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["mape"] == 0.0


# ------------------------------------------------------------- baselines


def test_persistence_predicts_the_previous_observation():
    s = _synthetic_series()
    history, test = s.iloc[:500], s.iloc[500:]
    result = persistence(history, test)

    expected = s.shift(1).loc[test.index].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(result.y_pred, expected)
    # The first test prediction must come from the last history value, not NaN
    assert result.y_pred[0] == pytest.approx(history.iloc[-1])


def test_seasonal_naive_predicts_the_same_slot_one_day_earlier():
    s = _synthetic_series()
    history, test = s.iloc[:500], s.iloc[500:]
    result = seasonal_naive(history, test, period=144)

    expected = s.shift(144).loc[test.index].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(result.y_pred, expected)


def test_baselines_never_read_the_future():
    """Changing a future value must not change any earlier prediction."""
    s = _synthetic_series()
    history, test = s.iloc[:500], s.iloc[500:]
    base = persistence(history, test).y_pred

    perturbed = s.copy()
    perturbed.iloc[-1] = 1e6
    after = persistence(perturbed.iloc[:500], perturbed.iloc[500:]).y_pred

    np.testing.assert_allclose(base[:-1], after[:-1])


def test_baselines_report_zero_training_time():
    s = _synthetic_series()
    assert persistence(s.iloc[:500], s.iloc[500:]).train_seconds == 0.0
    assert seasonal_naive(s.iloc[:500], s.iloc[500:]).train_seconds == 0.0


# ------------------------------------------------------------- windowing


def test_window_target_is_the_step_after_its_history():
    s = _synthetic_series(n=400)
    X, y, stamps = make_xy_windows(s, s.index, seq_len=10)

    # The first usable target is x[10]; its window is x[0..9]
    np.testing.assert_allclose(X[0, :, 0], s.to_numpy()[0:10], rtol=1e-6)
    assert y[0] == pytest.approx(s.to_numpy()[10], rel=1e-6)
    assert stamps[0] == s.index[10]
    # Targets with fewer than seq_len prior observations are skipped
    assert len(X) == len(s) - 10


def test_windows_never_contain_their_own_target():
    """The window must end one step before the target, not include it."""
    s = _synthetic_series(n=400)
    X, y, stamps = make_xy_windows(s, s.index, seq_len=10)
    values = s.to_numpy()

    for i in (0, 5, len(X) - 1):
        target_pos = s.index.get_loc(stamps[i])
        np.testing.assert_allclose(
            X[i, :, 0], values[target_pos - 10 : target_pos], rtol=1e-6
        )
        assert y[i] == pytest.approx(values[target_pos], rel=1e-6)


def test_scaler_roundtrips():
    scaler = ScalerParams(mean=1511.4, std=1389.7)
    values = np.array([0.0, 500.0, 5000.0])
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(values)), values, rtol=1e-12
    )


# --------------------------------------------------- saved-artifact checks

pytestmark_needs_artifacts = pytest.mark.skipif(
    not (PRED_DIR / "lstm_square_5161.parquet").exists(),
    reason="prediction artifacts not built; run src/evaluate_squares.py",
)


@pytestmark_needs_artifacts
@pytest.mark.parametrize("square_id", SQUARES)
@pytest.mark.parametrize("stem", MODEL_STEMS)
def test_saved_metrics_match_a_recomputation(square_id, stem):
    """The JSON metrics must be reproducible from the stored predictions."""
    df = pd.read_parquet(PRED_DIR / f"{stem}_square_{square_id}.parquet")
    meta = json.loads((PRED_DIR / f"{stem}_square_{square_id}.json").read_text())

    recomputed = metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
    for key in ["mae", "rmse", "mape"]:
        assert recomputed[key] == pytest.approx(meta[key], rel=1e-9)


@pytestmark_needs_artifacts
@pytest.mark.parametrize("square_id", SQUARES)
@pytest.mark.parametrize("stem", MODEL_STEMS)
def test_predictions_cover_exactly_the_assignment_week(square_id, stem):
    df = pd.read_parquet(PRED_DIR / f"{stem}_square_{square_id}.parquet")
    stamps = pd.to_datetime(df["datetime"]).dt.tz_convert("Europe/Rome")

    assert len(df) == STEPS_PER_WEEK
    assert stamps.min() == TEST_START
    assert stamps.max() == TEST_END
    assert stamps.is_monotonic_increasing
    assert not stamps.duplicated().any()


@pytestmark_needs_artifacts
@pytest.mark.parametrize("square_id", SQUARES)
def test_every_model_sees_identical_ground_truth(square_id):
    """A ranking is only meaningful if the models are scored on one series.

    The tolerance is 1e-5 rather than exact because the two model families
    reach the same truth by different routes: SARIMA and the baselines read
    the float64 raw series, while the LSTM and CNN read it back from the
    float32 window tensors that PyTorch requires. The resulting disagreement
    is at most ~2e-6 relative (under 0.0003 in absolute traffic units against
    an MAE of ~86), so it cannot affect any reported comparison -- but it is
    real, and pinning it here stops it growing unnoticed.
    """
    reference = pd.read_parquet(
        PRED_DIR / f"lstm_square_{square_id}.parquet"
    )["y_true"].to_numpy(dtype=np.float64)
    for stem in MODEL_STEMS:
        other = pd.read_parquet(
            PRED_DIR / f"{stem}_square_{square_id}.parquet"
        )["y_true"].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(other, reference, rtol=1e-5)


@pytestmark_needs_artifacts
@pytest.mark.parametrize("square_id", SQUARES)
def test_scaler_is_fitted_on_training_data_only(square_id):
    """Guards the most damaging leak: scaler statistics seeing val or test."""
    meta = json.loads((SEQ_DIR / f"square_{square_id}" / "meta.json").read_text())
    train = pd.read_parquet(SEQ_DIR / f"square_{square_id}" / "train_raw.parquet")
    values = train["internet"].to_numpy(dtype=np.float64)

    assert meta["scaler"]["mean"] == pytest.approx(values.mean(), rel=1e-9)
    assert meta["scaler"]["std"] == pytest.approx(values.std(), rel=1e-9)


@pytestmark_needs_artifacts
@pytest.mark.parametrize("square_id", SQUARES)
def test_splits_are_chronological_and_disjoint(square_id):
    d = SEQ_DIR / f"square_{square_id}"
    train = pd.read_parquet(d / "train_raw.parquet").index
    val = pd.read_parquet(d / "val_raw.parquet").index
    test = pd.read_parquet(d / "test_raw.parquet").index

    assert train.max() < val.min() < val.max() < test.min()
    assert test.min() == TEST_START and test.max() == TEST_END
    assert len(train.intersection(val)) == 0
    assert len(val.intersection(test)) == 0
