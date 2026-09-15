"""Trivial reference forecasters for the one-step-ahead task.

Two baselines that require no fitting and answer the question a reader asks
first: *does any learned model actually beat "nothing changed"?*

* **Persistence** — :math:`\\hat{x}(t+1) = x(t)`. The random-walk reference.
* **Seasonal naive** — :math:`\\hat{x}(t+1) = x(t+1-144)`, i.e. the same
  10-minute slot one day earlier. This is the natural reference given the
  dominant lag-144 ACF peak found in the exploratory analysis, and it is the
  bar SARIMA must clear, since SARIMA(1,0,1)x(0,1,0,144) is exactly this
  baseline plus an ARMA correction on the seasonally differenced series.

Both consume the *true* history at every step, matching the one-step-ahead
protocol used by SARIMA, the LSTM and the 1D-CNN: at time ``t`` the forecaster
sees observations up to and including ``t``, never beyond.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

SEASONAL_PERIOD = 144  # 24 h at 10-minute resolution


@dataclass
class BaselineResult:
    name: str
    label: str
    timestamps: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    mae: float
    mape: float
    rmse: float
    train_seconds: float
    infer_seconds: float


def metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps)) * 100.0),
    }


def _shifted_forecast(
    history: pd.Series,
    test: pd.Series,
    lag: int,
    name: str,
    label: str,
) -> BaselineResult:
    """Forecast every test point as the observation ``lag`` steps earlier.

    ``history`` must end immediately before ``test`` begins; the two are
    concatenated so the first ``lag`` test points draw their reference from
    real observations rather than being dropped.
    """
    extended = pd.concat([history, test]).astype(np.float64)
    extended = extended[~extended.index.duplicated(keep="last")].sort_index()

    t0 = time.perf_counter()
    y_pred = extended.shift(lag).loc[test.index].to_numpy(dtype=np.float64)
    infer_seconds = time.perf_counter() - t0

    if np.isnan(y_pred).any():
        raise ValueError(f"{label}: insufficient history for lag {lag}")

    y_true = test.to_numpy(dtype=np.float64)
    m = metrics(y_true, y_pred)
    return BaselineResult(
        name=name,
        label=label,
        timestamps=test.index,
        y_true=y_true,
        y_pred=y_pred,
        mae=m["mae"],
        mape=m["mape"],
        rmse=m["rmse"],
        train_seconds=0.0,  # no parameters are estimated
        infer_seconds=infer_seconds,
    )


def persistence(history: pd.Series, test: pd.Series) -> BaselineResult:
    """x̂(t+1) = x(t)."""
    return _shifted_forecast(history, test, 1, "persistence", "Persistence")


def seasonal_naive(
    history: pd.Series,
    test: pd.Series,
    period: int = SEASONAL_PERIOD,
) -> BaselineResult:
    """x̂(t+1) = x(t+1-144): same slot, previous day."""
    return _shifted_forecast(
        history, test, period, "seasonal_naive", "Seasonal naive (lag 144)"
    )


def save_predictions(
    result: BaselineResult,
    path: PathLike,
    *,
    square_id: int,
) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "datetime": result.timestamps,
            "y_true": result.y_true,
            "y_pred": result.y_pred,
            "model": result.name,
            "config": result.label,
            "square_id": square_id,
        }
    ).to_parquet(path, index=False)

    meta = {
        "square_id": square_id,
        "model": result.name,
        "config": result.label,
        "mae": result.mae,
        "mape": result.mape,
        "rmse": result.rmse,
        "train_seconds": result.train_seconds,
        "infer_seconds": result.infer_seconds,
        "test_start": str(result.timestamps[0]),
        "test_end": str(result.timestamps[-1]),
        "implementation": (
            "Closed-form shift of the observed series; no parameters are "
            "estimated, so training time is zero by construction."
        ),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path
