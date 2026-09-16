"""SARIMA one-step-ahead forecasting for a single Milan square.

Seasonal period is fixed at ``s=144`` (daily ACF peak). We implement
SARIMA(p,d,q)×(0,1,0,144) by seasonally differencing at lag 144 and fitting
a low-order ARIMA on the differenced series. That keeps the Kalman state
small (full (P,D,Q,s) MLE with s=144 is impractical at 10-minute resolution).

Limitations called out in the experiment log:
* Weekly seasonality (lag 1008) is not modelled.
* Seasonal AR/MA terms (P,Q > 0) are omitted for computational tractability;
  daily structure enters through seasonal differencing (D=1, s=144).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.prepare_sequences import PreparedSquare, prepare_square  # noqa: E402

PathLike = Union[str, Path]

SEASONAL_PERIOD = 144  # daily; weekly (1008) not modelled

# Grid over non-seasonal (p,d,q); seasonal part fixed at (0,1,0,144).
DEFAULT_ORDER_GRID: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 1),
    (1, 0, 2),
    (2, 0, 2),
    (1, 1, 1),
    (0, 1, 1),
    (2, 1, 1),
)


@dataclass(frozen=True)
class SarimaConfig:
    """SARIMA(p,d,q)×(0,1,0,s) with s=144."""

    order: Tuple[int, int, int]
    seasonal_period: int = SEASONAL_PERIOD

    @property
    def seasonal_order(self) -> Tuple[int, int, int, int]:
        return (0, 1, 0, self.seasonal_period)

    def label(self) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        return f"SARIMA({p},{d},{q})x({P},{D},{Q},{s})"


@dataclass
class GridResult:
    config: SarimaConfig
    val_mae: float
    val_rmse: float
    fit_seconds: float
    aic: Optional[float]
    status: str
    error: Optional[str] = None


@dataclass
class SarimaForecastResult:
    config: SarimaConfig
    timestamps: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    mae: float
    mape: float
    rmse: float
    train_seconds: float
    infer_seconds: float
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def fit_seconds(self) -> float:
        """Back-compat alias for train_seconds."""
        return self.train_seconds


def metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps)) * 100.0)
    return {"mae": mae, "mape": mape, "rmse": rmse}


def _seasonal_difference(series: pd.Series, s: int = SEASONAL_PERIOD) -> pd.Series:
    return series.diff(s).dropna()


def _fit_arima_on_seasonal_diff(
    history: pd.Series,
    config: SarimaConfig,
    *,
    maxiter: int = 100,
):
    """Fit ARIMA(p,d,q) on Δ_s history ⇔ SARIMA(p,d,q)×(0,1,0,s)."""
    sd = _seasonal_difference(history, config.seasonal_period)
    model = ARIMA(
        sd.astype(np.float64),
        order=config.order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        # ARIMA.fit uses the statespace MLE by default in statsmodels ≥0.12
        return model.fit()


def one_step_forecast(
    fit_result,
    history: pd.Series,
    horizon: pd.Series,
    config: SarimaConfig,
) -> np.ndarray:
    """One-step-ahead level forecasts on ``horizon``.

    Parameters are held fixed (fit on ``history`` only). The seasonally
    differenced extended series is filtered; predictions are inverted with
    ``ŷ_t = Δ̂_s y_t + y_{t-s}`` so each step uses the true lag-s level.
    """
    s = config.seasonal_period
    extended = pd.concat([history, horizon]).astype(np.float64)
    extended = extended[~extended.index.duplicated(keep="last")].sort_index()
    sd_ext = _seasonal_difference(extended, s)

    model = ARIMA(
        sd_ext,
        order=config.order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        filtered = model.filter(fit_result.params)

    # One-step predictions of the seasonally differenced process
    sd_pred = filtered.predict(start=horizon.index[0], end=horizon.index[-1])
    lag = extended.shift(s).loc[horizon.index]
    level_pred = sd_pred.to_numpy(dtype=np.float64) + lag.to_numpy(dtype=np.float64)
    return level_pred


def evaluate_config_on_val(
    train: pd.Series,
    val: pd.Series,
    config: SarimaConfig,
    *,
    maxiter: int = 100,
) -> GridResult:
    import traceback

    t0 = time.perf_counter()
    try:
        res = _fit_arima_on_seasonal_diff(train, config, maxiter=maxiter)
        y_pred = one_step_forecast(res, train, val, config)
        m = metrics(val.to_numpy(dtype=np.float64), y_pred)
        aic = float(res.aic) if np.isfinite(res.aic) else None
        return GridResult(
            config=config,
            val_mae=m["mae"],
            val_rmse=m["rmse"],
            fit_seconds=time.perf_counter() - t0,
            aic=aic,
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001 — record failed combos
        tb = traceback.format_exc()
        print(
            f"\n----- FULL TRACEBACK for {config.label()} -----\n"
            f"{tb}"
            f"----- end traceback -----\n",
            flush=True,
        )
        return GridResult(
            config=config,
            val_mae=float("inf"),
            val_rmse=float("inf"),
            fit_seconds=time.perf_counter() - t0,
            aic=None,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def iter_configs(
    orders: Sequence[Tuple[int, int, int]] = DEFAULT_ORDER_GRID,
    s: int = SEASONAL_PERIOD,
) -> List[SarimaConfig]:
    return [SarimaConfig(order=o, seasonal_period=s) for o in orders]


def grid_search(
    train: pd.Series,
    val: pd.Series,
    configs: Optional[Iterable[SarimaConfig]] = None,
    *,
    maxiter: int = 100,
) -> List[GridResult]:
    configs = list(configs) if configs is not None else iter_configs()
    results: List[GridResult] = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Fitting {cfg.label()} ...", flush=True)
        gr = evaluate_config_on_val(train, val, cfg, maxiter=maxiter)
        if gr.status == "ok":
            aic_s = f"{gr.aic:.1f}" if gr.aic is not None else "—"
            print(
                f"    val MAE={gr.val_mae:.4f}  RMSE={gr.val_rmse:.4f}  "
                f"AIC={aic_s}  ({gr.fit_seconds:.1f}s)",
                flush=True,
            )
        else:
            print(f"    FAILED ({gr.fit_seconds:.1f}s): {gr.error}", flush=True)
        results.append(gr)
    return results


def select_best(results: Sequence[GridResult]) -> GridResult:
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        raise RuntimeError("All SARIMA grid configurations failed")
    return min(ok, key=lambda r: r.val_mae)


def fit_and_forecast_test(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    config: SarimaConfig,
    *,
    maxiter: int = 200,
    refit_on: str = "train_val",
) -> SarimaForecastResult:
    if refit_on == "train_val":
        history = pd.concat([train, val]).sort_index()
    elif refit_on == "train":
        history = train
    else:
        raise ValueError("refit_on must be 'train_val' or 'train'")

    t0 = time.perf_counter()
    res = _fit_arima_on_seasonal_diff(history, config, maxiter=maxiter)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = one_step_forecast(res, history, test, config)
    infer_seconds = time.perf_counter() - t1

    y_true = test.to_numpy(dtype=np.float64)
    m = metrics(y_true, y_pred)

    return SarimaForecastResult(
        config=config,
        timestamps=test.index,
        y_true=y_true,
        y_pred=y_pred,
        mae=m["mae"],
        mape=m["mape"],
        rmse=m["rmse"],
        train_seconds=train_seconds,
        infer_seconds=infer_seconds,
        train_end=history.index[-1],
        test_start=test.index[0],
        test_end=test.index[-1],
    )


def append_experiments_log(
    results: Sequence[GridResult],
    best: GridResult,
    test_result: SarimaForecastResult,
    path: PathLike,
    *,
    square_id: int,
) -> None:
    from src.experiments_log import TimingStats, append_model_run

    notes = [
        f"- Implemented as **SARIMA(p,d,q)×(0,1,0,{SEASONAL_PERIOD})**: seasonal "
        f"difference at lag {SEASONAL_PERIOD} (daily), then ARIMA(p,d,q) on the "
        "differenced series.",
        "- **Why not full (P,D,Q,s) MLE?** With s=144 the Kalman state is "
        "O(s); full seasonal AR/MA MLE is impractical at 10-minute resolution. "
        "D=1 at s=144 still encodes the dominant daily period from EDA.",
        "- **Limitation — weekly seasonality:** ACF peak at lag 1008 is not "
        "modelled (single seasonal period only).",
        "- Validation: one-step-ahead MAE on val (9–15 Dec).",
        f"- Best by val MAE during grid search: {best.val_mae:.4f}.",
    ]
    grid_rows = []
    for r in results:
        grid_rows.append(
            {
                "config": r.config.label(),
                "val_mae": None if not np.isfinite(r.val_mae) else r.val_mae,
                "val_rmse": None if not np.isfinite(r.val_rmse) else r.val_rmse,
                "aic": r.aic,
                "train_s": r.fit_seconds,
                "status": r.status if r.status == "ok" else f"failed: {r.error}",
            }
        )

    append_model_run(
        model_name="SARIMA",
        square_id=square_id,
        best_config=best.config.label(),
        test_metrics={
            "mae": test_result.mae,
            "mape": test_result.mape,
            "rmse": test_result.rmse,
        },
        timing=TimingStats(
            train_seconds=test_result.train_seconds,
            infer_seconds=test_result.infer_seconds,
            notes="final refit on train∪val; inference = one-step filter over test week",
        ),
        grid_rows=grid_rows,
        notes=notes,
        path=path,
        replace_existing_section=True,
    )


def save_predictions(
    result: SarimaForecastResult,
    path: PathLike,
    *,
    square_id: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "datetime": result.timestamps,
            "y_true": result.y_true,
            "y_pred": result.y_pred,
            "model": "sarima",
            "config": result.config.label(),
            "square_id": square_id,
        }
    )
    df.to_parquet(path, index=False)
    meta = {
        "square_id": square_id,
        "model": "sarima",
        "config": result.config.label(),
        "order": list(result.config.order),
        "seasonal_order": list(result.config.seasonal_order),
        "mae": result.mae,
        "mape": result.mape,
        "rmse": result.rmse,
        "train_seconds": result.train_seconds,
        "infer_seconds": result.infer_seconds,
        "fit_seconds": result.train_seconds,  # back-compat
        "train_end": str(result.train_end),
        "test_start": str(result.test_start),
        "test_end": str(result.test_end),
        "implementation": (
            f"SARIMA(p,d,q)x(0,1,0,{SEASONAL_PERIOD}) via seasonal differencing "
            "+ ARIMA; one-step Kalman predictions inverted to levels."
        ),
        "limitation": (
            "Weekly lag-1008 seasonality not modelled; seasonal AR/MA (P,Q) "
            "omitted for tractability at s=144."
        ),
    }
    try:
        from src.experiments_log import collect_hardware_fingerprint

        meta["hardware"] = collect_hardware_fingerprint().to_dict()
    except Exception:
        pass
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def run_sarima_pipeline(
    square_id: int = 5161,
    *,
    prepared: Optional[PreparedSquare] = None,
    experiments_path: PathLike = _ROOT / "report" / "experiments.md",
    predictions_path: Optional[PathLike] = None,
    maxiter_grid: int = 100,
    maxiter_final: int = 200,
) -> SarimaForecastResult:
    if prepared is None:
        prepared = prepare_square(square_id=square_id, seq_len=144)
    if predictions_path is None:
        predictions_path = (
            _ROOT
            / "data"
            / "processed"
            / "predictions"
            / f"sarima_square_{square_id}.parquet"
        )

    train = prepared.series.train_raw
    val = prepared.series.val_raw
    test = prepared.series.test_raw

    print(
        f"SARIMA grid search on square {square_id} "
        f"(s={SEASONAL_PERIOD}, form (p,d,q)x(0,1,0,s))\n"
        f"  train n={len(train)}, val n={len(val)}, test n={len(test)}"
    )

    results = grid_search(train, val, maxiter=maxiter_grid)
    best = select_best(results)
    print(f"\nBest config: {best.config.label()}  val MAE={best.val_mae:.4f}")

    print("Refitting on train∪val and forecasting test week (one-step) ...")
    test_result = fit_and_forecast_test(
        train, val, test, best.config, maxiter=maxiter_final, refit_on="train_val"
    )
    print(
        f"Test MAE={test_result.mae:.4f}  MAPE={test_result.mape:.2f}%  "
        f"RMSE={test_result.rmse:.4f}\n"
        f"  train/fit={test_result.train_seconds:.3f}s  "
        f"one-step infer={test_result.infer_seconds:.3f}s"
    )

    append_experiments_log(
        results, best, test_result, experiments_path, square_id=square_id
    )
    save_predictions(test_result, predictions_path, square_id=square_id)
    print(f"Logged → {experiments_path}")
    print(f"Predictions → {predictions_path}")
    return test_result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="SARIMA grid search + Dec 16–22 test forecast"
    )
    parser.add_argument("--square", type=int, default=5161)
    args = parser.parse_args()
    run_sarima_pipeline(square_id=args.square)


if __name__ == "__main__":
    main()
