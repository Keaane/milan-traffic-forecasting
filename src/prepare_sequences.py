"""Train/val/test splits and sequence windows for one-step-ahead forecasting.

Designed for two consumption patterns:

* **SARIMA** — chronological raw (and optionally normalised) Series splits,
  with no windowing.
* **LSTM / 1D-CNN** — sliding windows of length ``seq_len`` producing
  ``(X, y)`` tensors, where ``X`` is the history up to time ``t`` and ``y``
  is the one-step-ahead target ``x(t+1)``.

Normalisation statistics (mean / std) are fit on the **training raw series
only** and then applied to val/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PathLike = Union[str, Path]

DEFAULT_PARQUET = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "internet_traffic.parquet"
)
TZ = "Europe/Rome"
FREQ = "10min"

# Assignment eval week: 16–22 Dec 2013 inclusive
DEFAULT_TEST_START = "2013-12-16"
DEFAULT_TEST_END = "2013-12-23"  # exclusive → through end of Dec 22

# Validation: final week before the test hold-out (9–15 Dec)
DEFAULT_VAL_START = "2013-12-09"


@dataclass(frozen=True)
class ScalerParams:
    """Standard-score scaler fit on training data only."""

    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def transform_series(self, s: pd.Series) -> pd.Series:
        out = self.transform(s.to_numpy(dtype=np.float64))
        return pd.Series(out, index=s.index, name=s.name)

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "ScalerParams":
        return cls(mean=float(d["mean"]), std=float(d["std"]))


@dataclass
class SeriesBundle:
    """Chronological series splits for SARIMA / inspection.

    ``*_raw`` are in the original internet-traffic scale.
    ``*_norm`` use the training-only scaler.
    """

    square_id: int
    full_raw: pd.Series
    train_raw: pd.Series
    val_raw: pd.Series
    test_raw: pd.Series
    full_norm: pd.Series
    train_norm: pd.Series
    val_norm: pd.Series
    test_norm: pd.Series
    scaler: ScalerParams
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    val_start: pd.Timestamp


@dataclass
class WindowBundle:
    """Sliding-window tensors for LSTM / 1D-CNN (normalised scale).

    Shapes
    ------
    X_* : (n_samples, seq_len, 1)
    y_* : (n_samples,)
    timestamps_* : DatetimeIndex of each target ``y`` (the ``t+1`` instant)
    """

    seq_len: int
    X_train: np.ndarray
    y_train: np.ndarray
    timestamps_train: pd.DatetimeIndex
    X_val: np.ndarray
    y_val: np.ndarray
    timestamps_val: pd.DatetimeIndex
    X_test: np.ndarray
    y_test: np.ndarray
    timestamps_test: pd.DatetimeIndex
    scaler: ScalerParams

    def shapes(self) -> dict:
        return {
            "X_train": self.X_train.shape,
            "y_train": self.y_train.shape,
            "X_val": self.X_val.shape,
            "y_val": self.y_val.shape,
            "X_test": self.X_test.shape,
            "y_test": self.y_test.shape,
        }


@dataclass
class PreparedSquare:
    """Full prep artifact: series (SARIMA) + windows (LSTM/CNN)."""

    series: SeriesBundle
    windows: WindowBundle

    @property
    def square_id(self) -> int:
        return self.series.square_id

    @property
    def scaler(self) -> ScalerParams:
        return self.series.scaler

    @property
    def seq_len(self) -> int:
        return self.windows.seq_len


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _ts(label: str) -> pd.Timestamp:
    return pd.Timestamp(label, tz=TZ)


def load_square_series(
    square_id: int = 5161,
    parquet_path: PathLike = DEFAULT_PARQUET,
) -> pd.Series:
    """Load one square's Internet traffic as a regular 10-min Series."""
    parquet_path = _as_path(parquet_path)
    table = pq.read_table(
        parquet_path,
        filters=[("square_id", "==", int(square_id))],
        columns=["timestamp", "internet"],
    )
    df = table.to_pandas()
    if df.empty:
        raise ValueError(f"No rows for square_id={square_id} in {parquet_path}")

    dt = (
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        .dt.tz_convert(TZ)
    )
    series = (
        pd.Series(df["internet"].to_numpy(dtype=np.float64), index=dt, name="internet")
        .sort_index()
        .groupby(level=0)
        .sum()  # safety if duplicates ever appear
        .asfreq(FREQ)
    )
    n_missing = int(series.isna().sum())
    if n_missing:
        series = series.interpolate(limit_direction="both")
    return series


def fit_scaler(train_raw: pd.Series, eps: float = 1e-8) -> ScalerParams:
    """Fit mean/std on training values only."""
    values = train_raw.to_numpy(dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < eps:
        std = 1.0
    return ScalerParams(mean=mean, std=std)


def chronological_split(
    series: pd.Series,
    *,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
    val_start: str = DEFAULT_VAL_START,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Split into train / val / test by calendar boundaries (Europe/Rome).

    Default layout
    --------------
    * **train** — start → 9 Dec 2013 00:00 (exclusive of val)
    * **val**   — 9 Dec → 16 Dec 2013 (week before the assignment hold-out)
    * **test**  — 16 Dec → 23 Dec 2013 (16–22 Dec inclusive)
    """
    t_val = _ts(val_start)
    t_test = _ts(test_start)
    t_test_end = _ts(test_end)

    if not (t_val < t_test < t_test_end):
        raise ValueError("Require val_start < test_start < test_end")

    train = series[series.index < t_val].copy()
    val = series[(series.index >= t_val) & (series.index < t_test)].copy()
    test = series[(series.index >= t_test) & (series.index < t_test_end)].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty split: train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test, t_val, t_test, t_test_end


def make_xy_windows(
    full_norm: pd.Series,
    target_index: pd.DatetimeIndex,
    seq_len: int = 144,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build one-step-ahead windows whose **targets** fall in ``target_index``.

    For each target time ``t_y`` in ``target_index``::

        X = full_norm[t_y - seq_len : t_y]   # length seq_len, ends at t_y - 10min
        y = full_norm[t_y]

    History may extend into an earlier split (e.g. test targets may use
    train/val history). Targets with insufficient history are skipped.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")

    values = full_norm.to_numpy(dtype=np.float64)
    index = full_norm.index
    pos = {ts: i for i, ts in enumerate(index)}

    Xs: list[np.ndarray] = []
    ys: list[float] = []
    ts_out: list[pd.Timestamp] = []

    for t_y in target_index:
        j = pos.get(t_y)
        if j is None or j < seq_len:
            continue
        # history is indices [j - seq_len, ..., j - 1]; target at j
        Xs.append(values[j - seq_len : j])
        ys.append(values[j])
        ts_out.append(t_y)

    if not Xs:
        raise ValueError("No valid windows for the requested target index")

    X = np.stack(Xs, axis=0).astype(np.float32)[..., np.newaxis]  # (n, L, 1)
    y = np.asarray(ys, dtype=np.float32)
    return X, y, pd.DatetimeIndex(ts_out, tz=TZ)


def prepare_square(
    square_id: int = 5161,
    *,
    seq_len: int = 144,
    parquet_path: PathLike = DEFAULT_PARQUET,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
    val_start: str = DEFAULT_VAL_START,
) -> PreparedSquare:
    """End-to-end prep for one square: series splits + neural windows."""
    full_raw = load_square_series(square_id, parquet_path)
    train_raw, val_raw, test_raw, t_val, t_test, t_test_end = chronological_split(
        full_raw,
        test_start=test_start,
        test_end=test_end,
        val_start=val_start,
    )
    scaler = fit_scaler(train_raw)

    full_norm = scaler.transform_series(full_raw)
    train_norm = scaler.transform_series(train_raw)
    val_norm = scaler.transform_series(val_raw)
    test_norm = scaler.transform_series(test_raw)

    series = SeriesBundle(
        square_id=square_id,
        full_raw=full_raw,
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw=test_raw,
        full_norm=full_norm,
        train_norm=train_norm,
        val_norm=val_norm,
        test_norm=test_norm,
        scaler=scaler,
        test_start=t_test,
        test_end=t_test_end,
        val_start=t_val,
    )

    # Targets strictly inside each split; history may cross earlier splits.
    X_tr, y_tr, ts_tr = make_xy_windows(full_norm, train_raw.index, seq_len)
    X_va, y_va, ts_va = make_xy_windows(full_norm, val_raw.index, seq_len)
    X_te, y_te, ts_te = make_xy_windows(full_norm, test_raw.index, seq_len)

    windows = WindowBundle(
        seq_len=seq_len,
        X_train=X_tr,
        y_train=y_tr,
        timestamps_train=ts_tr,
        X_val=X_va,
        y_val=y_va,
        timestamps_val=ts_va,
        X_test=X_te,
        y_test=y_te,
        timestamps_test=ts_te,
        scaler=scaler,
    )
    return PreparedSquare(series=series, windows=windows)


def save_prepared(
    prepared: PreparedSquare,
    out_dir: PathLike,
) -> Path:
    """Persist series splits, scaler, and window arrays under ``out_dir``."""
    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = prepared.square_id

    # Series (parquet-friendly DataFrames)
    for name, s in {
        "full_raw": prepared.series.full_raw,
        "train_raw": prepared.series.train_raw,
        "val_raw": prepared.series.val_raw,
        "test_raw": prepared.series.test_raw,
        "full_norm": prepared.series.full_norm,
        "train_norm": prepared.series.train_norm,
        "val_norm": prepared.series.val_norm,
        "test_norm": prepared.series.test_norm,
    }.items():
        s.rename("internet").to_frame().to_parquet(out_dir / f"{name}.parquet")

    meta = {
        "square_id": sid,
        "seq_len": prepared.seq_len,
        "scaler": prepared.scaler.to_dict(),
        "val_start": str(prepared.series.val_start),
        "test_start": str(prepared.series.test_start),
        "test_end": str(prepared.series.test_end),
        "n_train_raw": len(prepared.series.train_raw),
        "n_val_raw": len(prepared.series.val_raw),
        "n_test_raw": len(prepared.series.test_raw),
        "window_shapes": prepared.windows.shapes(),
    }
    pd.Series(meta).to_json(out_dir / "meta.json")

    np.savez_compressed(
        out_dir / "windows.npz",
        X_train=prepared.windows.X_train,
        y_train=prepared.windows.y_train,
        X_val=prepared.windows.X_val,
        y_val=prepared.windows.y_val,
        X_test=prepared.windows.X_test,
        y_test=prepared.windows.y_test,
        timestamps_train=prepared.windows.timestamps_train.asi8,
        timestamps_val=prepared.windows.timestamps_val.asi8,
        timestamps_test=prepared.windows.timestamps_test.asi8,
    )
    return out_dir


def load_windows_npz(windows: WindowBundle, split: str = "train"):
    """Convenience: return ``(X, y)`` for a named split."""
    mapping = {
        "train": (windows.X_train, windows.y_train),
        "val": (windows.X_val, windows.y_val),
        "test": (windows.X_test, windows.y_test),
    }
    if split not in mapping:
        raise KeyError(f"split must be one of {list(mapping)}")
    return mapping[split]


def summarize(prepared: PreparedSquare) -> str:
    s = prepared.series
    w = prepared.windows
    lines = [
        f"Square {prepared.square_id} | seq_len={prepared.seq_len}",
        f"  train raw: {s.train_raw.index.min()} → {s.train_raw.index.max()}  (n={len(s.train_raw)})",
        f"  val   raw: {s.val_raw.index.min()} → {s.val_raw.index.max()}  (n={len(s.val_raw)})",
        f"  test  raw: {s.test_raw.index.min()} → {s.test_raw.index.max()}  (n={len(s.test_raw)})",
        f"  scaler (train-only): mean={s.scaler.mean:.4f}, std={s.scaler.std:.4f}",
        f"  windows X_train {w.X_train.shape}, X_val {w.X_val.shape}, X_test {w.X_test.shape}",
        f"  y scale: norm  |  inverse via prepared.scaler.inverse_transform",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare sequences for square forecasting")
    parser.add_argument("--square", type=int, default=5161)
    parser.add_argument("--seq-len", type=int, default=144)
    parser.add_argument(
        "--parquet",
        default=str(DEFAULT_PARQUET),
        help="Path to internet_traffic.parquet",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: data/processed/sequences/square_<id>)",
    )
    args = parser.parse_args()

    prepared = prepare_square(
        square_id=args.square,
        seq_len=args.seq_len,
        parquet_path=args.parquet,
    )
    print(summarize(prepared))

    out = (
        _as_path(args.out)
        if args.out
        else DEFAULT_PARQUET.parent / "sequences" / f"square_{args.square}"
    )
    save_prepared(prepared, out)
    print(f"\nSaved under {out}")


if __name__ == "__main__":
    main()
