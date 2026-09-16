"""1D-CNN one-step-ahead forecasting on windowed Milan traffic sequences.

Same data contract as ``lstm.py``: windows from
``data/processed/sequences/square_<id>/`` (X shape ``(n, 144, 1)``,
normalised). Small grid over filters / kernel size / depth; early stopping
on validation MAE (raw scale). Logs via ``src.experiments_log.append_model_run``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.experiments_log import TimingStats, append_model_run  # noqa: E402
from src.models.lstm import (  # noqa: E402
    load_sequence_bundle,
    metrics,
    predict_numpy,
)
from src.prepare_sequences import ScalerParams  # noqa: E402

PathLike = Union[str, Path]

DEFAULT_SEQ_DIR = _ROOT / "data" / "processed" / "sequences"
DEFAULT_PRED_DIR = _ROOT / "data" / "processed" / "predictions"
DEFAULT_EXPERIMENTS = _ROOT / "report" / "experiments.md"

# Small grid: filters × kernel × depth = 2 × 2 × 2 = 8
DEFAULT_GRID: Tuple[Dict, ...] = (
    {"num_filters": 32, "kernel_size": 3, "num_layers": 1},
    {"num_filters": 32, "kernel_size": 5, "num_layers": 1},
    {"num_filters": 64, "kernel_size": 3, "num_layers": 1},
    {"num_filters": 64, "kernel_size": 5, "num_layers": 1},
    {"num_filters": 32, "kernel_size": 3, "num_layers": 2},
    {"num_filters": 32, "kernel_size": 5, "num_layers": 2},
    {"num_filters": 64, "kernel_size": 3, "num_layers": 2},
    {"num_filters": 64, "kernel_size": 5, "num_layers": 2},
)


@dataclass(frozen=True)
class CnnConfig:
    num_filters: int = 64
    kernel_size: int = 3
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 6

    def label(self) -> str:
        return (
            f"CNN1D(filters={self.num_filters}, kernel={self.kernel_size}, "
            f"layers={self.num_layers}, dropout={self.dropout})"
        )


@dataclass
class GridResult:
    config: CnnConfig
    val_mae: float
    val_rmse: float
    train_seconds: float
    best_epoch: int
    status: str
    error: Optional[str] = None
    state_dict: Optional[dict] = None


@dataclass
class CnnForecastResult:
    config: CnnConfig
    timestamps: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    mae: float
    mape: float
    rmse: float
    train_seconds: float
    infer_seconds: float


class TrafficCNN1D(nn.Module):
    """Causal-ish 1D CNN over the time axis (channels-first after permute)."""

    def __init__(
        self,
        *,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 144,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd for symmetric padding")
        pad = kernel_size // 2
        layers: List[nn.Module] = []
        in_ch = 1
        for i in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(in_ch, num_filters, kernel_size=kernel_size, padding=pad),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = num_filters
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(num_filters * seq_len, 1)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, 1) → (batch, 1, seq)
        x = x.transpose(1, 2)
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h).squeeze(-1)


def _device() -> torch.device:
    return torch.device("cpu")


def _loaders(
    bundle: dict,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    def make(X, y, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return (
        make(bundle["X_train"], bundle["y_train"], True),
        make(bundle["X_val"], bundle["y_val"], False),
    )


def train_one(
    config: CnnConfig,
    bundle: dict,
    *,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> GridResult:
    device = device or _device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_len = int(bundle["X_train"].shape[1])
    model = TrafficCNN1D(
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        seq_len=seq_len,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    train_loader, val_loader = _loaders(bundle, config.batch_size)
    scaler: ScalerParams = bundle["scaler"]

    best_val_mae = float("inf")
    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = config.patience

    t0 = time.perf_counter()
    try:
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            val_pred_norm = predict_numpy(
                model, bundle["X_val"], batch_size=config.batch_size, device=device
            )
            y_true = scaler.inverse_transform(bundle["y_val"])
            y_pred = scaler.inverse_transform(val_pred_norm)
            m = metrics(y_true, y_pred)

            if m["mae"] < best_val_mae - 1e-6:
                best_val_mae = m["mae"]
                best_val_rmse = m["rmse"]
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_left = config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        train_seconds = time.perf_counter() - t0
        if best_state is None:
            raise RuntimeError("Training produced no usable checkpoint")

        return GridResult(
            config=config,
            val_mae=best_val_mae,
            val_rmse=best_val_rmse,
            train_seconds=train_seconds,
            best_epoch=best_epoch,
            status="ok",
            state_dict=best_state,
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(
            f"\n----- FULL TRACEBACK for {config.label()} -----\n"
            f"{traceback.format_exc()}"
            f"----- end traceback -----\n",
            flush=True,
        )
        return GridResult(
            config=config,
            val_mae=float("inf"),
            val_rmse=float("inf"),
            train_seconds=time.perf_counter() - t0,
            best_epoch=0,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def iter_configs(grid: Sequence[Dict] = DEFAULT_GRID) -> List[CnnConfig]:
    return [CnnConfig(**params) for params in grid]


def grid_search(
    bundle: dict,
    configs: Optional[Iterable[CnnConfig]] = None,
) -> List[GridResult]:
    configs = list(configs) if configs is not None else iter_configs()
    results: List[GridResult] = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Training {cfg.label()} ...", flush=True)
        gr = train_one(cfg, bundle)
        if gr.status == "ok":
            print(
                f"    val MAE={gr.val_mae:.4f}  RMSE={gr.val_rmse:.4f}  "
                f"best_epoch={gr.best_epoch}  train={gr.train_seconds:.1f}s",
                flush=True,
            )
        else:
            print(f"    FAILED ({gr.train_seconds:.1f}s): {gr.error}", flush=True)
        results.append(gr)
    return results


def select_best(results: Sequence[GridResult]) -> GridResult:
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        raise RuntimeError("All CNN grid configurations failed")
    return min(ok, key=lambda r: r.val_mae)


def evaluate_best_on_test(
    best: GridResult,
    bundle: dict,
) -> CnnForecastResult:
    device = _device()
    cfg = best.config
    seq_len = int(bundle["X_train"].shape[1])
    model = TrafficCNN1D(
        num_filters=cfg.num_filters,
        kernel_size=cfg.kernel_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        seq_len=seq_len,
    ).to(device)
    assert best.state_dict is not None
    model.load_state_dict(best.state_dict)
    model.eval()

    t0 = time.perf_counter()
    y_pred_norm = predict_numpy(
        model, bundle["X_test"], batch_size=cfg.batch_size, device=device
    )
    infer_seconds = time.perf_counter() - t0

    scaler: ScalerParams = bundle["scaler"]
    y_true = scaler.inverse_transform(bundle["y_test"])
    y_pred = scaler.inverse_transform(y_pred_norm)
    m = metrics(y_true, y_pred)

    ts = bundle["timestamps_test"]
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert("Europe/Rome")
    else:
        ts = ts.tz_localize("UTC").tz_convert("Europe/Rome")

    return CnnForecastResult(
        config=cfg,
        timestamps=pd.DatetimeIndex(ts),
        y_true=y_true,
        y_pred=y_pred,
        mae=m["mae"],
        mape=m["mape"],
        rmse=m["rmse"],
        train_seconds=best.train_seconds,
        infer_seconds=infer_seconds,
    )


def log_cnn_run(
    results: Sequence[GridResult],
    best: GridResult,
    test_result: CnnForecastResult,
    *,
    square_id: int,
    path: PathLike = DEFAULT_EXPERIMENTS,
) -> None:
    notes = [
        "- Input: same windowed sequences as LSTM (length 144, train-only "
        "standardisation).",
        "- Architecture: stacked Conv1d + ReLU + Dropout, then linear head over "
        "flattened feature map; Adam; MSE on normalised targets; early stopping "
        "on validation MAE (raw scale).",
        f"- Early-stopping patience={best.config.patience}, "
        f"max_epochs={best.config.max_epochs}, batch_size={best.config.batch_size}.",
        f"- Best checkpoint at epoch {best.best_epoch} "
        f"(val MAE={best.val_mae:.4f}).",
        "- Receptive field is local (kernel×depth); weekly lag-1008 is not "
        "explicitly encoded.",
    ]
    grid_rows = []
    for r in results:
        grid_rows.append(
            {
                "config": r.config.label(),
                "val_mae": None if not np.isfinite(r.val_mae) else r.val_mae,
                "val_rmse": None if not np.isfinite(r.val_rmse) else r.val_rmse,
                "train_s": r.train_seconds,
                "best_epoch": r.best_epoch if r.status == "ok" else None,
                "status": r.status if r.status == "ok" else f"failed: {r.error}",
            }
        )

    append_model_run(
        model_name="1D-CNN",
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
            notes="train = full fit with early stopping for best config; "
            "infer = one forward pass over Dec 16–22 test windows",
        ),
        grid_rows=grid_rows,
        notes=notes,
        path=path,
        replace_existing_section=True,
    )


def save_predictions(
    result: CnnForecastResult,
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
            "model": "cnn1d",
            "config": result.config.label(),
            "square_id": square_id,
        }
    )
    df.to_parquet(path, index=False)
    meta = {
        "square_id": square_id,
        "model": "cnn1d",
        "config": result.config.label(),
        "num_filters": result.config.num_filters,
        "kernel_size": result.config.kernel_size,
        "num_layers": result.config.num_layers,
        "dropout": result.config.dropout,
        "mae": result.mae,
        "mape": result.mape,
        "rmse": result.rmse,
        "train_seconds": result.train_seconds,
        "infer_seconds": result.infer_seconds,
    }
    try:
        from src.experiments_log import collect_hardware_fingerprint

        meta["hardware"] = collect_hardware_fingerprint().to_dict()
    except Exception:
        pass
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def save_checkpoint(best: GridResult, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "num_filters": best.config.num_filters,
                "kernel_size": best.config.kernel_size,
                "num_layers": best.config.num_layers,
                "dropout": best.config.dropout,
                "lr": best.config.lr,
                "batch_size": best.config.batch_size,
            },
            "state_dict": best.state_dict,
            "val_mae": best.val_mae,
            "best_epoch": best.best_epoch,
        },
        path,
    )
    return path


def run_cnn_pipeline(
    square_id: int = 5161,
    *,
    seq_dir: PathLike = DEFAULT_SEQ_DIR,
    experiments_path: PathLike = DEFAULT_EXPERIMENTS,
    predictions_path: Optional[PathLike] = None,
) -> CnnForecastResult:
    if predictions_path is None:
        predictions_path = DEFAULT_PRED_DIR / f"cnn_square_{square_id}.parquet"

    bundle = load_sequence_bundle(square_id, seq_dir)
    print(
        f"1D-CNN grid on square {square_id} | "
        f"X_train {bundle['X_train'].shape}, X_val {bundle['X_val'].shape}, "
        f"X_test {bundle['X_test'].shape}"
    )

    results = grid_search(bundle)
    best = select_best(results)
    print(
        f"\nBest: {best.config.label()}  val MAE={best.val_mae:.4f}  "
        f"(epoch {best.best_epoch}, train {best.train_seconds:.1f}s)"
    )

    test_result = evaluate_best_on_test(best, bundle)
    print(
        f"Test MAE={test_result.mae:.4f}  MAPE={test_result.mape:.2f}%  "
        f"RMSE={test_result.rmse:.4f}\n"
        f"  train/fit={test_result.train_seconds:.3f}s  "
        f"one-step infer={test_result.infer_seconds:.3f}s"
    )

    log_cnn_run(
        results, best, test_result, square_id=square_id, path=experiments_path
    )
    save_predictions(test_result, predictions_path, square_id=square_id)
    ckpt = DEFAULT_PRED_DIR / f"cnn_square_{square_id}.pt"
    save_checkpoint(best, ckpt)
    print(f"Logged → {experiments_path}")
    print(f"Predictions → {predictions_path}")
    print(f"Checkpoint → {ckpt}")
    return test_result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="1D-CNN grid search + test forecast")
    parser.add_argument("--square", type=int, default=5161)
    parser.add_argument("--quick", action="store_true", help="Single-config smoke test")
    args = parser.parse_args()

    if args.quick:
        bundle = load_sequence_bundle(args.square)
        cfg = CnnConfig(
            num_filters=32,
            kernel_size=3,
            num_layers=1,
            max_epochs=5,
            patience=3,
        )
        results = grid_search(bundle, configs=[cfg])
        best = select_best(results)
        test_result = evaluate_best_on_test(best, bundle)
        log_cnn_run(results, best, test_result, square_id=args.square)
        save_predictions(
            test_result,
            DEFAULT_PRED_DIR / f"cnn_square_{args.square}.parquet",
            square_id=args.square,
        )
        print(
            f"QUICK Test MAE={test_result.mae:.4f}  "
            f"train={test_result.train_seconds:.2f}s "
            f"infer={test_result.infer_seconds:.3f}s"
        )
    else:
        run_cnn_pipeline(square_id=args.square)


if __name__ == "__main__":
    main()
