"""Refit best-of-5161 configs on additional squares; plots + comparison tables.

Best hyperparameters (from square 5161 grid search):
  SARIMA(1,0,1)×(0,1,0,144)
  LSTM(layers=2, hidden=64, dropout=0.0)
  CNN1D(filters=32, kernel=5, layers=1, dropout=0.1)

Two square sets are evaluated:

* ``PRIMARY_SQUARES`` — the three areas with the highest total Internet
  traffic (Task 2, item 2). These are the "three geographical areas" for
  which the brief requires 9 overlay plots and 3 performance tables.
* ``EXTENSION_SQUARES`` — squares 4159 and 4556, named individually in the
  brief's exploratory task. Carried through the same pipeline so the
  comparison also spans a lower-traffic regime.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from functools import lru_cache
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.experiments_log import TimingStats, append_model_run
from src.models.baselines import (
    persistence,
    save_predictions as baseline_save_preds,
    seasonal_naive,
)
from src.models.cnn import (
    CnnConfig,
    evaluate_best_on_test as cnn_eval,
    save_checkpoint as cnn_save_ckpt,
    save_predictions as cnn_save_preds,
    train_one as cnn_train,
)
from src.models.lstm import (
    LstmConfig,
    evaluate_best_on_test as lstm_eval,
    load_sequence_bundle,
    metrics,
    save_checkpoint as lstm_save_ckpt,
    save_predictions as lstm_save_preds,
    train_one as lstm_train,
)
from src.models.sarima import (
    SarimaConfig,
    fit_and_forecast_test,
    one_step_forecast,
    save_predictions as sarima_save_preds,
    _fit_arima_on_seasonal_diff,
)
from src.prepare_sequences import prepare_square, save_prepared

# The three highest-total-traffic areas identified in the exploratory analysis.
PRIMARY_SQUARES = [5161, 5059, 5259]
# Individually named in the brief's EDA task; kept as a lower-traffic contrast.
EXTENSION_SQUARES = [4159, 4556]
SQUARES = PRIMARY_SQUARES + EXTENSION_SQUARES
# 5161 is grid-searched separately; every other square is a fixed-config refit.
EXTRA_SQUARES = [s for s in SQUARES if s != 5161]

BEST_SARIMA = SarimaConfig(order=(1, 0, 1), seasonal_period=144)
BEST_LSTM = LstmConfig(num_layers=2, hidden_size=64, dropout=0.0)
BEST_CNN = CnnConfig(num_filters=32, kernel_size=5, num_layers=1, dropout=0.1)

PRED_DIR = _ROOT / "data" / "processed" / "predictions"
FIG_DIR = _ROOT / "report" / "figures"
RESULTS_MD = _ROOT / "report" / "results.md"
EXPERIMENTS_MD = _ROOT / "report" / "experiments.md"

MODEL_COLORS = {
    "sarima": "#1f4e79",
    "lstm": "#c45c26",
    "cnn1d": "#6b8f71",
}
MODEL_LABELS = {
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal naive",
    "sarima": "SARIMA",
    "lstm": "LSTM",
    "cnn1d": "1D-CNN",
}
# Reference forecasters reported alongside the three selected models.
BASELINE_KEYS = ["persistence", "seasonal_naive"]
MODEL_KEYS = ["sarima", "lstm", "cnn1d"]


@lru_cache(maxsize=None)
def _prepared(square_id: int):
    """Prepare one square once per process.

    ``prepare_square`` scans the 375 MB tidy parquet, so the SARIMA run, the
    baselines and the window export would otherwise pay for it three times
    per square.
    """
    return prepare_square(square_id=square_id, seq_len=144)


def _ensure_sequences(square_id: int) -> None:
    out = _ROOT / "data" / "processed" / "sequences" / f"square_{square_id}"
    if not (out / "windows.npz").exists():
        save_prepared(_prepared(square_id), out)
        print(f"Prepared sequences for square {square_id} -> {out}")


def run_baselines(square_id: int) -> List[dict]:
    """Persistence and seasonal-naive references for one square.

    No parameters are estimated, so these are computed for every square
    (including 5161) rather than refit from a grid search.
    """
    prepared = _prepared(square_id)
    history = pd.concat(
        [prepared.series.train_raw, prepared.series.val_raw]
    ).sort_index()
    test = prepared.series.test_raw

    rows: List[dict] = []
    for fn in (persistence, seasonal_naive):
        result = fn(history, test)
        path = PRED_DIR / f"{result.name}_square_{square_id}.parquet"
        baseline_save_preds(result, path, square_id=square_id)
        print(
            f"  {result.label} square {square_id}: test MAE={result.mae:.4f}  "
            f"MAPE={result.mape:.2f}%  RMSE={result.rmse:.4f}"
        )
        rows.append(
            {
                "model": result.name,
                "square_id": square_id,
                "mae": result.mae,
                "mape": result.mape,
                "rmse": result.rmse,
                "train_seconds": result.train_seconds,
                "infer_seconds": result.infer_seconds,
                "config": result.label,
                "path": str(path),
            }
        )
    return rows


def run_sarima_fixed(square_id: int) -> dict:
    prepared = _prepared(square_id)
    train, val, test = (
        prepared.series.train_raw,
        prepared.series.val_raw,
        prepared.series.test_raw,
    )

    # Val metrics with params fit on train only
    t0 = time.perf_counter()
    res_train = _fit_arima_on_seasonal_diff(train, BEST_SARIMA)
    train_for_val_s = time.perf_counter() - t0
    val_pred = one_step_forecast(res_train, train, val, BEST_SARIMA)
    val_m = metrics(val.to_numpy(dtype=np.float64), val_pred)

    # Final: refit train∪val, forecast test
    result = fit_and_forecast_test(
        train, val, test, BEST_SARIMA, refit_on="train_val"
    )
    # Replace train_seconds with final refit time already in result;
    # keep val info for logging.
    path = PRED_DIR / f"sarima_square_{square_id}.parquet"
    sarima_save_preds(result, path, square_id=square_id)

    append_model_run(
        model_name="SARIMA",
        square_id=square_id,
        best_config=BEST_SARIMA.label() + " [fixed from 5161]",
        test_metrics={"mae": result.mae, "mape": result.mape, "rmse": result.rmse},
        timing=TimingStats(
            train_seconds=result.train_seconds,
            infer_seconds=result.infer_seconds,
            notes="refit winning 5161 config on this square (no new grid search)",
        ),
        grid_rows=[
            {
                "config": BEST_SARIMA.label(),
                "val_mae": val_m["mae"],
                "val_rmse": val_m["rmse"],
                "train_s": train_for_val_s,
                "status": "ok (fixed config)",
            }
        ],
        notes=[
            f"- Refit of square-5161 winning config on square {square_id} "
            "(no per-square grid search).",
            f"- Val MAE (train-only fit) = {val_m['mae']:.4f}.",
        ],
        path=EXPERIMENTS_MD,
        replace_existing_section=True,
    )
    print(
        f"  SARIMA square {square_id}: test MAE={result.mae:.4f}  "
        f"MAPE={result.mape:.2f}%  RMSE={result.rmse:.4f}  "
        f"train={result.train_seconds:.3f}s infer={result.infer_seconds:.3f}s"
    )
    return {
        "model": "sarima",
        "square_id": square_id,
        "mae": result.mae,
        "mape": result.mape,
        "rmse": result.rmse,
        "train_seconds": result.train_seconds,
        "infer_seconds": result.infer_seconds,
        "config": BEST_SARIMA.label(),
        "path": str(path),
    }


def run_lstm_fixed(square_id: int) -> dict:
    bundle = load_sequence_bundle(square_id)
    gr = lstm_train(BEST_LSTM, bundle)
    if gr.status != "ok":
        raise RuntimeError(f"LSTM failed on {square_id}: {gr.error}")
    result = lstm_eval(gr, bundle)
    path = PRED_DIR / f"lstm_square_{square_id}.parquet"
    lstm_save_preds(result, path, square_id=square_id)
    lstm_save_ckpt(gr, PRED_DIR / f"lstm_square_{square_id}.pt")

    append_model_run(
        model_name="LSTM",
        square_id=square_id,
        best_config=BEST_LSTM.label() + " [fixed from 5161]",
        test_metrics={"mae": result.mae, "mape": result.mape, "rmse": result.rmse},
        timing=TimingStats(
            train_seconds=result.train_seconds,
            infer_seconds=result.infer_seconds,
            notes="refit winning 5161 config on this square (no new grid search)",
        ),
        grid_rows=[
            {
                "config": BEST_LSTM.label(),
                "val_mae": gr.val_mae,
                "val_rmse": gr.val_rmse,
                "train_s": gr.train_seconds,
                "best_epoch": gr.best_epoch,
                "status": "ok (fixed config)",
            }
        ],
        notes=[
            f"- Refit of square-5161 winning config on square {square_id} "
            "(no per-square grid search).",
            f"- Best checkpoint epoch={gr.best_epoch}, val MAE={gr.val_mae:.4f}.",
        ],
        path=EXPERIMENTS_MD,
        replace_existing_section=True,
    )
    print(
        f"  LSTM square {square_id}: test MAE={result.mae:.4f}  "
        f"MAPE={result.mape:.2f}%  RMSE={result.rmse:.4f}  "
        f"train={result.train_seconds:.3f}s infer={result.infer_seconds:.3f}s"
    )
    return {
        "model": "lstm",
        "square_id": square_id,
        "mae": result.mae,
        "mape": result.mape,
        "rmse": result.rmse,
        "train_seconds": result.train_seconds,
        "infer_seconds": result.infer_seconds,
        "config": BEST_LSTM.label(),
        "path": str(path),
    }


def run_cnn_fixed(square_id: int) -> dict:
    bundle = load_sequence_bundle(square_id)
    gr = cnn_train(BEST_CNN, bundle)
    if gr.status != "ok":
        raise RuntimeError(f"CNN failed on {square_id}: {gr.error}")
    result = cnn_eval(gr, bundle)
    path = PRED_DIR / f"cnn_square_{square_id}.parquet"
    cnn_save_preds(result, path, square_id=square_id)
    cnn_save_ckpt(gr, PRED_DIR / f"cnn_square_{square_id}.pt")

    append_model_run(
        model_name="1D-CNN",
        square_id=square_id,
        best_config=BEST_CNN.label() + " [fixed from 5161]",
        test_metrics={"mae": result.mae, "mape": result.mape, "rmse": result.rmse},
        timing=TimingStats(
            train_seconds=result.train_seconds,
            infer_seconds=result.infer_seconds,
            notes="refit winning 5161 config on this square (no new grid search)",
        ),
        grid_rows=[
            {
                "config": BEST_CNN.label(),
                "val_mae": gr.val_mae,
                "val_rmse": gr.val_rmse,
                "train_s": gr.train_seconds,
                "best_epoch": gr.best_epoch,
                "status": "ok (fixed config)",
            }
        ],
        notes=[
            f"- Refit of square-5161 winning config on square {square_id} "
            "(no per-square grid search).",
            f"- Best checkpoint epoch={gr.best_epoch}, val MAE={gr.val_mae:.4f}.",
        ],
        path=EXPERIMENTS_MD,
        replace_existing_section=True,
    )
    print(
        f"  CNN square {square_id}: test MAE={result.mae:.4f}  "
        f"MAPE={result.mape:.2f}%  RMSE={result.rmse:.4f}  "
        f"train={result.train_seconds:.3f}s infer={result.infer_seconds:.3f}s"
    )
    return {
        "model": "cnn1d",
        "square_id": square_id,
        "mae": result.mae,
        "mape": result.mape,
        "rmse": result.rmse,
        "train_seconds": result.train_seconds,
        "infer_seconds": result.infer_seconds,
        "config": BEST_CNN.label(),
        "path": str(path),
    }


def load_metrics_from_json(square_id: int, model_key: str) -> dict:
    path = PRED_DIR / f"{model_key}_square_{square_id}.json"
    meta = json.loads(path.read_text())
    return {
        "model": model_key if model_key != "cnn" else "cnn1d",
        "square_id": square_id,
        "mae": meta["mae"],
        "mape": meta["mape"],
        "rmse": meta["rmse"],
        "train_seconds": meta.get("train_seconds", meta.get("fit_seconds")),
        "infer_seconds": meta.get("infer_seconds"),
        "config": meta.get("config"),
        "path": str(PRED_DIR / f"{model_key}_square_{square_id}.parquet"),
    }


def plot_overlays() -> List[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "timezone": "Europe/Rome",
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    saved: List[Path] = []
    file_keys = [("sarima", "sarima"), ("lstm", "lstm"), ("cnn1d", "cnn")]

    for square_id in SQUARES:  # 3 primary + 2 extension squares

        for model_name, file_stem in file_keys:
            pq = PRED_DIR / f"{file_stem}_square_{square_id}.parquet"
            df = pd.read_parquet(pq)
            df["datetime"] = pd.to_datetime(df["datetime"])
            if df["datetime"].dt.tz is not None:
                df["datetime"] = df["datetime"].dt.tz_convert("Europe/Rome")

            fig, ax = plt.subplots(figsize=(11, 3.6))
            ax.plot(
                df["datetime"],
                df["y_true"],
                color="#333333",
                lw=0.9,
                label="Actual",
                alpha=0.9,
            )
            ax.plot(
                df["datetime"],
                df["y_pred"],
                color=MODEL_COLORS[model_name],
                lw=0.9,
                label=f"{MODEL_LABELS[model_name]} predicted",
                alpha=0.9,
            )
            ax.set_title(
                f"Square {square_id} — {MODEL_LABELS[model_name]} "
                f"(16–22 Dec 2013, one-step)"
            )
            ax.set_xlabel("Date (Europe/Rome)")
            ax.set_ylabel("Internet traffic")
            ax.legend(frameon=False, loc="upper right")
            fig.tight_layout()
            out = FIG_DIR / f"pred_{model_name}_square_{square_id}_dec16_22.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            saved.append(out)
            print(f"  Saved {out.name}")
    return saved


def _metrics_table(rows_by_model: Dict[str, dict]) -> List[str]:
    """One performance table: baselines first, then the three selected models.

    The lowest MAE among the *selected models* is bolded; baselines are shown
    for reference and are deliberately excluded from that comparison.
    """
    lines = [
        "| Model | MAE | MAPE (%) | RMSE | Train (s) | Infer (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    present_models = [m for m in MODEL_KEYS if m in rows_by_model]
    best_mae = min(rows_by_model[m]["mae"] for m in present_models)
    for key in BASELINE_KEYS + MODEL_KEYS:
        r = rows_by_model.get(key)
        if r is None:
            continue
        label = MODEL_LABELS[key]
        cells = (
            f"{r['mae']:.4f} | {r['mape']:.2f} | {r['rmse']:.4f} | "
            f"{r['train_seconds']:.3f} | {r['infer_seconds']:.3f}"
        )
        if key in MODEL_KEYS and r["mae"] == best_mae:
            lines.append(
                f"| **{label}** | "
                + " | ".join(f"**{c.strip()}**" for c in cells.split("|"))
                + " |"
            )
        else:
            marker = "*" if key in BASELINE_KEYS else ""
            lines.append(f"| {marker}{label}{marker} | {cells} |")
    return lines


def write_results_md(rows: List[dict]) -> Path:
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    by_square: Dict[int, Dict[str, dict]] = {s: {} for s in SQUARES}
    for r in rows:
        by_square[int(r["square_id"])][r["model"]] = r

    lines = [
        "# Forecasting results",
        "",
        "One-step-ahead Internet traffic forecasts for the evaluation week "
        "**16-22 December 2013** (1008 ten-minute steps). Hyperparameters were "
        "selected by grid search on square **5161** and the winning configs "
        "refit on every other square without a new search.",
        "",
        "Two trivial references are reported in every table: *persistence* "
        "(x(t+1)=x(t)) and *seasonal naive* (x(t+1)=x(t+1-144), same slot one "
        "day earlier). They estimate no parameters, so their training time is "
        "zero by construction. They are not candidates for the comparison; "
        "they fix the scale against which the three selected models are read.",
        "",
        "## Winning configs (from square 5161)",
        "",
        f"- **SARIMA:** `{BEST_SARIMA.label()}`",
        f"- **LSTM:** `{BEST_LSTM.label()}`",
        f"- **1D-CNN:** `{BEST_CNN.label()}`",
        "",
        "---",
        "",
        "# Required areas: the three highest-traffic squares",
        "",
        "Tables I-III below are the three tables required by Task 4, item III.",
        "",
    ]

    roman = ["I", "II", "III"]
    for i, square_id in enumerate(PRIMARY_SQUARES):
        rank = ["highest", "2nd highest", "3rd highest"][i]
        lines.append(f"## Table {roman[i]} - Square {square_id} ({rank} total traffic)")
        lines.append("")
        lines.extend(_metrics_table(by_square[square_id]))
        lines.append("")
        lines.append("Prediction overlays for this square:")
        for m in MODEL_KEYS:
            lines.append(
                f"- `report/figures/pred_{m}_square_{square_id}_dec16_22.png`"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "# Extension areas: squares 4159 and 4556",
            "",
            "These two squares are named individually in the exploratory task. "
            "They are carried through the identical pipeline to test whether the "
            "model ranking holds in a much lower-traffic regime.",
            "",
        ]
    )
    for square_id in EXTENSION_SQUARES:
        lines.append(f"## Square {square_id}")
        lines.append("")
        lines.extend(_metrics_table(by_square[square_id]))
        lines.append("")
        lines.append("Prediction overlays for this square:")
        for m in MODEL_KEYS:
            lines.append(
                f"- `report/figures/pred_{m}_square_{square_id}_dec16_22.png`"
            )
        lines.append("")

    lines.extend(
        [
            "## Cross-square notes",
            "",
            "- Absolute errors (MAE/RMSE) are not directly comparable across "
            "squares because baseline traffic levels differ by roughly an order "
            "of magnitude; MAPE is the cross-area metric.",
            "- SARIMA is refit on train+val before forecasting the test week; the "
            "LSTM and 1D-CNN are trained on train only, with val reserved for "
            "early stopping. The neural models therefore see *less* data, which "
            "makes their margin over SARIMA a conservative estimate.",
            "- Timing was recorded on the hardware fingerprint in "
            "`report/experiments.md` using `time.perf_counter()`.",
            "",
        ]
    )
    RESULTS_MD.write_text("\n".join(lines))
    return RESULTS_MD

def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for s in SQUARES:
        _ensure_sequences(s)

    all_rows: List[dict] = []

    # Existing 5161 results (already grid-searched)
    print("Loading square 5161 metrics from prior runs ...")
    for stem, key in [("sarima", "sarima"), ("lstm", "lstm"), ("cnn", "cnn1d")]:
        meta = load_metrics_from_json(5161, stem)
        meta["model"] = key
        all_rows.append(meta)

    others = ", ".join(str(s) for s in EXTRA_SQUARES)
    print(f"\nRefitting best-of-5161 configs on squares {others} ...")
    for square_id in EXTRA_SQUARES:
        print(f"\n=== Square {square_id} ===")
        all_rows.append(run_sarima_fixed(square_id))
        all_rows.append(run_lstm_fixed(square_id))
        all_rows.append(run_cnn_fixed(square_id))

    print("\nComputing trivial baselines for every square ...")
    for square_id in SQUARES:
        all_rows.extend(run_baselines(square_id))

    print(f"\nGenerating {3 * len(SQUARES)} prediction overlays ...")
    plot_overlays()

    print("\nWriting comparison tables ...")
    write_results_md(all_rows)
    print(f"Results → {RESULTS_MD}")
    print("Done.")


if __name__ == "__main__":
    main()
