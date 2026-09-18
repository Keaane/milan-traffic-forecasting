"""Post-hoc analysis of the saved test-week predictions.

Every quantitative claim in Sections 6 and 7 of the report is produced here
rather than typed by hand, so the report and the artifacts under
``data/processed/predictions/`` cannot drift apart. Writes
``report/results_stats.json`` and the failure-case figures.

    python3 src/analyze_results.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = Union[str, Path]

_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = _ROOT / "data" / "processed" / "predictions"
FIG_DIR = _ROOT / "report" / "figures"
STATS_JSON = _ROOT / "report" / "results_stats.json"

PRIMARY_SQUARES = [5161, 5059, 5259]
EXTENSION_SQUARES = [4159, 4556]
ALL_SQUARES = PRIMARY_SQUARES + EXTENSION_SQUARES

# Prediction files are stored under the model's own stem
FILE_STEM = {
    "persistence": "persistence",
    "seasonal_naive": "seasonal_naive",
    "sarima": "sarima",
    "lstm": "lstm",
    "cnn1d": "cnn",
}
MODEL_LABELS = {
    "persistence": "Persistence",
    "seasonal_naive": "Seasonal naive",
    "sarima": "SARIMA",
    "lstm": "LSTM",
    "cnn1d": "1D-CNN",
}
TZ = "Europe/Rome"
NIGHT_HOURS = (3, 4, 5)
DAY_HOURS = range(9, 18)


def load_predictions(square_id: int, model: str) -> pd.DataFrame:
    """Test-week predictions for one (square, model), indexed by timestamp."""
    path = PRED_DIR / f"{FILE_STEM[model]}_square_{square_id}.parquet"
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_convert(TZ)
    df = df.set_index("datetime").sort_index()
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    df["ape"] = 100.0 * df["abs_err"] / df["y_true"]
    return df


def load_meta(square_id: int, model: str) -> dict:
    path = PRED_DIR / f"{FILE_STEM[model]}_square_{square_id}.json"
    return json.loads(path.read_text())


def _pct_change(new: float, old: float) -> float:
    """Percentage improvement of ``new`` over ``old`` (positive = better)."""
    return 100.0 * (old - new) / old


def skill_scores() -> List[dict]:
    """MAE of each model relative to the persistence baseline, per square.

    A skill score above zero means the model beats "nothing changed"; below
    zero means it does not, which is the result worth reporting honestly.
    """
    rows = []
    for square_id in ALL_SQUARES:
        base = load_meta(square_id, "persistence")
        for model in ["seasonal_naive", "sarima", "lstm", "cnn1d"]:
            meta = load_meta(square_id, model)
            rows.append(
                {
                    "square_id": square_id,
                    "model": model,
                    "mae": meta["mae"],
                    "mape": meta["mape"],
                    "rmse": meta["rmse"],
                    "persistence_mae": base["mae"],
                    "mae_skill_vs_persistence_pct": _pct_change(
                        meta["mae"], base["mae"]
                    ),
                    "mape_skill_vs_persistence_pct": _pct_change(
                        meta["mape"], base["mape"]
                    ),
                    "beats_persistence_mae": bool(meta["mae"] < base["mae"]),
                }
            )
    return rows


def accuracy_compute_tradeoff() -> List[dict]:
    """LSTM accuracy gain over the 1D-CNN against its training-cost multiple."""
    rows = []
    for square_id in ALL_SQUARES:
        lstm, cnn = load_meta(square_id, "lstm"), load_meta(square_id, "cnn1d")
        rows.append(
            {
                "square_id": square_id,
                "lstm_mae": lstm["mae"],
                "cnn_mae": cnn["mae"],
                "mae_improvement_pct": _pct_change(lstm["mae"], cnn["mae"]),
                "lstm_train_s": lstm["train_seconds"],
                "cnn_train_s": cnn["train_seconds"],
                "train_time_ratio": lstm["train_seconds"] / cnn["train_seconds"],
                "lstm_infer_s": lstm["infer_seconds"],
                "cnn_infer_s": cnn["infer_seconds"],
            }
        )
    return rows


def hourly_error_profile(square_id: int) -> dict:
    """Mean APE by hour of day, per model, plus night/day aggregates."""
    out: Dict[str, dict] = {}
    for model in ["persistence", "sarima", "lstm", "cnn1d"]:
        df = load_predictions(square_id, model)
        hour = df.index.hour
        by_hour = df.groupby(hour)["ape"].mean()
        night_mask = np.isin(hour, NIGHT_HOURS)
        day_mask = np.isin(hour, list(DAY_HOURS))
        out[model] = {
            "mape_by_hour": {int(h): float(v) for h, v in by_hour.items()},
            "night_mape_03_05": float(df.loc[night_mask, "ape"].mean()),
            "day_mape_09_17": float(df.loc[day_mask, "ape"].mean()),
            "night_share_of_ape_mass_pct": float(
                100.0 * df.loc[night_mask, "ape"].sum() / df["ape"].sum()
            ),
        }
    return out


def worst_day_analysis(square_id: int) -> dict:
    """The calendar day on which each model does worst, by mean APE."""
    out = {}
    for model in ["persistence", "sarima", "lstm", "cnn1d"]:
        df = load_predictions(square_id, model)
        by_day = df.groupby(df.index.normalize())["ape"].mean()
        worst = by_day.idxmax()
        out[model] = {
            "worst_day": str(worst.date()),
            "worst_day_name": worst.day_name(),
            "worst_day_mape": float(by_day.max()),
            "best_day": str(by_day.idxmin().date()),
            "best_day_mape": float(by_day.min()),
            "mape_by_day": {
                str(d.date()): float(v) for d, v in by_day.items()
            },
        }
    return out


def plot_failure_case(
    square_id: int,
    model: str,
    reference: str = "sarima",
    fig_path: PathLike = None,
) -> Path:
    """Full test week plus a zoom on the night where ``model`` does worst."""
    plt.rcParams.update(
        {"timezone": TZ, "figure.dpi": 120, "savefig.dpi": 200,
         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
    )
    df = load_predictions(square_id, model)
    ref = load_predictions(square_id, reference)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4))
    axes[0].plot(df.index, df["y_true"], color="#333333", lw=0.9, label="Actual")
    axes[0].plot(
        df.index, df["y_pred"], color="#6b8f71", lw=0.9,
        label=f"{MODEL_LABELS[model]} predicted",
    )
    for day in pd.date_range(df.index[0].normalize(), df.index[-1], freq="1D"):
        axes[0].axvspan(
            day + pd.Timedelta(hours=3), day + pd.Timedelta(hours=6),
            color="#c45c26", alpha=0.12, zorder=0,
        )
    axes[0].set_title(
        f"Square {square_id} — {MODEL_LABELS[model]}, 16–22 Dec "
        "(03:00–06:00 shaded)"
    )
    axes[0].set_ylabel("Internet traffic")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=TZ))

    # Zoom on the night with the largest mean APE
    night = df[np.isin(df.index.hour, NIGHT_HOURS)]
    worst_night = night.groupby(night.index.normalize())["ape"].mean().idxmax()
    lo = worst_night + pd.Timedelta(hours=0)
    hi = worst_night + pd.Timedelta(hours=9)
    z, zr = df.loc[lo:hi], ref.loc[lo:hi]
    axes[1].plot(z.index, z["y_true"], color="#333333", lw=1.4, label="Actual")
    axes[1].plot(
        z.index, z["y_pred"], color="#6b8f71", lw=1.4,
        label=f"{MODEL_LABELS[model]} predicted",
    )
    axes[1].plot(
        zr.index, zr["y_pred"], color="#1f4e79", lw=1.2, ls="--",
        label=f"{MODEL_LABELS[reference]} predicted",
    )
    axes[1].axvspan(
        worst_night + pd.Timedelta(hours=3),
        worst_night + pd.Timedelta(hours=6),
        color="#c45c26", alpha=0.12, zorder=0,
    )
    axes[1].set_title(
        f"Zoom — {worst_night.strftime('%a %d %b')} 00:00–09:00 "
        f"(worst night for {MODEL_LABELS[model]})"
    )
    axes[1].set_xlabel("Time (Europe/Rome)")
    axes[1].set_ylabel("Internet traffic")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))

    fig.tight_layout()
    out = Path(
        fig_path or FIG_DIR / f"failure_{model}_square_{square_id}_overnight.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_weekend_failure(square_id: int = 5259, fig_path: PathLike = None) -> Path:
    """All three models over the test week of the most weekday-skewed square."""
    plt.rcParams.update({"timezone": TZ, "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10})
    colors = {"sarima": "#1f4e79", "lstm": "#c45c26", "cnn1d": "#6b8f71"}

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    truth = load_predictions(square_id, "lstm")
    ax.plot(truth.index, truth["y_true"], color="#333333", lw=1.1, label="Actual")
    for model, color in colors.items():
        df = load_predictions(square_id, model)
        ax.plot(
            df.index, df["y_pred"], color=color, lw=0.9, alpha=0.85,
            label=f"{MODEL_LABELS[model]} predicted",
        )
    # Saturday 21 and Sunday 22 December
    ax.axvspan(
        pd.Timestamp("2013-12-21", tz="Europe/Rome"),
        pd.Timestamp("2013-12-23", tz="Europe/Rome"),
        color="#c45c26", alpha=0.12, zorder=0, label="Weekend (21–22 Dec)",
    )
    ax.set_title(
        f"Square {square_id} — the weekend collapse the models must absorb"
    )
    ax.set_xlabel("Date (Europe/Rome)")
    ax.set_ylabel("Internet traffic")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=TZ))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()

    out = Path(fig_path or FIG_DIR / f"failure_weekend_square_{square_id}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    stats = {
        "skill_vs_persistence": skill_scores(),
        "accuracy_compute_tradeoff": accuracy_compute_tradeoff(),
        "hourly_error_profile": {
            str(s): hourly_error_profile(s) for s in PRIMARY_SQUARES
        },
        "worst_day": {str(s): worst_day_analysis(s) for s in ALL_SQUARES},
    }

    figures = [
        plot_failure_case(5161, "cnn1d"),
        plot_weekend_failure(5259),
    ]
    stats["figures"] = [str(p.relative_to(_ROOT)) for p in figures]

    STATS_JSON.write_text(json.dumps(stats, indent=2))
    print(f"Stats  → {STATS_JSON}")
    for p in figures:
        print(f"Figure → {p.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
