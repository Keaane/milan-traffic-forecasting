"""Exploratory analyses that back Sections 4.1, 4.4 and 4.5 of the report.

Three analyses live here, each producing a figure and a small dict of
statistics so the notebook and the report quote the same numbers:

* :func:`spatial_analysis` — maps the 10,000 square totals onto the 100x100
  Milan lattice and measures how clustered the hotspots are (Moran's I under
  rook contiguity). Answers the *spatial* half of the data characterisation.
* :func:`decomposition_analysis` — MSTL decomposition of the busiest square
  into trend, daily (144) and weekly (1008) seasonal components plus a
  remainder, with the variance share attributable to each.
* :func:`calendar_analysis` — quantifies the Italian and Milanese public
  holidays that fall inside the observation window, which explain the three
  largest deviations from the ordinary weekly rhythm.

Run as a script to regenerate every figure under ``report/figures/`` and
write the statistics to ``report/eda_stats.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

PathLike = Union[str, Path]

_ROOT = Path(__file__).resolve().parents[1]
PROC = _ROOT / "data" / "processed"
FIG_DIR = _ROOT / "report" / "figures"
STATS_JSON = _ROOT / "report" / "eda_stats.json"

GRID_SIDE = 100  # the Milan grid is 100 x 100 squares
DAILY = 144  # 24 h at 10-minute resolution
WEEKLY = 7 * DAILY

# Public holidays inside the 1 Nov 2013 - 1 Jan 2014 observation window.
# Sant'Ambrogio is a Milan-only holiday: the city's patron saint.
HOLIDAYS: Dict[str, str] = {
    "2013-11-01": "All Saints (national)",
    "2013-12-07": "Sant'Ambrogio (Milan only)",
    "2013-12-08": "Immacolata (national)",
    "2013-12-25": "Christmas Day (national)",
    "2013-12-26": "Santo Stefano (national)",
    "2014-01-01": "New Year's Day (national)",
}


# Every timestamp in this project is Europe/Rome. Matplotlib formats date
# ticks in UTC unless told otherwise, which silently shifts a midnight tick
# onto the previous day.
TZ = "Europe/Rome"


def _style() -> None:
    plt.rcParams.update(
        {
            "timezone": TZ,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def load_square_series(square_id: int) -> pd.Series:
    """Read a prepared square's full raw series."""
    path = PROC / "sequences" / f"square_{square_id}" / "full_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python3 src/prepare_sequences.py "
            f"--square {square_id}` first."
        )
    return pd.read_parquet(path)["internet"]


# ---------------------------------------------------------------- spatial


def totals_to_grid(totals: pd.Series) -> np.ndarray:
    """Map square IDs onto the 100x100 lattice.

    Uses the dataset's row-major numbering: square ``i`` sits at row
    ``(i-1)//100`` counting from the south edge and column ``(i-1)%100``
    counting from the west edge. Missing IDs become NaN.
    """
    grid = np.full((GRID_SIDE, GRID_SIDE), np.nan, dtype=np.float64)
    idx = totals.index.to_numpy(dtype=np.int64) - 1
    grid[idx // GRID_SIDE, idx % GRID_SIDE] = totals.to_numpy(dtype=np.float64)
    return grid


def morans_i(grid: np.ndarray) -> float:
    """Global Moran's I under rook contiguity (4-neighbour adjacency).

    Positive values mean like-valued squares sit next to each other. The
    expected value under spatial randomness is ``-1/(n-1)``, i.e. ~0 here.
    """
    z = grid - np.nanmean(grid)
    z = np.nan_to_num(z, nan=0.0)
    valid = (~np.isnan(grid)).astype(np.float64)

    # Sum of z_i * z_j over rook-adjacent pairs, counted once per ordered pair
    cross = (
        np.sum(z[:, :-1] * z[:, 1:])
        + np.sum(z[:-1, :] * z[1:, :])
    ) * 2.0
    # Number of ordered adjacent pairs where both cells carry data
    w_sum = (
        np.sum(valid[:, :-1] * valid[:, 1:])
        + np.sum(valid[:-1, :] * valid[1:, :])
    ) * 2.0

    n = float(valid.sum())
    denom = np.sum(z**2)
    return float((n / w_sum) * (cross / denom))


def spatial_analysis(
    totals: pd.Series,
    *,
    top_k: int = 20,
    fig_path: PathLike = None,
) -> dict:
    """Heatmap of total traffic over the Milan lattice + clustering statistic."""
    _style()
    grid = totals_to_grid(totals)
    log_grid = np.log10(grid)

    top_ids = totals.nlargest(top_k).index.to_numpy(dtype=np.int64)
    top_rc = np.stack([(top_ids - 1) // GRID_SIDE, (top_ids - 1) % GRID_SIDE], axis=1)
    # Bounding box of the top-k squares, in squares
    span_rows = int(top_rc[:, 0].max() - top_rc[:, 0].min() + 1)
    span_cols = int(top_rc[:, 1].max() - top_rc[:, 1].min() + 1)

    stats = {
        "morans_i_log10_total": morans_i(log_grid),
        "morans_i_expected_under_randomness": -1.0 / (int((~np.isnan(grid)).sum()) - 1),
        "top_k": top_k,
        "top_k_row_span": span_rows,
        "top_k_col_span": span_cols,
        "top_k_share_of_grid_area_pct": 100.0 * span_rows * span_cols / grid.size,
        "top_k_share_of_total_traffic_pct": float(
            100.0 * totals.nlargest(top_k).sum() / totals.sum()
        ),
        "top3_ids": [int(i) for i in totals.nlargest(3).index],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    im = axes[0].imshow(
        log_grid, origin="lower", cmap="magma", interpolation="nearest"
    )
    axes[0].set_title("Total Internet traffic, log₁₀ (Nov 2013 – Jan 2014)")
    axes[0].set_xlabel("Grid column (west → east)")
    axes[0].set_ylabel("Grid row (south → north)")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="log₁₀ total traffic")

    # Offsets chosen so the three labels do not collide
    label_offsets = [(9, -11), (-9, 7), (9, 7)]
    for sid, (r, c), off in zip(top_ids[:3], top_rc[:3], label_offsets):
        axes[0].plot(c, r, marker="o", ms=7, mfc="none", mec="#7fdbff", mew=1.6)
        axes[0].annotate(
            str(sid),
            (c, r),
            textcoords="offset points",
            xytext=off,
            ha="left" if off[0] > 0 else "right",
            color="#7fdbff",
            fontsize=8,
            fontweight="bold",
        )

    # Zoom on the bounding box of the top-k squares, with a margin
    pad = 6
    r0 = max(0, int(top_rc[:, 0].min()) - pad)
    r1 = min(GRID_SIDE, int(top_rc[:, 0].max()) + pad + 1)
    c0 = max(0, int(top_rc[:, 1].min()) - pad)
    c1 = min(GRID_SIDE, int(top_rc[:, 1].max()) + pad + 1)
    im2 = axes[1].imshow(
        log_grid[r0:r1, c0:c1],
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        extent=(c0 - 0.5, c1 - 0.5, r0 - 0.5, r1 - 0.5),
    )
    axes[1].scatter(
        top_rc[:, 1],
        top_rc[:, 0],
        s=52,
        facecolors="none",
        edgecolors="#7fdbff",
        linewidths=1.3,
        label=f"top {top_k} squares",
    )
    axes[1].set_title(
        f"Central zoom — top {top_k} squares span {span_rows}x{span_cols} cells"
    )
    axes[1].set_xlabel("Grid column")
    axes[1].set_ylabel("Grid row")
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)
    fig.colorbar(im2, ax=axes[1], shrink=0.85, label="log₁₀ total traffic")

    fig.tight_layout()
    out = Path(fig_path or FIG_DIR / "spatial_traffic_grid.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    stats["figure"] = str(out.relative_to(_ROOT))
    return stats


# ---------------------------------------------------------- decomposition


def decomposition_analysis(
    series: pd.Series,
    *,
    square_id: int,
    fig_path: PathLike = None,
) -> dict:
    """MSTL decomposition into trend + daily + weekly seasonality + remainder.

    A single-period STL cannot separate the daily cycle from the weekly one;
    MSTL fits both, which is what the two ACF peaks (lag 144 and lag 1008)
    call for.
    """
    _style()
    res = MSTL(series.astype(float), periods=(DAILY, WEEKLY)).fit()

    seasonal = res.seasonal
    daily = seasonal.iloc[:, 0]
    weekly = seasonal.iloc[:, 1]
    trend, resid = res.trend, res.resid

    total_var = float(np.var(series.to_numpy(dtype=np.float64)))
    stats = {
        "square_id": square_id,
        "var_share_daily_pct": 100.0 * float(np.var(daily)) / total_var,
        "var_share_weekly_pct": 100.0 * float(np.var(weekly)) / total_var,
        "var_share_trend_pct": 100.0 * float(np.var(trend)) / total_var,
        "var_share_remainder_pct": 100.0 * float(np.var(resid)) / total_var,
        "daily_peak_to_trough": float(daily.max() - daily.min()),
        "weekly_peak_to_trough": float(weekly.max() - weekly.min()),
        "remainder_std": float(np.std(resid)),
        "series_std": float(np.std(series)),
        # How much of the signal a perfect seasonal-only forecast would leave
        "seasonal_plus_trend_share_pct": 100.0
        * (1.0 - float(np.var(resid)) / total_var),
    }

    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    panels = [
        (series, "Observed", "#333333"),
        (trend, "Trend", "#1f4e79"),
        (daily, f"Daily seasonal (period {DAILY})", "#c45c26"),
        (weekly, f"Weekly seasonal (period {WEEKLY})", "#6b8f71"),
        (resid, "Remainder", "#8a8a8a"),
    ]
    for ax, (data, title, color) in zip(axes, panels):
        ax.plot(data.index, data.to_numpy(dtype=np.float64), lw=0.6, color=color)
        ax.set_ylabel(title, fontsize=9)
        ax.axhline(0, color="#dddddd", lw=0.5, zorder=0)
    axes[0].set_title(
        f"MSTL decomposition — square {square_id} "
        f"(daily {DAILY} + weekly {WEEKLY})"
    )
    axes[-1].set_xlabel("Date (Europe/Rome)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=TZ))
    fig.tight_layout()

    out = Path(fig_path or FIG_DIR / f"square_{square_id}_mstl_decomposition.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    stats["figure"] = str(out.relative_to(_ROOT))
    return stats


# -------------------------------------------------------------- calendar


def calendar_analysis(
    series: pd.Series,
    *,
    square_id: int,
    fig_path: PathLike = None,
) -> dict:
    """Compare holiday days against the same weekday in ordinary weeks."""
    _style()
    daily_total = series.resample("1D").sum()
    daily_total.index = daily_total.index.tz_convert("Europe/Rome")

    dates = daily_total.index.normalize()
    holiday_dates = pd.to_datetime(list(HOLIDAYS)).tz_localize("Europe/Rome")
    is_holiday = dates.isin(holiday_dates)

    rows = []
    for d, label in HOLIDAYS.items():
        ts = pd.Timestamp(d, tz="Europe/Rome")
        if ts not in set(dates):
            continue
        weekday = ts.day_name()
        same_weekday = daily_total[(dates.day_name() == weekday) & (~is_holiday)]
        if same_weekday.empty:
            continue
        actual = float(daily_total.loc[daily_total.index.normalize() == ts].iloc[0])
        typical = float(same_weekday.median())
        rows.append(
            {
                "date": d,
                "label": label,
                "weekday": weekday,
                "daily_total": actual,
                "typical_same_weekday": typical,
                "deviation_pct": 100.0 * (actual - typical) / typical,
            }
        )

    stats = {"square_id": square_id, "holidays": rows}

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(
        daily_total.index,
        daily_total.to_numpy(dtype=np.float64),
        color="#1f4e79",
        lw=1.2,
        marker="o",
        ms=3,
        label="Daily total traffic",
    )
    y_top = ax.get_ylim()[1]
    for i, row in enumerate(rows):
        ts = pd.Timestamp(row["date"], tz="Europe/Rome")
        ax.axvline(ts, color="#c45c26", ls="--", lw=1.0, alpha=0.8)
        # Alternate the label height so adjacent holidays do not overlap
        ax.annotate(
            f"{row['label'].split(' (')[0]}  {row['deviation_pct']:+.0f}%",
            (ts, y_top * (1.0 if i % 2 == 0 else 0.72)),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7.5,
            color="#c45c26",
        )
    # The evaluation week
    ax.axvspan(
        pd.Timestamp("2013-12-16", tz="Europe/Rome"),
        pd.Timestamp("2013-12-23", tz="Europe/Rome"),
        color="#6b8f71",
        alpha=0.14,
        label="Test week (16–22 Dec)",
    )
    ax.set_title(
        f"Square {square_id} — daily totals with Italian / Milanese public holidays"
    )
    ax.set_xlabel("Date (Europe/Rome)")
    ax.set_ylabel("Daily total Internet traffic")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=TZ))
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()

    out = Path(fig_path or FIG_DIR / f"square_{square_id}_calendar_effects.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    stats["figure"] = str(out.relative_to(_ROOT))
    return stats


# ---------------------------------------------------------------- regimes


# Ordinary weeks only: the Christmas shutdown from 23 Dec would swamp the
# weekday/weekend contrast we are trying to measure.
ORDINARY_START = "2013-11-01"
ORDINARY_END = "2013-12-20"


def regime_analysis(
    square_ids,
    *,
    fig_path: PathLike = None,
) -> dict:
    """Classify squares by weekly rhythm and depth of the night trough.

    Two scalars separate the evaluation squares into distinct regimes:

    * ``weekend_weekday_ratio`` — median weekend daily total over median
      weekday daily total. Above 1 means a leisure/nightlife area; well below
      1 means an office/business area that empties at the weekend.
    * ``night_share_pct`` — share of all traffic falling in 01:00-05:00. A
      small value means a deep night trough, which is where percentage-based
      error metrics bite hardest.
    """
    _style()
    rows = []
    profiles = {}
    for sid in square_ids:
        series = load_square_series(sid)
        daily = series.resample("1D").sum()
        ordinary = daily.loc[ORDINARY_START:ORDINARY_END]
        weekend = float(ordinary[ordinary.index.dayofweek >= 5].median())
        weekday = float(ordinary[ordinary.index.dayofweek < 5].median())

        hour = series.index.hour
        night_share = 100.0 * float(series[(hour >= 1) & (hour < 5)].sum()) / float(
            series.sum()
        )

        rows.append(
            {
                "square_id": int(sid),
                "weekday_median_daily": weekday,
                "weekend_median_daily": weekend,
                "weekend_weekday_ratio": weekend / weekday,
                "night_share_pct": night_share,
                "regime": "leisure / weekend-peaking"
                if weekend / weekday > 1.0
                else "office / weekday-peaking",
            }
        )
        # Mean intra-day profile, normalised, for the shape panel
        by_slot = series.groupby(
            series.index.hour * 6 + series.index.minute // 10
        ).mean()
        profiles[int(sid)] = by_slot / by_slot.mean()

    stats = {"squares": rows}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = plt.cm.viridis(np.linspace(0.08, 0.86, len(rows)))

    for (row, color) in zip(rows, colors):
        sid = row["square_id"]
        prof = profiles[sid]
        axes[0].plot(
            prof.index / 6.0,
            prof.to_numpy(dtype=np.float64),
            lw=1.5,
            color=color,
            label=f"{sid} (we/wd {row['weekend_weekday_ratio']:.2f})",
        )
    axes[0].set_title("Mean intra-day profile (normalised to each square's mean)")
    axes[0].set_xlabel("Hour of day (Europe/Rome)")
    axes[0].set_ylabel("Traffic / square mean")
    axes[0].set_xticks(range(0, 25, 4))
    axes[0].axvspan(1, 5, color="#cccccc", alpha=0.3, zorder=0)
    axes[0].legend(frameon=False, fontsize=8)

    for row, color in zip(rows, colors):
        axes[1].scatter(
            row["weekend_weekday_ratio"],
            row["night_share_pct"],
            s=90,
            color=color,
            zorder=3,
        )
        axes[1].annotate(
            str(row["square_id"]),
            (row["weekend_weekday_ratio"], row["night_share_pct"]),
            textcoords="offset points",
            xytext=(8, 3),
            fontsize=9,
        )
    axes[1].axvline(1.0, color="#999999", ls="--", lw=1.0)
    axes[1].set_title("Weekly rhythm vs depth of the night trough")
    axes[1].set_xlabel("Weekend / weekday median daily traffic")
    axes[1].set_ylabel("Share of traffic in 01:00-05:00 (%)")
    y_lo, y_hi = axes[1].get_ylim()
    axes[1].set_ylim(y_lo - 0.12 * (y_hi - y_lo), y_hi)
    y_label = axes[1].get_ylim()[0] + 0.02 * (y_hi - y_lo)
    axes[1].annotate("← office-like", (0.97, y_label), ha="right", fontsize=8, color="#666666")
    axes[1].annotate("leisure-like →", (1.03, y_label), ha="left", fontsize=8, color="#666666")

    fig.tight_layout()
    out = Path(fig_path or FIG_DIR / "square_regimes.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    stats["figure"] = str(out.relative_to(_ROOT))
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate EDA figures/stats")
    parser.add_argument("--square", type=int, default=5161)
    parser.add_argument(
        "--regime-squares",
        default="5161,5059,5259,4159,4556",
        help="Comma-separated squares for the regime comparison",
    )
    args = parser.parse_args()

    totals = pd.read_parquet(PROC / "per_square_totals.parquet")["total_internet"]
    series = load_square_series(args.square)
    regime_squares = [int(s) for s in args.regime_squares.split(",") if s.strip()]

    out = {
        "spatial": spatial_analysis(totals),
        "decomposition": decomposition_analysis(series, square_id=args.square),
        "calendar": calendar_analysis(series, square_id=args.square),
        "regimes": regime_analysis(regime_squares),
    }
    STATS_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nStats → {STATS_JSON}")


if __name__ == "__main__":
    main()
