"""Chunked loader for Telecom Italia Milan SMS/call/internet daily files.

Collapses country_code splits by summing internet activity per
(square_id, timestamp), then downcasts types for a compact parquet.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import psutil

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RAW_COLUMNS = [
    "square_id",
    "timestamp",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
]

PathLike = Union[str, Path]


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def _rss_mb() -> float:
    """Current process resident set size in MiB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


class _RssSampler:
    """Background thread that tracks peak RSS while a load runs."""

    def __init__(self, interval: float = 0.05) -> None:
        import threading

        self.interval = interval
        self.peak = _rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_RssSampler":
        self.peak = _rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.peak = max(self.peak, _rss_mb())

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, _rss_mb())


def _read_chunk(
    path: Path,
    *,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    """Read a daily TSV, keeping only square_id / timestamp / internet."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        usecols=["square_id", "timestamp", "internet"],
        dtype={
            "square_id": "int32",
            "timestamp": "int64",
            "internet": "float64",
        },
        na_values=[""],
        keep_default_na=True,
        chunksize=chunksize,
    )


def _aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Sum internet across country_code splits for one chunk."""
    chunk = chunk.dropna(subset=["internet"])
    if chunk.empty:
        return pd.DataFrame(columns=["square_id", "timestamp", "internet"])

    grouped = (
        chunk.groupby(["square_id", "timestamp"], sort=False, as_index=False)[
            "internet"
        ]
        .sum()
    )
    grouped["square_id"] = grouped["square_id"].astype(np.uint16)
    grouped["internet"] = grouped["internet"].astype(np.float32)
    return grouped


def load_day_optimized(
    path: PathLike,
    *,
    chunksize: int = 500_000,
    peak_rss: Optional[list] = None,
) -> pd.DataFrame:
    """Load one daily file with chunked aggregation and dtype downcasts.

    Returns a tidy frame with columns square_id (uint16), timestamp (int64),
    internet (float32), one row per (square_id, timestamp).

    If ``peak_rss`` is a one-element list, it is updated in-place with the
    maximum RSS (MiB) observed while loading.
    """
    path = _as_path(path)
    parts: list[pd.DataFrame] = []
    peak = _rss_mb()

    for chunk in _read_chunk(path, chunksize=chunksize):
        peak = max(peak, _rss_mb())
        agg = _aggregate_chunk(chunk)
        peak = max(peak, _rss_mb())
        if not agg.empty:
            parts.append(agg)

    if not parts:
        empty = pd.DataFrame(
            {
                "square_id": pd.Series(dtype=np.uint16),
                "timestamp": pd.Series(dtype=np.int64),
                "internet": pd.Series(dtype=np.float32),
            }
        )
        if peak_rss is not None:
            peak_rss[0] = max(peak_rss[0], peak)
        return empty

    # Chunks can split the same (square_id, timestamp) across boundaries,
    # so re-aggregate once after concatenating.
    out = pd.concat(parts, ignore_index=True)
    peak = max(peak, _rss_mb())
    out = (
        out.groupby(["square_id", "timestamp"], sort=False, as_index=False)[
            "internet"
        ]
        .sum()
    )
    out["square_id"] = out["square_id"].astype(np.uint16)
    out["internet"] = out["internet"].astype(np.float32)
    peak = max(peak, _rss_mb())
    if peak_rss is not None:
        peak_rss[0] = max(peak_rss[0], peak)
    return out


def load_day_naive(path: PathLike) -> pd.DataFrame:
    """Naive baseline: read the full file into memory, then aggregate."""
    path = _as_path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        na_values=[""],
        keep_default_na=True,
    )
    df["internet"] = pd.to_numeric(df["internet"], errors="coerce")
    df = df.dropna(subset=["internet"])
    out = (
        df.groupby(["square_id", "timestamp"], sort=False, as_index=False)[
            "internet"
        ]
        .sum()
    )
    return out


def process_days_to_parquet(
    raw_dir: PathLike,
    out_path: PathLike,
    *,
    pattern: str = "sms-call-internet-mi-*.txt",
    chunksize: int = 500_000,
    files: Optional[Iterable[PathLike]] = None,
) -> pd.DataFrame:
    """Process one or more daily files and write a tidy parquet dataset.

    If ``files`` is given, only those paths are processed; otherwise all
    files matching ``pattern`` under ``raw_dir`` are used.
    Returns the combined DataFrame that was written.
    """
    raw_dir = _as_path(raw_dir)
    out_path = _as_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if files is None:
        day_files = sorted(raw_dir.glob(pattern))
    else:
        day_files = [_as_path(p) for p in files]

    if not day_files:
        raise FileNotFoundError(f"No input files found under {raw_dir}")

    parts: list[pd.DataFrame] = []
    for day_file in day_files:
        parts.append(load_day_optimized(day_file, chunksize=chunksize))

    combined = pd.concat(parts, ignore_index=True)
    combined["square_id"] = combined["square_id"].astype(np.uint16)
    combined["internet"] = combined["internet"].astype(np.float32)
    combined.to_parquet(out_path, index=False)
    return combined


def per_square_totals(df: pd.DataFrame) -> pd.Series:
    """Sum internet traffic per square_id over the full observation period."""
    totals = df.groupby("square_id", sort=False)["internet"].sum()
    totals.name = "total_internet"
    return totals.astype(np.float64)


def _benchmark_one(
    mode: str,
    path: str,
    chunksize: int,
) -> dict:
    """Worker used in a fresh subprocess for an unbiased RSS measurement."""
    import gc

    gc.collect()
    baseline = _rss_mb()
    with _RssSampler() as sampler:
        t0 = time.perf_counter()
        if mode == "naive":
            df = load_day_naive(path)
        else:
            df = load_day_optimized(path, chunksize=chunksize)
        elapsed = time.perf_counter() - t0
    peak = max(sampler.peak, _rss_mb())
    return {
        "seconds": elapsed,
        "rss_before_mb": baseline,
        "rss_peak_mb": peak,
        "rss_delta_mb": peak - baseline,
        "rows_out": len(df),
        "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }


def benchmark_loaders(
    path: PathLike,
    *,
    chunksize: int = 500_000,
) -> dict:
    """Compare naive full-file load vs optimized chunked load on one day.

    Each approach runs in a fresh subprocess so RSS deltas are not polluted
    by allocator retention from the other approach.
    """
    import multiprocessing as mp

    path = _as_path(path)
    results: dict = {"file": str(path), "chunksize": chunksize}

    ctx = mp.get_context("spawn")

    def _run_isolated(mode: str) -> dict:
        # One-shot process per mode so the second run cannot inherit the
        # first run's retained heap / RSS.
        with ctx.Pool(1) as pool:
            return pool.apply(_benchmark_one, (mode, str(path), chunksize))

    results["naive"] = _run_isolated("naive")
    results["optimized"] = _run_isolated("optimized")

    results["speedup"] = (
        results["naive"]["seconds"] / results["optimized"]["seconds"]
        if results["optimized"]["seconds"] > 0
        else float("inf")
    )
    results["rss_delta_reduction_mb"] = (
        results["naive"]["rss_delta_mb"] - results["optimized"]["rss_delta_mb"]
    )
    results["frame_memory_reduction_mb"] = (
        results["naive"]["memory_mb"] - results["optimized"]["memory_mb"]
    )
    results["row_count_match"] = (
        results["naive"]["rows_out"] == results["optimized"]["rows_out"]
    )
    return results


def format_benchmark(results: dict) -> str:
    """Pretty-print benchmark results for the report / CLI."""
    n = results["naive"]
    o = results["optimized"]
    lines = [
        f"File: {results['file']}",
        f"Chunk size: {results['chunksize']:,}",
        "",
        "Naive (full-file read + groupby):",
        f"  wall time:     {n['seconds']:.2f} s",
        f"  RSS delta:     {n['rss_delta_mb']:.1f} MiB "
        f"(peak {n['rss_peak_mb']:.1f} MiB)",
        f"  frame memory:  {n['memory_mb']:.2f} MiB",
        f"  rows out:      {n['rows_out']:,}",
        f"  dtypes:        {n['dtypes']}",
        "",
        "Optimized (chunked + country collapse + downcast):",
        f"  wall time:     {o['seconds']:.2f} s",
        f"  RSS delta:     {o['rss_delta_mb']:.1f} MiB "
        f"(peak {o['rss_peak_mb']:.1f} MiB)",
        f"  frame memory:  {o['memory_mb']:.2f} MiB",
        f"  rows out:      {o['rows_out']:,}",
        f"  dtypes:        {o['dtypes']}",
        "",
        f"Speedup (time):              {results['speedup']:.2f}x",
        f"RSS delta reduction:         {results['rss_delta_reduction_mb']:.1f} MiB",
        f"Frame memory reduction:      {results['frame_memory_reduction_mb']:.2f} MiB",
        f"Output row counts match:     {results['row_count_match']}",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Load Milan traffic files and/or benchmark memory strategy."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw"),
        help="Directory containing daily TSV files",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Single daily file to process (default: first file in raw-dir)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output parquet path",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk for the optimized loader",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run naive vs optimized benchmark on the selected file",
    )
    parser.add_argument(
        "--bench-json",
        default=None,
        help=(
            "Persist benchmark results as JSON (default: "
            "report/memory_benchmark.json when --benchmark is given)"
        ),
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write optimized tidy parquet for the selected file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Stream all daily files into data/processed/internet_traffic.parquet",
    )
    args = parser.parse_args()

    raw_dir = _as_path(args.raw_dir)

    if args.all:
        import gc

        import pyarrow as pa
        import pyarrow.parquet as pq

        day_files = sorted(raw_dir.glob("sms-call-internet-mi-*.txt"))
        if not day_files:
            raise SystemExit(f"No daily files found in {raw_dir}")
        out = (
            _as_path(args.out)
            if args.out
            else raw_dir.parent / "processed" / "internet_traffic.parquet"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        totals = np.zeros(10001, dtype=np.float64)
        writer = None
        n_rows = 0
        t0 = time.perf_counter()
        for i, day_file in enumerate(day_files, 1):
            day = load_day_optimized(day_file, chunksize=args.chunksize)
            day_tot = day.groupby("square_id", sort=False)["internet"].sum()
            totals[day_tot.index.to_numpy(dtype=np.int64)] += day_tot.to_numpy(
                dtype=np.float64
            )
            table = pa.Table.from_pandas(day, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
            n_rows += len(day)
            del day, table, day_tot
            if i % 10 == 0 or i == len(day_files):
                gc.collect()
                print(
                    f"  [{i:02d}/{len(day_files)}] rows={n_rows:,} "
                    f"elapsed={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
        writer.close()
        sid = np.arange(1, 10001)
        ser = pd.Series(totals[1:], index=sid, name="total_internet")
        ser = ser[ser > 0]
        ser.to_frame().to_parquet(out.parent / "per_square_totals.parquet")
        print(f"Wrote {n_rows:,} rows -> {out}")
        print(
            f"Top square {int(ser.idxmax())} = {ser.max():,.1f} | "
            f"saved per_square_totals.parquet"
        )
        return

    if args.file:
        day_file = _as_path(args.file)
    else:
        candidates = sorted(raw_dir.glob("sms-call-internet-mi-*.txt"))
        if not candidates:
            raise SystemExit(f"No daily files found in {raw_dir}")
        day_file = candidates[0]

    # Default (no flags): benchmark + write one-day parquet.
    # --benchmark alone: benchmark only. --write-parquet alone: write only.
    do_benchmark = args.benchmark or not args.write_parquet
    do_write = args.write_parquet or not args.benchmark

    if do_benchmark:
        results = benchmark_loaders(day_file, chunksize=args.chunksize)
        print(format_benchmark(results))

        bench_json = (
            _as_path(args.bench_json)
            if args.bench_json
            else Path(__file__).resolve().parents[1] / "report" / "memory_benchmark.json"
        )
        bench_json.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(results)
        try:
            from src.experiments_log import collect_hardware_fingerprint

            payload["hardware"] = collect_hardware_fingerprint().to_dict()
        except Exception:  # noqa: BLE001 - fingerprint is a nicety, not a requirement
            pass
        bench_json.write_text(json.dumps(payload, indent=2))
        print(f"\nBenchmark JSON -> {bench_json}")

    if do_write:
        out = (
            _as_path(args.out)
            if args.out
            else _as_path(raw_dir).parents[0]
            / "processed"
            / f"{day_file.stem}.parquet"
        )
        df = process_days_to_parquet(
            raw_dir,
            out,
            files=[day_file],
            chunksize=args.chunksize,
        )
        totals = per_square_totals(df)
        print(
            f"\nWrote {len(df):,} rows to {out}\n"
            f"Squares: {totals.shape[0]:,} | "
            f"Top square {int(totals.idxmax())} = {totals.max():,.1f}"
        )


if __name__ == "__main__":
    main()
