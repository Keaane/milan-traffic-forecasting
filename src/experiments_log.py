"""Shared experiment logging: hardware fingerprint + timing helpers.

Use for every model (SARIMA, LSTM, 1D-CNN) so the report has consistent
hardware context and separate training vs one-step inference timings.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

DEFAULT_EXPERIMENTS_PATH = (
    Path(__file__).resolve().parents[1] / "report" / "experiments.md"
)


@dataclass(frozen=True)
class HardwareFingerprint:
    """Machine description for timing reproducibility notes."""

    system: str
    release: str
    machine: str
    processor: str
    cpu_brand: str
    cpu_count_logical: Optional[int]
    cpu_count_physical: Optional[int]
    ram_gb: Optional[float]
    python_version: str
    collected_at_utc: str

    def to_markdown(self) -> str:
        phys = self.cpu_count_physical if self.cpu_count_physical else "—"
        logical = self.cpu_count_logical if self.cpu_count_logical else "—"
        ram = f"{self.ram_gb:.1f} GiB" if self.ram_gb is not None else "—"
        return "\n".join(
            [
                "## Hardware fingerprint",
                "",
                f"- Collected (UTC): `{self.collected_at_utc}`",
                f"- OS: `{self.system} {self.release}` ({self.machine})",
                f"- CPU brand: `{self.cpu_brand}`",
                f"- `platform.processor()`: `{self.processor or '—'}`",
                f"- CPU cores: physical={phys}, logical={logical} "
                f"(`os.cpu_count()`={os.cpu_count()})",
                f"- RAM: {ram}",
                f"- Python: `{self.python_version}`",
                "",
                "Timing method: wall-clock via `time.perf_counter()` in-process; "
                "training/fit and one-step inference/forecast are measured separately.",
                "",
                "",
            ]
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _macos_cpu_brand() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def _linux_cpu_brand() -> Optional[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _physical_cpu_count() -> Optional[int]:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _ram_gb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return None


def collect_hardware_fingerprint() -> HardwareFingerprint:
    uname = platform.uname()
    brand = (
        _macos_cpu_brand()
        or _linux_cpu_brand()
        or platform.processor()
        or uname.processor
        or "unknown"
    )
    return HardwareFingerprint(
        system=uname.system,
        release=uname.release,
        machine=uname.machine,
        processor=platform.processor() or uname.processor or "",
        cpu_brand=brand,
        cpu_count_logical=os.cpu_count(),
        cpu_count_physical=_physical_cpu_count(),
        ram_gb=_ram_gb(),
        python_version=platform.python_version(),
        collected_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


@dataclass
class TimingStats:
    """Wall-clock seconds for training vs one-step inference."""

    train_seconds: float
    infer_seconds: float
    notes: str = ""

    def to_dict(self) -> dict:
        d = {
            "train_seconds": self.train_seconds,
            "infer_seconds": self.infer_seconds,
        }
        if self.notes:
            d["notes"] = self.notes
        return d


def ensure_experiments_header(
    path: PathLike = DEFAULT_EXPERIMENTS_PATH,
    *,
    hardware: Optional[HardwareFingerprint] = None,
    force_refresh_hardware: bool = False,
) -> HardwareFingerprint:
    """Ensure experiments.md exists with a hardware block at the top.

    If a hardware section already exists, it is left unchanged unless
    ``force_refresh_hardware`` is True.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hw = hardware or collect_hardware_fingerprint()

    if not path.exists():
        path.write_text(
            "# Experiments log\n\n"
            "Grid-search results, test metrics, and timing for forecasting models.\n\n"
            + hw.to_markdown()
        )
        return hw

    text = path.read_text()
    has_hw = "## Hardware fingerprint" in text
    if has_hw and not force_refresh_hardware:
        return hw

    # Strip an existing hardware section (through the blank line after notes)
    if has_hw:
        text = re.sub(
            r"## Hardware fingerprint\n.*?(?=\n## |\Z)",
            "",
            text,
            count=1,
            flags=re.DOTALL,
        )

    # Insert hardware after the title / intro paragraph
    if text.startswith("# Experiments log"):
        parts = text.split("\n", 1)
        rest = parts[1].lstrip("\n") if len(parts) > 1 else ""
        # Keep a short intro if present before first ##
        intro_end = rest.find("\n## ")
        if intro_end == -1:
            intro, remainder = rest.rstrip() + "\n\n", ""
        else:
            intro, remainder = rest[: intro_end + 1], rest[intro_end + 1 :]
        path.write_text(
            "# Experiments log\n\n"
            + intro.lstrip()
            + hw.to_markdown()
            + remainder
        )
    else:
        path.write_text(hw.to_markdown() + "\n" + text)
    return hw


def append_model_run(
    *,
    model_name: str,
    square_id: int,
    best_config: str,
    test_metrics: Mapping[str, float],
    timing: TimingStats,
    grid_rows: Sequence[Mapping[str, Any]],
    notes: Sequence[str] = (),
    path: PathLike = DEFAULT_EXPERIMENTS_PATH,
    replace_existing_section: bool = True,
    hardware: Optional[HardwareFingerprint] = None,
) -> None:
    """Append (or replace) a model section with grid table + split timings.

    Parameters
    ----------
    grid_rows :
        Iterable of mappings with keys used as table columns. Recommended:
        ``config``, ``val_mae``, ``val_rmse``, ``train_s`` (or ``fit_s``),
        ``status``, plus optional ``aic``, ``error``.
    test_metrics :
        Must include ``mae``, ``mape``, ``rmse``.
    replace_existing_section :
        If True, remove a previous ``## {model_name} ... square {id}`` block.
    """
    path = Path(path)
    ensure_experiments_header(path, hardware=hardware)

    heading = f"## {model_name} — square {square_id}"
    body_notes = list(notes) + [
        f"- Best config: **{best_config}**",
        (
            f"- Test week (one-step): MAE={test_metrics['mae']:.4f}, "
            f"MAPE={test_metrics['mape']:.2f}%, RMSE={test_metrics['rmse']:.4f}"
        ),
        (
            f"- **Timing (wall-clock):** train/fit = "
            f"**{timing.train_seconds:.3f}s**, one-step inference/forecast = "
            f"**{timing.infer_seconds:.3f}s**"
            + (f" ({timing.notes})" if timing.notes else "")
        ),
    ]

    # Build grid table from union of keys in order
    preferred = [
        "config",
        "val_mae",
        "val_rmse",
        "aic",
        "train_s",
        "fit_s",
        "infer_s",
        "status",
        "error",
    ]
    keys: list[str] = []
    for k in preferred:
        if any(k in row for row in grid_rows) and k not in keys:
            keys.append(k)
    for row in grid_rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    header = "| " + " | ".join(keys) + " |"
    sep_parts = []
    for k in keys:
        if k in {"config", "status", "error"}:
            sep_parts.append("---")
        else:
            sep_parts.append("---:")
    sep = "|" + "|".join(sep_parts) + "|"

    def _fmt(key: str, val: Any) -> str:
        if val is None:
            return "—"
        if isinstance(val, float):
            if key in {"val_mae", "val_rmse", "mae", "rmse"}:
                return f"{val:.4f}"
            if key in {"mape"}:
                return f"{val:.2f}"
            if key.endswith("_s") or key.endswith("seconds"):
                return f"{val:.3f}"
            if key == "aic":
                return f"{val:.1f}"
            return f"{val:.4f}"
        return str(val)

    lines = [heading, ""] + body_notes + ["", header, sep]
    for row in grid_rows:
        cells = [_fmt(k, row.get(k)) for k in keys]
        # backtick config column
        if keys and keys[0] == "config" and cells[0] != "—":
            cells[0] = f"`{cells[0]}`"
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "---", ""])
    block = "\n".join(lines)

    text = path.read_text()
    if replace_existing_section:
        # Match this model+square section through the following --- or EOF
        pattern = re.compile(
            rf"## {re.escape(model_name)} — square {square_id}\n.*?(?=\n## |\Z)",
            flags=re.DOTALL,
        )
        text = pattern.sub("", text)

    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + block)
