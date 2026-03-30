"""
plotter.py
----------
Draws X / Y / Z time-series graphs from a :class:`engine.DataStore`.

The X-axis is seq_ts_ms normalised to 0 at the first sample, so it shows
elapsed milliseconds regardless of log origin.

Public API
~~~~~~~~~~
    plot_all(store: DataStore, save_dir: str | None = None) -> None
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from engine import DataStore, Channel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed(ch: Channel) -> np.ndarray:
    """Elapsed ms from first sample."""
    ts = ch["seq_ts_ms"].astype(np.int64)
    return ts - ts[0]


def _draw_axes(ax: plt.Axes, elapsed: np.ndarray, ch: Channel, title: str) -> None:
    """Plot X / Y / Z onto *ax*."""
    ax.plot(elapsed, ch["x"], label="X", marker="o", markersize=2, linewidth=1.1)
    ax.plot(elapsed, ch["y"], label="Y", marker="s", markersize=2, linewidth=1.1)
    ax.plot(elapsed, ch["z"], label="Z", marker="^", markersize=2, linewidth=1.1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Elapsed time (ms from first sample)")
    ax.set_ylabel("Acceleration (raw counts)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def _annotate_empty(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(0.5, 0.5, "No data found", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_axis_off()


def _add_stats_table(fig: plt.Figure, ch: Channel) -> None:
    """Append min / max / mean table below the figure."""
    if not ch:
        return
    rows = []
    for axis in ("x", "y", "z"):
        if axis not in ch:
            continue
        v = ch[axis].astype(np.float64)
        rows.append([axis.upper(), f"{v.min():,.0f}", f"{v.max():,.0f}", f"{v.mean():,.1f}"])
    if not rows:
        return
    tbl_ax = fig.add_axes([0.15, -0.18, 0.7, 0.14])
    tbl_ax.axis("off")
    tbl = tbl_ax.table(cellText=rows, colLabels=["Axis", "Min", "Max", "Mean"],
                       cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)


def _plot_single(
    ch: Channel,
    title: str,
    filename: str,
    save_dir: Optional[str],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.tight_layout(pad=3.0)
    if ch:
        _draw_axes(ax, _elapsed(ch), ch, f"{title}  ({len(ch):,} samples)")
    else:
        _annotate_empty(ax, title)
    _add_stats_table(fig, ch)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, filename)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plotter] Saved \u2192 {out}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_all(store: DataStore, save_dir: Optional[str] = None) -> None:
    """
    Draw one figure for each accel channel (accel_raw, accel_xyz) and show.

    Parameters
    ----------
    store    : DataStore returned by engine.parse_log()
    save_dir : If given, also save PNG files to this directory.
    """
    _plot_single(store["accel_raw"], "Accel \u2013 Raw Counts (SAE)",
                 "accel_raw_counts_sae.png", save_dir)
    _plot_single(store["accel_xyz"], "Accel XYZ Summary",
                 "accel_xyz.png", save_dir)
    plt.show()
