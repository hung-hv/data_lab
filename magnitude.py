"""
magnitude.py
------------
Computes and plots  |a| = √(X² + Y² + Z²)  using a :class:`engine.DataStore`.

Usage
-----
    python magnitude.py <log_file> [<log_file2> ...]  [--schema <path>]  [--save <dir>]
"""

import argparse
import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from engine import DataStore, Channel, load_schema, parse_log


# ---------------------------------------------------------------------------
# Magnitude helpers
# ---------------------------------------------------------------------------

def compute_magnitude(ch: Channel) -> np.ndarray:
    """Return |a| = √(X²+Y²+Z²) as float64 array."""
    return np.sqrt(
        ch["x"].astype(np.float64) ** 2 +
        ch["y"].astype(np.float64) ** 2 +
        ch["z"].astype(np.float64) ** 2
    )


def _elapsed(ch: Channel) -> np.ndarray:
    ts = ch["seq_ts_ms"].astype(np.int64)
    return ts - ts[0]


def print_stats(label: str, mag: np.ndarray) -> None:
    if len(mag) == 0:
        print(f"  [{label}] no data")
        return
    print(
        f"  [{label}]  n={len(mag):,}  "
        f"min={mag.min():,.1f}  max={mag.max():,.1f}  "
        f"mean={mag.mean():,.1f}  std={mag.std():,.1f}"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_magnitude_ax(
    ax: plt.Axes,
    ts: np.ndarray,
    mag: np.ndarray,
    subtitle: str,
) -> None:
    ax.plot(ts, mag, color="steelblue", linewidth=1.0, marker="o", markersize=2, label="|a|")
    ax.axhline(mag.mean(), color="tomato", linestyle="--", linewidth=1.0,
               label=f"mean = {mag.mean():,.1f}")
    ax.set_title(subtitle, fontsize=11, fontweight="bold")
    ax.set_xlabel("Elapsed time (ms from first sample)")
    ax.set_ylabel("|a|  =  √(X²+Y²+Z²)  [raw counts]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def plot_magnitude(
    store: DataStore,
    log_path: str,
    save_dir: Optional[str] = None,
) -> None:
    filename = os.path.basename(log_path)
    active = [(n, store[n]) for n in ("accel_raw", "accel_xyz") if store[n]]

    n_plots = max(len(active), 1)
    fig, axes = plt.subplots(n_plots, 1, figsize=(13, 4.5 * n_plots), squeeze=False)
    fig.suptitle(
        f"Accel Magnitude  |a| = √(X²+Y²+Z²)\n{filename}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    if not active:
        axes[0][0].text(0.5, 0.5, "No data found", ha="center", va="center",
                        transform=axes[0][0].transAxes, fontsize=11, color="gray")
        axes[0][0].set_axis_off()
    else:
        for idx, (_, ch) in enumerate(active):
            mag = compute_magnitude(ch)
            _draw_magnitude_ax(
                axes[idx][0], _elapsed(ch), mag,
                f"{ch.description}  ({len(ch):,} samples)",
            )

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(filename)[0]
        out  = os.path.join(save_dir, f"{stem}_magnitude.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  [saved] \u2192 {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="magnitude",
        description="Plot √(X²+Y²+Z²) accel magnitude from hardware log files.",
    )
    p.add_argument("log_files", nargs="+", help="One or more log files to analyse.")
    p.add_argument("--schema", default=None,
                   help="Path to schema YAML (default: log_schema.yaml).")
    p.add_argument("--save", metavar="OUTPUT_DIR", default=None,
                   help="Directory to save PNG files (optional).")
    return p


def main() -> None:
    args   = build_arg_parser().parse_args()
    schema = load_schema(args.schema) if args.schema else load_schema()

    for log_path in args.log_files:
        print(f"\n\u2500\u2500 {log_path}")
        if not os.path.isfile(log_path):
            print(f"  [error] File not found: {log_path}")
            continue

        store = parse_log(log_path, schema)
        for name in store.channel_names():
            ch = store[name]
            print(f"  {name}: {len(ch):,} entries")
            if "x" in ch and "y" in ch and "z" in ch:
                print_stats(name, compute_magnitude(ch))

        plot_magnitude(store, log_path, save_dir=args.save)

    plt.show()


if __name__ == "__main__":
    main()
