"""
data_lab.py
-----------
Data analysis lab: applies a zero-phase Butterworth low-pass filter to
selected accel channels and plots original vs. filtered signals.

Usage
-----
    python data_lab.py <log_file>
                       [--schema  log_schema.yaml]
                       [--channel accel_xyz]
                       [--fields  x y z]
                       [--cutoff  5.0]     # Hz
                       [--order   4]
                       [--fs      auto]    # Hz  (auto = estimate from data)
                       [--save    <dir>]

Filter
------
    Butterworth IIR low-pass via scipy.signal.butter + sosfiltfilt.
    sosfiltfilt applies the filter forward then backward → zero phase distortion.

Sample-rate auto-detection
--------------------------
    Computed from the median inter-sample interval in seq_ts_ms.
    Override with --fs if the auto-detected rate is wrong.

Example
-------
    # Default: accel_xyz channel, x y z fields, 5 Hz cutoff, order 4
    python data_lab.py log_5_burst_read.txt

    # Custom: raw counts channel, 10 Hz cutoff, order 2
    python data_lab.py log_4.txt --channel accel_raw --cutoff 10 --order 2

    # Save graphs
    python data_lab.py log_8.txt --save ./graphs
"""

import argparse
import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.signal import butter, sosfiltfilt

from engine import Channel, DataStore, load_schema, parse_log


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def estimate_fs(ch: Channel) -> float:
    """
    Estimate sample rate (Hz) from the median inter-sample interval
    in seq_ts_ms.  Falls back to 50.0 Hz when there is insufficient data.
    """
    if "seq_ts_ms" not in ch or len(ch) < 2:
        return 50.0
    ts     = ch["seq_ts_ms"].astype(np.int64)
    deltas = np.diff(ts)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 50.0
    return 1000.0 / float(np.median(deltas))


def design_lowpass(cutoff_hz: float, fs_hz: float, order: int) -> np.ndarray:
    """
    Return SOS coefficients for a Butterworth low-pass filter.

    Raises
    ------
    ValueError
        When cutoff_hz ≥ Nyquist frequency (fs_hz / 2).
    """
    nyq = 0.5 * fs_hz
    if cutoff_hz >= nyq:
        raise ValueError(
            f"Cutoff frequency ({cutoff_hz:.2f} Hz) must be below the "
            f"Nyquist frequency ({nyq:.2f} Hz) for fs = {fs_hz:.1f} Hz."
        )
    return butter(order, cutoff_hz / nyq, btype="low", output="sos")


def apply_filter(data: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Apply zero-phase SOS filter to a 1-D float64 array."""
    if len(data) < 3 * (2 * (len(sos))):
        raise ValueError(
            f"Signal too short ({len(data)} samples) for filter order. "
            "Use a lower --order or collect more data."
        )
    return sosfiltfilt(sos, data.astype(np.float64))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _elapsed(ch: Channel) -> np.ndarray:
    ts = ch["seq_ts_ms"].astype(np.int64)
    return ts - ts[0]


def _fmt_ticker():
    return ticker.FuncFormatter(lambda v, _: f"{int(v):,}")


def plot_filtered(
    ch: Channel,
    fields: List[str],
    cutoff_hz: float,
    order: int,
    fs_hz: float,
    log_path: str,
    save_dir: Optional[str] = None,
) -> None:
    """
    Build one interactive figure with:
      - One subplot per field  (original overlaid with filtered)
      - One magnitude subplot  (|a_raw| vs |a_filtered|)
      - A slider at the bottom to adjust cutoff frequency in real time
      - A Reset button to restore the initial cutoff
    """
    filename   = os.path.basename(log_path)
    n_subplots = len(fields) + 1        # +1 for magnitude row
    ts         = _elapsed(ch)
    nyq        = 0.5 * fs_hz

    # Extra bottom margin for the slider + reset button
    fig, axes = plt.subplots(
        n_subplots, 1,
        figsize=(14, 3.8 * n_subplots),
        squeeze=False,
    )
    fig.subplots_adjust(bottom=0.13, top=0.93, hspace=0.55)

    # ── Pre-compute raw XYZ arrays (never change) ─────────────────────────
    raw_x = ch["x"].astype(np.float64) if "x" in ch else np.zeros(len(ch))
    raw_y = ch["y"].astype(np.float64) if "y" in ch else np.zeros(len(ch))
    raw_z = ch["z"].astype(np.float64) if "z" in ch else np.zeros(len(ch))
    mag_raw = np.sqrt(raw_x ** 2 + raw_y ** 2 + raw_z ** 2)

    def _filter_xyz(sos):
        out = {}
        for ax_name, raw_arr in (("x", raw_x), ("y", raw_y), ("z", raw_z)):
            if ax_name in ch:
                out[ax_name] = apply_filter(raw_arr, sos)
        return out

    def _mag_flt(fxyz):
        return np.sqrt(
            fxyz.get("x", raw_x) ** 2 +
            fxyz.get("y", raw_y) ** 2 +
            fxyz.get("z", raw_z) ** 2
        )

    def _field_title(fname, raw, flt, cutoff):
        noise_pct = (1 - flt.std() / raw.std()) * 100 if raw.std() > 0 else 0.0
        return (
            f"{fname.upper()}  --  raw std: {raw.std():,.1f}  |  "
            f"filtered std: {flt.std():,.1f}  |  "
            f"noise removed: {noise_pct:.1f}%  |  cutoff: {cutoff:.2f} Hz"
        )

    def _mag_title(mf, cutoff):
        return (
            f"|a|  --  raw mean: {mag_raw.mean():,.1f}  std: {mag_raw.std():,.1f}  |  "
            f"filtered mean: {mf.mean():,.1f}  std: {mf.std():,.1f}  |  "
            f"cutoff: {cutoff:.2f} Hz"
        )

    def _suptitle(cutoff):
        fig.suptitle(
            f"Data Lab  -  Butterworth Low-pass Filter   "
            f"[order={order}  fs={fs_hz:.1f} Hz  channel={ch.name}]\n"
            f"file: {filename}   cutoff: {cutoff:.2f} Hz",
            fontsize=11, fontweight="bold",
        )

    # ── Build initial filtered data ───────────────────────────────────────
    sos0         = design_lowpass(cutoff_hz, fs_hz, order)
    filtered_xyz = _filter_xyz(sos0)
    mf0          = _mag_flt(filtered_xyz)

    # ── Per-field subplots (save filtered line refs for live update) ───────
    flt_lines:  dict = {}   # fname -> filtered Line2D
    field_axes: dict = {}   # fname -> Axes

    for idx, fname in enumerate(fields):
        ax  = axes[idx][0]
        raw = ch[fname].astype(np.float64)
        flt = filtered_xyz.get(fname, apply_filter(raw, sos0))

        ax.plot(ts, raw, color="steelblue", linewidth=0.8, alpha=0.55,
                label=f"{fname.upper()} original")
        flt_line, = ax.plot(ts, flt, color="orangered", linewidth=1.5,
                            label=f"{fname.upper()} filtered")

        flt_lines[fname]  = flt_line
        field_axes[fname] = ax

        ax.set_ylabel(f"{fname.upper()}  [raw counts]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(_fmt_ticker())
        ax.set_title(_field_title(fname, raw, flt, cutoff_hz),
                     fontsize=9, loc="left", pad=4)

    # ── Magnitude subplot ─────────────────────────────────────────────────
    ax_mag = axes[len(fields)][0]

    ax_mag.plot(ts, mag_raw, color="steelblue", linewidth=0.8, alpha=0.55,
                label="|a| original")
    mag_flt_line, = ax_mag.plot(ts, mf0, color="orangered", linewidth=1.5,
                                label="|a| filtered")
    mean_line     = ax_mag.axhline(mf0.mean(), color="gray", linestyle=":",
                                   linewidth=1.0, label=f"mean = {mf0.mean():,.1f}")

    ax_mag.set_ylabel("|a|  =  sqrt(X^2+Y^2+Z^2)")
    ax_mag.set_xlabel("Elapsed time (ms from first sample)")
    ax_mag.set_title(_mag_title(mf0, cutoff_hz), fontsize=9, loc="left", pad=4)
    ax_mag.legend(loc="upper right", fontsize=8)
    ax_mag.grid(True, linestyle="--", alpha=0.4)
    ax_mag.xaxis.set_major_formatter(_fmt_ticker())

    _suptitle(cutoff_hz)

    # ── Cutoff-frequency slider ───────────────────────────────────────────
    slider_min = max(0.1, round(nyq * 0.005, 2))
    slider_max = round(nyq * 0.95, 2)

    ax_slider = fig.add_axes([0.13, 0.04, 0.62, 0.025])
    slider = Slider(
        ax       = ax_slider,
        label    = "Cutoff (Hz)",
        valmin   = slider_min,
        valmax   = slider_max,
        valinit  = cutoff_hz,
        color    = "orangered",
    )
    slider.label.set_fontsize(9)
    slider.valtext.set_fontsize(9)

    # ── Reset button ──────────────────────────────────────────────────────
    ax_reset  = fig.add_axes([0.82, 0.035, 0.07, 0.03])
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.85")

    # ── Live-update callback ──────────────────────────────────────────────
    def _update(val):
        new_cutoff = float(slider.val)
        try:
            new_sos = design_lowpass(new_cutoff, fs_hz, order)
        except ValueError:
            return

        new_fxyz = _filter_xyz(new_sos)
        new_mf   = _mag_flt(new_fxyz)

        for fname in fields:
            if fname not in flt_lines:
                continue
            raw = ch[fname].astype(np.float64)
            flt = new_fxyz.get(fname, apply_filter(raw, new_sos))
            flt_lines[fname].set_ydata(flt)
            field_axes[fname].set_title(
                _field_title(fname, raw, flt, new_cutoff),
                fontsize=9, loc="left", pad=4,
            )
            field_axes[fname].relim()
            field_axes[fname].autoscale_view()

        mag_flt_line.set_ydata(new_mf)
        mean_line.set_ydata([new_mf.mean(), new_mf.mean()])
        # rebuild legend so the mean label stays current
        ax_mag.legend(
            handles=[ax_mag.lines[0], mag_flt_line, mean_line],
            labels=["|a| original", "|a| filtered",
                    f"mean = {new_mf.mean():,.1f}"],
            loc="upper right", fontsize=8,
        )
        ax_mag.set_title(_mag_title(new_mf, new_cutoff),
                         fontsize=9, loc="left", pad=4)
        ax_mag.relim()
        ax_mag.autoscale_view()

        _suptitle(new_cutoff)
        fig.canvas.draw_idle()

    def _reset(event):
        slider.reset()

    slider.on_changed(_update)
    btn_reset.on_clicked(_reset)

    # Keep widget references alive so they aren't garbage-collected
    fig._data_lab_widgets = (slider, btn_reset)

    # ── Save (uses initial cutoff, before any slider interaction) ─────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(filename)[0]
        out  = os.path.join(
            save_dir,
            f"{stem}_{ch.name}_lowpass_c{cutoff_hz}Hz_o{order}.png",
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[data_lab] Saved -> {out}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    ch: Channel,
    fields: List[str],
    cutoff_hz: float,
    order: int,
    fs_hz: float,
) -> None:
    sos = design_lowpass(cutoff_hz, fs_hz, order)
    print(f"\n  {'Field':<8}  {'Raw mean':>10}  {'Raw std':>8}  "
          f"{'Filt mean':>10}  {'Filt std':>9}  {'Noise v':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*8}")
    for fname in fields:
        if fname not in ch:
            continue
        raw = ch[fname].astype(np.float64)
        flt = apply_filter(raw, sos)
        noise_pct = (1 - flt.std() / raw.std()) * 100 if raw.std() > 0 else 0.0
        print(f"  {fname.upper():<8}  {raw.mean():>10,.1f}  {raw.std():>8,.2f}  "
              f"{flt.mean():>10,.1f}  {flt.std():>9,.2f}  {noise_pct:>7.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data_lab",
        description=(
            "Apply a Butterworth low-pass filter to accel log data and "
            "plot original vs. filtered signals."
        ),
    )
    p.add_argument("log_file",
                   help="Path to the hardware log file.")
    p.add_argument("--schema",  default=None,
                   help="Path to schema YAML (default: log_schema.yaml).")
    p.add_argument("--channel", default="accel_xyz",
                   help="Channel name to analyse (default: accel_xyz).")
    p.add_argument("--fields",  nargs="+", default=["x", "y", "z"],
                   help="Fields to show in individual subplots (default: x y z).")
    p.add_argument("--cutoff",  type=float, default=5.0,
                   help="Low-pass cutoff frequency in Hz (default: 5.0).")
    p.add_argument("--order",   type=int,   default=4,
                   help="Butterworth filter order (default: 4).")
    p.add_argument("--fs",      type=float, default=None,
                   help="Sample rate Hz (default: auto-detected from data).")
    p.add_argument("--save",    metavar="OUTPUT_DIR", default=None,
                   help="Save PNG to this directory.")
    return p


def main() -> None:
    args   = build_arg_parser().parse_args()
    schema = load_schema(args.schema) if args.schema else load_schema()
    store  = parse_log(args.log_file, schema)

    ch = store[args.channel]
    if not ch:
        print(f"[data_lab] No data for channel '{args.channel}' in {args.log_file}")
        print(f"           Available channels: {store.channel_names()}")
        sys.exit(1)

    fs = args.fs if args.fs else estimate_fs(ch)

    print(f"[data_lab] File    : {args.log_file}")
    print(f"[data_lab] Channel : {ch.name}  ({len(ch):,} samples)")
    print(f"[data_lab] fs      : {fs:.2f} Hz  {'(auto-detected)' if not args.fs else '(manual)'}")
    print(f"[data_lab] Filter  : Butterworth LP  cutoff={args.cutoff} Hz  order={args.order}")

    fields  = [f for f in args.fields if f in ch]
    missing = [f for f in args.fields if f not in ch]
    if missing:
        print(f"[data_lab] Warning : fields not in channel: {missing}")
    if not fields:
        print("[data_lab] No valid fields to plot.")
        sys.exit(1)

    try:
        print_summary(ch, fields, args.cutoff, args.order, fs)
        plot_filtered(ch, fields, args.cutoff, args.order, fs,
                      args.log_file, save_dir=args.save)
    except ValueError as exc:
        print(f"[data_lab] Error: {exc}")
        sys.exit(1)

    plt.show()


if __name__ == "__main__":
    main()
