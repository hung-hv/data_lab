"""
main.py
-------
Entry point for the accel log visualiser.

Usage
-----
    python main.py <log_file> [--schema <path>] [--save <output_dir>]

Arguments
---------
    log_file          Path to the hardware log file to analyse.
    --schema <path>   Schema YAML (default: log_schema.yaml next to this file).
    --save   <dir>    Optional. Save PNG graphs to this directory in addition
                      to displaying them interactively.

Example
-------
    python main.py capture.log
    python main.py capture.log --save ./graphs
    python main.py capture.log --schema my_schema.yaml
"""

import argparse
import sys

from engine import load_schema, parse_log
from plotter import plot_all


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="accel_verify",
        description="Parse a hardware log file and graph accelerometer X/Y/Z output.",
    )
    p.add_argument("log_file", help="Path to the input log file.")
    p.add_argument("--schema", default=None,
                   help="Path to schema YAML (default: log_schema.yaml).")
    p.add_argument("--save", metavar="OUTPUT_DIR", default=None,
                   help="Directory to save PNG graphs (optional).")
    return p


def main() -> None:
    args   = build_arg_parser().parse_args()
    schema = load_schema(args.schema) if args.schema else load_schema()

    print(f"[main] Schema  : {len(schema.channels)} channel(s) — {[c.name for c in schema.channels]}")
    print(f"[main] Parsing : {args.log_file}")

    store = parse_log(args.log_file, schema)

    total = sum(len(store[n]) for n in store.channel_names())
    for name in store.channel_names():
        print(f"[main]   {name}: {len(store[name]):,} entries")

    if total == 0:
        print("[main] No accel entries found – check log format and schema.")
        sys.exit(1)

    plot_all(store, save_dir=args.save)


if __name__ == "__main__":
    main()
