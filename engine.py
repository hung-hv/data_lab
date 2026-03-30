"""
engine.py
---------
Schema-driven log parsing engine.

Loads a YAML schema (default: log_schema.yaml) that declares regex channels,
then parses any log file against those channels. Results are stored as NumPy
arrays for O(1) column access and vectorised arithmetic across millions of
samples.

Data hierarchy
~~~~~~~~~~~~~~
    DataStore
    └── channels: dict[str, Channel]
        └── Channel
            ├── name, description
            └── [field_name] → np.ndarray
                  datetime    object (str) – "03/28/2026 10:11:12"
                  seq_ts      object (str) – "98:382"
                  seq_ts_ms   int64        – 98382  (auto-derived from seq_ts)
                  x, y, z     int32        – raw counts (or any user field)

Extending to new log types
~~~~~~~~~~~~~~~~~~~~~~~~~~
Add an entry under ``channels:`` in log_schema.yaml. No Python changes needed.

Public API
~~~~~~~~~~
    load_schema(path: str = "log_schema.yaml") -> Schema
    parse_log(log_path: str, schema: Schema, encoding: str = "latin-1") -> DataStore
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Default schema path (same directory as this file)
# ---------------------------------------------------------------------------
_DEFAULT_SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_schema.yaml")

# ---------------------------------------------------------------------------
# Numpy dtype mapping
# ---------------------------------------------------------------------------
_DTYPE_MAP: Dict[str, object] = {
    "str":     object,
    "int8":    np.int8,
    "int16":   np.int16,
    "int32":   np.int32,
    "int64":   np.int64,
    "float32": np.float32,
    "float64": np.float64,
}


# ---------------------------------------------------------------------------
# Schema types  (loaded once, reused across many parse_log calls)
# ---------------------------------------------------------------------------

class FieldDef:
    __slots__ = ("name", "dtype")

    def __init__(self, name: str, dtype: str) -> None:
        self.name  = name
        self.dtype = dtype


class ChannelDef:
    __slots__ = ("name", "description", "pattern", "fields")

    def __init__(
        self,
        name: str,
        description: str,
        pattern: re.Pattern,
        fields: List[FieldDef],
    ) -> None:
        self.name        = name
        self.description = description
        self.pattern     = pattern
        self.fields      = fields


class Schema:
    __slots__ = ("version", "description", "channels")

    def __init__(self, version: str, description: str, channels: List[ChannelDef]) -> None:
        self.version     = version
        self.description = description
        self.channels    = channels

    def __repr__(self) -> str:
        names = [c.name for c in self.channels]
        return f"Schema(v{self.version}, channels={names})"


# ---------------------------------------------------------------------------
# Runtime data types
# ---------------------------------------------------------------------------

class Channel:
    """
    One parsed channel backed by parallel NumPy arrays.

    ch["x"]          → np.ndarray (int32)
    ch["seq_ts_ms"]  → np.ndarray (int64)
    "x" in ch        → bool
    len(ch)          → number of samples
    bool(ch)         → False when empty
    """

    def __init__(self, name: str, description: str, arrays: Dict[str, np.ndarray]) -> None:
        self.name        = name
        self.description = description
        self._arrays     = arrays

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if not self._arrays:
            return 0
        return len(next(iter(self._arrays.values())))

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self._arrays

    def keys(self) -> List[str]:
        return list(self._arrays.keys())

    def __repr__(self) -> str:
        return f"Channel({self.name!r}, n={len(self):,}, fields={self.keys()})"


class _EmptyChannel(Channel):
    """Sentinel returned when a channel name is absent from a DataStore."""

    def __init__(self) -> None:
        super().__init__("__empty__", "", {})

    def __bool__(self)                   -> bool:          return False
    def __getitem__(self, key: str)      -> np.ndarray:    return np.array([], dtype=np.float64)
    def __contains__(self, key: str)     -> bool:          return False


_EMPTY_CHANNEL = _EmptyChannel()


class DataStore:
    """
    Container returned by :func:`parse_log`.

    store["accel_xyz"]          → Channel  (empty sentinel if absent)
    "accel_xyz" in store        → True only when channel has ≥1 sample
    store.channel_names()       → list of non-empty channel names
    """

    def __init__(self, source_file: str, channels: Dict[str, Channel]) -> None:
        self.source_file = source_file
        self.channels    = channels

    def __getitem__(self, name: str) -> Channel:
        return self.channels.get(name, _EMPTY_CHANNEL)

    def __contains__(self, name: str) -> bool:
        ch = self.channels.get(name)
        return ch is not None and len(ch) > 0

    def channel_names(self) -> List[str]:
        """Return names of channels that have at least one sample."""
        return [n for n, ch in self.channels.items() if len(ch) > 0]

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}:{len(ch):,}" for n, ch in self.channels.items())
        return f"DataStore({os.path.basename(self.source_file)!r}, [{parts}])"


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

def load_schema(path: Optional[str] = None) -> Schema:
    """
    Load and compile a YAML schema file.

    Parameters
    ----------
    path : str, optional
        Path to the YAML schema.  Defaults to ``log_schema.yaml`` in the
        same directory as this module.

    Returns
    -------
    Schema
        Compiled schema ready to be passed to :func:`parse_log`.
    """
    schema_path = path or _DEFAULT_SCHEMA
    with open(schema_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    prefix = raw.get("log_prefix", "")
    channel_defs: List[ChannelDef] = []

    for ch_raw in raw.get("channels", []):
        # Substitute {log_prefix} placeholder
        pat_str = ch_raw["pattern"].replace("{log_prefix}", prefix)
        try:
            compiled = re.compile(pat_str, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex for channel '{ch_raw['name']}': {exc}\n"
                f"  Pattern: {pat_str}"
            ) from exc

        fields = [
            FieldDef(name=f["name"], dtype=f.get("type", "str"))
            for f in ch_raw.get("fields", [])
        ]
        channel_defs.append(ChannelDef(
            name        = ch_raw["name"],
            description = ch_raw.get("description", ch_raw["name"]),
            pattern     = compiled,
            fields      = fields,
        ))

    return Schema(
        version     = str(raw.get("version", "1.0")),
        description = raw.get("description", ""),
        channels    = channel_defs,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _seq_ts_to_ms(seq_ts_arr: np.ndarray) -> np.ndarray:
    """
    Convert an object array of 'major:minor' strings to int64 milliseconds.
    e.g. '98:382' → 98382
    """
    return np.fromiter(
        (int(s.split(":")[0]) * 1000 + int(s.split(":")[1]) for s in seq_ts_arr),
        dtype=np.int64,
        count=len(seq_ts_arr),
    )


def parse_log(
    log_path: str,
    schema: Schema,
    encoding: str = "latin-1",
) -> DataStore:
    """
    Parse *log_path* using *schema* and return a :class:`DataStore`.

    Strategy for performance
    ~~~~~~~~~~~~~~~~~~~~~~~~
    1. Accumulate matched values in plain Python lists  (O(1) append).
    2. After the file is fully read, convert each list to a NumPy array
       in one allocation — much faster than growing arrays incrementally.
    3. ``seq_ts_ms`` (int64, ms) is auto-derived from ``seq_ts`` if present.

    A line is tested against channels in declaration order; on the first
    match the scan moves to the next line (one line → one channel).

    Parameters
    ----------
    log_path : str
        Path to the raw log file.
    schema   : Schema
        Compiled schema from :func:`load_schema`.
    encoding : str
        File encoding (default ``latin-1`` handles most embedded firmware logs).

    Returns
    -------
    DataStore
    """
    # Accumulators: channel_name → {field_name → list of raw strings}
    accum: Dict[str, Dict[str, list]] = {
        ch.name: {f.name: [] for f in ch.fields}
        for ch in schema.channels
    }

    with open(log_path, encoding=encoding, errors="replace") as fh:
        for line in fh:
            for ch in schema.channels:
                m = ch.pattern.search(line)
                if m:
                    buf = accum[ch.name]
                    for f in ch.fields:
                        buf[f.name].append(m.group(f.name))
                    break  # one line → one channel

    # Convert lists → numpy arrays
    channels: Dict[str, Channel] = {}
    for ch in schema.channels:
        arrays: Dict[str, np.ndarray] = {}
        buf = accum[ch.name]

        for f in ch.fields:
            np_dtype = _DTYPE_MAP.get(f.dtype, object)
            arrays[f.name] = np.array(buf[f.name], dtype=np_dtype)

        # Auto-derive seq_ts_ms from seq_ts when present and non-empty
        if "seq_ts" in arrays and len(arrays["seq_ts"]) > 0:
            arrays["seq_ts_ms"] = _seq_ts_to_ms(arrays["seq_ts"])

        channels[ch.name] = Channel(
            name        = ch.name,
            description = ch.description,
            arrays      = arrays,
        )

    return DataStore(source_file=log_path, channels=channels)
