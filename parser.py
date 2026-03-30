"""
parser.py
---------
Parses a mixed hardware log file and extracts accelerometer entries.

Two supported line formats (the leading "Line NNN:" is optional):
  1. Raw counts (SAE):
     03/20/2026 16:24:55 (0094992791): 98:382 ROLLOVER:INF-0a:accel: X=320 Y=0 Z=8144 (raw counts, SAE)

  2. Accel XYZ summary (both variants are captured into accel_xyz):
     03/20/2026 16:24:55 (0094992791): 98:401 ROLLOVER:INF-0a:accel XYZ: X=320 Y=0 Z=8144
     03/24/2026 22:27:52 (0015970208):  0:062 ROLLOVER:INF-0a:XYZ: X=-184 Y=-836 Z=1844

Each parsed record is a dict:
  {
      "datetime":   str,   # "03/20/2026 16:24:55"
      "seq_ts":     str,   # sequence timestamp string, e.g. "98:382"
      "seq_ts_ms":  int,   # seq_ts converted to ms: 98*1000 + 382 = 98382
      "x":          int,
      "y":          int,
      "z":          int,
  }
"""

import re
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Shared prefix — the leading "Line NNN:" part is optional.
# Example: "03/20/2026 16:24:55 (0094992791): 98:382 ROLLOVER:INF-0a:"
_PREFIX = (
    r"(?:Line\s+\d+:\s+)?"                        # optional "Line NNN:" prefix (not captured)
    r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})"   # group 1 – datetime
    r"\s+\(\d+\):\s+"                             # hw timestamp (not captured)
    r"(\d+:\d+)\s+"                               # group 2 – sequence timestamp, e.g. "98:382"
)

# groups 3,4,5 – X, Y, Z values
_XYZ_VALUES = r"X=(-?\d+)\s+Y=(-?\d+)\s+Z=(-?\d+)"

# Pattern 1 – raw counts, SAE
_RAW_PATTERN = re.compile(
    _PREFIX + r".+?accel:\s+" + _XYZ_VALUES + r"\s+\(raw counts, SAE\)",
    re.IGNORECASE,
)

# Pattern 2 – accel XYZ summary (matches both "accel XYZ:" and bare "XYZ:")
_XYZ_PATTERN = re.compile(
    _PREFIX + r".+?(?:accel )?XYZ:\s+" + _XYZ_VALUES,
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Container for the two categories of parsed accel records."""
    raw_counts: List[dict] = field(default_factory=list)
    accel_xyz:  List[dict] = field(default_factory=list)


def _seq_ts_to_ms(seq_ts: str) -> int:
    """Convert sequence timestamp string to milliseconds.

    '98:382'  →  98 * 1000 + 382  =  98382 ms
    ' 0:063'  →   0 * 1000 +  63  =     63 ms
    """
    major, minor = seq_ts.split(":")
    return int(major) * 1000 + int(minor)


def _match_to_record(m: re.Match) -> dict:
    """Convert a regex match (5 groups) into a record dict.

    Group mapping:
      1 – datetime
      2 – seq_ts  (e.g. "98:382")
      3 – X
      4 – Y
      5 – Z
    """
    seq_ts = m.group(2)
    return {
        "datetime":  m.group(1),
        "seq_ts":    seq_ts,
        "seq_ts_ms": _seq_ts_to_ms(seq_ts),
        "x":         int(m.group(3)),
        "y":         int(m.group(4)),
        "z":         int(m.group(5)),
    }


def parse_log(filepath: str) -> ParseResult:
    """
    Read *filepath* line by line and extract all accelerometer entries.

    Returns a :class:`ParseResult` with two lists:
    - ``raw_counts`` – entries from ``accel: X=... (raw counts, SAE)`` lines
    - ``accel_xyz``  – entries from ``accel XYZ: X=...`` lines
    """
    result = ParseResult()

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RAW_PATTERN.search(line)
            if m:
                result.raw_counts.append(_match_to_record(m))
                continue  # a line won't match both patterns

            m = _XYZ_PATTERN.search(line)
            if m:
                result.accel_xyz.append(_match_to_record(m))

    return result
