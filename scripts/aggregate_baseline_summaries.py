"""
Combines every per-cell baseline_summary_{model}_{corpus}_{engine}.csv into
one baseline_summary_all.csv, for reading results across the full matrix
without opening a dozen separate files.

Deliberately a separate, run-once-after-the-fact script rather than a
shared file every sweep run appends to live: an append-safe write pattern
over a file multiple concurrent/sequential runs (e.g. one pod per cell for
a parallel sweep) all touch would need its own concurrency guarantees, and
atomicity over the Network Volume under concurrent access is unverified
(see data/build_index.py's atomic-write notes). Run this only after every
cell you care about has finished writing -- it's a read-only aggregation
over already-complete files, not a live participant in any run.

Usage:
    python -m scripts.aggregate_baseline_summaries
"""

import csv
import glob
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RESULTS_DIR  # noqa: E402

FIELDNAMES = ["model", "corpus", "n", "f1_clean", "exact_match",
              "f1_raw", "contains_answer_diagnostic"]


def aggregate(results_dir=None) -> Path:
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    per_cell_paths = sorted(glob.glob(str(results_dir / "baseline_summary_*_*_*.csv")))
    # Exclude a stale baseline_summary_all.csv from a prior run of this
    # same script, in case it's ever re-run without clearing the directory.
    per_cell_paths = [p for p in per_cell_paths if not p.endswith("baseline_summary_all.csv")]

    if not per_cell_paths:
        print(f"[aggregate] no baseline_summary_*.csv files found in {results_dir}")
        return None

    rows = []
    for path in per_cell_paths:
        with open(path, encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))

    out_path = results_dir / "baseline_summary_all.csv"
    fd, tmp_path = tempfile.mkstemp(dir=str(results_dir), prefix=".tmp_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"[aggregate] combined {len(per_cell_paths)} cell(s), {len(rows)} row(s) -> {out_path}")
    return out_path


if __name__ == "__main__":
    aggregate()
