import csv
import glob
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR
FIELDNAMES = ['model', 'corpus', 'n', 'f1_clean', 'exact_match', 'f1_raw', 'contains_answer_diagnostic']

def aggregate(results_dir=None) -> Path:
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    per_cell_paths = sorted(glob.glob(str(results_dir / 'baseline_summary_*_*_*.csv')))
    per_cell_paths = [p for p in per_cell_paths if not p.endswith('baseline_summary_all.csv')]
    if not per_cell_paths:
        print(f'[aggregate] no baseline_summary_*.csv files found in {results_dir}')
        return None
    rows = []
    for path in per_cell_paths:
        with open(path, encoding='utf-8', newline='') as f:
            rows.extend(csv.DictReader(f))
    out_path = results_dir / 'baseline_summary_all.csv'
    fd, tmp_path = tempfile.mkstemp(dir=str(results_dir), prefix='.tmp_', suffix='.csv')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f'[aggregate] combined {len(per_cell_paths)} cell(s), {len(rows)} row(s) -> {out_path}')
    return out_path
if __name__ == '__main__':
    aggregate()
