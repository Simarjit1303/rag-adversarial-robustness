"""Put the repo root on sys.path so tests can `import scripts...` / `import config`
regardless of pytest's import mode or the directory it's invoked from."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
