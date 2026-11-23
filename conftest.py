"""Put the repo root on sys.path so `from src.x import y` works under pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
