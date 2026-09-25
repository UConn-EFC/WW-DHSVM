"""pytest configuration: make the package importable from a checkout.

The repository is not installed as a package; the existing tests insert
``~/ww_dhsvm`` into ``sys.path``.  This file inserts the repository root
that contains this ``tests`` folder instead, so the suite runs from any
clone location.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
