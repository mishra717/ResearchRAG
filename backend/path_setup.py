"""
path_setup.py
─────────────
Standalone module that fixes sys.path BEFORE uvicorn tries to import main.py.
Used via the app factory pattern: uvicorn path_setup:create_app
"""

import sys
from pathlib import Path

# Insert project root at position 0 — works in every process
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


def create_app():
    """App factory — uvicorn calls this which triggers the path fix first."""
    from main import app
    return app
