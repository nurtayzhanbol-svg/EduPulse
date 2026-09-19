"""Vercel entrypoint: exposes the FastAPI + Socket.IO ASGI app as a serverless function."""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Vercel's function filesystem is read-only except /tmp.
os.environ.setdefault("EDUPULSE_DB", "/tmp/edupulse.db")

from main import socket_app as app  # noqa: E402,F401
