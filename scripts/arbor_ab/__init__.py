"""Shared library for the Arbor A/B test scripts (scripts/*.py). Stateless: all state lives in files under the repo."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
