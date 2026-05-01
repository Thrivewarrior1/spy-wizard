"""Pytest config: add the repo root to sys.path so tests can import
top-level modules (scraper, main, models) without an installed package.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
