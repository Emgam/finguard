"""Minimal test runner for environments without pytest (tests are pytest-compatible).

Usage:  python tools/run_tests.py        # or simply: pytest
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from unittest import SkipTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))   # shared test helpers such as _synthetic

failed = passed = skipped = 0
for path in sorted((ROOT / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {path.name}::{name}")
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP {path.name}::{name} ({exc})")
        except Exception:
            failed += 1
            print(f"FAIL {path.name}::{name}\n{traceback.format_exc()}")
print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
sys.exit(1 if failed else 0)
