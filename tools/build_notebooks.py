"""Build .ipynb notebooks from their percent-format sources (notebooks/src/*.py).

The .py sources are the reviewable, diff-friendly version (same convention as jupytext):
    # %% [markdown]      starts a markdown cell (lines prefixed with "# ")
    # %%                 starts a code cell

Usage:  python tools/build_notebooks.py            # all notebooks (clears their outputs)
        python tools/build_notebooks.py 05 06       # only notebooks whose name starts with these prefixes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "notebooks" / "src", ROOT / "notebooks"

META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}


def _lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def parse(path: Path) -> dict:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(body)})
        elif body:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(body)})

    for line in path.read_text().split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    prefixes = tuple(sys.argv[1:])
    for src in sorted(SRC.glob("*.py")):
        if prefixes and not src.stem.startswith(prefixes):
            continue
        nb = parse(src)
        for i, c in enumerate(nb["cells"]):
            c["id"] = f"{src.stem[:2]}-{i:03d}"
        out = OUT / f"{src.stem}.ipynb"
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        print("built", out.relative_to(ROOT), f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    sys.exit(main())
