"""Execute notebooks headlessly and store their outputs, without Jupyter.

Prefer the standard tool where it is installed:
    jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb

This fallback exists for environments without Jupyter (it was used to produce the
committed outputs). It runs each code cell in one shared namespace, captures printed text,
`display(...)` calls, the value of a trailing expression, and every open matplotlib figure,
and writes them into the notebook as standard nbformat v4 outputs.

Usage:  python tools/run_notebooks.py notebooks/01_data_validation.ipynb [...]
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _bundle(obj) -> dict:
    data = {"text/plain": repr(obj)}
    html = getattr(obj, "_repr_html_", None)
    if callable(html):
        try:
            h = html()
            if h:
                data["text/html"] = h
        except Exception:  # pragma: no cover
            pass
    if hasattr(obj, "to_string") and "text/html" not in data:
        try:
            data["text/plain"] = obj.to_string()
        except Exception:
            pass
    return data


def _figures() -> list[dict]:
    outs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        outs.append({"output_type": "display_data", "metadata": {},
                     "data": {"image/png": base64.b64encode(buf.getvalue()).decode(), "text/plain": f"<Figure {num}>"}})
    plt.close("all")
    return outs


def run(path: Path) -> float:
    nb = json.loads(path.read_text())
    os.chdir(path.parent)
    sys.path.insert(0, str(ROOT / "src"))
    ns: dict = {"__name__": "__main__"}
    t0 = time.time()
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        outputs: list[dict] = []
        buf = io.StringIO()

        def flush():
            text = buf.getvalue()
            if text:
                outputs.append({"output_type": "stream", "name": "stdout", "text": text})
                buf.seek(0)
                buf.truncate()

        def display(*objs):
            flush()
            for o in objs:
                outputs.extend(_figures()) if o is None else outputs.append(
                    {"output_type": "display_data", "metadata": {}, "data": _bundle(o)})

        ns["display"] = display
        src = "".join(cell["source"])
        tree = ast.parse(src)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                result = eval(compile(ast.Expression(last.value), f"<cell {count}>", "eval"), ns) if last else None
        except Exception as exc:
            flush()
            outputs.append({"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc),
                            "traceback": traceback.format_exc().split("\n")})
            cell["outputs"], cell["execution_count"] = outputs, count
            path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
            raise RuntimeError(f"{path.name}: cell {count} failed: {exc}") from exc
        flush()
        outputs.extend(_figures())
        if result is not None and not hasattr(result, "savefig"):
            outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": _bundle(result)})
        cell["outputs"], cell["execution_count"] = outputs, count
        print(f"  cell {count:>2} done  ({time.time() - t0:5.1f}s)", flush=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return time.time() - t0


if __name__ == "__main__":
    paths = [Path(p).resolve() for p in sys.argv[1:]]
    if len(paths) == 1:
        print("running", paths[0].name, flush=True)
        took = run(paths[0])
        print(f"finished {paths[0].name} in {took:.0f}s", flush=True)
    else:  # one fresh process per notebook: isolated namespaces, memory released in between
        import subprocess
        for p in paths:
            code = subprocess.call([sys.executable, __file__, str(p)])
            if code != 0:
                sys.exit(code)
