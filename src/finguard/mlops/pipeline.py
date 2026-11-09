"""Checks of the DVC pipeline definition (``dvc.yaml``) that run without DVC or the data.

* ``validate()``: the structure DVC accepts (stage and output keys, as in DVC 3's schema), each
  output declared once and never inside another output, every command's module or script
  present, every dependency either in the repository, produced by a stage, or external input
  (the Kaggle files, the downloaded documents), no cycle, and outputs that Git tracks marked
  ``cache: false`` (DVC refuses to cache a file Git already tracks).
* ``status()``: for a machine that already ran the pipeline by hand, which stages have all their
  outputs, so ``dvc commit -f`` can record them in dvc.lock without retraining anything.

    python -m finguard.mlops pipeline            # status and the dvc commands to run
"""
from __future__ import annotations

import importlib.util
import shlex
import subprocess
from pathlib import Path

from finguard import config

DVC_YAML = config.ROOT / "dvc.yaml"
STAGE_KEYS = {"cmd", "wdir", "deps", "params", "vars", "frozen", "meta", "desc", "always_changed", "outs", "metrics",
              "plots", "matrix"}
OUT_KEYS = {"cache", "persist", "checkpoint", "remote", "push", "desc", "type", "labels", "meta"}
PLOT_KEYS = OUT_KEYS | {"template", "x", "y", "x_label", "y_label", "title", "header"}
EXTERNAL = ("data/raw/", "data/corpus/raw")             # inputs that come from outside the repository


def load(path: Path = DVC_YAML) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _entries(stage: dict, key: str) -> list[tuple[str, dict]]:
    out = []
    for e in stage.get(key) or []:
        if isinstance(e, str):
            out.append((e, {}))
        else:
            (path, props), = e.items()
            out.append((path, props or {}))
    return out


def outputs(stage: dict) -> list[tuple[str, dict, str]]:
    return [(p, props, k) for k in ("outs", "metrics", "plots") for p, props in _entries(stage, k)]


def _under(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def git_tracked(root: Path = config.ROOT) -> set[str] | None:
    try:
        p = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return set(p.stdout.splitlines()) if p.returncode == 0 and p.stdout else None


def validate(doc: dict | None = None, root: Path = config.ROOT, tracked: set[str] | None = None) -> list[str]:
    """Problems that would make DVC refuse the file or run stages in a wrong order (empty = fine)."""
    doc = doc if doc is not None else load(root / "dvc.yaml")
    stages = doc.get("stages") or {}
    problems: list[str] = []
    owner: dict[str, str] = {}
    for name, st in stages.items():
        for k in set(st) - STAGE_KEYS:
            problems.append(f"{name}: unknown key '{k}'")
        if "cmd" not in st:
            problems.append(f"{name}: no cmd")
        if not all(isinstance(d, str) for d in st.get("deps") or []):
            problems.append(f"{name}: deps must be plain paths")
        for path, props, kind in outputs(st):
            allowed = PLOT_KEYS if kind == "plots" else OUT_KEYS
            for k in set(props) - allowed:
                problems.append(f"{name}: output {path}: unknown key '{k}'")
            if path in owner:
                problems.append(f"{path}: output of both {owner[path]} and {name}")
            owner[path] = name
    paths = sorted(owner)
    for a in paths:
        for b in paths:
            if a != b and _under(b, a):
                problems.append(f"overlapping outputs: {b} ({owner[b]}) is inside {a} ({owner[a]})")
    # commands: modules and scripts exist
    for name, st in stages.items():
        cmds = st["cmd"] if isinstance(st.get("cmd"), list) else [st.get("cmd", "")]
        for c in cmds:
            words = shlex.split(c)
            if words[:2] == ["python", "-m"] and len(words) > 2:
                try:
                    found = importlib.util.find_spec(words[2]) is not None
                except ModuleNotFoundError:
                    found = False
                if not found:
                    problems.append(f"{name}: module {words[2]} not found")
            elif words[:1] == ["python"] and len(words) > 1 and not (root / words[1]).exists():
                problems.append(f"{name}: script {words[1]} not found")
    # dependencies: in the repository, produced by a stage, or external
    def produced_by(d: str) -> list[str]:
        return [s for p, s in owner.items() if _under(d, p) or _under(p, d)]

    edges: dict[str, set[str]] = {n: set() for n in stages}
    for name, st in stages.items():
        for d in st.get("deps") or []:
            makers = [s for s in produced_by(d) if s != name]
            edges[name] |= set(makers)
            if not makers and not d.startswith(EXTERNAL) and not (root / d).exists():
                problems.append(f"{name}: dependency {d} is not in the repository and no stage produces it")
        for p in st.get("params") or []:
            files = [p] if isinstance(p, str) else list(p)
            for f in files:
                if f.endswith((".yaml", ".yml", ".json", ".toml", ".py")) and not (root / f).exists():
                    problems.append(f"{name}: params file {f} not found")
    # cycles (depth-first search over "stage depends on stage")
    state: dict[str, int] = {}

    def visit(n: str, trail: list[str]) -> None:
        state[n] = 1
        for m in edges[n]:
            if state.get(m) == 1:
                problems.append("cycle: " + " -> ".join(trail + [n, m]))
            elif not state.get(m):
                visit(m, trail + [n])
        state[n] = 2

    for n in stages:
        if not state.get(n):
            visit(n, [])
    # Git-tracked outputs cannot be cached by DVC
    tracked = tracked if tracked is not None else git_tracked(root)
    if tracked:
        for path, props, _ in outputs_all(stages):
            if props.get("cache", True) and any(_under(t, path) for t in tracked):
                problems.append(f"{path} is tracked by Git: declare it with 'cache: false'")
    return problems


def outputs_all(stages: dict) -> list[tuple[str, dict, str]]:
    return [o for st in stages.values() for o in outputs(st)]


def status(root: Path = config.ROOT) -> list[dict]:
    """Per stage: missing dependencies and outputs on this machine."""
    doc = load(root / "dvc.yaml")
    rows = []
    for name, st in (doc.get("stages") or {}).items():
        miss_out = [p for p, _, _ in outputs(st) if not (root / p).exists()]
        miss_dep = [d for d in st.get("deps") or [] if not (root / d).exists()]
        rows.append({"stage": name, "outputs": len(outputs(st)), "missing_outputs": miss_out, "missing_deps": miss_dep,
                     "ready": not miss_out and not miss_dep})
    return rows


def report(root: Path = config.ROOT, log=print) -> int:
    problems = validate(root=root)
    log(f"dvc.yaml: {len(load(root / 'dvc.yaml')['stages'])} stages, "
        + ("structure OK" if not problems else f"{len(problems)} problem(s):"))
    for p in problems:
        log(f"  - {p}")
    rows = status(root)
    log("\nstage              outputs  on this machine")
    for r in rows:
        if r["ready"]:
            state = "all present"
        else:
            miss = r["missing_outputs"] + [f"(dep) {d}" for d in r["missing_deps"]]
            state = f"missing {len(miss)}: " + ", ".join(miss[:3]) + (" ..." if len(miss) > 3 else "")
        log(f"  {r['stage']:<17} {r['outputs']:>4}    {state}")
    ready = [r["stage"] for r in rows if r["ready"]]
    if not (root / ".dvc").exists():
        log("\nDVC is not initialised here yet: dvc init")
    if ready:
        log("\nrecord the existing outputs in dvc.lock (nothing is recomputed):")
        log("  dvc commit -f " + (" ".join(ready) if len(ready) < len(rows) else ""))
        log("then `dvc status` lists what is out of date and `dvc repro` reruns only that.")
    return 1 if problems else 0
