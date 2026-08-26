"""Run-scoped cache of parsed Python module ASTs.

Repo-wide rule passes (phantom refs, API hallucination coverage, API
signature checks, dead-code liveness) each need the same files parsed.
This cache lets them share one tree per file while preserving each call
site's read semantics: sources are cached per (path, read mode), and a
tree is parsed once per distinct source text, so call sites with
different read modes still share a tree whenever the decoded source is
identical (the common case of a regular, valid-UTF-8 file).

The cache is cleared at the start of each analyzer run (and again when
the run finishes, to release memory). Derived per-tree caches can
register a clear callback so their id(tree)-keyed entries never outlive
the trees stored here.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

from skylos.core.safe_cache_io import read_text_no_symlink

# Read modes. Each mirrors the exact read semantics of the call sites it
# replaces; adding a mode must not change an existing one.
MODE_REPLACE = "replace"  # read_text(encoding="utf-8", errors="replace")
MODE_STRICT = "strict"  # read_text(encoding="utf-8")
MODE_SAFE_REPLACE_2MB = "safe_replace_2mb"  # no-symlink read, 2 MiB cap
MODE_SAFE_IGNORE_1MB = "safe_ignore_1mb"  # no-symlink read, 1 MiB cap

_SAFE_MODE_ARGS = {
    MODE_SAFE_REPLACE_2MB: {"max_bytes": 2 * 1024 * 1024, "errors": "replace"},
    MODE_SAFE_IGNORE_1MB: {"max_bytes": 1_000_000, "errors": "ignore"},
}

_MISSING = object()

_sources: dict[tuple[str, str], str | None] = {}
_trees: dict[str, dict[str, ast.Module | None]] = {}
_dependent_clears: list[Callable[[], None]] = []


def _read_source(path: Path, mode: str) -> str | None:
    if mode == MODE_REPLACE:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    if mode == MODE_STRICT:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
    safe_args = _SAFE_MODE_ARGS[mode]
    return read_text_no_symlink(path, encoding="utf-8", **safe_args)


def load_python_source(path: Path, mode: str) -> str | None:
    """Return the decoded source for ``path`` under ``mode``, or None."""
    key = (str(path), mode)
    source = _sources.get(key, _MISSING)
    if source is _MISSING:
        source = _read_source(Path(path), mode)
        _sources[key] = source
    return source


def load_python_module(path: Path, mode: str) -> tuple[str | None, ast.Module | None]:
    """Return ``(source, tree)`` for ``path`` under read mode ``mode``.

    ``source`` is None when the file cannot be read under that mode;
    ``tree`` is None when the file cannot be read or does not parse.
    """
    source = load_python_source(path, mode)
    if source is None:
        return None, None
    per_path = _trees.setdefault(str(path), {})
    tree = per_path.get(source, _MISSING)
    if tree is _MISSING:
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            tree = None
        per_path[source] = tree
    return source, tree


def register_dependent_clear(callback: Callable[[], None]) -> None:
    """Register a callback run whenever this cache is cleared.

    Used by caches keyed on id(tree) for trees held here, so their
    entries are dropped before the trees can be garbage collected.
    """
    if callback not in _dependent_clears:
        _dependent_clears.append(callback)


def clear_python_ast_cache() -> None:
    for callback in _dependent_clears:
        callback()
    _sources.clear()
    _trees.clear()
