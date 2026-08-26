from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from skylos.analysis.ast_cache import MODE_STRICT, load_python_module


@dataclass(frozen=True)
class ParsedPythonFile:
    path: Path
    tree: ast.Module


def parse_python_files(files: Iterable[Path]) -> list[ParsedPythonFile]:
    parsed: list[ParsedPythonFile] = []
    for path in files:
        _, tree = load_python_module(path, MODE_STRICT)
        if tree is None:
            continue
        parsed.append(ParsedPythonFile(path=path, tree=tree))
    return parsed
