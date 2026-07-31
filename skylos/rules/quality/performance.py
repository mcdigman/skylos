from __future__ import annotations
import ast
from pathlib import Path
from skylos.rules.base import SkylosRule


def _get_loop_target_name(node: ast.For) -> str | None:
    if isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _simple_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _constant_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _constant_int(node.operand)
        if value is not None:
            return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _unwrap_iterable(node: ast.AST) -> ast.AST:
    while (
        isinstance(node, ast.Call)
        and _simple_call_name(node.func) in {"enumerate", "iter", "reversed"}
        and node.args
    ):
        node = node.args[0]
    return node


def _range_stop(node: ast.AST) -> ast.AST | None:
    node = _unwrap_iterable(node)
    if not isinstance(node, ast.Call):
        return None
    if _simple_call_name(node.func) not in {"prange", "range"}:
        return None
    if len(node.args) == 1:
        return node.args[0]
    if 2 <= len(node.args) <= 3:
        return node.args[1]
    return None


def _is_statically_bounded(node: ast.AST) -> bool:
    node = _unwrap_iterable(node)
    if isinstance(node, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, str)):
        return True
    if isinstance(node, ast.Call) and _simple_call_name(node.func) in {
        "prange",
        "range",
    }:
        return bool(node.args) and all(
            _constant_int(arg) is not None for arg in node.args
        )
    return False


def _cardinality_signature(node: ast.AST) -> tuple[str, str]:
    node = _unwrap_iterable(node)
    stop = _range_stop(node)
    if stop is not None:
        if (
            isinstance(stop, ast.Call)
            and _simple_call_name(stop.func) == "len"
            and len(stop.args) == 1
        ):
            return ("collection", ast.dump(stop.args[0], include_attributes=False))
        return ("bound", ast.dump(stop, include_attributes=False))
    return ("collection", ast.dump(node, include_attributes=False))


def _is_outer_index_bound(outer: ast.For, inner: ast.For) -> bool:
    outer_name = _get_loop_target_name(outer)
    outer_stop = _range_stop(outer.iter)
    stop = _range_stop(inner.iter)
    if outer_name is None or outer_stop is None or stop is None:
        return False
    if isinstance(stop, ast.Name):
        return stop.id == outer_name
    if not isinstance(stop, ast.BinOp):
        return False
    if isinstance(stop.left, ast.Name) and stop.left.id == outer_name:
        return (
            isinstance(stop.op, (ast.Add, ast.Sub))
            and _constant_int(stop.right) is not None
        )
    if isinstance(stop.right, ast.Name) and stop.right.id == outer_name:
        return isinstance(stop.op, ast.Add) and _constant_int(stop.left) is not None
    return False


def _inner_iterates_over_outer(outer: ast.For, inner: ast.For) -> bool:
    outer_name = _get_loop_target_name(outer)
    if outer_name is None:
        return False

    inner_iter = inner.iter

    if isinstance(inner_iter, ast.Attribute):
        if isinstance(inner_iter.value, ast.Name) and inner_iter.value.id == outer_name:
            return True

    if isinstance(inner_iter, ast.Subscript):
        if isinstance(inner_iter.value, ast.Name) and inner_iter.value.id == outer_name:
            return True

    if isinstance(inner_iter, ast.Call):
        for arg in inner_iter.args:
            if isinstance(arg, ast.Name) and arg.id == outer_name:
                return True
            if isinstance(arg, ast.Attribute):
                if isinstance(arg.value, ast.Name) and arg.value.id == outer_name:
                    return True

    return False


def _has_same_input_scale(outer: ast.For, inner: ast.For) -> bool:
    if _is_statically_bounded(outer.iter) or _is_statically_bounded(inner.iter):
        return False
    if _inner_iterates_over_outer(outer, inner):
        return False
    if _cardinality_signature(outer.iter) == _cardinality_signature(inner.iter):
        return True
    return _is_outer_index_bound(outer, inner)


def _is_file_read_context(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return True

    receiver = node.func.value

    if isinstance(receiver, ast.Name):
        name = receiver.id.lower()

        non_file_hints = (
            "response",
            "resp",
            "reply",
            "buf",
            "buffer",
            "stringio",
            "bytesio",
            "stream",
            "bio",
            "sio",
            "stdin",
            "stdout",
            "stderr",
        )
        if any(hint in name for hint in non_file_hints):
            return False

    return True


def _qualified_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _call_chain_has_limiter(node: ast.AST) -> bool:
    current = node
    while isinstance(current, ast.Call):
        func = current.func
        if isinstance(func, ast.Attribute) and func.attr in {
            "first",
            "get",
            "head",
            "limit",
            "one",
            "one_or_none",
            "paginate",
            "take",
        }:
            return True
        current = func.value if isinstance(func, ast.Attribute) else func
    if isinstance(current, ast.Attribute):
        return _call_chain_has_limiter(current.value)
    return False


def _looks_like_orm_query(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        if node.attr == "query":
            return True
        return _looks_like_orm_query(node.value)
    if isinstance(node, ast.Call):
        func_name = _qualified_name(node.func)
        if func_name.endswith((".query", ".select")):
            return True
        if isinstance(node.func, ast.Attribute):
            return _looks_like_orm_query(node.func.value)
    return False


def _is_unbounded_orm_all(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "all":
        return False
    if node.args or node.keywords:
        return False
    receiver = node.func.value
    return _looks_like_orm_query(receiver) and not _call_chain_has_limiter(receiver)


class PerformanceRule(SkylosRule):
    rule_id = "SKY-P401"
    name = "Performance Checks"

    def __init__(self, ignore_list=None):
        self.ignore_list = ignore_list or []

    def _is_pandas_read(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "read_csv":
            return True
        return False

    def _find_nested_loops(self, outer: ast.For, body: list[ast.stmt]) -> list[dict]:
        findings_list = []
        for child in body:
            if isinstance(child, ast.For):
                if _has_same_input_scale(outer, child):
                    findings_list.append(child)
            elif isinstance(child, ast.If):
                findings_list.extend(self._find_nested_loops(outer, child.body))
                if child.orelse:
                    findings_list.extend(self._find_nested_loops(outer, child.orelse))
            elif isinstance(child, ast.With):
                findings_list.extend(self._find_nested_loops(outer, child.body))
            elif isinstance(child, ast.Try):
                findings_list.extend(self._find_nested_loops(outer, child.body))
                for handler in child.handlers:
                    findings_list.extend(self._find_nested_loops(outer, handler.body))
                if child.orelse:
                    findings_list.extend(self._find_nested_loops(outer, child.orelse))
                if child.finalbody:
                    findings_list.extend(
                        self._find_nested_loops(outer, child.finalbody)
                    )
        return findings_list

    def visit_node(self, node, context):
        findings = []

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "read",
                "readlines",
            ):
                if "SKY-P401" not in self.ignore_list:
                    if _is_file_read_context(node):
                        findings.append(
                            {
                                "rule_id": "SKY-P401",
                                "kind": "performance",
                                "severity": "LOW",
                                "type": "function",
                                "name": node.func.attr,
                                "simple_name": node.func.attr,
                                "value": "memory_load",
                                "threshold": 0,
                                "message": f"Potential Memory Risk: '{node.func.attr}()' loads entire file into RAM. Consider iterating line-by-line for large files.",
                                "file": context.get("filename"),
                                "basename": Path(context.get("filename", "")).name,
                                "line": node.lineno,
                                "col": node.col_offset,
                            }
                        )

            if self._is_pandas_read(node):
                if "SKY-P402" not in self.ignore_list:
                    has_chunk = False
                    for kw in node.keywords:
                        if kw.arg == "chunksize":
                            has_chunk = True
                            break

                    if not has_chunk:
                        findings.append(
                            {
                                "rule_id": "SKY-P402",
                                "kind": "performance",
                                "severity": "LOW",
                                "type": "function",
                                "name": "read_csv",
                                "simple_name": "read_csv",
                                "value": "no_chunk",
                                "threshold": 0,
                                "message": "Pandas Memory Risk: read_csv used without 'chunksize'. Large files may crash RAM.",
                                "file": context.get("filename"),
                                "basename": Path(context.get("filename", "")).name,
                                "line": node.lineno,
                                "col": node.col_offset,
                            }
                        )

            if _is_unbounded_orm_all(node):
                if "SKY-P404" not in self.ignore_list:
                    findings.append(
                        {
                            "rule_id": "SKY-P404",
                            "kind": "performance",
                            "severity": "MEDIUM",
                            "type": "function",
                            "name": "all",
                            "simple_name": "all",
                            "value": "unbounded_query",
                            "threshold": 0,
                            "message": "Unbounded ORM .all() call may load an entire table. Add a limit, pagination, or streaming boundary.",
                            "file": context.get("filename"),
                            "basename": Path(context.get("filename", "")).name,
                            "line": node.lineno,
                            "col": node.col_offset,
                        }
                    )

        if isinstance(node, ast.For):
            if "SKY-P403" not in self.ignore_list:
                suspect_loops = self._find_nested_loops(node, node.body)
                for inner_loop in suspect_loops:
                    findings.append(
                        {
                            "rule_id": "SKY-P403",
                            "kind": "performance",
                            "severity": "LOW",
                            "type": "loop",
                            "name": "nested_loop",
                            "simple_name": "for",
                            "value": "O(N^2)",
                            "threshold": 0,
                            "message": "Potential Quadratic Work: nested loops appear to scale with the same input. Verify all-pairs work is necessary; for repeated key matching, pre-index the inner collection.",
                            "file": context.get("filename"),
                            "basename": Path(context.get("filename", "")).name,
                            "line": inner_loop.lineno,
                            "col": inner_loop.col_offset,
                        }
                    )

        return findings if findings else None
