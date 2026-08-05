from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias, cast

from skylos.rules.base import SkylosRule


Finding: TypeAlias = dict[str, str | int]
LoopNode: TypeAlias = ast.For | ast.AsyncFor | ast.While | ast.comprehension
ScanOperation: TypeAlias = Literal["count", "index", "membership"]
_ORIGIN_WRAPPERS = frozenset(
    {"enumerate", "float", "int", "iter", "list", "reversed", "sorted", "str", "tuple"}
)
_ALIAS_WRAPPERS = frozenset({"float", "int", "str"})
_TRY_NODE_TYPES = (ast.Try, getattr(ast, "TryStar", ast.Try))


class _Origin(NamedTuple):
    level: int
    keyed: bool


class _LoopContext(NamedTuple):
    node: LoopNode
    fixed: bool
    flattening: bool
    body: list[ast.stmt] | None


def _is_small_literal_iterable(node: ast.expr) -> bool:
    if isinstance(node, ast.Dict):
        return len(node.keys) <= 8
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return len(node.elts) <= 8
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "set", "tuple"}
        and len(node.args) == 1
        and not node.keywords
        and _is_small_literal_iterable(node.args[0])
    )


def _is_local_list(node: ast.expr) -> bool:
    if isinstance(node, ast.List):
        return not _is_small_literal_iterable(node)
    return isinstance(node, ast.ListComp) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and bool(node.args)
        and (
            len(node.args) != 1
            or not isinstance(node.args[0], (ast.Dict, ast.List, ast.Set, ast.Tuple))
        )
    )


def _is_list_extension(node: ast.expr) -> bool:
    return isinstance(node, (ast.List, ast.ListComp)) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
    )


def _grown_list_name(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.attr in {"append", "extend", "insert"}
    ):
        return node.func.value.id
    if (
        isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and isinstance(node.target, ast.Name)
        and _is_list_extension(node.value)
    ):
        return node.target.id
    return None


def _grown_lists(body: list[ast.stmt]) -> set[str]:
    grown: set[str] = set()
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Lambda)
        ):
            continue
        name = _grown_list_name(node)
        if name is not None:
            grown.add(name)
        stack.extend(ast.iter_child_nodes(node))
    return grown


def _structured_blocks(node: ast.stmt) -> list[list[ast.stmt]]:
    if isinstance(node, (ast.AsyncWith, ast.With)):
        return [node.body]
    if not isinstance(node, _TRY_NODE_TYPES):
        return []
    try_node = cast(ast.Try, node)
    blocks = [
        try_node.body,
        *(handler.body for handler in try_node.handlers),
        try_node.orelse,
        try_node.finalbody,
    ]
    return [block for block in blocks if block]


def _wrapped_guard_siblings(
    node: ast.If, body: list[ast.stmt]
) -> tuple[list[ast.stmt], list[ast.stmt]] | None:
    for index, statement in enumerate(body):
        if statement is node:
            return body[:index], body[index + 1 :]
        blocks = _structured_blocks(statement)
        for block_index, block in enumerate(blocks):
            nested = _wrapped_guard_siblings(node, block)
            if nested is None:
                continue
            before, after = nested
            alternate = [
                sibling
                for sibling_index, sibling_block in enumerate(blocks)
                if sibling_index != block_index
                for sibling in sibling_block
            ]
            return (
                [*body[:index], *alternate, *before],
                [*after, *body[index + 1 :]],
            )
    return None


def _list_parameters(node: ast.AsyncFunctionDef | ast.FunctionDef) -> set[str]:
    parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return {
        parameter.arg
        for parameter in parameters
        if isinstance(parameter.annotation, ast.Subscript)
        and isinstance(parameter.annotation.value, ast.Name)
        and parameter.annotation.value.id == "list"
    }


class _P403Collector(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.findings: list[Finding] = []
        self.bindings: dict[str, _Origin] = {}
        self.loops: list[_LoopContext] = []
        self.local_lists: set[str] = set()
        self.selective_generators: set[int] = set()

    def _visit_scope(
        self, body: list[ast.stmt], parameter_lists: set[str] | None = None
    ) -> None:
        saved = self.bindings, self.loops, self.local_lists
        self.bindings, self.loops = {}, []
        self.local_lists = parameter_lists or set()
        for statement in body:
            self.visit(statement)
        self.bindings, self.loops, self.local_lists = saved

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_scope(node.body)

    def visit_FunctionDef(self, node: ast.AsyncFunctionDef | ast.FunctionDef) -> None:
        self._visit_scope(node.body, _list_parameters(node))

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        saved = self.bindings, self.loops, self.local_lists
        self.bindings, self.loops, self.local_lists = {}, [], set()
        self.visit(node.body)
        self.bindings, self.loops, self.local_lists = saved

    def _origin(self, node: ast.expr) -> _Origin | None:
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            origin = self._origin(node.value)
            return None if origin is None else _Origin(origin.level, True)
        if isinstance(node, ast.Subscript):
            return self._subscript_origin(node)
        if isinstance(node, ast.Call) and len(node.args) == 1:
            return self._call_origin(node)
        if isinstance(node, (ast.BinOp, ast.FormattedValue, ast.JoinedStr)):
            return self._composite_origin(node)
        return None

    def _subscript_origin(self, node: ast.Subscript) -> _Origin | None:
        value_origin = self._origin(node.value)
        slice_origin = self._origin(node.slice)
        if value_origin is None:
            return None if slice_origin is None else _Origin(slice_origin.level, True)
        if slice_origin is None or slice_origin.level == value_origin.level:
            return _Origin(value_origin.level, True)
        return None

    def _call_origin(self, node: ast.Call) -> _Origin | None:
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            return None
        return self._origin(node.args[0]) if function_name in _ORIGIN_WRAPPERS else None

    def _composite_origin(
        self, node: ast.BinOp | ast.FormattedValue | ast.JoinedStr
    ) -> _Origin | None:
        origins = [
            self.bindings[child.id]
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in self.bindings
        ]
        return (
            _Origin(max(origin.level for origin in origins), True) if origins else None
        )

    def _track_assignment(self, target: ast.expr, value: ast.expr) -> None:
        if not isinstance(target, ast.Name):
            return
        if _is_local_list(value) or (
            isinstance(value, ast.Name) and value.id in self.local_lists
        ):
            self.local_lists.add(target.id)
        else:
            self.local_lists.discard(target.id)
        origin = self._origin(value)
        if origin is None:
            self.bindings.pop(target.id, None)
        else:
            self.bindings[target.id] = origin

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
            self._track_assignment(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if node.value is None:
            return
        self.visit(node.value)
        self._track_assignment(node.target, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        if not (
            isinstance(node.op, ast.Add)
            and isinstance(node.target, ast.Name)
            and _is_list_extension(node.value)
        ):
            return
        if _is_local_list(node.value) or any(not loop.fixed for loop in self.loops):
            self.local_lists.add(node.target.id)

    def _push_loop(
        self,
        node: LoopNode,
        target: ast.expr,
        iterable: ast.expr,
        body: list[ast.stmt] | None,
    ) -> dict[str, _Origin]:
        old_bindings = self.bindings.copy()
        flattening = self._origin(iterable) is not None
        self.loops.append(
            _LoopContext(
                node=node,
                fixed=_is_small_literal_iterable(iterable),
                flattening=flattening,
                body=body,
            )
        )
        if body is not None and any(not loop.fixed for loop in self.loops):
            self.local_lists.update(_grown_lists(body))
        level = len(self.loops) - 1
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                self.bindings[child.id] = _Origin(level, False)
        return old_bindings

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        old_bindings = self._push_loop(node, node.target, node.iter, node.body)
        for statement in node.body:
            self.visit(statement)
        self.loops.pop()
        self.bindings = old_bindings
        for statement in node.orelse:
            self.visit(statement)

    visit_AsyncFor = visit_For

    @staticmethod
    def _indexed_assignment(
        statement: ast.stmt, test_names: set[str]
    ) -> tuple[ast.Assign, str, ast.expr] | None:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        value = statement.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Subscript):
            return None
        if not isinstance(value.slice, ast.Name) or value.slice.id not in test_names:
            return None
        return statement, value.slice.id, value.value

    def _while_binding(
        self, node: ast.While
    ) -> tuple[ast.Assign, str, ast.expr] | None:
        test_names = {
            child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)
        }
        for statement in node.body:
            binding = self._indexed_assignment(statement, test_names)
            if binding is not None:
                return binding
        return None

    def visit_While(self, node: ast.While) -> None:
        binding = self._while_binding(node)
        if binding is None:
            self.loops.append(_LoopContext(node, False, False, node.body))
            self.local_lists.update(_grown_lists(node.body))
            self.visit(node.test)
            for statement in node.body:
                self.visit(statement)
            self.loops.pop()
            for statement in node.orelse:
                self.visit(statement)
            return
        assignment, index_name, iterable = binding
        loop_body = [
            statement
            for statement in node.body
            if statement is not assignment
            and not (
                isinstance(statement, ast.AugAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == index_name
            )
        ]
        old_bindings = self._push_loop(node, assignment.targets[0], iterable, loop_body)
        self.visit(node.test)
        for statement in node.body:
            if statement is not assignment:
                self.visit(statement)
        self.loops.pop()
        self.bindings = old_bindings
        for statement in node.orelse:
            self.visit(statement)

    def _matching_comparison(
        self,
        test: ast.expr,
        operator: type[ast.cmpop],
        *,
        allow_and: bool = True,
    ) -> ast.Compare | None:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            if not isinstance(test.operand, ast.Compare):
                return None
            test = test.operand
            operator = ast.NotEq if operator is ast.Eq else ast.Eq
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], operator)
        ):
            return test
        if allow_and and isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            for value in test.values:
                comparison = self._matching_comparison(value, operator)
                if comparison is not None and self._is_join_comparison(comparison):
                    return comparison
        return None

    def _is_join_comparison(self, comparison: ast.Compare) -> bool:
        left = self._origin(comparison.left)
        right = self._origin(comparison.comparators[0])
        if left is None or right is None or left.level == right.level:
            return False
        if len(self.loops) - 1 not in {left.level, right.level}:
            return False
        if not left.keyed and not right.keyed:
            return False
        for level in (left.level, right.level):
            loop = self.loops[level]
            if loop.fixed or loop.flattening:
                return False
        return True

    def _is_alias_expression(self, node: ast.expr) -> bool:
        return self._origin(node) is not None and all(
            not isinstance(child, ast.Call)
            or (isinstance(child.func, ast.Name) and child.func.id in _ALIAS_WRAPPERS)
            for child in ast.walk(node)
        )

    def _is_alias_statement(self, node: ast.stmt) -> bool:
        return isinstance(node, ast.Pass) or (
            isinstance(node, (ast.AnnAssign, ast.Assign))
            and node.value is not None
            and self._is_alias_expression(node.value)
        )

    @staticmethod
    def _has_substantive_work(body: list[ast.stmt]) -> bool:
        return any(
            not isinstance(statement, (ast.Continue, ast.Pass)) for statement in body
        )

    @staticmethod
    def _only_continues(body: list[ast.stmt]) -> bool:
        return all(
            isinstance(statement, (ast.Continue, ast.Pass)) for statement in body
        ) and any(isinstance(statement, ast.Continue) for statement in body)

    def _finding(
        self, name: str, simple_name: str, message: str, node: ast.AST
    ) -> Finding:
        return {
            "rule_id": "SKY-P403",
            "kind": "performance",
            "severity": "LOW",
            "type": "loop",
            "name": name,
            "simple_name": simple_name,
            "value": "O(N^2)",
            "threshold": 0,
            "message": message,
            "file": self.filename,
            "basename": Path(self.filename).name,
            "line": getattr(node, "lineno", 0),
            "col": getattr(node, "col_offset", 0),
        }

    def _add_join_finding(self, node: ast.AST) -> None:
        self.findings.append(
            self._finding(
                "nested_loop",
                "for",
                "Nested-loop equality join repeatedly scans the inner iterable. Index inner items by key in a dict before the loop.",
                node,
            )
        )

    @staticmethod
    def _guard_siblings(
        node: ast.If, loop: _LoopContext
    ) -> tuple[list[ast.stmt], list[ast.stmt]] | None:
        if loop.body is None:
            return None
        return _wrapped_guard_siblings(node, loop.body)

    def _positive_guard(
        self, node: ast.If, before: list[ast.stmt], after: list[ast.stmt]
    ) -> ast.Compare | None:
        comparison = self._matching_comparison(node.test, ast.Eq)
        valid = (
            comparison is not None
            and self._has_substantive_work(node.body)
            and not self._has_substantive_work(node.orelse)
            and all(self._is_alias_statement(statement) for statement in before)
            and all(self._is_alias_statement(statement) for statement in after)
        )
        return comparison if valid else None

    def _continue_guard(
        self, node: ast.If, before: list[ast.stmt], after: list[ast.stmt]
    ) -> ast.Compare | None:
        comparison = self._matching_comparison(node.test, ast.NotEq, allow_and=False)
        valid = (
            comparison is not None
            and self._only_continues(node.body)
            and not node.orelse
            and all(self._is_alias_statement(statement) for statement in before)
            and self._has_substantive_work(after)
        )
        return comparison if valid else None

    def visit_If(self, node: ast.If) -> None:
        if len(self.loops) < 2:
            self.generic_visit(node)
            return
        loop = self.loops[-1]
        siblings = self._guard_siblings(node, loop)
        if siblings is not None:
            before, after = siblings
            comparison = self._positive_guard(node, before, after)
            if comparison is None:
                comparison = self._continue_guard(node, before, after)
            if comparison is not None and self._is_join_comparison(comparison):
                self._add_join_finding(loop.node)
        self.generic_visit(node)

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        result_nodes: tuple[ast.expr, ...],
        *,
        selective_result: bool = False,
    ) -> None:
        old_bindings = self.bindings.copy()
        pushed = 0
        for generator in generators:
            self.visit(generator.iter)
            self._push_loop(generator, generator.target, generator.iter, None)
            pushed += 1
            for condition in generator.ifs:
                if len(self.loops) >= 2:
                    comparison = self._matching_comparison(condition, ast.Eq)
                    if comparison is not None and self._is_join_comparison(comparison):
                        self._add_join_finding(condition)
                self.visit(condition)
        for result_node in result_nodes:
            if selective_result and len(self.loops) >= 2:
                comparison = self._matching_comparison(result_node, ast.Eq)
                if comparison is not None and self._is_join_comparison(comparison):
                    self._add_join_finding(result_node)
            self.visit(result_node)
        for _ in range(pushed):
            self.loops.pop()
        self.bindings = old_bindings

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, (node.key, node.value))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(
            node.generators,
            (node.elt,),
            selective_result=id(node) in self.selective_generators,
        )

    def _add_scan_finding(
        self, name: str, operation: ScanOperation, node: ast.AST
    ) -> None:
        messages = {
            "membership": f"Membership checks repeatedly scan locally built list '{name}'. Use a set when duplicates are irrelevant, or maintain counts with collections.Counter.",
            "count": f"'{name}.count(...)' repeatedly scans a locally built list. Maintain counts in a dict or collections.Counter.",
            "index": f"'{name}.index(...)' repeatedly scans a locally built list. Build a value-to-index dict before the loop.",
        }
        finding = self._finding(
            f"list_{operation}", operation, messages[operation], node
        )
        if name in self.local_lists and not any(loop.fixed for loop in self.loops):
            self.findings.append(finding)

    def visit_Call(self, node: ast.Call) -> None:
        selective_generator = (
            isinstance(node.func, ast.Name)
            and node.func.id == "any"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
        )
        if selective_generator:
            self.selective_generators.add(id(node.args[0]))
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            name = node.func.value.id
            operation = node.func.attr
            if (
                operation == "extend"
                and node.args
                and not _is_small_literal_iterable(node.args[0])
            ):
                self.local_lists.add(name)
            elif (
                self.loops
                and operation in {"append", "insert"}
                and any(not loop.fixed for loop in self.loops)
            ):
                self.local_lists.add(name)
            if self.loops and operation in {"count", "index"}:
                self._add_scan_finding(name, cast(ScanOperation, operation), node)
        self.generic_visit(node)
        if selective_generator:
            self.selective_generators.remove(id(node.args[0]))

    def visit_Compare(self, node: ast.Compare) -> None:
        if self.loops:
            for operator, comparator in zip(node.ops, node.comparators, strict=True):
                if isinstance(operator, (ast.In, ast.NotIn)) and isinstance(
                    comparator, ast.Name
                ):
                    self._add_scan_finding(comparator.id, "membership", node)
        self.generic_visit(node)


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

    def visit_node(self, node, context):
        findings = []

        if isinstance(node, ast.Module) and "SKY-P403" not in self.ignore_list:
            filename = context.get("filename", "")
            collector = _P403Collector(filename if isinstance(filename, str) else "")
            collector.visit(node)
            findings.extend(collector.findings)

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

        return findings if findings else None
