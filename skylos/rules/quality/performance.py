from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

from skylos.rules.base import SkylosRule


LoopNode = ast.For | ast.AsyncFor | ast.While | ast.comprehension
_ORIGIN_WRAPPERS = frozenset(
    {"enumerate", "float", "int", "iter", "list", "reversed", "sorted", "str", "tuple"}
)
_ALIAS_WRAPPERS = frozenset({"float", "int", "str"})
_NON_LIST_CALLS = frozenset(
    {"Counter", "OrderedDict", "bytearray", "bytes", "defaultdict", "deque", "dict", "frozenset", "set"}
)
_TRY_NODE_TYPES = (ast.Try, getattr(ast, "TryStar", ast.Try))


class _Origin(NamedTuple):
    level: int
    keyed: bool


class _LoopContext(NamedTuple):
    node: LoopNode
    fixed: bool
    flattening: bool
    body: list[ast.stmt] | None


def _call_func_name(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _method_call_parts(node):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    ):
        return node.func.value.id, node.func.attr
    return None


def _set_membership(names, name, member):
    if member:
        names.add(name)
    else:
        names.discard(name)


def _is_small_literal_iterable(node):
    if isinstance(node, ast.Dict):
        return len(node.keys) <= 8
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return len(node.elts) <= 8
    return (
        _call_func_name(node) in {"list", "set", "tuple"}
        and len(node.args) == 1
        and not node.keywords
        and _is_small_literal_iterable(node.args[0])
    )


def _is_local_list(node):
    if isinstance(node, ast.List):
        return not _is_small_literal_iterable(node)
    return isinstance(node, ast.ListComp) or (
        _call_func_name(node) == "list"
        and bool(node.args)
        and (
            len(node.args) != 1
            or not isinstance(node.args[0], (ast.Dict, ast.List, ast.Set, ast.Tuple))
        )
    )


def _is_list_extension(node):
    return isinstance(node, (ast.List, ast.ListComp)) or _call_func_name(node) == "list"


def _list_augassign_target(node):
    if (
        isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and isinstance(node.target, ast.Name)
        and _is_list_extension(node.value)
    ):
        return node.target.id
    return None


def _grown_list_name(node):
    parts = _method_call_parts(node)
    if parts is not None and parts[1] in {"append", "extend", "insert"}:
        return parts[0]
    return _list_augassign_target(node)


def _grown_lists(body):
    grown = set()
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Lambda)):
            continue
        name = _grown_list_name(node)
        if name is not None:
            grown.add(name)
        stack.extend(ast.iter_child_nodes(node))
    return grown


def _structured_blocks(node):
    if isinstance(node, (ast.AsyncWith, ast.With)):
        return [node.body]
    if not isinstance(node, _TRY_NODE_TYPES):
        return []
    blocks = [node.body, *(handler.body for handler in node.handlers), node.orelse, node.finalbody]
    return [block for block in blocks if block]


def _wrapped_guard_siblings(node, body):
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
            return [*body[:index], *alternate, *before], [*after, *body[index + 1 :]]
    return None


def _has_substantive_work(body):
    return any(
        not isinstance(statement, (ast.Continue, ast.Pass)) for statement in body
    )


def _only_continues(body):
    return not _has_substantive_work(body) and any(
        isinstance(statement, ast.Continue) for statement in body
    )


def _has_impure_call(node):
    return any(
        isinstance(child, ast.Call) and _call_func_name(child) not in _ORIGIN_WRAPPERS
        for child in ast.walk(node)
    )


def _list_parameters(node):
    parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return {
        parameter.arg
        for parameter in parameters
        if isinstance(parameter.annotation, ast.Subscript)
        and isinstance(parameter.annotation.value, ast.Name)
        and parameter.annotation.value.id == "list"
    }


class _P403Collector(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.findings = []
        self.bindings = {}
        self.loops = []
        self.local_lists = set()
        self.small_literals = set()
        self.non_lists = set()
        self.loop_assigned = set()
        self.selective_generators = set()

    def _visit_scope(self, body, parameter_lists = None):
        saved = (
            self.bindings,
            self.loops,
            self.local_lists,
            self.small_literals,
            self.non_lists,
            self.loop_assigned,
        )
        self.bindings, self.loops = {}, []
        self.local_lists = parameter_lists or set()
        self.small_literals = self.small_literals.copy()
        self.non_lists = self.non_lists.copy()
        self.loop_assigned = set()
        for statement in body:
            self.visit(statement)
        (
            self.bindings,
            self.loops,
            self.local_lists,
            self.small_literals,
            self.non_lists,
            self.loop_assigned,
        ) = saved

    def visit_Module(self, node):
        self._visit_scope(node.body)

    def visit_FunctionDef(self, node):
        self._visit_scope(node.body, _list_parameters(node))

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._visit_scope([node.body])

    def _origin(self, node):
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

    def _subscript_origin(self, node):
        value_origin = self._origin(node.value)
        slice_origin = self._origin(node.slice)
        if value_origin is None:
            return None if slice_origin is None else _Origin(slice_origin.level, True)
        if slice_origin is None or slice_origin.level == value_origin.level:
            return _Origin(value_origin.level, True)
        return None

    def _call_origin(self, node):
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            return None
        return self._origin(node.args[0]) if function_name in _ORIGIN_WRAPPERS else None

    def _composite_origin(self, node):
        origins = [
            self.bindings[child.id]
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in self.bindings
        ]
        return _Origin(max(origin.level for origin in origins), True) if origins else None

    def _track_assignment(self, target, value):
        if not isinstance(target, ast.Name):
            return
        name = target.id
        alias = value.id if isinstance(value, ast.Name) else None
        _set_membership(
            self.local_lists, name, _is_local_list(value) or alias in self.local_lists
        )
        _set_membership(
            self.small_literals,
            name,
            _is_small_literal_iterable(value) or alias in self.small_literals,
        )
        _set_membership(
            self.non_lists,
            name,
            _call_func_name(value) in _NON_LIST_CALLS or alias in self.non_lists,
        )
        _set_membership(self.loop_assigned, name, bool(self.loops))
        origin = self._origin(value)
        if origin is None:
            self.bindings.pop(name, None)
        else:
            self.bindings[name] = origin

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
            self._track_assignment(target, node.value)

    def visit_AnnAssign(self, node):
        self.visit(node.target)
        if node.value is None:
            return
        self.visit(node.value)
        self._track_assignment(node.target, node.value)

    def _in_unbounded_loop(self):
        return any(not loop.fixed for loop in self.loops)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        name = _list_augassign_target(node)
        if name is None:
            return
        self.small_literals.discard(name)
        if _is_local_list(node.value) or self._in_unbounded_loop():
            self.local_lists.add(name)

    def _push_loop(self, node, target, iterable, body):
        old_bindings = self.bindings.copy()
        flattening = self._origin(iterable) is not None or (
            isinstance(iterable, ast.Name) and iterable.id in self.loop_assigned
        )
        fixed = _is_small_literal_iterable(iterable) or (
            isinstance(iterable, ast.Name) and iterable.id in self.small_literals
        )
        self.loops.append(_LoopContext(node, fixed, flattening, body))
        if body is not None and self._in_unbounded_loop():
            self.local_lists.update(_grown_lists(body))
        level = len(self.loops) - 1
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                self.bindings[child.id] = _Origin(level, False)
        return old_bindings

    def visit_For(self, node):
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
    def _indexed_assignment(statement, test_names):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return None
        target = statement.targets[0]
        value = statement.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Subscript):
            return None
        if not isinstance(value.slice, ast.Name) or value.slice.id not in test_names:
            return None
        return statement, value.slice.id, value.value

    def _while_binding(self, node):
        test_names = {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
        for statement in node.body:
            binding = self._indexed_assignment(statement, test_names)
            if binding is not None:
                return binding
        return None

    def visit_While(self, node):
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

    def _matching_comparison(self, test, operator, *, allow_and = True):
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

    def _is_join_comparison(self, comparison):
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

    def _is_alias_expression(self, node):
        return self._origin(node) is not None and all(
            not isinstance(child, ast.Call)
            or (isinstance(child.func, ast.Name) and child.func.id in _ALIAS_WRAPPERS)
            for child in ast.walk(node)
        )

    def _is_alias_statement(self, node):
        return isinstance(node, ast.Pass) or (
            isinstance(node, (ast.AnnAssign, ast.Assign))
            and node.value is not None
            and self._is_alias_expression(node.value)
        )

    def _finding(self, name, simple_name, message, node):
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

    def _add_join_finding(self, node):
        self.findings.append(
            self._finding(
                "nested_loop",
                "for",
                "Nested-loop equality join repeatedly scans the inner iterable. Index inner items by key in a dict before the loop.",
                node,
            )
        )

    @staticmethod
    def _guard_siblings(node, loop):
        if loop.body is None:
            return None
        return _wrapped_guard_siblings(node, loop.body)

    def _only_aliases(self, statements):
        return all(self._is_alias_statement(statement) for statement in statements)

    def _positive_guard(self, node, before, after):
        comparison = self._matching_comparison(node.test, ast.Eq)
        valid = (
            comparison is not None
            and _has_substantive_work(node.body)
            and not _has_substantive_work(node.orelse)
            and self._only_aliases(before)
            and self._only_aliases(after)
        )
        return comparison if valid else None

    def _continue_guard(self, node, before, after):
        comparison = self._matching_comparison(node.test, ast.NotEq, allow_and=False)
        valid = (
            comparison is not None
            and _only_continues(node.body)
            and not node.orelse
            and self._only_aliases(before)
            and _has_substantive_work(after)
        )
        return comparison if valid else None

    def visit_If(self, node):
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

    def _check_join_condition(self, node):
        if len(self.loops) < 2:
            return
        comparison = self._matching_comparison(node, ast.Eq)
        if comparison is not None and self._is_join_comparison(comparison):
            self._add_join_finding(node)

    def _visit_comprehension(self, generators, result_nodes, *, selective_result = False):
        old_bindings = self.bindings.copy()
        pushed = 0
        checkable = not any(
            _has_impure_call(condition)
            for generator in generators
            for condition in generator.ifs
        )
        for generator in generators:
            self.visit(generator.iter)
            self._push_loop(generator, generator.target, generator.iter, None)
            pushed += 1
            for condition in generator.ifs:
                if checkable:
                    self._check_join_condition(condition)
                self.visit(condition)
        for result_node in result_nodes:
            if selective_result and checkable:
                self._check_join_condition(result_node)
            self.visit(result_node)
        for _ in range(pushed):
            self.loops.pop()
        self.bindings = old_bindings

    def visit_ListComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node):
        self._visit_comprehension(node.generators, (node.key, node.value))

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(
            node.generators,
            (node.elt,),
            selective_result=id(node) in self.selective_generators,
        )

    def _add_scan_finding(self, name, operation, node):
        if (
            name not in self.local_lists
            or name in self.non_lists
            or all(loop.fixed for loop in self.loops)
        ):
            return
        messages = {
            "membership": f"Membership checks repeatedly scan locally built list '{name}'. Use a set when duplicates are irrelevant, or maintain counts with collections.Counter.",
            "count": f"'{name}.count(...)' repeatedly scans a locally built list. Maintain counts in a dict or collections.Counter.",
            "index": f"'{name}.index(...)' repeatedly scans a locally built list. Build a value-to-index dict before the loop.",
        }
        self.findings.append(
            self._finding(f"list_{operation}", operation, messages[operation], node)
        )

    def visit_Call(self, node):
        selective_generator = (
            _call_func_name(node) == "any"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
        )
        if selective_generator:
            self.selective_generators.add(id(node.args[0]))
        parts = _method_call_parts(node)
        if parts is not None:
            name, operation = parts
            if operation in {"append", "extend", "insert"}:
                self.small_literals.discard(name)
            if operation == "extend" and node.args and not _is_small_literal_iterable(node.args[0]):
                self.local_lists.add(name)
            elif self.loops and operation in {"append", "insert"} and self._in_unbounded_loop():
                self.local_lists.add(name)
            if self.loops and operation in {"count", "index"}:
                self._add_scan_finding(name, operation, node)
        self.generic_visit(node)
        if selective_generator:
            self.selective_generators.remove(id(node.args[0]))

    def visit_Compare(self, node):
        if self.loops:
            for operator, comparator in zip(node.ops, node.comparators, strict=True):
                if isinstance(operator, (ast.In, ast.NotIn)) and isinstance(comparator, ast.Name):
                    self._add_scan_finding(comparator.id, "membership", node)
        self.generic_visit(node)


def _is_file_read_context(node):
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


def _qualified_name(node):
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _call_chain_has_limiter(node):
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


def _looks_like_orm_query(node):
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


def _is_unbounded_orm_all(node):
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
