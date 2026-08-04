from __future__ import annotations
import ast
from pathlib import Path
from skylos.rules.base import SkylosRule


_COMPREHENSIONS = (ast.DictComp, ast.GeneratorExp, ast.ListComp, ast.SetComp)
_INDEX_ITERATORS = frozenset({"prange", "range"})
_ITERABLE_WRAPPERS = frozenset(
    {"enumerate", "iter", "list", "reversed", "sorted", "tuple"}
)
# Equality only. ``x in y`` is a substring test as often as a membership test,
# and no index replaces a substring search, so membership is handled by the
# narrower linear-scan check where the container is known to be a list.
_JOIN_OPS = (ast.Eq,)
_MEMBERSHIP_OPS = (ast.In, ast.NotIn)
# `remove` is deliberately absent: a repeated `list.remove` in a loop is a
# question about mutation during iteration, and no set-shaped rewrite preserves
# its order, duplicate and ValueError semantics.
_SCAN_METHODS = frozenset({"count", "index"})
_SCAN_REMEDIES = {
    "count": (
        "counts occurrences in the list '{name}' on every iteration. Build a "
        "collections.Counter over '{name}' once, then read the count directly."
    ),
    "index": (
        "scans the list '{name}' for a position on every iteration. Build a "
        "dict of value to first index over '{name}' once, then look the "
        "position up directly."
    ),
    "membership": (
        "rescans the list '{name}' on every iteration. Track the same contents "
        "in a set beside '{name}' and test that instead."
    ),
}
# Bodies that run when they are called, not where they are written. A loop that
# only defines one of these does not run its contents per iteration.
_SCOPES = (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda)
_SEQUENCE_BUILDERS = frozenset({"list", "sorted", "tuple"})
_SEQUENCE_GROWERS = frozenset({"append", "extend", "insert"})
_SMALL_ITERABLE_LIMIT = 8


def _simple_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _constant_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            return None
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _constant_int(node.operand)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _unwrap_iterable(node: ast.AST) -> ast.AST:
    """Strip wrappers that do not change how many items an iteration visits."""
    while (
        isinstance(node, ast.Call)
        and _simple_call_name(node.func) in _ITERABLE_WRAPPERS
        and node.args
    ):
        node = node.args[0]
    return node


def _is_index_iterator(node: ast.AST | None) -> bool:
    if node is None:
        return False
    node = _unwrap_iterable(node)
    return (
        isinstance(node, ast.Call) and _simple_call_name(node.func) in _INDEX_ITERATORS
    )


def _constant_range_length(node: ast.Call) -> int | None:
    bounds = [_constant_int(arg) for arg in node.args]
    if not bounds or any(bound is None for bound in bounds):
        return None
    if len(bounds) == 1:
        return max(0, bounds[0])
    step = bounds[2] if len(bounds) == 3 else 1
    if step == 0:
        return None
    return max(0, -(-(bounds[1] - bounds[0]) // step))


def _is_small_iterable(node: ast.AST | None) -> bool:
    """Does this iterable visibly hold only a handful of items?

    Deliberately narrow: a literal display, a short string, or a fully constant
    ``range`` that is actually short. ``range(10000)`` is constant but not
    small, so replacing a parameter with a hard-coded bound never silences a
    finding.
    """
    if node is None:
        return False
    node = _unwrap_iterable(node)
    if isinstance(node, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
        elements = node.keys if isinstance(node, ast.Dict) else node.elts
        return len(elements) <= _SMALL_ITERABLE_LIMIT
    if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, str)):
        return len(node.value) <= _SMALL_ITERABLE_LIMIT
    if isinstance(node, ast.Call) and _simple_call_name(node.func) in _INDEX_ITERATORS:
        length = _constant_range_length(node)
        return length is not None and length <= _SMALL_ITERABLE_LIMIT
    return False


def _names_in(node: ast.AST) -> frozenset[str]:
    return frozenset(
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    )


def _assigned_names(body: list) -> frozenset[str]:
    """Names rebound by assignment in ``body``.

    Loop targets are deliberately excluded: they belong to the nested loop that
    binds them, not to the enclosing block.
    """
    names: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            targets: list = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                names.update(_names_in(target))
    return frozenset(names)


class _Loop:
    """A loop candidate: what it binds, what it iterates, and what it runs."""

    __slots__ = ("body", "ifs", "index_like", "iterable", "names", "node", "results")

    def __init__(
        self,
        node: ast.AST,
        names: frozenset[str],
        iterable,
        body: list,
        ifs: list | None = None,
        results: list | None = None,
    ):
        self.node = node
        self.names = names
        self.iterable = iterable
        self.body = body
        # Filters and element expressions of a comprehension clause. Both are
        # empty for statement loops, whose guards are `if` statements found by
        # walking the body instead.
        self.ifs = ifs or []
        self.results = results or []
        self.index_like = _is_index_iterator(iterable)

    @property
    def is_comprehension(self) -> bool:
        return bool(self.ifs or self.results)


def _as_loop(node: ast.AST) -> _Loop | None:
    if isinstance(node, (ast.AsyncFor, ast.For)):
        return _Loop(node, _names_in(node.target), node.iter, list(node.body))
    if isinstance(node, ast.While):
        # A while loop has no target, so the values it advances are whatever it
        # rebinds each pass.
        return _Loop(node, _assigned_names(node.body), None, list(node.body))
    return None


def _comprehension_loops(node: ast.AST) -> list[_Loop]:
    """Model each comprehension clause as a loop over the clauses after it."""
    if not isinstance(node, _COMPREHENSIONS):
        return []
    results = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
    loops = []
    for position, clause in enumerate(node.generators):
        body: list = list(clause.ifs)
        for later in node.generators[position + 1 :]:
            body.append(later.iter)
            body.extend(later.ifs)
        body.extend(results)
        loops.append(
            _Loop(
                node,
                _names_in(clause.target),
                clause.iter,
                body,
                list(clause.ifs),
                results,
            )
        )
    return loops


def _is_join_key(operand: ast.AST, names: frozenset[str], index_like: bool) -> bool:
    """Does ``operand`` read a value produced by a loop over ``names``?

    A bare counter from ``for i in range(n)`` does not qualify. Comparing two
    counters cannot be replaced by a lookup table, so the operand has to
    dereference the counter, as in ``rows[i].key``.
    """
    if not _names_in(operand) & names:
        return False
    return not (index_like and isinstance(operand, ast.Name))


def _iterates_over_outer(outer: _Loop, inner: _Loop) -> bool:
    """Is the inner loop walking a value the outer loop handed it?

    ``for row in rows: for cell in row`` visits each cell once overall, so it is
    linear in the flattened input rather than quadratic in either loop.
    """
    return bool(_names_in(_unwrap_iterable(inner.iterable)) & outer.names)


def _is_side(
    operand: ast.AST, own: frozenset[str], other: frozenset[str], index_like: bool
) -> bool:
    return not (_names_in(operand) & other) and _is_join_key(operand, own, index_like)


def _compare_pair(
    node: ast.Compare, op_type, outer: _Loop, inner: _Loop, outer_names: frozenset[str]
) -> ast.Compare | None:
    """Does ``node`` compare an outer value directly against an inner value?

    Chained comparisons are checked pair by pair, so ``a.rank < limit ==
    b.rank`` does not count: the ``==`` there relates ``limit`` to the inner
    value, and no index over the inner collection removes the ``<``.
    """
    operands = [node.left, *node.comparators]
    for position, op in enumerate(node.ops):
        if not isinstance(op, op_type):
            continue
        left, right = operands[position], operands[position + 1]
        for first, second in ((left, right), (right, left)):
            if _is_side(first, outer_names, inner.names, outer.index_like) and _is_side(
                second, inner.names, outer_names, inner.index_like
            ):
                return node
    return None


def _selecting_equality(
    test: ast.AST, outer: _Loop, inner: _Loop, outer_names: frozenset[str]
) -> ast.Compare | None:
    """An equality that decides, on its own, which pairs get the work.

    ``and`` is fine - the surviving terms still filter an indexed bucket. ``or``
    is not: ``module.startswith(package + ".") or module == package`` still has
    to try every package, so indexing the equality buys nothing.
    """
    if isinstance(test, ast.BoolOp):
        if isinstance(test.op, ast.Or):
            return None
        for value in test.values:
            found = _selecting_equality(value, outer, inner, outer_names)
            if found is not None:
                return found
        return None
    if isinstance(test, ast.Compare):
        return _compare_pair(test, ast.Eq, outer, inner, outer_names)
    return None


def _selecting_inequality(
    test: ast.AST, outer: _Loop, inner: _Loop, outer_names: frozenset[str]
) -> ast.Compare | None:
    """The inverted spelling: skip the non-matches, then do the work.

    Only a bare ``!=`` or ``not (... == ...)`` counts. Anything compound would
    change which pairs survive the skip, so it is left alone.
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        if isinstance(test.operand, ast.Compare):
            return _compare_pair(test.operand, ast.Eq, outer, inner, outer_names)
        return None
    if isinstance(test, ast.Compare):
        return _compare_pair(test, ast.NotEq, outer, inner, outer_names)
    return None


def _is_substantive(body: list) -> bool:
    """Does this branch do anything beyond falling through to the next pair?"""
    return any(
        not isinstance(statement, (ast.Continue, ast.Pass)) for statement in body
    )


def _is_skip(body: list) -> bool:
    return len(body) == 1 and isinstance(body[0], ast.Continue)


def _does_pair_work(statements: list, inner: _Loop) -> bool:
    """Is there work here that runs whether or not the pair matched?

    Assignments are excluded: deriving a value from the inner element is setup
    for the match, not the work the match is supposed to be guarding.
    """
    return any(
        _names_in(statement) & inner.names
        and not isinstance(statement, (ast.AnnAssign, ast.Assign))
        for statement in statements
    )


def _guarded_statement_lists(statement: ast.stmt):
    """Statement lists that run under the same pair as their parent."""
    if isinstance(statement, ast.If):
        yield statement.body
        yield statement.orelse
    elif isinstance(statement, (ast.AsyncWith, ast.With)):
        yield statement.body
    elif isinstance(statement, ast.Try):
        yield statement.body
        yield statement.orelse
        yield statement.finalbody
        for handler in statement.handlers:
            yield handler.body


def _scan_statements(
    statements: list, outer: _Loop, inner: _Loop, outer_names: frozenset[str]
) -> ast.Compare | None:
    for position, statement in enumerate(statements):
        if isinstance(statement, ast.If) and not statement.orelse:
            siblings = statements[:position] + statements[position + 1 :]
            equality = _selecting_equality(statement.test, outer, inner, outer_names)
            if (
                equality is not None
                and _is_substantive(statement.body)
                and not _does_pair_work(siblings, inner)
            ):
                return equality

            inequality = _selecting_inequality(
                statement.test, outer, inner, outer_names
            )
            if (
                inequality is not None
                and _is_skip(statement.body)
                and _does_pair_work(statements[position + 1 :], inner)
                and not _does_pair_work(statements[:position], inner)
            ):
                return inequality

        for nested in _guarded_statement_lists(statement):
            found = _scan_statements(nested, outer, inner, outer_names)
            if found is not None:
                return found
    return None


def _join_evidence(outer: _Loop, inner: _Loop) -> ast.Compare | None:
    """Find the equality that explains why the nesting exists.

    It is not enough for a qualifying ``==`` to appear somewhere in the inner
    body. The match has to decide which pairs get the work, because that is the
    thing a dict or set built once from the inner collection replaces. An
    equality that merely skips one pair, or that tallies a count while every
    pair is processed anyway, leaves the pairwise work exactly where it was.
    """
    if inner.iterable is None:
        # A while loop has no collection to pre-index.
        return None
    if _iterates_over_outer(outer, inner):
        return None
    outer_names = outer.names - inner.names
    if inner.is_comprehension:
        # A comprehension clause selects per pair rather than branching, so its
        # filters are the guard. The element counts too when it *is* the
        # comparison, as in `any(a.key == b.key for a in xs for b in ys)`;
        # an element that merely contains one somewhere does not.
        for test in [*inner.ifs, *inner.results]:
            found = _selecting_equality(test, outer, inner, outer_names)
            if found is not None:
                return found
        return None
    return _scan_statements(inner.body, outer, inner, outer_names)


def _walk_body(statements: list):
    """Yield nodes reachable from these statements without entering a new scope."""
    for statement in statements:
        if not isinstance(statement, _SCOPES):
            yield from _walk_scope(statement)


def _walk_scope(node: ast.AST):
    """Yield every node in this scope, stopping at nested function scopes."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if not isinstance(child, _SCOPES):
                stack.append(child)


def _collect_loops(node: ast.AST, enclosing: tuple, found: list) -> None:
    if isinstance(node, _SCOPES):
        return

    chain = enclosing
    for comprehension_loop in _comprehension_loops(node):
        found.append((comprehension_loop, chain))
        chain += (comprehension_loop,)

    loop = _as_loop(node)
    if loop is not None:
        found.append((loop, enclosing))
        chain = enclosing + (loop,)
        # The iterable and the while condition are evaluated outside the loop.
        outside = node.iter if isinstance(node, (ast.AsyncFor, ast.For)) else node.test
        _collect_loops(outside, enclosing, found)
        for child in [*node.body, *node.orelse]:
            _collect_loops(child, chain, found)
        return

    for child in ast.iter_child_nodes(node):
        _collect_loops(child, chain, found)


def _scope_loops(scope: ast.AST) -> list[tuple[_Loop, tuple]]:
    """Every loop in this scope, paired with its enclosing loops outermost first."""
    found: list = []
    for child in ast.iter_child_nodes(scope):
        _collect_loops(child, (), found)
    return found


def _sequence_source(node: ast.AST | None) -> bool:
    if isinstance(node, (ast.List, ast.ListComp, ast.Tuple)):
        return True
    return (
        isinstance(node, ast.Call)
        and _simple_call_name(node.func) in _SEQUENCE_BUILDERS
    )


def _scannable_sequences(scope: ast.AST) -> frozenset[str]:
    """Names holding a list whose membership test is a linear scan.

    A name qualifies only when every binding of it is list-like, so a name later
    rebound to a ``set`` is excluded, and when it is either built at runtime or
    grown in place. A frozen literal such as ``ORDER = ["low", "high"]`` is a
    fixed lookup table rather than a collection that scales, so it is left
    alone.
    """
    list_like: dict[str, bool] = {}
    grown: set[str] = set()
    built: set[str] = set()

    for node in _walk_scope(scope):
        targets: list = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                grown.add(node.target.id)
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            sequence_like = _sequence_source(node.value)
            list_like[target.id] = list_like.get(target.id, True) and sequence_like
            if sequence_like and not isinstance(node.value, (ast.List, ast.Tuple)):
                built.add(target.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SEQUENCE_GROWERS
            and isinstance(node.func.value, ast.Name)
        ):
            grown.add(node.func.value.id)

    return frozenset(
        name
        for name, sequence_like in list_like.items()
        if sequence_like and not name.isupper() and (name in grown or name in built)
    )


def _linear_scans(
    loop: _Loop, sequences: frozenset[str]
) -> list[tuple[ast.AST, str, str]]:
    """Find lookups inside ``loop`` that rescan a list on every iteration.

    Each kind is reported separately because each needs a different replacement:
    a set answers membership, a ``Counter`` answers ``count``, and a position
    map answers ``index``. Suggesting one remedy for all three would propose a
    rewrite that does not preserve behaviour.
    """
    found = []
    for node in _walk_body(loop.body):
        if isinstance(node, ast.Compare) and len(node.comparators) == 1:
            container = node.comparators[0]
            if (
                any(isinstance(op, _MEMBERSHIP_OPS) for op in node.ops)
                and isinstance(container, ast.Name)
                and container.id in sequences
                and _names_in(node.left) & loop.names
            ):
                found.append((node, container.id, "membership"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _SCAN_METHODS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in sequences
            and any(_names_in(arg) & loop.names for arg in node.args)
        ):
            found.append((node, node.func.value.id, node.func.attr))
    return found


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

    def _quadratic_finding(self, node, context, message: str, value: str) -> dict:
        return {
            "rule_id": "SKY-P403",
            "kind": "performance",
            "severity": "LOW",
            "type": "loop",
            "name": "quadratic_lookup",
            "simple_name": "for",
            "value": value,
            "threshold": 0,
            "message": message,
            "file": context.get("filename"),
            "basename": Path(context.get("filename", "")).name,
            "line": node.lineno,
            "col": node.col_offset,
        }

    def _find_quadratic_lookups(self, scope, context) -> list[dict]:
        """Report loops whose repeated lookups a pre-built index would remove."""
        loops = _scope_loops(scope)
        if not loops:
            return []

        findings = []
        sequences = _scannable_sequences(scope)

        for loop, enclosing in loops:
            for outer in reversed(enclosing):
                # The nearest enclosing loop that explains the match wins, so a
                # deep nest reports once rather than once per level.
                if _is_small_iterable(outer.iterable) or _is_small_iterable(
                    loop.iterable
                ):
                    continue
                evidence = _join_evidence(outer, loop)
                if evidence is None:
                    continue
                findings.append(
                    self._quadratic_finding(
                        loop.node,
                        context,
                        "Quadratic Lookup: this nested loop rescans the inner "
                        f"collection to match '{ast.unparse(evidence)}'. Build a "
                        "dict or set from the inner collection once before the "
                        "outer loop, then look the match up directly.",
                        "join_scan",
                    )
                )
                break

            for lookup, container, kind in _linear_scans(loop, sequences):
                remedy = _SCAN_REMEDIES[kind].format(name=container)
                findings.append(
                    self._quadratic_finding(
                        lookup,
                        context,
                        f"Quadratic Lookup: '{ast.unparse(lookup)}' {remedy}",
                        f"linear_scan_{kind}",
                    )
                )

        return findings

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

        # SKY-P403 runs once per scope so that a loop nest is judged as a whole
        # and each finding is reported exactly once.
        if isinstance(node, (*_SCOPES, ast.Module)):
            if "SKY-P403" not in self.ignore_list:
                findings.extend(self._find_quadratic_lookups(node, context))

        return findings if findings else None
