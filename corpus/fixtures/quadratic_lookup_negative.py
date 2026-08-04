"""Nested loops that no pre-built index removes.

Every loop here is labelled negative for SKY-P403. They fall into three groups:
work that is inherently pairwise, equalities that do not decide which pairs get
the work, and lookups whose container is fixed or already hashed.
"""

from collections.abc import Iterable
from typing import Any

SEVERITY_ORDER = ["low", "medium", "high"]


def pairwise_distance(points: list, n: int) -> float:
    """Triangular all-pairs work. Every pair genuinely has to be visited."""
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(points[i] - points[j])
    return total


def fill_square_grid(values: Any, n: int) -> None:
    """Element count is n**2, so touching each element is linear in the output."""
    for row in range(n):
        for col in range(n):
            values[row, col] = row + col


def walk_rectangular_dimensions(n_rows: int, n_cols: int) -> list:
    return [(row, col) for row in range(n_rows) for col in range(n_cols)]


def flatten_rows(matrix: Iterable[Iterable[Any]]) -> list:
    """The inner loop walks what the outer loop handed it: linear overall."""
    out = []
    for row in matrix:
        for value in row:
            out.append(value)
    return out


def skip_self_then_compare_all(entries: dict) -> list:
    """The equality removes one pair; the rest of the work is still pairwise."""
    out = []
    for name, info in entries.items():
        for other_name, other_info in entries.items():
            if other_name == name:
                continue
            out.append(info.score - other_info.score)
    return out


def tally_matches_while_scoring_every_pair(users: Iterable, orders: Iterable) -> int:
    """Every pair is scored regardless, so indexing the equality changes nothing."""
    hits = 0
    for user in users:
        for order in orders:
            score(user, order)
            if user.id == order.user_id:
                hits += 1
    return hits


def prefix_or_exact_match(modules: Iterable[str], packages: Iterable[str]) -> list:
    """The prefix test still has to try every package."""
    out = []
    for module in modules:
        for package in packages:
            if module.startswith(package + ".") or module == package:
                out.append(module)
    return out


def compare_against_a_shared_threshold(
    users: Iterable, orders: Iterable, limit: int
) -> list:
    """The `==` relates `limit` to the inner value, not the two loop values."""
    out = []
    for user in users:
        for order in orders:
            if user.rank < limit == order.rank:
                out.append((user, order))
    return out


def rank_against_a_fixed_table(rows: Iterable) -> list:
    """A frozen literal is a lookup table, not a collection that scales."""
    return [SEVERITY_ORDER.index(row.severity) for row in rows]


def membership_against_a_set(values: Iterable, raw: Iterable) -> list:
    allowed = set(raw)
    return [value for value in values if value in allowed]


def defer_the_lookup_to_a_callback(values: Iterable, raw: Iterable) -> list:
    """The nested body runs when the handler is called, not once per iteration."""
    ordered = list(raw)
    handlers = []
    for value in values:

        def later(value=value):
            return value in ordered

        handlers.append(later)
    return handlers


def score(user: Any, order: Any) -> float:
    return 0.0
