"""Quadratic lookups that a pre-built index removes.

Every loop here is labelled positive for SKY-P403: each one repeats a lookup
whose result a dict, set or Counter built once would answer directly.
"""

from collections.abc import Iterable
from typing import Any


def join_two_collections(users: Iterable[Any], orders: Iterable[Any]) -> list:
    """The plain join: index orders by user_id and the inner loop disappears."""
    matched = []
    for user in users:
        for order in orders:
            if user.id == order.user_id:
                matched.append((user, order))
    return matched


def join_written_as_a_skip(users: Iterable[Any], orders: Iterable[Any]) -> list:
    """The same join spelled as an early continue."""
    matched = []
    for user in users:
        for order in orders:
            if user.id != order.user_id:
                continue
            matched.append((user, order))
    return matched


def join_in_a_comprehension(users: Iterable[Any], orders: Iterable[Any]) -> list:
    return [
        (user, order) for user in users for order in orders if user.id == order.user_id
    ]


def join_as_a_generator_element(users: Iterable[Any], orders: Iterable[Any]) -> bool:
    return any(user.id == order.user_id for user in users for order in orders)


def membership_against_a_growing_list(values: Iterable[Any]) -> list:
    """The canonical accidental quadratic: keep a set beside `seen`."""
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def position_lookup_per_iteration(values: Iterable[Any], raw: Iterable[Any]) -> list:
    """`index` needs a value-to-first-position map, not a set."""
    ordered = sorted(raw)
    return [ordered.index(value) for value in values]


def frequency_lookup_per_iteration(values: Iterable[Any], raw: Iterable[Any]) -> list:
    """`count` needs a Counter, not a set."""
    ordered = list(raw)
    return [ordered.count(value) for value in values]
