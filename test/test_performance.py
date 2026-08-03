import unittest
import ast
from skylos.rules.quality.performance import PerformanceRule


class TestPerformanceRule(unittest.TestCase):
    def _analyze(self, source_code, ignore_list=None):
        tree = ast.parse(source_code)
        rule = PerformanceRule(ignore_list=ignore_list)
        context = {"filename": "test_perf.py"}
        all_findings = []

        for node in ast.walk(tree):
            res = rule.visit_node(node, context)
            if res:
                all_findings.extend(res)

        return all_findings

    def test_detect_file_read_memory_risk(self):
        code = """
def process(f):
    content = f.read()
    lines = f.readlines()
    other = f.readline()
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0]["rule_id"], "SKY-P401")
        self.assertEqual(findings[1]["rule_id"], "SKY-P401")

    def test_detect_pandas_no_chunk(self):
        code = """
import pandas as pd
df = pd.read_csv("large_file.csv") # Bad
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule_id"], "SKY-P402")

    def test_allow_pandas_with_chunk(self):
        code = """
import pandas as pd
df = pd.read_csv("large_file.csv", chunksize=1000) # Good
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 0)

    def _p403(self, source_code, ignore_list=None):
        return [
            finding
            for finding in self._analyze(source_code, ignore_list=ignore_list)
            if finding["rule_id"] == "SKY-P403"
        ]

    def test_detect_cross_collection_join(self):
        code = """
def match(users, orders):
    for user in users:
        for order in orders:
            if user.id == order.user_id:
                consume(user, order)
"""
        findings = self._p403(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["severity"], "MEDIUM")
        self.assertEqual(findings[0]["value"], "join_scan")
        self.assertEqual(findings[0]["line"], 4)
        self.assertIn("user.id == order.user_id", findings[0]["message"])

    def test_detect_same_collection_pair_match(self):
        code = """
def duplicates(objects):
    for first in objects:
        for second in objects:
            if first == second:
                consume(first)
"""
        self.assertEqual(len(self._p403(code)), 1)

    def test_detect_join_through_index_loops(self):
        code = """
def match(rows, others):
    for i in range(len(rows)):
        for j in range(len(others)):
            if rows[i].key == others[j].key:
                consume(i, j)
"""
        self.assertEqual(len(self._p403(code)), 1)

    def test_detect_join_in_comprehension_and_generator(self):
        code = """
def paired(users, orders):
    return [(u, o) for u in users for o in orders if u.id == o.user_id]

def matched(users, orders):
    return any(u.id == o.user_id for u in users for o in orders)
"""
        self.assertEqual(len(self._p403(code)), 2)

    def test_detect_join_under_while_and_through_branches(self):
        code = """
def drain(users, orders):
    i = 0
    while i < len(users):
        user = users[i]
        if user:
            try:
                for order in orders:
                    if user.id == order.user_id:
                        consume(user, order)
            except KeyError:
                pass
        i += 1
"""
        self.assertEqual(len(self._p403(code)), 1)

    def test_detect_linear_scan_of_growing_list(self):
        code = """
def unique(values):
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
"""
        findings = self._p403(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["value"], "linear_scan")
        self.assertIn("seen", findings[0]["message"])

    def test_detect_linear_scan_via_list_methods(self):
        code = """
def positions(values, raw):
    ordered = sorted(raw)
    for value in values:
        consume(ordered.index(value))
"""
        self.assertEqual(len(self._p403(code)), 1)

    def test_allow_all_pairs_work_without_a_lookup(self):
        code = """
def pairwise(points, n):
    for i in range(n):
        for j in range(i + 1, n):
            consume(points[i], points[j])

def grid(values, n):
    for row in range(n):
        for col in range(n):
            values[row, col] = row + col

def rectangle(n_rows, n_cols):
    for row in range(n_rows):
        for col in range(n_cols):
            consume(row, col)

def axes(values):
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            values[row, col] = 0
"""
        self.assertEqual(self._p403(code), [])

    def test_allow_bare_index_comparison(self):
        code = """
def skip_self(points, n):
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            consume(points[i], points[j])
"""
        self.assertEqual(self._p403(code), [])

    def test_allow_iteration_over_the_outer_element(self):
        code = """
def flatten(matrix):
    for row in matrix:
        for value in row:
            consume(value)

def self_edges(graph):
    for source, targets in graph.items():
        for target in sorted(targets):
            if source == target:
                consume(source)
"""
        self.assertEqual(self._p403(code), [])

    def test_allow_small_literal_collections(self):
        code = """
def fixed(values):
    for value in values:
        for flag in (1, 2, 3):
            if value.flag == flag:
                consume(value)
"""
        self.assertEqual(self._p403(code), [])

    def test_allow_fixed_lookup_tables(self):
        code = """
ORDER = ["low", "high"]

def rank(rows):
    local = ["low", "high"]
    for row in rows:
        consume(local.index(row.severity))
        if row.severity in ORDER:
            consume(row)
"""
        self.assertEqual(self._p403(code), [])

    def test_allow_set_membership(self):
        code = """
def filtered(values, raw):
    allowed = set(raw)
    for value in values:
        if value in allowed:
            consume(value)

def rebound(values, raw):
    allowed = list(raw)
    allowed = set(allowed)
    for value in values:
        if value in allowed:
            consume(value)
"""
        self.assertEqual(self._p403(code), [])

    def test_allow_loop_invariant_lookup(self):
        code = """
def constant_probe(values, raw, key):
    ordered = sorted(raw)
    for value in values:
        if key in ordered:
            consume(value)
"""
        self.assertEqual(self._p403(code), [])

    def test_allow_matches_no_index_can_replace(self):
        code = """
def substring(values, others):
    for value in values:
        for other in others:
            if value.tag in other.tags:
                consume(value, other)

def prefixes(modules, packages):
    for module in modules:
        for package in packages:
            if module.startswith(package + ".") or module == package:
                consume(module)

def scanner(quotes, text):
    for quote in quotes:
        index = 0
        while index < len(text):
            char = text[index]
            if char == quote:
                consume(char)
            index += 1
"""
        self.assertEqual(self._p403(code), [])

    def test_rewrites_that_hide_the_shape_still_report(self):
        rewrites = [
            "for user in users:\n        for order in list(orders):",
            "for user in users:\n        for order in sorted(orders):",
            "for user in users:\n        for index, order in enumerate(orders):",
        ]
        for header in rewrites:
            code = f"""
def match(users, orders):
    {header}
            if user.id == order.user_id:
                consume(user, order)
"""
            with self.subTest(header=header):
                self.assertEqual(len(self._p403(code)), 1)

    def test_hard_coded_bounds_do_not_suppress(self):
        code = """
def match(rows, others):
    for i in range(10000):
        for j in range(10000):
            if rows[i].key == others[j].key:
                consume(i, j)
"""
        self.assertEqual(len(self._p403(code)), 1)

    def test_allow_short_constant_and_literal_ranges(self):
        code = """
def signed(rows, others):
    for i in range(-2, 2):
        for j in range(6, 0, -1):
            if rows[i].key == others[j].key:
                consume(i, j)

def letters(values):
    for value in values:
        for char in "abc":
            if value.tag == char:
                consume(value)
"""
        self.assertEqual(self._p403(code), [])

    def test_detect_join_in_dict_comprehension_and_prange(self):
        code = """
def indexed(users, orders):
    return {u.id: o for u in users for o in orders if u.id == o.user_id}

def kernel(rows, others, n):
    for i in prange(n):
        for j in prange(n):
            if rows[i].key == others[j].key:
                consume(i, j)
"""
        self.assertEqual(len(self._p403(code)), 2)

    def test_detect_join_in_async_loops(self):
        code = """
async def match(users, orders):
    async for user in users:
        async for order in orders:
            if user.id == order.user_id:
                consume(user, order)
"""
        self.assertEqual(len(self._p403(code)), 1)

    def test_detect_linear_scan_of_annotated_and_concatenated_list(self):
        code = """
def collect(values, raw):
    seen: list = []
    seen += raw
    for value in values:
        if value not in seen:
            consume(value)
"""
        self.assertEqual(len(self._p403(code)), 1)

    def test_allow_nested_function_scopes_independently(self):
        code = """
def outer(users, orders):
    for user in users:

        def inner():
            for order in orders:
                if user.id == order.user_id:
                    consume(order)

        inner()
"""
        self.assertEqual(self._p403(code), [])

    def test_reports_each_nest_once(self):
        code = """
def deep(groups, users, orders):
    for group in groups:
        for user in users:
            for order in orders:
                if user.id == order.user_id:
                    consume(group, user, order)
"""
        findings = self._p403(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["line"], 5)

    def test_detect_unbounded_orm_all(self):
        code = """
def list_users():
    return User.query.all()
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule_id"], "SKY-P404")

    def test_detect_unbounded_sqlalchemy_session_query_all(self):
        code = """
def list_users(db):
    return db.session.query(User).all()
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule_id"], "SKY-P404")

    def test_allow_limited_orm_all(self):
        code = """
def list_users():
    return User.query.limit(100).all()
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 0)

    def test_allow_django_queryset_all_because_it_is_lazy(self):
        code = """
def get_queryset():
    return User.objects.all()

def filtered_queryset():
    return User.objects.filter(active=True).all()
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 0)

    def test_ignore_list_logic(self):
        code = """
def ignore_me(f, users, orders):
    f.read()
    for user in users:
        for order in orders:
            if user.id == order.user_id:
                pass
"""
        findings = self._analyze(code, ignore_list=[])
        self.assertEqual(len(findings), 2)

        findings = self._analyze(code, ignore_list=["SKY-P401"])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule_id"], "SKY-P403")

        findings = self._analyze(code, ignore_list=["SKY-P403"])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule_id"], "SKY-P401")


if __name__ == "__main__":
    unittest.main()
