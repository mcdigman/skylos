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

    def test_detect_nested_loops(self):
        code = """
def heavy(n):
    for i in range(n):
        print(i)
        for j in range(n):
            print(j)
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule_id"], "SKY-P403")
        self.assertNotIn("itertools", findings[0]["message"])

    def test_detect_triangular_same_scale_loops(self):
        code = """
def pairwise(n):
    for i in range(n):
        for j in range(i + 1, n):
            consume(i, j)

def lower_triangle(n):
    for row in range(n):
        for col in range(row + 1):
            consume(row, col)
"""
        findings = self._analyze(code)
        self.assertEqual(
            [finding["rule_id"] for finding in findings],
            ["SKY-P403", "SKY-P403"],
        )

    def test_allow_statically_bounded_nested_loops(self):
        code = """
def fixed_grid():
    for i in range(10):
        for j in range(10):
            consume(i, j)

def fixed_cases():
    for left in [1, 2, 3]:
        for right in (4, 5, 6):
            consume(left, right)
"""
        findings = self._analyze(code)
        self.assertFalse(any(finding["rule_id"] == "SKY-P403" for finding in findings))

    def test_allow_distinct_dimensions_and_array_axes(self):
        code = """
def rectangular(n_rows, n_cols, values):
    for row in range(n_rows):
        for col in range(n_cols):
            consume(row, col)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            values[row, col] = row + col
"""
        findings = self._analyze(code)
        self.assertFalse(any(finding["rule_id"] == "SKY-P403" for finding in findings))

    def test_allow_iteration_over_outer_element(self):
        code = """
def flatten(matrix):
    for row in matrix:
        for value in row:
            consume(value)

def expand_limits(limits):
    for limit in limits:
        for value in range(limit + 1):
            consume(value)
"""
        findings = self._analyze(code)
        self.assertFalse(any(finding["rule_id"] == "SKY-P403" for finding in findings))

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
def ignore_me(f, n):
    f.read()
    for i in range(n):
        for j in range(n):
            pass
"""
        findings = self._analyze(code, ignore_list=[])
        self.assertEqual(len(findings), 2)

        findings = self._analyze(code, ignore_list=["SKY-P401"])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule_id"], "SKY-P403")


if __name__ == "__main__":
    unittest.main()
