import unittest
import ast
from pathlib import Path

from skylos.rules.quality.performance import PerformanceRule


class TestPerformanceRule(unittest.TestCase):
    def test_module_dispatch_is_registered(self):
        path = (
            Path(__file__).parent.parent / "skylos" / "analysis" / "file_processing.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        registered: set[str] = set()
        for mapping in (node for node in ast.walk(tree) if isinstance(node, ast.Dict)):
            for key, value in zip(mapping.keys, mapping.values, strict=True):
                if isinstance(key, ast.Name) and key.id == "PerformanceRule":
                    registered.update(
                        item.attr
                        for item in ast.walk(value)
                        if isinstance(item, ast.Attribute)
                    )
        self.assertIn("Module", registered)

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

    def test_detect_nested_loop_join_variants(self):
        cases = {
            "direct": """
for user in users:
    for order in orders:
        if user.id == order.user_id:
            emit(user, order)
""",
            "inverse guard and aliases": """
for user in users:
    user_key: int = user.id
    for order in orders:
        order_key: int = order.user_id
        if user_key != order_key:
            continue
        emit(user, order)
""",
            "async loops": """
async def match():
    async for user in users:
        async for order in orders:
            if user.id == order.user_id:
                await emit(user, order)
""",
            "destructured targets": """
for _, user in users:
    for _, order in orders:
        if user.id == order.user_id:
            emit(user, order)
""",
            "double-negative guard": """
for user in users:
    for order in orders:
        if not user.id != order.user_id:
            emit(user, order)
""",
            "harmless else": """
for user in users:
    for order in orders:
        if user.id == order.user_id:
            emit(user, order)
        else:
            pass
""",
            "indexed while": """
for user in users:
    index = 0
    while index < len(orders):
        order = orders[index]
        if user.id == order.user_id:
            emit(user, order)
        index += 1
""",
            "with guard": """
for user in users:
    for order in orders:
        with lock:
            if user.id == order.user_id:
                emit(user, order)
""",
            "try guard": """
for user in users:
    for order in orders:
        try:
            if user.id == order.user_id:
                emit(user, order)
        except LookupError:
            pass
""",
        }
        for label, code in cases.items():
            with self.subTest(label):
                findings = self._analyze(code)
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["name"], "nested_loop")
                self.assertIn("dict", findings[0]["message"])

    def test_detect_comprehension_join_variants(self):
        cases = {
            "list comprehension": """
pairs = [
    (user, order)
    for user in users
    for order in orders
    if user.id == order.user_id
]
""",
            "generator expression": """
pairs = (
    (user, order)
    for user in users
    for order in orders
    if user.id == order.user_id
)
""",
            "set comprehension": """
pairs = {
    (user, order)
    for user in users
    for order in orders
    if user.id == order.user_id
}
""",
            "dict comprehension": """
pairs = {
    user.id: order
    for user in users
    for order in orders
    if user.id == order.user_id
}
""",
            "comparison result": """
matched = any(
    user.id == order.user_id
    for user in users
    for order in orders
)
""",
            "formatted outer key": """
paths = [
    next(path for path in artifacts if path.name == f"{group}_{seed}")
    for group in groups
    for seed in seeds
]
""",
        }
        for label, code in cases.items():
            with self.subTest(label):
                findings = self._analyze(code)
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["name"], "nested_loop")

    def test_ignore_noisy_nested_loop_shapes(self):
        cases = {
            "no equality join": """
for left in lefts:
    for right in rights:
        compare(left, right)
""",
            "incidental equality": """
for left in lefts:
    for right in rights:
        same = left.id == right.id
        compare(left, right, same)
""",
            "required all pairs": """
for left in lefts:
    for right in rights:
        compare(left, right)
        if left.id == right.id:
            remember(left, right)
""",
            "work assigned before guard": """
for left in lefts:
    for right in rights:
        score = expensive_score(left, right)
        if left.id == right.id:
            remember(score)
""",
            "or predicate": """
for left in lefts:
    for right in rights:
        if left.id == right.id or left.fallback == right.fallback:
            remember(left, right)
""",
            "non-equivalent inverse guard": """
for left in lefts:
    for right in rights:
        if left.id != right.id and right.active:
            continue
        remember(left, right)
""",
            "negated compound predicate": """
for left in lefts:
    for right in rights:
        if not (left.id != right.id and right.active):
            remember(left, right)
""",
            "bare indices": """
for i in left_indices:
    for j in right_indices:
        if i == j:
            remember(i, j)
""",
            "flatten outer element": """
for parent in parents:
    for child in iter(parent.children):
        if parent.id == child.parent_id:
            remember(parent, child)
""",
            "matching continue": """
for left in lefts:
    for right in rights:
        if left.id == right.id:
            continue
""",
            "deferred definition": """
for left in lefts:
    def later():
        for right in rights:
            if left.id == right.id:
                remember(left, right)
""",
            "deferred lambda": """
for left in lefts:
    later = lambda: (
        right for right in rights if left.id == right.id
    )
""",
            "fixed literal iterables": """
for left in [first, second]:
    for right in [third, fourth]:
        if left.id == right.id:
            remember(left, right)
""",
            "work beside wrapped guard": """
for left in lefts:
    for right in rights:
        with lock:
            compare(left, right)
            if left.id == right.id:
                remember(left, right)
""",
            "work in exception handler": """
for left in lefts:
    for right in rights:
        try:
            if left.id == right.id:
                remember(left, right)
        except LookupError:
            recover(left, right)
""",
        }
        for label, code in cases.items():
            with self.subTest(label):
                self.assertEqual(self._analyze(code), [])

    def test_detect_repeated_local_list_scans(self):
        code = """
values = list(source)
for item in items:
    if item in values:
        emit(item)
    values.count(item)
    values.index(item)
"""
        findings = self._analyze(code)
        self.assertEqual(
            [finding["name"] for finding in findings],
            ["list_membership", "list_count", "list_index"],
        )
        self.assertIn("set", findings[0]["message"])
        self.assertIn("duplicates", findings[0]["message"])
        self.assertIn("Counter", findings[1]["message"])
        self.assertIn("value-to-index dict", findings[2]["message"])

    def test_detect_list_grown_in_loop(self):
        cases = {
            "append": """
seen = []
for item in items:
    if item in seen:
        emit(item)
    seen.append(item)
""",
            "augmented assignment": """
seen = []
for item in items:
    if item in seen:
        emit(item)
    seen += [item]
""",
        }
        for label, code in cases.items():
            with self.subTest(label):
                findings = self._analyze(code)
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["name"], "list_membership")

    def test_detect_list_scan_in_plain_while(self):
        code = """
queue = list(start)
while queue:
    item = queue.pop()
    if item in queue:
        emit(item)
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["name"], "list_membership")

    def test_detect_list_scan_in_assignment_target(self):
        cases = {
            "assignment": "output[values.index(item)] = item",
            "annotated assignment": "output[values.index(item)]: str = item",
        }
        for label, assignment in cases.items():
            with self.subTest(label):
                code = f"""
values = list(source)
for item in items:
    {assignment}
"""
                findings = self._analyze(code)
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0]["name"], "list_index")

    def test_detect_annotated_list_parameter_scan(self):
        code = """
def unique(items, seen: list[str]):
    for item in items:
        if item not in seen:
            seen.append(item)
"""
        findings = self._analyze(code)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["name"], "list_membership")

    def test_ignore_fixed_lookup_tables_and_remove(self):
        code = """
fixed = ["alpha", "beta", "gamma"]
also_fixed = list(("alpha", "beta", "gamma"))
dynamic = list(source)
for item in items:
    if item in fixed:
        emit(item)
    if item in also_fixed:
        emit(item)
    dynamic.remove(item)
"""
        self.assertEqual(self._analyze(code), [])

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
def ignore_me(f):
    f.read()
    for item in items:
        for other in others:
            if item.id == other.item_id:
                emit(item, other)
"""
        findings = self._analyze(code, ignore_list=[])
        self.assertEqual(len(findings), 2)

        findings = self._analyze(code, ignore_list=["SKY-P401"])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule_id"], "SKY-P403")


if __name__ == "__main__":
    unittest.main()
