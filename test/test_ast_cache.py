import json
import os

import pytest

from skylos.analysis import ast_cache
from skylos.analysis.ast_cache import (
    MODE_IGNORE,
    MODE_REPLACE,
    MODE_SAFE_IGNORE_1MB,
    MODE_SAFE_REPLACE_2MB,
    MODE_STRICT,
    clear_python_ast_cache,
    load_python_module,
    load_python_source,
    mode_max_bytes,
    register_dependent_clear,
)
from skylos.analyzer import Skylos
from skylos.rules.ai_defect.api_signature_hallucination import (
    MAX_PYTHON_API_SIGNATURE_SOURCE_BYTES,
)
from skylos.rules.ai_defect.python_api_hallucination import (
    scan_python_local_api_hallucinations,
)

ALL_MODES = (
    MODE_REPLACE,
    MODE_STRICT,
    MODE_IGNORE,
    MODE_SAFE_REPLACE_2MB,
    MODE_SAFE_IGNORE_1MB,
)


@pytest.fixture(autouse=True)
def _isolated_cache():
    clear_python_ast_cache()
    yield
    clear_python_ast_cache()


def _write(path, source):
    path.write_text(  # skylos: ignore[SKY-D324] pytest tmp_path fixture
        source,
        encoding="utf-8",
    )
    return path


def test_every_read_mode_shares_one_source_object_and_one_tree(tmp_path):
    path = _write(tmp_path / "module.py", "x = 1\n")

    results = [load_python_module(path, mode) for mode in ALL_MODES]

    assert len({id(source) for source, _ in results}) == 1
    assert len({id(tree) for _, tree in results}) == 1
    assert len(ast_cache._trees[str(path)]) == 1


def test_safe_modes_refuse_symlinks_that_plain_modes_follow(tmp_path):
    target = _write(tmp_path / "real.py", "x = 1\n")
    link = tmp_path / "link.py"
    link.symlink_to(target)

    assert load_python_source(link, MODE_REPLACE) == "x = 1\n"
    assert load_python_source(link, MODE_SAFE_REPLACE_2MB) is None


def test_edited_file_is_reread_even_when_the_size_is_unchanged(tmp_path):
    path = _write(tmp_path / "module.py", "x = 1\n")
    first_source, first_tree = load_python_module(path, MODE_REPLACE)
    assert first_source == "x = 1\n"

    _write(path, "x = 9\n")
    os.utime(path, ns=(0, 0))

    second_source, second_tree = load_python_module(path, MODE_REPLACE)
    assert second_source == "x = 9\n"
    assert second_tree is not first_tree


def test_editing_a_file_runs_registered_dependent_clears(tmp_path):
    calls = []

    def _record():
        calls.append(1)

    register_dependent_clear(_record)
    try:
        path = _write(tmp_path / "module.py", "x = 1\n")
        load_python_module(path, MODE_REPLACE)
        calls.clear()

        _write(path, "x = 1\ny = 2\n")
        load_python_module(path, MODE_REPLACE)

        assert calls
    finally:
        ast_cache._dependent_clears.remove(_record)


def test_unreadable_and_unparsable_files_return_none(tmp_path):
    unparsable = _write(tmp_path / "broken.py", "def (:\n")
    source, tree = load_python_module(unparsable, MODE_IGNORE)
    assert source == "def (:\n"
    assert tree is None

    assert load_python_module(tmp_path / "absent.py", MODE_REPLACE) == (None, None)


def test_clear_drops_every_cache(tmp_path):
    path = _write(tmp_path / "module.py", "x = 1\n")
    load_python_module(path, MODE_REPLACE)
    assert ast_cache._sources

    clear_python_ast_cache()

    assert not ast_cache._sources
    assert not ast_cache._trees
    assert not ast_cache._source_pool
    assert not ast_cache._path_tokens


def test_mode_max_bytes_is_the_documented_cap():
    assert mode_max_bytes(MODE_SAFE_IGNORE_1MB) == 1_000_000
    assert mode_max_bytes(MODE_SAFE_IGNORE_1MB) == MAX_PYTHON_API_SIGNATURE_SOURCE_BYTES
    assert mode_max_bytes(MODE_SAFE_REPLACE_2MB) == 2 * 1024 * 1024
    assert mode_max_bytes(MODE_REPLACE) is None


def test_repeated_rule_scans_see_edits_made_between_them(tmp_path):
    lib = _write(tmp_path / "lib.py", "def real():\n    return 1\n")
    app = _write(tmp_path / "app.py", "import lib\n\nlib.missing()\n")
    files = [lib, app]

    findings, _ = scan_python_local_api_hallucinations(
        tmp_path, files, target_files=files
    )
    assert [finding["simple_name"] for finding in findings] == ["missing"]

    _write(lib, "def real():\n    return 1\n\n\ndef missing():\n    return 2\n")

    findings, _ = scan_python_local_api_hallucinations(
        tmp_path, files, target_files=files
    )
    assert findings == []


def test_analyze_releases_the_run_scoped_cache(tmp_path):
    _write(tmp_path / "module.py", "import os\n\n\ndef f():\n    return os.path\n")

    Skylos().analyze(str(tmp_path), enable_ai_defects=True)

    assert not ast_cache._sources
    assert not ast_cache._trees


def test_one_unparsable_file_does_not_suppress_phantom_calls_elsewhere(tmp_path):
    _write(tmp_path / "a_broken.py", "def (:\n")
    _write(
        tmp_path / "b_phantom.py",
        "def process(data):\n    clean = sanitize_input(data)\n    return clean\n",
    )

    result = json.loads(Skylos().analyze(str(tmp_path), enable_ai_defects=True))

    rule_ids = {finding.get("rule_id") for finding in result.get("ai_defects", [])}
    assert "SKY-L012" in rule_ids
