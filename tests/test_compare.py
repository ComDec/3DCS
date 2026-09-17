"""Tests for reproduce/compare.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reproduce" / "compare.py"


def _load():
    spec = importlib.util.spec_from_file_location("compare", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXPECTED = """table,model,metric,variant,paper_value,expected_value,tolerance,notes
1,gemnet,spearman,paper,0.560,0.559754,0.001,
1,gemnet,kendall,paper,0.336,0.335809,,default tolerance
1,unimol,spearman,paper,0.697,0.697176,0.001,embeddings not available
1,gemnet,lie,v2,,,0.001,pending
"""


def _write(tmp_path, results: str):
    exp = tmp_path / "expected.csv"
    res = tmp_path / "results.csv"
    exp.write_text(EXPECTED)
    res.write_text(results)
    return exp, res


def test_pass_missing_noexpected(tmp_path):
    exp, res = _write(
        tmp_path, "table,model,metric,variant,value\n1,gemnet,spearman,paper,0.5601\n1,gemnet,kendall,paper,0.3350\n"
    )
    mod = _load()
    code = mod.main(["--expected", str(exp), "--results", str(res)])
    assert code == 0
    report, _ = mod.compare(mod._read(exp, mod.KEY), mod._read(res, mod.KEY))
    assert [r["status"] for r in report] == ["PASS", "PASS", "MISSING", "NO_EXPECTED"]
    assert mod.main(["--expected", str(exp), "--results", str(res), "--strict"]) == 1


def test_fail_exit_code_cli(tmp_path):
    exp, res = _write(tmp_path, "table,model,metric,variant,value\n1,gemnet,spearman,paper,0.57\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--expected", str(exp), "--results", str(res)], capture_output=True, text=True
    )
    assert proc.returncode == 1
    assert "FAIL" in proc.stdout


def test_nothing_compared(tmp_path):
    exp, res = _write(tmp_path, "table,model,metric,variant,value\n9,x,y,z,1.0\n")
    assert _load().main(["--expected", str(exp), "--results", str(res)]) == 2


def test_nan_value_fails(tmp_path):
    exp, res = _write(tmp_path, "table,model,metric,variant,value\n1,gemnet,spearman,paper,nan\n")
    assert _load().main(["--expected", str(exp), "--results", str(res)]) == 1
