"""Tests for CLI argument parsing and help output."""
from __future__ import annotations

import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "three_dbench", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "3DBench CLI" in result.stdout


def test_cli_convert_help():
    result = subprocess.run(
        [sys.executable, "-m", "three_dbench", "convert", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "chirality" in result.stdout


def test_cli_evaluate_help():
    result = subprocess.run(
        [sys.executable, "-m", "three_dbench", "evaluate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "chirality" in result.stdout
    assert "rotation" in result.stdout
    assert "traj" in result.stdout
