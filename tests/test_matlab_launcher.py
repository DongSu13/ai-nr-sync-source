"""Portable entry point must not mutate archived, hash-frozen source files."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))


def test_resolve_matlab_prefers_explicit_bin_over_path(tmp_path):
    from matlab_launcher import resolve_matlab_path

    executable = tmp_path / "custom-matlab"
    executable.write_bytes(b"stub")
    result = resolve_matlab_path({"MATLAB_BIN": str(executable)}, lambda _: "wrong-matlab")
    assert result == executable


def test_resolve_matlab_uses_path_and_reports_missing(tmp_path):
    from matlab_launcher import resolve_matlab_path

    executable = tmp_path / "matlab"
    executable.write_bytes(b"stub")
    assert resolve_matlab_path({}, lambda _: str(executable)) == executable
    with pytest.raises(FileNotFoundError, match="MATLAB_BIN"):
        resolve_matlab_path({}, lambda _: None)


def test_v3_launch_always_uses_one_worker(monkeypatch, tmp_path):
    from matlab_launcher import launch
    import run_nr_v3_pipeline

    calls = []
    original = run_nr_v3_pipeline.MATLAB
    monkeypatch.setattr(run_nr_v3_pipeline, "run", lambda stage, workers: calls.append((stage, workers, run_nr_v3_pipeline.MATLAB)))
    executable = tmp_path / "matlab"
    launch("v3", "development", executable)
    assert calls == [("development", 1, executable)]
    assert run_nr_v3_pipeline.MATLAB == original


def test_cli_routes_to_launch_without_running_matlab(monkeypatch, tmp_path):
    import matlab_launcher

    executable = tmp_path / "matlab"
    calls = []
    monkeypatch.setattr(matlab_launcher, "resolve_matlab_path", lambda: executable)
    monkeypatch.setattr(matlab_launcher, "launch", lambda pipeline, action, path: calls.append((pipeline, action, path)))
    matlab_launcher.main(["v4", "final"])
    assert calls == [("v4", "final", executable)]
