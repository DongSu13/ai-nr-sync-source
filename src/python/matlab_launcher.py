"""Portable, single-worker entry point for MATLAB campaigns."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil
import sys
from typing import Callable, Mapping


def resolve_matlab_path(
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> Path:
    environment = os.environ if env is None else env
    configured = environment.get("MATLAB_BIN")
    candidate = configured if configured else which("matlab")
    if not candidate or not Path(candidate).is_file():
        raise FileNotFoundError(
            "MATLAB executable not found; set MATLAB_BIN to its path or add matlab to PATH"
        )
    return Path(candidate)


def launch(pipeline: str, action: str, matlab_path: Path) -> object:
    modules = {"v3": "run_nr_v3_pipeline", "v4": "run_nr_v4_pipeline"}
    allowed = {"v3": {"development", "final", "all"},
               "v4": {"development", "final", "all"}}
    if pipeline not in modules or action not in allowed[pipeline]:
        raise ValueError(f"unsupported pipeline action: {pipeline} {action}")
    module = importlib.import_module(modules[pipeline])
    original_matlab = module.MATLAB
    original_argv = sys.argv
    module.MATLAB = Path(matlab_path)
    try:
        if pipeline == "v3":
            return module.run(action, 1)
        if pipeline == "v4":
            sys.argv = [str(module.__file__), action]
            return module.main()
    finally:
        module.MATLAB = original_matlab
        sys.argv = original_argv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", choices=("v3", "v4"))
    parser.add_argument("action", help="development|final|all")
    args = parser.parse_args(argv)
    if args.action not in ({"v3": {"development", "final", "all"},
                           "v4": {"development", "final", "all"}}[args.pipeline]):
        parser.error(f"invalid action {args.action!r} for {args.pipeline}")
    launch(args.pipeline, args.action, resolve_matlab_path())


if __name__ == "__main__":
    main()
