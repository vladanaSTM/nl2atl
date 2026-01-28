#!/usr/bin/env python3
"""Consolidated CLI for nl2atl with subcommands."""

import argparse
import importlib
import sys
from typing import Callable, Dict, Tuple


_COMMANDS: Dict[str, Tuple[str, str]] = {
    "run-all": ("src.cli.run_all_experiments", "main"),
    "run-single": ("src.cli.run_single_experiment", "main"),
    "run-array": ("src.cli.run_experiment_array", "main"),
    "aggregate-seeds": ("src.cli.aggregate_seeds", "main"),
    "llm-judge": ("src.cli.run_llm_judge", "main"),
    "judge-agreement": ("src.cli.run_judge_agreement", "main"),
    "classify-difficulty": ("src.cli.classify_difficulty", "main"),
    "model-efficiency": ("src.cli.run_model_efficiency", "main"),
    "generate-eval-reports": ("src.cli.generate_eval_reports", "main"),
    "slurm": ("src.cli.slurm", "main"),
}


def _load_handler(command: str) -> Callable[[], None]:
    module_name, func_name = _COMMANDS[command]
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _dispatch(command: str, args: list) -> None:
    handler = _load_handler(command)
    sys.argv = [f"nl2atl {command}"] + args
    handler()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nl2atl",
        description="NL2ATL consolidated CLI",
    )
    parser.add_argument("command", choices=sorted(_COMMANDS.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    _dispatch(args.command, args.args)


if __name__ == "__main__":
    main()
