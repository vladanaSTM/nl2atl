#!/usr/bin/env python3
"""Submit nl2atl jobs to SLURM from the consolidated CLI."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PARTITION = "A100"
DEFAULT_GRES = "gpu:1"
DEFAULT_CPUS_PER_TASK = 8
DEFAULT_MEM = "32G"
DEFAULT_TIME = "06:00:00"


def _normalize_extra_args(extra_args: List[str]) -> List[str]:
    return [arg for arg in extra_args if arg != "--"]


def _render_command(module: str, extra_args: List[str]) -> str:
    args = " ".join(shlex.quote(arg) for arg in extra_args)
    if args:
        return f'"$PYTHON_BIN" -m {module} {args}'
    return f'"$PYTHON_BIN" -m {module}'


def _render_sbatch_script(
    *,
    job_name: str,
    partition: str,
    nodes: int,
    gres: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    output_path: str,
    error_path: str,
    array_range: Optional[str],
    logs_dir: str,
    python_bin: str,
    repo_root: Path,
    command: str,
    sbatch_args: List[str],
) -> str:
    lines = ["#!/usr/bin/env bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --partition={partition}")
    lines.append(f"#SBATCH --nodes={nodes}")
    lines.append(f"#SBATCH --gres={gres}")
    lines.append(f"#SBATCH --cpus-per-task={cpus_per_task}")
    lines.append(f"#SBATCH --mem={mem}")
    lines.append(f"#SBATCH --time={time_limit}")
    lines.append(f"#SBATCH --output={output_path}")
    lines.append(f"#SBATCH --error={error_path}")
    if array_range:
        lines.append(f"#SBATCH --array={array_range}")
    for extra in sbatch_args:
        extra = extra.strip()
        if not extra:
            continue
        if extra.startswith("#SBATCH"):
            lines.append(extra)
        else:
            lines.append(f"#SBATCH {extra}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(logs_dir)}",
            f"PYTHON_BIN=${{PYTHON_BIN:-{shlex.quote(python_bin)}}}",
            f"REPO_ROOT=${{REPO_ROOT:-{shlex.quote(str(repo_root))}}}",
            'export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"',
            'cd "$REPO_ROOT"',
            "",
            command,
            "",
        ]
    )

    return "\n".join(lines)


def _compute_array_range(extra_args: List[str], repo_root: Path) -> str:
    cmd = [
        sys.executable,
        "-m",
        "src.cli.run_experiment_array",
        "--count",
        *extra_args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip().splitlines()
    if not output:
        raise ValueError("Unable to resolve task count for array submission.")
    count = int(output[-1])
    if count <= 0:
        raise ValueError("Task count resolved to 0; check filters.")
    return f"0-{count - 1}"


def _write_script(script: str, path: Optional[Path]) -> Path:
    if path is None:
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch")
        tmp.write(script)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    return path


def _submit(script_path: Path) -> None:
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("sbatch not found. Are you on a SLURM cluster?") from exc

    message = result.stdout.strip() or "Submitted SLURM job."
    print(message)


def _add_common_slurm_args(
    parser: argparse.ArgumentParser, *, default_job_name: str
) -> None:
    parser.add_argument("--job-name", default=default_job_name)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gres", default=DEFAULT_GRES)
    parser.add_argument("--cpus-per-task", type=int, default=DEFAULT_CPUS_PER_TASK)
    parser.add_argument("--mem", default=DEFAULT_MEM)
    parser.add_argument("--time", dest="time_limit", default=DEFAULT_TIME)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--error", default=None)
    parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional SBATCH lines, e.g. --sbatch-arg='--constraint=a100'.",
    )
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--script-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-submit", action="store_true")


def _handle_array(args: argparse.Namespace, extra_args: List[str]) -> None:
    extra_args = _normalize_extra_args(extra_args)
    repo_root = Path(args.repo_root).resolve()

    array_range = args.array
    if array_range is None:
        array_range = _compute_array_range(extra_args, repo_root)

    output_path = args.output or f"{args.logs_dir}/%x_%A_%a.out"
    error_path = args.error or f"{args.logs_dir}/%x_%A_%a.err"

    command = _render_command("src.cli.run_experiment_array", extra_args)
    script = _render_sbatch_script(
        job_name=args.job_name,
        partition=args.partition,
        nodes=args.nodes,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time_limit=args.time_limit,
        output_path=output_path,
        error_path=error_path,
        array_range=array_range,
        logs_dir=args.logs_dir,
        python_bin=args.python_bin,
        repo_root=repo_root,
        command=command,
        sbatch_args=args.sbatch_arg,
    )

    if args.dry_run and not args.no_submit:
        print(script)
        return

    script_path = _write_script(
        script, Path(args.script_path) if args.script_path else None
    )

    if args.no_submit:
        if args.script_path is None:
            print(script)
        else:
            print(f"Wrote SLURM script to {script_path}")
        return

    _submit(script_path)


def _handle_llm_judge(args: argparse.Namespace, extra_args: List[str]) -> None:
    extra_args = _normalize_extra_args(extra_args)
    repo_root = Path(args.repo_root).resolve()

    output_path = args.output or f"{args.logs_dir}/%x_%j.out"
    error_path = args.error or f"{args.logs_dir}/%x_%j.err"

    command = _render_command("src.cli.run_llm_judge", extra_args)
    script = _render_sbatch_script(
        job_name=args.job_name,
        partition=args.partition,
        nodes=args.nodes,
        gres=args.gres,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time_limit=args.time_limit,
        output_path=output_path,
        error_path=error_path,
        array_range=None,
        logs_dir=args.logs_dir,
        python_bin=args.python_bin,
        repo_root=repo_root,
        command=command,
        sbatch_args=args.sbatch_arg,
    )

    if args.dry_run and not args.no_submit:
        print(script)
        return

    script_path = _write_script(
        script, Path(args.script_path) if args.script_path else None
    )

    if args.no_submit:
        if args.script_path is None:
            print(script)
        else:
            print(f"Wrote SLURM script to {script_path}")
        return

    _submit(script_path)


def _handle_status(args: argparse.Namespace) -> None:
    user = args.user
    if not user and not args.all_users:
        user = getpass.getuser()

    cmd = ["squeue"]
    if not args.all_users and user:
        cmd.extend(["-u", user])
    if args.name:
        cmd.extend(["-n", args.name])
    if args.partition:
        cmd.extend(["-p", args.partition])
    if args.format:
        cmd.extend(["-o", args.format])

    try:
        subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False,
        )
    except FileNotFoundError as exc:
        raise SystemExit("squeue not found. Are you on a SLURM cluster?") from exc

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit and monitor nl2atl jobs on SLURM."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    array_parser = subparsers.add_parser(
        "array",
        help="Submit a SLURM array for nl2atl run-array.",
    )
    _add_common_slurm_args(array_parser, default_job_name="nl2atl-array")
    array_parser.add_argument(
        "--array",
        default=None,
        help="Override SLURM array range (e.g., 0-99). Default: computed from run-array --count.",
    )

    judge_parser = subparsers.add_parser(
        "llm-judge",
        help="Submit a SLURM job for nl2atl llm-judge.",
    )
    _add_common_slurm_args(judge_parser, default_job_name="nl2atl-llm-judge")

    status_parser = subparsers.add_parser(
        "status",
        help="Show SLURM queue status for the current user.",
    )
    status_parser.add_argument(
        "--user",
        default=None,
        help="Show jobs for this user (default: current user).",
    )
    status_parser.add_argument(
        "--all-users",
        action="store_true",
        help="Show jobs for all users (ignores --user).",
    )
    status_parser.add_argument(
        "--name",
        default=None,
        help="Filter by job name.",
    )
    status_parser.add_argument(
        "--partition",
        default=None,
        help="Filter by partition.",
    )
    status_parser.add_argument(
        "--format",
        default=None,
        help="Custom squeue output format string (see squeue -o).",
    )

    args, extra = parser.parse_known_args()

    if args.subcommand == "array":
        _handle_array(args, extra)
    elif args.subcommand == "llm-judge":
        _handle_llm_judge(args, extra)
    elif args.subcommand == "status":
        _handle_status(args)


if __name__ == "__main__":
    main()
