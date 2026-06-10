"""Installer for the optional genVITAMIN NL2ATL integration."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_ENV_VALUES: Dict[str, str] = {
    "NL2ATL_URL": "http://localhost:8081",
    "NL2ATL_MODEL": "qwen-3b",
    "NL2ATL_FEW_SHOT": "true",
    "NL2ATL_MAX_NEW_TOKENS": "128",
    "NL2ATL_TIMEOUT": "300",
}

CONFIG_BLOCK = """    # NL2ATL Integration (optional)
    NL2ATL_URL: Optional[str] = None
    NL2ATL_MODEL: str = "qwen-3b"
    NL2ATL_FEW_SHOT: bool = True
    NL2ATL_NUM_FEW_SHOT: Optional[int] = None
    NL2ATL_ADAPTER: Optional[str] = None
    NL2ATL_MAX_NEW_TOKENS: int = 128
    NL2ATL_TIMEOUT: float = 300.0

"""

CLIENT_IMPORT = "from api.services.nl2atl_client import generate_with_nl2atl"
SETTINGS_IMPORT = "from api.core.config import settings"
SERVICE_IMPORT = "from api.services import get_ai_service"

OLD_GENERATE_BLOCK = """        logger.info(
            f"Generating {request.logic_type.value} formula using knowledge base first"
        )
        # Use AI service to generate the formula (which uses knowledge base internally)
        formula = await ai_service.generate_formula(
            description=request.description, logic=request.logic_type.value
        )
"""

NEW_GENERATE_BLOCK = """        formula = generate_with_nl2atl(
            description=request.description,
            logic=request.logic_type.value,
            settings=settings,
            logger=logger,
        )
        if formula:
            logger.info("Generated formula using NL2ATL")
        else:
            logger.info(
                "Generating %s formula using knowledge base first",
                request.logic_type.value,
            )
            # Use genVITAMIN's native generator as fallback.
            formula = await ai_service.generate_formula(
                description=request.description, logic=request.logic_type.value
            )
"""

NL2ATL_CLIENT_SOURCE = (
    r'''"""Optional NL2ATL client for genVITAMIN formula generation."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Optional, Protocol


class NL2ATLSettings(Protocol):
    """Settings required by the NL2ATL bridge."""

    NL2ATL_URL: Optional[str]
    NL2ATL_MODEL: str
    NL2ATL_FEW_SHOT: bool
    NL2ATL_NUM_FEW_SHOT: Optional[int]
    NL2ATL_ADAPTER: Optional[str]
    NL2ATL_MAX_NEW_TOKENS: int
    NL2ATL_TIMEOUT: float


def generate_with_nl2atl(
    description: str,
    logic: str,
    settings: NL2ATLSettings,
    logger: Optional[Any] = None,
) -> Optional[str]:
    """Return an ATL formula from NL2ATL, or None so genVITAMIN can fall back."""
    if logic.upper() != "ATL":
        _log(logger, "info", "NL2ATL supports ATL only; using genVITAMIN fallback")
        return None

    base_url = (settings.NL2ATL_URL or "").strip()
    if not base_url:
        _log(logger, "debug", "NL2ATL_URL is not configured")
        return None

    payload = {
        "description": description,
        "model": settings.NL2ATL_MODEL,
        "few_shot": settings.NL2ATL_FEW_SHOT,
        "num_few_shot": settings.NL2ATL_NUM_FEW_SHOT,
        "adapter": _blank_to_none(settings.NL2ATL_ADAPTER),
        "max_new_tokens": settings.NL2ATL_MAX_NEW_TOKENS,
    }
    payload = {key: value for key, value in payload.items() if value is not None}

    request = urllib.request.Request(
        url=base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request, timeout=settings.NL2ATL_TIMEOUT
        ) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        _log(logger, "warning", f"NL2ATL returned HTTP {exc.code}: {_error_body(exc)}")
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _log(logger, "warning", f"NL2ATL is unavailable: {exc}")
        return None

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        _log(logger, "warning", f"NL2ATL returned invalid JSON: {exc}")
        return None

    formula = parsed.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        _log(logger, "warning", "NL2ATL response did not include a formula")
        return None

    return normalize_formula(formula)


def normalize_formula(formula: str) -> str:
    """Convert NL2ATL coalition brackets into genVITAMIN's ATL syntax."""
    return re.sub(r"<<\s*([^>]+?)\s*>>", r"<\1>", formula.strip())


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    return body[:500] if body else exc.reason


def _log(logger: Optional[Any], level: str, message: str) -> None:
    if logger is None:
        return
    log_method = getattr(logger, level, None)
    if log_method is not None:
        log_method(message)
'''
)


class IntegrationError(RuntimeError):
    """Raised when the genVITAMIN integration cannot be installed."""


@dataclass
class InstallOptions:
    """Options for installing the genVITAMIN integration."""

    genvitamin_path: Path
    nl2atl_url: str = DEFAULT_ENV_VALUES["NL2ATL_URL"]
    model: str = DEFAULT_ENV_VALUES["NL2ATL_MODEL"]
    few_shot: bool = True
    num_few_shot: Optional[int] = None
    adapter: Optional[str] = None
    max_new_tokens: int = int(DEFAULT_ENV_VALUES["NL2ATL_MAX_NEW_TOKENS"])
    timeout: float = float(DEFAULT_ENV_VALUES["NL2ATL_TIMEOUT"])
    dry_run: bool = False
    no_backup: bool = False
    skip_env: bool = False
    skip_env_example: bool = False
    force_env: bool = False


@dataclass
class InstallReport:
    """Summary of an integration installation or status check."""

    repo_root: Path
    changed: List[Path] = field(default_factory=list)
    unchanged: List[Path] = field(default_factory=list)
    missing: List[Path] = field(default_factory=list)
    backups: List[Path] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def changed_count(self) -> int:
        return len(self.changed)


def install(options: InstallOptions) -> InstallReport:
    """Install the NL2ATL hook into a genVITAMIN checkout."""
    repo_root = options.genvitamin_path.resolve()
    paths = _resolve_paths(repo_root)
    _validate_repo(paths)

    report = InstallReport(repo_root=repo_root)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    generate_text = paths.generate_file.read_text(encoding="utf-8")
    config_text = paths.config_file.read_text(encoding="utf-8")

    _write_if_changed(
        paths.client_file,
        NL2ATL_CLIENT_SOURCE,
        options,
        report,
        timestamp,
    )
    _write_if_changed(
        paths.generate_file,
        patch_generate_text(generate_text),
        options,
        report,
        timestamp,
    )
    _write_if_changed(
        paths.config_file,
        patch_config_text(config_text),
        options,
        report,
        timestamp,
    )

    env_values = _env_values(options)
    if not options.skip_env:
        current_env_text = ""
        if paths.env_file.exists():
            current_env_text = paths.env_file.read_text(encoding="utf-8")
        env_text = _upsert_env_text(
            current_env_text,
            env_values,
            force=options.force_env,
        )
        _write_if_changed(paths.env_file, env_text, options, report, timestamp)

    if not options.skip_env_example and paths.env_example_file.exists():
        env_example_values = dict(env_values)
        env_example_values.setdefault("NL2ATL_NUM_FEW_SHOT", "")
        env_example_values.setdefault("NL2ATL_ADAPTER", "")
        env_example_text = _upsert_env_text(
            paths.env_example_file.read_text(encoding="utf-8"),
            env_example_values,
            force=False,
            comment_existing=True,
        )
        _write_if_changed(
            paths.env_example_file,
            env_example_text,
            options,
            report,
            timestamp,
        )

    if options.dry_run:
        report.notes.append("Dry run only; no files were written.")
    return report


def status(genvitamin_path: Path) -> InstallReport:
    """Return a lightweight status report for a genVITAMIN checkout."""
    repo_root = genvitamin_path.resolve()
    paths = _resolve_paths(repo_root)
    report = InstallReport(repo_root=repo_root)

    status_paths = [
        paths.generate_file,
        paths.config_file,
        paths.client_file,
        paths.env_file,
    ]
    for path in status_paths:
        if path.exists():
            report.unchanged.append(path)
        else:
            report.missing.append(path)

    if paths.generate_file.exists():
        text = paths.generate_file.read_text(encoding="utf-8")
        if "generate_with_nl2atl" in text:
            report.notes.append("generate.py contains the NL2ATL hook.")
        elif "_call_nl2atl" in text:
            report.notes.append("generate.py contains the older inline NL2ATL hook.")
        else:
            report.notes.append("generate.py does not contain an NL2ATL hook.")

    if paths.config_file.exists():
        text = paths.config_file.read_text(encoding="utf-8")
        if "NL2ATL_URL" in text:
            report.notes.append("config.py contains NL2ATL settings.")
        else:
            report.notes.append("config.py does not contain NL2ATL settings.")

    return report


def patch_generate_text(text: str) -> str:
    """Patch genVITAMIN's formula generation route to try NL2ATL first."""
    if "_call_nl2atl" in text and "generate_with_nl2atl" not in text:
        raise IntegrationError(
            "generate.py contains the older inline NL2ATL patch. Restore a clean "
            "genVITAMIN generate.py or reinstall genVITAMIN, then run this installer."
        )

    text = _ensure_import(text, SETTINGS_IMPORT, after=SERVICE_IMPORT)
    text = _ensure_import(text, CLIENT_IMPORT, after=SERVICE_IMPORT)

    if "generate_with_nl2atl(" in text:
        return text
    if OLD_GENERATE_BLOCK not in text:
        raise IntegrationError(
            "Could not find the expected genVITAMIN formula generation block. "
            "The upstream file may have changed; patch it manually using the README."
        )
    return text.replace(OLD_GENERATE_BLOCK, NEW_GENERATE_BLOCK, 1)


def patch_config_text(text: str) -> str:
    """Patch genVITAMIN settings with optional NL2ATL configuration."""
    text = _ensure_optional_import(text)
    if "NL2ATL_URL" in text:
        return text

    for marker in (
        "    # ChromaDB Configuration\n",
        "    # Embeddings Configuration\n",
        "    class Config:\n",
    ):
        if marker in text:
            return text.replace(marker, CONFIG_BLOCK + marker, 1)

    raise IntegrationError(
        "Could not find a safe insertion point in api/core/config.py."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nl2atl genvitamin",
        description=(
            "Install or inspect the optional NL2ATL integration for genVITAMIN."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser(
        "install", help="Patch a genVITAMIN checkout to call the NL2ATL API."
    )
    _add_path_argument(install_parser)
    install_parser.add_argument(
        "--nl2atl-url",
        default=DEFAULT_ENV_VALUES["NL2ATL_URL"],
    )
    install_parser.add_argument("--model", default=DEFAULT_ENV_VALUES["NL2ATL_MODEL"])
    install_parser.add_argument("--adapter", default=None)
    install_parser.add_argument("--num-few-shot", type=int, default=None)
    install_parser.add_argument(
        "--no-few-shot",
        action="store_false",
        dest="few_shot",
        help="Disable few-shot prompting in NL2ATL requests.",
    )
    install_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(DEFAULT_ENV_VALUES["NL2ATL_MAX_NEW_TOKENS"]),
    )
    install_parser.add_argument(
        "--timeout", type=float, default=float(DEFAULT_ENV_VALUES["NL2ATL_TIMEOUT"])
    )
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--no-backup", action="store_true")
    install_parser.add_argument("--skip-env", action="store_true")
    install_parser.add_argument("--skip-env-example", action="store_true")
    install_parser.add_argument(
        "--force-env",
        action="store_true",
        help="Overwrite existing NL2ATL_* values in api/.env.",
    )
    install_parser.set_defaults(few_shot=True)

    status_parser = subparsers.add_parser(
        "status", help="Show whether a genVITAMIN checkout has the NL2ATL hook."
    )
    _add_path_argument(status_parser)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    if argv is None:
        import sys

        args_list = sys.argv[1:]
    else:
        args_list = list(argv)

    if not args_list:
        args_list = ["install"]
    elif args_list[0].startswith("-") and args_list[0] not in {"-h", "--help"}:
        args_list = ["install"] + args_list

    parser = build_parser()
    args = parser.parse_args(args_list)

    if args.command == "install":
        report = install(
            InstallOptions(
                genvitamin_path=Path(args.genvitamin_path),
                nl2atl_url=args.nl2atl_url,
                model=args.model,
                few_shot=args.few_shot,
                num_few_shot=args.num_few_shot,
                adapter=args.adapter,
                max_new_tokens=args.max_new_tokens,
                timeout=args.timeout,
                dry_run=args.dry_run,
                no_backup=args.no_backup,
                skip_env=args.skip_env,
                skip_env_example=args.skip_env_example,
                force_env=args.force_env,
            )
        )
        _print_report(report, title="genVITAMIN NL2ATL integration installed")
    elif args.command == "status":
        _print_report(status(Path(args.genvitamin_path)), title="genVITAMIN status")


@dataclass
class _GenVitaminPaths:
    root: Path
    generate_file: Path
    config_file: Path
    client_file: Path
    env_file: Path
    env_example_file: Path


def _resolve_paths(repo_root: Path) -> _GenVitaminPaths:
    api_root = repo_root / "api"
    return _GenVitaminPaths(
        root=repo_root,
        generate_file=api_root / "routes" / "ai" / "generate.py",
        config_file=api_root / "core" / "config.py",
        client_file=api_root / "services" / "nl2atl_client.py",
        env_file=api_root / ".env",
        env_example_file=api_root / "env.example",
    )


def _validate_repo(paths: _GenVitaminPaths) -> None:
    missing = [
        path
        for path in (paths.generate_file, paths.config_file, paths.client_file.parent)
        if not path.exists()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise IntegrationError(
            "This does not look like a supported genVITAMIN checkout. Missing:\n"
            + formatted
        )


def _write_if_changed(
    path: Path,
    new_text: str,
    options: InstallOptions,
    report: InstallReport,
    timestamp: str,
) -> None:
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    if old_text == new_text:
        report.unchanged.append(path)
        return

    report.changed.append(path)
    if options.dry_run:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if old_text is not None and not options.no_backup:
        backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
        shutil.copy2(path, backup_path)
        report.backups.append(backup_path)
    path.write_text(new_text, encoding="utf-8")


def _env_values(options: InstallOptions) -> Dict[str, str]:
    values = {
        "NL2ATL_URL": options.nl2atl_url,
        "NL2ATL_MODEL": options.model,
        "NL2ATL_FEW_SHOT": str(options.few_shot).lower(),
        "NL2ATL_MAX_NEW_TOKENS": str(options.max_new_tokens),
        "NL2ATL_TIMEOUT": f"{options.timeout:g}",
    }
    if options.num_few_shot is not None:
        values["NL2ATL_NUM_FEW_SHOT"] = str(options.num_few_shot)
    if options.adapter:
        values["NL2ATL_ADAPTER"] = options.adapter
    return values


def _upsert_env_text(
    text: str,
    defaults: Dict[str, str],
    *,
    force: bool,
    comment_existing: bool = False,
) -> str:
    lines = text.splitlines(keepends=True)
    seen = set()
    updated: List[str] = []

    for line in lines:
        key = _env_key(line, include_commented=comment_existing)
        if key in defaults:
            seen.add(key)
            if force:
                updated.append(_env_line(key, defaults[key], comment=comment_existing))
            else:
                updated.append(line)
        else:
            updated.append(line)

    missing = [key for key in defaults if key not in seen]
    if missing:
        if updated and not updated[-1].endswith("\n"):
            updated[-1] += "\n"
        if updated and updated[-1].strip():
            updated.append("\n")
        updated.append("# NL2ATL integration for ATL formula generation\n")
        for key in missing:
            updated.append(_env_line(key, defaults[key], comment=comment_existing))

    if not updated:
        updated.append("# NL2ATL integration for ATL formula generation\n")
        for key, value in defaults.items():
            updated.append(_env_line(key, value, comment=comment_existing))

    return "".join(updated)


def _env_key(line: str, *, include_commented: bool = False) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        if not include_commented:
            return None
        stripped = stripped[1:].strip()
    if "=" not in stripped:
        return None
    return stripped.split("=", 1)[0].strip()


def _env_line(key: str, value: str, *, comment: bool) -> str:
    prefix = "# " if comment else ""
    return f"{prefix}{key}={value}\n"


def _ensure_import(text: str, import_line: str, *, after: str) -> str:
    if import_line in text:
        return text
    if after not in text:
        raise IntegrationError(f"Could not find import anchor in generate.py: {after}")
    return text.replace(after, f"{after}\n{import_line}", 1)


def _ensure_optional_import(text: str) -> str:
    if "Optional" in text:
        return text
    if "from typing import List\n" in text:
        return text.replace(
            "from typing import List\n",
            "from typing import List, Optional\n",
            1,
        )
    if "from typing import " in text:
        lines = text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if line.startswith("from typing import "):
                lines[index] = line.rstrip("\n") + ", Optional\n"
                return "".join(lines)
    return "from typing import Optional\n" + text


def _add_path_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--genvitamin-path",
        default="genVITAMIN",
        help="Path to the genVITAMIN repository root.",
    )


def _print_report(report: InstallReport, *, title: str) -> None:
    print(title)
    print(f"Repository: {report.repo_root}")
    if report.changed:
        print("Changed:")
        for path in report.changed:
            print(f"  - {path}")
    if report.unchanged:
        print("Unchanged:")
        for path in report.unchanged:
            print(f"  - {path}")
    if report.missing:
        print("Missing:")
        for path in report.missing:
            print(f"  - {path}")
    if report.backups:
        print("Backups:")
        for path in report.backups:
            print(f"  - {path}")
    for note in report.notes:
        print(note)


if __name__ == "__main__":
    main()
