"""Apply genVITAMIN backend patch to use NL2ATL generator."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch genVITAMIN backend to call NL2ATL"
    )
    parser.add_argument(
        "--genvitamin-path",
        default="genVITAMIN",
        help="Path to genVITAMIN repo root (contains api/ and frontend/)",
    )
    args = parser.parse_args()

    repo_root = Path(args.genvitamin_path).resolve()
    target_file = repo_root / "api" / "routes" / "ai" / "generate.py"
    config_file = repo_root / "api" / "core" / "config.py"
    env_file = repo_root / ".env"

    if not target_file.exists():
        raise SystemExit(f"Target file not found: {target_file}")
    if not config_file.exists():
        raise SystemExit(f"Config file not found: {config_file}")

    override_file = Path(__file__).resolve().parent / "overrides" / "generate.py"
    override_config = Path(__file__).resolve().parent / "overrides" / "config.py"
    if not override_file.exists():
        raise SystemExit(f"Override file not found: {override_file}")
    if not override_config.exists():
        raise SystemExit(f"Override file not found: {override_config}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = target_file.with_suffix(f".py.bak.{timestamp}")
    backup_config = config_file.with_suffix(f".py.bak.{timestamp}")

    shutil.copy2(target_file, backup_file)
    shutil.copy2(config_file, backup_config)
    shutil.copy2(override_file, target_file)
    shutil.copy2(override_config, config_file)

    _upsert_env(env_file)

    print("✅ genVITAMIN patched successfully")
    print(f"Backup saved at: {backup_file}")
    print(f"Backup saved at: {backup_config}")
    print(f"Patched file: {target_file}")
    print(f"Patched file: {config_file}")
    print(f"Updated env: {env_file}")


def _upsert_env(env_file: Path) -> None:
    defaults = {
        "NL2ATL_URL": "http://localhost:8081",
        "NL2ATL_MODEL": "qwen-3b",
        "NL2ATL_FEW_SHOT": "true",
        "NL2ATL_ADAPTER": "qwen-3b_finetuned_few_shot/final",
        "NL2ATL_MAX_NEW_TOKENS": "128",
        "NL2ATL_TIMEOUT": "300",
    }

    existing = {}
    lines = []
    if env_file.exists():
        with env_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                existing[key.strip()] = value.strip()

    updated = []
    used_keys = set()
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in defaults:
                updated.append(f"{key}={defaults[key]}\n")
                used_keys.add(key)
            else:
                updated.append(line)
        else:
            updated.append(line)

    for key, value in defaults.items():
        if key not in used_keys and key not in existing:
            updated.append(f"{key}={value}\n")

    if not updated:
        updated = [f"{k}={v}\n" for k, v in defaults.items()]

    with env_file.open("w", encoding="utf-8") as f:
        f.writelines(updated)


if __name__ == "__main__":
    main()
