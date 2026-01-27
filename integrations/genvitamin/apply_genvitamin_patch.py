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
        required=True,
        help="Path to genVITAMIN repo root (contains api/ and frontend/)",
    )
    args = parser.parse_args()

    repo_root = Path(args.genvitamin_path).resolve()
    target_file = repo_root / "api" / "routes" / "ai" / "generate.py"

    if not target_file.exists():
        raise SystemExit(f"Target file not found: {target_file}")

    override_file = Path(__file__).resolve().parent / "overrides" / "generate.py"
    if not override_file.exists():
        raise SystemExit(f"Override file not found: {override_file}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = target_file.with_suffix(f".py.bak.{timestamp}")

    shutil.copy2(target_file, backup_file)
    shutil.copy2(override_file, target_file)

    print("✅ genVITAMIN patched successfully")
    print(f"Backup saved at: {backup_file}")
    print(f"Patched file: {target_file}")


if __name__ == "__main__":
    main()
