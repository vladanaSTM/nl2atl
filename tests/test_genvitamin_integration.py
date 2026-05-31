from pathlib import Path

import pytest

from src.integrations import genvitamin

GENERATE_PY = '''"""Formula generation endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.schemas.requests import LogicType
from api.services import get_ai_service

router = APIRouter()
logger = logging.getLogger(__name__)

ai_service = get_ai_service()


class GenerateFormulaRequest(BaseModel):
    description: str
    logic_type: LogicType
    complexity: Optional[str] = "medium"


class GenerateFormulaResponse(BaseModel):
    formula: str
    logic_type: str
    description: str
    explanation: str


@router.post("/", response_model=GenerateFormulaResponse)
async def generate_formula(request: GenerateFormulaRequest):
    try:
        if not request.description.strip():
            raise HTTPException(status_code=400, detail="Description cannot be empty")

        logger.info(
            f"Generating {request.logic_type.value} formula using knowledge base first"
        )
        # Use AI service to generate the formula (which uses knowledge base internally)
        formula = await ai_service.generate_formula(
            description=request.description, logic=request.logic_type.value
        )
        return GenerateFormulaResponse(
            formula=formula,
            logic_type=request.logic_type.value,
            description=request.description,
            explanation="",
        )
    except HTTPException:
        raise
'''


CONFIG_PY = """from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STREAM_ERROR_DELAY: float = 0.02

    # ChromaDB Configuration
    CHROMADB_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"


settings = Settings()
"""


def test_install_patches_genvitamin_tree(tmp_path):
    root = _write_genvitamin_tree(tmp_path)

    report = genvitamin.install(
        genvitamin.InstallOptions(genvitamin_path=root, no_backup=True)
    )

    assert report.changed_count == 5
    assert (root / "api/services/nl2atl_client.py").exists()

    generate_text = (root / "api/routes/ai/generate.py").read_text(encoding="utf-8")
    assert "from api.core.config import settings" in generate_text
    assert (
        "from api.services.nl2atl_client import generate_with_nl2atl" in generate_text
    )
    assert "formula = generate_with_nl2atl(" in generate_text
    assert "Generated formula using NL2ATL" in generate_text

    config_text = (root / "api/core/config.py").read_text(encoding="utf-8")
    assert "NL2ATL_URL: Optional[str] = None" in config_text
    assert "NL2ATL_ADAPTER: Optional[str] = None" in config_text

    env_text = (root / "api/.env").read_text(encoding="utf-8")
    assert "OLLAMA_MODEL=custom" in env_text
    assert "NL2ATL_URL=http://localhost:8081" in env_text
    assert "NL2ATL_ADAPTER" not in env_text

    env_example_text = (root / "api/env.example").read_text(encoding="utf-8")
    assert "# NL2ATL_URL=http://localhost:8081" in env_example_text
    assert "# NL2ATL_NUM_FEW_SHOT=" in env_example_text
    assert "# NL2ATL_ADAPTER=" in env_example_text


def test_install_is_idempotent(tmp_path):
    root = _write_genvitamin_tree(tmp_path)
    options = genvitamin.InstallOptions(genvitamin_path=root, no_backup=True)

    genvitamin.install(options)
    report = genvitamin.install(options)

    assert report.changed == []
    assert report.unchanged


def test_install_preserves_existing_env_values(tmp_path):
    root = _write_genvitamin_tree(tmp_path)
    env_file = root / "api/.env"
    env_file.write_text("NL2ATL_URL=http://remote:9000\n", encoding="utf-8")

    genvitamin.install(genvitamin.InstallOptions(genvitamin_path=root, no_backup=True))

    env_text = env_file.read_text(encoding="utf-8")
    assert "NL2ATL_URL=http://remote:9000" in env_text
    assert "NL2ATL_MODEL=qwen-3b" in env_text


def test_dry_run_does_not_write_files(tmp_path):
    root = _write_genvitamin_tree(tmp_path)

    report = genvitamin.install(
        genvitamin.InstallOptions(genvitamin_path=root, dry_run=True)
    )

    assert report.changed
    assert not (root / "api/services/nl2atl_client.py").exists()
    assert "generate_with_nl2atl" not in (root / "api/routes/ai/generate.py").read_text(
        encoding="utf-8"
    )


def test_client_normalizes_coalition_syntax():
    namespace = {}
    exec(genvitamin.NL2ATL_CLIENT_SOURCE, namespace)

    assert (
        namespace["normalize_formula"]("<<Controller>>F safe") == "<Controller>F safe"
    )


def test_main_help_shows_genvitamin_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        genvitamin.main(["--help"])

    assert exc.value.code == 0
    assert "install" in capsys.readouterr().out


def _write_genvitamin_tree(tmp_path: Path) -> Path:
    root = tmp_path / "genVITAMIN"
    (root / "api/routes/ai").mkdir(parents=True)
    (root / "api/core").mkdir(parents=True)
    (root / "api/services").mkdir(parents=True)
    (root / "api/routes/ai/generate.py").write_text(GENERATE_PY, encoding="utf-8")
    (root / "api/core/config.py").write_text(CONFIG_PY, encoding="utf-8")
    (root / "api/.env").write_text("OLLAMA_MODEL=custom\n", encoding="utf-8")
    (root / "api/env.example").write_text("DEBUG=false\n", encoding="utf-8")
    return root
