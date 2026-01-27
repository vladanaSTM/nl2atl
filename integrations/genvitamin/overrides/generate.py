"""Formula generation endpoints."""

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.schemas.requests import LogicType
from api.services import get_ai_service
from api.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Use shared AIService instance
ai_service = get_ai_service()


# ============================================================================
# Request/Response Models
# ============================================================================


class GenerateFormulaRequest(BaseModel):
    """Request schema for formula generation."""

    description: str
    logic_type: LogicType
    complexity: Optional[str] = "medium"  # simple, medium, complex


class GenerateFormulaResponse(BaseModel):
    """Response schema for formula generation."""

    formula: str
    logic_type: str
    description: str
    explanation: str


# ============================================================================
# Helpers
# ============================================================================


def _call_nl2atl(description: str, logic: str) -> Optional[str]:
    """Call NL2ATL API if configured, otherwise return None."""
    base_url = settings.NL2ATL_URL
    if not base_url:
        return None

    # NL2ATL currently generates ATL only
    if logic != "ATL":
        return None

    payload = {
        "description": description,
        "model": settings.NL2ATL_MODEL,
        "few_shot": settings.NL2ATL_FEW_SHOT,
        "num_few_shot": settings.NL2ATL_NUM_FEW_SHOT,
        "adapter": settings.NL2ATL_ADAPTER,
        "max_new_tokens": settings.NL2ATL_MAX_NEW_TOKENS,
    }

    url = base_url.rstrip("/") + "/generate"
    timeout = settings.NL2ATL_TIMEOUT
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed.get("formula")
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        logger.warning(f"NL2ATL call failed: {exc}")
        return None


# ============================================================================
# Route Handlers
# ============================================================================


@router.post("/", response_model=GenerateFormulaResponse)
async def generate_formula(request: GenerateFormulaRequest):
    """Generate temporal logic formulas using AI with knowledge base support."""
    logger.info(
        f"Received generate_formula request: logic={request.logic_type.value}, description={request.description[:50]}..."
    )

    try:
        if not request.description.strip():
            logger.warning("Generate formula request with empty description")
            raise HTTPException(status_code=400, detail="Description cannot be empty")

        # Try NL2ATL first if configured
        formula = _call_nl2atl(request.description, request.logic_type.value)
        if formula:
            logger.info("Generated formula using NL2ATL")
        else:
            logger.info(
                f"Generating {request.logic_type.value} formula using knowledge base first"
            )
            # Use AI service to generate the formula (which uses knowledge base internally)
            formula = await ai_service.generate_formula(
                description=request.description, logic=request.logic_type.value
            )

        logger.info(f"Successfully generated formula (length: {len(formula)})")

        # Extract explanation if present in the response
        if "#" in formula:
            parts = formula.split("#", 1)
            formula_part = parts[0].strip()
            explanation = parts[1].strip() if len(parts) > 1 else ""
        else:
            formula_part = formula.strip()
            explanation = f"Generated {request.logic_type.value} formula based on: {request.description}"

        logger.info(f"Returning generated formula: {formula_part[:50]}...")
        return GenerateFormulaResponse(
            formula=formula_part,
            logic_type=request.logic_type.value,
            description=request.description,
            explanation=explanation,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating formula: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating formula: {str(e)}"
        )
