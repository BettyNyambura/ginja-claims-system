import io
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.decision import adjudicate
from app.extraction.pdf_parser import parse_invoice, validate_extraction

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class ClaimPayload(BaseModel):
    claim_id:                   str
    member_id:                  str
    provider_id:                str
    diagnosis_code:             str
    procedure_code:             str
    claimed_amount:             float = Field(..., gt=0)
    approved_tariff_amount:     float = Field(..., gt=0)
    date_of_service:            str
    provider_type:              str
    historical_claim_frequency: int   = Field(..., ge=0)
    location:                   str
    procedure_name:             Optional[str] = None
    member_name:                Optional[str] = None


class AdjudicationResult(BaseModel):
    claim_id:   str
    risk_score: float
    decision:   str
    confidence: float
    reason:     str
    source:     str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/adjudicate", response_model=AdjudicationResult)
def adjudicate_claim(claim: ClaimPayload):
    """
    Adjudicate a single claim submitted as a JSON payload.

    Returns a decision of PASS, FLAG, or FAIL with a risk score and explanation.
    """
    try:
        result = adjudicate(claim.model_dump())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adjudication error: {str(e)}")


@router.post("/adjudicate/batch")
def adjudicate_batch(file: UploadFile = File(...)):
    """
    Adjudicate a batch of claims from an uploaded CSV or JSON file.

    CSV must include all required claim fields as column headers.
    JSON must be an array of claim objects.

    Returns a list of adjudication results.
    """
    filename = file.filename.lower()

    try:
        contents = file.file.read()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only CSV or JSON files are supported.")

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {str(e)}")

    required_columns = [
        "claim_id", "member_id", "provider_id", "diagnosis_code",
        "procedure_code", "claimed_amount", "approved_tariff_amount",
        "date_of_service", "provider_type", "historical_claim_frequency", "location",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")

    results = []
    for _, row in df.iterrows():
        try:
            result = adjudicate(row.to_dict())
            results.append(result)
        except Exception as e:
            results.append({
                "claim_id": row.get("claim_id", "UNKNOWN"),
                "error":    str(e),
            })

    return {"total": len(results), "results": results}


@router.post("/adjudicate/pdf")
async def adjudicate_pdf(file: UploadFile = File(...)):
    """
    Extract claim data from a PDF invoice and adjudicate it.

    Expects a standard Nairobi Lifecare-style invoice PDF.
    Returns extracted fields + adjudication result.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    parsed = parse_invoice(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    issues = validate_extraction(parsed)
    if issues:
        raise HTTPException(
            status_code=422,
            detail=f"PDF extraction incomplete: {issues}"
        )

    # Fill in defaults for fields not present in invoice PDFs
    claim = {
        "claim_id":                   parsed.get("hospital_no", "PDF-CLAIM"),
        "member_id":                  parsed["member_id"],
        "provider_id":                "P000",
        "diagnosis_code":             "UNKNOWN",
        "procedure_code":             "UNKNOWN",
        "claimed_amount":             parsed["claimed_amount"],
        "approved_tariff_amount":     parsed["claimed_amount"],  # no tariff in PDF
        "date_of_service":            parsed["date_of_service"],
        "provider_type":              parsed.get("visit_type", "Clinic"),
        "historical_claim_frequency": 1,
        "location":                   "Nairobi",
        "member_name":                parsed.get("member_name"),
    }

    result = adjudicate(claim)

    return {
        "extracted": {k: v for k, v in parsed.items() if k != "raw_text"},
        "adjudication": result,
    }


@router.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "Ginja Claims Engine"}