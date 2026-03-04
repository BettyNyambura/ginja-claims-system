import re
import io
import os
import json
import base64
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()


def _extract_text_digital(pdf_path: str) -> str:
    """Extract text from a digitally generated PDF using PyMuPDF."""
    doc  = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text.strip()


def _pdf_page_to_base64(pdf_path: str, page_num: int = 0) -> str:
    """Convert a PDF page to a compressed base64 JPEG string."""
    from PIL import Image

    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    mat  = fitz.Matrix(1.5, 1.5)
    pix  = page.get_pixmap(matrix=mat)
    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    doc.close()

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_structured_groq(pdf_path: str) -> dict:
    """
    Use Groq's vision model to extract structured fields from a scanned
    or handwritten PDF invoice.

    Returns a dictionary of extracted fields.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Groq SDK not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env file")

    client    = Groq(api_key=api_key)
    image_b64 = _pdf_page_to_base64(pdf_path)

    prompt = """Extract fields from this medical invoice or claim form.
Return ONLY a valid JSON object with exactly these keys, no explanation, no markdown:
{
  "member_id": "",
  "member_name": "",
  "date_of_service": "",
  "claimed_amount": 0,
  "hospital_no": "",
  "visit_type": "",
  "line_items": [{"service": "", "amount": 0}]
}

Rules:
- member_id: the membership or insurance number (digits only, no prefix)
- member_name: first name then last name
- date_of_service: in format DD-MM-YYYY
- claimed_amount: the final total as a number
- line_items: each procedure/service with its amount as a number
- If a field is not found, use null
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    raw = re.sub(r"```json|```", "", raw).strip()

    return json.loads(raw)


def extract_text(pdf_path: str) -> tuple[str, str]:
    """
    Extract text from a PDF using PyMuPDF.
    Returns (text, method).
    """
    digital_text = _extract_text_digital(pdf_path)
    if len(digital_text) > 50:
        return digital_text, "pymupdf"
    return "", "groq"


def parse_invoice(pdf_path: str) -> dict:
    """
    Extract structured claim fields from a PDF invoice.

    Strategy:
        1. Try PyMuPDF for digital/system-generated PDFs (fast, free)
        2. Fall back to Groq vision model for scanned or handwritten documents

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary of extracted fields.
    """
    text, method = extract_text(pdf_path)

    # ── Groq vision fallback ──────────────────────────────────────────────────
    if method == "groq":
        print("Digital extraction insufficient, using Groq vision model...")
        try:
            groq_result = _extract_structured_groq(pdf_path)

            # Normalise member_id
            member_id = groq_result.get("member_id")
            if member_id and not str(member_id).startswith("M"):
                member_id = f"M{member_id}"

            return {
                "member_id":          member_id,
                "member_name":        groq_result.get("member_name"),
                "hospital_no":        groq_result.get("hospital_no"),
                "visit_type":         groq_result.get("visit_type"),
                "date_of_service":    groq_result.get("date_of_service"),
                "claimed_amount":     float(groq_result["claimed_amount"]) if groq_result.get("claimed_amount") else None,
                "line_items":         groq_result.get("line_items", []),
                "extraction_method":  "groq_vision",
                "raw_text":           "",
            }
        except Exception as e:
            print(f"Groq extraction failed: {e}")
            # Return empty result — validation will catch missing fields
            return {
                "member_id": None, "member_name": None, "hospital_no": None,
                "visit_type": None, "date_of_service": None, "claimed_amount": None,
                "line_items": [], "extraction_method": "groq_vision_failed", "raw_text": "",
            }

    # ── PyMuPDF regex parsing ─────────────────────────────────────────────────
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    result = {
        "member_id":          None,
        "member_name":        None,
        "hospital_no":        None,
        "visit_type":         None,
        "date_of_service":    None,
        "claimed_amount":     None,
        "line_items":         [],
        "extraction_method":  "pymupdf",
        "raw_text":           text,
    }

    for i, line in enumerate(lines):

        # Member number
        if re.search(r"member no", line, re.IGNORECASE):
            match = re.search(r"\d{5,}", line)
            if match:
                result["member_id"] = f"M{match.group()}"
            elif i + 1 < len(lines):
                next_match = re.search(r"\d{5,}", lines[i + 1])
                if next_match:
                    result["member_id"] = f"M{next_match.group()}"

        # Member name
        if re.search(r"full name", line, re.IGNORECASE):
            name = re.sub(r"full name[:\s]*", "", line, flags=re.IGNORECASE).strip()
            result["member_name"] = name if name else (lines[i + 1] if i + 1 < len(lines) else None)

        # Hospital number
        if re.search(r"hospital no", line, re.IGNORECASE):
            match = re.search(r"\d{4,}", line)
            if match:
                result["hospital_no"] = match.group()
            elif i + 1 < len(lines):
                next_match = re.search(r"\d{4,}", lines[i + 1])
                if next_match:
                    result["hospital_no"] = next_match.group()

        # Visit type
        if re.search(r"visit type", line, re.IGNORECASE):
            visit = re.sub(r"visit type[:\s]*", "", line, flags=re.IGNORECASE).strip()
            result["visit_type"] = visit if visit else (lines[i + 1] if i + 1 < len(lines) else None)

        # Date of service
        date_match = re.search(
            r"(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}"
            r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            line
        )
        if date_match and not result["date_of_service"]:
            result["date_of_service"] = date_match.group(1)

        # Total amount
        if re.search(r"\bTotal\b", line, re.IGNORECASE):
            match = re.search(r"([\d,]+(?:\.\d{2})?)", line)
            if match:
                result["claimed_amount"] = float(match.group(1).replace(",", ""))
            elif i + 1 < len(lines):
                next_match = re.search(r"([\d,]+(?:\.\d{2})?)", lines[i + 1])
                if next_match:
                    result["claimed_amount"] = float(next_match.group(1).replace(",", ""))

    # Line items
    amount_pattern = re.compile(r"^(.+?)\s+([\d,]+(?:\.\d{2})?)$")
    skip_keywords  = [
        "total", "amount", "service", "invoice", "thank", "kes",
        "description", "outpatient", "inpatient", "hospital", "date",
        "phone", "visit", "member", "insurance", "signature", "system",
        "choosing", "required",
    ]
    date_pattern = re.compile(
        r"\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    )

    for line in lines:
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        if date_pattern.search(line):
            continue
        match = amount_pattern.match(line)
        if match:
            service = match.group(1).strip()
            amount  = float(match.group(2).replace(",", ""))
            if amount > 0 and len(service) > 2:
                result["line_items"].append({"service": service, "amount": amount})

    # Fallback: sum line items if total not found
    if result["claimed_amount"] is None and result["line_items"]:
        result["claimed_amount"] = sum(item["amount"] for item in result["line_items"])

    return result


def validate_extraction(parsed: dict) -> list[str]:
    """Validate required fields were extracted."""
    issues   = []
    required = ["member_id", "date_of_service", "claimed_amount"]

    for field in required:
        if not parsed.get(field):
            issues.append(f"Missing: {field}")

    amount = parsed.get("claimed_amount")
    if amount is not None and amount <= 0:
        issues.append("Invalid: claimed_amount must be > 0")

    return issues


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_invoice.pdf"

    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)

    parsed = parse_invoice(path)
    issues = validate_extraction(parsed)

    print(json.dumps({k: v for k, v in parsed.items() if k != "raw_text"}, indent=2))
    print(f"\nExtraction method: {parsed['extraction_method']}")

    if issues:
        print(f"Validation issues: {issues}")
    else:
        print("Extraction validated successfully.")