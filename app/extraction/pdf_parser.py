import re
from pathlib import Path
import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    """Extract raw text from all pages of a PDF."""
    doc  = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def parse_invoice(pdf_path: str) -> dict:
    """
    Extract structured claim fields from a Nairobi Lifecare-style PDF invoice.

    Designed around the invoice format:
        - Member No, Hospital No, Date
        - Line items: Service Description | Amount (KES)
        - Total amount

    Args:
        pdf_path: Path to the PDF invoice file.

    Returns:
        Dictionary of extracted fields. Missing fields default to None.
    """
    text  = extract_text(pdf_path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    result = {
        "member_id":       None,
        "member_name":     None,
        "hospital_no":     None,
        "visit_type":      None,
        "date_of_service": None,
        "claimed_amount":  None,
        "line_items":      [],
        "raw_text":        text,
    }

    for i, line in enumerate(lines):

        # Member number — handle inline "Member No: 1538500" or split across lines
        if re.search(r"member no", line, re.IGNORECASE):
            match = re.search(r"\d{5,}", line)  # member numbers are 5+ digits
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
            r"(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4})",
            line
        )
        if date_match and not result["date_of_service"]:
            result["date_of_service"] = date_match.group(1)

        # Total amount — handles same line or next line
        if re.search(r"\bTotal\b", line, re.IGNORECASE):
            match = re.search(r"([\d,]+(?:\.\d{2})?)", line)
            if match:
                result["claimed_amount"] = float(match.group(1).replace(",", ""))
            elif i + 1 < len(lines):
                next_match = re.search(r"([\d,]+(?:\.\d{2})?)", lines[i + 1])
                if next_match:
                    result["claimed_amount"] = float(next_match.group(1).replace(",", ""))

    # Line items — skip any line that looks like a date, header, or footer
    amount_pattern = re.compile(r"^(.+?)\s+([\d,]+(?:\.\d{2})?)$")
    skip_keywords  = [
        "total", "amount", "service", "invoice", "thank", "kes",
        "description", "outpatient", "inpatient", "hospital", "date",
        "phone", "visit", "member", "insurance", "signature", "system",
        "choosing", "required",
    ]
    date_pattern = re.compile(
        r"\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}"
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
                result["line_items"].append({
                    "service": service,
                    "amount":  amount,
                })

    # Fallback: if claimed_amount still None, sum up line items
    if result["claimed_amount"] is None and result["line_items"]:
        result["claimed_amount"] = sum(item["amount"] for item in result["line_items"])

    return result


def validate_extraction(parsed: dict) -> list[str]:
    """
    Validate that required fields were successfully extracted.

    Returns:
        List of validation issue descriptions.
    """
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
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_invoice.pdf"

    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)

    parsed = parse_invoice(path)
    issues = validate_extraction(parsed)

    print(json.dumps({k: v for k, v in parsed.items() if k != "raw_text"}, indent=2))

    if issues:
        print(f"\nValidation issues: {issues}")
    else:
        print("\nExtraction validated successfully.")