from app.core.rules import apply_rules
from app.core.model import predict


# Probability thresholds mapping to decisions
PASS_THRESHOLD = 0.30
FAIL_THRESHOLD = 0.70


def _score_to_decision(risk_score: float) -> str:
    if risk_score < PASS_THRESHOLD:
        return "PASS"
    elif risk_score < FAIL_THRESHOLD:
        return "FLAG"
    else:
        return "FAIL"


def _build_reason(decision: str, claim: dict, rule_reason: str | None, risk_score: float) -> str:
    """
    Build a reason string that reflects what actually happened in THIS claim,
    not generic feature importances.
    """
    parts = []

    # Rule-based reason always takes priority if present
    if rule_reason:
        parts.append(rule_reason)

    # For PASS decisions — explain why it looks clean
    if decision == "PASS":
        claimed = float(claim.get("claimed_amount", 0))
        tariff  = float(claim.get("approved_tariff_amount", 1))
        freq    = int(claim.get("historical_claim_frequency", 0))
        dev     = ((claimed - tariff) / tariff) * 100 if tariff > 0 else 0

        observations = []
        if abs(dev) <= 15:
            observations.append(f"claimed amount is within {abs(dev):.1f}% of approved tariff")
        if freq <= 10:
            observations.append(f"claim frequency is normal ({freq})")
        if observations:
            parts.append(f"Claim appears legitimate: {', '.join(observations)}")
        else:
            parts.append(f"Risk score {risk_score:.2f} is below threshold — no significant anomalies detected")

    # For FLAG/FAIL decisions — explain the specific anomalies found
    else:
        claimed = float(claim.get("claimed_amount", 0))
        tariff  = float(claim.get("approved_tariff_amount", 1))
        freq    = int(claim.get("historical_claim_frequency", 0))
        dev     = ((claimed - tariff) / tariff) * 100 if tariff > 0 else 0

        anomalies = []
        if dev > 15:
            anomalies.append(f"amount exceeds tariff by {dev:.1f}%")
        if dev < -15:
            anomalies.append(f"amount is {abs(dev):.1f}% below tariff (possible underbilling)")
        if freq > 12:
            anomalies.append(f"abnormal claim frequency ({freq})")

        if anomalies:
            parts.append(f"Anomalies detected: {', '.join(anomalies)}")
        else:
            parts.append(f"ML model flagged elevated risk (score: {risk_score:.2f})")

    return "; ".join(parts) if parts else f"Risk score: {risk_score:.2f}"


def adjudicate(claim: dict) -> dict:
    """
    Run full adjudication on a single claim.

    Pipeline:
        1. Apply deterministic rules — hard violations bypass ML entirely.
        2. Run ML model to get probability score.
        3. If soft rule flags exist, nudge the ML score upward slightly.
        4. Map final score to decision: PASS / FLAG / FAIL.
        5. Return structured output with decision, score, confidence, and reason.

    Args:
        claim: Raw claim dictionary matching the expected schema.

    Returns:
        Adjudication result dictionary.
    """
    rule_result = apply_rules(claim)

    # Hard rule override — ML not needed
    if rule_result["override"]:
        return {
            "claim_id":   claim.get("claim_id", "UNKNOWN"),
            "risk_score": 1.0,
            "decision":   rule_result["decision"],
            "confidence": 1.0,
            "reason":     rule_result["reason"],
            "source":     "rule_engine",
        }

    # ML inference
    ml_result  = predict(claim)
    risk_score = ml_result["risk_score"]

    # Soft rule nudge: if rule flags exist, push score toward FLAG zone
    soft_flags = rule_result.get("soft_flags", [])
    if soft_flags and risk_score < FAIL_THRESHOLD:
        nudge      = min(0.15 * len(soft_flags), 0.25)
        risk_score = min(risk_score + nudge, 0.99)

    decision   = _score_to_decision(risk_score)
    confidence = round(
        1 - risk_score if decision == "PASS" else
        risk_score     if decision == "FAIL" else
        1 - abs(risk_score - 0.5) * 2,
        4
    )

    reason = _build_reason(
        decision=decision,
        claim=claim,
        rule_reason=rule_result.get("reason"),
        risk_score=risk_score,
    )

    return {
        "claim_id":   claim.get("claim_id", "UNKNOWN"),
        "risk_score": round(risk_score, 4),
        "decision":   decision,
        "confidence": confidence,
        "reason":     reason,
        "source":     "ml_model",
    }