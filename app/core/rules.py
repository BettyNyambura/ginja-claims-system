from typing import Optional


# Thresholds — easy to tune without retraining the model
TARIFF_DEVIATION_HARD_FAIL_PCT  = 200.0   # claimed > 3x tariff → instant FAIL
TARIFF_DEVIATION_FLAG_PCT       = 30.0    # claimed > 30% above tariff → FLAG
HIGH_FREQUENCY_HARD_FAIL        = 40      # >40 claims in period → instant FAIL
HIGH_FREQUENCY_FLAG             = 15      # >15 claims in period → FLAG


def apply_rules(claim: dict) -> Optional[dict]:
    """
    Apply deterministic business rules to a claim before or alongside ML scoring.

    Hard rules override the ML model — they represent absolute policy violations
    that no probability score can overturn (e.g. billing 3x the approved tariff).

    Soft rules feed into the final decision blending logic in decision.py.

    Args:
        claim: Raw claim dictionary.

    Returns:
        dict with keys: override (bool), decision (str|None), reason (str|None)
        If override=True, the ML model is bypassed and this decision stands.
    """
    claimed  = float(claim.get("claimed_amount", 0))
    tariff   = float(claim.get("approved_tariff_amount", 1))
    freq     = int(claim.get("historical_claim_frequency", 0))

    deviation_pct = ((claimed - tariff) / tariff) * 100 if tariff > 0 else 0

    # --- Hard rules (override ML) ---

    if deviation_pct >= TARIFF_DEVIATION_HARD_FAIL_PCT:
        return {
            "override": True,
            "decision": "FAIL",
            "reason":   f"Claimed amount exceeds tariff by {deviation_pct:.1f}% (threshold: {TARIFF_DEVIATION_HARD_FAIL_PCT}%)",
        }

    if freq >= HIGH_FREQUENCY_HARD_FAIL:
        return {
            "override": True,
            "decision": "FAIL",
            "reason":   f"Historical claim frequency of {freq} exceeds hard limit of {HIGH_FREQUENCY_HARD_FAIL}",
        }

    # --- Soft rules (inform ML blending, no override) ---

    soft_flags = []

    if deviation_pct >= TARIFF_DEVIATION_FLAG_PCT:
        soft_flags.append(f"Amount exceeds tariff by {deviation_pct:.1f}%")

    if freq >= HIGH_FREQUENCY_FLAG:
        soft_flags.append(f"Abnormal claim frequency: {freq}")

    if claimed == 0:
        soft_flags.append("Claimed amount is zero")

    if claimed < 0:
        return {
            "override": True,
            "decision": "FAIL",
            "reason":   "Negative claimed amount is invalid",
        }

    return {
        "override":    False,
        "decision":    None,
        "reason":      "; ".join(soft_flags) if soft_flags else None,
        "soft_flags":  soft_flags,
    }