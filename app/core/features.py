import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer fraud-signal features from raw claims data.

    Each feature is chosen because it captures a known fraud or misbilling pattern:

    1. amount_deviation_pct:
       How much the claimed amount deviates from the approved tariff as a percentage.
       Fraudulent providers inflate claims above the tariff — this is the strongest signal.

    2. is_above_tariff:
       Binary flag — claimed amount exceeds approved tariff.
       Even small overages are worth flagging in rule-based logic.

    3. high_frequency_flag:
       Binary flag — member or provider submitting claims at an abnormally high rate.
       Claim mills and ghost patients tend to generate unusually high volumes.

    4. frequency_x_deviation:
       Interaction feature combining frequency and amount deviation.
       A provider billing slightly above tariff many times is more suspicious than once.

    5. provider_type_encoded:
       Ordinal encoding of provider type by inherent risk level.
       Pharmacies and labs have historically higher misbilling rates in emerging markets.

    6. location_encoded:
       Label encoding of location. Certain regions may have higher fraud rates —
       useful for the model to learn geographic patterns.

    7. log_claimed_amount:
       Log-transformed claimed amount to reduce skew from very large procedures
       (e.g. surgeries) dominating the model.

    Args:
        df: Raw claims DataFrame.

    Returns:
        DataFrame with engineered feature columns appended.
    """
    df = df.copy()

    # 1. Amount deviation from approved tariff (%)
    df["amount_deviation_pct"] = (
        (df["claimed_amount"] - df["approved_tariff_amount"])
        / df["approved_tariff_amount"]
    ) * 100

    # 2. Binary: is the claimed amount above tariff?
    df["is_above_tariff"] = (df["claimed_amount"] > df["approved_tariff_amount"]).astype(int)

    # 3. Binary: abnormally high claim frequency (threshold: >12 claims in period)
    df["high_frequency_flag"] = (df["historical_claim_frequency"] > 12).astype(int)

    # 4. Interaction: frequency × deviation — catches systematic overbilling
    df["frequency_x_deviation"] = (
        df["historical_claim_frequency"] * df["amount_deviation_pct"]
    )

    # 5. Provider type risk encoding
    provider_risk = {
        "Hospital":   1,
        "Clinic":     2,
        "Specialist": 3,
        "Lab":        4,
        "Pharmacy":   5,
    }
    df["provider_type_encoded"] = df["provider_type"].map(provider_risk).fillna(3)

    # 6. Location encoding
    locations = sorted(df["location"].unique())
    location_map = {loc: idx for idx, loc in enumerate(locations)}
    df["location_encoded"] = df["location"].map(location_map).fillna(0)

    # 7. Log-transformed claimed amount (handles large surgery outliers)
    df["log_claimed_amount"] = np.log1p(df["claimed_amount"])

    return df


# Features the model will train on — everything else is metadata
FEATURE_COLUMNS = [
    "amount_deviation_pct",
    "is_above_tariff",
    "high_frequency_flag",
    "frequency_x_deviation",
    "provider_type_encoded",
    "location_encoded",
    "log_claimed_amount",
    "historical_claim_frequency",
]