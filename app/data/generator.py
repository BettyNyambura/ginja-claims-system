import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
np.random.seed(42)
random.seed(42)

# --- Constants reflecting Kenyan healthcare context ---

PROVIDER_TYPES = ["Hospital", "Clinic", "Pharmacy", "Lab", "Specialist"]

DIAGNOSIS_CODES = [
    "J06.9",  # Upper respiratory infection
    "A09",    # Diarrhoea and gastroenteritis
    "K29.7",  # Gastritis
    "J18.9",  # Pneumonia
    "B50.9",  # Malaria
    "E11.9",  # Type 2 diabetes
    "I10",    # Hypertension
    "K35.9",  # Appendicitis
    "N39.0",  # Urinary tract infection
    "S00.9",  # Superficial injury
]

# procedure_code: (name, base_tariff_KES)
PROCEDURE_CODES = {
    "99213": ("Consultation",                   1500),
    "76645": ("Ultrasound Breast",              2800),
    "85025": ("Full Blood Count",               1200),
    "87081": ("Malaria RDT",                     800),
    "93000": ("ECG",                            2500),
    "71046": ("Chest X-Ray",                    3500),
    "99232": ("Inpatient Visit",                5000),
    "27447": ("Knee Replacement",             180000),
    "47562": ("Laparoscopic Cholecystectomy", 120000),
    "99285": ("Emergency Visit",                8000),
}

LOCATIONS = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika"]

KENYAN_FIRST_NAMES = [
    "Wanjiku", "Kamau", "Otieno", "Achieng", "Muthoni", "Njoroge",
    "Akinyi", "Odhiambo", "Waweru", "Chebet", "Kipchoge", "Nafula",
    "Zawadi", "Amani", "Baraka", "Fatuma", "Halima", "Imani",
]

KENYAN_LAST_NAMES = [
    "Kamau", "Otieno", "Mwangi", "Ochieng", "Kariuki", "Mutua",
    "Njuguna", "Omondi", "Wambua", "Rotich", "Langat", "Simiyu",
    "Kimani", "Auma", "Nekesa", "Wekesa", "Korir", "Chege",
]


def _random_date(start_days_ago: int = 365) -> str:
    delta = timedelta(days=random.randint(0, start_days_ago))
    return (datetime.today() - delta).strftime("%Y-%m-%d")


def _random_kenyan_name() -> str:
    return f"{random.choice(KENYAN_FIRST_NAMES)} {random.choice(KENYAN_LAST_NAMES)}"


def _generate_claim(is_fraud: bool) -> dict:
    procedure_code = random.choice(list(PROCEDURE_CODES.keys()))
    procedure_name, base_tariff = PROCEDURE_CODES[procedure_code]

    if is_fraud:
        # Fraud patterns:
        # - Inflate claimed amount significantly above tariff
        # - Abnormally high claim frequency
        claimed_amount  = round(base_tariff * random.uniform(1.5, 4.0), 2)
        claim_frequency = random.randint(15, 60)
    else:
        # Legitimate: small natural variance around approved tariff
        claimed_amount  = round(base_tariff * random.uniform(0.90, 1.15), 2)
        claim_frequency = random.randint(1, 10)

    return {
        "claim_id":                   str(uuid.uuid4())[:8].upper(),
        "member_id":                  f"M{random.randint(1_000_000, 9_999_999)}",
        "member_name":                _random_kenyan_name(),
        "provider_id":                f"P{random.randint(100, 999)}",
        "provider_name":              fake.company(),
        "diagnosis_code":             random.choice(DIAGNOSIS_CODES),
        "procedure_code":             procedure_code,
        "procedure_name":             procedure_name,
        "claimed_amount":             claimed_amount,
        "approved_tariff_amount":     base_tariff,
        "date_of_service":            _random_date(),
        "provider_type":              random.choice(PROVIDER_TYPES),
        "historical_claim_frequency": claim_frequency,
        "location":                   random.choice(LOCATIONS),
        "is_fraud":                   int(is_fraud),
    }


def generate_dataset(n_legitimate: int = 700, n_fraud: int = 300) -> pd.DataFrame:
    """
    Generate a synthetic claims dataset with labelled fraud cases.

    Fraud patterns baked in:
      - claimed_amount significantly above approved_tariff_amount
      - historical_claim_frequency abnormally high

    Args:
        n_legitimate: Number of legitimate (non-fraud) claims to generate.
        n_fraud:      Number of fraudulent claims to generate.

    Returns:
        Shuffled DataFrame of claims with is_fraud label column.
    """
    claims = (
        [_generate_claim(is_fraud=False) for _ in range(n_legitimate)] +
        [_generate_claim(is_fraud=True)  for _ in range(n_fraud)]
    )
    df = pd.DataFrame(claims).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    df = generate_dataset()
    df.to_csv("data/claims.csv", index=False)

    total = len(df)
    fraud = int(df["is_fraud"].sum())
    legit = total - fraud

    print(f"Generated {total} claims — {legit} legitimate, {fraud} fraud")
    print(df.head(10).to_string(index=False))