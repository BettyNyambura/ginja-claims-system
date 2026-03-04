# Ginja Claims Engine

An AI-powered healthcare claims adjudication prototype built for Ginja AI / Eden Care.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │           FastAPI Application        │
                        │                                      │
          JSON payload  │  POST /api/v1/adjudicate             │
          CSV / JSON    │  POST /api/v1/adjudicate/batch       │
          PDF invoice   │  POST /api/v1/adjudicate/pdf         │
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │         Decision Engine              │
                        │   (app/core/decision.py)            │
                        │                                      │
                        │  1. Rule Engine  →  hard override?  │
                        │  2. ML Model     →  risk score 0–1  │
                        │  3. Soft nudge   →  rule-adjusted   │
                        │  4. Threshold    →  PASS/FLAG/FAIL  │
                        └──────────────┬──────────────────────┘
                                       │
               ┌───────────────────────┼────────────────────────┐
               │                       │                        │
   ┌───────────▼──────┐   ┌────────────▼──────────┐  ┌────────▼────────────┐
   │  Rule Engine      │   │   ML Model (XGBoost)  │  │  PDF Parser         │
   │  (rules.py)       │   │   (model.py)          │  │  (pdf_parser.py)    │
   │                   │   │                       │  │                     │
   │  Hard rules:      │   │  Feature engineering  │  │  PyMuPDF extraction │
   │  >200% tariff     │   │  → XGBoost classifier │  │  Regex field parsing│
   │  >40 freq         │   │  → probability 0–1    │  │  Validation         │
   └───────────────────┘   └───────────────────────┘  └─────────────────────┘
```

---

## Model Explanation

### Why XGBoost?

XGBoost was chosen as the primary classifier for several reasons:

- **Tabular data performance**: XGBoost consistently outperforms other models on structured tabular data, which is exactly what claims data is.
- **Probability output**: It natively produces a calibrated probability score (0–1), which maps directly to the PASS/FLAG/FAIL thresholds.
- **Built-in feature importance**: Provides SHAP-compatible feature importances for explainability without additional tooling.
- **Handles class imbalance**: The `scale_pos_weight` parameter adjusts for the 70/30 legitimate-to-fraud split.
- **Fast training**: Trains in seconds on modest hardware — no GPU required.

### Decision Thresholds

| Risk Score | Decision |
|------------|----------|
| 0.00 – 0.30 | ✅ PASS |
| 0.30 – 0.70 | ⚠️ FLAG |
| 0.70 – 1.00 | ❌ FAIL |

Hard rules (e.g. claimed amount > 3× tariff) **override** the ML score entirely.

---

## Feature Engineering

| Feature | Why it matters |
|---|---|
| `amount_deviation_pct` | % deviation of claimed amount from approved tariff. The strongest fraud signal — fraudulent providers inflate claims above tariff. |
| `is_above_tariff` | Binary flag. Even small overages warrant attention. |
| `high_frequency_flag` | Binary flag for abnormally high claim frequency. Claim mills and ghost patients generate high volumes. |
| `frequency_x_deviation` | Interaction feature. A provider billing 30% above tariff once is different from doing it 40 times. |
| `provider_type_encoded` | Ordinal risk encoding. Pharmacies and labs have historically higher misbilling rates in emerging markets. |
| `location_encoded` | Geographic encoding. Allows the model to learn regional fraud patterns over time. |
| `log_claimed_amount` | Log-transformed amount to reduce skew from high-value procedures (surgeries) dominating gradient updates. |
| `historical_claim_frequency` | Raw frequency. High absolute values are suspicious independent of deviation. |

---

## Assumptions & Trade-offs

**Synthetic training data**: No real historical claims were available. The model is trained on synthetic data with realistic Kenyan healthcare pricing and fraud patterns baked in. In production, this would be replaced with labelled historical claims from Eden Care's database.

**Tariff data**: Approved tariff amounts are hardcoded per procedure code. In production, these would be pulled from a dynamic tariff schedule API (NHIF rates or insurer-specific schedules).

**PDF extraction**: The parser is calibrated to Nairobi Lifecare-style invoice PDFs. Other hospital formats will require regex pattern additions.

**No duplicate detection across requests**: The current prototype processes each claim independently. Production would require a persistent store to detect duplicate claim IDs or member–procedure–date combinations.

**Model retraining**: The model is trained once at startup. A production system would retrain on a schedule (weekly/monthly) as new labelled data accumulates, using a pipeline like MLflow or Prefect.

---

## How to Run

### Local

```bash
# 1. Clone and install dependencies
git clone https://github.com/your-username/ginja-claims-engine
cd ginja-claims-engine
pip install -r requirements.txt

# 2. Train the model
python -m app.core.model

# 3. Start the API
uvicorn app.main:app --reload
```

API will be available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### Docker

```bash
docker build -t ginja-claims-engine .
docker run -p 8000:8000 ginja-claims-engine
```

---

## Example Request

```bash
curl -X POST http://localhost:8000/api/v1/adjudicate \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "C001",
    "member_id": "M1538500",
    "provider_id": "P120",
    "diagnosis_code": "J06.9",
    "procedure_code": "99213",
    "claimed_amount": 15000,
    "approved_tariff_amount": 1500,
    "date_of_service": "2026-02-12",
    "provider_type": "Clinic",
    "historical_claim_frequency": 35,
    "location": "Nairobi"
  }'
```

Expected response:

```json
{
  "claim_id": "C001",
  "risk_score": 0.97,
  "decision": "FAIL",
  "confidence": 0.97,
  "reason": "Claimed amount exceeds tariff by 900.0% (threshold: 200%)",
  "source": "rule_engine"
}
```

---

## Potential Improvements

- **SHAP explainability**: Replace simple feature importance with SHAP values for per-claim explanations.
- **Duplicate detection**: Add Redis-backed claim deduplication across requests.
- **Dynamic tariff schedule**: Pull approved tariffs from a live NHIF/insurer API instead of hardcoded values.
- **Model monitoring**: Track prediction drift and data drift using Evidently AI.
- **Active learning**: Flag borderline cases (FLAG zone) for human review and feed reviewed labels back into retraining.
- **Multi-model ensemble**: Combine XGBoost with an Isolation Forest for unsupervised anomaly detection on truly novel fraud patterns.
- **Streamlit dashboard**: Visual claims review interface for human adjudicators.