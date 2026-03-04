# Ginja Claims Engine

An AI-powered healthcare claims adjudication prototype built for Ginja AI / Eden Care.

Designed to make healthcare payments faster, more accurate, and fraud-resistant across emerging markets.

---

## Project Structure

```
ginja-claims-engine/
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints
│   ├── core/
│   │   ├── features.py            # Feature engineering
│   │   ├── model.py               # XGBoost training & inference
│   │   ├── rules.py               # Deterministic rule engine
│   │   └── decision.py            # Combines ML + rules → final output
│   ├── data/
│   │   └── generator.py           # Synthetic claims data generator
│   ├── extraction/
│   │   └── pdf_parser.py          # PDF extraction (PyMuPDF + Groq vision)
│   └── main.py                    # FastAPI app entry point
├── models/                        # Saved model and scaler (auto-generated)
├── data/                          # Generated claims CSV (auto-generated)
├── dashboard.py                   # Streamlit dashboard
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Architecture

```
                     ┌──────────────────────────────────────────┐
                     │             Input Layer                   │
                     │                                           │
       JSON payload  │  POST /api/v1/adjudicate                 │
       CSV / JSON    │  POST /api/v1/adjudicate/batch            │
       PDF invoice   │  POST /api/v1/adjudicate/pdf             │
                     └─────────────────┬────────────────────────┘
                                       │
                     ┌─────────────────▼────────────────────────┐
                     │            PDF Extraction                 │
                     │         (app/extraction/pdf_parser.py)   │
                     │                                           │
                     │  Digital PDF → PyMuPDF (fast, free)      │
                     │  Scanned/Handwritten → Groq Vision LLM   │
                     └─────────────────┬────────────────────────┘
                                       │
                     ┌─────────────────▼────────────────────────┐
                     │           Decision Engine                 │
                     │         (app/core/decision.py)           │
                     │                                           │
                     │  1. Rule Engine  →  hard override?       │
                     │  2. ML Model     →  risk score 0–1       │
                     │  3. Soft nudge   →  rule-adjusted score  │
                     │  4. Threshold    →  PASS / FLAG / FAIL   │
                     └──────┬──────────────────┬────────────────┘
                            │                  │
          ┌─────────────────▼──┐    ┌──────────▼─────────────────┐
          │   Rule Engine       │    │   ML Model (XGBoost)        │
          │   (rules.py)        │    │   (model.py + features.py)  │
          │                     │    │                              │
          │  >200% tariff → FAIL│    │  8 engineered features       │
          │  >40 freq → FAIL    │    │  probability score 0–1       │
          │  Soft flags → nudge │    │  feature importance reasons  │
          └─────────────────────┘    └──────────────────────────────┘
```

---

## Model Explanation

### Why XGBoost?

XGBoost was chosen as the primary classifier for the following reasons:

- **Tabular data performance**: XGBoost consistently outperforms other algorithms on structured tabular data, which is exactly what healthcare claims are.
- **Calibrated probability output**: It natively produces a probability score (0–1) that maps directly to the PASS/FLAG/FAIL thresholds without additional calibration.
- **Built-in feature importance**: Provides interpretable feature importances per prediction, supporting the explainability requirement.
- **Class imbalance handling**: The `scale_pos_weight` parameter corrects for the 70/30 legitimate-to-fraud split in the training data.
- **No GPU required**: Trains in seconds on a standard laptop — critical for a prototype that needs to run anywhere.

### Decision Thresholds

| Risk Score   | Decision | Meaning                              |
|--------------|----------|--------------------------------------|
| 0.00 – 0.30  | PASS     | Claim approved for payment           |
| 0.30 – 0.70  | FLAG     | Claim held for manual review         |
| 0.70 – 1.00  | FAIL     | Claim rejected                       |

Hard rules (e.g. claimed amount > 3× tariff, frequency > 40) **override** the ML score entirely and return an immediate FAIL regardless of the model's output.

### Hybrid Rule + ML Decisioning

The decision engine runs in two layers:

1. **Hard rules** are evaluated first. If triggered, they bypass the ML model entirely and return an immediate decision with a human-readable reason.
2. **Soft rules** (e.g. amount 30% above tariff) do not override the model — instead they apply a small upward nudge to the ML risk score, pushing borderline legitimate claims into the FLAG zone for human review.
3. The **ML model** handles the remaining cases, returning a probability score with the top contributing features as the explanation.

---

## Feature Engineering

| Feature | Why it matters |
|---|---|
| `amount_deviation_pct` | Percentage deviation of claimed amount from approved tariff. The single strongest fraud signal — fraudulent providers systematically inflate claims above the approved rate. |
| `is_above_tariff` | Binary flag. Even small overages are a weak signal worth capturing independently of the magnitude. |
| `high_frequency_flag` | Binary flag for abnormally high claim frequency (> 12). Claim mills and ghost patient schemes generate unusually high submission volumes. |
| `frequency_x_deviation` | Interaction feature combining frequency and deviation. A provider billing 40% above tariff once differs significantly from doing so 40 times. |
| `provider_type_encoded` | Ordinal risk encoding by provider type. Pharmacies and labs carry higher misbilling risk in emerging market contexts. |
| `location_encoded` | Geographic label encoding. Enables the model to learn regional fraud patterns from historical data over time. |
| `log_claimed_amount` | Log-transformed claimed amount. Reduces skew introduced by high-value procedures (e.g. surgeries at KES 180,000) dominating gradient updates. |
| `historical_claim_frequency` | Raw submission frequency as a standalone feature, capturing absolute volume independently of the binary flag. |

---

## PDF Extraction Pipeline

The parser supports two document types automatically:

**Digital PDFs** (system-generated invoices like Nairobi Lifecare):
- Extracted using PyMuPDF — fast, free, no external API call
- Regex patterns parse member ID, name, date, line items, and total

**Scanned or handwritten documents** (e.g. Eden Care claim forms):
- Detected when PyMuPDF returns fewer than 50 characters
- Falls back to Groq's `llama-4-scout-17b-16e-instruct` vision model
- Sends a compressed JPEG of each page with a structured JSON prompt
- Returns clean, typed fields regardless of handwriting quality

---

## Assumptions & Trade-offs

**Synthetic training data**: No real historical claims were available for this prototype. The model is trained on 1,000 synthetic claims generated with realistic Kenyan healthcare pricing (KES), ICD-10 diagnosis codes, and baked-in fraud patterns. In production this would be replaced with labelled historical claims from Eden Care's database.

**Hardcoded tariff schedule**: Approved tariff amounts are defined per CPT procedure code in the generator. In production these would be pulled from a live NHIF tariff schedule or insurer-specific rate API.

**No cross-request duplicate detection**: Claims are scored independently per request. A production system would require a persistent store (e.g. PostgreSQL or Redis) to detect duplicate claim IDs or identical member–procedure–date combinations across submissions.

**Groq vision for scanned docs**: The Groq API is free tier but rate-limited. For high-volume production use, a self-hosted vision model (e.g. LLaVA via Ollama) or a dedicated document AI service would be more appropriate.

**Model retraining**: The model is trained once at container build time. A production system would retrain on a weekly or monthly schedule as new labelled data accumulates, orchestrated through a pipeline such as Prefect or Apache Airflow with experiment tracking via MLflow.

**SSL verification**: SSL verification is disabled in the dashboard for local development. In production, all clients connect over verified HTTPS via Google-managed certificates on Cloud Run.

---

## How to Run

### Prerequisites

- Python 3.11+
- A Groq API key (free at https://console.groq.com) — only needed for scanned PDF extraction

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ginja-claims-engine
cd ginja-claims-engine

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
touch .env
# Edit .env and add your GROQ_API_KEY

# 5. Train the model
python -m app.core.model

# 6. Start the API
uvicorn app.main:app --reload
```

API available at: `http://localhost:8000`
Interactive API docs: `http://localhost:8000/docs`

### Streamlit Dashboard

In a second terminal:

```bash
streamlit run dashboard.py
```

Dashboard available at: `http://localhost:8501`

### Docker

```bash
# Build and start everything
docker-compose up --build

# Stop
docker-compose down
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/adjudicate` | Adjudicate a single claim (JSON payload) |
| POST | `/api/v1/adjudicate/batch` | Adjudicate a batch of claims (CSV or JSON file) |
| POST | `/api/v1/adjudicate/pdf` | Extract and adjudicate from a PDF invoice |
| GET  | `/api/v1/health` | Health check |

---

## Example Request & Response

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

## Model Retraining Strategy

In production the model would retrain on the following schedule:

1. **Trigger**: Weekly batch job or when labelled claim volume exceeds 500 new records
2. **Data pipeline**: New claims reviewed by human adjudicators are labelled and appended to the training dataset
3. **Training**: XGBoost retrained with the full updated dataset, hyperparameters tuned via cross-validation
4. **Validation**: New model must achieve AUC-ROC > 0.90 and precision on FAIL class > 0.85 before promotion
5. **Deployment**: Blue-green model swap — new model loaded into a staging slot, shadow-scored against live traffic for 24 hours before full cutover
6. **Monitoring**: Prediction drift tracked weekly using Evidently AI; data drift alerts trigger retraining outside the normal schedule

---

## Potential Improvements

- **SHAP explainability**: Replace feature importance with SHAP values for per-claim, per-feature contribution scores visible in the dashboard.
- **Duplicate detection**: Redis-backed deduplication across requests using claim ID + member ID + date + procedure as a composite key.
- **Dynamic tariff schedule**: Live integration with NHIF tariff API or insurer rate tables instead of hardcoded procedure costs.
- **Isolation Forest**: Add an unsupervised anomaly detector running in parallel to catch novel fraud patterns not present in training data.
- **Active learning loop**: Claims in the FLAG zone routed to human reviewers; their decisions fed back as labels for the next retraining cycle.
- **Multi-tenant support**: Provider and insurer-specific models trained on their own historical data for higher precision.
- **Cloud deployment**: Container deployed to GCP Cloud Run or AWS ECS with auto-scaling and a managed PostgreSQL instance for claim history.

---

## Evaluation Notes

This prototype was built to demonstrate real production thinking, not just a working demo. Key engineering decisions made deliberately:

- Hard rules run before ML to prevent the model from overriding obvious policy violations
- Synthetic data fraud patterns are grounded in real healthcare fraud research (tariff inflation, claim mills, ghost patients)
- PDF extraction degrades gracefully — digital PDFs use a fast local parser, scanned documents escalate to a vision model, and failures return clear validation errors rather than silent bad data
- The decision reason field is always computed from the actual claim values, not generic feature names, so every output is human-readable and auditable