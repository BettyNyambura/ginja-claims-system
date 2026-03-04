import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os
import urllib3
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

DECISION_COLORS = {
    "PASS": "#0d9488",
    "FLAG": "#b45309",
    "FAIL": "#b91c1c",
}

DECISION_BG = {
    "PASS": "#f0fdfa",
    "FLAG": "#fffbeb",
    "FAIL": "#fff1f2",
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()
session.verify = False
# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ginja Claims Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styles ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8fafc; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e3a5f;
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label { color: #cbd5e1 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: white !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Buttons */
    .stButton > button {
        background-color: #1e3a5f;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #2d5282; }

    /* Form submit button */
    .stFormSubmitButton > button {
        background-color: #1e3a5f;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }

    /* Section headers */
    h2, h3 { color: #1e3a5f; }

    /* Download button */
    .stDownloadButton > button {
        background-color: #0d9488;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def risk_gauge(score: float, decision: str):
    color = DECISION_COLORS[decision]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score * 100, 1),
        delta={"reference": 30, "valueformat": ".1f"},
        number={"suffix": "%", "font": {"size": 34, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#f0fdfa"},
                {"range": [30, 70], "color": "#fffbeb"},
                {"range": [70, 100], "color": "#fff1f2"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score * 100,
            },
        },
        title={"text": "Risk Score", "font": {"size": 16, "color": "#64748b"}},
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor="white",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def decision_card(decision: str, confidence: float):
    color  = DECISION_COLORS[decision]
    bg     = DECISION_BG[decision]
    labels = {"PASS": "Approved", "FLAG": "Under Review", "FAIL": "Rejected"}
    label  = labels[decision]
    st.markdown(f"""
    <div style="
        background: {bg};
        border-left: 5px solid {color};
        border-radius: 10px;
        padding: 24px 28px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    ">
        <p style="margin:0; font-size:0.85rem; color:#64748b; letter-spacing:1px; text-transform:uppercase; font-weight:600">Decision</p>
        <p style="margin:4px 0 0; font-size:2rem; font-weight:800; color:{color}">{decision}</p>
        <p style="margin:2px 0 0; font-size:1rem; color:{color}; opacity:0.8">{label}</p>
        <hr style="border-color:{color}33; margin:12px 0">
        <p style="margin:0; font-size:0.9rem; color:#475569">Confidence: <strong style="color:{color}">{confidence*100:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)


def show_result(result: dict):
    st.markdown("---")
    st.markdown("### Adjudication Result")

    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        decision_card(result["decision"], result["confidence"])

    with col2:
        st.plotly_chart(risk_gauge(result["risk_score"], result["decision"]), width="stretch")

    with col3:
        st.markdown("""
        <div style="background:white; border-radius:10px; padding:20px 24px;
                    border:1px solid #e2e8f0; box-shadow:0 1px 3px rgba(0,0,0,0.06); height:100%">
        """, unsafe_allow_html=True)
        st.markdown(f"**Claim ID** &nbsp; `{result['claim_id']}`")
        st.markdown(f"**Risk Score** &nbsp; `{result['risk_score']}`")
        st.markdown(f"**Evaluated by** &nbsp; `{result['source']}`")
        st.markdown("**Reason**")
        color = DECISION_COLORS[result["decision"]]
        st.markdown(f"""
        <div style="background:{DECISION_BG[result['decision']]};
                    border:1px solid {color}44;
                    border-radius:8px; padding:12px 16px;
                    color:#334155; font-size:0.92rem; line-height:1.6">
            {result['reason']}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def color_decision(val):
    colors = {"PASS": "#0d9488", "FLAG": "#b45309", "FAIL": "#b91c1c"}
    return f"color: {colors.get(val, 'black')}; font-weight: 700"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px">
        <p style="font-size:1.4rem; font-weight:800; margin:0; letter-spacing:0.5px">Ginja Claims</p>
        <p style="font-size:0.8rem; opacity:0.7; margin:0">AI-powered adjudication engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**API Status**")
    try:
        r = session.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            st.markdown("""
            <div style="background:#134e4a; border-radius:6px; padding:8px 12px; font-size:0.85rem">
                API online
            </div>""", unsafe_allow_html=True)
        else:
            st.error("API error")
    except Exception:
        st.error("API offline — run uvicorn first")

    st.divider()
    st.markdown("**Thresholds**")
    st.markdown("""
    <div style="font-size:0.88rem; line-height:2">
        <span style="color:#5eead4">■</span> PASS &nbsp; score &lt; 0.30<br>
        <span style="color:#fcd34d">■</span> FLAG &nbsp; score 0.30 – 0.70<br>
        <span style="color:#fca5a5">■</span> FAIL &nbsp; score &gt; 0.70
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Filters**")
    provider_filter = st.selectbox("Provider Type", ["All", "Clinic", "Pharmacy", "Hospital", "Lab", "Specialist"])
    decision_filter = st.multiselect("Decision", ["PASS", "FLAG", "FAIL"], default=["PASS", "FLAG", "FAIL"])


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5282 100%);
            border-radius: 14px; padding: 28px 36px; margin-bottom: 24px;
            box-shadow: 0 4px 16px rgba(30,58,95,0.15)">
    <p style="margin:0; color:#93c5fd; font-size:0.85rem; letter-spacing:1.5px; text-transform:uppercase; font-weight:600">Eden Care</p>
    <h1 style="margin:4px 0 0; color:white; font-size:2rem; font-weight:800">Claims Adjudication Dashboard</h1>
    <p style="margin:6px 0 0; color:#bfdbfe; font-size:0.95rem">Real-time AI-powered claims intelligence for emerging markets</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["  Single Claim  ", "  PDF Invoice  ", "  Batch Upload  "])


# ── Tab 1: Single Claim ───────────────────────────────────────────────────────

with tab1:
    st.markdown("### Submit a Claim for Adjudication")
    st.caption("Enter claim details below. All amounts in Kenyan Shillings (KES).")

    with st.form("single_claim_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Claim & Member Info**")
            claim_id    = st.text_input("Claim ID",                value="C001")
            member_id   = st.text_input("Member ID",               value="M1538500")
            provider_id = st.text_input("Provider ID",             value="P120")
            diagnosis   = st.text_input("Diagnosis Code (ICD-10)", value="K02.1")
            procedure   = st.text_input("Procedure Code (CPT)",    value="99213")
            location    = st.selectbox("Location", ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika"])

        with col2:
            st.markdown("**Financial & Provider Info**")
            claimed_amount  = st.number_input("Claimed Amount (KES)",         value=4300.0, min_value=1.0)
            tariff_amount   = st.number_input("Approved Tariff Amount (KES)", value=4300.0, min_value=1.0)
            claim_frequency = st.number_input("Historical Claim Frequency",   value=3,      min_value=0, step=1)
            date_of_service = st.date_input("Date of Service")
            provider_type   = st.selectbox("Provider Type", ["Hospital", "Clinic", "Pharmacy", "Lab", "Specialist"])

        st.form_submit_button("Run Adjudication", width="stretch")

    if st.session_state.get("FormSubmitter:single_claim_form-Run Adjudication"):
        payload = {
            "claim_id":                   claim_id,
            "member_id":                  member_id,
            "provider_id":                provider_id,
            "diagnosis_code":             diagnosis,
            "procedure_code":             procedure,
            "claimed_amount":             claimed_amount,
            "approved_tariff_amount":     tariff_amount,
            "date_of_service":            str(date_of_service),
            "provider_type":              provider_type,
            "historical_claim_frequency": claim_frequency,
            "location":                   location,
        }
        with st.spinner("Scoring claim..."):
            try:
                r = session.post(f"{API_BASE}/adjudicate", json=payload, timeout=10)
                if r.status_code == 200:
                    show_result(r.json())
                    with st.expander("Raw JSON"):
                        st.json(r.json())
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Could not reach API: {e}")


# ── Tab 2: PDF Invoice ────────────────────────────────────────────────────────

with tab2:
    st.markdown("### Adjudicate from PDF Invoice")
    st.caption("Upload a system-generated PDF invoice. Scanned documents require digital re-submission.")

    uploaded_pdf = st.file_uploader("Upload PDF Invoice", type=["pdf"])

    if uploaded_pdf:
        st.markdown(f"""
        <div style="background:white; border:1px solid #e2e8f0; border-radius:8px;
                    padding:12px 16px; margin:8px 0; font-size:0.9rem; color:#475569">
            {uploaded_pdf.name} &nbsp;·&nbsp; {uploaded_pdf.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

        if st.button("Extract and Adjudicate", width="stretch"):
            with st.spinner("Extracting and scoring..."):
                try:
                    r = session.post(
                        f"{API_BASE}/adjudicate/pdf",
                        files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")},
                        timeout=30,
                    )
                    if r.status_code == 200:
                        data = r.json()

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Extracted Fields")
                            extracted = data.get("extracted", {})
                            for k, v in extracted.items():
                                if k not in ["raw_text", "line_items", "extraction_method"]:
                                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

                        with col2:
                            st.markdown("#### Line Items")
                            items = extracted.get("line_items", [])
                            if items:
                                st.dataframe(pd.DataFrame(items), width="stretch")
                            else:
                                st.caption("No line items extracted")

                        show_result(data["adjudication"])

                        with st.expander("Raw JSON"):
                            st.json(data)
                    else:
                        st.error(f"API error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")


# ── Tab 3: Batch Upload ───────────────────────────────────────────────────────

with tab3:
    st.markdown("### Batch Claims Adjudication")
    st.caption("Upload a CSV or JSON file of claims to score in bulk.")

    uploaded_batch = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])

    if uploaded_batch:
        st.markdown(f"""
        <div style="background:white; border:1px solid #e2e8f0; border-radius:8px;
                    padding:12px 16px; margin:8px 0; font-size:0.9rem; color:#475569">
            {uploaded_batch.name} &nbsp;·&nbsp; {uploaded_batch.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run Batch Adjudication", width="stretch"):
            with st.spinner("Scoring all claims..."):
                try:
                    r = session.post(
                        f"{API_BASE}/adjudicate/batch",
                        files={"file": (uploaded_batch.name, uploaded_batch.getvalue())},
                        timeout=60,
                    )
                    if r.status_code == 200:
                        data    = r.json()
                        results = data["results"]
                        df      = pd.DataFrame(results)
                        counts  = df["decision"].value_counts()

                        # Metric cards
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Claims", data["total"])
                        col2.metric("Approved",     counts.get("PASS", 0))
                        col3.metric("Under Review", counts.get("FLAG", 0))
                        col4.metric("Rejected",     counts.get("FAIL", 0))

                        st.markdown("---")

                        # Results table
                        st.markdown("#### Results")
                        styled = df.style.map(color_decision, subset=["decision"])
                        st.dataframe(styled, width="stretch", height=380)

                        col_dl, col_chart = st.columns([1, 2])

                        with col_dl:
                            st.download_button(
                                label="Download Results CSV",
                                data=df.to_csv(index=False),
                                file_name="adjudication_results.csv",
                                mime="text/csv",
                                width="stretch",
                            )

                        with col_chart:
                            fig = go.Figure(go.Pie(
                                labels=list(counts.index),
                                values=list(counts.values),
                                marker_colors=[DECISION_COLORS.get(d, "#888") for d in counts.index],
                                hole=0.45,
                                textinfo="label+percent",
                                textfont_size=13,
                            ))
                            fig.update_layout(
                                title=dict(text="Decision Distribution", font=dict(size=15, color="#1e3a5f")),
                                height=320,
                                margin=dict(t=40, b=0, l=0, r=0),
                                paper_bgcolor="rgba(0,0,0,0)",
                                showlegend=False,
                            )
                            st.plotly_chart(fig, width="stretch")

                        with st.expander("Raw JSON"):
                            st.json(data)
                    else:
                        st.error(f"API error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top:48px; padding:16px 0; border-top:1px solid #e2e8f0;
            text-align:right; color:#94a3b8; font-size:0.82rem">
    Ginja AI &nbsp;·&nbsp; Eden Care Medical &nbsp;·&nbsp; Nairobi, Kenya
</div>
""", unsafe_allow_html=True)