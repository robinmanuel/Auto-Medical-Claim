"""
Medical Insurance Claim Intelligence System

"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from modules.predictor import ClaimPredictor
from modules.ocr       import extract_text, OCRResult
from modules.validator import generate_verification_report
from modules.fraud     import calculate_fraud_risk

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Medical Insurance Claim Intelligence System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0c0f1a; color: #e2e8f0; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1400px; }

/* ── title ── */
.page-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #e2e8f0;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.page-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0 0 1.8rem;
    font-weight: 400;
}

/* ── KPI cards ── */
.kpi-card {
    background: #131929;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { border-color: #334155; box-shadow: 0 4px 24px rgba(0,0,0,0.35); }
.kpi-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: #64748b; margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 1.65rem; font-weight: 700; color: #e2e8f0;
    font-family: 'DM Mono', monospace;
}
.kpi-sub { font-size: 0.78rem; color: #64748b; margin-top: 0.25rem; }

/* ── section headings ── */
.section-heading {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #6366f1; margin: 0 0 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-heading::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #1e293b, transparent);
}

/* ── approval badges ── */
.status-approve {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 2px solid #16a34a; border-radius: 14px;
    padding: 1.5rem 2rem; text-align: center;
    color: #86efac; font-size: 1.5rem; font-weight: 700; letter-spacing: 0.1em;
}
.status-review {
    background: linear-gradient(135deg, #1c1204, #451a03);
    border: 2px solid #d97706; border-radius: 14px;
    padding: 1.5rem 2rem; text-align: center;
    color: #fde68a; font-size: 1.5rem; font-weight: 700; letter-spacing: 0.1em;
}
.status-reject {
    background: linear-gradient(135deg, #1c0404, #450a0a);
    border: 2px solid #dc2626; border-radius: 14px;
    padding: 1.5rem 2rem; text-align: center;
    color: #fca5a5; font-size: 1.5rem; font-weight: 700; letter-spacing: 0.1em;
}

/* ── pills ── */
.pill {
    display: inline-block; padding: 0.2rem 0.65rem;
    border-radius: 100px; font-size: 0.72rem; font-weight: 600;
    margin: 0.2rem 0.2rem 0 0;
}
.pill-green  { background: rgba(34,197,94,0.15);  color: #86efac; }
.pill-red    { background: rgba(239,68,68,0.15);   color: #fca5a5; }

/* ── divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e293b, transparent);
    margin: 1.8rem 0;
}

/* ── widgets ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stFileUploader"] label {
    color: #94a3b8 !important; font-size: 0.82rem !important; font-weight: 500 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background: #0c0f1a !important; border-color: #1e293b !important; color: #e2e8f0 !important;
}
div[data-testid="stFileUploader"] section {
    background: #0c1020 !important; border: 1px dashed #334155 !important; border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important; letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
div[data-testid="stExpander"] {
    background: #0c0f1a !important; border: 1px solid #1e293b !important; border-radius: 10px !important;
}
div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.72rem !important; }
div[data-testid="stMetric"] div   { color: #e2e8f0 !important; font-family: 'DM Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_predictor():
    return ClaimPredictor()

predictor = load_predictor()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _gauge(value, title, max_val=100, color_thresholds=None):
    steps = color_thresholds or [
        (0, 0.35*max_val, "#22c55e"),
        (0.35*max_val, 0.65*max_val, "#f59e0b"),
        (0.65*max_val, max_val, "#ef4444"),
    ]
    bar_color = "#6366f1"
    for lo, hi, col in steps:
        if lo <= value <= hi:
            bar_color = col
            break
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"font": {"color": "#e2e8f0", "family": "DM Mono", "size": 28}},
        title={"text": title, "font": {"color": "#94a3b8", "size": 13, "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#334155",
                     "tickfont": {"color": "#475569", "size": 9}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#0c0f1a", "borderwidth": 0,
            "steps": [{"range": [s[0], s[1]], "color": "rgba(255,255,255,0.03)"} for s in steps],
            "threshold": {"line": {"color": bar_color, "width": 3}, "thickness": 0.75, "value": value},
        }
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
    )
    return fig


def _bar_chart(labels, values, colors):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.0f}" for v in values], textposition="outside",
        textfont={"color": "#94a3b8", "size": 11},
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"range": [0, 110], "gridcolor": "#1e293b", "tickfont": {"color": "#475569"}},
        yaxis={"gridcolor": "#1e293b", "tickfont": {"color": "#94a3b8"}},
    )
    return fig


def _approval_color(status):
    return {"APPROVED": "#22c55e", "UNDER REVIEW": "#f59e0b", "REJECTED": "#ef4444"}.get(status, "#64748b")


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class="page-title">Medical Insurance Claim Intelligence System</p>
<p class="page-subtitle">AI-powered insurance claim verification and risk assessment</p>
""", unsafe_allow_html=True)

if not predictor.is_loaded:
    st.warning(
        f"Model not found. {predictor.load_error} "
        "Place claim_model.joblib in the model/ directory. "
        "The app will use a rule-based fallback estimator.",
    )

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "predicted_amount": 0.0, "verification_score": 0.0, "fraud_score": 0.0,
    "approval_status": "Pending", "risk_label": "-", "fraud_label": "-",
    "ver_status": "-", "analyzed": False,
    "fraud_result": None, "ver_report": None, "prediction_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Predicted Claim</div>
      <div class="kpi-value">Rs.{st.session_state.predicted_amount:,.0f}</div>
      <div class="kpi-sub">{st.session_state.risk_label}</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Verification Score</div>
      <div class="kpi-value">{st.session_state.verification_score:.0f}<span style="font-size:1rem;color:#64748b">/100</span></div>
      <div class="kpi-sub">{st.session_state.ver_status}</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[2]:
    fc = _approval_color("APPROVED") if st.session_state.fraud_score < 35 else (
         _approval_color("UNDER REVIEW") if st.session_state.fraud_score < 65 else
         _approval_color("REJECTED"))
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Fraud Risk</div>
      <div class="kpi-value" style="color:{fc}">{st.session_state.fraud_score:.0f}<span style="font-size:1rem;color:#64748b">/100</span></div>
      <div class="kpi-sub">{st.session_state.fraud_label}</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[3]:
    ac = _approval_color(st.session_state.approval_status)
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Approval Status</div>
      <div class="kpi-value" style="color:{ac};font-size:1.2rem">{st.session_state.approval_status}</div>
      <div class="kpi-sub">{'Model active' if predictor.is_loaded else 'Fallback mode'}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1.1, 1.9], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT — FORM
# ─────────────────────────────────────────────────────────────────────────────
with left_col:
    st.markdown('<p class="section-heading">Patient Information</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        age              = st.slider("Age", 18, 90, 35)
        weight           = st.slider("Weight (kg)", 30, 150, 70)
        no_of_dependents = st.slider("Dependents", 0, 10, 1)
    with col_b:
        bmi           = st.slider("BMI", 10.0, 55.0, 22.5, step=0.1)
        sex           = st.selectbox("Gender", ["Male", "Female"])
        bloodpressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])

    col_c, col_d = st.columns(2)
    with col_c:
        smoker     = st.selectbox("Smoker",           ["No", "Yes"])
        diabetes   = st.selectbox("Diabetes",         ["No", "Yes"])
        regular_ex = st.selectbox("Regular Exercise", ["No", "Yes"])
    with col_d:
        job_title = st.selectbox("Job Title", [
            "Engineer", "Doctor", "Teacher", "Lawyer", "Accountant",
            "Manager", "Nurse", "Driver", "Businessman", "Artist",
            "Scientist", "Technician", "Architect", "Police", "Other",
        ])
        city = st.selectbox("City", [
            "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
            "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Other",
        ])
        hereditary_diseases = st.selectbox("Hereditary Disease", [
            "None", "Diabetes", "Heart Disease", "Cancer", "Hypertension",
            "Epilepsy", "Alzheimer's", "Arthritis", "Other",
        ])

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Document Upload — medical bill only ──────────────────────────────────
    st.markdown('<p class="section-heading">Document Upload</p>', unsafe_allow_html=True)
    medical_bill = st.file_uploader("Medical Bill", type=["png", "jpg", "jpeg", "webp"], key="bill")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyse Claim", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with right_col:

    if analyze_btn:
        form_data = {
            "age": age, "sex": sex, "weight": weight, "bmi": bmi,
            "no_of_dependents": no_of_dependents, "smoker": smoker,
            "bloodpressure": bloodpressure, "diabetes": diabetes,
            "regular_ex": regular_ex, "job_title": job_title,
            "city": city, "hereditary_diseases": hereditary_diseases,
        }

        with st.spinner("Running analysis..."):
            # 1. Prediction
            pred = predictor.predict(form_data)

            # 2. OCR — medical bill only
            ocr_results: dict[str, OCRResult] = {}
            if medical_bill is not None:
                ocr_results["medical_bill"] = extract_text(medical_bill)

            # 3. Verification — medical bill only
            doc_input = {
                "medical_bill": {
                    "uploaded": medical_bill is not None,
                    "ocr": ocr_results.get("medical_bill"),
                },
                "prescription": {"uploaded": False, "ocr": None},
                "claim_form":   {"uploaded": False, "ocr": None},
            }
            ver_report = generate_verification_report(doc_input)

            # 4. Fraud
            fraud_result = calculate_fraud_risk(form_data, pred["predicted_amount"])

            # 5. Approval
            vs = ver_report.overall_score
            fr = fraud_result.score
            risk_label = pred["risk_label"]
            
            # Higher approval rate for low claims
            if risk_label == "Low Claim":
                if vs >= 70 and fr < 50:
                    approval = "APPROVED"
                elif fr > 75 or vs < 40:
                    approval = "REJECTED"
                else:
                    approval = "UNDER REVIEW"
            else:
                # Standard thresholds for medium/high claims
                if vs >= 80 and fr < 40:
                    approval = "APPROVED"
                elif fr > 70 or vs < 50:
                    approval = "REJECTED"
                else:
                    approval = "UNDER REVIEW"

            st.session_state.update({
                "predicted_amount":   pred["predicted_amount"],
                "verification_score": vs,
                "fraud_score":        fr,
                "approval_status":    approval,
                "risk_label":         pred["risk_label"],
                "fraud_label":        fraud_result.label,
                "ver_status":         ver_report.status,
                "analyzed":           True,
                "fraud_result":       fraud_result,
                "ver_report":         ver_report,
                "prediction_result":  pred,
                "ocr_results":        ocr_results,
            })

        st.rerun()

    if st.session_state.analyzed:
        fraud_result = st.session_state.fraud_result
        ver_report   = st.session_state.ver_report
        pred         = st.session_state.prediction_result
        ocr_results  = st.session_state.get("ocr_results", {})

        # Approval banner
        status  = st.session_state.approval_status
        css_map = {"APPROVED": "status-approve", "UNDER REVIEW": "status-review", "REJECTED": "status-reject"}
        st.markdown(f"""
        <div class="{css_map.get(status, 'status-review')}">
          {status}
          <div style="font-size:0.85rem;font-weight:400;margin-top:0.4rem;opacity:0.8">
            {fraud_result.recommendation if fraud_result else ''}
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauges
        st.markdown('<p class="section-heading">Dashboard Analytics</p>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(
                _gauge(st.session_state.fraud_score, "Fraud Risk Score"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with g2:
            st.plotly_chart(
                _gauge(
                    st.session_state.verification_score, "Verification Score",
                    color_thresholds=[(0, 50, "#ef4444"), (50, 80, "#f59e0b"), (80, 100, "#22c55e")],
                ),
                use_container_width=True, config={"displayModeBar": False},
            )
        with g3:
            max_claim = 150_000
            st.plotly_chart(
                _gauge(
                    min(st.session_state.predicted_amount, max_claim),
                    "Claim Amount (Rs.)",
                    max_val=max_claim,
                    color_thresholds=[
                        (0, 50_000, "#22c55e"),
                        (50_000, 150_000, "#f59e0b"),
                        (150_000, max_claim, "#ef4444"),
                    ],
                ),
                use_container_width=True, config={"displayModeBar": False},
            )

        # Risk breakdown
        st.markdown('<p class="section-heading">Risk Breakdown</p>', unsafe_allow_html=True)
        if fraud_result:
            rule_labels = [r["rule"][:40] for r in fraud_result.triggered_rules]
            rule_values = [r["weight"] if r["triggered"] else 0 for r in fraud_result.triggered_rules]
            rule_colors = ["#ef4444" if r["triggered"] else "#1e293b" for r in fraud_result.triggered_rules]
            st.plotly_chart(
                _bar_chart(rule_labels, rule_values, rule_colors),
                use_container_width=True, config={"displayModeBar": False},
            )

        # Prediction details
        st.markdown('<p class="section-heading">Claim Prediction</p>', unsafe_allow_html=True)
        pm1, pm2, pm3 = st.columns(3)
        with pm1:
            st.metric("Predicted Amount", f"Rs.{pred['predicted_amount']:,.0f}")
        with pm2:
            st.metric("Risk Category", pred["risk_label"])
        with pm3:
            st.metric("Model", "Loaded" if pred.get("model_loaded") else "Fallback")

        # Verification
        st.markdown('<p class="section-heading">Document Verification</p>', unsafe_allow_html=True)
        if ver_report:
            vc1, vc2 = st.columns([1, 2])
            with vc1:
                st.metric("Score", f"{ver_report.overall_score:.0f} / 100")
                st.metric("Status", ver_report.status)
            with vc2:
                bill_issues = [i for i in ver_report.issues if "bill" in i.lower() or "medical" in i.lower()]
                if bill_issues:
                    for issue in bill_issues[:3]:
                        st.markdown(f'<span class="pill pill-red">{issue}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="pill pill-green">Medical bill verified</span>', unsafe_allow_html=True)

        # OCR text
        if ocr_results:
            st.markdown('<p class="section-heading">Extracted Text (OCR)</p>', unsafe_allow_html=True)
            res = ocr_results.get("medical_bill")
            if res:
                with st.expander(f"Medical Bill — confidence {res.confidence_pct}%  |  {res.word_count} words"):
                    if res.success and res.text:
                        st.code(res.text, language=None)
                    elif res.error:
                        st.warning(res.error)
                    else:
                        st.info("No text detected.")

    else:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center;
            justify-content:center; height:420px;
            background:#0c1020; border:1px dashed #1e293b;
            border-radius:14px; color:#334155; text-align:center; padding:2rem;">
          <div style="font-size:1.1rem;font-weight:600;color:#475569;margin-bottom:0.5rem">
            No analysis yet
          </div>
          <div style="font-size:0.85rem;color:#334155;max-width:320px;line-height:1.6">
            Fill in the patient information form, optionally upload a medical bill,
            then click <strong style="color:#6366f1">Analyse Claim</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;color:#1e293b;font-size:0.72rem;
     margin-top:3rem;padding-top:1.5rem;border-top:1px solid #0f172a;">
  Medical Insurance Claim Intelligence System  |
  For internal use only  |
  All predictions are probabilistic and require human review.
</div>
""", unsafe_allow_html=True)
