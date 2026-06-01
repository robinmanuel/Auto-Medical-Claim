# Medical Insurance Claim Intelligence System

> **AI-powered insurance claim verification and risk assessment**

---

## Features

| Module | Description |
|---|---|
| **Claim Prediction** | Loads a pre-trained `joblib` model and predicts the expected claim amount |
| **Document Verification** | EasyOCR extracts text from uploaded images and scores completeness (0-100) |
| **Fraud Detection** | Rule-based engine assigns a 0-100 risk score across 11 weighted risk signals |
| **Approval Engine** | Business-rule decision: APPROVED / UNDER REVIEW / REJECTED |
| **Dashboard** | Plotly gauges, bar charts, KPI cards — everything on one page |

---

## Project Structure

```
medical_insurance/
├── app.py                      # Single-page Streamlit application
├── generate_demo_model.py      # Creates a demo insurance_model.joblib
├── requirements.txt
├── README.md
├── model/
│   └── insurance_model.joblib  # Place your trained model here
└── modules/
    ├── __init__.py
    ├── predictor.py            # Model loading & inference
    ├── ocr.py                  # EasyOCR text extraction
    ├── validator.py            # Document verification scoring
    └── fraud.py                # Rule-based fraud risk engine
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


### 2. Add your model 

```

The model must accept a `pandas.DataFrame` with these columns (in order):

| Column | Type | Encoding |
|---|---|---|
| age | float | raw |
| sex | int | 0=male, 1=female |
| weight | float | raw (kg) |
| bmi | float | raw |
| no_of_dependents | int | raw |
| smoker | int | 0=no, 1=yes |
| bloodpressure | int | 0=low, 1=normal, 2=high |
| diabetes | int | 0=no, 1=yes |
| regular_ex | int | 0=no, 1=yes |
| job_title | int | label-encoded 0-14 |
| city | int | label-encoded 0-10 |
| hereditary_diseases | int | label-encoded 0-8 |



### 3. Run the application

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Module Details

### `modules/predictor.py`
- Loads `model/insurance_model.joblib` on startup (cached with `@st.cache_resource`)
- Encodes raw form values to numeric features matching training-time schema
- Falls back to a rule-based estimator if the model file is absent

### `modules/ocr.py`
- Wraps EasyOCR with graceful degradation when unavailable
- Returns `OCRResult` dataclass: `text`, `lines`, `confidence`, `success`, `error`
- Lazy-loads the OCR engine once per session

### `modules/validator.py`
- Scores each uploaded document (0-100) on three dimensions:
  1. **Upload presence** (20 pts)
  2. **OCR confidence** (0-30 pts)
  3. **Keyword hit-rate** (0-50 pts)
- Weighted average gives `overall_score`; status: `Valid` ≥80, `Needs Review` ≥50, `Insufficient` <50

### `modules/fraud.py`
- 11 weighted rules evaluated against form data + predicted amount
- Score normalised to 0-100; labels: `Low Risk` <35, `Medium Risk` <65, `High Risk` ≥65
- Rule weights are tunable in the `RULES` list

### Approval Engine (in `app.py`)
| Condition | Decision |
|---|---|
| Verification ≥ 80 **AND** Fraud < 40 | ✅ APPROVED |
| Fraud > 70 **OR** Verification < 50 | ❌ REJECTED |
| Otherwise | ⚠️ UNDER REVIEW |

---

## Customisation

- **Swap model:** Replace `model/insurance_model.joblib`; update `FEATURE_COLUMNS` and `ENCODINGS` in `predictor.py` if your schema differs.
- **Add fraud rules:** Extend the `RULES` list in `fraud.py`.
- **Adjust keyword lists:** Edit `REQUIRED_KEYWORDS` in `validator.py`.
- **Tune approval thresholds:** Edit the if/elif block in `app.py` under `# ── 5. Approval engine`.

---

## Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
plotly>=5.18.0
Pillow>=10.0.0
easyocr
```

---

*For internal use only. All AI predictions are probabilistic and require human adjuster review.*
