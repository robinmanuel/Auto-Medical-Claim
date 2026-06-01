"""
predictor.py — Claim amount prediction using the pre-trained joblib model.
"""

import os
import joblib
import numpy as np
import pandas as pd


# ── paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "claim_model.joblib")

# Feature order expected by the model
FEATURE_COLUMNS = [
    "age", "sex", "weight", "bmi", "hereditary_diseases", "no_of_dependents",
    "smoker", "city", "bloodpressure", "diabetes", "regular_ex", "job_title",
]

# Categorical → numeric maps (must match training-time encoding)
ENCODINGS = {
    "sex":      {"male": 0, "female": 1},
    "smoker":   {"no": 0, "yes": 1},
    "diabetes": {"no": 0, "yes": 1},
    "regular_ex": {"no": 0, "yes": 1},
}

# Ordinal blood-pressure map
BP_MAP = {"low": 0, "normal": 1, "high": 2}

# Simple label-encoding for high-cardinality categoricals
JOB_TITLES = [
    "engineer", "doctor", "teacher", "lawyer", "accountant",
    "manager", "nurse", "driver", "businessman", "artist",
    "scientist", "technician", "architect", "police", "other",
]

CITIES = [
    "mumbai", "delhi", "bangalore", "hyderabad", "chennai",
    "kolkata", "pune", "ahmedabad", "jaipur", "lucknow", "other",
]

HEREDITARY = [
    "none", "diabetes", "heart disease", "cancer", "hypertension",
    "epilepsy", "alzheimer's", "arthritis", "other",
]


def _label_encode(value: str, vocab: list) -> int:
    """Return index of value in vocab, defaulting to last bucket."""
    v = value.lower().strip()
    return vocab.index(v) if v in vocab else len(vocab) - 1


def _encode_input(data: dict) -> pd.DataFrame:
    """Convert raw form dict → numeric DataFrame row."""
    row = {}
    row["age"]             = float(data["age"])
    row["sex"]             = ENCODINGS["sex"].get(data["sex"].lower(), 0)
    row["weight"]          = float(data["weight"])
    row["bmi"]             = float(data["bmi"])
    row["no_of_dependents"]= int(data["no_of_dependents"])
    row["smoker"]          = ENCODINGS["smoker"].get(data["smoker"].lower(), 0)
    row["bloodpressure"]   = BP_MAP.get(data["bloodpressure"].lower(), 1)
    row["diabetes"]        = ENCODINGS["diabetes"].get(data["diabetes"].lower(), 0)
    row["regular_ex"]      = ENCODINGS["regular_ex"].get(data["regular_ex"].lower(), 0)
    row["job_title"]       = _label_encode(data["job_title"], JOB_TITLES)
    row["city"]            = _label_encode(data["city"], CITIES)
    row["hereditary_diseases"] = _label_encode(data["hereditary_diseases"], HEREDITARY)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _risk_category(amount: float) -> tuple[str, str]:
    """Return (label, colour) based on claim amount thresholds (in INR)."""
    if amount < 50_000:
        return "Low Claim", "#22c55e"
    elif amount < 150_000:
        return "Medium Claim", "#f59e0b"
    else:
        return "High Claim", "#ef4444"


def convert_to_indian_context(usd_prediction: float, input_df: pd.DataFrame) -> float:
    """Convert USD prediction to Indian context with risk adjustments.
    
    Parameters
    ----------
    usd_prediction : float
        The USD amount predicted by the model.
    input_df : pd.DataFrame
        Encoded feature DataFrame with smoker and diabetes columns.
    
    Returns
    -------
    float
        Adjusted claim amount in INR.
    """
    factor = 0.06  # Tune between 0.05 to 0.08 based on local claim patterns
    risk_multiplier = 1.0
    
    # Adjust multiplier based on risk factors
    if input_df['smoker'].iloc[0] == 1:
        risk_multiplier += 0.2
    
    if input_df['diabetes'].iloc[0] == 1:
        risk_multiplier += 0.15
    
    return usd_prediction * factor * 83 * risk_multiplier


class ClaimPredictor:
    """Loads a joblib model and exposes a predict() method."""

    def __init__(self):
        self._model = None
        self._loaded = False
        self._error: str | None = None
        self._load_model()

    # ── internal ─────────────────────────────────────────────────────────────
    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self._model = joblib.load(MODEL_PATH)
                self._loaded = True
            except Exception as exc:
                self._error = str(exc)
        else:
            self._error = (
                f"Model file not found at {MODEL_PATH}. "
                "Place insurance_model.joblib inside the model/ directory."
            )

    # ── public ───────────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._error

    def predict(self, form_data: dict) -> dict:
        """
        Parameters
        ----------
        form_data : dict  Keys match FEATURE_COLUMNS (raw / string values).

        Returns
        -------
        dict with keys:
            predicted_amount  float
            risk_label        str
            risk_color        str
            features_df       pd.DataFrame  (for display)
            model_loaded      bool
        """
        if not self._loaded:
            # Graceful fallback — rule-based estimate so the UI still works
            amount = _fallback_estimate(form_data)
        else:
            X = _encode_input(form_data)
            amount = float(self._model.predict(X)[0])
            amount = max(0.0, amount)          # clamp negatives
        
        # Convert to Indian context with risk adjustments
        X = _encode_input(form_data)
        amount = convert_to_indian_context(amount, X)

        label, color = _risk_category(amount)
        return {
            "predicted_amount": round(amount, 2),
            "risk_label": label,
            "risk_color": color,
            "features_df": _encode_input(form_data),
            "model_loaded": self._loaded,
        }


# ── fallback (no model file) ──────────────────────────────────────────────────
def _fallback_estimate(data: dict) -> float:
    """
    Simple rule-based estimate used when no model is available.
    Not for production — purely a UI demonstration fallback.
    """
    base = 5_000.0
    base += float(data.get("age", 30)) * 80
    base += float(data.get("bmi", 22)) * 50
    base += int(data.get("no_of_dependents", 0)) * 1_500
    if str(data.get("smoker", "no")).lower() == "yes":
        base += 8_000
    if str(data.get("diabetes", "no")).lower() == "yes":
        base += 5_000
    if str(data.get("bloodpressure", "normal")).lower() == "high":
        base += 4_000
    if str(data.get("regular_ex", "no")).lower() == "no":
        base += 2_000
    return base
