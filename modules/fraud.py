"""
fraud.py — Rule-based fraud risk scoring engine.

Returns a 0-100 fraud risk score plus a risk label and breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FraudResult:
    score: float = 0.0                   # 0-100
    label: str = "Low Risk"
    color: str = "#22c55e"
    triggered_rules: list[dict] = field(default_factory=list)   # {rule, weight, triggered}
    recommendation: str = ""


# ── rule definitions ──────────────────────────────────────────────────────────
# Each rule: (id, description, weight, evaluator_fn)
# evaluator_fn receives the raw form_data dict + predicted_amount float → bool

def _rule_high_claim(data: dict, amount: float) -> bool:
    return amount > 50_000

def _rule_very_high_claim(data: dict, amount: float) -> bool:
    return amount > 100_000

def _rule_smoker(data: dict, amount: float) -> bool:
    return str(data.get("smoker", "no")).lower() == "yes"

def _rule_high_bp(data: dict, amount: float) -> bool:
    return str(data.get("bloodpressure", "normal")).lower() == "high"

def _rule_diabetes(data: dict, amount: float) -> bool:
    return str(data.get("diabetes", "no")).lower() == "yes"

def _rule_multiple_hereditary(data: dict, amount: float) -> bool:
    hd = str(data.get("hereditary_diseases", "none")).lower()
    # Flag if multiple conditions mentioned (comma separated or known combined strings)
    return hd != "none" and ("," in hd or "and" in hd)

def _rule_no_exercise_smoker(data: dict, amount: float) -> bool:
    return (
        str(data.get("smoker", "no")).lower() == "yes"
        and str(data.get("regular_ex", "no")).lower() == "no"
    )

def _rule_young_high_claim(data: dict, amount: float) -> bool:
    return float(data.get("age", 30)) < 30 and amount > 40_000

def _rule_high_dependents(data: dict, amount: float) -> bool:
    return int(data.get("no_of_dependents", 0)) >= 5

def _rule_extreme_bmi(data: dict, amount: float) -> bool:
    bmi = float(data.get("bmi", 22))
    return bmi > 40 or bmi < 16

def _rule_combo_risk(data: dict, amount: float) -> bool:
    """Smoker + diabetes + high BP simultaneously."""
    return (
        str(data.get("smoker", "no")).lower() == "yes"
        and str(data.get("diabetes", "no")).lower() == "yes"
        and str(data.get("bloodpressure", "normal")).lower() == "high"
    )


RULES = [
    ("high_claim",           "Claim amount unusually high (>₹50k)",         12, _rule_high_claim),
    ("very_high_claim",      "Claim amount extremely high (>₹1L)",           20, _rule_very_high_claim),
    ("smoker",               "Patient is a smoker",                           8, _rule_smoker),
    ("high_bp",              "High blood pressure reported",                  8, _rule_high_bp),
    ("diabetes",             "Diabetes diagnosis present",                    8, _rule_diabetes),
    ("multiple_hereditary",  "Multiple hereditary diseases listed",           10, _rule_multiple_hereditary),
    ("no_exercise_smoker",   "Non-exercising smoker",                         7, _rule_no_exercise_smoker),
    ("young_high_claim",     "Young patient with high claim",                 10, _rule_young_high_claim),
    ("high_dependents",      "Unusually high number of dependents (≥5)",      5, _rule_high_dependents),
    ("extreme_bmi",          "Extreme BMI value (>40 or <16)",                7, _rule_extreme_bmi),
    ("combo_risk",           "Combined smoker + diabetes + high BP",          15, _rule_combo_risk),
]

_MAX_POSSIBLE = sum(w for _, _, w, _ in RULES)


def calculate_fraud_risk(form_data: dict, predicted_amount: float) -> FraudResult:
    """
    Parameters
    ----------
    form_data        : dict  Raw patient form values.
    predicted_amount : float Predicted claim amount (₹).

    Returns
    -------
    FraudResult
    """
    total_weight = 0.0
    triggered: list[dict] = []
    all_rules: list[dict] = []

    for rule_id, description, weight, fn in RULES:
        fired = fn(form_data, predicted_amount)
        entry = {
            "rule": description,
            "weight": weight,
            "triggered": fired,
        }
        all_rules.append(entry)
        if fired:
            total_weight += weight
            triggered.append(entry)

    # Normalise to 0-100
    raw_score = (total_weight / _MAX_POSSIBLE) * 100
    score = min(100.0, round(raw_score, 1))

    label, color = _risk_label(score)
    recommendation = _recommendation(score, triggered)

    return FraudResult(
        score=score,
        label=label,
        color=color,
        triggered_rules=all_rules,
        recommendation=recommendation,
    )


def _risk_label(score: float) -> tuple[str, str]:
    if score < 35:
        return "Low Risk",    "#22c55e"
    elif score < 65:
        return "Medium Risk", "#f59e0b"
    else:
        return "High Risk",   "#ef4444"


def _recommendation(score: float, triggered: list[dict]) -> str:
    if score < 35:
        return "Claim profile appears standard. Proceed with normal processing."
    elif score < 65:
        top = triggered[:2]
        reasons = "; ".join(r["rule"] for r in top)
        return f"Manual review recommended. Key concerns: {reasons}."
    else:
        return (
            "Escalate to senior adjuster for thorough investigation before approval. "
            "Multiple high-risk indicators detected."
        )
