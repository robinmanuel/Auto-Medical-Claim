"""
validator.py — Document verification and scoring module.

Generates a 0-100 verification score based on:
  1. Whether each document was uploaded
  2. Whether OCR succeeded
  3. OCR extraction confidence
  4. Presence of required keywords in extracted text
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Required keywords per document type
REQUIRED_KEYWORDS: dict[str, list[str]] = {
    "medical_bill": [
        "hospital", "patient", "amount", "total", "bill", "date",
        "invoice", "charge", "payment", "diagnosis",
    ],
    "prescription": [
        "doctor", "patient", "medicine", "tablet", "dose", "mg",
        "prescribed", "signature", "date", "pharmacy",
    ],
    "claim_form": [
        "claim", "policy", "insured", "signature", "date",
        "amount", "name", "address", "illness", "treatment",
    ],
}

# Weight given to each document in the final score
DOC_WEIGHTS = {
    "medical_bill": 0.40,
    "prescription": 0.30,
    "claim_form":   0.30,
}


@dataclass
class DocVerification:
    doc_type: str
    uploaded: bool = False
    ocr_success: bool = False
    ocr_confidence: float = 0.0          # 0-1
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    doc_score: float = 0.0               # 0-100 for this single document

    @property
    def keyword_hit_rate(self) -> float:
        total = len(REQUIRED_KEYWORDS.get(self.doc_type, []))
        return len(self.keywords_found) / total if total else 0.0


@dataclass
class VerificationReport:
    documents: list[DocVerification] = field(default_factory=list)
    overall_score: float = 0.0           # 0-100
    status: str = "Pending"             # Valid | Needs Review | Insufficient
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ── scoring helpers ───────────────────────────────────────────────────────────
def _score_single_doc(
    doc_type: str,
    uploaded: bool,
    ocr_text: str,
    ocr_confidence: float,
    ocr_success: bool,
) -> DocVerification:
    dv = DocVerification(doc_type=doc_type)

    if not uploaded:
        dv.doc_score = 0.0
        return dv

    dv.uploaded = True

    if not ocr_success:
        dv.doc_score = 15.0          # partial credit — at least it was uploaded
        return dv

    dv.ocr_success = True
    dv.ocr_confidence = ocr_confidence

    # Keyword check
    text_lower = ocr_text.lower()
    required = REQUIRED_KEYWORDS.get(doc_type, [])
    dv.keywords_found   = [kw for kw in required if kw in text_lower]
    dv.keywords_missing = [kw for kw in required if kw not in text_lower]

    # ── weighted sub-scores ──────────────────────────────────────────────────
    upload_score     = 20.0
    ocr_score        = ocr_confidence * 30          # 0-30
    keyword_score    = dv.keyword_hit_rate * 50     # 0-50

    dv.doc_score = min(100.0, upload_score + ocr_score + keyword_score)
    return dv


def generate_verification_report(
    doc_results: dict,   # {doc_type: {"uploaded": bool, "ocr": OCRResult | None}}
) -> VerificationReport:
    """
    Parameters
    ----------
    doc_results : dict
        Keys: "medical_bill", "prescription", "claim_form"
        Values: dict with keys "uploaded" (bool) and "ocr" (OCRResult or None)

    Returns
    -------
    VerificationReport
    """
    report = VerificationReport()
    weighted_total = 0.0

    for doc_type, weight in DOC_WEIGHTS.items():
        info = doc_results.get(doc_type, {})
        uploaded = info.get("uploaded", False)
        ocr_res  = info.get("ocr", None)

        if ocr_res is not None:
            ocr_text       = ocr_res.text
            ocr_confidence = ocr_res.confidence
            ocr_success    = ocr_res.success
        else:
            ocr_text       = ""
            ocr_confidence = 0.0
            ocr_success    = False

        dv = _score_single_doc(doc_type, uploaded, ocr_text, ocr_confidence, ocr_success)
        report.documents.append(dv)
        weighted_total += dv.doc_score * weight

    report.overall_score = round(weighted_total, 1)

    # ── issues & recommendations ─────────────────────────────────────────────
    for dv in report.documents:
        label = dv.doc_type.replace("_", " ").title()
        if not dv.uploaded:
            report.issues.append(f"{label} not uploaded.")
            report.recommendations.append(f"Upload {label} to improve score.")
        elif not dv.ocr_success:
            report.issues.append(f"{label}: OCR failed or no text detected.")
            report.recommendations.append(f"Re-upload a clearer image of {label}.")
        elif dv.keywords_missing:
            report.issues.append(
                f"{label}: Missing keywords — {', '.join(dv.keywords_missing[:3])}."
            )

    # ── status ────────────────────────────────────────────────────────────────
    s = report.overall_score
    if s >= 80:
        report.status = "Valid"
    elif s >= 50:
        report.status = "Needs Review"
    else:
        report.status = "Insufficient"

    return report
