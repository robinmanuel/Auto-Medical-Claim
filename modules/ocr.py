"""
ocr.py — Document text extraction using EasyOCR.
Replaces PaddleOCR for better Windows / Python 3.12 compatibility.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str = ""
    lines: list[str] = field(default_factory=list)
    confidence: float = 0.0
    success: bool = False
    error: Optional[str] = None
    engine_available: bool = True

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def confidence_pct(self) -> float:
        return round(self.confidence * 100, 1)


# ── lazy-load EasyOCR reader ──────────────────────────────────────────────────
_reader = None
_available: Optional[bool] = None


def _get_engine():
    global _reader, _available

    if _available is True:
        return _reader
    if _available is False:
        return None

    try:
        import easyocr  # type: ignore
        # gpu=False ensures it works on any machine; downloads model on first run
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _available = True
        logger.info("EasyOCR loaded successfully.")
        return _reader
    except Exception as exc:
        _available = False
        logger.warning("EasyOCR unavailable: %s", exc)
        return None


# ── public API ────────────────────────────────────────────────────────────────
def extract_text(uploaded_file) -> OCRResult:
    if uploaded_file is None:
        return OCRResult(error="No file provided.")

    try:
        img_bytes = uploaded_file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(pil_img)
    except Exception as exc:
        return OCRResult(error=f"Failed to read image: {exc}")

    engine = _get_engine()
    if engine is None:
        return OCRResult(
            engine_available=False,
            error="EasyOCR is not installed. Run: pip install easyocr",
            success=False,
        )

    try:
        # returns list of (bbox, text, confidence)
        raw = engine.readtext(img_array)
    except Exception as exc:
        return OCRResult(error=f"OCR inference failed: {exc}", success=False)

    if not raw:
        return OCRResult(text="", success=True, confidence=0.0,
                         error="No text detected in the image.")

    lines       = [item[1] for item in raw]
    confidences = [float(item[2]) for item in raw]
    full_text   = "\n".join(lines)
    mean_conf   = float(np.mean(confidences)) if confidences else 0.0

    return OCRResult(
        text=full_text,
        lines=lines,
        confidence=mean_conf,
        success=True,
    )


def summarise_results(results: dict) -> dict:
    all_texts, confidences = [], []
    success_count = 0
    for res in results.values():
        if res.success and res.text:
            all_texts.append(res.text)
            confidences.append(res.confidence)
            success_count += 1
    return {
        "total_words":     sum(len(t.split()) for t in all_texts),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "all_text":        "\n\n---\n\n".join(all_texts),
        "success_count":   success_count,
    }
