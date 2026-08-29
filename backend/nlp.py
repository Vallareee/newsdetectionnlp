"""TF-IDF + LR / Naive Bayes / Random Forest ensemble for news text."""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

TRUE_THRESHOLD = 0.40
FALSE_THRESHOLD = 0.60

_bundle = None


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_bundle():
    global _bundle
    if _bundle is not None:
        return _bundle
    path = MODELS_DIR / "ensemble.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "Trained models not found. Run: python scripts/train.py"
        )
    _bundle = joblib.load(path)
    return _bundle


def fake_probability(text: str) -> float:
    """Return mean P(fake) from the three sklearn models."""
    bundle = load_bundle()
    cleaned = clean_text(text)
    if not cleaned:
        return 0.5
    x = bundle["vectorizer"].transform([cleaned])
    probs = []
    for name in ("logistic_regression", "naive_bayes", "random_forest"):
        model = bundle["models"][name]
        # class order stored at train time: [REAL, FAKE] => index 1 is FAKE
        proba = model.predict_proba(x)[0]
        fake_idx = bundle["fake_index"]
        probs.append(float(proba[fake_idx]))
    return float(np.mean(probs))


def label_from_probability(p_fake: float) -> str:
    """Map P(fake) to True, False, or Misleading."""
    if p_fake < TRUE_THRESHOLD:
        return "True"
    if p_fake > FALSE_THRESHOLD:
        return "False"
    return "Misleading"


def score_text(text: str) -> dict:
    p_fake = fake_probability(text)
    label = label_from_probability(p_fake)
    return {
        "label": label,
        "p_fake": round(p_fake, 4),
        "likelihood_true": round(1.0 - p_fake, 4),
        "likelihood_true_pct": round((1.0 - p_fake) * 100, 1),
        "formula": {
            "features": "TF-IDF (1-2 grams, max 8000 features)",
            "models": "Logistic Regression + Naive Bayes + Random Forest (mean P(fake))",
            "rules": (
                f"True if P(fake) < {TRUE_THRESHOLD:.2f}; "
                f"False if P(fake) > {FALSE_THRESHOLD:.2f}; "
                "otherwise Misleading"
            ),
        },
    }
