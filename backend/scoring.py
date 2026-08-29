"""Combine a caption with reverse-image page context using the TF-IDF ensemble."""

from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity

from backend.nlp import clean_text, label_from_probability, load_bundle, score_text

MATCH_THRESHOLD = 0.22


def text_similarity(a: str, b: str) -> float:
    bundle = load_bundle()
    xa = bundle["vectorizer"].transform([clean_text(a)])
    xb = bundle["vectorizer"].transform([clean_text(b)])
    return float(cosine_similarity(xa, xb)[0, 0])


def score_claim_with_image_context(claim: str, context: str, first_appearance: dict | None) -> dict:
    """
    Match uploaded text with the first-appearance page context, then apply
    the same ensemble P(fake) rules as pure text.

    - If no claim: classify the scraped context alone.
    - If claim matches context: average P(fake) of claim and context
      (same formula, two observations of the story).
    - If claim and context diverge: treat as Misleading (image reused with
      unrelated wording — classic misinformation pattern).
    """
    claim = (claim or "").strip()
    context = (context or "").strip()

    if not claim and not context:
        return {
            "label": "Misleading",
            "p_fake": 0.5,
            "likelihood_true": 0.5,
            "likelihood_true_pct": 50.0,
            "match_score": 0.0,
            "matched": False,
            "mode": "insufficient_text",
            "claim_score": None,
            "context_score": None,
            "combined_text": "",
            "note": "No claim and no readable context from the first appearance page.",
        }

    if not claim:
        result = score_text(context)
        result.update(
            {
                "match_score": None,
                "matched": None,
                "mode": "image_context_only",
                "claim_score": None,
                "context_score": result,
                "combined_text": context[:500],
                "note": "No caption was provided, so the ensemble ran on the first-appearance page text.",
            }
        )
        return result

    claim_score = score_text(claim)
    if not context:
        claim_score.update(
            {
                "match_score": 0.0,
                "matched": False,
                "mode": "claim_only_no_context",
                "claim_score": claim_score,
                "context_score": None,
                "combined_text": claim[:500],
                "note": "Reverse image search did not yield page text; scored the uploaded caption only.",
            }
        )
        return claim_score

    sim = text_similarity(claim, context)
    context_score = score_text(context)

    if sim >= MATCH_THRESHOLD:
        p_fake = (claim_score["p_fake"] + context_score["p_fake"]) / 2.0
        combined = f"{claim}\n\n{context}"
        combined_score = score_text(combined)
        label = label_from_probability(p_fake)
        return {
            "label": label,
            "p_fake": round(p_fake, 4),
            "likelihood_true": round(1.0 - p_fake, 4),
            "likelihood_true_pct": round((1.0 - p_fake) * 100, 1),
            "match_score": round(sim, 4),
            "matched": True,
            "mode": "matched_claim_and_context",
            "claim_score": claim_score,
            "context_score": context_score,
            "combined_score": combined_score,
            "combined_text": combined[:500],
            "formula": claim_score["formula"],
            "note": (
                "Caption matched the first-appearance context. "
                "P(fake) is the mean of the two ensemble scores."
            ),
            "first_appearance": first_appearance,
        }

    # Low lexical overlap: image is likely being reused with a new story.
    return {
        "label": "Misleading",
        "p_fake": 0.5,
        "likelihood_true": 0.5,
        "likelihood_true_pct": 50.0,
        "match_score": round(sim, 4),
        "matched": False,
        "mode": "unmatched_reuse",
        "claim_score": claim_score,
        "context_score": context_score,
        "combined_text": claim[:500],
        "formula": claim_score["formula"],
        "note": (
            "The uploaded text does not match the first known use of this image, "
            "so the result is Misleading."
        ),
        "first_appearance": first_appearance,
    }
