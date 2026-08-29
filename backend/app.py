"""Fake news detection API: college NLP stack + reverse image search."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from backend.nlp import score_text  # noqa: E402
from backend.reverse_search import fetch_page_context, reverse_image_search  # noqa: E402
from backend.scoring import score_claim_with_image_context  # noqa: E402

FRONTEND = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/api/health")
def health():
    from backend.nlp import MODELS_DIR

    return jsonify(
        {
            "ok": True,
            "models": (MODELS_DIR / "ensemble.joblib").exists(),
            "reverse_image": bool(
                os.getenv("SERPAPI_API_KEY", "").strip()
                or os.getenv("BING_VISUAL_SEARCH_KEY", "").strip()
            ),
        }
    )


@app.post("/api/check-text")
def check_text():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or request.form.get("text") or "").strip()
    if len(text) < 8:
        return jsonify({"error": "Please enter a longer news snippet (at least a short sentence)."}), 400
    try:
        result = score_text(text)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    result["input"] = text[:400]
    return jsonify(result)


@app.post("/api/check-image")
def check_image():
    claim = (request.form.get("text") or "").strip()
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return jsonify({"error": "Please upload an image."}), 400

    raw = upload.read()
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:  # noqa: BLE001
        return jsonify({"error": "That file is not a readable image."}), 400

    search = reverse_image_search(raw, upload.filename)
    first = search.get("first_appearance")
    context_parts = []
    if first:
        context_parts.extend(
            [
                first.get("title") or "",
                first.get("snippet") or "",
                fetch_page_context(first["url"]) if first.get("url") else "",
            ]
        )
    context = " ".join(p for p in context_parts if p)

    try:
        scored = score_claim_with_image_context(claim, context, first)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(
        {
            **scored,
            "reverse_search": {
                "ok": search["ok"],
                "message": search.get("message"),
                "errors": search.get("errors") or [],
                "first_appearance": first,
                "other_matches": search.get("hits") or [],
            },
            "input": claim[:400],
        }
    )


def main():
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
