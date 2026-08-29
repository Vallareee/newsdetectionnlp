"""Reverse image search adapters (SerpAPI Google + Bing Visual Search)."""

from __future__ import annotations

import base64
import os
from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

USER_AGENT = (
    "Mozilla/5.0 (compatible; NewsCheck/1.0; +https://github.com/Vallareee/newsdetectionnlp)"
)


def _parse_date(value: str | None):
    if not value:
        return None
    try:
        return dateparser.parse(str(value), fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None


def _normalize_hits(raw: list[dict]) -> list[dict]:
    hits = []
    for item in raw:
        url = item.get("link") or item.get("url") or item.get("hostPageUrl")
        if not url:
            continue
        title = item.get("title") or item.get("name") or item.get("source") or ""
        snippet = (
            item.get("snippet")
            or item.get("text")
            or item.get("description")
            or ""
        )
        date_val = (
            item.get("date")
            or item.get("datePublished")
            or item.get("crawl_date")
            or item.get("posted")
        )
        parsed = _parse_date(date_val) if not isinstance(date_val, datetime) else date_val
        hits.append(
            {
                "url": url,
                "title": title,
                "snippet": snippet,
                "date": parsed.isoformat() if parsed else None,
                "_sort": parsed or datetime.max,
            }
        )
    hits.sort(key=lambda h: h["_sort"])
    for h in hits:
        h.pop("_sort", None)
    return hits


def search_serpapi(image_bytes: bytes) -> list[dict]:
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return []
    encoded = base64.b64encode(image_bytes).decode("ascii")
    resp = requests.post(
        "https://serpapi.com/search.json",
        data={
            "engine": "google_reverse_image",
            "image_content": encoded,
            "api_key": api_key,
        },
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    rows: list[dict] = []
    for key in ("image_results", "inline_images", "organic_results"):
        rows.extend(data.get(key) or [])
    kg = data.get("knowledge_graph") or {}
    if kg.get("source") or kg.get("link"):
        rows.insert(
            0,
            {
                "title": kg.get("title") or kg.get("header"),
                "link": kg.get("source", {}).get("link") if isinstance(kg.get("source"), dict) else kg.get("link"),
                "snippet": kg.get("description") or "",
                "date": kg.get("date"),
            },
        )
    return _normalize_hits(rows)


def search_bing(image_bytes: bytes, filename: str = "upload.jpg") -> list[dict]:
    key = os.getenv("BING_VISUAL_SEARCH_KEY", "").strip()
    if not key:
        return []
    resp = requests.post(
        "https://api.bing.microsoft.com/v7.0/images/visualsearch",
        headers={"Ocp-Apim-Subscription-Key": key},
        files={"image": (filename, image_bytes, "application/octet-stream")},
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    rows: list[dict] = []

    def walk(node: Any):
        if isinstance(node, dict):
            if node.get("hostPageUrl") or node.get("url"):
                rows.append(
                    {
                        "title": node.get("name") or node.get("hostPageDisplayUrl") or "",
                        "url": node.get("hostPageUrl") or node.get("url"),
                        "snippet": node.get("text") or node.get("description") or "",
                        "date": node.get("datePublished"),
                    }
                )
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return _normalize_hits(rows)


def reverse_image_search(image_bytes: bytes, filename: str = "upload.jpg") -> dict:
    """
    Query configured APIs and return the earliest dated match as first appearance.
    Requires SERPAPI_API_KEY and/or BING_VISUAL_SEARCH_KEY.
    """
    errors = []
    hits: list[dict] = []

    if os.getenv("SERPAPI_API_KEY", "").strip():
        try:
            hits.extend(search_serpapi(image_bytes))
        except requests.RequestException as exc:
            errors.append(f"SerpAPI: {exc}")

    if os.getenv("BING_VISUAL_SEARCH_KEY", "").strip():
        try:
            hits.extend(search_bing(image_bytes, filename))
        except requests.RequestException as exc:
            errors.append(f"Bing: {exc}")

    # de-duplicate by URL
    seen = set()
    unique = []
    for hit in hits:
        url = hit["url"].split("#")[0]
        if url in seen:
            continue
        seen.add(url)
        unique.append(hit)
    unique.sort(key=lambda h: _parse_date(h.get("date")) or datetime.max)

    if not unique:
        configured = bool(
            os.getenv("SERPAPI_API_KEY", "").strip()
            or os.getenv("BING_VISUAL_SEARCH_KEY", "").strip()
        )
        return {
            "ok": False,
            "hits": [],
            "first_appearance": None,
            "errors": errors,
            "message": (
                "Reverse image search returned no matches."
                if configured
                else "Set SERPAPI_API_KEY or BING_VISUAL_SEARCH_KEY in .env to enable reverse image search."
            ),
        }

    first = unique[0]
    return {
        "ok": True,
        "hits": unique[:8],
        "first_appearance": first,
        "errors": errors,
        "message": None,
    }


def fetch_page_context(url: str, limit_chars: int = 2500) -> str:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            timeout=10,
            allow_redirects=True,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        desc = ""
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if meta:
            desc = meta.get("content") or ""
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        body = " ".join(p for p in paras if p)
        combined = " ".join(part for part in (title, desc, body) if part)
        return combined[:limit_chars]
    except requests.RequestException:
        return ""
