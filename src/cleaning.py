# src/cleaning.py
import re
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TICKER_FILE = Path(__file__).parents[1] / "data" / "ticker_aliases.json"

_analyzer = None
_ticker_map = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer

def load_ticker_map():
    global _ticker_map
    if _ticker_map is None:
        if TICKER_FILE.exists():
            with open(TICKER_FILE, "r", encoding="utf-8") as f:
                _ticker_map = json.load(f)
        else:
            _ticker_map = {}
    return _ticker_map

def strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    # Remove scripts/styles
    for s in soup(["script", "style"]):
        s.decompose()
    text = soup.get_text(separator=" ")
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_timestamp(ts: Optional[str]):
    """
    Accepts ISO strings or other date strings; returns timezone-naive Python datetime
    (we store as timezone-aware in DB using SQLAlchemy server defaults or alignment docs).
    """
    if ts is None:
        return None
    try:
        dt = date_parser.parse(ts)
        return dt
    except Exception:
        return None

def map_tickers(text: str) -> List[str]:
    """
    Very simple keyword mapping. Returns list of matched tickers (no duplicates).
    """
    ticker_map = load_ticker_map()
    text_low = (text or "").lower()
    matched = []
    for ticker, keywords in ticker_map.items():
        for kw in keywords:
            if kw.lower() in text_low:
                matched.append(ticker)
                break
    return list(dict.fromkeys(matched))  # preserve order and dedupe

def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple dedupe on (url) then title+body fingerprint.
    records: list of dicts with keys 'url','title','body','published_at','source'
    """
    seen_urls = set()
    seen_fingerprints = set()
    out = []
    for r in records:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        if url and url in seen_urls:
            continue
        # fingerprint
        fp = (title + "|" + (body[:300] if body else "")).lower()
        if fp in seen_fingerprints:
            continue
        if url:
            seen_urls.add(url)
        seen_fingerprints.add(fp)
        out.append(r)
    return out

def label_sentiment(text: str) -> Dict[str, Any]:
    analyzer = get_analyzer()
    s = analyzer.polarity_scores(text or "")
    # assign label thresholds (VADER compound)
    comp = s.get("compound", 0.0)
    if comp >= 0.05:
        label = "positive"
    elif comp <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {
        "neg": s.get("neg"),
        "neu": s.get("neu"),
        "pos": s.get("pos"),
        "compound": comp,
        "label": label,
    }
