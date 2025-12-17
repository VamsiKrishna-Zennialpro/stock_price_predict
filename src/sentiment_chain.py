# src/sentiment_chain.py
"""
LLM-based sentiment classifier using Google Gemini.

- Provider: Gemini (primary), HF (optional fallback)
- JSON-only output
- Langfuse observability
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai

from src.observability.langfuse_client import get_langfuse

# -------------------------------------------------
# Env & config
# -------------------------------------------------
load_dotenv(Path(__file__).parents[1] / ".env")

SENTIMENT_LLM_PROVIDER = os.getenv("SENTIMENT_LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
HF_SENTIMENT_MODEL = os.getenv("HF_SENTIMENT_MODEL")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------------------------------
# Prompt (JSON only)
# -------------------------------------------------
_PROMPT_TEMPLATE = """You are a financial news sentiment classifier.

Return ONLY valid JSON with:
- "label": "positive", "neutral", or "negative"
- "confidence": number between 0 and 1
- "rationale": max 40 words

Article title:
\"\"\"{title}\"\"\"

Article body:
\"\"\"{body}\"\"\"
"""

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _sanitize(parsed: Dict[str, Any]) -> Dict[str, Any]:
    label = str(parsed.get("label", "neutral")).lower()
    if label not in ("positive", "neutral", "negative"):
        label = "neutral"

    try:
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.5

    rationale = str(parsed.get("rationale", ""))[:300]

    return {
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
    }

def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        return {
            "label": "neutral",
            "confidence": 0.0,
            "rationale": "Failed to parse Gemini output",
        }

# -------------------------------------------------
# Gemini sentiment classifier
# -------------------------------------------------
def _classify_gemini(title: str, body: str) -> Dict[str, Any]:
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = _PROMPT_TEMPLATE.format(
        title=title or "",
        body=body or "",
    )

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 256,
        },
    )

    parsed = _parse_json(response.text.strip())
    return _sanitize(parsed)

# -------------------------------------------------
# Public API (with Langfuse tracing)
# -------------------------------------------------
def classify_text(title: str, body: str) -> Dict[str, Any]:
    lf = get_langfuse()
    start = time.time()

    with lf.trace(
        name="llm_sentiment_classification",
        input={
            "title": title[:300],
            "body": body[:1000],
        },
        metadata={
            "provider": "gemini",
            "model": GEMINI_MODEL,
            "model_version": "llm-gemini-v1",
        },
    ) as trace:
        try:
            result = _classify_gemini(title, body)

            latency = time.time() - start
            trace.output = result
            trace.score(name="confidence", value=result["confidence"])
            trace.metadata["latency_sec"] = latency

            return result

        except Exception as e:
            trace.error(str(e))
            raise
