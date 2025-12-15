# src/sentiment_chain.py
"""
LLM-based sentiment classifier for news items.

Behavior:
- Configurable provider via .env:
  - SENTIMENT_LLM_PROVIDER=openai (requires OPENAI_API_KEY)
  - otherwise uses a local HF model (HF_SENTIMENT_MODEL)
- Uses a clear instruction prompt that returns JSON with:
  - label: "positive" | "neutral" | "negative"
  - confidence: float in [0..1] (LLM-produced heuristic)
  - rationale: short explanation (<= 50 words)

Functions:
- classify_text(text) -> dict
- batch_classify(rows) -> yields dicts
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List
from src.observability.langfuse_client import get_langfuse
import time


from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")

SENT_LLM_PROVIDER = os.getenv("SENTIMENT_LLM_PROVIDER", os.getenv("SENTIMENT_LLM_PROVIDER", "openai"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
HF_SENTIMENT_MODEL = os.getenv("HF_SENTIMENT_MODEL", "tiiuae/falcon-7b-instruct")  # example; change if too heavy
HF_BATCH_SIZE = int(os.getenv("HF_BATCH_SIZE", "8"))

# Prompt template — keep it deterministic and ask for JSON
_PROMPT_TEMPLATE = """You are a concise financial news sentiment classifier.
Given the article title and body, return a JSON object with three fields:
  - "label": one of "positive", "neutral", or "negative" (pick the single most appropriate label).
  - "confidence": a number between 0.0 and 1.0 representing how confident you are (0 = not confident, 1 = fully confident).
  - "rationale": a short (max ~40 words) explanation for the label.

Important: Output must be valid JSON and nothing else.

Article title:
\"\"\"{title}\"\"\"

Article body:
\"\"\"{body}\"\"\"
"""

# --------------------
# OpenAI provider impl
# --------------------
def _classify_openai(title: str, body: str) -> Dict[str, Any]:
    import openai
    openai.api_key = OPENAI_API_KEY
    prompt = _PROMPT_TEMPLATE.format(title=title or "", body=body or "")
    # Use chat completion (gpt-3.5-turbo or gpt-4 if you have access)
    model = os.getenv("OPENAI_SENTiment_MODEL", "gpt-3.5-turbo-16k")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # Try to parse JSON; handle common issues
    try:
        parsed = json.loads(text)
    except Exception:
        # Attempt to extract JSON substring
        import re
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = {"label": "neutral", "confidence": 0.0, "rationale": "could not parse model output"}
        else:
            parsed = {"label": "neutral", "confidence": 0.0, "rationale": "no JSON in model output"}
    # sanitize
    parsed = _sanitize_parsed(parsed)
    return parsed

# --------------------
# HF local provider impl (transformers)
# --------------------
def _classify_hf(title: str, body: str) -> Dict[str, Any]:
    """
    Use a local transformer text-generation model to produce the same JSON.
    This requires a model capable of instruction following (e.g., flan-* or instruct models).
    """
    from transformers import pipeline, set_seed

    model_name = os.getenv("HF_SENTIMENT_MODEL", HF_SENTIMENT_MODEL)
    # use text-generation pipeline
    pipe = pipeline("text-generation", model=model_name, truncation=True, device_map="auto" if os.getenv("HF_USE_GPU") else None)
    prompt = _PROMPT_TEMPLATE.format(title=title or "", body=body or "")
    # generate
    gen = pipe(prompt, max_new_tokens=200, do_sample=False, return_full_text=False)
    text = gen[0]["generated_text"].strip()
    # try parse JSON
    try:
        parsed = json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = {"label": "neutral", "confidence": 0.0, "rationale": "could not parse HF output"}
        else:
            parsed = {"label": "neutral", "confidence": 0.0, "rationale": "no JSON in HF output"}
    parsed = _sanitize_parsed(parsed)
    return parsed

# --------------------
# helpers
# --------------------
def _sanitize_parsed(d: Dict[str, Any]) -> Dict[str, Any]:
    label = d.get("label", "neutral")
    if label not in ("positive", "neutral", "negative"):
        # try to normalise from synonyms
        lab_low = str(label).lower()
        if "pos" in lab_low or "good" in lab_low or "+" in lab_low:
            label = "positive"
        elif "neg" in lab_low or "bad" in lab_low or "-" in lab_low:
            label = "negative"
        else:
            label = "neutral"
    conf = d.get("confidence", None)
    try:
        conf = float(conf)
        if conf < 0:
            conf = 0.0
        if conf > 1:
            conf = 1.0
    except Exception:
        # Fallback: assign heuristic confidences based on label if missing
        conf = {"positive": 0.75, "neutral": 0.5, "negative": 0.75}.get(label, 0.5)
    rationale = (d.get("rationale") or "")[:400]
    return {"label": label, "confidence": float(conf), "rationale": rationale}

def classify_text(title: str, body: str) -> Dict[str, Any]:
    lf = get_langfuse()
    start = time.time()

    with lf.trace(
        name="llm_sentiment_classification",
        metadata={
            "provider": os.getenv("SENTIMENT_LLM_PROVIDER"),
            "model_version": "llm-v1",
        },
        input={
            "title": title[:300],
            "body": body[:1000],
        }
    ) as trace:

        try:
            provider = os.getenv("SENTIMENT_LLM_PROVIDER", SENT_LLM_PROVIDER).lower()
            if provider == "openai":
                result = _classify_openai(title, body)
            else:
                result = _classify_hf(title, body)

            latency = time.time() - start

            trace.output = result
            trace.score(
                name="confidence",
                value=result.get("confidence", 0.0)
            )
            trace.metadata["latency_sec"] = latency

            return result

        except Exception as e:
            trace.error(str(e))
            raise


def batch_classify(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    items: iterable of dicts with keys 'title', 'body'
    returns: list of parsed classification dicts (aligned to input order)
    Note: for HF you may want to implement batching — left simple for clarity.
    """
    out = []
    for it in items:
        title = it.get("title") or ""
        body = it.get("body") or ""
        res = classify_text(title, body)
        out.append(res)
    return out
