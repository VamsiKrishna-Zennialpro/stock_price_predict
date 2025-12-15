# ui/app.py
import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY", "super-secret-prod-key")

HEADERS = {"X-API-KEY": API_KEY}

st.set_page_config(page_title="AI Stock Sentiment Dashboard", layout="wide")

st.title("📈 AI Stock Sentiment & Price Predictor")

ticker = st.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

st.divider()

# ---------------------
# Price Prediction
# ---------------------
st.subheader("🔮 Next-Day Price Prediction")

if st.button("Predict Next Day"):
    resp = requests.post(
        f"{API_BASE}/predict",
        json={"ticker": ticker},
        headers=HEADERS,
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        st.metric(
            label="Prediction",
            value=data["prediction"],
            delta=f"Confidence {data['confidence']:.2f}"
        )
    else:
        st.error("Prediction failed")

# ---------------------
# Sentiment Analysis
# ---------------------
st.subheader("📰 News Sentiment Analyzer")

title = st.text_input("News Title")
body = st.text_area("News Body", height=120)

if st.button("Analyze Sentiment"):
    resp = requests.post(
        f"{API_BASE}/sentiment",
        json={"title": title, "body": body},
        headers=HEADERS,
        timeout=30,
    )
    if resp.status_code == 200:
        s = resp.json()
        st.success(f"Sentiment: {s['label']} (confidence {s['confidence']:.2f})")
        st.caption(s["rationale"])
    else:
        st.error("Sentiment analysis failed")

# ---------------------
# Semantic Retrieval
# ---------------------
st.subheader("🔍 Semantic News Search")

query = st.text_input("Search news (semantic)")
if st.button("Search"):
    resp = requests.get(
        f"{API_BASE}/retrieve",
        params={"query": query, "ticker": ticker},
        headers=HEADERS,
        timeout=30,
    )
    if resp.status_code == 200:
        results = resp.json()["results"]
        for r in results:
            st.markdown(f"**{r['metadata'].get('title','')}**")
            st.caption(r["text"][:400])
            st.divider()
    else:
        st.error("Search failed")
