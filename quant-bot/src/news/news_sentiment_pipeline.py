#!/usr/bin/env python3
"""
News sentiment & event pipeline (starter)

Features:
 - Ingest news from NewsAPI / RSS feeds
 - Compute finance-aware sentiment with FinBERT
 - Simple event extraction (regex + keyword classifier)
 - Produce news_signal payload and POST to trading API (/api/signals/news)
 - Save raw and parsed articles to disk for replay/audit

Configure via environment variables or .env:
 - NEWSAPI_KEY
 - TRADING_SIGNALS_URL (e.g., http://localhost:8000/api/signals/news)
 - SOURCE_WHITELIST (comma separated)
"""
import os
import re
import json
import time
import math
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
TRADING_SIGNALS_URL = os.getenv("TRADING_SIGNALS_URL", "http://localhost:8000/api/signals/news")
ARCHIVE_DIR = Path(os.getenv("NEWS_ARCHIVE", "data/news_archive"))
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_WHITELIST = set(x.strip().lower() for x in os.getenv("SOURCE_WHITELIST", "reuters,coindesk,cointelegraph,bbc,cnn,forbes").split(","))

# Use FinBERT for finance sentiment
MODEL_NAME = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Warning: Could not load FinBERT model {MODEL_NAME}: {e}")
    sentiment_pipe = None

# Simple event keywords -> event_type mapping
EVENT_KEYWORDS = {
    "hack": "exchange_hack",
    "security breach": "exchange_hack",
    "investigation": "regulatory_investigation",
    "banned": "regulation",
    "ban": "regulation",
    "ban on": "regulation",
    "legal": "regulation",
    "sued": "legal_action",
    "delist": "delisting",
    "listed": "listing",
    "fork": "fork",
    "upgrade": "protocol_upgrade",
    "merge": "protocol_upgrade",
    "hard fork": "fork",
    "airdrop": "airdrop",
    "inflation": "macro",
    "interest rate": "macro",
    "valuation": "macro",
    "bankrupt": "bankruptcy",
    "outage": "exchange_outage",
    "exploit": "exploit",
}

# Very small source trust map (expand)
SOURCE_TRUST = {
    "reuters": 1.0,
    "bloomberg": 0.95,
    "cointelegraph": 0.7,
    "coindesk": 0.75,
    "twitter": 0.4,
}

# Ticker regex mapping helper
TICKER_RE = re.compile(r"\b([A-Z]{2,5}|[A-Z]{2,5}[-/][A-Z]{2,5})\b")

def is_trusted_source(src_name: str) -> bool:
    return src_name and src_name.lower() in SOURCE_WHITELIST

def fetch_top_headlines(query: str = "crypto OR bitcoin OR ethereum", page_size: int = 20):
    if not NEWSAPI_KEY:
        print("Warning: NEWSAPI_KEY not set. Skipping fetch.")
        return []
        
    # Use NewsAPI
    url = ("https://newsapi.org/v2/everything?"
           f"q={requests.utils.quote(query)}&pageSize={page_size}&apiKey={NEWSAPI_KEY}")
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print("NewsAPI error", r.status_code, r.text)
            return []
        data = r.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def extract_text_from_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=6, headers={"User-Agent":"news-bot/1.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        texts = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(texts)
    except Exception:
        return ""

def finbert_sentiment(text: str) -> Dict[str, Any]:
    if not sentiment_pipe:
        return {"label": "NEUTRAL", "score": 0.0, "value": 0.0}
        
    if not text or len(text.strip()) < 10:
        return {"label": "NEUTRAL", "score": 0.0, "value": 0.0}
    try:
        # FinBERT returns labels like 'positive','negative','neutral' with scores
        out = sentiment_pipe(text[:512])  # truncate to first 512 tokens chars
        if isinstance(out, list) and len(out) > 0:
            o = out[0]
            lbl = o.get("label", "NEUTRAL").upper()
            score = float(o.get("score", 0.0))
            # Map to -1..1
            if lbl == "NEGATIVE":
                val = -score
            elif lbl == "POSITIVE":
                val = score
            else:
                val = 0.0
            return {"label": lbl, "score": score, "value": float(val)}
        return {"label": "NEUTRAL", "score": 0.0, "value": 0.0}
    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.0, "value": 0.0}

def extract_event_and_tickers(title: str, content: str) -> Dict[str, Any]:
    text = (title or "") + "\n" + (content or "")
    text_lower = text.lower()
    event = None
    for k, v in EVENT_KEYWORDS.items():
        if k in text_lower:
            event = v
            break
    tickers = []
    # naive ticker extraction
    for m in TICKER_RE.findall(text):
        # ignore common words like 'THE', 'USD' — simple heuristic
        if len(m) <= 5 and not m.lower() in ["usd", "eur", "the", "and"]:
            tickers.append(m)
    return {"event": event or "news", "tickers": list(set(tickers))}

def compute_source_trust(source: str):
    s = source.lower() if source else ""
    return SOURCE_TRUST.get(s, 0.5)

def compute_signal(sentiment_value: float, event: str, source_trust: float, corroboration:int=1) -> Dict[str, Any]:
    # base magnitude
    impact_map = {
        "exchange_hack": 1.5,
        "regulation": 1.4,
        "bankruptcy": 1.5,
        "protocol_upgrade": 0.8,
        "listing": 0.6,
        "delisting": 1.0,
        "fork": 0.8,
        "news": 0.5,
        "exchange_outage": 1.2,
        "exploit": 1.5,
    }
    event_impact = impact_map.get(event, 0.6)
    # corroboration boost
    coro_boost = math.log1p(corroboration)
    raw = sentiment_value * event_impact * source_trust * coro_boost
    # clamp [-1,1]
    score = max(-1.0, min(1.0, raw))
    # confidence heuristic
    confidence = min(0.99, source_trust * (0.5 + abs(sentiment_value))) * (0.5 + 0.5*(coro_boost/ (1+coro_boost)))
    decay_seconds = 3600  # default 1h
    if event in ("regulation", "bankruptcy"):
        decay_seconds = 24*3600
    elif event in ("exchange_hack", "exploit"):
        decay_seconds = 6*3600
    return {"score": float(score), "confidence": float(confidence), "decay": int(decay_seconds), "event": event, "impact": float(event_impact)}

def post_signal_to_trading(signal: Dict[str, Any]):
    try:
        r = requests.post(TRADING_SIGNALS_URL, json=signal, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print("post signal error", e)
        return False

def archive_article(article_raw: Dict[str,Any], parsed: Dict[str,Any], out_dir:Path):
    name = f"{int(time.time())}_{hash(article_raw.get('url','')) & 0xffffffff}.json"
    payload = {"raw": article_raw, "parsed": parsed}
    try:
        (out_dir / name).write_text(json.dumps(payload, indent=2))
    except Exception as e:
        print(f"Error archiving article: {e}")

# Main ingest loop
def run_pipeline():
    print("Starting news pipeline...")
    while True:
        try:
            articles = fetch_top_headlines()
            print(f"Fetched {len(articles)} articles")
            
            # group by title/url to reduce duplicates
            seen = set()
            for a in articles:
                url = a.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                source = a.get("source", {}).get("name", "").lower()
                title = a.get("title", "")
                description = a.get("description", "")
                content = a.get("content", "") or extract_text_from_url(url)
                published_at = a.get("publishedAt", datetime.utcnow().isoformat())
                trusted = is_trusted_source(source)
                source_trust = compute_source_trust(source)
                parsed_event = extract_event_and_tickers(title, content)
                sentiment = finbert_sentiment(title + "\n" + (description or "") + "\n" + (content or ""))
                # simple corroboration: count other headlines containing same ticker or title keywords (naive)
                corroboration = 1
                if parsed_event["tickers"]:
                    tk = parsed_event["tickers"][0]
                    corroboration = sum(1 for ar in articles if tk in (ar.get("title","") + ar.get("description","") + (ar.get("content") or "")).upper())
                sig = compute_signal(sentiment.get("value", 0.0), parsed_event["event"], source_trust, corroboration=corroboration)
                signal_payload = {
                    "timestamp": published_at,
                    "source": source,
                    "url": url,
                    "title": title,
                    "event": parsed_event["event"],
                    "tickers": parsed_event["tickers"],
                    "sentiment": sentiment,
                    "score": sig["score"],
                    "confidence": sig["confidence"],
                    "decay_seconds": sig["decay"],
                    "impact": sig["impact"],
                    "raw_article": {
                        "description": description,
                    }
                }
                # archival
                archive_article(a, signal_payload, ARCHIVE_DIR)
                # safety: only post signals if confidence > 0.4 and abs(score) > 0.15
                if sig["confidence"] >= 0.4 and abs(sig["score"]) >= 0.15:
                    ok = post_signal_to_trading(signal_payload)
                    print("posted signal", ok, signal_payload["title"][:80], "score", sig["score"], "conf", sig["confidence"])
                else:
                    print("skipping low-confidence signal", sig)
            
            print("Cycle complete. Sleeping for 5 minutes...")
            time.sleep(300)
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_pipeline()
