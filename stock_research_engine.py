"""
stock_research_engine.py

Full research layer for every current Plan 2 Top-50 stock.

This layer DOES NOT change:
- Plan 2 ranking
- BUY/HOLD/SELL signal
- Capital allocation

Configured for Groq free/on-demand tier (openai/gpt-oss-20b, 8,000 TPM):
- Uses reasoning_effort="low" and include_reasoning=False so reasoning tokens
  do not exhaust completion budget.
- Omits brittle server-side json_object validation and parses JSON cleanly in Python.
- Sends a compact evidence payload (~400 tokens) and paces requests + respects
  HTTP 429 Retry-After headers to stay within the 8,000 TPM limit.

Outputs:
    stock_research_ai.csv
    stock_research_sources.csv
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

TOP50 = Path("live_plan2_top50.csv")
INTEL = Path("advanced_market_intelligence.csv")
SECTORS = Path("sector_wise_top50.csv")

OUT = Path("stock_research_ai.csv")
SOURCES_OUT = Path("stock_research_sources.csv")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"

# Free tier for openai/gpt-oss-20b is 8,000 TPM.
# Compact prompt + concise JSON output uses ~1,000-1,200 tokens/request.
# Pacing at ~10s per stock keeps throughput comfortably under 8,000 TPM.
REQUEST_PACE_SECONDS = float(os.getenv("GROQ_PACE_SECONDS", "10.0"))

STRING_FIELDS = [
    "executive_summary",
    "plan2_interpretation",
    "fundamental_picture",
    "quarterly_picture",
    "price_move_explanation",
    "volume_liquidity",
    "sector_impact",
    "global_impact",
    "government_policy_impact",
    "rbi_impact",
    "sebi_impact",
    "evidence_confidence",
]

LIST_FIELDS = [
    "positive_factors",
    "negative_factors",
    "consider",
    "do_not_assume",
    "watch_next",
    "risk_flags",
]


def s(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def num(v, digits=2):
    try:
        val = float(v)
        if pd.isna(val):
            return None
        return round(val, digits)
    except Exception:
        return None


def safe_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def get_symbol(row):
    for c in ("Stock", "Symbol", "Ticker"):
        if c in row and s(row[c]):
            return s(row[c])
    return ""


def get_plan2_data():
    top = load_csv(TOP50)
    if top.empty:
        raise RuntimeError("live_plan2_top50.csv is missing or empty.")

    top["Stock"] = top.apply(get_symbol, axis=1)

    intel = load_csv(INTEL)
    if not intel.empty:
        intel["Stock"] = intel.apply(get_symbol, axis=1)
        cols = [
            "Stock", "Momentum %", "Momentum Quality", "AI View",
            "Evidence Confidence", "Risk Flags", "1D Return %",
            "5D Return %", "20D Return %", "Volume vs 20D Avg",
            "Catalyst Assessment", "Why Selected", "Market Proxy",
            "Market 20D %",
        ]
        cols = [c for c in cols if c in intel.columns]
        if len(cols) > 1:
            top = top.merge(
                intel[cols].drop_duplicates("Stock"),
                on="Stock", how="left", suffixes=("", "_intel"),
            )

    sec = load_csv(SECTORS)
    if not sec.empty:
        sec["Stock"] = sec.apply(get_symbol, axis=1)
        if "Sector" in sec.columns:
            top = top.merge(
                sec[["Stock", "Sector"]].drop_duplicates("Stock"),
                on="Stock", how="left", suffixes=("", "_sector"),
            )
            if "Sector_sector" in top.columns:
                if "Sector" not in top.columns:
                    top["Sector"] = top["Sector_sector"]
                else:
                    top["Sector"] = top["Sector"].fillna(top["Sector_sector"])
                top.drop(columns=["Sector_sector"], inplace=True)

    # Avoid Pandas4Warning when filling mixed float64/object columns
    return top.astype(object).fillna("")


def price_and_volume(symbol):
    try:
        df = yf.download(
            symbol + ".NS",
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            return {"status": "unavailable"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Close"])
        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float) if "Volume" in df else pd.Series(index=df.index, dtype=float)

        def ret(days):
            if len(close) <= days:
                return None
            return round(float((close.iloc[-1] / close.iloc[-days - 1] - 1) * 100), 1)

        avg20 = volume.tail(20).mean() if len(volume) else None
        lastvol = volume.iloc[-1] if len(volume) else None

        out = {
            "status": "ok",
            "price": round(float(close.iloc[-1]), 2),
            "r1d": ret(1),
            "r5d": ret(5),
            "r20d": ret(20),
            "r3m": ret(63),
            "r1y": ret(252),
            "hi52w": round(float(close.tail(252).max()), 2),
            "lo52w": round(float(close.tail(252).min()), 2),
            "vol_ratio": round(float(lastvol) / float(avg20), 2) if avg20 and avg20 > 0 else None,
        }
        return {k: v for k, v in out.items() if v is not None}
    except Exception:
        return {"status": "error"}


def fundamentals(symbol):
    out = {"status": "unavailable"}
    try:
        t = yf.Ticker(symbol + ".NS")
        info = t.info or {}
        key_map = {
            "marketCap": "mcap_cr",
            "trailingPE": "pe",
            "forwardPE": "fpe",
            "priceToBook": "pb",
            "returnOnEquity": "roe",
            "debtToEquity": "de",
            "profitMargins": "pm",
            "operatingMargins": "om",
            "revenueGrowth": "rev_g",
            "earningsGrowth": "earn_g",
            "dividendYield": "div_y",
        }
        for raw_k, short_k in key_map.items():
            val = info.get(raw_k)
            if val is not None:
                if raw_k == "marketCap":
                    out[short_k] = num(float(val) / 1e7, 0)
                else:
                    out[short_k] = num(val, 2)
        if info.get("industry"):
            out["industry"] = s(info.get("industry"))
        if len(out) > 1:
            out["status"] = "ok"
    except Exception:
        pass
    return out


def quarterly_data(symbol):
    result = {"status": "unavailable", "q": []}
    try:
        t = yf.Ticker(symbol + ".NS")
        q = t.quarterly_income_stmt
        if q is None or q.empty:
            return result
        q = q.iloc[:, :2]  # Latest 2 quarters to keep token count small
        rows = []
        for col in q.columns:
            item = {"p": str(col)[:10]}
            for field, short in (("Total Revenue", "rev_cr"), ("Net Income", "ni_cr"), ("Diluted EPS", "eps")):
                if field in q.index:
                    value = q.loc[field, col]
                    if pd.notna(value):
                        item[short] = round(float(value) / 1e7, 1) if short.endswith("_cr") else round(float(value), 2)
            rows.append(item)
        result["status"] = "ok"
        result["q"] = rows
    except Exception:
        pass
    return result


def recent_news(symbol):
    result = []
    try:
        t = yf.Ticker(symbol + ".NS")
        news = t.news or []
        for item in news[:6]:
            content = item.get("content", item)
            title = content.get("title", "")
            publisher = content.get("provider", {}).get("displayName", "")
            url = (
                content.get("canonicalUrl", {}).get("url")
                or content.get("clickThroughUrl", {}).get("url")
                or ""
            )
            pub = content.get("pubDate", "")
            if title:
                result.append({
                    "title": s(title),
                    "publisher": s(publisher),
                    "date": s(pub),
                    "url": s(url),
                })
    except Exception:
        pass
    return result


def risk_flags(price, fund, quarterly):
    flags = []
    vr = price.get("vol_ratio")
    if vr is not None and vr >= 3:
        flags.append("Unusual volume")
    dte = fund.get("de")
    if dte is not None and dte > 150:
        flags.append("High debt/equity")
    pe = fund.get("pe")
    if pe is not None and pe > 60:
        flags.append("High trailing P/E")
    rg = fund.get("rev_g")
    if rg is not None and rg < 0:
        flags.append("Negative revenue growth")
    eg = fund.get("earn_g")
    if eg is not None and eg < 0:
        flags.append("Negative earnings growth")
    if quarterly.get("status") != "ok":
        flags.append("Quarterly data unavailable")
    if price.get("status") != "ok":
        flags.append("Price data unavailable")
    return flags


def collect_stock(row):
    symbol = s(row["Stock"])
    price = price_and_volume(symbol)
    fund = fundamentals(symbol)
    qtr = quarterly_data(symbol)
    news_full = recent_news(symbol)

    # Compact prompt payload to stay well under Groq's 8,000 TPM free-tier limit
    plan2_compact = {}
    for k, short in (
        ("ID Rank", "id_rank"),
        ("Momentum Rank", "mom_rank"),
        ("Momentum %", "mom_pct"),
        ("ID", "id"),
        ("Portfolio Signal", "signal"),
        ("Sector", "sector"),
        ("Momentum Quality", "mom_q"),
        ("Risk Flags", "plan2_risks"),
        ("Market 20D %", "mkt_20d"),
    ):
        if k in row.index and s(row[k]):
            plan2_compact[short] = s(row[k])

    compact_news = [
        f"{n['title'][:110]} ({n['date'][:10]})"
        for n in news_full[:3]
    ]

    evidence = {
        "stock": symbol,
        "plan2": plan2_compact,
        "px": price,
        "fund": fund,
        "qtr": qtr,
        "news": compact_news,
        "flags": risk_flags(price, fund, qtr),
    }
    return evidence, news_full


def extract_json_object(text):
    """Extract a JSON object from raw model output (handles code fences or stray text)."""
    text = s(text)
    if not text:
        raise ValueError("Empty response content from model")

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: locate outermost {...}
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(cleaned[start:end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not parse JSON from model output: {text[:180]}")


def normalize_report(symbol, parsed, fallback_flags):
    out = {"Stock": symbol}
    for f in STRING_FIELDS:
        val = parsed.get(f, "")
        out[f] = s(val) if val else ("Low" if f == "evidence_confidence" else "No clear evidence available.")
    for f in LIST_FIELDS:
        val = parsed.get(f, [])
        if isinstance(val, str):
            val = [val] if val.strip() else []
        elif not isinstance(val, list):
            val = []
        out[f] = [s(x) for x in val if s(x)]
    if not out["risk_flags"] and fallback_flags:
        out["risk_flags"] = fallback_flags
    return out


def fallback_report(symbol, reason="AI research unavailable for this run.", flags=None):
    return {
        "Stock": symbol,
        "executive_summary": reason,
        "plan2_interpretation": "Use the Plan 2 fields supplied on the dashboard.",
        "fundamental_picture": "No AI conclusion.",
        "quarterly_picture": "No AI conclusion.",
        "price_move_explanation": "No clear evidence available.",
        "volume_liquidity": "No AI conclusion.",
        "sector_impact": "Not established.",
        "global_impact": "Not established.",
        "government_policy_impact": "Not established.",
        "rbi_impact": "Not established.",
        "sebi_impact": "Not established.",
        "positive_factors": [],
        "negative_factors": [],
        "consider": ["Retry after the AI/data service is available."],
        "do_not_assume": ["Do not infer a catalyst from price movement alone."],
        "watch_next": ["Next quarterly result", "New company filings", "Material sector/policy changes"],
        "risk_flags": flags if flags else ["AI unavailable"],
        "evidence_confidence": "Low",
    }


def groq_research(api_key, evidence):
    system_prompt = (
        "You are an evidence-first Indian equity analyst. Use ONLY the supplied JSON. "
        "Never invent facts, target prices, or causes. If missing, write 'No clear evidence available.' "
        "For global_impact, government_policy_impact, rbi_impact, sebi_impact write 'Not established.' "
        "unless directly supported by supplied news. Keep every string brief (1-2 sentences, <=25 words) "
        "and each array <=3 short items. "
        "Reply with ONLY a raw JSON object (no markdown, no explanation) having keys: "
        "executive_summary, plan2_interpretation, fundamental_picture, quarterly_picture, "
        "price_move_explanation, volume_liquidity, sector_impact, global_impact, "
        "government_policy_impact, rbi_impact, sebi_impact, evidence_confidence (High/Medium/Low), "
        "positive_factors, negative_factors, consider, do_not_assume, watch_next, risk_flags."
    )

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.1,
        "reasoning_effort": "low",
        "include_reasoning": False,
        "max_completion_tokens": 1800,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Evidence JSON: " + safe_json(evidence)},
        ],
    }

    last_err = None
    for attempt in range(4):
        try:
            r = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90,
            )

            if r.status_code == 429 or r.status_code >= 500:
                retry_hdr = r.headers.get("retry-after") or r.headers.get("Retry-After")
                if retry_hdr:
                    try:
                        wait = min(65.0, float(retry_hdr) + 1.5)
                    except Exception:
                        wait = min(60.0, 12.0 * (attempt + 1))
                else:
                    # Parse "Please try again in X.Xs" if present in Groq error message
                    m = re.search(r"try again in ([0-9.]+)s", r.text, flags=re.IGNORECASE)
                    wait = min(65.0, float(m.group(1)) + 2.0) if m else min(60.0, 12.0 * (attempt + 1))
                print(f"  Groq HTTP {r.status_code}; waiting {wait:.1f}s before retry {attempt + 1}/4...")
                time.sleep(wait)
                last_err = f"HTTP {r.status_code}: {r.text[:220]}"
                continue

            if not r.ok:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                print(f"  Groq rejected request: {last_err}")
                break

            data = r.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            parsed = extract_json_object(content)
            return normalize_report(evidence["stock"], parsed, evidence.get("flags", []))
        except Exception as exc:
            last_err = str(exc)
            print(f"  Attempt {attempt + 1}/4 parse/request error: {last_err[:180]}")
            time.sleep(4 * (attempt + 1))

    raise RuntimeError(last_err or "Groq request failed")


def main():
    top = get_plan2_data()
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    results = []
    source_rows = []

    if not api_key:
        print("WARNING: GROQ_API_KEY is not set. Writing fallback research CSV without crashing pipeline.")
        for _, row in top.iterrows():
            symbol = s(row["Stock"])
            results.append(fallback_report(symbol, "GROQ_API_KEY secret is not configured."))
        pd.DataFrame(results).to_csv(OUT, index=False)
        pd.DataFrame(source_rows, columns=["Stock", "Source Type", "Publisher", "Date", "Title", "URL"]).to_csv(SOURCES_OUT, index=False)
        return

    print(f"Using Groq model: {GROQ_MODEL} | Pace: {REQUEST_PACE_SECONDS}s/stock")
    ok_count = 0

    for i, (_, row) in enumerate(top.iterrows(), start=1):
        symbol = s(row["Stock"])
        t0 = time.time()
        print(f"[{i}/{len(top)}] Collecting {symbol}")
        evidence, news_full = collect_stock(row)

        for n in news_full:
            source_rows.append({
                "Stock": symbol,
                "Source Type": "News discovery",
                "Publisher": n.get("publisher", ""),
                "Date": n.get("date", ""),
                "Title": n.get("title", ""),
                "URL": n.get("url", ""),
            })

        try:
            report = groq_research(api_key, evidence)
            results.append(report)
            ok_count += 1
            print(f"  AI research complete: {symbol} (success {ok_count}/{i})")
        except Exception as exc:
            results.append(fallback_report(symbol, flags=evidence.get("flags", []) or ["AI unavailable"]))
            print(f"  AI research failed for {symbol}: {exc}")

        elapsed = time.time() - t0
        sleep_needed = max(0.0, REQUEST_PACE_SECONDS - elapsed)
        if i < len(top) and sleep_needed > 0:
            time.sleep(sleep_needed)

    pd.DataFrame(results).to_csv(OUT, index=False)
    pd.DataFrame(
        source_rows,
        columns=["Stock", "Source Type", "Publisher", "Date", "Title", "URL"],
    ).to_csv(SOURCES_OUT, index=False)

    print(f"Wrote {OUT} ({ok_count}/{len(results)} AI reports succeeded)")
    print(f"Wrote {SOURCES_OUT} ({len(source_rows)} source rows)")


if __name__ == "__main__":
    main()
