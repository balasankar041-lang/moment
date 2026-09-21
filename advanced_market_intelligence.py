"""
Advanced Free Market Intelligence Engine
========================================
Evidence-first intelligence for Plan 2 Top 50 stocks.

No paid AI/API key required.

Adds:
- Plan 2 selection explanation
- Momentum / ID quality
- Risk flags
- Recent price behaviour (when yfinance is available)
- Volume anomaly
- Market context (Nifty 500 proxy via ^CRSLDX when available)
- Sector context from sector_wise_top50.csv
- News headline collection from Yahoo RSS when network access is available
- Evidence-based catalyst labels; never invents a reason
- Overall market summary
- Stock-level concise explanation

This module NEVER changes Plan 2 ranking or ₹20,000 allocation.
"""

from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET
import urllib.request
import ssl
import pandas as pd
import numpy as np

TOP50 = Path("live_plan2_top50.csv")
SECTORS = Path("sector_wise_top50.csv")
OUT = Path("advanced_market_intelligence.csv")
SUMMARY = Path("market_intelligence_summary.csv")

try:
    import yfinance as yf
except Exception:
    yf = None

UA = "Mozilla/5.0 (compatible; Plan2MarketIntelligence/1.0)"


def get(row, *names, default=np.nan):
    for name in names:
        if name in row.index:
            v = row[name]
            if pd.notna(v) and str(v).strip():
                try:
                    return float(v)
                except Exception:
                    return v
    return default


def stock_name(row):
    return str(row.get("Stock", row.get("Symbol", row.get("Ticker", "")))).strip()


def fetch_news(symbol):
    """Best-effort public RSS headlines. Empty result is valid."""
    try:
        q = quote(f"{symbol} NSE stock")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=8, context=ctx) as r:
            root = ET.fromstring(r.read())
        items = []
        for item in root.findall(".//item")[:5]:
            title = item.findtext("title") or ""
            pub = item.findtext("pubDate") or ""
            if title:
                items.append(f"{title} [{pub}]")
        return items
    except Exception:
        return []


def price_context(symbol):
    if yf is None:
        return {}
    try:
        t = yf.Ticker(symbol + ".NS")
        d = t.history(period="3mo", auto_adjust=False)
        if d is None or d.empty or len(d) < 25:
            return {}
        close = d["Close"].dropna()
        vol = d["Volume"].dropna()
        last = float(close.iloc[-1])
        ret1 = float(close.iloc[-1] / close.iloc[-2] - 1) * 100
        ret5 = float(close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else np.nan
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else np.nan
        avg20v = float(vol.tail(20).mean()) if len(vol) >= 20 else np.nan
        lastv = float(vol.iloc[-1]) if len(vol) else np.nan
        vol_ratio = lastv / avg20v if avg20v > 0 else np.nan
        high20 = float(close.tail(20).max())
        low20 = float(close.tail(20).min())
        return {
            "Last Price": last,
            "1D Return %": ret1,
            "5D Return %": ret5,
            "20D Return %": ret20,
            "Volume vs 20D Avg": vol_ratio,
            "20D High Distance %": (last / high20 - 1) * 100 if high20 else np.nan,
            "20D Low Distance %": (last / low20 - 1) * 100 if low20 else np.nan,
        }
    except Exception:
        return {}


def market_context():
    if yf is None:
        return {}
    for ticker in ["^CRSLDX", "^NSEI"]:
        try:
            d = yf.Ticker(ticker).history(period="1mo", auto_adjust=False)
            c = d["Close"].dropna()
            if len(c) >= 6:
                return {
                    "Market Proxy": ticker,
                    "Market 1D %": float((c.iloc[-1] / c.iloc[-2] - 1) * 100),
                    "Market 5D %": float((c.iloc[-1] / c.iloc[-6] - 1) * 100),
                    "Market 20D %": float((c.iloc[-1] / c.iloc[-21] - 1) * 100) if len(c) >= 21 else np.nan,
                }
        except Exception:
            pass
    return {}


def momentum_quality(x):
    if not np.isfinite(x):
        return "Unknown"
    if x >= 75:
        return "Very strong"
    if x >= 40:
        return "Strong"
    if x >= 15:
        return "Positive"
    if x >= 0:
        return "Weak positive"
    return "Negative"


if not TOP50.exists():
    raise SystemExit("live_plan2_top50.csv not found")

df = pd.read_csv(TOP50)
if df.empty:
    pd.DataFrame().to_csv(OUT, index=False)
    raise SystemExit("Top50 is empty")

sector_map = {}
if SECTORS.exists():
    sec = pd.read_csv(SECTORS)
    for _, r in sec.iterrows():
        s = stock_name(r)
        if s:
            sector_map[s] = str(r.get("Sector", "Miscellaneous"))

mkt = market_context()
rows = []

for _, r in df.iterrows():
    symbol = stock_name(r)
    momentum = get(r, "Momentum %", "Momentum", "Momentum_Return")
    ident = get(r, "ID", "Id")
    mrank = get(r, "Momentum Rank", "Momentum_Rank", "Rank")
    idrank = get(r, "ID Rank", "ID_Rank")
    signal = str(r.get("Signal", "HOLD")).upper()
    sector = sector_map.get(symbol, "Miscellaneous")

    pc = price_context(symbol)
    headlines = fetch_news(symbol)

    evidence = []
    risks = []

    if np.isfinite(mrank):
        evidence.append(f"momentum rank #{int(mrank)}")
    if np.isfinite(momentum):
        evidence.append(f"12–2M momentum {momentum:.1f}%")
    if np.isfinite(idrank):
        evidence.append(f"ID rank #{int(idrank)}")
    if np.isfinite(ident):
        evidence.append(f"ID {ident:.4f}")

    mq = momentum_quality(momentum)
    if np.isfinite(momentum) and momentum < 10:
        risks.append("weak momentum")
    if np.isfinite(momentum) and momentum < 0:
        risks.append("negative momentum")
    if np.isfinite(idrank) and idrank > 40:
        risks.append("lower final rank")

    price_reason = "Recent price data unavailable"
    if pc:
        r1, r5, r20 = pc.get("1D Return %"), pc.get("5D Return %"), pc.get("20D Return %")
        vr = pc.get("Volume vs 20D Avg")
        price_reason = f"1D {r1:+.2f}%, 5D {r5:+.2f}%, 20D {r20:+.2f}%"
        if np.isfinite(vr) and vr >= 2:
            price_reason += f"; unusual volume ({vr:.1f}x 20D average)"
            evidence.append("unusual volume")
        if np.isfinite(r20) and np.isfinite(mkt.get("Market 20D %", np.nan)):
            rel = r20 - mkt["Market 20D %"]
            if rel >= 5:
                evidence.append("outperforming market over 20D")
            elif rel <= -5:
                risks.append("underperforming market over 20D")

    news_text = "No public headline evidence retrieved"
    if headlines:
        news_text = " | ".join(headlines[:3])

    if signal == "BUY":
        action = "NEW TOP 50 ENTRY"
    elif signal == "SELL":
        action = "TOP 50 EXIT / REVIEW"
    else:
        action = "HOLD / CONTINUES"

    # No headline is interpreted as no catalyst, not as a negative catalyst.
    if headlines:
        catalyst = "News evidence available; headline requires manual verification."
    else:
        catalyst = "No clear catalyst identified from available public headlines."

    explanation = (
        f"{symbol}: {action}. "
        f"Plan 2 evidence: {', '.join(evidence) if evidence else 'current ranking data'}. "
        f"Momentum quality: {mq}. "
        f"Price context: {price_reason}. "
        f"Sector: {sector}. "
        f"{catalyst}"
    )

    why_selected = (
        f"Plan 2 Top 50: momentum rank {mrank if np.isfinite(mrank) else 'N/A'}, "
        f"ID rank {idrank if np.isfinite(idrank) else 'N/A'}, "
        f"momentum {momentum:.2f}%."
        if np.isfinite(momentum)
        else (
            f"Plan 2 Top 50: momentum rank {mrank if np.isfinite(mrank) else 'N/A'}, "
            f"ID rank {idrank if np.isfinite(idrank) else 'N/A'}."
        )
    )

    # Confidence is evidence availability, not a prediction or recommendation.
    confidence_parts = 0
    if pc:
        confidence_parts += 1
    if mkt:
        confidence_parts += 1
    if headlines:
        confidence_parts += 1
    confidence = "High" if confidence_parts >= 3 else ("Medium" if confidence_parts >= 1 else "Low")

    # Keep the AI View descriptive: report the Plan 2 signal and available
    # evidence rather than turning the layer into a new trading rule.
    quality_view = mq if mq and str(mq).strip().lower() != "unknown" else ""
    ai_view = f"{action}" + (f"; {quality_view}" if quality_view else "")

    rows.append({
        "Stock": symbol,
        "Sector": sector,
        "Signal": signal,
        "Momentum %": momentum,
        "Momentum Quality": mq,
        "Momentum Rank": mrank,
        "ID": ident,
        "ID Rank": idrank,
        **pc,
        **mkt,
        "AI View": ai_view,
        "Why Selected": why_selected,
        "Risk Flags": "; ".join(risks) if risks else "No major Plan 2 risk flag",
        "Evidence Confidence": confidence,
        "News Evidence": news_text,
        "Catalyst Assessment": catalyst,
        "AI-Style Explanation": explanation,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

summary = pd.DataFrame([{
    "Top 50 Stocks": len(out),
    "BUY": int((out["Signal"] == "BUY").sum()),
    "HOLD": int((out["Signal"] == "HOLD").sum()),
    "SELL": int((out["Signal"] == "SELL").sum()),
    "News Evidence Stocks": int((out["News Evidence"] != "No public headline evidence retrieved").sum()),
    "Market Proxy": mkt.get("Market Proxy", "Unavailable"),
    "Market 1D %": mkt.get("Market 1D %", np.nan),
    "Market 5D %": mkt.get("Market 5D %", np.nan),
    "Market 20D %": mkt.get("Market 20D %", np.nan),
}])
summary.to_csv(SUMMARY, index=False)

print(f"Created {OUT} with {len(out)} stocks")
print(f"Created {SUMMARY}")
