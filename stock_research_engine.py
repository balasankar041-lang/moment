"""
stock_research_engine.py

Full research layer for every current Plan 2 Top-50 stock.

This layer DOES NOT change:
- Plan 2 ranking
- BUY/HOLD/SELL signal
- ₹20,000 allocation

It gathers available evidence and asks Groq to explain it.

Outputs:
    stock_research_ai.csv
    stock_research_sources.csv

Evidence buckets:
1. Plan 2 metrics
2. Price / volume behaviour
3. Fundamentals / fact-sheet data
4. Quarterly results
5. Balance-sheet / cash-flow data
6. Recent company news
7. Sector context
8. Global / macro context where available
9. Government / RBI / SEBI relevance
10. Risk flags
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
GROQ_MODEL = "openai/gpt-oss-20b"

UA = {
    "User-Agent": "Plan2-Research/1.0 research dashboard"
}

OFFICIAL_SOURCE_LINKS = {
    "NSE Corporate Filings": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    "SEBI Circulars": "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&smid=0SEBI&ssid=7",
    "RBI Press Releases": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
}


def s(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def num(v):
    try:
        return float(v)
    except Exception:
        return None


def safe_json(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)


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
            "Stock", "Momentum Quality", "Intelligence Score",
            "Evidence Confidence", "Risk Flags", "Recent 1M Return",
            "Recent Volatility", "Average Daily Traded Value",
            "Volume Anomaly", "Catalyst", "Why Selected"
        ]
        cols = [c for c in cols if c in intel.columns]
        if cols:
            top = top.merge(
                intel[cols].drop_duplicates("Stock"),
                on="Stock", how="left"
            )

    sec = load_csv(SECTORS)
    if not sec.empty:
        sec["Stock"] = sec.apply(get_symbol, axis=1)
        if "Sector" in sec.columns:
            top = top.merge(
                sec[["Stock", "Sector"]].drop_duplicates("Stock"),
                on="Stock", how="left", suffixes=("", "_sector")
            )
            if "Sector_sector" in top.columns:
                if "Sector" not in top.columns:
                    top["Sector"] = top["Sector_sector"]
                else:
                    top["Sector"] = top["Sector"].fillna(top["Sector_sector"])
                top.drop(columns=["Sector_sector"], inplace=True)

    return top.fillna("")


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
            return {"data_status": "unavailable"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Close"])
        close = df["Close"].astype(float)
        volume = df["Volume"].astype(float) if "Volume" in df else pd.Series(index=df.index, dtype=float)

        def ret(days):
            if len(close) <= days:
                return None
            return (close.iloc[-1] / close.iloc[-days-1] - 1) * 100

        avg20 = volume.tail(20).mean() if len(volume) else None
        lastvol = volume.iloc[-1] if len(volume) else None

        return {
            "data_status": "available",
            "price": float(close.iloc[-1]),
            "return_1d_pct": ret(1),
            "return_5d_pct": ret(5),
            "return_20d_pct": ret(20),
            "return_3m_pct": ret(63),
            "return_6m_pct": ret(126),
            "return_1y_pct": ret(252),
            "high_52w": float(close.tail(252).max()),
            "low_52w": float(close.tail(252).min()),
            "volume_last": num(lastvol),
            "volume_avg20": num(avg20),
            "volume_ratio": (float(lastvol) / float(avg20)) if avg20 and avg20 > 0 else None,
        }
    except Exception as exc:
        return {"data_status": "error", "error": str(exc)[:200]}


def fundamentals(symbol):
    out = {"data_status": "unavailable"}

    try:
        t = yf.Ticker(symbol + ".NS")
        info = t.info or {}

        keys = [
            "marketCap", "enterpriseValue", "trailingPE", "forwardPE",
            "priceToBook", "enterpriseToEbitda", "returnOnEquity",
            "returnOnAssets", "debtToEquity", "profitMargins",
            "operatingMargins", "grossMargins", "revenueGrowth",
            "earningsGrowth", "dividendYield", "bookValue",
            "trailingEps", "forwardEps", "sharesOutstanding",
            "heldPercentInsiders", "heldPercentInstitutions"
        ]

        for k in keys:
            if k in info and info[k] is not None:
                out[k] = info[k]

        out["sector_yf"] = info.get("sector", "")
        out["industry_yf"] = info.get("industry", "")
        out["data_status"] = "available"

    except Exception as exc:
        out["error"] = str(exc)[:200]

    return out


def quarterly_data(symbol):
    result = {
        "data_status": "unavailable",
        "quarters": []
    }

    try:
        t = yf.Ticker(symbol + ".NS")
        q = t.quarterly_income_stmt

        if q is None or q.empty:
            return result

        q = q.iloc[:, :4]

        wanted = [
            "Total Revenue",
            "Operating Revenue",
            "EBITDA",
            "EBIT",
            "Net Income",
            "Diluted EPS",
            "Basic EPS",
        ]

        rows = []
        for col in q.columns:
            item = {"period": str(col)}
            for field in wanted:
                if field in q.index:
                    value = q.loc[field, col]
                    if pd.notna(value):
                        item[field] = float(value)
            rows.append(item)

        result["data_status"] = "available"
        result["quarters"] = rows

    except Exception as exc:
        result["error"] = str(exc)[:200]

    return result


def balance_cashflow(symbol):
    result = {
        "data_status": "unavailable",
        "balance_sheet": {},
        "cash_flow": {}
    }

    try:
        t = yf.Ticker(symbol + ".NS")

        bs = t.balance_sheet
        cf = t.cashflow

        def latest(table, names):
            if table is None or table.empty:
                return None
            for name in names:
                if name in table.index:
                    value = table.loc[name].iloc[0]
                    if pd.notna(value):
                        return float(value)
            return None

        result["balance_sheet"] = {
            "total_assets": latest(bs, ["Total Assets"]),
            "total_debt": latest(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation"]),
            "cash": latest(bs, ["Cash Cash Equivalents And Short Term Investments"]),
            "equity": latest(bs, ["Stockholders Equity", "Common Stock Equity"]),
        }

        result["cash_flow"] = {
            "operating_cashflow": latest(cf, ["Operating Cash Flow"]),
            "free_cashflow": latest(cf, ["Free Cash Flow"]),
            "capital_expenditure": latest(cf, ["Capital Expenditure"]),
        }

        result["data_status"] = "available"

    except Exception as exc:
        result["error"] = str(exc)[:200]

    return result


def recent_news(symbol):
    """
    Yahoo Finance news is used only as a discovery feed.
    AI must not treat a headline as a confirmed fact unless the source itself
    supports the claim.
    """
    result = []

    try:
        t = yf.Ticker(symbol + ".NS")
        news = t.news or []

        for item in news[:8]:
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


def macro_context():
    # Official links are supplied so the AI can distinguish source availability.
    # This layer does not invent a current policy event.
    return {
        "official_sources": OFFICIAL_SOURCE_LINKS,
        "note": (
            "Current policy/global event text is only included when fetched "
            "successfully from a source. Otherwise report as unavailable."
        ),
    }


def risk_flags(price, fund, quarterly):
    flags = []

    vr = price.get("volume_ratio")
    if vr is not None and vr >= 3:
        flags.append("Unusual volume")

    dte = fund.get("debtToEquity")
    if dte is not None and dte > 150:
        flags.append("High debt/equity")

    pe = fund.get("trailingPE")
    if pe is not None and pe > 60:
        flags.append("High trailing P/E")

    rg = fund.get("revenueGrowth")
    if rg is not None and rg < 0:
        flags.append("Negative revenue growth")

    eg = fund.get("earningsGrowth")
    if eg is not None and eg < 0:
        flags.append("Negative earnings growth")

    if quarterly.get("data_status") != "available":
        flags.append("Quarterly data unavailable")

    if price.get("data_status") != "available":
        flags.append("Price data unavailable")

    return flags


def collect_stock(row):
    symbol = s(row["Stock"])

    price = price_and_volume(symbol)
    fund = fundamentals(symbol)
    qtr = quarterly_data(symbol)
    bs = balance_cashflow(symbol)
    news = recent_news(symbol)

    return {
        "stock": symbol,
        "plan2": {k: s(row[k]) for k in row.index if k in [
            "Momentum 12-2", "Momentum Rank", "ID", "ID Rank",
            "Portfolio Signal", "Signal", "Price", "Live Price",
            "Sector", "Momentum Quality", "Intelligence Score",
            "Evidence Confidence", "Risk Flags"
        ]},
        "price_volume": price,
        "fundamentals": fund,
        "quarterly": qtr,
        "balance_cashflow": bs,
        "recent_news": news,
        "risk_flags": risk_flags(price, fund, qtr),
        "macro_policy_sources": macro_context(),
    }


def groq_research(api_key, evidence):
    schema = {
        "type": "object",
        "properties": {
            "stock": {"type": "string"},
            "executive_summary": {"type": "string"},
            "plan2_interpretation": {"type": "string"},
            "fundamental_picture": {"type": "string"},
            "quarterly_picture": {"type": "string"},
            "price_move_explanation": {"type": "string"},
            "volume_liquidity": {"type": "string"},
            "sector_impact": {"type": "string"},
            "global_impact": {"type": "string"},
            "government_policy_impact": {"type": "string"},
            "rbi_impact": {"type": "string"},
            "sebi_impact": {"type": "string"},
            "positive_factors": {
                "type": "array",
                "items": {"type": "string"}
            },
            "negative_factors": {
                "type": "array",
                "items": {"type": "string"}
            },
            "consider": {
                "type": "array",
                "items": {"type": "string"}
            },
            "do_not_assume": {
                "type": "array",
                "items": {"type": "string"}
            },
            "watch_next": {
                "type": "array",
                "items": {"type": "string"}
            },
            "risk_flags": {
                "type": "array",
                "items": {"type": "string"}
            },
            "evidence_confidence": {"type": "string"}
        },
        "required": [
            "stock", "executive_summary", "plan2_interpretation",
            "fundamental_picture", "quarterly_picture",
            "price_move_explanation", "volume_liquidity",
            "sector_impact", "global_impact",
            "government_policy_impact", "rbi_impact", "sebi_impact",
            "positive_factors", "negative_factors", "consider",
            "do_not_assume", "watch_next", "risk_flags",
            "evidence_confidence"
        ],
        "additionalProperties": False
    }

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.1,
        "reasoning_effort": "high",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evidence-first Indian equity research analyst. "
                    "Use ONLY the supplied data. Never invent a reason for a price move, "
                    "policy effect, quarterly result, news event or global impact. "
                    "If evidence is missing, say 'No clear evidence available'. "
                    "Do not provide guaranteed returns, target prices, or certainty. "
                    "Do not alter Plan 2 or the capital allocation. "
                    "Distinguish documented facts from inference. "
                    "For Government/RBI/SEBI/global impact, say Not established "
                    "unless the supplied evidence supports a connection. "
                    "A news headline is not proof of causation. "
                    "Return only the requested JSON schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a complete research report for this stock. "
                    "Explain what can reasonably be considered and what should not "
                    "be assumed. Evidence:\n" + safe_json(evidence)
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "stock_research",
                "strict": True,
                "schema": schema,
            },
        },
    }

    r = requests.post(
        GROQ_URL,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    r.raise_for_status()

    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def main():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY GitHub Secret is not available.")

    top = get_plan2_data()

    results = []
    source_rows = []

    for i, (_, row) in enumerate(top.iterrows(), start=1):
        symbol = s(row["Stock"])
        print(f"[{i}/{len(top)}] Collecting {symbol}")

        evidence = collect_stock(row)

        for n in evidence.get("recent_news", []):
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
            report["Stock"] = symbol
            results.append(report)
            print(f"  AI research complete: {symbol}")
        except Exception as exc:
            results.append({
                "Stock": symbol,
                "executive_summary": "AI research unavailable for this run.",
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
                "risk_flags": ["AI unavailable"],
                "evidence_confidence": "Low",
            })
            print(f"  AI research failed for {symbol}: {exc}")

        # Keep requests paced for free-tier operation.
        time.sleep(1)

    pd.DataFrame(results).to_csv(OUT, index=False)
    pd.DataFrame(source_rows).to_csv(SOURCES_OUT, index=False)

    print(f"Wrote {OUT} ({len(results)} stocks)")
    print(f"Wrote {SOURCES_OUT} ({len(source_rows)} source rows)")


if __name__ == "__main__":
    main()
