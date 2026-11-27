"""
stock_research_engine.py

Full research layer for every current Plan 2 Top-50 stock.

This layer DOES NOT change:
- Plan 2 ranking
- BUY/HOLD/SELL signal
- Capital allocation

Outputs:
    stock_research_ai.csv
    stock_research_sources.csv
"""

import json
import os
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
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]
# Deduplicate preserving order
MODELS = list(dict.fromkeys(FALLBACK_MODELS))

OFFICIAL_SOURCE_LINKS = {
    "NSE Corporate Filings": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    "SEBI Circulars": "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&smid=0SEBI&ssid=7",
    "RBI Press Releases": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
}

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


def num(v):
    try:
        val = float(v)
        return val if pd.notna(val) else None
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
            return round(float((close.iloc[-1] / close.iloc[-days - 1] - 1) * 100), 2)

        avg20 = volume.tail(20).mean() if len(volume) else None
        lastvol = volume.iloc[-1] if len(volume) else None

        return {
            "data_status": "available",
            "price": round(float(close.iloc[-1]), 2),
            "return_1d_pct": ret(1),
            "return_5d_pct": ret(5),
            "return_20d_pct": ret(20),
            "return_3m_pct": ret(63),
            "return_6m_pct": ret(126),
            "return_1y_pct": ret(252),
            "high_52w": round(float(close.tail(252).max()), 2),
            "low_52w": round(float(close.tail(252).min()), 2),
            "volume_last": num(lastvol),
            "volume_avg20": num(avg20),
            "volume_ratio": round(float(lastvol) / float(avg20), 2) if avg20 and avg20 > 0 else None,
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
            "heldPercentInsiders", "heldPercentInstitutions",
        ]
        for k in keys:
            if k in info and info[k] is not None:
                out[k] = info[k]
        out["sector_yf"] = info.get("sector", "")
        out["industry_yf"] = info.get("industry", "")
        out["data_status"] = "available" if len(out) > 3 else "unavailable"
    except Exception as exc:
        out["error"] = str(exc)[:200]
    return out


def quarterly_data(symbol):
    result = {"data_status": "unavailable", "quarters": []}
    try:
        t = yf.Ticker(symbol + ".NS")
        q = t.quarterly_income_stmt
        if q is None or q.empty:
            return result
        q = q.iloc[:, :4]
        wanted = [
            "Total Revenue", "Operating Revenue", "EBITDA",
            "EBIT", "Net Income", "Diluted EPS", "Basic EPS",
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
    result = {"data_status": "unavailable", "balance_sheet": {}, "cash_flow": {}}
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


def macro_context():
    return {
        "official_sources": OFFICIAL_SOURCE_LINKS,
        "note": (
            "Current policy/global event text is only included when fetched "
            "successfully from a source. Otherwise report as Not established."
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

    plan2_keys = [
        "Momentum 12-2", "Momentum %", "Momentum Rank", "ID", "ID Rank",
        "Portfolio Signal", "Signal", "Live Price", "Sector",
        "Momentum Quality", "AI View", "Evidence Confidence",
        "Risk Flags", "Why Selected", "Catalyst Assessment",
    ]

    return {
        "stock": symbol,
        "plan2": {k: s(row[k]) for k in row.index if k in plan2_keys},
        "price_volume": price,
        "fundamentals": fund,
        "quarterly": qtr,
        "balance_cashflow": bs,
        "recent_news": news,
        "risk_flags": risk_flags(price, fund, qtr),
        "macro_policy_sources": macro_context(),
    }


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
        "You are an evidence-first Indian equity research analyst. "
        "Use ONLY the supplied JSON data. Never invent a reason for a price move, "
        "policy effect, quarterly result, news event, or global impact. "
        "If evidence is missing, write 'No clear evidence available.' "
        "For government_policy_impact, rbi_impact, sebi_impact, and global_impact, "
        "write 'Not established.' unless the supplied evidence directly supports a connection. "
        "Do not give target prices, guaranteed returns, or alter Plan 2 ranking/allocation. "
        "Return a valid JSON object containing EXACTLY these keys: "
        "stock (string), executive_summary (string), plan2_interpretation (string), "
        "fundamental_picture (string), quarterly_picture (string), price_move_explanation (string), "
        "volume_liquidity (string), sector_impact (string), global_impact (string), "
        "government_policy_impact (string), rbi_impact (string), sebi_impact (string), "
        "positive_factors (array of strings), negative_factors (array of strings), "
        "consider (array of strings), do_not_assume (array of strings), "
        "watch_next (array of strings), risk_flags (array of strings), "
        "evidence_confidence (string: High, Medium, or Low)."
    )

    last_err = None
    for model in MODELS:
        payload = {
            "model": model,
            "temperature": 0.1,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Produce a concise, evidence-only JSON research report for this stock.\n"
                        "Evidence:\n" + safe_json(evidence)
                    ),
                },
            ],
        }

        for attempt in range(3):
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
                    wait = min(30, 3 * (2 ** attempt))
                    print(f"  Groq HTTP {r.status_code} on {model}; retrying in {wait}s...")
                    time.sleep(wait)
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    continue

                if not r.ok:
                    last_err = f"{model} HTTP {r.status_code}: {r.text[:300]}"
                    print(f"  Groq model {model} rejected request: {last_err}")
                    break

                content = r.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return normalize_report(evidence["stock"], parsed, evidence.get("risk_flags", []))
            except Exception as exc:
                last_err = f"{model}: {exc}"
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(last_err or "All Groq models failed")


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
            results.append(report)
            print(f"  AI research complete: {symbol}")
        except Exception as exc:
            results.append(fallback_report(symbol, flags=evidence.get("risk_flags", []) or ["AI unavailable"]))
            print(f"  AI research failed for {symbol}: {exc}")

        time.sleep(2.2)

    pd.DataFrame(results).to_csv(OUT, index=False)
    pd.DataFrame(
        source_rows,
        columns=["Stock", "Source Type", "Publisher", "Date", "Title", "URL"],
    ).to_csv(SOURCES_OUT, index=False)

    print(f"Wrote {OUT} ({len(results)} stocks)")
    print(f"Wrote {SOURCES_OUT} ({len(source_rows)} source rows)")


if __name__ == "__main__":
    main()
