"""
Financial Data service using yfinance.
Port: 8107
"""

import logging
import time
from typing import Optional

import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("findata")

app = FastAPI(title="Local FinData Service", version="1.0.0")

# Simple TTL cache: {key: (data, timestamp)}
_cache: dict[str, tuple[object, float]] = {}
CACHE_TTL = 60  # seconds


def _get_cached(key: str):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
        del _cache[key]
    return None


def _set_cached(key: str, data: object):
    _cache[key] = (data, time.time())
    # Evict old entries if cache grows too large
    if len(_cache) > 200:
        cutoff = time.time() - CACHE_TTL
        for k in list(_cache):
            if _cache[k][1] < cutoff:
                del _cache[k]


def _df_to_records(df):
    """Convert a pandas DataFrame to a list of dicts with string index."""
    if df is None or df.empty:
        return []
    df = df.copy()
    if hasattr(df.index, "strftime"):
        df.index = df.index.strftime("%Y-%m-%d")
    df = df.reset_index()
    # Convert any remaining Timestamp columns
    for col in df.columns:
        if hasattr(df[col], "dt"):
            try:
                df[col] = df[col].astype(str)
            except Exception:
                pass
    return df.to_dict("records")


# --- Request models ---

class TickerRequest(BaseModel):
    ticker: str

class HistoryRequest(BaseModel):
    ticker: str
    period: str = "1mo"
    interval: str = "1d"

class FinancialsRequest(BaseModel):
    ticker: str
    statement: str = "income"  # income, balance, cashflow

class DownloadRequest(BaseModel):
    tickers: str  # comma-separated
    period: str = "5d"
    interval: str = "1d"


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "service": "findata", "library": "yfinance"}


@app.post("/quote")
async def quote(req: TickerRequest):
    try:
        cache_key = f"quote:{req.ticker}"
        cached = _get_cached(cache_key)
        if cached:
            return JSONResponse(cached)

        t = yf.Ticker(req.ticker)
        info = t.fast_info

        result = {
            "ticker": req.ticker.upper(),
            "price": getattr(info, "last_price", None),
            "previous_close": getattr(info, "previous_close", None),
            "open": getattr(info, "open", None),
            "day_low": getattr(info, "day_low", None),
            "day_high": getattr(info, "day_high", None),
            "volume": getattr(info, "last_volume", None),
            "market_cap": getattr(info, "market_cap", None),
            "fifty_day_average": getattr(info, "fifty_day_average", None),
            "two_hundred_day_average": getattr(info, "two_hundred_day_average", None),
            "currency": getattr(info, "currency", None),
            "exchange": getattr(info, "exchange", None),
        }

        _set_cached(cache_key, result)
        return JSONResponse(result)
    except Exception as e:
        log.error("quote error for %s: %s", req.ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/history")
async def history(req: HistoryRequest):
    try:
        cache_key = f"history:{req.ticker}:{req.period}:{req.interval}"
        cached = _get_cached(cache_key)
        if cached:
            return JSONResponse(cached)

        t = yf.Ticker(req.ticker)
        df = t.history(period=req.period, interval=req.interval)
        records = _df_to_records(df)

        result = {
            "ticker": req.ticker.upper(),
            "period": req.period,
            "interval": req.interval,
            "count": len(records),
            "data": records,
        }

        _set_cached(cache_key, result)
        return JSONResponse(result)
    except Exception as e:
        log.error("history error for %s: %s", req.ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/info")
async def info(req: TickerRequest):
    try:
        cache_key = f"info:{req.ticker}"
        cached = _get_cached(cache_key)
        if cached:
            return JSONResponse(cached)

        t = yf.Ticker(req.ticker)
        data = t.info

        if not data or data.get("trailingPegRatio") is None and data.get("symbol") is None:
            return JSONResponse({"ticker": req.ticker.upper(), "error": "Ticker not found or no data available"})

        # Clean up any non-serializable values
        clean = {}
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                clean[k] = v
            else:
                clean[k] = str(v)

        _set_cached(cache_key, clean)
        return JSONResponse(clean)
    except Exception as e:
        log.error("info error for %s: %s", req.ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/financials")
async def financials(req: FinancialsRequest):
    try:
        cache_key = f"financials:{req.ticker}:{req.statement}"
        cached = _get_cached(cache_key)
        if cached:
            return JSONResponse(cached)

        t = yf.Ticker(req.ticker)

        if req.statement == "income":
            annual = _df_to_records(t.income_stmt)
            quarterly = _df_to_records(t.quarterly_income_stmt)
        elif req.statement == "balance":
            annual = _df_to_records(t.balance_sheet)
            quarterly = _df_to_records(t.quarterly_balance_sheet)
        elif req.statement == "cashflow":
            annual = _df_to_records(t.cashflow)
            quarterly = _df_to_records(t.quarterly_cashflow)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown statement type: {req.statement}. Use: income, balance, cashflow")

        result = {
            "ticker": req.ticker.upper(),
            "statement": req.statement,
            "annual": annual,
            "quarterly": quarterly,
        }

        _set_cached(cache_key, result)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        log.error("financials error for %s: %s", req.ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/news")
async def news(req: TickerRequest):
    try:
        cache_key = f"news:{req.ticker}"
        cached = _get_cached(cache_key)
        if cached:
            return JSONResponse(cached)

        t = yf.Ticker(req.ticker)
        articles = t.news or []

        # Extract key fields
        clean_articles = []
        for a in articles[:20]:
            content = a.get("content", {}) if isinstance(a.get("content"), dict) else {}
            clean_articles.append({
                "title": content.get("title") or a.get("title", ""),
                "publisher": content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else a.get("publisher", ""),
                "link": content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else a.get("link", ""),
                "published": content.get("pubDate") or a.get("providerPublishTime", ""),
                "summary": content.get("summary", ""),
            })

        result = {
            "ticker": req.ticker.upper(),
            "count": len(clean_articles),
            "articles": clean_articles,
        }

        _set_cached(cache_key, result)
        return JSONResponse(result)
    except Exception as e:
        log.error("news error for %s: %s", req.ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyst")
async def analyst(req: TickerRequest):
    try:
        cache_key = f"analyst:{req.ticker}"
        cached = _get_cached(cache_key)
        if cached:
            return JSONResponse(cached)

        t = yf.Ticker(req.ticker)

        recs = _df_to_records(t.recommendations)
        upgrades = _df_to_records(t.upgrades_downgrades)

        result = {
            "ticker": req.ticker.upper(),
            "recommendations": recs[:20] if recs else [],
            "upgrades_downgrades": upgrades[:20] if upgrades else [],
        }

        _set_cached(cache_key, result)
        return JSONResponse(result)
    except Exception as e:
        log.error("analyst error for %s: %s", req.ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download")
async def download(req: DownloadRequest):
    try:
        tickers = [s.strip() for s in req.tickers.split(",") if s.strip()]
        if not tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        if len(tickers) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 tickers per request")

        cache_key = f"download:{','.join(tickers)}:{req.period}:{req.interval}"
        cached = _get_cached(cache_key)
        if cached:
            return JSONResponse(cached)

        df = yf.download(tickers, period=req.period, interval=req.interval, group_by="ticker", threads=True)

        results = {}
        if len(tickers) == 1:
            results[tickers[0].upper()] = _df_to_records(df)
        else:
            for ticker in tickers:
                try:
                    ticker_df = df[ticker.upper()] if ticker.upper() in df.columns.get_level_values(0) else None
                    if ticker_df is not None:
                        results[ticker.upper()] = _df_to_records(ticker_df)
                    else:
                        results[ticker.upper()] = []
                except Exception:
                    results[ticker.upper()] = []

        result = {
            "tickers": [t.upper() for t in tickers],
            "period": req.period,
            "interval": req.interval,
            "data": results,
        }

        _set_cached(cache_key, result)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        log.error("download error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8107)
