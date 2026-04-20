import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query


# Load both workspace-level and backend-local env files if present.
_THIS_FILE = Path(__file__).resolve()
_BACKEND_DIR = _THIS_FILE.parents[1]
_WORKSPACE_DIR = _THIS_FILE.parents[2]
load_dotenv(_WORKSPACE_DIR / ".env")
load_dotenv(_BACKEND_DIR / ".env")
load_dotenv()


ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "finance-module/1.0 (contact: dev@example.com)")
DEFAULT_SYMBOLS = os.getenv("FINANCE_NEWS_DEFAULT_SYMBOLS", "AAPL,MSFT")
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "20"))
TRADESTIE_TIMEOUT_SEC = max(4, min(REQUEST_TIMEOUT_SEC, 8))


CURATED_COMPANIES: List[Dict[str, Any]] = [
    {"symbol": "AAPL", "name": "Apple Inc.", "cik": 320193},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "cik": 789019},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "cik": 1652044},
    {"symbol": "AMZN", "name": "Amazon.com, Inc.", "cik": 1018724},
    {"symbol": "META", "name": "Meta Platforms, Inc.", "cik": 1326801},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "cik": 1045810},
    {"symbol": "TSLA", "name": "Tesla, Inc.", "cik": 1318605},
    {"symbol": "BRK.B", "name": "Berkshire Hathaway Inc.", "cik": 1067983},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "cik": 19617},
    {"symbol": "V", "name": "Visa Inc.", "cik": 1403161},
    {"symbol": "MA", "name": "Mastercard Incorporated", "cik": 1141391},
    {"symbol": "UNH", "name": "UnitedHealth Group Incorporated", "cik": 731766},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "cik": 200406},
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "cik": 34088},
    {"symbol": "PG", "name": "The Procter & Gamble Company", "cik": 80424},
    {"symbol": "HD", "name": "The Home Depot, Inc.", "cik": 354950},
    {"symbol": "KO", "name": "The Coca-Cola Company", "cik": 21344},
    {"symbol": "PEP", "name": "PepsiCo, Inc.", "cik": 77476},
    {"symbol": "PFE", "name": "Pfizer Inc.", "cik": 78003},
    {"symbol": "WMT", "name": "Walmart Inc.", "cik": 104169},
]


class TTLCache:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if not item:
            return None
        if item["expires_at"] <= time.time():
            self._data.pop(key, None)
            return None
        return item["value"]

    def set(self, key: str, value: Any, ttl_sec: int) -> None:
        self._data[key] = {"value": value, "expires_at": time.time() + ttl_sec}


cache = TTLCache()
finance_router = APIRouter(prefix="/ai/finance", tags=["Finance Intelligence"])


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _iso_from_epoch(epoch_value: Any) -> Optional[str]:
    ts = _to_int(epoch_value, 0)
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _iso_from_alpha_time(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    # Alpha format: 20250301T130015
    try:
        dt = datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _split_price_range(value: Optional[str]) -> Dict[str, Optional[float]]:
    if not value:
        return {"low": None, "high": None}

    parts = [part.strip() for part in str(value).split("-") if part.strip()]
    if len(parts) != 2:
        return {"low": None, "high": None}

    low = _to_float(parts[0], 0.0)
    high = _to_float(parts[1], 0.0)
    return {
        "low": low if low > 0 else None,
        "high": high if high > 0 else None,
    }


def _news_timestamp(item: Dict[str, Any]) -> float:
    published = (item.get("published_at") or "").strip()
    if not published:
        return 0.0

    try:
        normalized = published.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        pass

    try:
        return datetime.strptime(published, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _http_json_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 1,
) -> Any:
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
            if response.status_code >= 500 and attempt < retries:
                time.sleep(0.2 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Upstream request failed ({response.status_code}): {url}")
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.2 * (attempt + 1))
                continue
            break

    raise HTTPException(status_code=504, detail=f"Upstream timeout/unreachable: {url} ({last_error})")


def _symbols_csv(raw: str) -> str:
    cleaned = [_normalize_symbol(s) for s in raw.split(",") if s.strip()]
    return ",".join(cleaned) if cleaned else DEFAULT_SYMBOLS


def _sec_company_records() -> List[Dict[str, Any]]:
    ck = "sec_company_records"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": SEC_USER_AGENT},
        retries=1,
    )

    rows: List[Dict[str, Any]] = []
    for _, row in payload.items():
        symbol = _normalize_symbol(str(row.get("ticker", "")))
        name = str(row.get("title", "")).strip()
        cik = _to_int(row.get("cik_str"))
        if symbol and name:
            rows.append({"symbol": symbol, "name": name, "cik": cik, "source": "sec"})

    rows.sort(key=lambda x: x["name"])
    cache.set(ck, rows, ttl_sec=86400)
    return rows


def _company_catalog(query: str, limit: int) -> List[Dict[str, Any]]:
    q = query.strip().lower()
    seen = set()
    out: List[Dict[str, Any]] = []

    def maybe_add(item: Dict[str, Any], source: str) -> None:
        symbol = _normalize_symbol(str(item.get("symbol", "")))
        name = str(item.get("name", "")).strip()
        if not symbol or not name:
            return

        if q and q not in symbol.lower() and q not in name.lower():
            return

        if symbol in seen:
            return

        seen.add(symbol)
        out.append(
            {
                "symbol": symbol,
                "name": name,
                "cik": _to_int(item.get("cik")),
                "source": source,
            }
        )

    for row in CURATED_COMPANIES:
        maybe_add(row, "curated")

    try:
        for row in _sec_company_records():
            maybe_add(row, "sec")
    except Exception:
        # Keep curated results available even if SEC fetch fails.
        pass

    out.sort(key=lambda x: x["name"])
    return out[:limit]


def _alpha_company_overview(symbol: str) -> Dict[str, Any]:
    if not ALPHA_VANTAGE_API_KEY:
        return {}

    ck = f"alpha_overview:{symbol}"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://www.alphavantage.co/query",
        params={"function": "OVERVIEW", "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY},
        retries=0,
    )

    if payload.get("Note"):
        raise HTTPException(status_code=429, detail=payload["Note"])

    if not payload.get("Symbol"):
        return {}

    result = {
        "symbol": symbol,
        "name": payload.get("Name"),
        "description": payload.get("Description"),
        "sector": payload.get("Sector"),
        "industry": payload.get("Industry"),
        "employees": _to_int(payload.get("FullTimeEmployees"), 0),
        "market_cap": _to_float(payload.get("MarketCapitalization")),
        "pe_ratio": _to_float(payload.get("PERatio")),
        "eps": _to_float(payload.get("EPS")),
        "week_52_high": _to_float(payload.get("52WeekHigh")),
        "week_52_low": _to_float(payload.get("52WeekLow")),
        "dividend_yield": _to_float(payload.get("DividendYield")),
        "country": payload.get("Country"),
        "address": payload.get("Address"),
        "exchange": payload.get("Exchange"),
        "currency": payload.get("Currency"),
        "website": payload.get("OfficialSite"),
        "provider": "alpha_vantage",
    }
    cache.set(ck, result, ttl_sec=21600)
    return result


def _finnhub_profile(symbol: str) -> Dict[str, Any]:
    if not FINNHUB_API_KEY:
        return {}

    ck = f"finnhub_profile:{symbol}"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://finnhub.io/api/v1/stock/profile2",
        params={"symbol": symbol, "token": FINNHUB_API_KEY},
        retries=0,
    )

    if not payload or not payload.get("ticker"):
        return {}

    market_cap_m = _to_float(payload.get("marketCapitalization"), 0.0)
    result = {
        "symbol": symbol,
        "name": payload.get("name"),
        "industry": payload.get("finnhubIndustry"),
        "country": payload.get("country"),
        "currency": payload.get("currency"),
        "exchange": payload.get("exchange"),
        "ipo": payload.get("ipo"),
        "website": payload.get("weburl"),
        "logo_url": payload.get("logo"),
        "market_cap": market_cap_m * 1_000_000 if market_cap_m else 0.0,
        "provider": "finnhub",
    }
    cache.set(ck, result, ttl_sec=21600)
    return result


def _fmp_profile(symbol: str) -> Dict[str, Any]:
    if not FMP_API_KEY:
        return {}

    ck = f"fmp_profile:{symbol}"
    cached = cache.get(ck)
    if cached:
        return cached

    # Use non-legacy stable endpoint. Legacy /api/v3/profile is commonly blocked on newer plans.
    payload = _http_json_get(
        "https://financialmodelingprep.com/stable/profile",
        params={"symbol": symbol, "apikey": FMP_API_KEY},
        retries=0,
    )

    if not isinstance(payload, list) or not payload:
        return {}

    row = payload[0]
    range_bounds = _split_price_range(row.get("range"))
    employees_value = _to_int(row.get("fullTimeEmployees"), 0)
    market_cap_value = _to_float(row.get("marketCap"), 0.0)

    result = {
        "symbol": symbol,
        "name": row.get("companyName"),
        "description": row.get("description"),
        "sector": row.get("sector"),
        "industry": row.get("industry"),
        "employees": employees_value if employees_value > 0 else None,
        "market_cap": market_cap_value if market_cap_value > 0 else None,
        "ceo": row.get("ceo"),
        "country": row.get("country"),
        "exchange": row.get("exchange"),
        "currency": row.get("currency"),
        "website": row.get("website"),
        "logo_url": row.get("image") or row.get("logo"),
        "ipo": row.get("ipoDate"),
        "week_52_low": range_bounds["low"],
        "week_52_high": range_bounds["high"],
        "dividend_yield": _to_float(row.get("lastDividend")),
        "provider": "fmp",
    }
    cache.set(ck, result, ttl_sec=21600)
    return result


def _merge_company_profiles(symbol: str) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "symbol": symbol,
        "name": symbol,
        "description": None,
        "sector": None,
        "industry": None,
        "employees": None,
        "market_cap": None,
        "pe_ratio": None,
        "eps": None,
        "week_52_high": None,
        "week_52_low": None,
        "dividend_yield": None,
        "country": None,
        "address": None,
        "exchange": None,
        "currency": None,
        "website": None,
        "logo_url": None,
        "ceo": None,
        "ipo": None,
    }

    sources: List[str] = []
    errors: List[str] = []

    providers = [
        ("alpha_vantage", _alpha_company_overview),
        ("finnhub", _finnhub_profile),
        ("fmp", _fmp_profile),
    ]

    for provider_name, fetcher in providers:
        try:
            data = fetcher(symbol)
        except Exception as exc:
            errors.append(f"{provider_name}: {exc}")
            continue

        if not data:
            continue

        sources.append(provider_name)
        for key, value in data.items():
            if key in profile and _is_present(value):
                if not _is_present(profile.get(key)):
                    profile[key] = value

    if not _is_present(profile.get("name")):
        profile["name"] = symbol

    if not _is_present(profile.get("employees")):
        profile["employees"] = None

    return {"profile": profile, "sources": sources, "errors": errors}


def _alpha_quote(symbol: str) -> Dict[str, Any]:
    if not ALPHA_VANTAGE_API_KEY:
        raise HTTPException(status_code=503, detail="ALPHA_VANTAGE_API_KEY missing")

    ck = f"alpha_quote:{symbol}"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY},
        retries=0,
    )

    if payload.get("Note"):
        raise HTTPException(status_code=429, detail=payload["Note"])

    q = payload.get("Global Quote", {})
    if not q:
        raise HTTPException(status_code=404, detail=f"No quote data for {symbol}")

    result = {
        "symbol": q.get("01. symbol", symbol),
        "price": _to_float(q.get("05. price")),
        "open": _to_float(q.get("02. open")),
        "high": _to_float(q.get("03. high")),
        "low": _to_float(q.get("04. low")),
        "previous_close": _to_float(q.get("08. previous close")),
        "change": _to_float(q.get("09. change")),
        "change_percent": q.get("10. change percent", "0%"),
        "volume": _to_int(q.get("06. volume")),
        "latest_trading_day": q.get("07. latest trading day"),
        "provider": "alpha_vantage",
    }
    cache.set(ck, result, ttl_sec=45)
    return result


def _finnhub_quote(symbol: str) -> Dict[str, Any]:
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="FINNHUB_API_KEY missing")

    ck = f"finnhub_quote:{symbol}"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol, "token": FINNHUB_API_KEY},
        retries=0,
    )

    current_price = _to_float(payload.get("c"), 0.0)
    if current_price <= 0:
        raise HTTPException(status_code=404, detail=f"No finnhub quote for {symbol}")

    result = {
        "symbol": symbol,
        "price": current_price,
        "open": _to_float(payload.get("o")),
        "high": _to_float(payload.get("h")),
        "low": _to_float(payload.get("l")),
        "previous_close": _to_float(payload.get("pc")),
        "change": _to_float(payload.get("d")),
        "change_percent": f"{_to_float(payload.get('dp')):.2f}%",
        "volume": 0,
        "latest_trading_day": (_iso_from_epoch(payload.get("t")) or "")[:10],
        "provider": "finnhub",
    }
    cache.set(ck, result, ttl_sec=45)
    return result


def _fmp_stable_quote(symbol: str) -> Dict[str, Any]:
    if not FMP_API_KEY:
        raise HTTPException(status_code=503, detail="FMP_API_KEY missing")

    ck = f"fmp_quote:{symbol}"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://financialmodelingprep.com/stable/quote",
        params={"symbol": symbol, "apikey": FMP_API_KEY},
        retries=0,
    )

    if not isinstance(payload, list) or not payload:
        raise HTTPException(status_code=404, detail=f"No FMP quote for {symbol}")

    row = payload[0]
    current_price = _to_float(row.get("price"), 0.0)
    if current_price <= 0:
        raise HTTPException(status_code=404, detail=f"No FMP quote for {symbol}")

    change_percent = _to_float(row.get("changePercentage"), 0.0)

    result = {
        "symbol": _normalize_symbol(str(row.get("symbol", symbol))),
        "price": current_price,
        "open": _to_float(row.get("open")),
        "high": _to_float(row.get("dayHigh")),
        "low": _to_float(row.get("dayLow")),
        "previous_close": _to_float(row.get("previousClose")),
        "change": _to_float(row.get("change")),
        "change_percent": f"{change_percent:.2f}%",
        "volume": _to_int(row.get("volume")),
        "latest_trading_day": (_iso_from_epoch(row.get("timestamp")) or "")[:10],
        "provider": "fmp",
    }
    cache.set(ck, result, ttl_sec=45)
    return result


def _get_quote(symbol: str) -> Dict[str, Any]:
    errors: List[str] = []

    for provider in (_alpha_quote, _finnhub_quote, _fmp_stable_quote):
        try:
            return provider(symbol)
        except Exception as exc:
            errors.append(str(exc))

    raise HTTPException(status_code=503, detail=f"Quote unavailable for {symbol}: {' | '.join(errors)}")


def _alpha_candles(symbol: str, points: int = 30) -> Dict[str, Any]:
    if not ALPHA_VANTAGE_API_KEY:
        raise HTTPException(status_code=503, detail="ALPHA_VANTAGE_API_KEY missing")

    ck = f"alpha_candles:{symbol}:{points}"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://www.alphavantage.co/query",
        params={
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": ALPHA_VANTAGE_API_KEY,
        },
        retries=0,
    )

    if payload.get("Note"):
        raise HTTPException(status_code=429, detail=payload["Note"])

    series = payload.get("Time Series (Daily)", {})
    if not series:
        raise HTTPException(status_code=404, detail=f"No candle data for {symbol}")

    dates = sorted(series.keys(), reverse=True)[:points]
    rows: List[Dict[str, Any]] = []
    for day in dates:
        row = series[day]
        rows.append(
            {
                "date": day,
                "open": _to_float(row.get("1. open")),
                "high": _to_float(row.get("2. high")),
                "low": _to_float(row.get("3. low")),
                "close": _to_float(row.get("4. close")),
                "volume": _to_int(row.get("6. volume")),
            }
        )

    rows.reverse()
    result = {"symbol": symbol, "provider": "alpha_vantage", "candles": rows}
    cache.set(ck, result, ttl_sec=300)
    return result


def _finnhub_candles(symbol: str, points: int = 30) -> Dict[str, Any]:
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="FINNHUB_API_KEY missing")

    ck = f"finnhub_candles:{symbol}:{points}"
    cached = cache.get(ck)
    if cached:
        return cached

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=max(150, points * 3))

    payload = _http_json_get(
        "https://finnhub.io/api/v1/stock/candle",
        params={
            "symbol": symbol,
            "resolution": "D",
            "from": int(start.timestamp()),
            "to": int(now.timestamp()),
            "token": FINNHUB_API_KEY,
        },
        retries=0,
    )

    if payload.get("s") != "ok":
        raise HTTPException(status_code=404, detail=f"No finnhub candle data for {symbol}")

    candles: List[Dict[str, Any]] = []
    times = payload.get("t", [])
    opens = payload.get("o", [])
    highs = payload.get("h", [])
    lows = payload.get("l", [])
    closes = payload.get("c", [])
    volumes = payload.get("v", [])

    max_n = min(len(times), len(opens), len(highs), len(lows), len(closes), len(volumes))
    for i in range(max_n):
        date_iso = _iso_from_epoch(times[i])
        candles.append(
            {
                "date": (date_iso or "")[:10],
                "open": _to_float(opens[i]),
                "high": _to_float(highs[i]),
                "low": _to_float(lows[i]),
                "close": _to_float(closes[i]),
                "volume": _to_int(volumes[i]),
            }
        )

    candles = candles[-points:]
    result = {"symbol": symbol, "provider": "finnhub", "candles": candles}
    cache.set(ck, result, ttl_sec=300)
    return result


def _fmp_stable_candles(symbol: str, points: int = 30) -> Dict[str, Any]:
    if not FMP_API_KEY:
        raise HTTPException(status_code=503, detail="FMP_API_KEY missing")

    ck = f"fmp_candles:{symbol}:{points}"
    cached = cache.get(ck)
    if cached:
        return cached

    today = datetime.now(timezone.utc).date()
    from_date = today - timedelta(days=max(points * 4, 365))

    payload = _http_json_get(
        "https://financialmodelingprep.com/stable/historical-price-eod/full",
        params={
            "symbol": symbol,
            "from": from_date.isoformat(),
            "to": today.isoformat(),
            "apikey": FMP_API_KEY,
        },
        retries=0,
    )

    rows_source: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        rows_source = payload
    elif isinstance(payload, dict):
        rows_source = payload.get("historical") or []

    if not rows_source:
        raise HTTPException(status_code=404, detail=f"No FMP candle data for {symbol}")

    candles: List[Dict[str, Any]] = []
    for row in rows_source:
        date_value = str(row.get("date", "")).strip()[:10]
        close_value = _to_float(row.get("close"), 0.0)
        if not date_value or close_value <= 0:
            continue

        candles.append(
            {
                "date": date_value,
                "open": _to_float(row.get("open")),
                "high": _to_float(row.get("high")),
                "low": _to_float(row.get("low")),
                "close": close_value,
                "volume": _to_int(row.get("volume")),
            }
        )

    if not candles:
        raise HTTPException(status_code=404, detail=f"No FMP candle data for {symbol}")

    candles.sort(key=lambda row: row["date"])
    candles = candles[-points:]

    result = {"symbol": symbol, "provider": "fmp", "candles": candles}
    cache.set(ck, result, ttl_sec=300)
    return result


def _get_candles(symbol: str, points: int = 30) -> Dict[str, Any]:
    errors: List[str] = []

    for provider in (_alpha_candles, _finnhub_candles, _fmp_stable_candles):
        try:
            return provider(symbol, points)
        except Exception as exc:
            errors.append(str(exc))

    raise HTTPException(status_code=503, detail=f"Candles unavailable for {symbol}: {' | '.join(errors)}")


def _marketaux_news(symbols_csv: str, limit: int) -> Dict[str, Any]:
    if not MARKETAUX_API_KEY:
        raise HTTPException(status_code=503, detail="MARKETAUX_API_KEY missing")

    ck = f"marketaux_news:{symbols_csv}:{limit}"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://api.marketaux.com/v1/news/all",
        params={
            "api_token": MARKETAUX_API_KEY,
            "symbols": symbols_csv,
            "language": "en",
            "filter_entities": "true",
            "limit": min(max(limit, 1), 50),
        },
        retries=0,
    )

    items = []
    for row in payload.get("data", []):
        entities = row.get("entities") or []
        tickers = []
        for entity in entities:
            ticker = _normalize_symbol(str(entity.get("symbol", "")))
            if ticker:
                tickers.append(ticker)

        sentiment_score = entities[0].get("sentiment_score") if entities else None
        items.append(
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "source": row.get("source"),
                "published_at": row.get("published_at"),
                "tickers": tickers,
                "sentiment_score": sentiment_score,
                "summary": row.get("description"),
                "provider": "marketaux",
            }
        )

    result = {"provider": "marketaux", "items": items}
    cache.set(ck, result, ttl_sec=90)
    return result


def _alpha_news(symbols_csv: str, limit: int) -> Dict[str, Any]:
    if not ALPHA_VANTAGE_API_KEY:
        raise HTTPException(status_code=503, detail="ALPHA_VANTAGE_API_KEY missing")

    ck = f"alpha_news:{symbols_csv}:{limit}"
    cached = cache.get(ck)
    if cached:
        return cached

    payload = _http_json_get(
        "https://www.alphavantage.co/query",
        params={
            "function": "NEWS_SENTIMENT",
            "tickers": symbols_csv,
            "limit": min(max(limit, 1), 50),
            "apikey": ALPHA_VANTAGE_API_KEY,
        },
        retries=0,
    )

    if payload.get("Note"):
        raise HTTPException(status_code=429, detail=payload["Note"])

    items = []
    for row in payload.get("feed", []):
        ticker_sentiment = row.get("ticker_sentiment") or []
        tickers = []
        for item in ticker_sentiment:
            ticker = _normalize_symbol(str(item.get("ticker", "")))
            if ticker:
                tickers.append(ticker)

        published = _iso_from_alpha_time(row.get("time_published"))
        items.append(
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "source": row.get("source"),
                "published_at": published,
                "tickers": tickers,
                "sentiment_score": _to_float(row.get("overall_sentiment_score")),
                "summary": row.get("summary"),
                "provider": "alpha_vantage_news",
            }
        )

    result = {"provider": "alpha_vantage_news", "items": items}
    cache.set(ck, result, ttl_sec=90)
    return result


def _finnhub_news(symbol: str, limit: int) -> Dict[str, Any]:
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="FINNHUB_API_KEY missing")

    ck = f"finnhub_news:{symbol}:{limit}"
    cached = cache.get(ck)
    if cached:
        return cached

    today = datetime.now(timezone.utc).date()
    from_date = today - timedelta(days=10)

    payload = _http_json_get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": symbol,
            "from": from_date.isoformat(),
            "to": today.isoformat(),
            "token": FINNHUB_API_KEY,
        },
        retries=0,
    )

    items = []
    if isinstance(payload, list):
        for row in payload[: min(max(limit, 1), 50)]:
            items.append(
                {
                    "title": row.get("headline"),
                    "url": row.get("url"),
                    "source": row.get("source"),
                    "published_at": _iso_from_epoch(row.get("datetime")),
                    "tickers": [symbol],
                    "sentiment_score": None,
                    "summary": row.get("summary"),
                    "provider": "finnhub",
                }
            )

    result = {"provider": "finnhub", "items": items}
    cache.set(ck, result, ttl_sec=90)
    return result


def _sec_ticker_map() -> Dict[str, int]:
    ck = "sec_ticker_map"
    cached = cache.get(ck)
    if cached:
        return cached

    records = _sec_company_records()
    mapping = {row["symbol"]: _to_int(row.get("cik")) for row in records if row.get("symbol")}
    cache.set(ck, mapping, ttl_sec=86400)
    return mapping


def _sec_filings(symbol: str, limit: int) -> Dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    ck = f"sec_filings:{symbol}:{limit}"
    cached = cache.get(ck)
    if cached:
        return cached

    ticker_map = _sec_ticker_map()
    cik = ticker_map.get(symbol)
    if not cik:
        return {"provider": "sec", "items": []}

    cik10 = str(cik).zfill(10)
    payload = _http_json_get(
        f"https://data.sec.gov/submissions/CIK{cik10}.json",
        headers={"User-Agent": SEC_USER_AGENT},
        retries=0,
    )

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    max_n = min(limit, len(forms), len(dates), len(accessions), len(docs))
    items = []
    for i in range(max_n):
        acc_no_dash = str(accessions[i]).replace("-", "")
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/{docs[i]}"
        items.append(
            {
                "title": f"{symbol} filed {forms[i]}",
                "url": filing_url,
                "source": "SEC EDGAR",
                "published_at": f"{dates[i]}T00:00:00+00:00",
                "tickers": [symbol],
                "sentiment_score": None,
                "summary": None,
                "provider": "sec",
            }
        )

    result = {"provider": "sec", "items": items}
    cache.set(ck, result, ttl_sec=900)
    return result


def _tradestie_sentiment(limit: int) -> Dict[str, Any]:
    ck = f"tradestie_sentiment:{limit}"
    cached = cache.get(ck)
    if cached:
        return cached

    try:
        response = requests.get("https://api.tradestie.com/v1/apps/reddit", timeout=TRADESTIE_TIMEOUT_SEC)
    except requests.Timeout as exc:
        raise HTTPException(status_code=504, detail="Tradestie timed out") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Tradestie unreachable ({exc.__class__.__name__})") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Tradestie sentiment request failed ({response.status_code})")

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Tradestie response was not valid JSON") from exc

    if not isinstance(payload, list):
        return {"provider": "tradestie", "items": []}

    items = []
    for row in payload[: min(max(limit, 1), 50)]:
        ticker = _normalize_symbol(str(row.get("ticker", "")))
        sentiment = row.get("sentiment")
        score = _to_float(row.get("sentiment_score"))
        comments = _to_int(row.get("no_of_comments"))
        items.append(
            {
                "title": f"WSB sentiment {ticker}: {sentiment} ({comments} comments)",
                "url": "https://tradestie.com/apps/reddit/api/",
                "source": "Tradestie / WallStreetBets",
                "published_at": datetime.now(timezone.utc).isoformat(),
                "tickers": [ticker] if ticker else [],
                "sentiment_score": score,
                "summary": "Retail crowd sentiment pulse from social activity",
                "provider": "tradestie",
            }
        )

    result = {"provider": "tradestie", "items": items}
    cache.set(ck, result, ttl_sec=90)
    return result


def _dedupe_news(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for item in items:
        key = (item.get("url") or item.get("title") or "").strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


def _format_provider_error(provider_name: str, exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = str(exc.detail)
    else:
        detail = str(exc)

    lowered = detail.lower()
    if "timed out" in lowered or "timeout" in lowered:
        detail = "timed out"
    elif "unreachable" in lowered or "name resolution" in lowered or "failed to establish a new connection" in lowered:
        detail = "unreachable"

    if len(detail) > 180:
        detail = f"{detail[:177]}..."

    return f"{provider_name}: {detail}"


def _build_news(symbols_csv: str, limit: int) -> Dict[str, Any]:
    symbols = [_normalize_symbol(s) for s in symbols_csv.split(",") if s.strip()]
    first_symbol = symbols[0] if symbols else "AAPL"

    providers_used: List[str] = []
    provider_errors: List[str] = []
    merged_items: List[Dict[str, Any]] = []

    provider_calls = [
        ("marketaux", lambda: _marketaux_news(symbols_csv, limit)),
        ("alpha_vantage_news", lambda: _alpha_news(symbols_csv, limit)),
        ("finnhub", lambda: _finnhub_news(first_symbol, limit)),
        ("sec", lambda: _sec_filings(first_symbol, max(4, min(limit, 12)))),
        ("tradestie", lambda: _tradestie_sentiment(max(6, min(limit, 15)))),
    ]

    for provider_name, call in provider_calls:
        try:
            data = call()
            items = data.get("items", [])
            if items:
                merged_items.extend(items)
                providers_used.append(provider_name)
        except Exception as exc:
            provider_errors.append(_format_provider_error(provider_name, exc))

    deduped = _dedupe_news(merged_items)
    deduped.sort(key=_news_timestamp, reverse=True)

    return {
        "provider": "+".join(providers_used) if providers_used else "none",
        "providers": providers_used,
        "provider_errors": provider_errors,
        "items": deduped[:limit],
    }


def _build_company_intel(symbol: str, points: int, news_limit: int) -> Dict[str, Any]:
    normalized_symbol = _normalize_symbol(symbol)

    quote_data = None
    quote_error = None
    candles_data: List[Dict[str, Any]] = []
    candles_provider: Optional[str] = None
    candles_error = None

    profile_bundle = _merge_company_profiles(normalized_symbol)

    try:
        quote_data = _get_quote(normalized_symbol)
    except HTTPException as exc:
        quote_error = exc.detail

    try:
        candles_bundle = _get_candles(normalized_symbol, points=points)
        candles_data = candles_bundle.get("candles", [])
        candles_provider = candles_bundle.get("provider")
    except HTTPException as exc:
        candles_error = exc.detail

    news_data = _build_news(normalized_symbol, news_limit)

    return {
        "symbol": normalized_symbol,
        "company": profile_bundle.get("profile"),
        "company_sources": profile_bundle.get("sources", []),
        "company_provider_errors": profile_bundle.get("errors", []),
        "quote": quote_data,
        "quote_error": quote_error,
        "candles": candles_data,
        "candles_provider": candles_provider,
        "candles_error": candles_error,
        "news": news_data.get("items", []),
        "news_provider": news_data.get("provider"),
        "news_providers": news_data.get("providers", []),
        "news_provider_errors": news_data.get("provider_errors", []),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@finance_router.get("/providers")
def providers() -> Dict[str, Any]:
    return {
        "alpha_vantage_configured": bool(ALPHA_VANTAGE_API_KEY),
        "marketaux_configured": bool(MARKETAUX_API_KEY),
        "finnhub_configured": bool(FINNHUB_API_KEY),
        "fmp_configured": bool(FMP_API_KEY),
        "sec_enabled": True,
        "tradestie_enabled": True,
    }


@finance_router.get("/companies")
def companies(query: str = Query("", max_length=100), limit: int = Query(600, ge=10, le=3000)) -> Dict[str, Any]:
    items = _company_catalog(query=query, limit=limit)
    return {"count": len(items), "items": items}


@finance_router.get("/company-profile")
def company_profile(symbol: str = Query(..., min_length=1, max_length=16)) -> Dict[str, Any]:
    normalized = _normalize_symbol(symbol)
    bundle = _merge_company_profiles(normalized)
    return {
        "symbol": normalized,
        "company": bundle.get("profile"),
        "company_sources": bundle.get("sources", []),
        "company_provider_errors": bundle.get("errors", []),
    }


@finance_router.get("/quote")
def quote(symbol: str = Query(..., min_length=1, max_length=16)) -> Dict[str, Any]:
    return _get_quote(_normalize_symbol(symbol))


@finance_router.get("/candles")
def candles(symbol: str = Query(..., min_length=1, max_length=16), points: int = Query(30, ge=10, le=200)) -> Dict[str, Any]:
    return _get_candles(_normalize_symbol(symbol), points)


@finance_router.get("/news")
def news(symbols: str = Query(DEFAULT_SYMBOLS), limit: int = Query(15, ge=1, le=50)) -> Dict[str, Any]:
    symbols_csv = _symbols_csv(symbols)
    return _build_news(symbols_csv, limit)


@finance_router.get("/company-intel")
def company_intel(
    symbol: str = Query("AAPL", min_length=1, max_length=16),
    points: int = Query(45, ge=10, le=200),
    news_limit: int = Query(15, ge=1, le=50),
) -> Dict[str, Any]:
    return _build_company_intel(symbol=symbol, points=points, news_limit=news_limit)


@finance_router.get("/overview")
def overview(
    symbol: str = Query("AAPL", min_length=1, max_length=16),
    points: int = Query(30, ge=10, le=200),
    news_limit: int = Query(12, ge=1, le=50),
) -> Dict[str, Any]:
    intel = _build_company_intel(symbol=symbol, points=points, news_limit=news_limit)
    return {
        "symbol": intel["symbol"],
        "company": intel.get("company"),
        "quote": intel.get("quote"),
        "quote_error": intel.get("quote_error"),
        "candles": intel.get("candles", []),
        "candles_error": intel.get("candles_error"),
        "news": intel.get("news", []),
        "news_provider": intel.get("news_provider"),
        "generated_at": intel.get("generated_at"),
    }
