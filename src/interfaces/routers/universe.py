"""Universe API."""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.repositories.models import StockUniverse
from src.shared.market import get_market, market_is_us
from src.shared.us_universe import get_us_universe_tickers

router = APIRouter()


class StockInfo(BaseModel):
    stock_id: str
    name: str
    market_cap: int
    rank: int


class UniverseResponse(BaseModel):
    name: str
    description: str
    total: int
    stocks: list[StockInfo]
    updated_at: datetime | None


class UniverseStats(BaseModel):
    total: int
    min_market_cap: int
    max_market_cap: int
    updated_at: datetime | None


def _load_stocks(session: Session):
    stmt = select(StockUniverse).order_by(StockUniverse.rank)
    result = session.execute(stmt)
    return result.scalars().all()


@router.get("", response_model=UniverseResponse)
async def get_universe(session: Session = Depends(get_db)):
    market = get_market()
    stocks = _load_stocks(session)
    updated_at = stocks[0].updated_at if stocks else None

    return UniverseResponse(
        name=market.universe_name,
        description=market.universe_description,
        total=len(stocks),
        stocks=[
            StockInfo(
                stock_id=s.stock_id,
                name=s.name,
                market_cap=s.market_cap,
                rank=s.rank,
            )
            for s in stocks
        ],
        updated_at=updated_at,
    )


@router.get("/stats", response_model=UniverseStats)
async def get_universe_stats(session: Session = Depends(get_db)):
    stocks = _load_stocks(session)

    if not stocks:
        return UniverseStats(total=0, min_market_cap=0, max_market_cap=0, updated_at=None)

    return UniverseStats(
        total=len(stocks),
        min_market_cap=min(s.market_cap for s in stocks),
        max_market_cap=max(s.market_cap for s in stocks),
        updated_at=stocks[0].updated_at,
    )


@router.get("/ids")
async def get_stock_ids(session: Session = Depends(get_db)):
    stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
    result = session.execute(stmt)
    stock_ids = [row[0] for row in result.fetchall()]
    return {"stock_ids": stock_ids, "total": len(stock_ids)}


async def _sync_us_universe(session: Session):
    import yfinance as yf

    tickers = get_us_universe_tickers()
    now = datetime.now()
    rows = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info or {}
            market_cap = int(info.get("market_cap") or 0)
            name = ticker
            rows.append({"stock_id": ticker, "name": name, "market_cap": market_cap})
        except Exception:
            rows.append({"stock_id": ticker, "name": ticker, "market_cap": 0})

    rows.sort(key=lambda x: x["market_cap"], reverse=True)

    session.execute(StockUniverse.__table__.delete())
    for rank, row in enumerate(rows, 1):
        session.add(
            StockUniverse(
                stock_id=row["stock_id"],
                name=row["name"],
                market_cap=row["market_cap"],
                rank=rank,
                updated_at=now,
            )
        )
    session.commit()

    return {"success": True, "total": len(rows), "updated_at": now.isoformat(), "market": "us"}


async def _sync_tw_universe(session: Session):
    import httpx

    price_url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL"
    share_url = "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS"

    async with httpx.AsyncClient() as client:
        price_resp = await client.get(price_url, timeout=30)
        share_resp = await client.get(share_url, params={"selectType": "ALLBUT0999"}, timeout=30)

    price_data = price_resp.json()
    share_data = share_resp.json()

    if price_data.get("stat") != "OK" or share_data.get("stat") != "OK":
        return {"success": False, "error": "Failed to fetch data from TWSE"}

    price_map = {}
    for row in price_data.get("data", []):
        if len(row) < 8:
            continue
        stock_id = row[0].strip()
        name = row[1].strip()
        try:
            close = float(row[7].replace(",", "")) if row[7] not in ("--", "-", "") else 0
            price_map[stock_id] = {"close": close, "name": name}
        except Exception:
            continue

    shares_map = {}
    for row in share_data.get("data", []):
        if len(row) < 4:
            continue
        stock_id = row[0].strip()
        try:
            issued_shares = int(row[3].replace(",", ""))
            shares_map[stock_id] = issued_shares
        except Exception:
            continue

    stocks = []
    for stock_id, price_info in price_map.items():
        name = price_info["name"]
        if not stock_id.isdigit() or len(stock_id) != 4:
            continue
        if stock_id.startswith("0"):
            continue
        if "-KY" in name or "KY" in name:
            continue
        if "*" in name or "-創" in name:
            continue
        if price_info["close"] <= 0:
            continue
        if stock_id not in shares_map:
            continue

        market_cap = price_info["close"] * shares_map[stock_id]
        stocks.append({"stock_id": stock_id, "name": name, "market_cap": round(market_cap / 1e8)})

    stocks.sort(key=lambda x: x["market_cap"], reverse=True)
    top100 = stocks[:100]

    session.execute(StockUniverse.__table__.delete())
    now = datetime.now()
    for rank, s in enumerate(top100, 1):
        session.add(
            StockUniverse(
                stock_id=s["stock_id"],
                name=s["name"],
                market_cap=s["market_cap"],
                rank=rank,
                updated_at=now,
            )
        )
    session.commit()

    return {"success": True, "total": len(top100), "updated_at": now.isoformat(), "market": "tw"}


@router.post("/sync")
async def sync_universe(session: Session = Depends(get_db)):
    if market_is_us():
        return await _sync_us_universe(session)
    return await _sync_tw_universe(session)
