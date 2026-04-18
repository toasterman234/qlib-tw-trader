"""US market sync API."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
import yfinance as yf

from src.interfaces.dependencies import get_db
from src.repositories.models import StockDaily, StockDailyAdj, StockUniverse, TradingCalendar
from src.services.sync_service import SyncService

router = APIRouter()


class SyncCalendarResponse(BaseModel):
    start_date: str
    end_date: str
    new_dates: int
    total_dates: int


class SyncStockResponse(BaseModel):
    stock_id: str
    fetched: int
    inserted: int
    missing_dates: list[str]


class SyncBulkResponse(BaseModel):
    date: str
    total: int
    inserted: int
    days_synced: int = 1
    error: str | None = None


class SyncAllResponse(BaseModel):
    stocks: int
    total_inserted: int
    errors: list[dict]


class DataStatusItem(BaseModel):
    stock_id: str
    name: str
    rank: int
    earliest_date: str | None
    latest_date: str | None
    total_records: int
    missing_count: int
    coverage_pct: float


class DataStatusResponse(BaseModel):
    trading_days: int
    start_date: str
    end_date: str
    stocks: list[DataStatusItem]


class MonthlyStatusItem(BaseModel):
    stock_id: str
    name: str
    rank: int
    earliest_month: str | None
    latest_month: str | None
    total_records: int
    missing_count: int
    coverage_pct: float


class MonthlyStatusResponse(BaseModel):
    expected_months: int
    start_year: int
    end_year: int
    stocks: list[MonthlyStatusItem]


class MonthlyStockResponse(BaseModel):
    stock_id: str
    fetched: int
    inserted: int
    missing_months: list[str]


def _to_decimal(value) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(round(float(value), 4)))
    except Exception:
        return None


def _get_universe(session: Session):
    return session.execute(select(StockUniverse).order_by(StockUniverse.rank)).scalars().all()


def _placeholder_error(name: str) -> str:
    return f"{name} sync is not implemented for US mode yet."


async def _sync_calendar_impl(start_date: date, end_date: date, session: Session):
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat(), auto_adjust=False)
    if df.empty:
        return 0, 0

    trading_dates = {idx.date() for idx in df.index}
    stmt = select(TradingCalendar.date).where(TradingCalendar.date >= start_date, TradingCalendar.date <= end_date)
    existing = {row[0] for row in session.execute(stmt).fetchall()}

    new_dates = trading_dates - existing
    for d in new_dates:
        session.add(TradingCalendar(date=d, is_trading_day=True))
    session.commit()
    return len(new_dates), len(trading_dates)


async def _sync_stock_daily_impl(stock_id: str, start_date: date, end_date: date, session: Session):
    trading_dates = set(SyncService(session).get_trading_dates(start_date, end_date))
    if not trading_dates:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    stmt = select(StockDaily.date).where(StockDaily.stock_id == stock_id, StockDaily.date >= start_date, StockDaily.date <= end_date)
    existing_dates = {row[0] for row in session.execute(stmt).fetchall()}
    missing_dates = sorted(trading_dates - existing_dates)
    if not missing_dates:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    ticker = yf.Ticker(stock_id)
    df = ticker.history(start=min(missing_dates).isoformat(), end=(max(missing_dates) + timedelta(days=1)).isoformat(), auto_adjust=False)
    if df.empty:
        return {"fetched": 0, "inserted": 0, "missing_dates": [d.isoformat() for d in missing_dates]}

    inserted = 0
    for idx, row in df.iterrows():
        r_date = idx.date()
        if r_date not in missing_dates or r_date in existing_dates:
            continue
        open_val = _to_decimal(row.get("Open"))
        high_val = _to_decimal(row.get("High"))
        low_val = _to_decimal(row.get("Low"))
        close_val = _to_decimal(row.get("Close"))
        if open_val is None or close_val is None:
            continue
        session.add(
            StockDaily(
                stock_id=stock_id,
                date=r_date,
                open=open_val,
                high=high_val or open_val,
                low=low_val or open_val,
                close=close_val,
                volume=int(row.get("Volume") or 0),
            )
        )
        inserted += 1
    session.commit()

    final_existing = {row[0] for row in session.execute(stmt).fetchall()}
    still_missing = sorted(trading_dates - final_existing)
    return {"fetched": len(df), "inserted": inserted, "missing_dates": [d.isoformat() for d in still_missing]}


async def _sync_adj_impl(stock_id: str, start_date: date, end_date: date, session: Session):
    trading_dates = set(SyncService(session).get_trading_dates(start_date, end_date))
    if not trading_dates:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    stmt = select(StockDailyAdj.date).where(StockDailyAdj.stock_id == stock_id, StockDailyAdj.date >= start_date, StockDailyAdj.date <= end_date)
    existing_dates = {row[0] for row in session.execute(stmt).fetchall()}
    missing_dates = sorted(trading_dates - existing_dates)
    if not missing_dates:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    ticker = yf.Ticker(stock_id)
    df = ticker.history(start=min(missing_dates).isoformat(), end=(max(missing_dates) + timedelta(days=1)).isoformat(), auto_adjust=True)
    if df.empty:
        return {"fetched": 0, "inserted": 0, "missing_dates": [d.isoformat() for d in missing_dates]}

    inserted = 0
    for idx, row in df.iterrows():
        r_date = idx.date()
        if r_date not in missing_dates or r_date in existing_dates:
            continue
        close_val = _to_decimal(row.get("Close"))
        if close_val is None:
            continue
        session.add(StockDailyAdj(stock_id=stock_id, date=r_date, adj_close=close_val))
        inserted += 1
    session.commit()

    final_existing = {row[0] for row in session.execute(stmt).fetchall()}
    still_missing = sorted(trading_dates - final_existing)
    return {"fetched": len(df), "inserted": inserted, "missing_dates": [d.isoformat() for d in still_missing]}


def _empty_daily_status(session: Session, start_date: date, end_date: date):
    universe = _get_universe(session)
    return {
        "trading_days": SyncService(session).count_trading_days(start_date, end_date),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "stocks": [
            {
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_date": None,
                "latest_date": None,
                "total_records": 0,
                "missing_count": 0,
                "coverage_pct": 0.0,
            }
            for stock in universe
        ],
    }


def _empty_monthly_status(session: Session, start_year: int, end_year: int):
    universe = _get_universe(session)
    return {
        "expected_months": 0,
        "start_year": start_year,
        "end_year": end_year,
        "stocks": [
            {
                "stock_id": stock.stock_id,
                "name": stock.name,
                "rank": stock.rank,
                "earliest_month": None,
                "latest_month": None,
                "total_records": 0,
                "missing_count": 0,
                "coverage_pct": 0.0,
            }
            for stock in universe
        ],
    }


@router.post("/calendar", response_model=SyncCalendarResponse)
async def sync_calendar(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    new_count, total = await _sync_calendar_impl(start_date, end_date, session)
    return SyncCalendarResponse(start_date=start_date.isoformat(), end_date=end_date.isoformat(), new_dates=new_count, total_dates=total)


@router.post("/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_stock(stock_id: str, start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = await _sync_stock_daily_impl(stock_id, start_date, end_date, session)
    return SyncStockResponse(stock_id=stock_id, fetched=result["fetched"], inserted=result["inserted"], missing_dates=result["missing_dates"])


@router.post("/bulk", response_model=SyncBulkResponse)
async def sync_bulk(target_date: date = Query(default=None), session: Session = Depends(get_db)):
    service = SyncService(session)
    if target_date is not None:
        universe = _get_universe(session)
        inserted = 0
        for stock in universe:
            res = await _sync_stock_daily_impl(stock.stock_id, target_date, target_date, session)
            inserted += res["inserted"]
        return SyncBulkResponse(date=target_date.isoformat(), total=len(universe), inserted=inserted)

    dates = service.get_recent_trading_dates(7)
    total = 0
    inserted = 0
    universe = _get_universe(session)
    for d in dates:
        for stock in universe:
            res = await _sync_stock_daily_impl(stock.stock_id, d, d, session)
            inserted += res["inserted"]
        total += len(universe)
    return SyncBulkResponse(date=f"{dates[0].isoformat()}~{dates[-1].isoformat()}" if dates else "", total=total, inserted=inserted, days_synced=len(dates))


@router.post("/all", response_model=SyncAllResponse)
async def sync_all(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    await _sync_calendar_impl(start_date, end_date, session)
    universe = _get_universe(session)
    total_inserted = 0
    errors = []
    for stock in universe:
        try:
            res = await _sync_stock_daily_impl(stock.stock_id, start_date, end_date, session)
            total_inserted += res["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock.stock_id, "error": str(e)})
    return SyncAllResponse(stocks=len(universe), total_inserted=total_inserted, errors=errors)


@router.get("/status", response_model=DataStatusResponse)
async def get_data_status(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = SyncService(session)._get_daily_status(StockDaily, start_date, end_date)
    return DataStatusResponse(trading_days=result["trading_days"], start_date=result["start_date"], end_date=result["end_date"], stocks=[DataStatusItem(**s) for s in result["stocks"]])


@router.get("/all-status")
async def get_all_sync_status(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), start_year: int = Query(default=2020), end_year: int = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    if end_year is None:
        end_year = date.today().year
    stock_daily = SyncService(session)._get_daily_status(StockDaily, start_date, end_date)
    adj = SyncService(session)._get_daily_status(StockDailyAdj, start_date, end_date)
    placeholder = _empty_daily_status(session, start_date, end_date)
    monthly = _empty_monthly_status(session, start_year, end_year)
    return {
        "stock_daily": stock_daily,
        "per": placeholder,
        "institutional": placeholder,
        "margin": placeholder,
        "adj": adj,
        "shareholding": placeholder,
        "securities_lending": placeholder,
        "monthly_revenue": monthly,
    }


@router.get("/adj/status", response_model=DataStatusResponse)
async def get_adj_status(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = SyncService(session)._get_daily_status(StockDailyAdj, start_date, end_date)
    return DataStatusResponse(trading_days=result["trading_days"], start_date=result["start_date"], end_date=result["end_date"], stocks=[DataStatusItem(**s) for s in result["stocks"]])


@router.post("/adj/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_adj_stock(stock_id: str, start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = await _sync_adj_impl(stock_id, start_date, end_date, session)
    return SyncStockResponse(stock_id=stock_id, fetched=result["fetched"], inserted=result["inserted"], missing_dates=result["missing_dates"])


@router.post("/adj/bulk", response_model=SyncBulkResponse)
async def sync_adj_bulk(target_date: date = Query(default=None), session: Session = Depends(get_db)):
    service = SyncService(session)
    universe = _get_universe(session)
    if target_date is not None:
        inserted = 0
        for stock in universe:
            res = await _sync_adj_impl(stock.stock_id, target_date, target_date, session)
            inserted += res["inserted"]
        return SyncBulkResponse(date=target_date.isoformat(), total=len(universe), inserted=inserted)

    dates = service.get_recent_trading_dates(7)
    total = 0
    inserted = 0
    for d in dates:
        for stock in universe:
            res = await _sync_adj_impl(stock.stock_id, d, d, session)
            inserted += res["inserted"]
        total += len(universe)
    return SyncBulkResponse(date=f"{dates[0].isoformat()}~{dates[-1].isoformat()}" if dates else "", total=total, inserted=inserted, days_synced=len(dates))


@router.post("/adj/all", response_model=SyncAllResponse)
async def sync_adj_all(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    await _sync_calendar_impl(start_date, end_date, session)
    universe = _get_universe(session)
    total_inserted = 0
    errors = []
    for stock in universe:
        try:
            res = await _sync_adj_impl(stock.stock_id, start_date, end_date, session)
            total_inserted += res["inserted"]
        except Exception as e:
            errors.append({"stock_id": stock.stock_id, "error": str(e)})
    return SyncAllResponse(stocks=len(universe), total_inserted=total_inserted, errors=errors)


@router.get("/per/status", response_model=DataStatusResponse)
async def get_per_status(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = _empty_daily_status(session, start_date, end_date)
    return DataStatusResponse(trading_days=result["trading_days"], start_date=result["start_date"], end_date=result["end_date"], stocks=[DataStatusItem(**s) for s in result["stocks"]])


@router.post("/per/bulk", response_model=SyncBulkResponse)
async def sync_per_bulk(target_date: date = Query(default=None), session: Session = Depends(get_db)):
    d = target_date or date.today()
    return SyncBulkResponse(date=d.isoformat(), total=0, inserted=0, error=_placeholder_error("PER/PBR/dividend yield"))


@router.post("/per/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_per_stock(stock_id: str, session: Session = Depends(get_db)):
    return SyncStockResponse(stock_id=stock_id, fetched=0, inserted=0, missing_dates=[])


@router.post("/per/all", response_model=SyncAllResponse)
async def sync_per_all(session: Session = Depends(get_db)):
    return SyncAllResponse(stocks=len(_get_universe(session)), total_inserted=0, errors=[{"stock_id": "*", "error": _placeholder_error("PER/PBR/dividend yield") }])


@router.get("/institutional/status", response_model=DataStatusResponse)
async def get_institutional_status(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = _empty_daily_status(session, start_date, end_date)
    return DataStatusResponse(trading_days=result["trading_days"], start_date=result["start_date"], end_date=result["end_date"], stocks=[DataStatusItem(**s) for s in result["stocks"]])


@router.post("/institutional/bulk", response_model=SyncBulkResponse)
async def sync_institutional_bulk(target_date: date = Query(default=None), session: Session = Depends(get_db)):
    d = target_date or date.today()
    return SyncBulkResponse(date=d.isoformat(), total=0, inserted=0, error=_placeholder_error("Institutional flow"))


@router.post("/institutional/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_institutional_stock(stock_id: str, session: Session = Depends(get_db)):
    return SyncStockResponse(stock_id=stock_id, fetched=0, inserted=0, missing_dates=[])


@router.post("/institutional/all", response_model=SyncAllResponse)
async def sync_institutional_all(session: Session = Depends(get_db)):
    return SyncAllResponse(stocks=len(_get_universe(session)), total_inserted=0, errors=[{"stock_id": "*", "error": _placeholder_error("Institutional flow") }])


@router.get("/margin/status", response_model=DataStatusResponse)
async def get_margin_status(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = _empty_daily_status(session, start_date, end_date)
    return DataStatusResponse(trading_days=result["trading_days"], start_date=result["start_date"], end_date=result["end_date"], stocks=[DataStatusItem(**s) for s in result["stocks"]])


@router.post("/margin/bulk", response_model=SyncBulkResponse)
async def sync_margin_bulk(target_date: date = Query(default=None), session: Session = Depends(get_db)):
    d = target_date or date.today()
    return SyncBulkResponse(date=d.isoformat(), total=0, inserted=0, error=_placeholder_error("Margin / short interest"))


@router.post("/margin/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_margin_stock(stock_id: str, session: Session = Depends(get_db)):
    return SyncStockResponse(stock_id=stock_id, fetched=0, inserted=0, missing_dates=[])


@router.post("/margin/all", response_model=SyncAllResponse)
async def sync_margin_all(session: Session = Depends(get_db)):
    return SyncAllResponse(stocks=len(_get_universe(session)), total_inserted=0, errors=[{"stock_id": "*", "error": _placeholder_error("Margin / short interest") }])


@router.get("/shareholding/status", response_model=DataStatusResponse)
async def get_shareholding_status(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = _empty_daily_status(session, start_date, end_date)
    return DataStatusResponse(trading_days=result["trading_days"], start_date=result["start_date"], end_date=result["end_date"], stocks=[DataStatusItem(**s) for s in result["stocks"]])


@router.post("/shareholding/bulk", response_model=SyncBulkResponse)
async def sync_shareholding_bulk(target_date: date = Query(default=None), session: Session = Depends(get_db)):
    d = target_date or date.today()
    return SyncBulkResponse(date=d.isoformat(), total=0, inserted=0, error=_placeholder_error("Shareholding") )


@router.post("/shareholding/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_shareholding_stock(stock_id: str, session: Session = Depends(get_db)):
    return SyncStockResponse(stock_id=stock_id, fetched=0, inserted=0, missing_dates=[])


@router.post("/shareholding/all", response_model=SyncAllResponse)
async def sync_shareholding_all(session: Session = Depends(get_db)):
    return SyncAllResponse(stocks=len(_get_universe(session)), total_inserted=0, errors=[{"stock_id": "*", "error": _placeholder_error("Shareholding") }])


@router.get("/securities-lending/status", response_model=DataStatusResponse)
async def get_securities_lending_status(start_date: date = Query(default=date(2020, 1, 1)), end_date: date = Query(default=None), session: Session = Depends(get_db)):
    if end_date is None:
        end_date = date.today()
    result = _empty_daily_status(session, start_date, end_date)
    return DataStatusResponse(trading_days=result["trading_days"], start_date=result["start_date"], end_date=result["end_date"], stocks=[DataStatusItem(**s) for s in result["stocks"]])


@router.post("/securities-lending/stock/{stock_id}", response_model=SyncStockResponse)
async def sync_securities_lending_stock(stock_id: str, session: Session = Depends(get_db)):
    return SyncStockResponse(stock_id=stock_id, fetched=0, inserted=0, missing_dates=[])


@router.post("/securities-lending/all", response_model=SyncAllResponse)
async def sync_securities_lending_all(session: Session = Depends(get_db)):
    return SyncAllResponse(stocks=len(_get_universe(session)), total_inserted=0, errors=[{"stock_id": "*", "error": _placeholder_error("Securities lending") }])


@router.get("/monthly-revenue/status", response_model=MonthlyStatusResponse)
async def get_monthly_revenue_status(start_year: int = Query(default=2020), end_year: int = Query(default=None), session: Session = Depends(get_db)):
    if end_year is None:
        end_year = date.today().year
    result = _empty_monthly_status(session, start_year, end_year)
    return MonthlyStatusResponse(expected_months=result["expected_months"], start_year=result["start_year"], end_year=result["end_year"], stocks=[MonthlyStatusItem(**s) for s in result["stocks"]])


@router.post("/monthly-revenue/stock/{stock_id}", response_model=MonthlyStockResponse)
async def sync_monthly_revenue_stock(stock_id: str, session: Session = Depends(get_db)):
    return MonthlyStockResponse(stock_id=stock_id, fetched=0, inserted=0, missing_months=[])


@router.post("/monthly-revenue/all", response_model=SyncAllResponse)
async def sync_monthly_revenue_all(session: Session = Depends(get_db)):
    return SyncAllResponse(stocks=len(_get_universe(session)), total_inserted=0, errors=[{"stock_id": "*", "error": _placeholder_error("Monthly revenue") }])
