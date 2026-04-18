from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import yfinance as yf
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.repositories.models import (
    StockDaily,
    StockDailyAdj,
    StockDailyInstitutional,
    StockDailyMargin,
    StockDailyPER,
    StockDailySecuritiesLending,
    StockDailyShareholding,
    StockMonthlyRevenue,
    StockUniverse,
    TradingCalendar,
)
from src.services.sync_service import SyncService
from src.shared.market import get_market


class USSyncService(SyncService):
    """US-market sync service built on Yahoo Finance."""

    def __init__(self, session: Session):
        super().__init__(session)
        self.market = get_market()

    def _yf_symbol(self, stock_id: str) -> str:
        return f"{stock_id}{self.market.yf_suffix}"

    def _empty_daily_status(self, start_date: date, end_date: date) -> dict:
        return self._get_daily_status(StockDaily, start_date, end_date)

    async def sync_trading_calendar(self, start_date: date, end_date: date) -> int:
        ticker = yf.Ticker(self.market.calendar_symbol)
        df = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat())
        if df.empty:
            return 0
        trading_dates = {idx.date() for idx in df.index}
        stmt = select(TradingCalendar.date).where(
            TradingCalendar.date >= start_date,
            TradingCalendar.date <= end_date,
        )
        existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        for d in trading_dates - existing:
            self._session.add(TradingCalendar(date=d, is_trading_day=True))
        self._session.commit()
        return len(trading_dates - existing)

    async def sync_stock_daily(self, stock_id: str, start_date: date, end_date: date) -> dict:
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        stmt = select(StockDaily.date).where(
            StockDaily.stock_id == stock_id,
            StockDaily.date >= start_date,
            StockDaily.date <= end_date,
        )
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        df = yf.Ticker(self._yf_symbol(stock_id)).history(
            start=min(missing_dates).isoformat(),
            end=(max(missing_dates) + timedelta(days=1)).isoformat(),
            auto_adjust=False,
        )
        if df.empty:
            return {"fetched": 0, "inserted": 0, "missing_dates": [d.isoformat() for d in missing_dates]}

        inserted = 0
        for idx, row in df.iterrows():
            r_date = idx.date()
            if r_date not in missing_dates or r_date in existing_dates:
                continue
            open_val = self._safe_decimal(row.get("Open"))
            close_val = self._safe_decimal(row.get("Close"))
            if open_val is None or close_val is None:
                continue
            self._session.add(
                StockDaily(
                    stock_id=stock_id,
                    date=r_date,
                    open=open_val,
                    high=self._safe_decimal(row.get("High")) or open_val,
                    low=self._safe_decimal(row.get("Low")) or open_val,
                    close=close_val,
                    volume=self._safe_int(row.get("Volume")),
                )
            )
            inserted += 1
        self._session.commit()

        stmt = select(StockDaily.date).where(
            StockDaily.stock_id == stock_id,
            StockDaily.date >= start_date,
            StockDaily.date <= end_date,
        )
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)
        return {"fetched": len(df), "inserted": inserted, "missing_dates": [d.isoformat() for d in still_missing]}

    async def sync_stock_daily_bulk(self, target_date: date) -> dict:
        stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
        stock_ids = [row[0] for row in self._session.execute(stmt).fetchall()]
        if not stock_ids:
            return {"date": target_date.isoformat(), "total": 0, "inserted": 0}

        stmt = select(StockDaily.stock_id).where(StockDaily.date == target_date)
        existing_stocks = {row[0] for row in self._session.execute(stmt).fetchall()}
        pending = [sid for sid in stock_ids if sid not in existing_stocks]
        if not pending:
            return {"date": target_date.isoformat(), "total": len(stock_ids), "inserted": 0}

        symbols = [self._yf_symbol(sid) for sid in pending]
        data = yf.download(symbols, start=target_date.isoformat(), end=(target_date + timedelta(days=1)).isoformat(), progress=False, auto_adjust=False, group_by="ticker")
        if data.empty:
            return {"date": target_date.isoformat(), "total": len(stock_ids), "inserted": 0, "error": "No Yahoo Finance data returned"}

        inserted = 0
        if len(symbols) == 1:
            sid = pending[0]
            open_val = self._safe_decimal(data["Open"].iloc[0] if "Open" in data else None)
            close_val = self._safe_decimal(data["Close"].iloc[0] if "Close" in data else None)
            if open_val is not None and close_val is not None:
                self._session.add(StockDaily(stock_id=sid, date=target_date, open=open_val, high=self._safe_decimal(data["High"].iloc[0]) or open_val, low=self._safe_decimal(data["Low"].iloc[0]) or open_val, close=close_val, volume=self._safe_int(data["Volume"].iloc[0])))
                inserted += 1
        else:
            for sid in pending:
                symbol = self._yf_symbol(sid)
                try:
                    frame = data[symbol]
                    if frame.empty:
                        continue
                    row = frame.iloc[0]
                    open_val = self._safe_decimal(row.get("Open"))
                    close_val = self._safe_decimal(row.get("Close"))
                    if open_val is None or close_val is None:
                        continue
                    self._session.add(StockDaily(stock_id=sid, date=target_date, open=open_val, high=self._safe_decimal(row.get("High")) or open_val, low=self._safe_decimal(row.get("Low")) or open_val, close=close_val, volume=self._safe_int(row.get("Volume"))))
                    inserted += 1
                except Exception:
                    continue
        self._session.commit()
        return {"date": target_date.isoformat(), "total": len(stock_ids), "inserted": inserted}

    async def sync_all_stocks(self, start_date: date, end_date: date) -> dict:
        await self.sync_trading_calendar(start_date, end_date)
        stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
        stock_ids = [row[0] for row in self._session.execute(stmt).fetchall()]
        total_inserted = 0
        errors = []
        for stock_id in stock_ids:
            try:
                result = await self.sync_stock_daily(stock_id, start_date, end_date)
                total_inserted += result["inserted"]
            except Exception as e:
                errors.append({"stock_id": stock_id, "error": str(e)})
        return {"stocks": len(stock_ids), "total_inserted": total_inserted, "errors": errors}

    async def sync_adj(self, stock_id: str, start_date: date, end_date: date) -> dict:
        trading_dates = set(self.get_trading_dates(start_date, end_date))
        if not trading_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}
        stmt = select(StockDailyAdj.date).where(StockDailyAdj.stock_id == stock_id, StockDailyAdj.date >= start_date, StockDailyAdj.date <= end_date)
        existing_dates = {row[0] for row in self._session.execute(stmt).fetchall()}
        missing_dates = sorted(trading_dates - existing_dates)
        if not missing_dates:
            return {"fetched": 0, "inserted": 0, "missing_dates": []}

        df = yf.download(self._yf_symbol(stock_id), start=min(missing_dates).isoformat(), end=(max(missing_dates) + timedelta(days=1)).isoformat(), progress=False, auto_adjust=False)
        if df.empty:
            return {"fetched": 0, "inserted": 0, "missing_dates": [d.isoformat() for d in missing_dates]}

        inserted = 0
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        for idx, row in df.iterrows():
            r_date = idx.date()
            if r_date not in missing_dates or r_date in existing_dates:
                continue
            adj_close = self._safe_decimal(row.get(col))
            if adj_close is None:
                continue
            self._session.add(StockDailyAdj(stock_id=stock_id, date=r_date, adj_close=adj_close))
            inserted += 1
        self._session.commit()
        stmt = select(StockDailyAdj.date).where(StockDailyAdj.stock_id == stock_id, StockDailyAdj.date >= start_date, StockDailyAdj.date <= end_date)
        final_existing = {row[0] for row in self._session.execute(stmt).fetchall()}
        still_missing = sorted(trading_dates - final_existing)
        return {"fetched": len(df), "inserted": inserted, "missing_dates": [d.isoformat() for d in still_missing]}

    async def sync_adj_bulk(self, target_date: date) -> dict:
        stmt = select(StockUniverse.stock_id).order_by(StockUniverse.rank)
        stock_ids = [row[0] for row in self._session.execute(stmt).fetchall()]
        inserted = 0
        for sid in stock_ids:
            try:
                result = await self.sync_adj(sid, target_date, target_date)
                inserted += result["inserted"]
            except Exception:
                continue
        return {"date": target_date.isoformat(), "total": len(stock_ids), "inserted": inserted}

    def get_adj_status(self, start_date: date, end_date: date) -> dict:
        return self._get_daily_status(StockDailyAdj, start_date, end_date)

    def get_per_status(self, start_date: date, end_date: date) -> dict:
        return self._get_daily_status(StockDailyPER, start_date, end_date)

    async def sync_per_bulk(self, target_date: date) -> dict:
        return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "PER/PBR sync is not implemented for US mode yet"}

    async def sync_per(self, stock_id: str, start_date: date, end_date: date) -> dict:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    def get_institutional_status(self, start_date: date, end_date: date) -> dict:
        return self._get_daily_status(StockDailyInstitutional, start_date, end_date)

    async def sync_institutional_bulk(self, target_date: date) -> dict:
        return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "Institutional flow sync is not implemented for US mode yet"}

    async def sync_institutional(self, stock_id: str, start_date: date, end_date: date) -> dict:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    def get_margin_status(self, start_date: date, end_date: date) -> dict:
        return self._get_daily_status(StockDailyMargin, start_date, end_date)

    async def sync_margin_bulk(self, target_date: date) -> dict:
        return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "Margin short-interest sync is not implemented for US mode yet"}

    async def sync_margin(self, stock_id: str, start_date: date, end_date: date) -> dict:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    def get_shareholding_status(self, start_date: date, end_date: date) -> dict:
        return self._get_daily_status(StockDailyShareholding, start_date, end_date)

    async def sync_shareholding_bulk(self, target_date: date) -> dict:
        return {"date": target_date.isoformat(), "total": 0, "inserted": 0, "error": "Shareholding sync is not implemented for US mode yet"}

    async def sync_shareholding(self, stock_id: str, start_date: date, end_date: date) -> dict:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    def get_securities_lending_status(self, start_date: date, end_date: date) -> dict:
        return self._get_daily_status(StockDailySecuritiesLending, start_date, end_date)

    async def sync_securities_lending(self, stock_id: str, start_date: date, end_date: date) -> dict:
        return {"fetched": 0, "inserted": 0, "missing_dates": []}

    def get_monthly_revenue_status(self, start_year: int, end_year: int) -> dict:
        universe = self._session.execute(select(StockUniverse).order_by(StockUniverse.rank)).scalars().all()
        stocks = [{"stock_id": s.stock_id, "name": s.name, "rank": s.rank, "earliest_month": None, "latest_month": None, "total_records": 0, "missing_count": 0, "coverage_pct": 0.0} for s in universe]
        return {"expected_months": 0, "start_year": start_year, "end_year": end_year, "stocks": stocks}

    async def sync_monthly_revenue(self, stock_id: str, start_year: int, end_year: int) -> dict:
        return {"fetched": 0, "inserted": 0, "missing_months": []}

    async def sync_monthly_revenue_all(self, start_year: int, end_year: int) -> dict:
        stmt = select(StockUniverse.stock_id)
        stock_ids = [row[0] for row in self._session.execute(stmt).fetchall()]
        return {"stocks": len(stock_ids), "total_inserted": 0, "errors": []}
