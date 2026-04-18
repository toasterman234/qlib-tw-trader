"""
Qlib .bin export service.

Converts SQLite-backed US market data into Qlib-compatible binary files.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.daily import AdjCloseRepository, OHLCVRepository
from src.repositories.models import StockUniverse, TradingCalendar


@dataclass
class ExportConfig:
    start_date: date
    end_date: date
    output_dir: Path
    include_fields: list[str] | None = None


@dataclass
class ExportResult:
    stocks_exported: int
    fields_per_stock: int
    total_files: int
    calendar_days: int
    output_path: str
    errors: list[dict]


class QlibExporter:
    """Qlib .bin exporter for the US-only fork."""

    DAILY_FIELDS = {
        "open": ("ohlcv", "open"),
        "high": ("ohlcv", "high"),
        "low": ("ohlcv", "low"),
        "close": ("ohlcv", "close"),
        "volume": ("ohlcv", "volume"),
        "adj_close": ("adj", "adj_close"),
    }

    def __init__(self, session: Session):
        self._session = session
        self._repos = {
            "ohlcv": OHLCVRepository(self._session),
            "adj": AdjCloseRepository(self._session),
        }

    def export(self, config: ExportConfig) -> ExportResult:
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        calendar = self._get_trading_calendar(config.start_date, config.end_date)
        if not calendar:
            raise ValueError("No trading days found in the specified range")

        self._write_calendar(output_dir / "calendars", calendar)

        stocks = self._get_stock_universe()
        if not stocks:
            raise ValueError("Stock universe is empty")

        fields = config.include_fields or list(self.DAILY_FIELDS.keys())

        errors = []
        stocks_exported = 0
        total_files = 0

        for stock in stocks:
            stock_id = stock.stock_id
            try:
                files_written = self._export_stock(
                    stock_id=stock_id,
                    calendar=calendar,
                    fields=fields,
                    output_dir=output_dir / "features" / stock_id,
                    start_date=config.start_date,
                    end_date=config.end_date,
                )
                stocks_exported += 1
                total_files += files_written
            except Exception as e:
                errors.append({"stock_id": stock_id, "error": str(e)})

        self._write_instruments(output_dir / "instruments", [s.stock_id for s in stocks], config.start_date, config.end_date)

        return ExportResult(
            stocks_exported=stocks_exported,
            fields_per_stock=len(fields),
            total_files=total_files,
            calendar_days=len(calendar),
            output_path=str(output_dir),
            errors=errors,
        )

    def _get_trading_calendar(self, start_date: date, end_date: date) -> list[date]:
        stmt = (
            select(TradingCalendar.date)
            .where(TradingCalendar.date >= start_date)
            .where(TradingCalendar.date <= end_date)
            .where(TradingCalendar.is_trading_day == True)
            .order_by(TradingCalendar.date)
        )
        return list(self._session.execute(stmt).scalars().all())

    def _get_stock_universe(self) -> list[StockUniverse]:
        stmt = select(StockUniverse).order_by(StockUniverse.rank)
        return list(self._session.execute(stmt).scalars().all())

    def _write_calendar(self, calendar_dir: Path, dates: list[date]):
        calendar_dir.mkdir(parents=True, exist_ok=True)
        with open(calendar_dir / "day.txt", "w") as f:
            for d in dates:
                f.write(f"{d.strftime('%Y-%m-%d')}\n")

    def _write_instruments(self, instruments_dir: Path, stock_ids: list[str], start_date: date, end_date: date):
        instruments_dir.mkdir(parents=True, exist_ok=True)
        with open(instruments_dir / "all.txt", "w") as f:
            for stock_id in stock_ids:
                f.write(f"{stock_id}\t{start_date}\t{end_date}\n")

    def _export_stock(self, stock_id: str, calendar: list[date], fields: list[str], output_dir: Path, start_date: date, end_date: date) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)
        date_to_idx = {d: i for i, d in enumerate(calendar)}
        n_days = len(calendar)
        data_cache = self._load_stock_data(stock_id, start_date, end_date)
        start_index = self._find_start_index(data_cache, date_to_idx)

        files_written = 0
        for field in fields:
            if field not in self.DAILY_FIELDS:
                continue
            source, attr = self.DAILY_FIELDS[field]
            arr = np.full(n_days, np.nan, dtype=np.float32)
            records = data_cache.get(source, [])
            for rec in records:
                if rec.date in date_to_idx:
                    idx = date_to_idx[rec.date]
                    value = getattr(rec, attr, None)
                    if value is not None:
                        arr[idx] = float(value)

            bin_path = output_dir / f"{field}.day.bin"
            with open(bin_path, "wb") as f:
                np.array([start_index], dtype="<f").tofile(f)
                arr[start_index:].astype("<f").tofile(f)
            files_written += 1

        return files_written

    def _find_start_index(self, data_cache: dict, date_to_idx: dict[date, int]) -> int:
        min_idx = float("inf")
        ohlcv_records = data_cache.get("ohlcv", [])
        if ohlcv_records:
            for rec in ohlcv_records:
                if rec.date in date_to_idx:
                    min_idx = min(min_idx, date_to_idx[rec.date])
                    break
        return int(min_idx) if min_idx != float("inf") else 0

    def _load_stock_data(self, stock_id: str, start_date: date, end_date: date) -> dict:
        ohlcv_raw = self._repos["ohlcv"].get(stock_id, start_date, end_date)
        ohlcv = [rec for rec in ohlcv_raw if rec.open > 0 and rec.high > 0 and rec.low > 0 and rec.close > 0]
        return {
            "ohlcv": ohlcv,
            "adj": self._repos["adj"].get(stock_id, start_date, end_date),
        }

    def get_available_fields(self) -> list[dict]:
        return [{"name": name, "source": source, "attribute": attr} for name, (source, attr) in self.DAILY_FIELDS.items()]
