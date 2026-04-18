"""US-only dataset catalog and smoke-test API."""

from datetime import date, timedelta
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel
import yfinance as yf

from src.shared.us_universe import get_us_universe_tickers

router = APIRouter()


class DatasetInfo(BaseModel):
    name: str
    display_name: str
    category: str
    source: str
    status: Literal["available", "needs_accumulation", "not_implemented", "pending"]
    description: str | None = None
    requires_stock_id: bool = True


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfo]
    total: int


class TestResult(BaseModel):
    dataset: str
    success: bool
    record_count: int
    sample_data: list[dict] | None = None
    error: str | None = None


US_DATASETS = [
    DatasetInfo(name="USEquityPrice", display_name="Daily OHLCV", category="technical", source="yfinance", status="available", description="US daily open, high, low, close, and volume"),
    DatasetInfo(name="USEquityPriceAdj", display_name="Adjusted Close", category="technical", source="yfinance", status="available", description="US adjusted close series from Yahoo Finance"),
    DatasetInfo(name="USEquityValuation", display_name="Valuation Snapshot", category="fundamental", source="custom", status="pending", description="Placeholder for PE, PB, and yield in the US-only fork"),
    DatasetInfo(name="USInstitutionalFlow", display_name="Institutional Flow", category="chips", source="custom", status="pending", description="Not implemented yet in the US-only fork"),
    DatasetInfo(name="USShortInterest", display_name="Short Interest / Margin", category="chips", source="custom", status="pending", description="Not implemented yet in the US-only fork"),
    DatasetInfo(name="USOwnership", display_name="Ownership / Holdings", category="chips", source="custom", status="pending", description="Not implemented yet in the US-only fork"),
    DatasetInfo(name="USRevenue", display_name="Revenue / Fundamentals", category="fundamental", source="custom", status="pending", description="Not implemented yet in the US-only fork"),
]


def _all_datasets() -> list[DatasetInfo]:
    return US_DATASETS


@router.get("", response_model=DatasetListResponse)
async def list_datasets(category: str | None = Query(None, description="Filter category"), status: str | None = Query(None, description="Filter status")):
    datasets = _all_datasets()
    if category:
        datasets = [d for d in datasets if d.category == category]
    if status:
        datasets = [d for d in datasets if d.status == status]
    return DatasetListResponse(datasets=datasets, total=len(datasets))


async def _test_us_dataset(dataset_name: str, stock_id: str, days: int) -> TestResult:
    if dataset_name not in {"USEquityPrice", "USEquityPriceAdj"}:
        return TestResult(dataset=dataset_name, success=False, record_count=0, error=f"{dataset_name} is not implemented yet in the US-only fork")

    if not stock_id:
        tickers = get_us_universe_tickers()
        stock_id = tickers[0] if tickers else "AAPL"

    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    ticker = yf.Ticker(stock_id)
    auto_adjust = dataset_name == "USEquityPriceAdj"
    df = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat(), auto_adjust=auto_adjust)
    if df.empty:
        return TestResult(dataset=dataset_name, success=False, record_count=0, error=f"No data returned for {stock_id}")

    sample = []
    for idx, row in df.head(3).iterrows():
        sample.append({
            "date": idx.date().isoformat(),
            "open": None if auto_adjust else float(row.get("Open", 0) or 0),
            "high": None if auto_adjust else float(row.get("High", 0) or 0),
            "low": None if auto_adjust else float(row.get("Low", 0) or 0),
            "close": float(row.get("Close", 0) or 0),
            "volume": int(row.get("Volume", 0) or 0),
        })
    return TestResult(dataset=dataset_name, success=True, record_count=len(df), sample_data=sample)


@router.get("/test/{dataset_name}", response_model=TestResult)
async def test_dataset(dataset_name: str, stock_id: str = Query("", description="Stock symbol"), days: int = Query(5, description="Test days", ge=1, le=30)):
    dataset = next((d for d in _all_datasets() if d.name == dataset_name), None)
    if not dataset:
        return TestResult(dataset=dataset_name, success=False, record_count=0, error=f"Dataset not found: {dataset_name}")
    if dataset.status == "not_implemented":
        return TestResult(dataset=dataset_name, success=False, record_count=0, error="Dataset not implemented yet")
    try:
        return await _test_us_dataset(dataset_name, stock_id, days)
    except Exception as e:
        return TestResult(dataset=dataset_name, success=False, record_count=0, error=str(e))


@router.get("/categories")
async def list_categories():
    categories = {}
    for d in _all_datasets():
        if d.category not in categories:
            categories[d.category] = {"name": d.category, "count": 0, "available": 0}
        categories[d.category]["count"] += 1
        if d.status == "available":
            categories[d.category]["available"] += 1
    return {
        "categories": [
            {"id": k, "name": _category_name(k), "total": v["count"], "available": v["available"]}
            for k, v in categories.items()
        ]
    }


def _category_name(cat: str) -> str:
    return {
        "technical": "Technical",
        "chips": "Flow / Ownership",
        "fundamental": "Fundamental",
        "derivatives": "Derivatives",
        "macro": "Macro",
    }.get(cat, cat)
