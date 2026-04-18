"""Dataset catalog and smoke-test API."""

import os
from datetime import date, timedelta
from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel
import httpx
import yfinance as yf

from src.shared.market import get_market, market_is_us
from src.shared.us_universe import get_us_universe_tickers

router = APIRouter()

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.getenv("FINMIND_KEY", "")


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


TW_DATASETS = [
    DatasetInfo(name="TaiwanStockPrice", display_name="日K線", category="technical", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockPriceAdj", display_name="還原股價", category="technical", source="yfinance", status="available"),
    DatasetInfo(name="TaiwanStockPER", display_name="PER/PBR/殖利率", category="technical", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockMarginPurchaseShortSale", display_name="個股融資融券", category="chips", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockInstitutionalInvestorsBuySell", display_name="個股三大法人", category="chips", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockShareholding", display_name="外資持股", category="chips", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockSecuritiesLending", display_name="借券明細", category="chips", source="twse/finmind", status="available"),
    DatasetInfo(name="TaiwanStockMonthRevenue", display_name="月營收", category="fundamental", source="finmind", status="available"),
    DatasetInfo(name="TaiwanStockDayTrading", display_name="當沖成交量值", category="technical", source="finmind", status="pending", description="非核心因子，資料完整度低"),
    DatasetInfo(name="TaiwanStockHoldingSharesPer", display_name="股權分級表", category="chips", source="twse", status="pending", description="籌碼集中度，需累積"),
    DatasetInfo(name="TaiwanStockTradingDailyReport", display_name="分點資料", category="chips", source="twse", status="pending", description="主力券商進出，需累積"),
    DatasetInfo(name="TaiwanFuturesDaily", display_name="期貨日成交", category="derivatives", source="finmind", status="pending", requires_stock_id=False, description="待確認個股期貨對應"),
    DatasetInfo(name="TaiwanOptionDaily", display_name="選擇權日成交", category="derivatives", source="finmind", status="pending", requires_stock_id=False, description="待確認個股選擇權對應"),
    DatasetInfo(name="TaiwanFuturesInstitutionalInvestors", display_name="期貨三大法人", category="derivatives", source="finmind", status="pending", requires_stock_id=False, description="待確認個股期貨對應"),
]

US_DATASETS = [
    DatasetInfo(name="USEquityPrice", display_name="Daily OHLCV", category="technical", source="yfinance", status="available", description="US daily open, high, low, close, and volume"),
    DatasetInfo(name="USEquityPriceAdj", display_name="Adjusted Close", category="technical", source="yfinance", status="available", description="US adjusted close series from Yahoo Finance"),
    DatasetInfo(name="USEquityValuation", display_name="Valuation Snapshot", category="fundamental", source="yfinance", status="pending", description="Placeholder for PE, PB, and yield in US mode"),
    DatasetInfo(name="USInstitutionalFlow", display_name="Institutional Flow", category="chips", source="custom", status="pending", description="Not implemented in US MVP"),
    DatasetInfo(name="USShortInterest", display_name="Short Interest / Margin", category="chips", source="custom", status="pending", description="Not implemented in US MVP"),
    DatasetInfo(name="USOwnership", display_name="Ownership / Holdings", category="chips", source="custom", status="pending", description="Not implemented in US MVP"),
    DatasetInfo(name="USRevenue", display_name="Revenue / Fundamentals", category="fundamental", source="custom", status="pending", description="Not implemented in US MVP"),
]


def _all_datasets() -> list[DatasetInfo]:
    return US_DATASETS if market_is_us() else TW_DATASETS


@router.get("", response_model=DatasetListResponse)
async def list_datasets(category: str | None = Query(None, description="Filter category"), status: str | None = Query(None, description="Filter status")):
    datasets = _all_datasets()
    if category:
        datasets = [d for d in datasets if d.category == category]
    if status:
        datasets = [d for d in datasets if d.status == status]
    return DatasetListResponse(datasets=datasets, total=len(datasets))


async def _test_tw_dataset(dataset: DatasetInfo, dataset_name: str, stock_id: str, days: int) -> TestResult:
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    params = {
        "dataset": dataset_name,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    if dataset.requires_stock_id:
        params["data_id"] = stock_id
    else:
        if dataset_name == "TaiwanFuturesDaily":
            params["data_id"] = "TX"
        elif dataset_name == "TaiwanOptionDaily":
            params["data_id"] = "TXO"
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN

    async with httpx.AsyncClient() as client:
        resp = await client.get(FINMIND_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    if data.get("status") != 200:
        return TestResult(dataset=dataset_name, success=False, record_count=0, error=data.get("msg", "Unknown error"))
    records = data.get("data", [])
    return TestResult(dataset=dataset_name, success=True, record_count=len(records), sample_data=records[:3] if records else None)


async def _test_us_dataset(dataset_name: str, stock_id: str, days: int) -> TestResult:
    if dataset_name not in {"USEquityPrice", "USEquityPriceAdj"}:
        return TestResult(dataset=dataset_name, success=False, record_count=0, error=f"{dataset_name} is not implemented in US mode yet")

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
        if market_is_us():
            return await _test_us_dataset(dataset_name, stock_id, days)
        return await _test_tw_dataset(dataset, dataset_name, stock_id or "2330", days)
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
    market = get_market()
    if market.code == "us":
        return {
            "technical": "Technical",
            "chips": "Flow / Ownership",
            "fundamental": "Fundamental",
            "derivatives": "Derivatives",
            "macro": "Macro",
        }.get(cat, cat)
    return {
        "technical": "技術面",
        "chips": "籌碼面",
        "fundamental": "基本面",
        "derivatives": "衍生品",
        "macro": "總經指標",
    }.get(cat, cat)
