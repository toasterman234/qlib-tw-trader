from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketConfig:
    code: str
    display_name: str
    app_title: str
    app_description: str
    timezone: str
    qlib_region: str
    calendar_symbol: str
    yf_suffix: str
    universe_name: str
    universe_description: str
    supports_tw_only_datasets: bool


TW_MARKET = MarketConfig(
    code="tw",
    display_name="Taiwan",
    app_title="QLib Trader API",
    app_description="Taiwan equity research, prediction, and backtesting API",
    timezone=os.getenv("APP_TIMEZONE", "Asia/Taipei"),
    qlib_region="cn",
    calendar_symbol="0050.TW",
    yf_suffix=".TW",
    universe_name="tw100",
    universe_description="Top 100 Taiwan equities by market cap (ex ETF / KY)",
    supports_tw_only_datasets=True,
)

US_MARKET = MarketConfig(
    code="us",
    display_name="United States",
    app_title="QLib Trader API",
    app_description="US equity research, prediction, and backtesting API",
    timezone=os.getenv("APP_TIMEZONE", "America/New_York"),
    qlib_region="us",
    calendar_symbol="SPY",
    yf_suffix="",
    universe_name="us-core-100",
    universe_description="Curated US large-cap universe shipped with the app",
    supports_tw_only_datasets=False,
)


def get_market() -> MarketConfig:
    code = os.getenv("APP_MARKET", "tw").strip().lower()
    return US_MARKET if code == "us" else TW_MARKET


def market_is_us() -> bool:
    return get_market().code == "us"


def market_is_tw() -> bool:
    return get_market().code == "tw"
