"""
測試所有 Bulk Adapter
"""

import asyncio
from datetime import date

from src.adapters.twse import (
    TwseBulkOHLCVAdapter,
    TwseBulkPERAdapter,
    TwseBulkInstitutionalAdapter,
    TwseBulkMarginAdapter,
    TwseBulkShareholdingAdapter,
)


async def main():
    target = date(2026, 1, 29)
    print(f"測試日期: {target}\n")

    adapters = [
        ("OHLCV", TwseBulkOHLCVAdapter()),
        ("PER", TwseBulkPERAdapter()),
        ("Institutional", TwseBulkInstitutionalAdapter()),
        ("Margin", TwseBulkMarginAdapter()),
        ("Shareholding", TwseBulkShareholdingAdapter()),
    ]

    for name, adapter in adapters:
        print(f"{'=' * 50}")
        print(f"測試 {name}")
        print(f"{'=' * 50}")

        try:
            data = await adapter.fetch_all(target)
            print(f"取得 {len(data)} 筆資料")

            if data:
                # 找台積電
                tsmc = [d for d in data if getattr(d, "stock_id", "") == "2330"]
                if tsmc:
                    print(f"台積電 (2330): {tsmc[0]}")
                else:
                    print(f"範例: {data[0]}")
            else:
                print("無資料")
        except Exception as e:
            print(f"錯誤: {e}")

        print()
        await asyncio.sleep(2)  # 限流


if __name__ == "__main__":
    asyncio.run(main())
