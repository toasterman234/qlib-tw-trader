"""
測試 DataService 功能
"""

import asyncio
from datetime import date

from src.repositories import init_db
from src.services import DataService, Dataset


async def test_ohlcv():
    """測試日K線取得"""
    print("=" * 60)
    print("測試 1: 取得日K線（OHLCV）")
    print("=" * 60)

    ds = DataService()

    # 取得台積電最近一週資料
    start = date(2026, 1, 20)
    end = date(2026, 1, 29)

    print(f"查詢 2330 日K線: {start} ~ {end}")
    data = await ds.get_ohlcv("2330", start, end)

    if data:
        print(f"取得 {len(data)} 筆資料")
        for d in data[-3:]:  # 只顯示最後3筆
            print(f"  {d.date}: O={d.open} H={d.high} L={d.low} C={d.close} V={d.volume}")
    else:
        print("無資料")

    return len(data) > 0


async def test_per():
    """測試 PER 取得"""
    print("\n" + "=" * 60)
    print("測試 2: 取得 PER/PBR")
    print("=" * 60)

    ds = DataService()

    start = date(2026, 1, 20)
    end = date(2026, 1, 29)

    print(f"查詢 2330 PER: {start} ~ {end}")
    data = await ds.get_per("2330", start, end)

    if data:
        print(f"取得 {len(data)} 筆資料")
        for d in data[-3:]:
            print(f"  {d.date}: PE={d.pe_ratio} PB={d.pb_ratio} DY={d.dividend_yield}")
    else:
        print("無資料")

    return len(data) > 0


async def test_adj_close():
    """測試還原股價取得"""
    print("\n" + "=" * 60)
    print("測試 3: 取得還原股價（yfinance）")
    print("=" * 60)

    ds = DataService()

    start = date(2026, 1, 20)
    end = date(2026, 1, 29)

    print(f"查詢 2330 還原股價: {start} ~ {end}")
    data = await ds.get_adj_close("2330", start, end)

    if data:
        print(f"取得 {len(data)} 筆資料")
        for d in data[-3:]:
            print(f"  {d.date}: Adj Close = {d.adj_close}")
    else:
        print("無資料")

    return len(data) > 0


async def test_bulk():
    """測試 Bulk API（全市場）"""
    print("\n" + "=" * 60)
    print("測試 4: 取得全市場日K（Bulk）")
    print("=" * 60)

    ds = DataService()

    target = date(2026, 1, 29)
    print(f"查詢全市場日K: {target}")

    data = await ds.fetch_bulk(Dataset.OHLCV, target)

    if data:
        print(f"取得 {len(data)} 支股票資料")
        # 顯示前5支
        for d in data[:5]:
            print(f"  {d.stock_id}: O={d.open} C={d.close}")
    else:
        print("無資料（可能 OpenAPI 尚未更新或非交易日）")

    return len(data) > 0


async def main():
    print("初始化資料庫...")
    init_db()

    results = {}

    try:
        results["ohlcv"] = await test_ohlcv()
    except Exception as e:
        print(f"OHLCV 測試失敗: {e}")
        results["ohlcv"] = False

    try:
        results["per"] = await test_per()
    except Exception as e:
        print(f"PER 測試失敗: {e}")
        results["per"] = False

    try:
        results["adj_close"] = await test_adj_close()
    except Exception as e:
        print(f"Adj Close 測試失敗: {e}")
        results["adj_close"] = False

    try:
        results["bulk"] = await test_bulk()
    except Exception as e:
        print(f"Bulk 測試失敗: {e}")
        results["bulk"] = False

    # 總結
    print("\n" + "=" * 60)
    print("測試結果")
    print("=" * 60)
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
