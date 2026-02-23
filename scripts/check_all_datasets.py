"""
完整的 FinMind Dataset ↔ TWSE API 對應表與可用性檢查
"""

import asyncio
from datetime import date
import httpx
import json

TODAY = "2026-01-29"
TODAY_ROC = "1150129"
TEST_STOCK = "2330"


def parse_date(raw: str) -> str | None:
    """解析各種日期格式，返回 YYYY-MM-DD"""
    if not raw:
        return None
    raw = str(raw).strip()

    # 格式: 1150128 (7位數民國年)
    if raw.isdigit() and len(raw) == 7:
        year = int(raw[:3]) + 1911
        return f"{year}-{raw[3:5]}-{raw[5:7]}"

    # 格式: 20260128 (8位數西元年)
    if raw.isdigit() and len(raw) == 8:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    # 格式: 115/01/28
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3:
            year = int(parts[0])
            if year < 1911:
                year += 1911
            return f"{year}-{parts[1]}-{parts[2]}"

    # 格式: 2026-01-28
    if "-" in raw and len(raw) >= 10:
        return raw[:10]

    return raw


# ============================================
# 完整的 FinMind Dataset ↔ TWSE 對應表
# ============================================

DATASET_MAPPING = {
    # ==================== 技術面 ====================
    "TaiwanStockInfo": {
        "category": "技術面",
        "name": "台股總覽",
        "twse_api": "OpenAPI /v1/exchangeReport/STOCK_DAY_ALL",
        "twse_url": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        "api_type": "openapi",
        "date_field": "Date",
        "update_time": "~15:00",
        "note": "從日K推算股票清單",
    },
    "TaiwanStockPrice": {
        "category": "技術面",
        "name": "日K線",
        "twse_api": "OpenAPI /v1/exchangeReport/STOCK_DAY_ALL",
        "twse_url": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        "api_type": "openapi",
        "date_field": "Date",
        "update_time": "~15:00",
        "note": "收盤後更新",
    },
    "TaiwanStockPER": {
        "category": "技術面",
        "name": "PER/PBR/殖利率",
        "twse_api": "OpenAPI /v1/exchangeReport/BWIBBU_ALL",
        "twse_url": "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL",
        "api_type": "openapi",
        "date_field": "Date",
        "update_time": "~15:00",
        "note": "收盤後更新",
    },
    "TaiwanStockDayTrading": {
        "category": "技術面",
        "name": "當沖成交量值",
        "twse_api": "RWD /afterTrading/BWISU",
        "twse_url": "https://www.twse.com.tw/rwd/zh/afterTrading/BWISU?response=json",
        "api_type": "rwd",
        "date_field": "date",
        "update_time": "~15:30",
        "note": "當沖資訊",
    },

    # ==================== 籌碼面 ====================
    "TaiwanStockMarginPurchaseShortSale": {
        "category": "籌碼面",
        "name": "個股融資融券",
        "twse_api": "RWD /marginTrading/MI_MARGN",
        "twse_url": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json",
        "api_type": "rwd",
        "date_field": "date",
        "update_time": "~19:00",
        "note": "融資融券餘額，更新較晚",
    },
    "TaiwanStockTotalMarginPurchaseShortSale": {
        "category": "籌碼面",
        "name": "整體融資融券",
        "twse_api": "RWD /marginTrading/MI_MARGN",
        "twse_url": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json",
        "api_type": "rwd",
        "date_field": "date",
        "update_time": "~19:00",
        "note": "同上，取彙總資料",
    },
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "category": "籌碼面",
        "name": "個股三大法人",
        "twse_api": "RWD /fund/T86",
        "twse_url": "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALL",
        "api_type": "rwd",
        "date_field": "date",
        "update_time": "~16:30",
        "note": "三大法人買賣超",
    },
    "TaiwanStockTotalInstitutionalInvestors": {
        "category": "籌碼面",
        "name": "整體三大法人",
        "twse_api": "RWD /fund/TWT38U",
        "twse_url": "https://www.twse.com.tw/rwd/zh/fund/TWT38U?response=json",
        "api_type": "rwd",
        "date_field": "date",
        "update_time": "~16:30",
        "note": "三大法人彙總",
    },
    "TaiwanStockShareholding": {
        "category": "籌碼面",
        "name": "外資持股",
        "twse_api": "RWD /fund/MI_QFIIS",
        "twse_url": "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?response=json",
        "api_type": "rwd",
        "date_field": "date",
        "update_time": "T+1",
        "note": "外資持股比例，隔天更新",
    },
    "TaiwanStockSecuritiesLending": {
        "category": "籌碼面",
        "name": "借券明細",
        "twse_api": "RWD /lending/t13sa900",
        "twse_url": "https://www.twse.com.tw/rwd/zh/lending/TWT93U?response=json",
        "api_type": "rwd",
        "date_field": "date",
        "update_time": "~17:00",
        "note": "借券餘額",
    },

    # ==================== 基本面 ====================
    "TaiwanStockDividendResult": {
        "category": "基本面",
        "name": "除權息結果",
        "twse_api": "RWD /exRight/TWT49U",
        "twse_url": "https://www.twse.com.tw/rwd/zh/exRight/TWT49U?response=json",
        "api_type": "rwd",
        "date_field": "date",
        "update_time": "事件型",
        "note": "除權息日才有資料",
    },
    "TaiwanStockMonthRevenue": {
        "category": "基本面",
        "name": "月營收",
        "twse_api": "N/A",
        "twse_url": None,
        "api_type": "finmind_only",
        "date_field": "date",
        "update_time": "每月10日前",
        "note": "FinMind 獨有",
    },
    "TaiwanStockCashFlowsStatement": {
        "category": "基本面",
        "name": "現金流量表",
        "twse_api": "N/A",
        "twse_url": None,
        "api_type": "finmind_only",
        "date_field": "date",
        "update_time": "季報",
        "note": "FinMind 獨有",
    },
    "TaiwanStockFinancialStatements": {
        "category": "基本面",
        "name": "綜合損益表",
        "twse_api": "N/A",
        "twse_url": None,
        "api_type": "finmind_only",
        "date_field": "date",
        "update_time": "季報",
        "note": "FinMind 獨有",
    },
    "TaiwanStockBalanceSheet": {
        "category": "基本面",
        "name": "資產負債表",
        "twse_api": "N/A",
        "twse_url": None,
        "api_type": "finmind_only",
        "date_field": "date",
        "update_time": "季報",
        "note": "FinMind 獨有",
    },
    "TaiwanStockDividend": {
        "category": "基本面",
        "name": "股利政策",
        "twse_api": "N/A",
        "twse_url": None,
        "api_type": "finmind_only",
        "date_field": "date",
        "update_time": "年度",
        "note": "FinMind 獨有",
    },

    # ==================== 衍生品 ====================
    "TaiwanFuturesDaily": {
        "category": "衍生品",
        "name": "期貨日成交",
        "twse_api": "TAIFEX",
        "twse_url": None,
        "api_type": "finmind_only",
        "finmind_data_id": "TX",
        "date_field": "date",
        "update_time": "~15:00",
        "note": "期交所資料，FinMind 有整合",
    },
    "TaiwanOptionDaily": {
        "category": "衍生品",
        "name": "選擇權日成交",
        "twse_api": "TAIFEX",
        "twse_url": None,
        "api_type": "finmind_only",
        "finmind_data_id": "TXO",
        "date_field": "date",
        "update_time": "~15:00",
        "note": "期交所資料，FinMind 有整合",
    },
    "TaiwanFuturesInstitutionalInvestors": {
        "category": "衍生品",
        "name": "期貨三大法人",
        "twse_api": "TAIFEX",
        "twse_url": None,
        "api_type": "finmind_only",
        "finmind_data_id": "TX",
        "date_field": "date",
        "update_time": "~15:00",
        "note": "期交所資料，FinMind 有整合",
    },
    "TaiwanOptionInstitutionalInvestors": {
        "category": "衍生品",
        "name": "選擇權三大法人",
        "twse_api": "TAIFEX",
        "twse_url": None,
        "api_type": "finmind_only",
        "finmind_data_id": "TXO",
        "date_field": "date",
        "update_time": "~15:00",
        "note": "期交所資料，FinMind 有整合",
    },

    # ==================== 其他 ====================
    "GoldPrice": {
        "category": "其他",
        "name": "黃金價格",
        "twse_api": "N/A",
        "twse_url": None,
        "api_type": "finmind_only",
        "date_field": "date",
        "update_time": "每日",
        "note": "FinMind 獨有",
    },
    "CrudeOilPrices": {
        "category": "其他",
        "name": "原油價格",
        "twse_api": "N/A",
        "twse_url": None,
        "api_type": "finmind_only",
        "finmind_data_id": "WTI",
        "date_field": "date",
        "update_time": "延遲數天",
        "note": "FinMind 獨有，更新慢",
    },
    "TaiwanExchangeRate": {
        "category": "其他",
        "name": "匯率",
        "twse_api": "N/A",
        "twse_url": None,
        "api_type": "finmind_only",
        "finmind_data_id": "USD",
        "date_field": "date",
        "update_time": "每日",
        "note": "FinMind 獨有",
    },
    "TaiwanStockPriceAdj": {
        "category": "技術面",
        "name": "還原股價",
        "twse_api": "N/A (用 yfinance)",
        "twse_url": None,
        "api_type": "yfinance",
        "date_field": "Date",
        "update_time": "~15:30",
        "note": "使用 yfinance 的 Adj Close",
    },
}


async def check_twse_openapi(client: httpx.AsyncClient, url: str, date_field: str) -> tuple[str | None, int]:
    """檢查 TWSE OpenAPI"""
    try:
        resp = await client.get(url, timeout=30, follow_redirects=True)
        if resp.status_code != 200:
            return None, 0
        data = resp.json()
        if not data or not isinstance(data, list):
            return None, 0
        latest = parse_date(data[0].get(date_field, ""))
        return latest, len(data)
    except:
        return None, 0


async def check_twse_rwd(client: httpx.AsyncClient, url: str) -> tuple[str | None, int]:
    """檢查 TWSE RWD API"""
    try:
        resp = await client.get(url, timeout=30, follow_redirects=True)
        if resp.status_code != 200:
            return None, 0
        data = resp.json()
        if data.get("stat") != "OK":
            return None, 0
        latest = parse_date(data.get("date", ""))
        rows = len(data.get("data", []))
        return latest, rows
    except:
        return None, 0


async def check_finmind(client: httpx.AsyncClient, dataset: str, data_id: str = None) -> tuple[str | None, int]:
    """檢查 FinMind API"""
    try:
        params = {
            "dataset": dataset,
            "start_date": "2026-01-01",
            "end_date": "2026-01-29",
        }
        if data_id:
            params["data_id"] = data_id

        resp = await client.get(
            "https://api.finmindtrade.com/api/v4/data",
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            return None, 0
        result = resp.json()
        if result.get("status") != 200:
            return None, 0
        data = result.get("data", [])
        if not data:
            return None, 0

        dates = [str(row.get("date", ""))[:10] for row in data if row.get("date")]
        latest = max(dates) if dates else None
        return latest, len(data)
    except:
        return None, 0


async def check_yfinance(stock_id: str) -> tuple[str | None, int]:
    """檢查 yfinance（同步）"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{stock_id}.TW")
        hist = ticker.history(period="5d")
        if hist.empty:
            return None, 0
        latest = hist.index[-1].strftime("%Y-%m-%d")
        return latest, len(hist)
    except:
        return None, 0


async def main():
    print(f"=== FinMind Dataset ↔ TWSE 對應表（{TODAY}）===\n")

    results = []

    async with httpx.AsyncClient() as client:
        for dataset, info in DATASET_MAPPING.items():
            api_type = info["api_type"]
            latest_date = None
            count = 0

            print(f"檢查 {dataset}...", end=" ", flush=True)

            if api_type == "openapi" and info["twse_url"]:
                latest_date, count = await check_twse_openapi(
                    client, info["twse_url"], info["date_field"]
                )
                await asyncio.sleep(2)

            elif api_type == "rwd" and info["twse_url"]:
                latest_date, count = await check_twse_rwd(client, info["twse_url"])
                await asyncio.sleep(2)

            elif api_type == "finmind_only":
                data_id = info.get("finmind_data_id", TEST_STOCK)
                latest_date, count = await check_finmind(client, dataset, data_id)
                await asyncio.sleep(1)

            elif api_type == "yfinance":
                latest_date, count = await check_yfinance(TEST_STOCK)

            has_today = latest_date == TODAY
            status = "✅" if has_today else "❌"
            print(status)

            results.append({
                "dataset": dataset,
                "category": info["category"],
                "name": info["name"],
                "twse_api": info["twse_api"],
                "update_time": info["update_time"],
                "latest_date": latest_date,
                "has_today": has_today,
                "count": count,
                "note": info["note"],
            })

    # 輸出表格
    print("\n" + "=" * 130)
    print(f"{'Dataset':<45} {'分類':<8} {'TWSE API':<25} {'更新時間':<12} {'1/29':<5} {'最新日期':<12} {'筆數'}")
    print("=" * 130)

    # 按分類排序
    categories = ["技術面", "籌碼面", "基本面", "衍生品", "其他"]
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            print(f"\n【{cat}】")
            for r in cat_results:
                status = "✅" if r["has_today"] else "❌"
                latest = r["latest_date"] or "N/A"
                print(f"  {r['dataset']:<43} {r['twse_api']:<25} {r['update_time']:<12} {status:<5} {latest:<12} {r['count']}")

    # 統計
    print("\n" + "=" * 130)
    has_today_count = sum(1 for r in results if r["has_today"])
    print(f"\n總計: {has_today_count}/{len(results)} 個資料集有 {TODAY} 的資料")

    # 分類統計
    print("\n各資料來源統計:")
    for api_type in ["openapi", "rwd", "finmind_only", "yfinance"]:
        type_results = [r for r in results if DATASET_MAPPING[r["dataset"]]["api_type"] == api_type]
        if type_results:
            ok = sum(1 for r in type_results if r["has_today"])
            print(f"  {api_type}: {ok}/{len(type_results)}")

    # 未更新原因分析
    no_today = [r for r in results if not r["has_today"]]
    if no_today:
        print(f"\n❌ 未有 {TODAY} 資料的原因分析:")

        delayed = [r for r in no_today if r["latest_date"] and r["latest_date"] < TODAY]
        periodic = [r for r in no_today if r["update_time"] in ["季報", "年度", "每月10日前", "事件型"]]
        no_data = [r for r in no_today if not r["latest_date"] and r not in periodic]

        if delayed:
            print("\n  【延遲更新】（有資料但不是今天）:")
            for r in delayed:
                print(f"    - {r['name']} ({r['dataset']}): 最新 {r['latest_date']}，預計 {r['update_time']}")

        if periodic:
            print("\n  【非每日資料】:")
            for r in periodic:
                print(f"    - {r['name']} ({r['dataset']}): {r['update_time']}")

        if no_data:
            print("\n  【無法取得】:")
            for r in no_data:
                print(f"    - {r['name']} ({r['dataset']}): {r['note']}")


if __name__ == "__main__":
    asyncio.run(main())
