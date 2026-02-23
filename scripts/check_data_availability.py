"""
檢查所有 Dataset 在 1/29 的資料可用性
優先使用 TWSE API，FinMind 獨有的才用 FinMind
"""

import asyncio
from datetime import date
import httpx

TODAY = date(2026, 1, 29)
TODAY_ROC = "1150129"  # 民國115年1月29日
TODAY_ROC_SLASH = "115/01/29"
TEST_STOCK = "2330"


def parse_roc_date(raw: str) -> str | None:
    """解析民國年日期，返回 YYYY-MM-DD"""
    if not raw:
        return None
    raw = str(raw).strip()

    # 格式: 1150128 (7位數)
    if raw.isdigit() and len(raw) == 7:
        year = int(raw[:3]) + 1911
        month = raw[3:5]
        day = raw[5:7]
        return f"{year}-{month}-{day}"

    # 格式: 115/01/28
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 3:
            year = int(parts[0]) + 1911
            return f"{year}-{parts[1]}-{parts[2]}"

    return raw


# TWSE OpenAPI 端點對應表
TWSE_DATASETS = {
    # === 技術面 ===
    "TaiwanStockPrice": {
        "twse_url": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        "twse_name": "每日收盤行情(全部)",
        "date_field": "日期",
        "note": "日K線",
    },
    "TaiwanStockPER": {
        "twse_url": "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL",
        "twse_name": "PER/PBR/殖利率(全部)",
        "date_field": "日期",
        "note": "本益比",
    },
    # === 籌碼面 ===
    "TaiwanStockInstitutionalInvestorsBuySell": {
        "twse_url": "https://openapi.twse.com.tw/v1/exchangeReport/TWT43U_ALL",
        "twse_name": "三大法人買賣超(全部)",
        "date_field": "日期",
        "note": "個股三大法人",
    },
    "TaiwanStockTotalInstitutionalInvestors": {
        "twse_url": "https://openapi.twse.com.tw/v1/exchangeReport/TWT38U_ALL",
        "twse_name": "三大法人彙總",
        "date_field": "日期",
        "note": "整體三大法人",
    },
    "TaiwanStockShareholding": {
        "twse_url": "https://openapi.twse.com.tw/v1/exchangeReport/MI_QFIIS_ALL",
        "twse_name": "外資持股(全部)",
        "date_field": "日期",
        "note": "外資持股",
    },
}

# 需要 follow redirect 的 TWSE 端點
TWSE_REDIRECT_DATASETS = {
    "TaiwanStockMarginPurchaseShortSale": {
        "twse_url": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json",
        "twse_name": "融資融券餘額",
        "note": "需解析特殊格式",
    },
    "TaiwanStockSecuritiesLending": {
        "twse_url": "https://www.twse.com.tw/rwd/zh/lending/t13sa900?response=json",
        "twse_name": "借券餘額",
        "note": "需解析特殊格式",
    },
}

# FinMind 獨有資料（需要不同參數）
FINMIND_DATASETS = {
    # 個股資料（需要 data_id）
    "TaiwanStockCashFlowsStatement": {
        "need_stock_id": True,
        "note": "現金流量表（季報）",
        "update_freq": "季度",
    },
    "TaiwanStockFinancialStatements": {
        "need_stock_id": True,
        "note": "綜合損益表（季報）",
        "update_freq": "季度",
    },
    "TaiwanStockBalanceSheet": {
        "need_stock_id": True,
        "note": "資產負債表（季報）",
        "update_freq": "季度",
    },
    "TaiwanStockDividend": {
        "need_stock_id": True,
        "note": "股利政策",
        "update_freq": "年度",
    },
    "TaiwanStockMonthRevenue": {
        "need_stock_id": True,
        "note": "月營收",
        "update_freq": "每月10日前",
    },
    # 期貨選擇權（需要不同的 data_id）
    "TaiwanFuturesDaily": {
        "need_stock_id": True,
        "data_id": "TX",  # 台指期
        "note": "期貨日成交",
        "update_freq": "每日",
    },
    "TaiwanOptionDaily": {
        "need_stock_id": True,
        "data_id": "TXO",  # 台指選
        "note": "選擇權日成交",
        "update_freq": "每日",
    },
    "TaiwanFuturesInstitutionalInvestors": {
        "need_stock_id": True,
        "data_id": "TX",
        "note": "期貨三大法人",
        "update_freq": "每日",
    },
    "TaiwanOptionInstitutionalInvestors": {
        "need_stock_id": True,
        "data_id": "TXO",
        "note": "選擇權三大法人",
        "update_freq": "每日",
    },
    # 總體經濟（不需要 data_id）
    "GoldPrice": {
        "need_stock_id": False,
        "note": "黃金價格",
        "update_freq": "每日",
    },
    "CrudeOilPrices": {
        "need_stock_id": True,
        "data_id": "WTI",
        "note": "原油價格",
        "update_freq": "每日(延遲)",
    },
    "TaiwanExchangeRate": {
        "need_stock_id": True,
        "data_id": "USD",
        "note": "匯率",
        "update_freq": "每日",
    },
    "InterestRate": {
        "need_stock_id": True,
        "data_id": "taiwan",
        "note": "央行利率",
        "update_freq": "不定期",
    },
    "GovernmentBonds": {
        "need_stock_id": True,
        "data_id": "United States 10-Year",
        "note": "美債",
        "update_freq": "每日",
    },
}


async def check_twse_openapi(client: httpx.AsyncClient, name: str, info: dict) -> dict:
    """檢查 TWSE OpenAPI 資料"""
    try:
        resp = await client.get(info["twse_url"], timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, list):
            return {
                "dataset": name,
                "source": "TWSE",
                "twse_endpoint": info["twse_name"],
                "has_today": False,
                "latest_date": None,
                "count": 0,
                "reason": "空資料或格式錯誤",
            }

        # 找日期
        date_field = info.get("date_field", "日期")
        first_row = data[0]
        raw_date = first_row.get(date_field, "")
        latest_date = parse_roc_date(raw_date)
        has_today = latest_date == "2026-01-29"

        return {
            "dataset": name,
            "source": "TWSE",
            "twse_endpoint": info["twse_name"],
            "has_today": has_today,
            "latest_date": latest_date,
            "count": len(data),
            "reason": None if has_today else f"最新: {latest_date}",
        }

    except Exception as e:
        return {
            "dataset": name,
            "source": "TWSE",
            "twse_endpoint": info["twse_name"],
            "has_today": False,
            "latest_date": None,
            "count": 0,
            "reason": f"錯誤: {str(e)[:60]}",
        }


async def check_twse_rwd(client: httpx.AsyncClient, name: str, info: dict) -> dict:
    """檢查 TWSE RWD API（舊版 API）"""
    try:
        resp = await client.get(info["twse_url"], timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()

        # RWD API 格式: {"stat": "OK", "date": "1150128", "data": [...]}
        stat = data.get("stat", "")
        if stat != "OK":
            return {
                "dataset": name,
                "source": "TWSE(RWD)",
                "twse_endpoint": info["twse_name"],
                "has_today": False,
                "latest_date": None,
                "count": 0,
                "reason": f"stat={stat}",
            }

        raw_date = data.get("date", "")
        latest_date = parse_roc_date(raw_date)
        has_today = latest_date == "2026-01-29"
        rows = data.get("data", [])

        return {
            "dataset": name,
            "source": "TWSE(RWD)",
            "twse_endpoint": info["twse_name"],
            "has_today": has_today,
            "latest_date": latest_date,
            "count": len(rows),
            "reason": None if has_today else f"最新: {latest_date}",
        }

    except Exception as e:
        return {
            "dataset": name,
            "source": "TWSE(RWD)",
            "twse_endpoint": info["twse_name"],
            "has_today": False,
            "latest_date": None,
            "count": 0,
            "reason": f"錯誤: {str(e)[:60]}",
        }


async def check_finmind(client: httpx.AsyncClient, name: str, info: dict) -> dict:
    """檢查 FinMind API 資料"""
    try:
        params = {
            "dataset": name,
            "start_date": "2026-01-01",
            "end_date": "2026-01-29",
        }

        if info.get("need_stock_id"):
            params["data_id"] = info.get("data_id", TEST_STOCK)

        resp = await client.get(
            "https://api.finmindtrade.com/api/v4/data",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") != 200:
            return {
                "dataset": name,
                "source": "FinMind",
                "twse_endpoint": f"獨有 ({info['note']})",
                "has_today": False,
                "latest_date": None,
                "count": 0,
                "reason": f"API: {result.get('msg', 'Unknown')[:40]}",
                "update_freq": info.get("update_freq", ""),
            }

        data = result.get("data", [])
        if not data:
            return {
                "dataset": name,
                "source": "FinMind",
                "twse_endpoint": f"獨有 ({info['note']})",
                "has_today": False,
                "latest_date": None,
                "count": 0,
                "reason": "無資料（可能延遲或季度資料）",
                "update_freq": info.get("update_freq", ""),
            }

        # 找最新日期（可能是 date 或其他欄位）
        dates = []
        for row in data:
            for field in ["date", "Date", "日期"]:
                if field in row and row[field]:
                    d = str(row[field])[:10]  # 取前10字元（去除時間）
                    dates.append(d)
                    break

        latest_date = max(dates) if dates else None
        has_today = latest_date == "2026-01-29"

        return {
            "dataset": name,
            "source": "FinMind",
            "twse_endpoint": f"獨有 ({info['note']})",
            "has_today": has_today,
            "latest_date": latest_date,
            "count": len(data),
            "reason": None if has_today else f"最新: {latest_date}",
            "update_freq": info.get("update_freq", ""),
        }

    except Exception as e:
        return {
            "dataset": name,
            "source": "FinMind",
            "twse_endpoint": f"獨有 ({info['note']})",
            "has_today": False,
            "latest_date": None,
            "count": 0,
            "reason": f"錯誤: {str(e)[:60]}",
            "update_freq": info.get("update_freq", ""),
        }


async def main():
    print(f"=== 資料可用性檢查 ({TODAY}) ===\n")

    results = []

    async with httpx.AsyncClient() as client:
        # 1. TWSE OpenAPI
        print("【1】TWSE OpenAPI...")
        for name, info in TWSE_DATASETS.items():
            print(f"  {name}...", end=" ", flush=True)
            result = await check_twse_openapi(client, name, info)
            results.append(result)
            status = "✅" if result["has_today"] else "❌"
            print(status)
            await asyncio.sleep(2)

        # 2. TWSE RWD (舊版)
        print("\n【2】TWSE RWD API（舊版）...")
        for name, info in TWSE_REDIRECT_DATASETS.items():
            print(f"  {name}...", end=" ", flush=True)
            result = await check_twse_rwd(client, name, info)
            results.append(result)
            status = "✅" if result["has_today"] else "❌"
            print(status)
            await asyncio.sleep(2)

        # 3. FinMind 獨有
        print("\n【3】FinMind 獨有資料...")
        for name, info in FINMIND_DATASETS.items():
            print(f"  {name}...", end=" ", flush=True)
            result = await check_finmind(client, name, info)
            results.append(result)
            status = "✅" if result["has_today"] else "❌"
            print(status)
            await asyncio.sleep(1)

    # 輸出結果表格
    print("\n" + "=" * 110)
    print(f"{'Dataset':<45} {'來源':<12} {'1/29':<5} {'最新日期':<12} {'筆數':<8} {'說明'}")
    print("=" * 110)

    for r in results:
        status = "✅" if r["has_today"] else "❌"
        latest = r["latest_date"] or "N/A"
        reason = r.get("reason", "") or ""
        count = r.get("count", 0)
        print(f"{r['dataset']:<45} {r['source']:<12} {status:<5} {latest:<12} {count:<8} {reason}")

    # 統計
    print("\n" + "=" * 110)
    has_today = [r for r in results if r["has_today"]]
    no_today = [r for r in results if not r["has_today"]]

    print(f"\n✅ 有 1/29 資料 ({len(has_today)}):")
    for r in has_today:
        print(f"  - {r['dataset']}")

    print(f"\n❌ 無 1/29 資料 ({len(no_today)}):")

    # 分類原因
    delayed = [r for r in no_today if "最新" in (r.get("reason") or "")]
    periodic = [r for r in no_today if r.get("update_freq") in ["季度", "年度", "每月10日前", "不定期"]]
    error = [r for r in no_today if "錯誤" in (r.get("reason") or "")]
    other = [r for r in no_today if r not in delayed and r not in periodic and r not in error]

    if delayed:
        print("\n  【延遲更新】:")
        for r in delayed:
            print(f"    - {r['dataset']}: {r.get('reason', '')}")

    if periodic:
        print("\n  【非每日更新】:")
        for r in periodic:
            print(f"    - {r['dataset']}: {r.get('update_freq', '')} ({r.get('reason', '')})")

    if error:
        print("\n  【API 錯誤】:")
        for r in error:
            print(f"    - {r['dataset']}: {r.get('reason', '')}")

    if other:
        print("\n  【其他】:")
        for r in other:
            print(f"    - {r['dataset']}: {r.get('reason', '')}")


if __name__ == "__main__":
    asyncio.run(main())
