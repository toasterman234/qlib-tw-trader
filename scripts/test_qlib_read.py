"""
使用 qlib 驗證導出的 .bin 資料
"""

import qlib
from qlib.data import D

# 初始化 qlib，使用導出的資料
qlib.init(provider_uri="data/qlib")

print("=== Qlib 初始化成功 ===\n")

# 測試讀取股票清單
instruments = D.instruments(market="all")
print(f"股票清單類型: {type(instruments)}")
print(f"股票清單: {instruments}\n")

# 測試讀取特徵
print("=== 讀取 2330 的特徵 ===")
fields = ["$close", "$adj_close", "$volume", "$foreign_buy", "$revenue"]

df = D.features(
    instruments=["2330"],
    fields=fields,
    start_time="2022-01-01",
    end_time="2025-01-01",
)

print(f"資料形狀: {df.shape}")
print(f"欄位: {df.columns.tolist()}")
print(f"\n前 10 筆:")
print(df.head(10))
print(f"\n後 10 筆:")
print(df.tail(10))

# 檢查 NaN 統計
print("\n=== NaN 統計 ===")
for col in df.columns:
    nan_count = df[col].isna().sum()
    total = len(df)
    print(f"{col}: {nan_count}/{total} NaN ({nan_count/total*100:.1f}%)")

# 測試多股票讀取
print("\n=== 讀取多股票 ===")
df_multi = D.features(
    instruments=["2330", "2317", "2454"],
    fields=["$close", "$volume"],
    start_time="2024-12-01",
    end_time="2024-12-31",
)
print(f"資料形狀: {df_multi.shape}")
print(df_multi)
