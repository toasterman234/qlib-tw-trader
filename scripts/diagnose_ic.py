"""診斷 IC 計算問題"""
import json
import pickle
import numpy as np
import pandas as pd
import os

# 設定環境變數避免 multiprocessing 問題
os.environ["JOBLIB_START_METHOD"] = "spawn"

def main():
    # 初始化 qlib（單線程模式）
    import qlib
    from qlib.config import REG_CN, C
    C.joblib_backend = "loky"
    C.kernels = 1  # 單線程
    qlib.init(provider_uri="data/qlib", region=REG_CN)
    from qlib.data import D

    # 讀取模型和因子
    with open("data/models/2022-07~2024-07/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("data/models/2022-07~2024-07/factors.json") as f:
        factors = json.load(f)

    factor_names = [f["name"] for f in factors]
    factor_exprs = [f["expression"] for f in factors]

    print("=== 模型資訊 ===")
    print(f"因子: {factor_names}")
    print(f"LightGBM 特徵數: {model.num_feature()}")
    print(f"LightGBM 樹數量: {model.num_trees()}")

    # 讀取股票清單
    with open("data/qlib/instruments/all.txt") as f:
        instruments = [line.strip().split()[0] for line in f if line.strip()]
    print(f"股票數: {len(instruments)}")

    # 載入驗證期資料
    from src.shared.constants import LABEL_EXPR
    label_expr = LABEL_EXPR
    all_fields = factor_exprs + [label_expr]
    all_names = factor_names + ["label"]

    print("\n=== 載入驗證資料 (2024-07-31 ~ 2025-01-01) ===")
    df = D.features(
        instruments=instruments,
        fields=all_fields,
        start_time="2024-07-31",
        end_time="2025-01-01",
    )
    df.columns = all_names

    print(f"原始形狀: {df.shape}")
    print(f"NaN 統計:\n{df.isna().sum()}")

    # 只保留完整資料
    df_clean = df.dropna()
    print(f"\n去除 NaN 後: {df_clean.shape}")

    if df_clean.empty:
        print("ERROR: 沒有完整資料！")
        return

    # 分離 X, y
    X = df_clean[factor_names]
    y = df_clean["label"]

    print(f"\n=== 特徵統計 ===")
    print(X.describe())

    print(f"\n=== 標籤統計 ===")
    print(y.describe())

    # Z-score 標準化（與訓練時相同）
    def zscore_by_date(data):
        return data.groupby(level="datetime", group_keys=False).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    X_norm = zscore_by_date(X)
    print(f"\n=== 標準化後特徵 ===")
    print(X_norm.describe())

    # 預測
    print(f"\n=== 模型預測 ===")
    predictions = model.predict(X_norm.values)
    print(f"預測值範圍: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"預測值標準差: {predictions.std():.6f}")

    # 建立 DataFrame
    pred_df = pd.DataFrame({
        "pred": predictions,
        "label": y.values,
    }, index=y.index)

    # 計算整體相關性
    overall_corr = pred_df["pred"].corr(pred_df["label"])
    print(f"\n整體相關係數: {overall_corr:.6f}")

    # 計算每日截面 IC
    def calc_ic(group):
        if len(group) < 5:
            return np.nan
        return group["pred"].corr(group["label"])

    daily_ic = pred_df.groupby(level="datetime").apply(calc_ic)
    daily_ic = daily_ic.dropna()

    print(f"\n=== 每日 IC 統計 ===")
    print(f"天數: {len(daily_ic)}")
    print(f"平均 IC: {daily_ic.mean():.6f}")
    print(f"IC 標準差: {daily_ic.std():.6f}")
    if daily_ic.std() > 0:
        print(f"ICIR: {daily_ic.mean() / daily_ic.std():.6f}")
    print(f"IC > 0 比例: {(daily_ic > 0).mean():.2%}")

    print(f"\n=== 每日 IC 樣本 ===")
    print(daily_ic.head(10))

if __name__ == "__main__":
    main()
