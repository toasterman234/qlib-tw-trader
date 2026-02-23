"""檢查訓練過程"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

# 初始化 qlib
import qlib
from qlib.config import REG_CN, C
C.kernels = 1
qlib.init(provider_uri="data/qlib", region=REG_CN)
from qlib.data import D

def main():
    # 讀取因子
    with open("data/models/2022-07~2024-07/factors.json") as f:
        factors = json.load(f)

    factor_names = [f["name"] for f in factors]
    factor_exprs = [f["expression"] for f in factors]

    # 讀取股票清單
    with open("data/qlib/instruments/all.txt") as f:
        instruments = [line.strip().split()[0] for line in f if line.strip()]

    # 訓練期和驗證期
    train_start, train_end = "2022-07-05", "2024-07-30"
    valid_start, valid_end = "2024-07-31", "2025-01-01"

    # 載入資料
    label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
    all_fields = factor_exprs + [label_expr]
    all_names = factor_names + ["label"]

    print("=== 載入完整資料 ===")
    df = D.features(
        instruments=instruments,
        fields=all_fields,
        start_time=train_start,
        end_time=valid_end,
    )
    df.columns = all_names
    df = df.dropna()
    print(f"資料形狀: {df.shape}")

    # 分割訓練/驗證
    train_mask = df.index.get_level_values("datetime") <= pd.Timestamp(train_end)
    valid_mask = df.index.get_level_values("datetime") >= pd.Timestamp(valid_start)

    train_df = df[train_mask]
    valid_df = df[valid_mask]

    print(f"訓練集: {train_df.shape}")
    print(f"驗證集: {valid_df.shape}")

    X_train = train_df[factor_names]
    y_train = train_df["label"]
    X_valid = valid_df[factor_names]
    y_valid = valid_df["label"]

    # Z-score 標準化
    def zscore_by_date(data):
        return data.groupby(level="datetime", group_keys=False).apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    X_train_norm = zscore_by_date(X_train)
    X_valid_norm = zscore_by_date(X_valid)

    print(f"\n=== 訓練前檢查 ===")
    print(f"X_train mean: {X_train_norm.mean().mean():.6f}")
    print(f"X_train std: {X_train_norm.std().mean():.6f}")
    print(f"y_train mean: {y_train.mean():.6f}")
    print(f"y_train std: {y_train.std():.6f}")

    # 訓練 LightGBM（帶詳細日誌）
    print(f"\n=== 訓練 LightGBM ===")
    train_data = lgb.Dataset(X_train_norm.values, label=y_train.values)
    valid_data = lgb.Dataset(X_valid_norm.values, label=y_valid.values, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 1,  # 顯示訓練過程
        "seed": 42,
    }

    # 自訂 callback 記錄訓練過程
    eval_results = {}

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(eval_results),
        ],
    )

    print(f"\n=== 訓練結果 ===")
    print(f"最佳迭代: {model.best_iteration}")
    print(f"樹數量: {model.num_trees()}")
    print(f"訓練 MSE: {eval_results['train']['l2'][model.best_iteration-1]:.8f}")
    print(f"驗證 MSE: {eval_results['valid']['l2'][model.best_iteration-1]:.8f}")

    # 預測並計算 IC
    pred_train = model.predict(X_train_norm.values)
    pred_valid = model.predict(X_valid_norm.values)

    print(f"\n=== 預測統計 ===")
    print(f"訓練集預測 std: {pred_train.std():.6f}")
    print(f"驗證集預測 std: {pred_valid.std():.6f}")
    print(f"訓練集預測範圍: [{pred_train.min():.6f}, {pred_train.max():.6f}]")
    print(f"驗證集預測範圍: [{pred_valid.min():.6f}, {pred_valid.max():.6f}]")

    # 計算 IC
    def calc_daily_ic(pred, y, index):
        df = pd.DataFrame({"pred": pred, "label": y}, index=index)
        ic = df.groupby(level="datetime").apply(
            lambda g: g["pred"].corr(g["label"]) if len(g) >= 5 else np.nan
        ).dropna()
        return ic

    train_ic = calc_daily_ic(pred_train, y_train.values, y_train.index)
    valid_ic = calc_daily_ic(pred_valid, y_valid.values, y_valid.index)

    print(f"\n=== IC 統計 ===")
    print(f"訓練集平均 IC: {train_ic.mean():.6f} (std: {train_ic.std():.6f})")
    print(f"驗證集平均 IC: {valid_ic.mean():.6f} (std: {valid_ic.std():.6f})")

    # 特徵重要性
    print(f"\n=== 特徵重要性 ===")
    importance = pd.DataFrame({
        "feature": factor_names,
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)
    print(importance)

if __name__ == "__main__":
    main()
