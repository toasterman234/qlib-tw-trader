# qlib-tw-trader

台灣股票交易與預測系統

## 專案目標

使用 qlib 進行台股預測，產生每日交易訊號。

## 快速指令

```bash
# 啟動後端
uvicorn src.interfaces.app:app --reload --port 8000

# 啟動前端
cd frontend && npm run dev

# Seed 因子
curl -X POST http://localhost:8000/api/v1/factors/seed

# 導出 qlib
curl -X POST http://localhost:8000/api/v1/qlib/export/sync \
  -H "Content-Type: application/json" \
  -d '{"start_date":"2022-01-01","end_date":"2025-01-01"}'
```

## 關鍵規則

- 時間：`Asia/Taipei` (UTC+8)
- 股票池：市值前 100 大（排除 ETF、KY）
- 因子挑選：IC 增量選擇法
- **禁止自行啟動伺服器**
- **資料庫路徑**：`data/data.db`（不是 `data/trader.db`）

## DVC 資料版本控制

使用 DVC 管理大型檔案，透過 Google Drive Desktop 自動同步雲端。

### 追蹤的檔案

| 檔案 | 大小 | 說明 |
|------|------|------|
| `data/data.db` | ~131MB | SQLite 資料庫 |
| `data/models/` | ~19MB | 訓練好的模型（154 週） |

### 常用指令

```bash
# 上傳變更（訓練新模型或更新資料庫後）
python -m dvc push

# 下載資料（在新電腦或 clone 後）
python -m dvc pull

# 檢查狀態
python -m dvc status
```

### 雲端位置

- 本地：`G:\My Drive\qlib-tw-trader`
- Google Drive Desktop 自動同步至雲端

### 新電腦設定

1. 安裝 Google Drive Desktop 並登入
2. 確認同步資料夾掛載為 `G:\My Drive`
3. Clone repo 後執行 `python -m dvc pull`

## Qlib 資料架構

**重要**：Qlib `.bin` 檔案是從資料庫動態導出的，不是靜態資料。

### 資料流程

```
資料庫 (stock_daily, etc.)
    ↓
QlibExporter.export()  ← 指定日期範圍
    ↓
data/qlib/*.bin
    ↓
ModelTrainer / WalkForwardBacktester 使用
```

### 日期範圍判斷

- **正確**：查詢資料庫 `stock_daily` 表的 `MIN(date)` / `MAX(date)`
- **錯誤**：讀取現有 qlib 檔案的日期範圍（可能是舊的導出）

### 訓練/回測前

訓練和回測前，系統會自動調用 `QlibExporter` 導出所需日期範圍的資料：

```python
# 範例：訓練時自動導出
exporter = QlibExporter(session)
exporter.export(ExportConfig(
    start_date=train_start - lookback_days,  # 預留因子計算緩衝
    end_date=valid_end,
))
```

### 模型命名

格式：`YYYYMM-{hash}`
- `YYYYMM`：valid_end 的年月
- `hash`：6 位 MD5（基於 run_id + valid_end + factor_count）

例：`202502-a1b2c3`

## DoubleEnsemble 模型

### 演算法（ICDM 2020）

使用 K 個 LightGBM 子模型的迭代集成，每輪之間進行：
- **Sample Reweighting (SR)**：根據 learning trajectory 調高難學樣本的權重
- **Feature Selection (FS)**：permutation importance 過濾低貢獻因子

### 為什麼用 DoubleEnsemble？

| 模型 | Qlib Alpha158 IC | 年化回報 |
|------|-------------------|----------|
| DoubleEnsemble | 0.0521 | 11.58% |
| LightGBM | 0.0448 | 9.08% |

內建 feature selection 解決了 LightGBM 單模型的 feature importance bias 問題。

### 實作

- `src/services/double_ensemble.py`：獨立 wrapper，相容 `lgb.Booster.predict()` 介面
- 不依賴 Qlib 的 `DEnsembleModel`（它要求 `DatasetH`）
- 增量更新：對每個子模型分別用 `lgb.train(init_model=...)` 更新

## 預測與交易時序

### Label 定義

```python
label_expr = "Ref($close, -3) / Ref($close, -1) - 1"
```

- T 日特徵 → 預測 T+1 收盤 ~ T+3 收盤的收益率（2-day return）
- 若要捕捉此收益，應在 T+1 開盤買入、T+3 收盤賣出

### 避免 Lookahead Bias

**Predictor:**
- `trade_date` = 預計交易日期（買入日）
- 系統自動使用 `trade_date - 1` 的特徵資料
- 返回 `feature_date`（實際使用的資料日期）

**Backtester:**
- 在 T 日交易時，使用 T-1 日的分數
- 預設使用 Open Price（更符合實際交易）

### 正確流程示例

要在 2/2 開盤買入：
1. 使用 2/1 收盤後的資料計算特徵
2. 模型預測 2/2→2/4 收益率（2-day return）
3. 2/2 開盤執行買入，2/4 收盤賣出

## Top-K 選股與 Tie-Breaking

### 已知問題：模型區分能力不足

目前模型可能產生大量相同分數的股票（例如 100 支股票只有 7 種分數）。
這導致 Top-K 選股結果取決於資料順序，而非模型判斷。

### Tie-Breaking 機制

為確保結果穩定可重現，Predictor 和 Backtester 都採用相同的排序邏輯：

```python
# 先按分數降序，再按股票代碼升序
df.sort_values(by=["score", "symbol"], ascending=[False, True]).head(top_k)
```

### 因子擴充（已實施）

已將因子從 30 個擴充至 ~300 個，包含：
- Alpha158 純 K 線因子（109 個）
- 台股籌碼因子（107 個）
- 交互因子（50 個）
- 增強因子（37 個）— 波動率 regime、長期動量、流動性、估值動態、市場微結構

需要重新訓練模型以驗證區分能力是否提升。

## Scripts（分析與實驗腳本）

`scripts/` 存放可重複執行的分析與實驗腳本（有別於 `sandbox/` 的臨時腳本）。

| 腳本 | 用途 | 執行方式 |
|------|------|---------|
| `evaluate_models.py` | 模型評估：策略回測 + 模型診斷 + 弱點分析，輸出到 `output/{hash}/` | `python scripts/evaluate_models.py` |
| `analyze_intraday_price.py` | 日內價格分析：各時段與收盤價偏差、最佳買賣時段 | `python scripts/analyze_intraday_price.py` |
| `simulate_timing.py` | 交易時段成本比較：早賣午買 vs 開盤 vs 收盤 | `python scripts/simulate_timing.py` |

## 資料來源

| 優先序 | 來源 | 限制 |
|--------|------|------|
| 1 | TWSE RWD | 當日 17:30 後 |
| 2 | FinMind | 600次/時 |
| 3 | yfinance | 無限制 |

**注意**：不要用 TWSE OpenAPI（`openapi.twse.com.tw`）
