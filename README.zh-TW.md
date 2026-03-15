[English](README.md) | [繁體中文](README.zh-TW.md)

# qlib-tw-trader

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Qlib](https://img.shields.io/badge/Qlib-Microsoft-0078D4)
![LightGBM](https://img.shields.io/badge/LightGBM-DoubleEnsemble-9ACD32)
![License](https://img.shields.io/badge/License-MIT-green)

> 台股量化交易系統 — DoubleEnsemble 模型、300+ 因子、Walk-Forward 回測與全端儀表板。

研究級的台股量化交易平台，針對台灣市值前 100 大個股。涵蓋完整流程：從 TWSE/FinMind/yfinance 資料擷取、因子工程（約 300 個因子）、Optuna 超參數搜尋的模型訓練、156 週 Walk-Forward 回測，到 React 監控儀表板 — 全程嚴格防止前視偏差。

## 關鍵績效（156 週 Walk-Forward）

| 指標 | LightGBM | DoubleEnsemble | 改善幅度 |
|------|----------|----------------|---------|
| **IC** | 0.0107 | 0.0166 | +55% |
| **IC Decay** | 78.5% | 56.0% | -23pp |
| **最佳策略 Sharpe** | 1.006 | 1.724 | +71% |
| **信號單調性** | 0.90 | 1.00 | 完美 |

## 功能特性

- **DoubleEnsemble 模型（ICDM 2020）** — K 個 LightGBM 子模型的迭代集成，結合樣本重新加權與特徵選擇。取代單一 LightGBM，IC 提升 55%、Sharpe 提升 71%。
- **約 300 個因子庫** — Alpha158 價量因子（109）、台股籌碼面因子（107）、交互因子（50）、增強因子（37），涵蓋波動率 regime、動量、流動性、估值與微結構。
- **Walk-Forward 回測** — 156 週滾動窗口（2023W01–2025W51），每週重新訓練模型，IC Decay 分析與 backtrader 多策略比較。
- **IC 增量選擇** — 逐步加入因子搭配 RD-Agent 去重複（0.99 閾值），從約 300 個候選因子中精選 30–50 個有效因子。
- **前視偏差防護** — T 日交易僅使用 T-1 日特徵。Label 定義為 2 日前瞻報酬（`Ref($close,-3)/Ref($close,-1)-1`），訓練與驗證集間設 7 日隔離期。
- **9 種資料來源** — 自動同步 OHLCV、還原收盤價、PER/PBR、三大法人買賣超、融資融券、月營收（PIT），來源為 TWSE、FinMind、yfinance。
- **全端儀表板** — React 18 + Vite + TailwindCSS 前端，8 個頁面，透過 Zustand 的 WebSocket 即時更新、權益曲線圖表與週曆導航。
- **Optuna 超參數搜尋** — 以 IC 為目標的貝葉斯最佳化，每次訓練執行 50 組試驗。

## 技術棧

| 層級 | 技術 |
|------|------|
| **後端** | FastAPI, SQLAlchemy 2.0, SQLite（WAL 模式）|
| **前端** | React 18, Vite, TailwindCSS, Zustand, Recharts, Lightweight Charts |
| **模型** | Qlib（Microsoft）, LightGBM, DoubleEnsemble（ICDM 2020）, Optuna |
| **回測** | backtrader |
| **資料** | TWSE OpenAPI, FinMind, yfinance, DVC + Google Drive |
| **即時通訊** | WebSocket |

## 快速開始

### 安裝

```bash
# 後端
pip install -r requirements.txt

# 前端
cd frontend && npm install
```

### 下載資料（DVC）

```bash
# 需要先安裝 Google Drive Desktop 並登入
python -m dvc pull
```

### 啟動

```bash
# 後端（port 8000）
uvicorn src.interfaces.app:app --reload --port 8000

# 前端（port 5173）
cd frontend && npm run dev
```

### 初始化因子

```bash
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## 系統架構

```
┌─────────────────────┐     ┌──────────────────────────────┐
│  React 儀表板        │◄───►│  FastAPI 後端                 │
│  (Vite + Tailwind)   │ WS  │                              │
│                      │     │  ┌─────────┐  ┌───────────┐  │
│  8 個頁面：           │     │  │ Adapters │  │ Services   │  │
│  - 儀表板            │     │  │ TWSE     │  │ Trainer    │  │
│  - 模型訓練          │     │  │ FinMind  │  │ Predictor  │  │
│  - 因子管理          │     │  │ yfinance │  │ Backtester │  │
│  - 品質監控          │     │  └────┬─────┘  └─────┬──────┘  │
│  - 預測結果          │     │       │              │         │
│  - 回測分析          │     │  ┌────▼──────────────▼──────┐  │
│  - 投資組合          │     │  │  SQLite (WAL) + Qlib .bin │  │
│  - 設定             │     │  └──────────────────────────┘  │
└─────────────────────┘     └──────────────────────────────┘
```

## API 文檔

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 資料來源

| 優先序 | 來源 | 說明 |
|--------|------|------|
| 1 | TWSE OpenAPI | 官方資料，當日 17:30 後可用 |
| 2 | FinMind | 第三方整合，600 次/時限制 |
| 3 | yfinance | 還原股價 |

## 文檔導覽

| 文檔 | 說明 |
|------|------|
| [API 設計](docs/api-design.md) | API 端點參考 |
| [訓練系統](docs/training-system.md) | 訓練流程、IC 計算、參數配置 |
| [資料集](docs/datasets.md) | 資料來源與更新時間 |
| [原始欄位](docs/raw-fields.md) | 30 個 Qlib 欄位定義 |
| [模型績效分析](reports/model-performance-analysis.md) | 156 週 Walk-Forward 回測分析 |

## 開發藍圖

- [ ] 增量學習 — 每日微調模型權重
- [ ] 排程系統 — 每日自動同步 + 訓練流程
- [ ] 動態策略參數 — 自適應 TopK 與持股週期

## 授權

MIT
