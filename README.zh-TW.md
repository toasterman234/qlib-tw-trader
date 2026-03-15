[English](README.md) | [繁體中文](README.zh-TW.md)

# qlib-tw-trader

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Qlib](https://img.shields.io/badge/Qlib-Microsoft-0078D4)
![LightGBM](https://img.shields.io/badge/LightGBM-DoubleEnsemble-9ACD32)
![License](https://img.shields.io/badge/License-MIT-green)

台股量化交易研究平台，涵蓋從資料擷取到模型評估的完整流程。目標股票池為台股市值前 100 大。

<!-- TODO: 加入儀表板截圖 -->
<!-- ![Dashboard](docs/images/dashboard.png) -->

## Walk-Forward 回測結果

所有結果皆為 **樣本外**，來自 156 週（3 年）Walk-Forward 回測，每週重新訓練模型。無 lookahead bias — T 日交易僅使用 T-1 日特徵。

### LightGBM vs DoubleEnsemble

| 指標 | LightGBM | DoubleEnsemble | 變化 |
|------|:--------:|:--------------:|:----:|
| Backtest IC | 0.0107 | **0.0166** | +55% |
| IC 衰減（驗證期 → 回測期） | 78.5% | **56.0%** | -23pp |
| 最佳策略 Sharpe | 1.006 | **1.724** | +71% |
| 最佳超額報酬 | +0.2% | **+23.9%** | |
| 分位數單調性（rho） | 0.90 | **1.00** | 完美 |
| Spread t-stat | 1.25 | **2.30** | 顯著 |

### 最佳策略：HoldDrop(K=10, H=3, D=1)

| 指標 | 數值 |
|------|:----:|
| 年化報酬 | 55.1% |
| 年化超額 | +23.9% |
| Sharpe Ratio | 1.724 |
| 最大回撤 | 38.7% |
| 週換手率 | 9.9% |
| t-stat | 1.89 |

<details>
<summary>年度績效分解</summary>

| 年度 | 超額 | Sharpe | 勝率 | MaxDD |
|:----:|:----:|:------:|:----:|:-----:|
| 2023 | +80.0% | 2.96 | 54.5% | 18.0% |
| 2024 | -8.4% | 0.45 | 47.9% | 21.2% |
| 2025 | +19.6% | 1.62 | 51.3% | 29.4% |

</details>

### 市場 Regime 表現

| Regime | Mean IC | 超額（bps/日） | 勝率 |
|:------:|:-------:|:--------------:|:----:|
| 熊市 | **0.0354** | +10.0 | **55.2%** |
| 盤整 | 0.0146 | **+13.4** | 50.0% |
| 牛市 | -0.0002 | +1.8 | 50.2% |

模型在股票分化大的熊市中排名能力最強。

## 運作原理

```
T-1 收盤     T 開盤      T+2 收盤
  |            |             |
  計算特徵 --> 買入信號  -->  賣出    （2 天持有期）
```

**流程：**

1. **資料擷取** — 從 TWSE/FinMind/yfinance 取得 OHLCV、PER/PBR、三大法人、融資融券、月營收
2. **因子計算** — 約 300 個因子送入 Qlib：Alpha158 量價因子（109）、台股籌碼因子（107）、交互因子（50）、增強因子（37）
3. **模型訓練** — DoubleEnsemble（ICDM 2020）：K 個 LightGBM 子模型搭配迭代式樣本加權 + 排列重要性特徵選擇。模型內建特徵選擇，無需預先篩選
4. **Walk-Forward** — 每週重新訓練，504 天滾動訓練窗口、100 天驗證期、7 天 embargo
5. **預測** — 每日對前 100 大股票進行截面排名，預測 2 天報酬
6. **評估** — 多策略回測（TopK、TopKDrop、HoldDrop）、IC 分析、Regime 分解

## 功能特色

- **DoubleEnsemble（ICDM 2020）** — 內建樣本加權和特徵選擇的迭代集成。IC 比單一 LightGBM 高 55%。[[論文]](https://arxiv.org/abs/2010.01265)
- **約 300 個因子庫** — Alpha158 量價因子、台股三大法人籌碼因子、交互因子、增強因子（波動率 regime、動量、流動性、微結構）
- **Walk-Forward 回測** — 156 週樣本外測試，包含 IC 衰減分析、分位數展開、多策略比較
- **嚴格防止 Lookahead Bias** — T 日交易僅用 T-1 特徵。訓練/驗證集間 7 天 embargo。以股票代碼排序確保可重現性
- **多來源資料同步** — TWSE OpenAPI、FinMind、yfinance 自動同步，優先順序降級
- **全端儀表板** — React 18 + WebSocket 即時更新、模型評估圖表、持倉追蹤、因子管理
- **Optuna 超參數搜尋** — 貝葉斯最佳化模型參數

## 技術棧

| 層級 | 技術 |
|------|------|
| **後端** | FastAPI, SQLAlchemy 2.0, SQLite (WAL mode) |
| **前端** | React 18, Vite, TailwindCSS, Zustand, Recharts |
| **模型** | Qlib (Microsoft), LightGBM, DoubleEnsemble, Optuna |
| **資料** | TWSE OpenAPI, FinMind, yfinance |
| **即時通訊** | WebSocket |

## 快速開始

### Docker（推薦）

```bash
git clone https://github.com/Docat0209/qlib-tw-trader.git
cd qlib-tw-trader

cp .env.example .env
# 編輯 .env 填入 FinMind API token（免費：https://finmindtrade.com/）

docker compose up --build
```

- 前端：http://localhost:3000
- 後端 API：http://localhost:8000
- Swagger 文件：http://localhost:8000/docs

### 手動安裝

```bash
# 後端
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn src.interfaces.app:app --reload --port 8000

# 前端（另開終端）
cd frontend && npm install && npm run dev

# 初始化因子庫（約 300 個因子）
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## 系統架構

```
┌──────────────────────────────────────────────┐
│              React Dashboard                  │
│        Vite + TailwindCSS + Zustand           │
└────────────────────┬─────────────────────────┘
                     │ HTTP / WebSocket
┌────────────────────▼─────────────────────────┐
│              FastAPI Backend                   │
│                                               │
│  ┌───────────┐  ┌──────────────────────────┐  │
│  │ Adapters  │  │ Services                 │  │
│  │  TWSE     │  │  ModelTrainer            │  │
│  │  FinMind  │  │  WalkForwardBacktester   │  │
│  │  yfinance │  │  Predictor               │  │
│  └─────┬─────┘  │  QlibExporter            │  │
│        │        │  DoubleEnsemble          │  │
│        │        └────────────┬─────────────┘  │
│  ┌─────▼─────────────────────▼─────────────┐  │
│  │   SQLite (WAL) + Qlib .bin exports      │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
```

### 專案結構

```
qlib-tw-trader/
├── src/
│   ├── adapters/           # TWSE, FinMind, yfinance 資料客戶端
│   ├── interfaces/         # FastAPI 路由、Schema、WebSocket
│   ├── repositories/       # 資料庫存取 + 因子定義（約 300 個）
│   ├── services/           # 訓練、預測、回測、Qlib 導出
│   └── shared/             # 常數、型別、週曆工具
├── frontend/               # React 18 SPA
├── tests/                  # pytest 測試套件（28 個測試）
├── scripts/                # 分析腳本（模型評估、時段分析、IC）
└── data/                   # 資料庫 + 模型 + Qlib 導出（gitignored）
```

## 儀表板頁面

| 頁面 | 說明 |
|------|------|
| **Dashboard** | 系統總覽、模型統計、快速操作 |
| **Factors** | 因子庫 CRUD、啟用/停用、去重複 |
| **Training** | 週曆、批量訓練、模型管理 |
| **Evaluation** | 綜合 IC 分析、權益曲線、因子重要性、CSV/JSON 匯出 |
| **Backtest** | Walk-Forward 結果、逐週 IC、策略比較 |
| **Quality** | IC 穩定性監控、Jaccard 相似度、ICIR 追蹤 |
| **Predictions** | 今日信號、Top-K 選股 |
| **Positions** | 當前持倉、交易歷史、持倉時間軸 |
| **Datasets** | 資料來源覆蓋率、同步狀態、新鮮度檢查 |

## 資料來源

| 優先序 | 來源 | 涵蓋範圍 |
|:------:|------|----------|
| 1 | TWSE OpenAPI | OHLCV、PER/PBR（每日 17:30 後可用） |
| 2 | FinMind | 三大法人、融資融券、月營收（免費 600 次/時） |
| 3 | yfinance | 還原收盤價（無限制） |

## 參考文獻

- **DoubleEnsemble**: Chuheng Zhang et al. "DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis." ICDM 2020. [[論文]](https://arxiv.org/abs/2010.01265)
- **Qlib**: Yang et al. "Qlib: An AI-oriented Quantitative Investment Platform." 2020. [[repo]](https://github.com/microsoft/qlib)

## 貢獻

歡迎貢獻。詳見 [CONTRIBUTING.md](CONTRIBUTING.md) 了解環境設定與開發流程。

## 授權

MIT。詳見 [LICENSE](LICENSE)。
