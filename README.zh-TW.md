[English](README.md) | [繁體中文](README.zh-TW.md)

# qlib-tw-trader

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Qlib](https://img.shields.io/badge/Qlib-Microsoft-0078D4)
![LightGBM](https://img.shields.io/badge/LightGBM-DoubleEnsemble-9ACD32)
![License](https://img.shields.io/badge/License-MIT-green)

> 台股量化交易系統 -- DoubleEnsemble 模型、300+ 因子、Walk-Forward 回測與全端儀表板。

研究級的台股量化交易平台，針對台灣市值前 100 大個股。涵蓋完整流程：從 TWSE/FinMind/yfinance 資料擷取、因子工程（約 300 個因子）、Optuna 超參數搜尋的模型訓練、156 週 Walk-Forward 回測，到 React 監控儀表板 -- 全程嚴格防止前視偏差。

## 關鍵績效（156 週 Walk-Forward）

| 指標 | LightGBM | DoubleEnsemble | 改善幅度 |
|------|----------|----------------|---------|
| **IC** | 0.0107 | 0.0166 | +55% |
| **IC Decay** | 78.5% | 56.0% | -23pp |
| **最佳策略 Sharpe** | 1.006 | 1.724 | +71% |
| **信號單調性** | 0.90 | 1.00 | 完美 |

## 功能特性

- **DoubleEnsemble 模型（ICDM 2020）** -- K 個 LightGBM 子模型的迭代集成，結合樣本重新加權與特徵選擇。取代單一 LightGBM，IC 提升 55%、Sharpe 提升 71%。
- **約 300 個因子庫** -- Alpha158 價量因子（109）、台股籌碼面因子（107）、交互因子（50）、增強因子（37），涵蓋波動率 regime、動量、流動性、估值與微結構。
- **Walk-Forward 回測** -- 156 週滾動窗口，每週重新訓練模型，IC Decay 分析與多策略比較。
- **IC 增量選擇** -- 逐步加入因子搭配去重複（0.99 閾值），從約 300 個候選因子中精選 30-50 個有效因子。
- **前視偏差防護** -- T 日交易僅使用 T-1 日特徵。Label 定義為 2 日前瞻報酬，訓練與驗證集間設 7 日隔離期。
- **9 種資料來源** -- 自動同步 OHLCV、還原收盤價、PER/PBR、三大法人買賣超、融資融券、月營收，來源為 TWSE、FinMind、yfinance。
- **全端儀表板** -- React 18 + Vite + TailwindCSS 前端，透過 WebSocket 即時更新、權益曲線圖表與週曆導航。
- **Optuna 超參數搜尋** -- 以 IC 為目標的貝葉斯最佳化，每次訓練執行 50 組試驗。

## 技術棧

| 層級 | 技術 |
|------|------|
| **後端** | FastAPI, SQLAlchemy 2.0, SQLite（WAL 模式）|
| **前端** | React 18, Vite, TailwindCSS, Zustand, Recharts, Lightweight Charts |
| **模型** | Qlib（Microsoft）, LightGBM, DoubleEnsemble（ICDM 2020）, Optuna |
| **回測** | backtrader |
| **資料** | TWSE OpenAPI, FinMind, yfinance, DVC + Google Drive |
| **即時通訊** | WebSocket |

## Docker 快速開始

最快的啟動方式：

```bash
# Clone
git clone https://github.com/your-username/qlib-tw-trader.git
cd qlib-tw-trader

# 設定環境變數
cp .env.example .env
# 編輯 .env，填入 FinMind API Token（選填）

# 啟動所有服務
docker compose up --build
```

- 後端 API：http://localhost:8000
- 前端：http://localhost:3000
- Swagger 文件：http://localhost:8000/docs

> **注意**：SQLite 資料庫（`data/data.db`）和訓練好的模型（`data/models/`）以 volume 掛載。若透過 DVC 有預建資料，請先放入 `data/` 目錄再啟動。

## 手動安裝

### 前置需求

- Python 3.12+
- Node.js 20+
- Git

### 後端

```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env

# 啟動伺服器
uvicorn src.interfaces.app:app --reload --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

前端開發伺服器運行在 http://localhost:5173，API 請求自動代理到後端。

### 下載預建資料（選填）

若已設定 Google Drive Desktop：

```bash
python -m dvc pull
```

### 初始化因子

```bash
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## 系統架構

```
                    ┌──────────────────────────────────────────┐
                    │             React Dashboard               │
                    │          Vite + TailwindCSS + Zustand     │
                    └─────────────────┬────────────────────────┘
                                      │ HTTP / WebSocket
                    ┌─────────────────▼────────────────────────┐
                    │             FastAPI Backend                │
                    │                                           │
                    │  ┌───────────┐  ┌──────────────────────┐  │
                    │  │ Adapters  │  │ Services             │  │
                    │  │  TWSE     │  │  Model Trainer       │  │
                    │  │  FinMind  │  │  Predictor           │  │
                    │  │  yfinance │  │  Walk-Forward Tester │  │
                    │  └─────┬─────┘  │  Factor Selection    │  │
                    │        │        │  Qlib Exporter       │  │
                    │        │        └──────────┬───────────┘  │
                    │  ┌─────▼───────────────────▼───────────┐  │
                    │  │  SQLite (WAL) + Qlib .bin exports   │  │
                    │  └────────────────────────────────────┘  │
                    └──────────────────────────────────────────┘
```

### 專案結構

```
qlib-tw-trader/
├── src/
│   ├── adapters/           # 外部資料來源客戶端
│   ├── flows/              # 工作流程編排
│   ├── interfaces/         # FastAPI 路由、Schema、WebSocket
│   ├── repositories/       # 資料庫存取層與因子定義
│   ├── services/           # 商業邏輯（訓練、預測、回測）
│   └── shared/             # 共用型別與工具
├── frontend/               # React SPA
├── tests/                  # pytest 測試
├── scripts/                # 可重現的分析腳本
├── data/                   # 資料庫、模型、qlib 匯出（gitignored）
└── docker-compose.yml      # 容器編排
```

## API 文件

後端啟動後可存取互動式 API 文件：

- **Swagger UI**：http://localhost:8000/docs
- **ReDoc**：http://localhost:8000/redoc

## 資料來源

| 優先序 | 來源 | 說明 |
|--------|------|------|
| 1 | TWSE OpenAPI | 官方資料，當日 17:30 後可用 |
| 2 | FinMind | 第三方整合，免費方案 600 次/時限制 |
| 3 | yfinance | 還原股價，無速率限制 |

## 開發藍圖

- [ ] 增量學習 -- 每日微調模型權重
- [ ] 排程系統 -- 每日自動同步 + 訓練流程
- [ ] 動態策略參數 -- 自適應 TopK 與持股週期
- [ ] 多市場支援 -- 擴展至台股以外的市場

## 貢獻

歡迎貢獻！請閱讀 [CONTRIBUTING.md](CONTRIBUTING.md) 了解安裝步驟、程式碼規範與開發流程。

## 授權

本專案採用 MIT 授權。詳見 [LICENSE](LICENSE)。
