[English](README.md) | [繁體中文](README.zh-TW.md)

# qlib-tw-trader

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Qlib](https://img.shields.io/badge/Qlib-Microsoft-0078D4)
![LightGBM](https://img.shields.io/badge/LightGBM-DoubleEnsemble-9ACD32)
![License](https://img.shields.io/badge/License-MIT-green)

A research-grade quantitative trading platform for Taiwan's top-100 market-cap stocks, covering the full pipeline from data ingestion to model evaluation.

<!-- TODO: Add dashboard screenshot -->
<!-- ![Dashboard](docs/images/dashboard.png) -->

## Walk-Forward Backtest Results

All results are **out-of-sample**, from a 156-week (3-year) Walk-Forward backtest with weekly model retraining. No lookahead bias -- T-day trades use only T-1 features.

### LightGBM vs DoubleEnsemble

| Metric | LightGBM | DoubleEnsemble | Change |
|--------|:--------:|:--------------:|:------:|
| Backtest IC | 0.0107 | **0.0166** | +55% |
| IC Decay (valid -> backtest) | 78.5% | **56.0%** | -23pp |
| Best Strategy Sharpe | 1.006 | **1.724** | +71% |
| Best Excess Return | +0.2% | **+23.9%** | |
| Quantile Monotonicity (rho) | 0.90 | **1.00** | Perfect |
| Spread t-stat | 1.25 | **2.30** | Significant |

### Best Strategy: HoldDrop(K=10, H=3, D=1)

| Metric | Value |
|--------|:-----:|
| Annualized Return | 55.1% |
| Annualized Excess | +23.9% |
| Sharpe Ratio | 1.724 |
| Max Drawdown | 38.7% |
| Turnover | 9.9%/week |
| t-stat | 1.89 |

<details>
<summary>Yearly breakdown</summary>

| Year | Excess | Sharpe | Win Rate | MaxDD |
|:----:|:------:|:------:|:--------:|:-----:|
| 2023 | +80.0% | 2.96 | 54.5% | 18.0% |
| 2024 | -8.4% | 0.45 | 47.9% | 21.2% |
| 2025 | +19.6% | 1.62 | 51.3% | 29.4% |

</details>

<details>
<summary>Comparison with Qlib official benchmarks</summary>

Our IC/ICIR is lower than [Qlib CSI300 benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks) (IC 0.052, ICIR 0.42), but **Sharpe is higher** (1.72 vs 1.34). This reflects structural differences:

| Factor | Qlib Benchmark | This Project |
|--------|:--------------:|:------------:|
| Universe | CSI300 (300 stocks) | TW100 (100 stocks) |
| Market | China A-shares (retail-driven) | Taiwan (institutional-heavy) |
| Holding | top-50 | top-10 (concentrated) |
| Label | 1-day return | 2-day return |

Direct IC comparison is not meaningful across these conditions. The relevant comparison is the **relative improvement** from LightGBM to DoubleEnsemble within the same setup.

</details>

### Market Regime Performance

| Regime | Mean IC | Excess (bps/day) | Win Rate |
|:------:|:-------:|:-----------------:|:--------:|
| Bear | **0.0354** | +10.0 | **55.2%** |
| Sideways | 0.0146 | **+13.4** | 50.0% |
| Bull | -0.0002 | +1.8 | 50.2% |

The model's ranking ability is strongest in bear markets where stock dispersion is high.

## How It Works

<p align="center">
  <img src="docs/images/pipeline.svg" alt="Trading Pipeline" width="600">
</p>

## Features

- **DoubleEnsemble (ICDM 2020)** -- Iterative ensemble with built-in sample reweighting and feature selection. +55% IC over single LightGBM. [[paper]](https://arxiv.org/abs/2010.01265)
- **~300 factor library** -- Alpha158 OHLCV factors, Taiwan institutional flow (foreign/trust/dealer net buy, margin), cross-interaction terms, and enhanced factors (volatility regime, momentum, liquidity, microstructure)
- **Walk-Forward backtesting** -- 156-week out-of-sample test with IC Decay analysis, quantile spread, and multi-strategy comparison
- **Strict lookahead bias prevention** -- T-day trades use T-1 features only. 7-day embargo between train/validation sets. Tie-breaking by stock symbol for reproducibility
- **Multi-source data sync** -- Auto-sync from TWSE OpenAPI, FinMind, and yfinance with priority fallback
- **Full-stack dashboard** -- React 18 + WebSocket real-time updates, model evaluation charts, position tracking, factor management
- **Optuna hyperparameter search** -- Bayesian optimization over model parameters

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, SQLAlchemy 2.0, SQLite (WAL mode) |
| **Frontend** | React 18, Vite, TailwindCSS, Zustand, Recharts |
| **Model** | Qlib (Microsoft), LightGBM, DoubleEnsemble, Optuna |
| **Data** | TWSE OpenAPI, FinMind, yfinance |
| **Real-time** | WebSocket |

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/Docat0209/qlib-tw-trader.git
cd qlib-tw-trader

cp .env.example .env
# Edit .env with your FinMind API token (free: https://finmindtrade.com/)

docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Swagger: http://localhost:8000/docs

### Manual Setup

```bash
# Backend
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn src.interfaces.app:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev

# Seed the factor library (~300 factors)
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## Architecture

<p align="center">
  <img src="docs/images/architecture.svg" alt="System Architecture" width="800">
</p>

### Project Structure

```
qlib-tw-trader/
├── src/
│   ├── adapters/           # TWSE, FinMind, yfinance data clients
│   ├── interfaces/         # FastAPI routes, schemas, WebSocket
│   ├── repositories/       # Database access + factor definitions (~300)
│   ├── services/           # Training, prediction, backtesting, Qlib export
│   └── shared/             # Constants, types, week utilities
├── frontend/               # React 18 SPA
├── tests/                  # pytest suite (28 tests)
├── scripts/                # Analysis scripts (model eval, timing, IC)
└── data/                   # Database + models + Qlib exports (gitignored)
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Dashboard** | System overview, model stats, quick actions |
| **Factors** | Factor library CRUD, enable/disable, deduplication |
| **Training** | Week calendar, batch training, model management |
| **Evaluation** | Aggregate IC analysis, equity curves, factor importance, CSV/JSON export |
| **Backtest** | Walk-Forward results, per-week IC, strategy comparison |
| **Quality** | IC stability monitoring, Jaccard similarity, ICIR tracking |
| **Predictions** | Today's signals, Top-K stock picks |
| **Positions** | Current holdings, trade history, holdings timeline |
| **Datasets** | Data source coverage, sync status, freshness checks |

## API

Interactive documentation at http://localhost:8000/docs when running.

Key endpoints:

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/sync/all` | Sync all data sources |
| `POST /api/v1/factors/seed` | Initialize ~300 factors |
| `POST /api/v1/models/train` | Train model for a specific week |
| `POST /api/v1/backtest/walk-forward` | Run Walk-Forward backtest |
| `GET /api/v1/backtest/walk-forward/summary` | Aggregated backtest metrics |
| `POST /api/v1/predictions/today/generate` | Generate today's predictions |
| `GET /api/v1/predictions/today` | Get current stock picks |
| `GET /api/v1/predictions/history` | Prediction history |

## Data Sources

| Priority | Source | Coverage |
|:--------:|--------|----------|
| 1 | TWSE OpenAPI | OHLCV, PER/PBR (available after 17:30 daily) |
| 2 | FinMind | Institutional trades, margin, monthly revenue (600 req/hr free) |
| 3 | yfinance | Adjusted close prices (no rate limit) |

## References

- **DoubleEnsemble**: Chuheng Zhang et al. "DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis." ICDM 2020. [[paper]](https://arxiv.org/abs/2010.01265)
- **Qlib**: Yang et al. "Qlib: An AI-oriented Quantitative Investment Platform." 2020. [[repo]](https://github.com/microsoft/qlib)
- **Alpha158**: Qlib built-in factor set. [[docs]](https://qlib.readthedocs.io/en/latest/component/data.html)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and development workflow.

## License

MIT. See [LICENSE](LICENSE).
