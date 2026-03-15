[English](README.md) | [繁體中文](README.zh-TW.md)

# qlib-tw-trader

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Qlib](https://img.shields.io/badge/Qlib-Microsoft-0078D4)
![LightGBM](https://img.shields.io/badge/LightGBM-DoubleEnsemble-9ACD32)
![License](https://img.shields.io/badge/License-MIT-green)

> Quantitative trading system for Taiwan stocks -- DoubleEnsemble model, 300+ factors, Walk-Forward backtesting, and a full-stack dashboard.

A research-grade quantitative trading platform targeting Taiwan's top-100 market-cap stocks. It handles the full pipeline: data ingestion from TWSE/FinMind/yfinance, factor engineering (~300 factors), model training with Optuna hyperparameter search, 156-week Walk-Forward backtesting, and a React dashboard for monitoring -- all with strict lookahead bias prevention.

<!-- TODO: Add screenshot of dashboard -->
<!-- ![Dashboard Screenshot](docs/images/dashboard.png) -->

## Key Results (156-Week Walk-Forward)

| Metric | LightGBM | DoubleEnsemble | Improvement |
|--------|----------|----------------|-------------|
| **IC** | 0.0107 | 0.0166 | +55% |
| **IC Decay** | 78.5% | 56.0% | -23pp |
| **Best Strategy Sharpe** | 1.006 | 1.724 | +71% |
| **Signal Monotonicity** | 0.90 | 1.00 | Perfect |

## Features

- **DoubleEnsemble model (ICDM 2020)** -- Iterative ensemble of K LightGBM sub-models with sample reweighting and feature selection. +55% IC and +71% Sharpe over single LightGBM.
- **~300 factor library** -- Alpha158 price-volume (109), Taiwan institutional flow (107), cross-interaction (50), and enhanced factors (37) covering volatility regime, momentum, liquidity, valuation, and microstructure.
- **Walk-Forward backtesting** -- 156-week rolling window with per-week model retraining, IC Decay analysis, and multi-strategy comparison.
- **IC incremental selection** -- Stepwise factor addition with deduplication (0.99 threshold) to select 30-50 effective factors from ~300 candidates.
- **Lookahead bias prevention** -- Trade on T uses T-1 features only. Label defined as 2-day forward return with 7-day embargo between train and validation sets.
- **9 data sources** -- Auto-sync OHLCV, adjusted close, PER/PBR, institutional trades, margin trading, and monthly revenue from TWSE, FinMind, and yfinance.
- **Full-stack dashboard** -- React 18 + Vite + TailwindCSS with WebSocket real-time updates, equity curve charts, and week calendar navigation.
- **Optuna hyperparameter search** -- Bayesian optimization over DoubleEnsemble parameters (50 trials per training run).

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, SQLAlchemy 2.0, SQLite (WAL mode) |
| **Frontend** | React 18, Vite, TailwindCSS, Zustand, Recharts, Lightweight Charts |
| **Model** | Qlib (Microsoft), LightGBM, DoubleEnsemble (ICDM 2020), Optuna |
| **Backtesting** | backtrader |
| **Data** | TWSE OpenAPI, FinMind, yfinance, DVC + Google Drive |
| **Real-time** | WebSocket |

## Quick Start with Docker

The fastest way to get running:

```bash
# Clone the repository
git clone https://github.com/your-username/qlib-tw-trader.git
cd qlib-tw-trader

# Configure environment
cp .env.example .env
# Edit .env with your FinMind API token (optional)

# Start all services
docker compose up --build
```

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- Swagger docs: http://localhost:8000/docs

> **Note**: The SQLite database (`data/data.db`) and trained models (`data/models/`) are mounted as volumes. If you have pre-existing data via DVC, place them in the `data/` directory before starting.

## Manual Setup

### Prerequisites

- Python 3.12+
- Node.js 20+
- Git

### Backend

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Start the server
uvicorn src.interfaces.app:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server runs at http://localhost:5173 and proxies API requests to the backend.

### Download Pre-built Data (Optional)

If you have Google Drive Desktop configured:

```bash
python -m dvc pull
```

### Initialize Factors

```bash
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## Architecture

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

### Project Structure

```
qlib-tw-trader/
├── src/
│   ├── adapters/           # External data source clients
│   ├── flows/              # Orchestration workflows
│   ├── interfaces/         # FastAPI routes, schemas, WebSocket
│   ├── repositories/       # Database access and factor definitions
│   ├── services/           # Business logic (training, prediction, backtesting)
│   └── shared/             # Shared types and utilities
├── frontend/               # React SPA
├── tests/                  # pytest test suite
├── scripts/                # Reproducible analysis scripts
├── data/                   # Database, models, qlib exports (gitignored)
└── docker-compose.yml      # Container orchestration
```

## API Documentation

Interactive API documentation is available when the backend is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Data Sources

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | TWSE OpenAPI | Official exchange data, available after 17:30 daily |
| 2 | FinMind | Third-party aggregator, 600 requests/hour (free tier) |
| 3 | yfinance | Adjusted close prices, no rate limit |

## Roadmap

- [ ] Incremental learning -- daily model weight fine-tuning
- [ ] Scheduler -- automated daily sync + training pipeline
- [ ] Dynamic strategy parameters -- adaptive TopK and holding periods
- [ ] Multi-market support -- extend beyond Taiwan stocks

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding standards, and the development workflow.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
