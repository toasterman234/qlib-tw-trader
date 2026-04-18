# QLib Trader (US Fork)

<p align="center">
  <strong>US equity research workspace built on Qlib, FastAPI, React, and LightGBM</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/Qlib-Microsoft-0078D4" alt="Qlib">
  <img src="https://img.shields.io/badge/LightGBM-DoubleEnsemble-9ACD32" alt="LightGBM">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

## What this fork is

This repository is a **US-only fork** of the original `qlib-tw-trader` project.

The goal of this fork is to turn the app into an English-first research workspace for **US equities only**.
That means:

- no Taiwan-market product path
- no FinMind requirement
- no Taiwan-specific setup in the default workflow
- no market toggle in the active runtime path

## Current status

This fork is in active conversion. It already includes:

- a US-only backend entrypoint
- a US sync router
- US trading calendar sync using `SPY`
- US OHLCV sync via Yahoo Finance
- US adjusted-close sync via Yahoo Finance
- a curated built-in US large-cap universe
- a simplified English-first datasets screen
- market-aware Qlib training/export foundations already pointed at the US path

What is still incomplete:

- several non-price US dataset families are placeholders for now
- some legacy Taiwan code still exists in the repository and is being removed
- runtime validation and CI coverage still need to be expanded for the US-only path

## Implemented US data path

### Live today

- **US daily OHLCV** via Yahoo Finance
- **US adjusted close** via Yahoo Finance
- **US trading calendar** inferred from `SPY`
- **US large-cap starter universe** shipped with the app

### Placeholder / planned

- valuation snapshots
- ownership / holdings
- institutional flow
- short-interest / margin-style datasets
- revenue / broader fundamentals

These placeholder dataset families are visible so the product surface is clear, but they are not yet fully implemented.

## Tech stack

| Layer | Technologies |
|-------|-------------|
| Backend | FastAPI, SQLAlchemy, SQLite |
| Frontend | React 18, Vite, TailwindCSS |
| Model | Qlib, LightGBM, DoubleEnsemble, Optuna |
| Market Data | Yahoo Finance |
| Realtime | WebSocket |

## Quick start

### Docker

```bash
git clone https://github.com/toasterman234/qlib-tw-trader.git
cd qlib-tw-trader
cp .env.example .env
docker compose up --build
```

Services:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Swagger: http://localhost:8000/docs

### Manual setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
cp .env.example .env
uvicorn src.interfaces.app:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Optional factor seeding:

```bash
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## Environment

Example `.env.example` now contains only US-oriented defaults.

Optional variables:

- `APP_TIMEZONE=America/New_York`
- `US_UNIVERSE_FILE=/absolute/path/to/tickers.txt`

`US_UNIVERSE_FILE` should contain one ticker per line, for example:

```text
AAPL
MSFT
NVDA
AMZN
GOOGL
META
```

## Product direction

This fork is being actively simplified into a cleaner US-equity application.

Planned cleanup still in progress:

- remove remaining Taiwan-only code paths from the repository
- remove old Taiwan sync/router implementations
- remove Taiwan-only factor families from the default product path
- add stronger testing and CI for the US-only workflow
- improve the model/backtest path for the US dataset set

## Architecture

Current active runtime path:

- FastAPI backend
- React frontend
- US sync router
- SQLite + local model artifacts
- Qlib export + training path

The app is intended to remain a **local-first research workspace**, not a hosted platform dependency.

## Important note on scope

This fork is not yet a fully finished production-grade US quant platform. It is a focused US-only conversion of the original project with a working price-data path and ongoing cleanup.

The guiding principle is:

> make the repo honest, usable, and clearly US-only first,
> then expand the deeper data/model/backtest coverage.

## License

MIT. See [LICENSE](LICENSE).
