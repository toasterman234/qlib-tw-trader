# English + US Market MVP Notes

This fork is being adapted from a Taiwan-first Qlib app into a dual-mode research workspace.

## Current switch

The backend now supports a market mode via environment variable:

```bash
APP_MARKET=us
```

Supported values:

- `tw` — original Taiwan-oriented mode
- `us` — US equity MVP mode

## What already changed

- Added market abstraction in `src/shared/market.py`
- Added curated US universe preset in `src/shared/us_universe.py`
- Made API metadata market-aware and English-friendly
- Made shared runtime constants market-aware
- Made universe sync dual-mode:
  - Taiwan mode keeps TWSE sync
  - US mode seeds a curated large-cap universe and enriches with Yahoo Finance market cap data
- Made default factor families market-aware:
  - US mode uses generic factor groups only
  - TW mode keeps Taiwan-specific factor families
- Rebranded the shell from `QLIB-TW` to `QLib Trader`

## US mode expectations

This is an MVP port, not full parity.

### Works reasonably well

- English shell direction
- US universe bootstrapping
- Generic factor seeding direction
- Backend market abstraction foundation

### Still Taiwan-first / incomplete

- Some sync endpoints still assume Taiwan-specific datasets
- Some Qlib-specific training/export code paths still need region-aware cleanup
- Some frontend pages and API comments still contain Chinese/Taiwan-specific wording
- TW-only datasets should eventually be hidden or replaced in US mode

## Recommended startup for US mode

```bash
export APP_MARKET=us
export APP_TIMEZONE=America/New_York
```

Optional universe override:

```bash
export US_UNIVERSE_FILE=/absolute/path/to/tickers.txt
```

One ticker per line, for example:

```text
AAPL
MSFT
NVDA
AMZN
GOOGL
META
```

## Practical next steps

1. Run the app with `APP_MARKET=us`
2. Trigger `/api/v1/universe/sync`
3. Verify the stock universe is populated with US tickers
4. Continue making sync, training, and dataset pages market-aware

## Goal of this fork

Make the project usable as an English-first US equity research MVP without destroying the original Taiwan path.
