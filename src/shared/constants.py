"""Training constants and label definitions."""

from zoneinfo import ZoneInfo

from src.shared.market import get_market

MARKET = get_market()

# === Timezone ===
TZ_APP = ZoneInfo(MARKET.timezone)
TZ_TAIPEI = TZ_APP  # backward-compatible alias for existing imports

# === Training windows ===
TRAIN_DAYS = 504
VALID_DAYS = 100
EMBARGO_DAYS = 7

# === Retrain cadence ===
RETRAIN_THRESHOLD_DAYS = 7

# === IC Deduplication ===
IC_DEDUP_THRESHOLD = 0.99

# === Quality monitoring ===
QUALITY_JACCARD_MIN = 0.3
QUALITY_IC_STD_MAX = 0.1
QUALITY_ICIR_MIN = 0.5

# === Label definition ===
LABEL_EXPR = "Ref($close, -3) / Ref($close, -1) - 1"
LABEL_DELAY_DAYS = 3
LABEL_EXTEND_DAYS = 10
LABEL_ENTRY_OFFSET = 1
LABEL_EXIT_OFFSET = 3

# === Factor lookback ===
LOOKBACK_DAYS = 400
