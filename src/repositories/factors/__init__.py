"""Factor definition exports."""

from src.repositories.factors.alpha158 import ALPHA158_FACTORS
from src.repositories.factors.enhanced import ENHANCED_FACTORS
from src.repositories.factors.interaction import INTERACTION_FACTORS
from src.repositories.factors.taiwan_chips import TAIWAN_CHIPS_FACTORS
from src.shared.market import market_is_us

GENERIC_FACTORS = ALPHA158_FACTORS + INTERACTION_FACTORS + ENHANCED_FACTORS
TW_FACTORS = GENERIC_FACTORS + TAIWAN_CHIPS_FACTORS
ALL_FACTORS = GENERIC_FACTORS if market_is_us() else TW_FACTORS

__all__ = [
    "ALPHA158_FACTORS",
    "TAIWAN_CHIPS_FACTORS",
    "INTERACTION_FACTORS",
    "ENHANCED_FACTORS",
    "GENERIC_FACTORS",
    "TW_FACTORS",
    "ALL_FACTORS",
]
