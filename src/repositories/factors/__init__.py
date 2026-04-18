from src.repositories.factors.alpha158 import ALPHA158_FACTORS
from src.repositories.factors.enhanced import ENHANCED_FACTORS
from src.repositories.factors.interaction import INTERACTION_FACTORS

GENERIC_FACTORS = ALPHA158_FACTORS + INTERACTION_FACTORS + ENHANCED_FACTORS
ALL_FACTORS = GENERIC_FACTORS

__all__ = [
    "ALPHA158_FACTORS",
    "INTERACTION_FACTORS",
    "ENHANCED_FACTORS",
    "GENERIC_FACTORS",
    "ALL_FACTORS",
]
