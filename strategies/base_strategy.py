"""Base strategy abstract class and Signal dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Signal:
    direction: SignalDirection
    strength: float  # 0.0 – 1.0
    confidence: float  # 0.0 – 1.0
    indicators_used: List[str] = field(default_factory=list)
    entry_reason: str = ""
    symbol: str = ""
    strategy_name: str = ""

    # Strategy-provided levels (optional — risk_engine uses these if set)
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    fvg_zone: Optional[Dict] = None

    @property
    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.FLAT and self.strength > 0

    def __repr__(self) -> str:
        return (
            f"Signal({self.direction.value} str={self.strength:.2f} "
            f"conf={self.confidence:.2f} '{self.entry_reason}')"
        )


class BaseStrategy(ABC):
    """Abstract base for all scalping strategies."""

    name: str = "BaseStrategy"

    def __init__(self, params: Optional[Dict] = None, **kwargs):
        self.params = params or self.default_params()

    @abstractmethod
    def compute_signal(self, df: pd.DataFrame, **kwargs: Any) -> Signal:
        """Compute a trading signal from the indicator-enriched DataFrame.

        Args:
            df: OHLCV DataFrame with all indicator columns populated.
            **kwargs: Optional extra data (df_m5, symbol_info, last_tick, settings).

        Returns:
            Signal with direction, strength, and confidence.
        """
        ...

    @abstractmethod
    def param_space(self, trial) -> Dict:
        """Define Optuna hyperparameter search space.

        Args:
            trial: An optuna.trial.Trial object.

        Returns:
            dict of parameter name -> suggested value.
        """
        ...

    @abstractmethod
    def required_indicators(self) -> List[str]:
        """Declare which DataFrame columns this strategy needs."""
        ...

    def default_params(self) -> Dict:
        """Return default parameters for this strategy."""
        return {}

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Check that all required indicator columns exist in the DataFrame."""
        required = self.required_indicators()
        missing = [col for col in required if col not in df.columns]
        return len(missing) == 0

    @staticmethod
    def flat_signal(reason: str = "") -> Signal:
        return Signal(
            direction=SignalDirection.FLAT,
            strength=0.0,
            confidence=0.0,
            entry_reason=reason,
        )
