"""MACD Zero-Cross Scalper — histogram crosses zero with EMA/VWAP filters."""

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, SignalDirection


class MACDZeroCrossScalper(BaseStrategy):
    name = "MACD_ZeroCross"

    def default_params(self) -> Dict:
        return {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "require_ema_align": True,
        }

    def required_indicators(self) -> List[str]:
        return ["macd_hist", "macd", "macd_signal", "ema_8", "ema_21", "vwap", "close"]

    def param_space(self, trial) -> Dict:
        return {
            "macd_fast": trial.suggest_int("macd_fast", 8, 16),
            "macd_slow": trial.suggest_int("macd_slow", 20, 32),
            "macd_signal": trial.suggest_int("macd_signal", 6, 12),
        }

    def compute_signal(self, df: pd.DataFrame, **kwargs) -> Signal:
        if not self.validate_dataframe(df) or len(df) < 30:
            return self.flat_signal("Insufficient data")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        hist = row["macd_hist"]
        prev_hist = prev["macd_hist"]
        close = row["close"]
        ema8 = row["ema_8"]
        ema21 = row["ema_21"]
        vwap = row["vwap"]

        if any(np.isnan(v) for v in [hist, prev_hist, close, ema8, ema21]):
            return self.flat_signal("NaN values")

        # Bullish cross: histogram crosses from negative to positive
        if prev_hist < 0 and hist > 0:
            if not self.params.get("require_ema_align", True) or ema8 > ema21:
                vwap_filter = np.isnan(vwap) or close > vwap
                if vwap_filter:
                    strength = min(1.0, abs(hist) / (abs(prev_hist) + abs(hist) + 1e-10))
                    strength = max(0.5, strength)
                    return Signal(
                        direction=SignalDirection.LONG,
                        strength=round(strength, 3),
                        confidence=round(0.6 + 0.2 * (1 if ema8 > ema21 else 0), 3),
                        indicators_used=["macd_hist", "ema_8", "ema_21", "vwap"],
                        entry_reason=f"MACD hist crossed zero bullish, EMA8>21={ema8 > ema21}, above VWAP",
                        strategy_name=self.name,
                    )

        # Bearish cross: histogram crosses from positive to negative
        if prev_hist > 0 and hist < 0:
            if not self.params.get("require_ema_align", True) or ema8 < ema21:
                vwap_filter = np.isnan(vwap) or close < vwap
                if vwap_filter:
                    strength = min(1.0, abs(hist) / (abs(prev_hist) + abs(hist) + 1e-10))
                    strength = max(0.5, strength)
                    return Signal(
                        direction=SignalDirection.SHORT,
                        strength=round(strength, 3),
                        confidence=round(0.6 + 0.2 * (1 if ema8 < ema21 else 0), 3),
                        indicators_used=["macd_hist", "ema_8", "ema_21", "vwap"],
                        entry_reason=f"MACD hist crossed zero bearish, EMA8<21={ema8 < ema21}, below VWAP",
                        strategy_name=self.name,
                    )

        return self.flat_signal("No MACD zero cross detected")
