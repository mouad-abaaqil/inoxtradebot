"""Volume Surge Momentum — massive volume spike with price breakout."""

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, SignalDirection


class VolumeSurgeMomentum(BaseStrategy):
    name = "Volume_Surge"

    def default_params(self) -> Dict:
        return {
            "vol_multiplier": 3.0,
            "breakout_bars": 5,
        }

    def required_indicators(self) -> List[str]:
        return ["vol_ratio", "vol_sma_20", "macd_hist", "close", "high", "low"]

    def param_space(self, trial) -> Dict:
        return {
            "vol_multiplier": trial.suggest_float("vol_multiplier", 2.0, 5.0),
            "breakout_bars": trial.suggest_int("breakout_bars", 3, 10),
        }

    def compute_signal(self, df: pd.DataFrame, **kwargs) -> Signal:
        if not self.validate_dataframe(df) or len(df) < 20:
            return self.flat_signal("Insufficient data")

        row = df.iloc[-1]
        vol_ratio = row["vol_ratio"]
        macd_hist = row["macd_hist"]
        close = row["close"]
        vol_mult = self.params["vol_multiplier"]
        n_bars = self.params["breakout_bars"]

        if any(np.isnan(v) for v in [vol_ratio, macd_hist, close]):
            return self.flat_signal("NaN values")

        # Volume surge check
        if vol_ratio < vol_mult:
            return self.flat_signal(f"Volume ratio {vol_ratio:.1f} < {vol_mult}")

        recent = df.tail(n_bars + 1).head(n_bars)  # last N bars before current
        prev_high = recent["high"].max()
        prev_low = recent["low"].min()

        # Bullish surge: close above previous N-bar high + MACD positive
        if close > prev_high and macd_hist > 0:
            strength = min(1.0, 0.5 + vol_ratio / 8.0)
            return Signal(
                direction=SignalDirection.LONG,
                strength=round(strength, 3),
                confidence=round(min(0.85, 0.5 + vol_ratio / 10.0), 3),
                indicators_used=["vol_ratio", "macd_hist", "high"],
                entry_reason=f"Volume surge LONG, vol={vol_ratio:.1f}x, breakout above {n_bars}-bar high, MACD+",
                strategy_name=self.name,
            )

        # Bearish surge: close below previous N-bar low + MACD negative
        if close < prev_low and macd_hist < 0:
            strength = min(1.0, 0.5 + vol_ratio / 8.0)
            return Signal(
                direction=SignalDirection.SHORT,
                strength=round(strength, 3),
                confidence=round(min(0.85, 0.5 + vol_ratio / 10.0), 3),
                indicators_used=["vol_ratio", "macd_hist", "low"],
                entry_reason=f"Volume surge SHORT, vol={vol_ratio:.1f}x, breakout below {n_bars}-bar low, MACD-",
                strategy_name=self.name,
            )

        return self.flat_signal("Volume surge but no price breakout confirmation")
