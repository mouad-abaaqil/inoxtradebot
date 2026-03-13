"""RSI Reversal Scalper — RSI extremes + volume spike confirmation."""

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, SignalDirection


class RSIReversalScalper(BaseStrategy):
    name = "RSI_Reversal"

    def default_params(self) -> Dict:
        return {
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "vol_ratio_threshold": 1.8,
            "rsi_period": 14,
        }

    def required_indicators(self) -> List[str]:
        return ["rsi_14", "vol_ratio", "atr_14", "ema_8", "ema_21"]

    def param_space(self, trial) -> Dict:
        return {
            "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 85),
            "vol_ratio_threshold": trial.suggest_float("vol_ratio_threshold", 1.2, 3.0),
        }

    def compute_signal(self, df: pd.DataFrame, **kwargs) -> Signal:
        if not self.validate_dataframe(df) or len(df) < 20:
            return self.flat_signal("Insufficient data")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        rsi = row["rsi_14"]
        vol_ratio = row["vol_ratio"]
        rsi_os = self.params["rsi_oversold"]
        rsi_ob = self.params["rsi_overbought"]
        vol_thresh = self.params["vol_ratio_threshold"]

        if np.isnan(rsi) or np.isnan(vol_ratio):
            return self.flat_signal("NaN indicator values")

        # Volume confirmation required
        if vol_ratio < vol_thresh:
            return self.flat_signal("Volume too low for reversal")

        # Oversold reversal (LONG)
        if rsi < rsi_os and prev["rsi_14"] < rsi_os:
            # RSI turning up from extreme oversold
            if rsi > prev["rsi_14"]:
                strength = min(1.0, (rsi_os - rsi + 10) / 30) * min(1.0, vol_ratio / 3.0)
                return Signal(
                    direction=SignalDirection.LONG,
                    strength=round(strength, 3),
                    confidence=round(min(0.9, vol_ratio / 4.0), 3),
                    indicators_used=["rsi_14", "vol_ratio", "atr_14"],
                    entry_reason=f"RSI reversal from oversold ({rsi:.1f}), volume spike {vol_ratio:.1f}x",
                    strategy_name=self.name,
                )

        # Overbought reversal (SHORT)
        if rsi > rsi_ob and prev["rsi_14"] > rsi_ob:
            if rsi < prev["rsi_14"]:
                strength = min(1.0, (rsi - rsi_ob + 10) / 30) * min(1.0, vol_ratio / 3.0)
                return Signal(
                    direction=SignalDirection.SHORT,
                    strength=round(strength, 3),
                    confidence=round(min(0.9, vol_ratio / 4.0), 3),
                    indicators_used=["rsi_14", "vol_ratio", "atr_14"],
                    entry_reason=f"RSI reversal from overbought ({rsi:.1f}), volume spike {vol_ratio:.1f}x",
                    strategy_name=self.name,
                )

        return self.flat_signal("No RSI extreme with volume confirmation")
