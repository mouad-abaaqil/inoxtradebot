"""VWAP Bounce Scalper — price touches VWAP with Stochastic confirmation."""

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, SignalDirection


class VWAPBounceScalper(BaseStrategy):
    name = "VWAP_Bounce"

    def default_params(self) -> Dict:
        return {
            "vwap_atr_band": 0.5,
            "stoch_oversold": 20,
            "stoch_overbought": 80,
        }

    def required_indicators(self) -> List[str]:
        return ["vwap", "atr_14", "stoch_k", "stoch_d", "close"]

    def param_space(self, trial) -> Dict:
        return {
            "vwap_atr_band": trial.suggest_float("vwap_atr_band", 0.2, 1.0),
            "stoch_oversold": trial.suggest_int("stoch_oversold", 10, 30),
            "stoch_overbought": trial.suggest_int("stoch_overbought", 70, 90),
        }

    def compute_signal(self, df: pd.DataFrame, **kwargs) -> Signal:
        if not self.validate_dataframe(df) or len(df) < 20:
            return self.flat_signal("Insufficient data")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        close = row["close"]
        vwap = row["vwap"]
        atr = row["atr_14"]
        stoch_k = row["stoch_k"]
        stoch_d = row["stoch_d"]
        prev_stoch_k = prev["stoch_k"]

        if any(np.isnan(v) for v in [close, vwap, atr, stoch_k, stoch_d]):
            return self.flat_signal("NaN values")

        band = atr * self.params["vwap_atr_band"]
        stoch_os = self.params["stoch_oversold"]
        stoch_ob = self.params["stoch_overbought"]

        distance_to_vwap = close - vwap

        # Long: price near/below VWAP, stochastic turning up from oversold
        if abs(distance_to_vwap) <= band and distance_to_vwap <= 0:
            if stoch_k < stoch_os and stoch_k > prev_stoch_k:
                strength = min(1.0, 0.6 + (stoch_os - stoch_k) / 50.0)
                return Signal(
                    direction=SignalDirection.LONG,
                    strength=round(strength, 3),
                    confidence=round(0.65, 3),
                    indicators_used=["vwap", "atr_14", "stoch_k", "stoch_d"],
                    entry_reason=f"VWAP bounce LONG, price near VWAP ({distance_to_vwap:.5f}), stoch turning up ({stoch_k:.1f})",
                    strategy_name=self.name,
                )

        # Short: price near/above VWAP, stochastic turning down from overbought
        if abs(distance_to_vwap) <= band and distance_to_vwap >= 0:
            if stoch_k > stoch_ob and stoch_k < prev_stoch_k:
                strength = min(1.0, 0.6 + (stoch_k - stoch_ob) / 50.0)
                return Signal(
                    direction=SignalDirection.SHORT,
                    strength=round(strength, 3),
                    confidence=round(0.65, 3),
                    indicators_used=["vwap", "atr_14", "stoch_k", "stoch_d"],
                    entry_reason=f"VWAP bounce SHORT, price near VWAP ({distance_to_vwap:.5f}), stoch turning down ({stoch_k:.1f})",
                    strategy_name=self.name,
                )

        return self.flat_signal("Price not near VWAP or no stochastic confirmation")
