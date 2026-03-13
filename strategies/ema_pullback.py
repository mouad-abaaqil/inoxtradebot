"""EMA Pullback Scalper — pullback to EMA8 in a confirmed trend."""

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, SignalDirection


class EMAPullbackScalper(BaseStrategy):
    name = "EMA_Pullback"

    def default_params(self) -> Dict:
        return {
            "rsi_low": 40,
            "rsi_high": 60,
            "atr_pullback_factor": 0.3,
        }

    def required_indicators(self) -> List[str]:
        return ["ema_8", "ema_21", "ema_50", "rsi_14", "atr_14", "close", "low", "high"]

    def param_space(self, trial) -> Dict:
        return {
            "rsi_low": trial.suggest_int("rsi_low", 30, 50),
            "rsi_high": trial.suggest_int("rsi_high", 50, 70),
            "atr_pullback_factor": trial.suggest_float("atr_pullback_factor", 0.1, 0.8),
        }

    def compute_signal(self, df: pd.DataFrame, **kwargs) -> Signal:
        if not self.validate_dataframe(df) or len(df) < 50:
            return self.flat_signal("Insufficient data")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        close = row["close"]
        low = row["low"]
        high = row["high"]
        ema8 = row["ema_8"]
        ema21 = row["ema_21"]
        ema50 = row["ema_50"]
        rsi = row["rsi_14"]
        atr = row["atr_14"]

        if any(np.isnan(v) for v in [close, ema8, ema21, ema50, rsi, atr]):
            return self.flat_signal("NaN values")

        rsi_low = self.params["rsi_low"]
        rsi_high = self.params["rsi_high"]
        pullback_band = atr * self.params["atr_pullback_factor"]

        # Bullish trend: EMA8 > EMA21 > EMA50
        if ema8 > ema21 > ema50:
            # Price pulled back to EMA8 zone
            if low <= ema8 + pullback_band and close > ema8:
                # RSI in neutral zone (not overextended)
                if rsi_low <= rsi <= rsi_high:
                    # Bounce: current close above previous close
                    if close > prev["close"]:
                        strength = min(1.0, 0.6 + (ema8 - ema21) / (atr + 1e-10) * 0.1)
                        return Signal(
                            direction=SignalDirection.LONG,
                            strength=round(strength, 3),
                            confidence=round(0.7, 3),
                            indicators_used=["ema_8", "ema_21", "ema_50", "rsi_14"],
                            entry_reason=f"EMA pullback LONG, trend aligned, RSI={rsi:.1f}, bounce off EMA8",
                            strategy_name=self.name,
                        )

        # Bearish trend: EMA8 < EMA21 < EMA50
        if ema8 < ema21 < ema50:
            if high >= ema8 - pullback_band and close < ema8:
                if rsi_low <= rsi <= rsi_high:
                    if close < prev["close"]:
                        strength = min(1.0, 0.6 + (ema21 - ema8) / (atr + 1e-10) * 0.1)
                        return Signal(
                            direction=SignalDirection.SHORT,
                            strength=round(strength, 3),
                            confidence=round(0.7, 3),
                            indicators_used=["ema_8", "ema_21", "ema_50", "rsi_14"],
                            entry_reason=f"EMA pullback SHORT, trend aligned, RSI={rsi:.1f}, rejection at EMA8",
                            strategy_name=self.name,
                        )

        return self.flat_signal("No trend alignment or pullback conditions")
