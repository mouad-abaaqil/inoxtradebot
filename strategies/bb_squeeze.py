"""BB Squeeze Breakout — Bollinger Band squeeze followed by breakout with volume."""

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy, Signal, SignalDirection


class BBSqueezeBreakout(BaseStrategy):
    name = "BB_Squeeze"

    def default_params(self) -> Dict:
        return {
            "squeeze_percentile": 20,
            "vol_ratio_min": 1.5,
            "lookback": 5,
        }

    def required_indicators(self) -> List[str]:
        return ["bb_upper", "bb_lower", "bb_width", "bb_width_percentile", "vol_ratio", "close"]

    def param_space(self, trial) -> Dict:
        return {
            "squeeze_percentile": trial.suggest_int("squeeze_percentile", 10, 30),
            "vol_ratio_min": trial.suggest_float("vol_ratio_min", 1.0, 2.5),
            "lookback": trial.suggest_int("lookback", 3, 10),
        }

    def compute_signal(self, df: pd.DataFrame, **kwargs) -> Signal:
        if not self.validate_dataframe(df) or len(df) < 30:
            return self.flat_signal("Insufficient data")

        row = df.iloc[-1]
        squeeze_pctile = self.params["squeeze_percentile"]
        vol_min = self.params["vol_ratio_min"]
        lookback = self.params["lookback"]

        bb_pctile = row.get("bb_width_percentile", np.nan)
        close = row["close"]
        bb_upper = row["bb_upper"]
        bb_lower = row["bb_lower"]
        vol_ratio = row["vol_ratio"]

        if any(np.isnan(v) for v in [bb_pctile, close, bb_upper, bb_lower, vol_ratio]):
            return self.flat_signal("NaN values")

        # Check if we were in a squeeze recently (within lookback bars)
        recent = df.tail(lookback)
        was_squeezed = (recent["bb_width_percentile"] < squeeze_pctile / 100.0).any()

        if not was_squeezed:
            return self.flat_signal("No recent BB squeeze")

        # Breakout above upper band with volume
        if close > bb_upper and vol_ratio >= vol_min:
            strength = min(1.0, 0.5 + vol_ratio / 5.0)
            return Signal(
                direction=SignalDirection.LONG,
                strength=round(strength, 3),
                confidence=round(min(0.85, vol_ratio / 3.0), 3),
                indicators_used=["bb_upper", "bb_width_percentile", "vol_ratio"],
                entry_reason=f"BB squeeze breakout UP, close={close:.5f} > upper={bb_upper:.5f}, vol={vol_ratio:.1f}x",
                strategy_name=self.name,
            )

        # Breakout below lower band with volume
        if close < bb_lower and vol_ratio >= vol_min:
            strength = min(1.0, 0.5 + vol_ratio / 5.0)
            return Signal(
                direction=SignalDirection.SHORT,
                strength=round(strength, 3),
                confidence=round(min(0.85, vol_ratio / 3.0), 3),
                indicators_used=["bb_lower", "bb_width_percentile", "vol_ratio"],
                entry_reason=f"BB squeeze breakout DOWN, close={close:.5f} < lower={bb_lower:.5f}, vol={vol_ratio:.1f}x",
                strategy_name=self.name,
            )

        return self.flat_signal("Squeeze detected but no breakout yet")
