"""Daily Trend Follower (DTF) for XAUUSD M1.

Trade ONLY in the direction of the Daily trend, entering on M1 pullbacks.
Never trade against the Daily bias.

Entry logic (8 steps):
1. Session gate: London (07-12 UTC) and New York (13-18 UTC)
2. Daily bias: close vs EMA20, EMA20 vs EMA50, slope → bull/bear/neutral
3. ATR M1 minimum >= 0.40
4. M1 pullback: price below EMA9 (LONG) or above EMA9 (SHORT), RSI in zone,
   price still above/below EMA50
5. Reversal confirmation: green candle + close > prev close (LONG), inverse for SHORT
6. Candle body ratio >= 0.40
7. Adaptive cooldown: max(60, int(180 / (atr / 0.8))) seconds
8. SL/TP: SL from N-bar swing ± 0.10 (default 3), TP1 = 1.2× SL, TP2 = 2.0× SL
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal, SignalDirection


class DailyTrendFollower(BaseStrategy):
    """Daily Trend Follower scalper for XAUUSD M1."""

    name = "Gold_Scalper"  # Registry compatibility

    def __init__(self, params=None, **kwargs):
        super().__init__(params=params, **kwargs)
        self._last_signal_time = 0.0

    def default_params(self) -> Dict:
        return {
            "min_atr": 0.40,
            "rsi_pullback_long_min": 35,
            "rsi_pullback_long_max": 55,
            "rsi_pullback_short_min": 45,
            "rsi_pullback_short_max": 65,
            "body_ratio_min": 0.40,
            "min_tp1_pips": 10,
            "max_sl_factor": 1.5,
            "sl_swing_bars": 5,
            "tp1_rr": 1.0,
            "tp2_rr": 2.0,
            "cooldown_base": 180,
            "session_windows": [(7, 12), (13, 18)],
        }

    def required_indicators(self) -> List[str]:
        return [
            "close", "open", "high", "low", "volume",
            "ema_9", "ema_50", "rsi_14", "atr_14",
            "vol_sma_20",
        ]

    def param_space(self, trial) -> Dict:
        return {
            "min_atr": trial.suggest_float("min_atr", 0.30, 0.60),
            "rsi_pullback_long_min": trial.suggest_int("rsi_pullback_long_min", 30, 40),
            "rsi_pullback_long_max": trial.suggest_int("rsi_pullback_long_max", 50, 60),
            "rsi_pullback_short_min": trial.suggest_int("rsi_pullback_short_min", 40, 50),
            "rsi_pullback_short_max": trial.suggest_int("rsi_pullback_short_max", 60, 70),
            "body_ratio_min": trial.suggest_float("body_ratio_min", 0.30, 0.55),
            "tp1_rr": trial.suggest_float("tp1_rr", 0.8, 1.5),
            "tp2_rr": trial.suggest_float("tp2_rr", 1.5, 3.0),
            "cooldown_base": trial.suggest_int("cooldown_base", 120, 300),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_signal(self, df: pd.DataFrame, **kwargs: Any) -> Signal:
        """DTF strategy -- 8-step signal pipeline."""

        settings = kwargs.get("settings")
        df_daily = kwargs.get("df_daily")
        p = self._resolve_params(settings)

        if not self.validate_dataframe(df) or len(df) < 25:
            return self.flat_signal("Insufficient M1 data")

        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None
        close = float(current["close"])
        open_ = float(current["open"])
        high = float(current["high"])
        low = float(current["low"])
        volume = float(current["volume"])
        atr = float(current["atr_14"])
        ema9 = float(current["ema_9"])
        ema50 = float(current["ema_50"])
        rsi = float(current["rsi_14"])
        vol_sma = float(current["vol_sma_20"])

        if any(np.isnan(v) for v in [close, open_, high, low, atr, ema9, ema50, rsi]):
            return self.flat_signal("NaN in M1 indicators")

        # -- STEP 1 -- SESSION GATE ------------------------------------
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        in_session = any(s <= hour < e for s, e in p["session_windows"])
        if not in_session:
            return self.flat_signal("SESSION_BLOCKED")

        # -- STEP 2 -- DAILY BIAS --------------------------------------
        bias = self._daily_bias(df_daily)
        if bias == "NEUTRAL":
            logger.info("[DTF] SKIP daily_neutral")
            return self.flat_signal("DAILY_NEUTRAL")

        # Direction is dictated by the daily trend
        direction = bias  # "LONG" or "SHORT"

        # -- STEP 3 -- ATR MINIMUM -------------------------------------
        if atr < p["min_atr"]:
            logger.info("[DTF] SKIP low_atr (atr={:.4f})", atr)
            return self.flat_signal("LOW_ATR")

        # -- STEP 4 -- M1 PULLBACK DETECTION ---------------------------
        if direction == "LONG":
            # Price pulled back below EMA9 but still above EMA50
            if not (close < ema9 or open_ < ema9):
                return self.flat_signal("NO_PULLBACK")
            if close < ema50:
                logger.info("[DTF] SKIP below_ema50 (LONG: close={:.2f} ema50={:.2f})", close, ema50)
                return self.flat_signal("BELOW_EMA50")
            if not (p["rsi_pullback_long_min"] <= rsi <= p["rsi_pullback_long_max"]):
                logger.info("[DTF] SKIP rsi_invalid (LONG rsi={:.1f})", rsi)
                return self.flat_signal("RSI_INVALID")
        else:
            # Price pulled back above EMA9 but still below EMA50
            if not (close > ema9 or open_ > ema9):
                return self.flat_signal("NO_PULLBACK")
            if close > ema50:
                logger.info("[DTF] SKIP above_ema50 (SHORT: close={:.2f} ema50={:.2f})", close, ema50)
                return self.flat_signal("ABOVE_EMA50")
            if not (p["rsi_pullback_short_min"] <= rsi <= p["rsi_pullback_short_max"]):
                logger.info("[DTF] SKIP rsi_invalid (SHORT rsi={:.1f})", rsi)
                return self.flat_signal("RSI_INVALID")

        # -- STEP 5 -- REVERSAL CONFIRMATION ---------------------------
        if prev is None:
            return self.flat_signal("Insufficient M1 data")
        prev_close = float(prev["close"])

        if direction == "LONG":
            if not (close > open_ and close > prev_close):
                logger.info("[DTF] SKIP no_reversal (LONG)")
                return self.flat_signal("NO_REVERSAL")
        else:
            if not (close < open_ and close < prev_close):
                logger.info("[DTF] SKIP no_reversal (SHORT)")
                return self.flat_signal("NO_REVERSAL")

        # -- STEP 6 -- CANDLE BODY RATIO -------------------------------
        candle_range = high - low
        if candle_range <= 0:
            return self.flat_signal("ZERO_RANGE_CANDLE")

        body_ratio = abs(close - open_) / candle_range
        if body_ratio < p["body_ratio_min"]:
            logger.info("[DTF] SKIP weak_candle (body={:.2f})", body_ratio)
            return self.flat_signal("WEAK_CANDLE")

        # -- STEP 7 -- ADAPTIVE COOLDOWN -------------------------------
        cooldown = max(60, int(p["cooldown_base"] / (atr / 0.8)))
        elapsed = time.time() - self._last_signal_time
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            logger.info("[DTF] SKIP cooldown ({:.0f}s remaining)", remaining)
            return self.flat_signal("COOLDOWN")

        # -- STEP 8 -- SL / TP CALCULATION -----------------------------
        digits = 2
        m1_tail = df.tail(p["sl_swing_bars"])

        if direction == "LONG":
            swing_low = float(m1_tail["low"].min())
            sl = swing_low - 0.10
            sl_distance = close - sl
        else:
            swing_high = float(m1_tail["high"].max())
            sl = swing_high + 0.10
            sl_distance = sl - close

        sl = round(sl, digits)

        # SL floor
        if sl_distance < 0.01:
            return self.flat_signal("SL_TOO_TIGHT")

        # SL cap
        max_sl = atr * p["max_sl_factor"]
        if sl_distance > max_sl:
            logger.info("[DTF] SKIP sl_too_wide (sl_dist={:.4f} > max={:.4f})", sl_distance, max_sl)
            return self.flat_signal("SL_TOO_WIDE")

        # TP based on R:R multiples of SL distance
        if direction == "LONG":
            tp1 = round(close + sl_distance * p["tp1_rr"], digits)
            tp2 = round(close + sl_distance * p["tp2_rr"], digits)
        else:
            tp1 = round(close - sl_distance * p["tp1_rr"], digits)
            tp2 = round(close - sl_distance * p["tp2_rr"], digits)

        # Minimum TP1 distance
        min_tp1 = p["min_tp1_pips"] * 0.01
        tp1_distance = abs(tp1 - close)
        if tp1_distance < min_tp1:
            logger.info("[DTF] SKIP tp1_too_small (tp1_dist={:.4f})", tp1_distance)
            return self.flat_signal("TP1_TOO_SMALL")

        # Update cooldown
        self._last_signal_time = time.time()

        # -- SIGNAL STRENGTH -------------------------------------------
        strength = 0.50

        # Strong daily slope bonus
        daily_slope = self._daily_slope(df_daily)
        if direction == "LONG" and daily_slope > 0.15:
            strength += 0.20
        elif direction == "SHORT" and daily_slope < -0.15:
            strength += 0.20
        elif abs(daily_slope) > 0.08:
            strength += 0.10

        # Volume confirmation
        if vol_sma > 0 and not np.isnan(vol_sma) and volume >= vol_sma * 1.2:
            strength += 0.15

        # Clean candle
        if body_ratio >= 0.60:
            strength += 0.10

        # RSI sweet spot
        if direction == "LONG" and rsi < 45:
            strength += 0.05
        elif direction == "SHORT" and rsi > 55:
            strength += 0.05

        strength = min(1.0, strength)

        # -- BUILD SIGNAL ----------------------------------------------
        entry_reason = (
            f"DTF {direction} | bias={bias} slope={daily_slope:+.3f} rsi={rsi:.1f} | "
            f"body={body_ratio:.2f} vol={volume:.0f} | "
            f"SL={sl:.2f} TP1={tp1:.2f} TP2={tp2:.2f}"
        )
        logger.info("[DTF] SIGNAL {} strength={:.2f} | {}", direction, strength, entry_reason)

        sig_direction = SignalDirection.LONG if direction == "LONG" else SignalDirection.SHORT

        return Signal(
            direction=sig_direction,
            strength=round(strength, 3),
            confidence=round(min(0.90, strength + 0.10), 3),
            indicators_used=["ema_9", "ema_50", "rsi_14", "atr_14", "vol_sma_20", "daily_ema20", "daily_ema50"],
            entry_reason=entry_reason,
            strategy_name=self.name,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _daily_bias(self, df_daily: Optional[pd.DataFrame]) -> str:
        """Determine daily trend direction: LONG, SHORT, or NEUTRAL.

        Bull: close > ema20 AND ema20 > ema50
        Bear: close < ema20 AND ema20 < ema50
        Otherwise: NEUTRAL
        """
        if df_daily is None or len(df_daily) < 3:
            return "NEUTRAL"

        last = df_daily.iloc[-1]
        close_d = float(last["close"])
        ema20_d = float(last.get("ema_20", np.nan))
        ema50_d = float(last.get("ema_50", np.nan))

        if any(np.isnan(v) for v in [close_d, ema20_d, ema50_d]):
            return "NEUTRAL"

        if close_d > ema20_d and ema20_d > ema50_d:
            return "LONG"
        elif close_d < ema20_d and ema20_d < ema50_d:
            return "SHORT"
        return "NEUTRAL"

    def _daily_slope(self, df_daily: Optional[pd.DataFrame]) -> float:
        """Compute normalised EMA20 slope over last 3 daily bars."""
        if df_daily is None or len(df_daily) < 4:
            return 0.0

        ema20_col = df_daily.get("ema_20")
        if ema20_col is None:
            return 0.0

        vals = ema20_col.iloc[-4:].values.astype(float)
        if any(np.isnan(vals)):
            return 0.0

        # Slope as % change over 3 bars
        if vals[0] == 0:
            return 0.0
        return (vals[-1] - vals[0]) / vals[0]

    def _resolve_params(self, settings) -> Dict:
        """Merge default params with settings overrides."""
        p = self.default_params()
        if settings is not None:
            p["min_atr"] = getattr(settings, "DTF_MIN_ATR", p["min_atr"])
            p["rsi_pullback_long_min"] = getattr(settings, "DTF_RSI_PULLBACK_LONG_MIN", p["rsi_pullback_long_min"])
            p["rsi_pullback_long_max"] = getattr(settings, "DTF_RSI_PULLBACK_LONG_MAX", p["rsi_pullback_long_max"])
            p["rsi_pullback_short_min"] = getattr(settings, "DTF_RSI_PULLBACK_SHORT_MIN", p["rsi_pullback_short_min"])
            p["rsi_pullback_short_max"] = getattr(settings, "DTF_RSI_PULLBACK_SHORT_MAX", p["rsi_pullback_short_max"])
            p["body_ratio_min"] = getattr(settings, "DTF_BODY_RATIO_MIN", p["body_ratio_min"])
            p["min_tp1_pips"] = getattr(settings, "DTF_MIN_TP1_PIPS", p["min_tp1_pips"])
            p["max_sl_factor"] = getattr(settings, "DTF_MAX_SL_FACTOR", p["max_sl_factor"])
            p["sl_swing_bars"] = getattr(settings, "DTF_SL_SWING_BARS", p["sl_swing_bars"])
            p["tp1_rr"] = getattr(settings, "DTF_TP1_RR", p["tp1_rr"])
            p["tp2_rr"] = getattr(settings, "DTF_TP2_RR", p["tp2_rr"])
            p["cooldown_base"] = getattr(settings, "DTF_COOLDOWN_BASE", p["cooldown_base"])
            # Parse session windows from settings string "7-12,13-18"
            raw = getattr(settings, "DTF_SESSION_WINDOWS", None)
            if isinstance(raw, str):
                windows = []
                for chunk in raw.split(","):
                    parts = chunk.strip().split("-")
                    if len(parts) == 2:
                        windows.append((int(parts[0]), int(parts[1])))
                if windows:
                    p["session_windows"] = windows
        return p


# Backward compatibility aliases
GoldScalper = DailyTrendFollower
EMARibbonScalper = DailyTrendFollower
ICTMomentumPullback = DailyTrendFollower
VWAPReversionScalper = DailyTrendFollower
