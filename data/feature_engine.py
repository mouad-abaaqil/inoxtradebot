"""Technical indicator computation using TA-Lib."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import talib
from loguru import logger

from data.market_data import MarketDataModule, NewCandleEvent, Timeframe


@dataclass
class FeatureVector:
    """Typed container for strategy consumption."""

    symbol: str
    timeframe: Timeframe

    # Momentum
    rsi_14: float = np.nan
    rsi_9: float = np.nan
    macd: float = np.nan
    macd_signal: float = np.nan
    macd_hist: float = np.nan
    stoch_k: float = np.nan
    stoch_d: float = np.nan

    # Trend
    ema_8: float = np.nan
    ema_21: float = np.nan
    ema_50: float = np.nan
    sma_200: float = np.nan

    # Volatility
    atr_14: float = np.nan
    bb_upper: float = np.nan
    bb_middle: float = np.nan
    bb_lower: float = np.nan
    bb_width: float = np.nan
    bb_width_percentile: float = np.nan

    # Volume
    vwap: float = np.nan
    obv: float = np.nan
    vol_sma_20: float = np.nan
    vol_ratio: float = np.nan

    # Price
    close: float = np.nan
    high: float = np.nan
    low: float = np.nan
    open: float = np.nan
    volume: float = np.nan

    # Cross-timeframe
    ema_8_5m: float = np.nan
    ema_21_5m: float = np.nan
    ema_8_15m: float = np.nan
    ema_21_15m: float = np.nan


class FeatureEngine:
    """Computes technical indicators on candle DataFrames."""

    def __init__(self, market_data: MarketDataModule, event_queue: asyncio.Queue):
        self._market_data = market_data
        self._queue = event_queue

    async def process_event(self, event: NewCandleEvent) -> Optional[FeatureVector]:
        """Called on each NewCandle event. Updates indicators and returns feature vector."""
        try:
            df = self._market_data.get_dataframe(event.symbol, event.timeframe)
            if df is None or len(df) < 50:
                return None

            df = self._compute_indicators(df)
            # Write back to market data buffer
            self._market_data._buffers[event.symbol][event.timeframe] = df

            if event.timeframe == Timeframe.M1:
                fv = self._build_feature_vector(event.symbol, df)
                return fv
            return None

        except Exception as e:
            logger.error("FeatureEngine error for {} {}: {}", event.symbol, event.timeframe.value, e)
            return None

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators in-place — fully vectorized."""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].astype(float).values

        # --- Momentum ---
        df["rsi_14"] = talib.RSI(close, timeperiod=14)
        df["rsi_9"] = talib.RSI(close, timeperiod=9)

        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df["stoch_k"] = slowk
        df["stoch_d"] = slowd

        # --- Trend ---
        df["ema_3"] = talib.EMA(close, timeperiod=3)
        df["ema_8"] = talib.EMA(close, timeperiod=8)
        df["ema_9"] = talib.EMA(close, timeperiod=9)
        df["ema_20"] = talib.EMA(close, timeperiod=20)
        df["ema_21"] = talib.EMA(close, timeperiod=21)
        df["ema_50"] = talib.EMA(close, timeperiod=50)
        df["sma_200"] = talib.SMA(close, timeperiod=200)
        df["adx"] = talib.ADX(high, low, close, timeperiod=14)

        # --- Volatility ---
        df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)

        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        df["bb_width"] = (bb_upper - bb_lower) / np.where(bb_middle != 0, bb_middle, np.nan)

        # BB width percentile (squeeze detector)
        bb_width_series = pd.Series(df["bb_width"].values)
        df["bb_width_percentile"] = bb_width_series.rolling(100, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ).values

        # --- Volume ---
        # VWAP — daily reset: cumsum resets at midnight UTC each day
        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume
        dates = df.index.date
        day_groups = np.concatenate([[True], dates[1:] != dates[:-1]])
        cum_tp_vol = np.zeros(len(df), dtype=float)
        cum_vol = np.zeros(len(df), dtype=float)
        running_tp_vol = 0.0
        running_vol = 0.0
        for k in range(len(df)):
            if day_groups[k]:
                running_tp_vol = 0.0
                running_vol = 0.0
            running_tp_vol += tp_vol[k]
            running_vol += volume[k]
            cum_tp_vol[k] = running_tp_vol
            cum_vol[k] = running_vol
        df["vwap"] = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)

        # VWAP bands and distance (for VRS strategy)
        atr_vals = df["atr_14"].values
        vwap_vals = df["vwap"].values
        df["vwap_upper"] = vwap_vals + atr_vals * 1.0
        df["vwap_lower"] = vwap_vals - atr_vals * 1.0
        df["vwap_dist"] = np.where(atr_vals > 0, (close - vwap_vals) / atr_vals, 0.0)

        df["obv"] = talib.OBV(close, volume)
        df["vol_sma_20"] = talib.SMA(volume, timeperiod=20)
        df["vol_ratio"] = np.where(
            df["vol_sma_20"] > 0,
            volume / df["vol_sma_20"].values,
            1.0,
        )

        return df

    def compute_for_symbol(self, symbol: str) -> None:
        """Recompute indicators for all timeframes of a symbol (used at startup)."""
        for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
            df = self._market_data.get_dataframe(symbol, tf)
            if df is not None and len(df) >= 50:
                self._market_data._buffers[symbol][tf] = self._compute_indicators(df)
                logger.debug("Indicators computed for {} {}", symbol, tf.value)
                if tf == Timeframe.M5:
                    logger.info(
                        "M5 columns: {}", df.columns.tolist()
                    )
                    last_row = df.iloc[-1]
                    logger.info(
                        "M5 last row: adx={} ema_20={} atr_14={} close={}",
                        last_row.get("adx", "MISSING"),
                        last_row.get("ema_20", "MISSING"),
                        last_row.get("atr_14", "MISSING"),
                        last_row.get("close", "MISSING"),
                    )

    def _build_feature_vector(self, symbol: str, df_1m: pd.DataFrame) -> FeatureVector:
        """Build a FeatureVector from the latest row of the 1m DataFrame."""
        row = df_1m.iloc[-1]

        fv = FeatureVector(
            symbol=symbol,
            timeframe=Timeframe.M1,
            close=_safe(row, "close"),
            high=_safe(row, "high"),
            low=_safe(row, "low"),
            open=_safe(row, "open"),
            volume=_safe(row, "volume"),
            rsi_14=_safe(row, "rsi_14"),
            rsi_9=_safe(row, "rsi_9"),
            macd=_safe(row, "macd"),
            macd_signal=_safe(row, "macd_signal"),
            macd_hist=_safe(row, "macd_hist"),
            stoch_k=_safe(row, "stoch_k"),
            stoch_d=_safe(row, "stoch_d"),
            ema_8=_safe(row, "ema_8"),
            ema_21=_safe(row, "ema_21"),
            ema_50=_safe(row, "ema_50"),
            sma_200=_safe(row, "sma_200"),
            atr_14=_safe(row, "atr_14"),
            bb_upper=_safe(row, "bb_upper"),
            bb_middle=_safe(row, "bb_middle"),
            bb_lower=_safe(row, "bb_lower"),
            bb_width=_safe(row, "bb_width"),
            bb_width_percentile=_safe(row, "bb_width_percentile"),
            vwap=_safe(row, "vwap"),
            obv=_safe(row, "obv"),
            vol_sma_20=_safe(row, "vol_sma_20"),
            vol_ratio=_safe(row, "vol_ratio"),
        )

        # Cross-timeframe features
        for tf, suffix in [(Timeframe.M5, "5m"), (Timeframe.M15, "15m")]:
            htf_df = self._market_data.get_dataframe(symbol, tf)
            if htf_df is not None and "ema_8" in htf_df.columns and not htf_df.empty:
                htf_row = htf_df.iloc[-1]
                setattr(fv, f"ema_8_{suffix}", _safe(htf_row, "ema_8"))
                setattr(fv, f"ema_21_{suffix}", _safe(htf_row, "ema_21"))

        return fv

    def get_latest_features(self, symbol: str) -> Optional[FeatureVector]:
        """Get the current feature vector for a symbol without waiting for an event."""
        df = self._market_data.get_dataframe(symbol, Timeframe.M1)
        if df is None or len(df) < 50:
            return None
        if "rsi_14" not in df.columns:
            df = self._compute_indicators(df)
            self._market_data._buffers[symbol][Timeframe.M1] = df
        return self._build_feature_vector(symbol, df)

    def get_dataframe(self, symbol: str, timeframe: Timeframe = Timeframe.M1) -> Optional[pd.DataFrame]:
        """Get the indicator-enriched DataFrame."""
        return self._market_data.get_dataframe(symbol, timeframe)


def _safe(row: pd.Series, col: str) -> float:
    """Safely extract a float from a Series row."""
    try:
        val = row.get(col, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        return float(val)
    except (TypeError, ValueError):
        return np.nan
