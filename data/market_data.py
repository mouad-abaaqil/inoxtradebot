"""MT5 connection, candle streaming, and in-memory buffer management."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import Settings


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H4 = "4h"
    D1 = "1d"


# Map our timeframes to MT5 constants
TF_MAP = {
    Timeframe.M1: mt5.TIMEFRAME_M1,
    Timeframe.M5: mt5.TIMEFRAME_M5,
    Timeframe.M15: mt5.TIMEFRAME_M15,
    Timeframe.H4: mt5.TIMEFRAME_H4,
    Timeframe.D1: mt5.TIMEFRAME_D1,
}


@dataclass
class NewCandleEvent:
    """Emitted when a bar closes on any timeframe."""
    symbol: str
    timeframe: Timeframe
    candle: pd.Series  # OHLCV row
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MarketDataModule:
    """Manages MT5 connection, streams candles, and maintains rolling buffers."""

    def __init__(self, settings: Settings, event_queue: asyncio.Queue):
        self._settings = settings
        self._queue = event_queue
        self._connected = False
        self._running = False

        # Buffers: symbol -> timeframe -> DataFrame
        self._buffers: Dict[str, Dict[Timeframe, pd.DataFrame]] = defaultdict(dict)

        # Track last known candle time per symbol/tf to detect new bars
        self._last_candle_time: Dict[str, Dict[Timeframe, Optional[datetime]]] = defaultdict(dict)

        # For 5m/15m aggregation from 1m candles
        self._m1_accumulator: Dict[str, List[pd.Series]] = defaultdict(list)

    # ------------------------------------------------------------------
    # MT5 Connection
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Initialize MT5 connection with retry."""
        for attempt in range(3):
            try:
                ok = await asyncio.to_thread(self._init_mt5)
                if ok:
                    self._connected = True
                    logger.info("MT5 connected successfully")
                    return True
                logger.warning("MT5 init failed, attempt {}/3", attempt + 1)
            except Exception as e:
                logger.error("MT5 connection error: {}", e)
            await asyncio.sleep(2 ** attempt)
        logger.critical("Failed to connect to MT5 after 3 attempts")
        return False

    def _init_mt5(self) -> bool:
        kwargs = {}
        if self._settings.MT5_PATH:
            kwargs["path"] = self._settings.MT5_PATH
        if not mt5.initialize(**kwargs):
            logger.error("mt5.initialize() failed: {}", mt5.last_error())
            return False
        authorized = mt5.login(
            login=self._settings.MT5_LOGIN,
            password=self._settings.MT5_PASSWORD,
            server=self._settings.MT5_SERVER,
        )
        if not authorized:
            logger.error("MT5 login failed: {}", mt5.last_error())
            mt5.shutdown()
            return False
        return True

    async def disconnect(self) -> None:
        self._running = False
        await asyncio.to_thread(mt5.shutdown)
        self._connected = False
        logger.info("MT5 disconnected")

    # ------------------------------------------------------------------
    # Data Fetching
    # ------------------------------------------------------------------

    def _fetch_candles(
        self, symbol: str, timeframe: Timeframe, count: int
    ) -> Optional[pd.DataFrame]:
        """Fetch historical candles from MT5."""
        rates = mt5.copy_rates_from_pos(symbol, TF_MAP[timeframe], 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No data for {} {}", symbol, timeframe.value)
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume",
            },
            inplace=True,
        )
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = self._validate_candles(df, symbol, timeframe)
        return df

    def _validate_candles(
        self, df: pd.DataFrame, symbol: str, timeframe: Timeframe
    ) -> pd.DataFrame:
        """Check for gaps, NaN prices, zero-volume bars."""
        nan_count = df[["open", "high", "low", "close"]].isna().sum().sum()
        if nan_count > 0:
            logger.warning("{} {} has {} NaN price values — forward filling", symbol, timeframe.value, nan_count)
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()

        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            logger.debug("{} {} has {} zero-volume bars", symbol, timeframe.value, zero_vol)

        return df

    async def load_historical(self) -> None:
        """Fetch and cache historical data on startup."""
        logger.info("Loading historical data...")
        for symbol in self._settings.SYMBOLS:
            # 1m: ~90 days ≈ 129,600 candles (assuming 24h * 60 * 90)
            bars_1m = min(self._settings.HISTORICAL_DAYS_1M * 24 * 60, 5000)
            df_1m = await asyncio.to_thread(
                self._fetch_candles, symbol, Timeframe.M1, bars_1m
            )
            if df_1m is not None:
                self._buffers[symbol][Timeframe.M1] = df_1m.tail(
                    self._settings.CANDLE_BUFFER_SIZE
                ).copy()
                logger.info(
                    "{} 1m loaded: {} bars", symbol, len(df_1m)
                )

            # 5m
            bars_5m = self._settings.HISTORICAL_DAYS_1M * 24 * 12
            df_5m = await asyncio.to_thread(
                self._fetch_candles, symbol, Timeframe.M5, min(bars_5m, 5000)
            )
            if df_5m is not None:
                self._buffers[symbol][Timeframe.M5] = df_5m.tail(
                    self._settings.CANDLE_BUFFER_SIZE
                ).copy()

            # 15m
            bars_15m = self._settings.HISTORICAL_DAYS_1M * 24 * 4
            df_15m = await asyncio.to_thread(
                self._fetch_candles, symbol, Timeframe.M15, min(bars_15m, 5000)
            )
            if df_15m is not None:
                self._buffers[symbol][Timeframe.M15] = df_15m.tail(
                    self._settings.CANDLE_BUFFER_SIZE
                ).copy()

            # Daily: 2 years
            df_d1 = await asyncio.to_thread(
                self._fetch_candles, symbol, Timeframe.D1, self._settings.HISTORICAL_DAYS_DAILY
            )
            if df_d1 is not None:
                self._buffers[symbol][Timeframe.D1] = df_d1.copy()
                logger.info("{} D1 loaded: {} bars", symbol, len(df_d1))

        logger.info("Historical data loaded for all symbols")

    # ------------------------------------------------------------------
    # Live Streaming
    # ------------------------------------------------------------------

    async def stream_candles(self) -> None:
        """Main loop: poll MT5 every second for new 1m bars."""
        self._running = True
        logger.info("Candle streaming started")

        while self._running:
            try:
                if not self._connected:
                    logger.warning("MT5 disconnected — attempting reconnect")
                    if not await self.connect():
                        await asyncio.sleep(5)
                        continue

                for symbol in self._settings.SYMBOLS:
                    await self._check_new_candles(symbol)

                await asyncio.sleep(1)

            except Exception as e:
                logger.error("Streaming error: {} — reconnecting", e)
                self._connected = False
                await asyncio.sleep(2)

    async def _check_new_candles(self, symbol: str) -> None:
        """Check if a new 1m bar has closed; if so, emit events and aggregate."""
        df = await asyncio.to_thread(self._fetch_candles, symbol, Timeframe.M1, 5)
        if df is None or df.empty:
            return

        last_time = self._last_candle_time.get(symbol, {}).get(Timeframe.M1)
        current_time = df.index[-2] if len(df) >= 2 else None  # completed bar is second-to-last

        if current_time is None or current_time == last_time:
            return

        # New 1m bar detected
        if symbol not in self._last_candle_time:
            self._last_candle_time[symbol] = {}
        self._last_candle_time[symbol][Timeframe.M1] = current_time

        new_candle = df.iloc[-2]  # last completed bar
        self._append_to_buffer(symbol, Timeframe.M1, new_candle)

        await self._queue.put(
            NewCandleEvent(symbol=symbol, timeframe=Timeframe.M1, candle=new_candle)
        )

        # Aggregate to 5m and 15m
        await self._aggregate_higher_tf(symbol, Timeframe.M5, 5, current_time)
        await self._aggregate_higher_tf(symbol, Timeframe.M15, 15, current_time)

    async def _aggregate_higher_tf(
        self, symbol: str, tf: Timeframe, period: int, candle_time: datetime
    ) -> None:
        """Check if a higher timeframe bar has completed."""
        if candle_time.minute % period == 0:
            df = await asyncio.to_thread(self._fetch_candles, symbol, tf, 3)
            if df is not None and len(df) >= 2:
                last_htf_time = self._last_candle_time.get(symbol, {}).get(tf)
                completed = df.index[-2]
                if completed != last_htf_time:
                    self._last_candle_time[symbol][tf] = completed
                    new_candle = df.iloc[-2]
                    self._append_to_buffer(symbol, tf, new_candle)
                    await self._queue.put(
                        NewCandleEvent(symbol=symbol, timeframe=tf, candle=new_candle)
                    )

    def _append_to_buffer(
        self, symbol: str, timeframe: Timeframe, candle: pd.Series
    ) -> None:
        """Append a candle to the rolling buffer, trimming to max size."""
        buf = self._buffers.get(symbol, {}).get(timeframe)
        if buf is None:
            self._buffers[symbol][timeframe] = pd.DataFrame([candle])
            return
        new_row = pd.DataFrame([candle])
        self._buffers[symbol][timeframe] = pd.concat([buf, new_row]).tail(
            self._settings.CANDLE_BUFFER_SIZE
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataframe(self, symbol: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        """Get the buffered DataFrame for a symbol/timeframe."""
        return self._buffers.get(symbol, {}).get(timeframe)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest close price for a symbol."""
        df = self.get_dataframe(symbol, Timeframe.M1)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current spread from MT5 tick data."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return tick.ask - tick.bid

    def get_average_spread(self, symbol: str, periods: int = 100) -> float:
        """Estimate average spread from recent tick history (simplified)."""
        info = mt5.symbol_info(symbol)
        if info is not None:
            return info.spread * self._settings.pip_value(symbol)
        return 0.0

    def is_market_open(self, symbol: str) -> bool:
        """Check if the market is currently open for this symbol.

        Gold (XAUUSD) trades Sunday 23:00 to Friday 22:00 UTC
        with a daily break 22:00-23:00 UTC (Mon-Thu).
        Falls back to MT5 symbol info if available.
        """
        # Try MT5 trade mode first
        info = mt5.symbol_info(symbol)
        if info is not None:
            if info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                return True

        # Manual schedule fallback (especially for gold)
        now = datetime.utcnow()
        weekday = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        hour = now.hour
        minute = now.minute

        sym = symbol.upper().replace("/", "")
        if "XAU" in sym:
            # Saturday: always closed
            if weekday == 5:
                return False
            # Sunday: only open from 23:00 UTC
            if weekday == 6:
                return hour >= 23
            # Friday: open until 22:00 UTC
            if weekday == 4:
                return hour < 22
            # Mon-Thu: closed during daily break 22:00-23:00 UTC
            if 22 <= hour < 23:
                return False
            return True
        else:
            # Forex: Sun 22:00 to Fri 22:00 UTC (no daily break)
            if weekday == 5:
                return False
            if weekday == 6:
                return hour >= 22
            if weekday == 4:
                return hour < 22
            return True

    @property
    def connected(self) -> bool:
        return self._connected
