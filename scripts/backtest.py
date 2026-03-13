#!/usr/bin/env python
"""Backtest Daily Trend Follower (DTF) on real MT5 historical data.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --days 30
    python scripts/backtest.py --days 90 --lot 0.04
    python scripts/backtest.py --from 2026-01-01 --to 2026-03-01
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time as _real_time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from io import StringIO
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import strategy & feature_engine
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# We import MT5 early to fail fast if not available
try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed.  pip install MetaTrader5")
    sys.exit(1)

import talib
from loguru import logger

# Suppress noisy strategy logging during backtest
logger.remove()
logger.add(sys.stderr, level="WARNING", format="{message}")

from config.settings import get_settings
from strategies.gold_scalper import DailyTrendFollower
from strategies.base_strategy import SignalDirection

# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def progress_iter(iterable, total, desc=""):
    """Wrap iterable in tqdm if available, else print every 2000 items."""
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=desc, ncols=90)
    else:
        for i, item in enumerate(iterable):
            if i % 2000 == 0:
                pct = i / total * 100 if total else 0
                print(f"  {desc}: {i}/{total} ({pct:.0f}%)", flush=True)
            yield item


# ---------------------------------------------------------------------------
# Backtest clock -- monkey-patches datetime.utcnow() and time.time()
# inside gold_scalper so session gate and cooldown use candle timestamps.
# ---------------------------------------------------------------------------
class _BacktestClock:
    """Mutable clock injected into gold_scalper module at import time."""

    def __init__(self):
        self.current_dt: datetime = datetime(2020, 1, 1)
        self.current_ts: float = 0.0

    # Replaces datetime.utcnow()
    def utcnow(self):
        return self.current_dt

    # Replaces time.time()
    def time(self):
        return self.current_ts


# ---------------------------------------------------------------------------
# Indicator computation (reuses exact same TA-Lib calls as feature_engine.py)
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators exactly like FeatureEngine._compute_indicators."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].astype(float).values

    # Momentum
    df["rsi_14"] = talib.RSI(close, timeperiod=14)
    df["rsi_9"] = talib.RSI(close, timeperiod=9)
    macd, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    slowk, slowd = talib.STOCH(high, low, close, 14, 3, 3)
    df["stoch_k"] = slowk
    df["stoch_d"] = slowd

    # Trend
    df["ema_3"] = talib.EMA(close, timeperiod=3)
    df["ema_8"] = talib.EMA(close, timeperiod=8)
    df["ema_9"] = talib.EMA(close, timeperiod=9)
    df["ema_20"] = talib.EMA(close, timeperiod=20)
    df["ema_21"] = talib.EMA(close, timeperiod=21)
    df["ema_50"] = talib.EMA(close, timeperiod=50)
    df["sma_200"] = talib.SMA(close, timeperiod=200)
    df["adx"] = talib.ADX(high, low, close, timeperiod=14)

    # Volatility
    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 20, 2, 2)
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / np.where(bb_middle != 0, bb_middle, np.nan)

    # Volume
    # VWAP — daily reset: cumsum resets at midnight UTC each day
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    dates = df.index.date
    day_groups = np.concatenate([[True], dates[1:] != dates[:-1]])
    cum_tp_vol = np.zeros(len(df), dtype=float)
    cum_vol_arr = np.zeros(len(df), dtype=float)
    running_tp_vol = 0.0
    running_vol = 0.0
    for k in range(len(df)):
        if day_groups[k]:
            running_tp_vol = 0.0
            running_vol = 0.0
        running_tp_vol += tp_vol[k]
        running_vol += volume[k]
        cum_tp_vol[k] = running_tp_vol
        cum_vol_arr[k] = running_vol
    df["vwap"] = np.where(cum_vol_arr > 0, cum_tp_vol / cum_vol_arr, np.nan)

    # VWAP bands and distance
    atr_vals = df["atr_14"].values
    vwap_vals = df["vwap"].values
    df["vwap_upper"] = vwap_vals + atr_vals * 1.0
    df["vwap_lower"] = vwap_vals - atr_vals * 1.0
    df["vwap_dist"] = np.where(atr_vals > 0, (close - vwap_vals) / atr_vals, 0.0)
    df["obv"] = talib.OBV(close, volume)
    df["vol_sma_20"] = talib.SMA(volume, timeperiod=20)
    df["vol_ratio"] = np.where(
        df["vol_sma_20"] > 0, volume / df["vol_sma_20"].values, 1.0
    )

    return df


def compute_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Daily indicators needed by DTF strategy (ema_20, ema_50)."""
    close = df["close"].values
    df["ema_20"] = talib.EMA(close, timeperiod=20)
    df["ema_50"] = talib.EMA(close, timeperiod=50)
    return df


# ---------------------------------------------------------------------------
# MT5 data fetching
# ---------------------------------------------------------------------------
def _mt5_call_with_timeout(func, args=(), kwargs=None, timeout=30):
    """Run an MT5 function in a thread with a timeout (seconds).

    Returns the result or raises TimeoutError.
    """
    if kwargs is None:
        kwargs = {}
    result = [None]
    exc = [None]

    def _target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(
            f"MT5 call {func.__name__} timed out after {timeout}s. "
            "Ensure MetaTrader 5 is running in the foreground."
        )
    if exc[0] is not None:
        raise exc[0]
    return result[0]


def connect_mt5(symbol: str = "XAUUSD") -> bool:
    settings = get_settings()
    kwargs = {}
    if settings.MT5_PATH:
        kwargs["path"] = settings.MT5_PATH
    if not mt5.initialize(**kwargs):
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}")
        return False
    if not mt5.login(
        login=settings.MT5_LOGIN,
        password=settings.MT5_PASSWORD,
        server=settings.MT5_SERVER,
    ):
        print(f"ERROR: MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    # Ensure symbol is available in Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"ERROR: symbol_select('{symbol}') failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    print(f"Symbol '{symbol}' selected in Market Watch.")

    return True


def _rates_to_df(rates) -> pd.DataFrame:
    """Convert MT5 rates array to a clean OHLCV DataFrame."""
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    return df


def _fetch_chunked(
    symbol: str, tf, dt_from: datetime, dt_to: datetime,
    chunk_days: int = 30, label: str = "", timeout: int = 30,
) -> pd.DataFrame:
    """Fetch data in chunks of chunk_days to stay under MT5's ~50k bar limit.

    Walks backward from dt_to to dt_from, fetching one chunk at a time,
    then concatenates and deduplicates.
    """
    frames: list[pd.DataFrame] = []
    current_to = dt_to

    while current_to > dt_from:
        current_from = max(dt_from, current_to - timedelta(days=chunk_days))
        print(
            f"  {label}: fetching {current_from.strftime('%Y-%m-%d')} -> "
            f"{current_to.strftime('%Y-%m-%d')} ...",
            flush=True,
        )
        try:
            rates = _mt5_call_with_timeout(
                mt5.copy_rates_range, args=(symbol, tf, current_from, current_to),
                timeout=timeout,
            )
        except TimeoutError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        if rates is None or len(rates) == 0:
            print(f"  {label}: chunk returned 0 bars (error: {mt5.last_error()})")
            if not frames:
                # First chunk failed — fatal
                print(f"ERROR: {label} fetch failed entirely")
                mt5.shutdown()
                sys.exit(1)
            break  # Earlier chunks succeeded, stop here

        df_chunk = _rates_to_df(rates)
        frames.append(df_chunk)
        print(f"  {label}: chunk {len(df_chunk):,} bars", flush=True)

        current_to = current_from

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(
        f"  {label} total: {len(df):,} bars ({df.index[0]} -> {df.index[-1]})",
        flush=True,
    )
    return df


def _fetch_m1(symbol: str, dt_from: datetime, dt_to: datetime, timeout: int = 30) -> pd.DataFrame:
    """Fetch M1 data in 30-day chunks (~43k bars each, under MT5 limit)."""
    return _fetch_chunked(symbol, mt5.TIMEFRAME_M1, dt_from, dt_to,
                          chunk_days=30, label="M1", timeout=timeout)


def _fetch_m5(symbol: str, dt_from: datetime, dt_to: datetime, timeout: int = 30) -> pd.DataFrame:
    """Fetch M5 data in 60-day chunks."""
    return _fetch_chunked(symbol, mt5.TIMEFRAME_M5, dt_from, dt_to,
                          chunk_days=60, label="M5", timeout=timeout)


def _fetch_h4(symbol: str, dt_from: datetime, dt_to: datetime, timeout: int = 30) -> pd.DataFrame:
    """Fetch H4 data in 60-day chunks."""
    return _fetch_chunked(symbol, mt5.TIMEFRAME_H4, dt_from, dt_to,
                          chunk_days=60, label="H4", timeout=timeout)


def _fetch_daily(symbol: str, dt_from: datetime, dt_to: datetime, timeout: int = 30) -> pd.DataFrame:
    """Fetch D1 data in 120-day chunks."""
    return _fetch_chunked(symbol, mt5.TIMEFRAME_D1, dt_from, dt_to,
                          chunk_days=120, label="D1", timeout=timeout)


def fetch_candles(symbol: str, tf, count: int) -> Optional[pd.DataFrame]:
    """Fetch recent candles using copy_rates_range with timeout."""
    dt_to = datetime.now()
    dt_from = dt_to - timedelta(days=max(count // 1440 + 2, 7))
    try:
        rates = _mt5_call_with_timeout(
            mt5.copy_rates_range, args=(symbol, tf, dt_from, dt_to), timeout=30,
        )
    except TimeoutError as e:
        print(f"ERROR: {e}")
        return None
    if rates is None or len(rates) == 0:
        print(f"fetch_candles failed: {mt5.last_error()}")
        return None
    return _rates_to_df(rates)


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------
COMMISSION_PER_LOT = 7.0  # USD per side
CONTRACT_SIZE = 100       # Gold: 100 oz per lot
SPREAD_HALF = 0.06        # 6 pips each side (12 pips total spread)


class TradeResult:
    __slots__ = (
        "entry_time", "direction", "entry_price", "sl", "tp1", "tp2",
        "exit_time", "exit_price", "exit_reason",
        "pnl_gross", "commission", "pnl_net",
        "signal_strength", "fvg_bottom", "fvg_top",
        "session",
    )

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)


def simulate_trade(
    direction: str,
    entry_price: float,
    sl: float,
    tp1: float,
    tp2: float,
    df_m1: pd.DataFrame,
    entry_idx: int,
    lot: float,
    signal_strength: float,
    fvg_bottom: float,
    fvg_top: float,
    entry_time: datetime,
) -> TradeResult:
    """Walk forward candle-by-candle after entry and simulate TP/SL/timeout."""
    tr = TradeResult()
    tr.entry_time = entry_time
    tr.direction = direction
    tr.entry_price = entry_price
    tr.sl = sl
    tr.tp1 = tp1
    tr.tp2 = tp2
    tr.signal_strength = signal_strength
    tr.fvg_bottom = fvg_bottom
    tr.fvg_top = fvg_top

    is_long = direction == "LONG"
    remaining_pct = 1.0       # fraction of position still open
    realised_pnl = 0.0        # cumulative gross pnl in price terms
    current_sl = sl
    tp1_hit = False
    max_bars = 15
    stale_check_bar = 5       # after 5 bars, check if trade has momentum
    stale_threshold = 0.25    # must have moved 25% toward TP1

    n = len(df_m1)
    for offset in range(1, max_bars + 1):
        bar_idx = entry_idx + offset
        if bar_idx >= n:
            # Ran out of data -- close at last available bar
            last_bar = df_m1.iloc[min(bar_idx - 1, n - 1)]
            exit_price = float(last_bar["close"])
            tr.exit_reason = "DATA_END"
            tr.exit_time = last_bar.name if hasattr(last_bar, "name") else entry_time
            price_pnl = (exit_price - entry_price) if is_long else (entry_price - exit_price)
            realised_pnl += price_pnl * remaining_pct
            remaining_pct = 0.0
            break

        bar = df_m1.iloc[bar_idx]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])
        bar_time = bar.name

        # --- Check SL first (pessimistic) ---
        sl_hit = (bar_low <= current_sl) if is_long else (bar_high >= current_sl)
        if sl_hit:
            price_pnl = (current_sl - entry_price) if is_long else (entry_price - current_sl)
            realised_pnl += price_pnl * remaining_pct
            remaining_pct = 0.0
            tr.exit_price = current_sl
            tr.exit_time = bar_time
            tr.exit_reason = "SL" if not tp1_hit else "BE"
            break

        # --- Check TP1 ---
        if not tp1_hit:
            tp1_hit_cond = (bar_high >= tp1) if is_long else (bar_low <= tp1)
            if tp1_hit_cond:
                price_pnl = (tp1 - entry_price) if is_long else (entry_price - tp1)
                close_pct = 0.60 * remaining_pct
                realised_pnl += price_pnl * close_pct
                remaining_pct -= close_pct
                tp1_hit = True
                current_sl = entry_price  # move SL to break-even

        # --- Check TP2 ---
        if tp1_hit and remaining_pct > 0:
            tp2_hit_cond = (bar_high >= tp2) if is_long else (bar_low <= tp2)
            if tp2_hit_cond:
                price_pnl = (tp2 - entry_price) if is_long else (entry_price - tp2)
                realised_pnl += price_pnl * remaining_pct
                remaining_pct = 0.0
                tr.exit_price = tp2
                tr.exit_time = bar_time
                tr.exit_reason = "TP2"
                break

        # --- Stale trade exit: no momentum after N bars ---
        if offset == stale_check_bar and not tp1_hit:
            tp1_dist = abs(tp1 - entry_price)
            if tp1_dist > 0:
                progress = (bar_close - entry_price) / tp1_dist if is_long else (entry_price - bar_close) / tp1_dist
                if progress < stale_threshold:
                    exit_price = bar_close
                    price_pnl = (exit_price - entry_price) if is_long else (entry_price - exit_price)
                    realised_pnl += price_pnl * remaining_pct
                    remaining_pct = 0.0
                    tr.exit_price = exit_price
                    tr.exit_time = bar_time
                    tr.exit_reason = "STALE"
                    break

        # --- Timeout ---
        if offset >= max_bars:
            exit_price = bar_close
            price_pnl = (exit_price - entry_price) if is_long else (entry_price - exit_price)
            realised_pnl += price_pnl * remaining_pct
            remaining_pct = 0.0
            tr.exit_price = exit_price
            tr.exit_time = bar_time
            tr.exit_reason = "TIMEOUT"
            break

    # If loop ended without break (shouldn't happen, but guard)
    if remaining_pct > 0:
        tr.exit_reason = tr.exit_reason or "TIMEOUT"

    # If only TP1 was hit and position still open at end
    if tr.exit_reason is None and tp1_hit:
        tr.exit_reason = "TP1"
        tr.exit_price = tp1

    # Calculate monetary PnL
    tr.pnl_gross = round(realised_pnl * CONTRACT_SIZE * lot, 2)
    tr.commission = round(COMMISSION_PER_LOT * lot * 2, 2)  # round-trip
    tr.pnl_net = round(tr.pnl_gross - tr.commission, 2)

    # Session tag
    hour = entry_time.hour if isinstance(entry_time, datetime) else 0
    if 7 <= hour < 12:
        tr.session = "London"
    elif 13 <= hour < 18:
        tr.session = "New York"
    else:
        tr.session = "Other"

    return tr


# ---------------------------------------------------------------------------
# Rejection category mapping
# ---------------------------------------------------------------------------
REJECTION_MAP = {
    "SESSION_BLOCKED": "session_gate",
    "DAILY_NEUTRAL": "daily_neutral",
    "LOW_ATR": "low_atr",
    "NO_PULLBACK": "no_pullback",
    "BELOW_EMA50": "below_ema50",
    "ABOVE_EMA50": "above_ema50",
    "RSI_INVALID": "rsi_invalid",
    "NO_REVERSAL": "no_reversal",
    "WEAK_CANDLE": "weak_candle",
    "COOLDOWN": "cooldown",
    "SL_TOO_WIDE": "sl_too_wide",
    "SL_TOO_TIGHT": "sl_too_tight",
    "TP1_TOO_SMALL": "tp1_too_small",
    "ZERO_RANGE_CANDLE": "zero_range",
    "Insufficient M1 data": "insufficient_data",
    "NaN in M1 indicators": "nan_indicators",
}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    trades: List[TradeResult],
    rejections: Dict[str, int],
    total_candles: int,
    elapsed_sec: float,
) -> str:
    """Build the full report as a string."""
    out = StringIO()

    def w(s=""):
        out.write(s + "\n")

    w("=" * 70)
    w("  DAILY TREND FOLLOWER -- BACKTEST REPORT")
    w("=" * 70)
    w(f"  Candles analysed : {total_candles:,}")
    w(f"  Execution time   : {elapsed_sec:.1f}s")
    w()

    # -- PERFORMANCE GLOBALE ----------------------------------------
    n = len(trades)
    if n == 0:
        w("  NO TRADES GENERATED")
        w()
        _write_rejections(w, rejections, total_candles)
        return out.getvalue()

    pnls = [t.pnl_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n * 100
    total_pnl = sum(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    biggest_win = max(pnls)
    biggest_loss = min(pnls)
    expectancy = (len(wins) / n * avg_win) + (len(losses) / n * avg_loss) if n > 0 else 0

    # Drawdown (starting equity = $10,000 as reference for %)
    starting_equity = 10_000.0
    cumulative = np.cumsum(pnls)
    equity_curve = starting_equity + cumulative
    peak = np.maximum.accumulate(equity_curve)
    drawdown = peak - equity_curve
    max_dd = drawdown.max()
    max_dd_idx = np.argmax(drawdown)
    max_dd_pct = max_dd / peak[max_dd_idx] * 100 if peak[max_dd_idx] > 0 else 0

    # Sharpe (annualised, trades as daily returns proxy)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
    else:
        sharpe = 0.0

    w("-- PERFORMANCE GLOBALE " + "-" * 47)
    w(f"  Total trades     : {n}")
    w(f"  Win Rate         : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    w(f"  Profit Factor    : {profit_factor:.2f}")
    w(f"  Total P&L        : ${total_pnl:.2f}")
    w(f"  Avg Win          : ${avg_win:.2f}")
    w(f"  Avg Loss         : ${avg_loss:.2f}")
    w(f"  Win/Loss ratio   : {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Win/Loss ratio   : inf")
    w(f"  Biggest win      : ${biggest_win:.2f}")
    w(f"  Biggest loss     : ${biggest_loss:.2f}")
    w(f"  Max drawdown     : ${max_dd:.2f}  ({max_dd_pct:.1f}%)")
    w(f"  Sharpe (annual)  : {sharpe:.2f}")
    w(f"  Expectancy/trade : ${expectancy:.2f}")
    w()

    # -- PAR SESSION ------------------------------------------------
    w("-- PAR SESSION " + "-" * 55)
    for session_name in ["London", "New York"]:
        st = [t for t in trades if t.session == session_name]
        if st:
            sw = [t.pnl_net for t in st if t.pnl_net > 0]
            wr = len(sw) / len(st) * 100
            sp = sum(t.pnl_net for t in st)
            w(f"  {session_name:12s}: {len(st):3d} trades | WR {wr:5.1f}% | P&L ${sp:8.2f}")
        else:
            w(f"  {session_name:12s}:   0 trades")
    blocked = rejections.get("session_gate", 0)
    w(f"  Hors session     : {blocked:,} signaux bloques par session gate")
    w()

    # -- PAR DIRECTION ----------------------------------------------
    w("-- PAR DIRECTION " + "-" * 52)
    for dir_name in ["LONG", "SHORT"]:
        dt = [t for t in trades if t.direction == dir_name]
        if dt:
            dw = [t.pnl_net for t in dt if t.pnl_net > 0]
            wr = len(dw) / len(dt) * 100
            dp = sum(t.pnl_net for t in dt)
            w(f"  {dir_name:6s}: {len(dt):3d} trades | WR {wr:5.1f}% | P&L ${dp:8.2f}")
        else:
            w(f"  {dir_name:6s}:   0 trades")
    w()

    # -- PAR MOIS ---------------------------------------------------
    w("-- PAR MOIS " + "-" * 57)
    w(f"  {'Month':10s} {'Trades':>7s} {'WR':>7s} {'P&L':>10s} {'PF':>7s}")
    w(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*10} {'-'*7}")
    monthly: Dict[str, List[TradeResult]] = defaultdict(list)
    for t in trades:
        if t.entry_time:
            key = t.entry_time.strftime("%Y-%m")
            monthly[key].append(t)
    for month_key in sorted(monthly.keys()):
        mt_list = monthly[month_key]
        mt_pnls = [t.pnl_net for t in mt_list]
        mt_wins = [p for p in mt_pnls if p > 0]
        mt_losses_neg = [p for p in mt_pnls if p <= 0]
        mt_wr = len(mt_wins) / len(mt_list) * 100 if mt_list else 0
        mt_total = sum(mt_pnls)
        mt_gp = sum(mt_wins) if mt_wins else 0
        mt_gl = abs(sum(mt_losses_neg)) if mt_losses_neg else 0
        mt_pf = mt_gp / mt_gl if mt_gl > 0 else float("inf")
        pf_str = f"{mt_pf:.2f}" if mt_pf < 999 else "inf"
        w(f"  {month_key:10s} {len(mt_list):7d} {mt_wr:6.1f}% ${mt_total:9.2f} {pf_str:>7s}")
    w()

    # -- ANALYSE DES REJETS -----------------------------------------
    _write_rejections(w, rejections, total_candles)

    # -- EXIT REASONS -----------------------------------------------
    w("-- EXIT REASONS " + "-" * 53)
    exit_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        exit_counts[t.exit_reason or "UNKNOWN"] += 1
    for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        w(f"  {reason:12s}: {count:4d}  ({count/n*100:5.1f}%)")
    w()

    return out.getvalue()


def _write_rejections(w, rejections: Dict[str, int], total_candles: int):
    w("-- ANALYSE DES REJETS " + "-" * 47)
    w(f"  Total cycles M1 analyses : {total_candles:,}")
    total_rej = sum(rejections.values())
    w(f"  Total rejets             : {total_rej:,}")
    w()
    # Sort by count descending
    for reason, count in sorted(rejections.items(), key=lambda x: -x[1]):
        pct = count / total_candles * 100 if total_candles > 0 else 0
        w(f"  {reason:20s}: {count:7,}  ({pct:5.1f}%)")
    w()


# ---------------------------------------------------------------------------
# Daily data alignment (no-lookahead)
# ---------------------------------------------------------------------------
def _get_daily_window(df_daily: pd.DataFrame, candle_time: datetime, lookback: int = 10) -> pd.DataFrame:
    """Return the most recent `lookback` completed daily bars before candle_time.

    Uses searchsorted to avoid lookahead — only bars strictly before candle_time's date.
    """
    candle_ts = pd.Timestamp(candle_time)
    # Daily bars are indexed by date; we want bars before today
    idx = np.searchsorted(df_daily.index.values, candle_ts.value, side="left")
    # idx points to first bar >= candle_time; use idx-1 as last completed bar
    end = idx
    start = max(0, end - lookback)
    if end <= 0:
        return df_daily.iloc[:0]  # empty
    return df_daily.iloc[start:end]


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------
def run_backtest(
    symbol: str = "XAUUSD",
    days: int = 90,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    lot: float = 0.04,
) -> None:
    start_wall = _real_time.time()

    # -- Connect MT5 -----------------------------------------------
    print("Connecting to MT5...")
    if not connect_mt5(symbol):
        sys.exit(1)
    print("MT5 connected.")

    # -- Fetch data ------------------------------------------------
    if date_from and date_to:
        dt_from = datetime.strptime(date_from, "%Y-%m-%d")
        dt_to = datetime.strptime(date_to, "%Y-%m-%d")
    else:
        dt_to = datetime.utcnow()
        dt_from = dt_to - timedelta(days=days)

    # DTF needs M1 + Daily data
    print(f"Fetching {symbol} M1 data...")
    df_m1 = _fetch_m1(symbol, dt_from - timedelta(hours=4), dt_to)

    print(f"Fetching {symbol} D1 data...")
    # Fetch extra daily history for EMA50 warm-up (~60 bars)
    df_daily = _fetch_daily(symbol, dt_from - timedelta(days=120), dt_to)

    mt5.shutdown()

    # -- Compute indicators ----------------------------------------
    print("Computing M1 indicators...")
    df_m1 = compute_indicators(df_m1)

    print("Computing Daily indicators...")
    df_daily = compute_daily_indicators(df_daily)
    print(f"  D1 bars: {len(df_daily)}, last ema_20={df_daily['ema_20'].iloc[-1]:.2f}, ema_50={df_daily['ema_50'].iloc[-1]:.2f}")

    # Trim to target date range (after warm-up)
    eval_mask = (df_m1.index >= pd.Timestamp(dt_from)) & (df_m1.index < pd.Timestamp(dt_to))
    eval_indices = np.where(eval_mask)[0]

    print(f"  Evaluating {len(eval_indices):,} M1 candles in [{dt_from.date()} -> {dt_to.date()})")

    # -- Strategy setup --------------------------------------------
    settings = get_settings()
    strategy = DailyTrendFollower()

    # Install backtest clock into gold_scalper module
    import strategies.gold_scalper as _gs_mod
    _original_datetime = _gs_mod.datetime
    _original_time = _gs_mod.time
    clock = _BacktestClock()
    _gs_mod.datetime = clock
    _gs_mod.time = clock

    # -- Walk-forward loop -----------------------------------------
    trades: List[TradeResult] = []
    rejections: Dict[str, int] = defaultdict(int)
    in_trade = False
    trade_exit_idx = -1  # index of last bar used by the current trade

    # Pre-compute arrays for fast access (avoid repeated pandas indexing)
    _m1_index_values = df_m1.index.values                    # numpy datetime64
    _m1_hours = df_m1.index.hour.values                      # numpy int array
    _m1_close_values = df_m1["close"].values.astype(float)   # numpy float array

    # Parse session windows for pre-filter
    _session_windows = strategy.default_params()["session_windows"]
    if settings:
        raw_sw = getattr(settings, "DTF_SESSION_WINDOWS", None)
        if isinstance(raw_sw, str):
            _session_windows = []
            for chunk in raw_sw.split(","):
                parts = chunk.strip().split("-")
                if len(parts) == 2:
                    _session_windows.append((int(parts[0]), int(parts[1])))

    # Cache daily windows to avoid re-slicing every candle
    _last_daily_date = None
    _cached_daily_window = None

    try:
        for i in progress_iter(eval_indices, total=len(eval_indices), desc="Backtesting"):
            # Skip if we are inside a trade's duration
            if in_trade and i <= trade_exit_idx:
                continue
            in_trade = False

            # -- Pre-filter: session gate (skip ~57% of candles cheaply) --
            hour = int(_m1_hours[i])
            in_session = any(s <= hour < e for s, e in _session_windows)
            if not in_session:
                rejections["session_gate"] += 1
                continue

            # -- Build M1 window (last 100 candles up to i inclusive) --
            m1_start = max(0, i - 99)
            df_m1_window = df_m1.iloc[m1_start: i + 1]
            if len(df_m1_window) < 50:
                rejections["insufficient_data"] += 1
                continue

            # -- Set backtest clock ------------------------------------
            candle_time = pd.Timestamp(_m1_index_values[i]).to_pydatetime()
            clock.current_dt = candle_time
            clock.current_ts = candle_time.timestamp()

            close_price = _m1_close_values[i]

            # -- Get daily window (no-lookahead, cached per date) ------
            candle_date = candle_time.date()
            if candle_date != _last_daily_date:
                _cached_daily_window = _get_daily_window(df_daily, candle_time, lookback=10)
                _last_daily_date = candle_date

            # -- Call strategy -----------------------------------------
            signal = strategy.compute_signal(
                df_m1_window,
                settings=settings,
                df_daily=_cached_daily_window,
            )

            # -- Track rejection ---------------------------------------
            if signal.direction == SignalDirection.FLAT:
                reason = signal.entry_reason
                category = REJECTION_MAP.get(reason, reason.lower() if reason else "unknown")
                rejections[category] += 1
                continue

            # -- Signal generated -- simulate trade ---------------------
            direction = signal.direction.value  # "LONG" or "SHORT"
            if direction == "LONG":
                entry_price = close_price + SPREAD_HALF  # buy at ask
            else:
                entry_price = close_price - SPREAD_HALF  # sell at bid

            tr = simulate_trade(
                direction=direction,
                entry_price=entry_price,
                sl=signal.sl,
                tp1=signal.tp1,
                tp2=signal.tp2,
                df_m1=df_m1,
                entry_idx=i,
                lot=lot,
                signal_strength=signal.strength,
                fvg_bottom=0,
                fvg_top=0,
                entry_time=candle_time,
            )
            trades.append(tr)

            # Mark bars consumed by this trade so we don't re-enter
            if tr.exit_time and isinstance(tr.exit_time, (datetime, pd.Timestamp)):
                # Find the M1 index of exit_time
                exit_ts = pd.Timestamp(tr.exit_time)
                exit_positions = np.searchsorted(df_m1.index.values, exit_ts.value, side="right")
                trade_exit_idx = min(exit_positions, len(df_m1) - 1)
            else:
                trade_exit_idx = min(i + 20, len(df_m1) - 1)
            in_trade = True

    finally:
        # Restore original modules
        _gs_mod.datetime = _original_datetime
        _gs_mod.time = _original_time

    elapsed = _real_time.time() - start_wall

    # -- Generate report -------------------------------------------
    report = generate_report(trades, rejections, len(eval_indices), elapsed)

    # Print to console
    print()
    print(report)

    # Save report
    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "backtest_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Save trade details CSV
    csv_path = os.path.join("results", "backtest_trades.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "entry_time", "direction", "entry_price", "sl", "tp1", "tp2",
            "exit_time", "exit_price", "exit_reason",
            "pnl_gross", "commission", "pnl_net",
            "signal_strength", "fvg_bottom", "fvg_top",
        ])
        for t in trades:
            writer.writerow([
                t.entry_time, t.direction, t.entry_price, t.sl, t.tp1, t.tp2,
                t.exit_time, t.exit_price, t.exit_reason,
                t.pnl_gross, t.commission, t.pnl_net,
                t.signal_strength, t.fvg_bottom, t.fvg_top,
            ])
    print(f"Trade details saved to {csv_path}")
    print(f"Total execution time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Backtest Daily Trend Follower strategy")
    parser.add_argument("--days", type=int, default=90, help="Number of calendar days (default: 90)")
    parser.add_argument("--lot", type=float, default=0.04, help="Fixed lot size (default: 0.04)")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol (default: XAUUSD)")
    parser.add_argument("--from", dest="date_from", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", type=str, default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    # Validate date args
    if (args.date_from is None) != (args.date_to is None):
        print("ERROR: --from and --to must be specified together")
        sys.exit(1)

    run_backtest(
        symbol=args.symbol,
        days=args.days,
        date_from=args.date_from,
        date_to=args.date_to,
        lot=args.lot,
    )


if __name__ == "__main__":
    main()
