"""Vectorbt-backed backtesting engine with walk-forward and overfitting detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from config.settings import Settings
from strategies.base_strategy import BaseStrategy, Signal, SignalDirection


@dataclass
class BacktestResult:
    """Comprehensive backtest metrics."""
    strategy_name: str
    symbol: str
    params: Dict
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    total_return: float = 0.0
    avg_trade_pnl: float = 0.0
    is_oos: bool = False  # out-of-sample flag

    def passes_qualification(self, settings: Settings) -> bool:
        return (
            self.sharpe_ratio > settings.MIN_OOS_SHARPE
            and self.profit_factor > settings.MIN_PROFIT_FACTOR
            and self.max_drawdown < settings.MAX_DRAWDOWN_PCT
            and self.win_rate > settings.MIN_WIN_RATE
            and self.trade_count >= settings.MIN_QUALIFYING_TRADES
        )


class BacktestEngine:
    """Run vectorized backtests and compute metrics."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        symbol: str,
        spread_cost: float = 0.0,
    ) -> BacktestResult:
        """Run a full backtest on the provided DataFrame."""
        if len(df) < 100:
            logger.warning("DataFrame too short for backtest: {} rows", len(df))
            return BacktestResult(
                strategy_name=strategy.name, symbol=symbol, params=strategy.params
            )

        # Generate signals for every row
        entries_long, entries_short, exits = self._generate_signals(strategy, df)

        # Apply spread as a cost per trade
        close = df["close"].values.astype(float)

        try:
            # Run long trades
            pf_long = vbt.Portfolio.from_signals(
                close=df["close"],
                entries=entries_long,
                exits=exits,
                direction="longonly",
                fees=spread_cost,
                freq="1T",  # 1-minute frequency
                init_cash=10000,
            )

            # Run short trades
            pf_short = vbt.Portfolio.from_signals(
                close=df["close"],
                entries=entries_short,
                exits=exits,
                direction="shortonly",
                fees=spread_cost,
                freq="1T",
                init_cash=10000,
            )

            # Combine metrics
            result = self._compute_metrics(pf_long, pf_short, strategy, symbol)
            return result

        except Exception as e:
            logger.error("Backtest failed for {} on {}: {}", strategy.name, symbol, e)
            return BacktestResult(
                strategy_name=strategy.name, symbol=symbol, params=strategy.params
            )

    def _generate_signals(
        self, strategy: BaseStrategy, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate entry/exit signals for every bar in the DataFrame."""
        n = len(df)
        entries_long = pd.Series(False, index=df.index)
        entries_short = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        # We need enough history for indicators, so start from bar 60
        min_bars = 60
        for i in range(min_bars, n):
            window = df.iloc[: i + 1]
            signal = strategy.compute_signal(window)

            if signal.direction == SignalDirection.LONG and signal.strength >= 0.5:
                entries_long.iloc[i] = True
            elif signal.direction == SignalDirection.SHORT and signal.strength >= 0.5:
                entries_short.iloc[i] = True

        return entries_long, entries_short, exits

    def _compute_metrics(
        self,
        pf_long: vbt.Portfolio,
        pf_short: vbt.Portfolio,
        strategy: BaseStrategy,
        symbol: str,
    ) -> BacktestResult:
        """Extract metrics from vectorbt portfolios."""
        # Combine stats from both long and short
        try:
            stats_l = pf_long.stats()
            stats_s = pf_short.stats()

            total_trades = int(stats_l.get("Total Trades", 0)) + int(stats_s.get("Total Trades", 0))
            total_return = float(pf_long.total_return()) + float(pf_short.total_return())

            # Use long portfolio for primary metrics (usually dominant)
            primary = pf_long if int(stats_l.get("Total Trades", 0)) >= int(stats_s.get("Total Trades", 0)) else pf_short
            p_stats = primary.stats()

            sharpe = float(p_stats.get("Sharpe Ratio", 0)) if not np.isnan(p_stats.get("Sharpe Ratio", 0)) else 0.0
            sortino = float(p_stats.get("Sortino Ratio", 0)) if not np.isnan(p_stats.get("Sortino Ratio", 0)) else 0.0
            calmar = float(p_stats.get("Calmar Ratio", 0)) if not np.isnan(p_stats.get("Calmar Ratio", 0)) else 0.0
            max_dd = abs(float(p_stats.get("Max Drawdown [%]", 0))) / 100.0
            win_rate = float(p_stats.get("Win Rate [%]", 0)) / 100.0

            # Profit factor
            trades = primary.trades.records_readable
            if len(trades) > 0 and "PnL" in trades.columns:
                gross_profit = trades.loc[trades["PnL"] > 0, "PnL"].sum()
                gross_loss = abs(trades.loc[trades["PnL"] < 0, "PnL"].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0
            else:
                profit_factor = 0.0

        except Exception as e:
            logger.error("Metrics computation error: {}", e)
            return BacktestResult(
                strategy_name=strategy.name, symbol=symbol, params=strategy.params
            )

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            params=dict(strategy.params),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            profit_factor=round(profit_factor, 4),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            trade_count=total_trades,
            total_return=round(total_return, 4),
        )

    def train_test_split(
        self, df: pd.DataFrame, train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """80/20 train/test split."""
        split_idx = int(len(df) * train_ratio)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def walk_forward(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        symbol: str,
        train_days: int = 30,
        test_days: int = 10,
        spread_cost: float = 0.0,
    ) -> List[BacktestResult]:
        """Walk-forward analysis with rolling windows."""
        results = []
        bars_per_day = 24 * 60  # 1m bars
        train_bars = train_days * bars_per_day
        test_bars = test_days * bars_per_day
        window = train_bars + test_bars

        if len(df) < window:
            logger.warning("Not enough data for walk-forward: {} < {}", len(df), window)
            return results

        start = 0
        while start + window <= len(df):
            train_df = df.iloc[start : start + train_bars]
            test_df = df.iloc[start + train_bars : start + window]

            result = self.run(strategy, test_df, symbol, spread_cost)
            result.is_oos = True
            results.append(result)

            start += test_bars  # slide by test window size

        return results

    def parameter_sensitivity(
        self,
        strategy_class,
        base_params: Dict,
        df: pd.DataFrame,
        symbol: str,
        spread_cost: float = 0.0,
        perturbation: float = 0.10,
    ) -> Tuple[bool, float]:
        """Test parameter sensitivity — Sharpe must not drop > threshold at ±10% perturbation."""
        base_strategy = strategy_class(params=dict(base_params))
        base_result = self.run(base_strategy, df, symbol, spread_cost)
        base_sharpe = base_result.sharpe_ratio

        if base_sharpe <= 0:
            return False, 1.0

        max_drop = 0.0
        for param_name, value in base_params.items():
            if not isinstance(value, (int, float)):
                continue

            for direction in [1 + perturbation, 1 - perturbation]:
                perturbed = dict(base_params)
                new_val = value * direction
                perturbed[param_name] = int(new_val) if isinstance(value, int) else new_val

                test_strategy = strategy_class(params=perturbed)
                test_result = self.run(test_strategy, df, symbol, spread_cost)

                if base_sharpe > 0:
                    drop = (base_sharpe - test_result.sharpe_ratio) / base_sharpe
                    max_drop = max(max_drop, drop)

        passes = max_drop < self._settings.PARAM_SENSITIVITY_THRESHOLD
        return passes, max_drop
