"""Optuna-based strategy optimizer with walk-forward validation."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional, Type

import optuna
from loguru import logger

from config.settings import Settings
from data.feature_engine import FeatureEngine
from data.market_data import MarketDataModule, Timeframe
from optimization.backtester import BacktestEngine, BacktestResult
from storage.db import Database
from storage.models import StrategyRecord
from strategies.base_strategy import BaseStrategy
from strategies.registry import StrategyRegistry


class StrategyOptimizer:
    """Run Optuna optimization for all registered strategies."""

    def __init__(
        self,
        settings: Settings,
        market_data: MarketDataModule,
        feature_engine: FeatureEngine,
        backtester: BacktestEngine,
        registry: StrategyRegistry,
        database: Database,
    ):
        self._settings = settings
        self._market_data = market_data
        self._feature_engine = feature_engine
        self._backtester = backtester
        self._registry = registry
        self._db = database

    async def run_optimization(self) -> Dict[str, Optional[BacktestResult]]:
        """Run full optimization cycle for all symbols and strategies."""
        logger.info("=== Starting optimization cycle ===")
        best_results: Dict[str, Optional[BacktestResult]] = {}

        for symbol in self._settings.SYMBOLS:
            best = await self._optimize_symbol(symbol)
            best_results[symbol] = best

            if best and best.passes_qualification(self._settings):
                self._promote_strategy(symbol, best)
            else:
                logger.info("No qualifying strategy found for {}", symbol)

        logger.info("=== Optimization cycle complete ===")
        return best_results

    async def _optimize_symbol(self, symbol: str) -> Optional[BacktestResult]:
        """Find the best strategy + params for a single symbol."""
        df = self._market_data.get_dataframe(symbol, Timeframe.M1)
        if df is None or len(df) < 1000:
            logger.warning("Insufficient data for optimization: {} has {} bars",
                           symbol, len(df) if df is not None else 0)
            return None

        # Ensure indicators are computed
        self._feature_engine.compute_for_symbol(symbol)
        df = self._market_data.get_dataframe(symbol, Timeframe.M1)

        # Calculate spread cost
        avg_spread = self._market_data.get_average_spread(symbol)
        spread_cost = avg_spread * 1.5  # realistic spread

        train_df, test_df = self._backtester.train_test_split(df)

        best_result: Optional[BacktestResult] = None
        best_strategy_name: Optional[str] = None

        for strategy_name in self._registry.registered_names:
            strategy_class = self._registry.get_strategy_class(strategy_name)
            if strategy_class is None:
                continue

            logger.info("Optimizing {} for {}", strategy_name, symbol)

            try:
                result = self._optimize_single(
                    strategy_class, train_df, test_df, symbol, spread_cost
                )
                if result is None:
                    continue

                # Record in database
                self._save_record(result, is_oos=True)

                if best_result is None or result.sharpe_ratio > best_result.sharpe_ratio:
                    best_result = result
                    best_strategy_name = strategy_name

            except Exception as e:
                logger.error("Optimization failed for {} on {}: {}", strategy_name, symbol, e)

        if best_result:
            logger.info(
                "Best for {}: {} (Sharpe={:.3f}, PF={:.3f}, WR={:.1%}, trades={})",
                symbol, best_result.strategy_name, best_result.sharpe_ratio,
                best_result.profit_factor, best_result.win_rate, best_result.trade_count,
            )

        return best_result

    def _optimize_single(
        self,
        strategy_class: Type[BaseStrategy],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        symbol: str,
        spread_cost: float,
    ) -> Optional[BacktestResult]:
        """Run Optuna optimization for a single strategy class."""

        def objective(trial: optuna.Trial) -> float:
            instance = strategy_class()
            params = instance.param_space(trial)
            candidate = strategy_class(params=params)
            result = self._backtester.run(candidate, train_df, symbol, spread_cost)

            if result.trade_count < 10:
                raise optuna.TrialPruned()

            return result.sharpe_ratio

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(objective, n_trials=self._settings.OPTUNA_TRIALS, show_progress_bar=False)

        if len(study.trials) == 0 or study.best_trial is None:
            return None

        best_params = study.best_trial.params
        best_candidate = strategy_class(params=best_params)

        # Validate on OOS test data
        oos_result = self._backtester.run(best_candidate, test_df, symbol, spread_cost)
        oos_result.is_oos = True

        # Parameter sensitivity check
        passes_sensitivity, max_drop = self._backtester.parameter_sensitivity(
            strategy_class, best_params, test_df, symbol, spread_cost
        )
        if not passes_sensitivity:
            logger.warning(
                "{} failed sensitivity test (max Sharpe drop: {:.1%})",
                strategy_class.name, max_drop,
            )

        return oos_result

    def _promote_strategy(self, symbol: str, result: BacktestResult) -> None:
        """Promote the best strategy to active status."""
        current = self._registry.get_active(symbol)

        # Only promote if better than current
        if current:
            current_result = self._backtester.run(
                current,
                self._market_data.get_dataframe(symbol, Timeframe.M1),
                symbol,
            )
            if current_result.sharpe_ratio >= result.sharpe_ratio:
                logger.info(
                    "Current {} still better for {} (Sharpe {:.3f} >= {:.3f})",
                    current.name, symbol, current_result.sharpe_ratio, result.sharpe_ratio,
                )
                return

        self._registry.set_active(symbol, result.strategy_name, result.params)
        self._db.deactivate_strategies(symbol, result.strategy_name)
        self._save_record(result, is_oos=True, is_active=True)
        logger.info("Promoted {} for {} (Sharpe={:.3f})", result.strategy_name, symbol, result.sharpe_ratio)

    def _save_record(
        self, result: BacktestResult, is_oos: bool = False, is_active: bool = False
    ) -> None:
        record = StrategyRecord(
            strategy_name=result.strategy_name,
            symbol=result.symbol,
            params_json=json.dumps(result.params),
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.sortino_ratio,
            calmar_ratio=result.calmar_ratio,
            profit_factor=result.profit_factor,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            trade_count=result.trade_count,
            total_return=result.total_return,
            is_oos=is_oos,
            is_active=is_active,
        )
        self._db.save_strategy_record(record)


# Need this import for type hints in _optimize_single
import pandas as pd
