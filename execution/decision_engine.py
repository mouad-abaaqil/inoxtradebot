"""Pre-trade decision checklist — all conditions must pass before order placement."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import Settings
from data.market_data import MarketDataModule
from execution.risk_engine import RiskEngine, RiskDecision
from sentiment.sentiment_analyzer import SentimentAnalyzer
from storage.db import Database
from storage.models import SignalLog
from strategies.base_strategy import Signal, SignalDirection


class DecisionEngine:
    """Orchestrates the pre-trade checklist before allowing order placement."""

    def __init__(
        self,
        settings: Settings,
        risk_engine: RiskEngine,
        market_data: MarketDataModule,
        sentiment: SentimentAnalyzer,
        database: Database,
    ):
        self._settings = settings
        self._risk = risk_engine
        self._market_data = market_data
        self._sentiment = sentiment
        self._db = database

    def evaluate_signal(self, signal: Signal) -> Optional[RiskDecision]:
        """Run the full pre-trade checklist.

        Returns RiskDecision if approved, None if rejected.
        Every signal (accepted or rejected) is logged for audit.
        """
        # Step 1: Sentiment check (stub: always True)
        if not self._sentiment.sentiment_allows(signal.direction.value):
            self._log_rejection(signal, "Sentiment filter blocked trade")
            return None

        # Step 2: High-impact event check
        if self._sentiment.is_high_impact_event_near():
            self._log_rejection(signal, "High-impact news event approaching")
            return None

        # Step 3: Full risk evaluation (includes all hard stops)
        risk_decision = self._risk.evaluate(signal)

        if not risk_decision.approved:
            self._log_rejection(signal, risk_decision.rejection_reason)
            logger.info(
                "Signal REJECTED for {}: {} | {}",
                signal.symbol, signal.strategy_name, risk_decision.rejection_reason,
            )
            return None

        # All checks passed
        self._log_acceptance(signal, risk_decision)
        logger.info(
            "Signal APPROVED for {}: {} | lots={} SL={} TP1={} TP2={}",
            signal.symbol, signal.strategy_name,
            risk_decision.lot_size, risk_decision.stop_loss,
            risk_decision.tp1, risk_decision.tp2,
        )
        return risk_decision

    def _log_rejection(self, signal: Signal, reason: str) -> None:
        log = SignalLog(
            symbol=signal.symbol,
            strategy_name=signal.strategy_name,
            direction=signal.direction.value,
            strength=signal.strength,
            confidence=signal.confidence,
            entry_reason=signal.entry_reason,
            accepted=False,
            rejection_reason=reason,
            indicators_json=json.dumps(signal.indicators_used),
        )
        self._db.log_signal(log)

    def _log_acceptance(self, signal: Signal, decision: RiskDecision) -> None:
        log = SignalLog(
            symbol=signal.symbol,
            strategy_name=signal.strategy_name,
            direction=signal.direction.value,
            strength=signal.strength,
            confidence=signal.confidence,
            entry_reason=signal.entry_reason,
            accepted=True,
            rejection_reason=None,
            indicators_json=json.dumps(signal.indicators_used),
        )
        self._db.log_signal(log)
