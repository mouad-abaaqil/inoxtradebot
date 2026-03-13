"""Main entry point — wires all modules together and runs the async event loop."""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config.settings import Settings, get_settings
from data.feature_engine import FeatureEngine
from data.market_data import MarketDataModule, NewCandleEvent, Timeframe
from execution.decision_engine import DecisionEngine
from execution.order_executor import OrderExecutor
from execution.risk_engine import RiskEngine
from execution.trade_manager import TradeManager
from notifications.telegram_bot import TelegramNotifier
from optimization.backtester import BacktestEngine
from optimization.optimizer import StrategyOptimizer
from sentiment.sentiment_analyzer import SentimentAnalyzer
from storage.db import Database
from storage.models import AccountSnapshot, Trade
from strategies.base_strategy import SignalDirection
from strategies.registry import StrategyRegistry
from strategies.gold_scalper import GoldScalper
from strategies.rsi_reversal import RSIReversalScalper
from strategies.macd_crossover import MACDZeroCrossScalper
from strategies.bb_squeeze import BBSqueezeBreakout
from strategies.vwap_bounce import VWAPBounceScalper
from strategies.ema_pullback import EMAPullbackScalper
from strategies.volume_surge import VolumeSurgeMomentum


def setup_logging(settings: Settings) -> None:
    """Configure loguru with rotation and structured output."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
    )
    logger.add(
        "logs/scalping_bot_{time:YYYY-MM-DD}.log",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}",
    )


class ScalpingBot:
    """Main bot orchestrator — initializes and runs all modules."""

    def __init__(self):
        self._settings = get_settings()
        setup_logging(self._settings)

        # Event bus
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # Initialize modules
        self._db = Database(self._settings.DATABASE_URL)
        self._market_data = MarketDataModule(self._settings, self._event_queue)
        self._feature_engine = FeatureEngine(self._market_data, self._event_queue)
        self._sentiment = SentimentAnalyzer()

        # Strategy registry — GoldScalper is default
        self._registry = StrategyRegistry()
        self._registry.register_all([
            GoldScalper,
            RSIReversalScalper,
            MACDZeroCrossScalper,
            BBSqueezeBreakout,
            VWAPBounceScalper,
            EMAPullbackScalper,
            VolumeSurgeMomentum,
        ])
        self._registry.set_default("Gold_Scalper")

        # Execution
        self._risk_engine = RiskEngine(self._settings, self._market_data, self._db)
        self._executor = OrderExecutor(self._settings, self._db)
        self._decision_engine = DecisionEngine(
            self._settings, self._risk_engine, self._market_data, self._sentiment, self._db
        )
        self._trade_manager = TradeManager(
            self._settings,
            self._market_data,
            self._executor,
            self._db,
            on_tp_hit=self._on_tp_hit,
            on_trade_closed=self._on_trade_closed,
            risk_engine=self._risk_engine,
        )

        # Optimization
        self._backtester = BacktestEngine(self._settings)
        self._optimizer = StrategyOptimizer(
            self._settings, self._market_data, self._feature_engine,
            self._backtester, self._registry, self._db,
        )

        # Notifications
        self._telegram = TelegramNotifier(
            self._settings, self._db, self._registry,
            risk_engine=self._risk_engine,
            order_executor=self._executor,
        )

        # Scheduler
        self._scheduler = AsyncIOScheduler()

        self._running = False

    async def start(self) -> None:
        """Initialize everything and start the bot."""
        logger.info("=" * 50)
        logger.info("Gold Scalping Bot starting...")
        logger.info("Demo mode: {}", self._settings.DEMO_MODE)
        logger.info("Symbols: {}", self._settings.SYMBOLS)
        logger.info("Risk per trade: {}%", self._settings.RISK_PER_TRADE * 100)
        logger.info("ATR SL multiplier: {}", self._settings.ATR_SL_MULTIPLIER)
        logger.info("Pause on {} consecutive losses: {} min",
                     self._settings.CONSECUTIVE_LOSS_PAUSE,
                     self._settings.PAUSE_DURATION_MINUTES)
        logger.info("=" * 50)

        # Connect to MT5
        connected = await self._market_data.connect()
        if not connected:
            logger.critical("Cannot connect to MT5 — exiting")
            return

        # Load historical data
        await self._market_data.load_historical()

        # Initialize daily loss tracking (5% drawdown kill switch)
        self._risk_engine.initialize_daily()

        # Compute initial indicators
        for symbol in self._settings.SYMBOLS:
            self._feature_engine.compute_for_symbol(symbol)

        # Ensure strategies are active for all symbols
        for symbol in self._settings.SYMBOLS:
            self._registry.ensure_active(symbol)

        # Start Telegram bot
        await self._telegram.start()
        await self._telegram.send_message(
            "Gold Scalping Bot started!\n"
            f"Symbol: {', '.join(self._settings.SYMBOLS)}\n"
            f"Strategy: Gold_Scalper\n"
            f"Demo mode: {self._settings.DEMO_MODE}\n"
            f"Risk: {self._settings.RISK_PER_TRADE*100:.0f}% per trade"
        )

        # Schedule tasks
        self._scheduler.add_job(
            self._optimizer.run_optimization,
            CronTrigger(day_of_week="sun", hour=0, minute=0),
            id="weekly_optimization",
        )
        self._scheduler.add_job(
            self._daily_reset,
            CronTrigger(hour=0, minute=5),  # 00:05 UTC daily
            id="daily_reset",
        )
        self._scheduler.add_job(
            self._take_snapshot,
            CronTrigger(minute="*/15"),  # every 15 min
            id="account_snapshot",
        )
        self._scheduler.add_job(
            self._send_daily_report,
            CronTrigger(hour=21, minute=0),  # 21:00 UTC
            id="daily_report",
        )
        self._scheduler.add_job(
            self._daily_backup,
            CronTrigger(hour=23, minute=55),
            id="daily_backup",
        )
        self._scheduler.start()

        self._running = True

        # Run all concurrent tasks
        await asyncio.gather(
            self._market_data.stream_candles(),
            self._process_events(),
            self._trade_manager.start_polling(),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False
        self._trade_manager.stop()
        self._scheduler.shutdown(wait=False)
        await self._telegram.send_message("Gold Scalping Bot shutting down")
        await self._telegram.stop()
        await self._market_data.disconnect()
        logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Main Trading Loop
    # ------------------------------------------------------------------

    async def _process_events(self) -> None:
        """Main event processing loop — consumes NewCandle events."""
        logger.info("Event processor started")
        while self._running:
            try:
                event: NewCandleEvent = await asyncio.wait_for(
                    self._event_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Event queue error: {}", e)
                continue

            try:
                # Update indicators
                feature_vector = await self._feature_engine.process_event(event)

                # Only trade on 1m candles
                if event.timeframe != Timeframe.M1 or feature_vector is None:
                    continue

                symbol = event.symbol
                strategy = self._registry.get_active(symbol)
                if strategy is None:
                    continue

                # Get indicator-enriched DataFrame
                df = self._feature_engine.get_dataframe(symbol, Timeframe.M1)
                if df is None:
                    continue

                # Gather extra data for strategies that need it (IMP etc.)
                import MetaTrader5 as _mt5
                _df_m5 = self._feature_engine.get_dataframe(symbol, Timeframe.M5)
                _sym_info = _mt5.symbol_info(symbol)
                _last_tick = _mt5.symbol_info_tick(symbol)

                # Fetch H4 data for HTF bias filter
                _df_h4 = None
                try:
                    from data.market_data import Timeframe as _TF
                    _df_h4 = self._feature_engine.get_dataframe(symbol, _TF.H4)
                except (AttributeError, KeyError):
                    pass  # H4 not in Timeframe enum yet — strategy handles None gracefully

                # Compute signal
                signal_obj = strategy.compute_signal(
                    df,
                    df_m5=_df_m5,
                    df_h4=_df_h4,
                    symbol_info=_sym_info,
                    last_tick=_last_tick,
                    settings=self._settings,
                )
                signal_obj.symbol = symbol
                signal_obj.strategy_name = strategy.name

                if not signal_obj.is_actionable:
                    continue

                logger.debug("Signal: {} {}", symbol, signal_obj)

                # Run through decision engine
                risk_decision = self._decision_engine.evaluate_signal(signal_obj)
                if risk_decision is None:
                    continue

                # Execute order
                order_result = await self._executor.execute_entry(signal_obj, risk_decision)

                if order_result.success and order_result.trade_record:
                    await self._telegram.send_entry_alert(
                        order_result.trade_record,
                        balance=self._risk_engine._get_balance(),
                        daily_pnl=self._db.get_daily_pnl(),
                        atr_pips=risk_decision.sl_pips / self._settings.ATR_SL_MULTIPLIER,
                    )

                # Run shadow strategies for comparison logging
                await self._run_shadow_strategies(symbol, df)

            except Exception as e:
                logger.error("Event processing error for {} {}: {}", event.symbol, event.timeframe.value, e)

    async def _run_shadow_strategies(self, symbol: str, df) -> None:
        """Run candidate strategies in shadow mode for comparison."""
        shadows = self._registry.get_shadows(symbol)
        for shadow in shadows:
            try:
                signal_obj = shadow.compute_signal(df)
                signal_obj.symbol = symbol
                signal_obj.strategy_name = f"[SHADOW]{shadow.name}"
                if signal_obj.is_actionable:
                    logger.debug("Shadow signal: {} {}", symbol, signal_obj)
            except Exception as e:
                logger.debug("Shadow strategy error: {}", e)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    async def _on_tp_hit(self, trade: Trade, tp_level: str, price: float) -> None:
        await self._telegram.send_tp_hit(trade, tp_level, price)

    async def _on_trade_closed(self, trade: Trade, price: float, pnl: float) -> None:
        await self._telegram.send_trade_closed(trade, price, pnl)

    # ------------------------------------------------------------------
    # Scheduled Tasks
    # ------------------------------------------------------------------

    async def _daily_reset(self) -> None:
        self._risk_engine.reset_daily()
        await self._telegram.send_message("Daily risk reset complete")

    async def _take_snapshot(self) -> None:
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info:
                snapshot = AccountSnapshot(
                    balance=info.balance,
                    equity=info.equity,
                    margin=info.margin,
                    free_margin=info.margin_free,
                    daily_pnl=self._db.get_daily_pnl(),
                    open_trades_count=len(self._db.get_open_trades()),
                )
                if info.balance > 0:
                    snapshot.daily_pnl_pct = snapshot.daily_pnl / info.balance
                self._db.save_snapshot(snapshot)
        except Exception as e:
            logger.error("Snapshot error: {}", e)

    async def _send_daily_report(self) -> None:
        await self._telegram.send_daily_report()

    async def _daily_backup(self) -> None:
        self._db.backup()


def main() -> None:
    bot = ScalpingBot()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(bot.stop()))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        loop.run_until_complete(bot.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
