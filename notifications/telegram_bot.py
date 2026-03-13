"""Async Telegram bot — notifications and command handlers."""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from config.settings import Settings
from notifications.formatters import TradeFormatter
from storage.db import Database
from storage.models import Trade
from strategies.registry import StrategyRegistry


class TelegramNotifier:
    """Async Telegram bot for trade alerts and remote control."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        registry: StrategyRegistry,
        risk_engine=None,
        order_executor=None,
    ):
        self._settings = settings
        self._db = database
        self._registry = registry
        self._risk_engine = risk_engine
        self._executor = order_executor
        self._app: Optional[Application] = None
        self._formatter = TradeFormatter()

    async def start(self) -> None:
        """Initialize and start the Telegram bot."""
        if not self._settings.TELEGRAM_BOT_TOKEN:
            logger.warning("No Telegram bot token — notifications disabled")
            return

        self._app = (
            Application.builder()
            .token(self._settings.TELEGRAM_BOT_TOKEN)
            .build()
        )

        # Register command handlers
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("strategy", self._cmd_strategy))
        self._app.add_handler(CommandHandler("pause", self._cmd_pause))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(CommandHandler("report", self._cmd_report))
        self._app.add_handler(CommandHandler("close_all", self._cmd_close_all))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")

    # ------------------------------------------------------------------
    # Send Methods
    # ------------------------------------------------------------------

    async def send_message(self, text: str) -> None:
        """Send a message to the authorized chat."""
        if not self._app or not self._settings.TELEGRAM_CHAT_ID:
            logger.debug("Telegram not configured, skipping message")
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._settings.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode=None,
            )
        except Exception as e:
            logger.error("Telegram send error: {}", e)

    async def send_entry_alert(self, trade: Trade, balance: float, daily_pnl: float, atr_pips: float = 0.0) -> None:
        msg = self._formatter.entry_alert(trade, balance, daily_pnl, atr_pips)
        await self.send_message(msg)

    async def send_tp_hit(self, trade: Trade, tp_level: str, price: float) -> None:
        msg = self._formatter.tp_hit_alert(trade, tp_level, price)
        await self.send_message(msg)

    async def send_trade_closed(self, trade: Trade, exit_price: float, pnl: float) -> None:
        msg = self._formatter.trade_closed_alert(trade, exit_price, pnl)
        await self.send_message(msg)

    async def send_kill_switch(self, reason: str) -> None:
        msg = self._formatter.kill_switch_alert(reason)
        await self.send_message(msg)

    async def send_daily_report(self) -> None:
        data = self._db.get_daily_report_data()
        balance = self._get_balance()
        msg = self._formatter.daily_report(data, balance)
        await self.send_message(msg)

    # ------------------------------------------------------------------
    # Command Handlers
    # ------------------------------------------------------------------

    def _authorized(self, update: Update) -> bool:
        """Check if message is from authorized chat."""
        return update.effective_chat.id == self._settings.TELEGRAM_CHAT_ID

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        trading_enabled = self._risk_engine.is_trading_enabled if self._risk_engine else False
        open_trades = len(self._db.get_open_trades())
        daily_pnl = self._db.get_daily_pnl()
        balance = self._get_balance()
        active = self._registry.get_all_active()
        msg = self._formatter.status_message(trading_enabled, open_trades, daily_pnl, balance, active)
        await update.message.reply_text(msg)

    async def _cmd_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        report = self._registry.status_report()
        await update.message.reply_text(report)

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        if self._risk_engine:
            self._risk_engine.pause_trading(60)
            await update.message.reply_text("\u23f8 Trading paused for 60 minutes")
        else:
            await update.message.reply_text("Risk engine not available")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        if self._risk_engine:
            self._risk_engine.deactivate_kill_switch()
            await update.message.reply_text("\u25b6 Trading resumed")
        else:
            await update.message.reply_text("Risk engine not available")

    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        data = self._db.get_daily_report_data()
        balance = self._get_balance()
        msg = self._formatter.daily_report(data, balance)
        await update.message.reply_text(msg)

    async def _cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        if self._executor:
            closed = await self._executor.close_all_positions()
            await update.message.reply_text(f"\u2705 Closed {closed} positions")
        else:
            await update.message.reply_text("Order executor not available")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_balance(self) -> float:
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info:
                return float(info.balance)
        except Exception:
            pass
        snap = self._db.get_latest_snapshot()
        return snap.balance if snap else 0.0
