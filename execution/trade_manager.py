"""Trade manager — TP ladder state machine, trailing stop, and position monitoring."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable, Coroutine, List, Optional

import MetaTrader5 as mt5
import numpy as np
from loguru import logger

from config.settings import Settings
from data.market_data import MarketDataModule, Timeframe
from execution.order_executor import OrderExecutor
from storage.db import Database
from storage.models import Trade, TradeDirection, TradeState


class TradeManager:
    """Monitors open trades, manages TP ladder, trailing stop, and state transitions.

    2-tier TP system:
      OPEN -> TP1_HIT (close 50%, move SL to breakeven) -> CLOSED at TP2 or SL
    """

    def __init__(
        self,
        settings: Settings,
        market_data: MarketDataModule,
        order_executor: OrderExecutor,
        database: Database,
        on_tp_hit: Optional[Callable] = None,  # callback for notifications
        on_trade_closed: Optional[Callable] = None,
        risk_engine=None,  # for daily loss tracking
    ):
        self._settings = settings
        self._market_data = market_data
        self._executor = order_executor
        self._db = database
        self._on_tp_hit = on_tp_hit
        self._on_trade_closed = on_trade_closed
        self._risk_engine = risk_engine
        self._running = False

    async def start_polling(self) -> None:
        """Poll open trades every TRADE_POLL_INTERVAL seconds."""
        self._running = True
        logger.info("Trade manager polling started ({}s interval)", self._settings.TRADE_POLL_INTERVAL)

        while self._running:
            try:
                await self._check_all_trades()
            except Exception as e:
                logger.error("Trade manager error: {}", e)
            await asyncio.sleep(self._settings.TRADE_POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    async def _check_all_trades(self) -> None:
        """Check all open/partially-closed trades."""
        open_trades = self._db.get_open_trades()
        for trade in open_trades:
            await self._check_trade(trade)

    async def _check_trade(self, trade: Trade) -> None:
        """Evaluate a single trade against current price."""
        current_price = self._get_current_price(trade.symbol, trade.direction)
        if current_price is None:
            return

        is_long = trade.direction == TradeDirection.LONG

        # Check SL hit
        if self._is_sl_hit(trade, current_price, is_long):
            await self._handle_sl_hit(trade, current_price)
            return

        # Check max duration (HFT: 15 minutes for Gold_Scalper)
        if await self._check_max_duration(trade, current_price):
            return

        # State machine transitions
        if trade.state == TradeState.OPEN:
            if self._is_tp1_hit(trade, current_price, is_long):
                await self._handle_tp1(trade, current_price)

        elif trade.state == TradeState.TP1_HIT:
            if self._is_tp2_hit(trade, current_price, is_long):
                await self._handle_tp2_close(trade, current_price)
            else:
                # Trailing stop logic after TP1
                await self._update_trailing_stop(trade, current_price, is_long)

    # ------------------------------------------------------------------
    # Price checks
    # ------------------------------------------------------------------

    def _is_sl_hit(self, trade: Trade, price: float, is_long: bool) -> bool:
        if is_long:
            return price <= trade.stop_loss
        return price >= trade.stop_loss

    async def _check_max_duration(self, trade: Trade, current_price: float) -> bool:
        """Force close trade if open for > 10 minutes (HFT max duration)."""
        if trade.opened_at is None:
            return False

        duration_minutes = (datetime.utcnow() - trade.opened_at).total_seconds() / 60.0

        if duration_minutes > 20:
            logger.warning(
                "MAX DURATION HIT {}: {} mins | force closing at market {}",
                trade.symbol, duration_minutes, current_price,
            )
            pnl = self._calculate_pnl(trade, current_price)
            pnl_pips = self._calculate_pnl_pips(trade, current_price)

            self._db.update_trade(
                trade.id,
                state=TradeState.CLOSED,
                exit_price=current_price,
                pnl=pnl,
                pnl_pips=pnl_pips,
                closed_at=datetime.utcnow(),
            )
            logger.info(
                "FORCED CLOSE (max duration): {} P&L=${:.2f} ({:.1f} pips)",
                trade.symbol, pnl, pnl_pips,
            )

            if self._on_trade_closed:
                await _safe_callback(self._on_trade_closed, trade, current_price, pnl)

            # Check daily loss limit after close
            await self._check_daily_loss_after_close()

            return True

        return False

    def _is_tp1_hit(self, trade: Trade, price: float, is_long: bool) -> bool:
        if trade.tp1_price is None:
            return False
        if is_long:
            return price >= trade.tp1_price
        return price <= trade.tp1_price

    def _is_tp2_hit(self, trade: Trade, price: float, is_long: bool) -> bool:
        if trade.tp2_price is None:
            return False
        if is_long:
            return price >= trade.tp2_price
        return price <= trade.tp2_price

    # ------------------------------------------------------------------
    # TP Handlers
    # ------------------------------------------------------------------

    async def _handle_tp1(self, trade: Trade, price: float) -> None:
        """TP1 hit: close 50%, move SL to breakeven."""
        close_volume = round(trade.lot_size * self._settings.TP1_CLOSE_PCT, 2)
        close_volume = max(0.01, close_volume)

        direction = SignalDirection_from_trade(trade.direction)

        if not self._position_exists_in_mt5(trade.ticket):
            logger.info("TP1: position already closed by MT5 natively: {} ticket={}", trade.symbol, trade.ticket)
            # Met à jour la DB pour éviter la boucle infinie
            pip = self._settings.pip_value(trade.symbol)
            is_long = trade.direction == TradeDirection.LONG
            be_sl = trade.entry_price + pip if is_long else trade.entry_price - pip
            self._db.update_trade(
                trade.id,
                state=TradeState.TP1_HIT,
                stop_loss=be_sl,
                tp1_hit_at=datetime.utcnow(),
            )
            return


        result = await self._executor.close_position(
            trade.symbol, trade.ticket, close_volume, direction
        )

        if result.success:
            # Move SL to breakeven (entry price + 1 pip buffer)
            pip = self._settings.pip_value(trade.symbol)
            is_long = trade.direction == TradeDirection.LONG
            be_sl = trade.entry_price + pip if is_long else trade.entry_price - pip

            await self._executor.modify_sl(
                trade.symbol, trade.ticket, be_sl, trade.tp2_price or 0
            )

            self._db.update_trade(
                trade.id,
                state=TradeState.TP1_HIT,
                stop_loss=be_sl,
                tp1_hit_at=datetime.utcnow(),
            )
            logger.info("TP1 HIT: {} closed {:.2f} lots, SL moved to BE {}", trade.symbol, close_volume, be_sl)

            if self._on_tp_hit:
                await _safe_callback(self._on_tp_hit, trade, "TP1", price)

    async def _handle_tp2_close(self, trade: Trade, price: float) -> None:
        """TP2 hit: close remaining position."""
        remaining = round(trade.lot_size * self._settings.TP2_CLOSE_PCT, 2)
        remaining = max(0.01, remaining)

        if self._position_exists_in_mt5(trade.ticket):
            direction = SignalDirection_from_trade(trade.direction)
            await self._executor.close_position(
                trade.symbol, trade.ticket, remaining, direction
            )
        else:
            logger.info("TP2 already closed by MT5 natively: {} ticket={}", trade.symbol, trade.ticket)

        pnl = self._calculate_pnl(trade, price)


        pnl_pips = self._calculate_pnl_pips(trade, price)

        self._db.update_trade(
            trade.id,
            state=TradeState.CLOSED,
            exit_price=price,
            pnl=pnl,
            pnl_pips=pnl_pips,
            closed_at=datetime.utcnow(),
        )
        logger.info("TP2 HIT / CLOSED: {} P&L=${:.2f} ({:.1f} pips)", trade.symbol, pnl, pnl_pips)

        if self._on_tp_hit:
            await _safe_callback(self._on_tp_hit, trade, "TP2", price)
        if self._on_trade_closed:
            await _safe_callback(self._on_trade_closed, trade, price, pnl)

        # Check daily loss limit after close
        await self._check_daily_loss_after_close()

    async def _handle_sl_hit(self, trade: Trade, price: float) -> None:
        """SL hit: full close."""
        pnl = self._calculate_pnl(trade, price)
        pnl_pips = self._calculate_pnl_pips(trade, price)

        self._db.update_trade(
            trade.id,
            state=TradeState.SL_HIT,
            exit_price=price,
            pnl=pnl,
            pnl_pips=pnl_pips,
            closed_at=datetime.utcnow(),
        )
        logger.warning("SL HIT: {} P&L=${:.2f} ({:.1f} pips)", trade.symbol, pnl, pnl_pips)

        if self._on_trade_closed:
            await _safe_callback(self._on_trade_closed, trade, price, pnl)

        # Check daily loss limit after close
        await self._check_daily_loss_after_close()

    # ------------------------------------------------------------------
    # Trailing Stop
    # ------------------------------------------------------------------

    async def _update_trailing_stop(self, trade: Trade, price: float, is_long: bool) -> None:
        """After TP1, trail the stop at 1x ATR behind price."""
        atr = self._get_atr(trade.symbol)
        if atr is None or atr <= 0:
            return

        if is_long:
            digits = self._settings.pip_digits(trade.symbol)
            new_sl = round(price - atr, digits)
            if new_sl > trade.stop_loss:
                await self._executor.modify_sl(
                    trade.symbol, trade.ticket, new_sl, trade.tp2_price or 0
                )
                self._db.update_trade(trade.id, stop_loss=new_sl)
                logger.debug("Trailing SL updated: {} -> {}", trade.symbol, new_sl)
        else:
            digits = self._settings.pip_digits(trade.symbol)
            new_sl = round(price + atr, digits)
            if new_sl < trade.stop_loss:
                await self._executor.modify_sl(
                    trade.symbol, trade.ticket, new_sl, trade.tp2_price or 0
                )
                self._db.update_trade(trade.id, stop_loss=new_sl)
                logger.debug("Trailing SL updated: {} -> {}", trade.symbol, new_sl)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _check_daily_loss_after_close(self) -> None:
        """After each trade close, check if daily 5% loss limit has been reached."""
        if self._risk_engine is None:
            return

        if self._risk_engine.check_daily_loss_limit():
            logger.critical("DAILY LOSS LIMIT TRIGGERED — all new entries blocked for rest of day")
            # Send Telegram alert (if available)
            try:
                # Import here to avoid circular imports
                from notifications.telegram_bot import TelegramNotifier
                # Try to find a telegram instance — this is a workaround
                # A better approach would be to pass telegram_notifier to TradeManager
                logger.info("⚠️ Daily 5% loss limit reached. No more trades today.")
            except Exception as e:
                logger.error("Cannot send Telegram alert: {}", e)


    def _position_exists_in_mt5(self, ticket: int) -> bool:
        """Vérifie que la position est toujours ouverte dans MT5."""
        positions = mt5.positions_get(ticket=ticket)
        return positions is not None and len(positions) > 0

    def _get_current_price(self, symbol: str, direction: TradeDirection) -> Optional[float]:
        
        
        """Get current bid/ask depending on trade direction."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            # Fallback to last close
            return self._market_data.get_latest_price(symbol)
        # For longs, exit at bid; for shorts, exit at ask
        return tick.bid if direction == TradeDirection.LONG else tick.ask

    def _get_atr(self, symbol: str) -> Optional[float]:
        df = self._market_data.get_dataframe(symbol, Timeframe.M1)
        if df is not None and "atr_14" in df.columns:
            val = df["atr_14"].iloc[-1]
            return float(val) if not np.isnan(val) else None
        return None

    def _get_contract_size(self, symbol: str) -> float:
        """Get contract size from MT5, with fallback."""
        info = mt5.symbol_info(symbol)
        if info is not None and info.trade_contract_size > 0:
            return float(info.trade_contract_size)
        sym = symbol.upper().replace("/", "")
        if "XAU" in sym:
            return 100.0
        return 100_000.0

    def _calculate_pnl(self, trade: Trade, exit_price: float) -> float:
        """PnL = price_diff * contract_size * lot_size."""
        contract_size = self._get_contract_size(trade.symbol)
        if trade.direction == TradeDirection.LONG:
            price_diff = exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - exit_price
        return price_diff * contract_size * trade.lot_size

    def _calculate_pnl_pips(self, trade: Trade, exit_price: float) -> float:
        pip_size = self._settings.pip_value(trade.symbol)
        if trade.direction == TradeDirection.LONG:
            return (exit_price - trade.entry_price) / pip_size
        return (trade.entry_price - exit_price) / pip_size


def SignalDirection_from_trade(direction: TradeDirection):
    """Convert TradeDirection to SignalDirection for order executor."""
    from strategies.base_strategy import SignalDirection
    return SignalDirection.LONG if direction == TradeDirection.LONG else SignalDirection.SHORT


async def _safe_callback(callback, *args):
    """Safely invoke an async or sync callback."""
    try:
        result = callback(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        logger.error("Callback error: {}", e)
