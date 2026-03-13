"""MT5 order placement with retry, slippage tracking, and demo mode."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
from loguru import logger

from config.settings import Settings
from execution.risk_engine import RiskDecision
from storage.db import Database
from storage.models import Trade, TradeDirection, TradeState
from strategies.base_strategy import Signal, SignalDirection


@dataclass
class OrderResult:
    success: bool
    ticket: int = 0
    fill_price: float = 0.0
    slippage: float = 0.0
    error: str = ""
    trade_record: Optional[Trade] = None


class OrderExecutor:
    """Place, modify, and close orders via MT5 with retry logic."""

    def __init__(self, settings: Settings, database: Database):
        self._settings = settings
        self._db = database

    async def execute_entry(
        self, signal: Signal, risk: RiskDecision
    ) -> OrderResult:
        """Place a market entry order with SL/TP."""

        # Demo mode: log but don't place real orders
        if self._settings.DEMO_MODE:
            return await self._demo_entry(signal, risk)

        order_type = mt5.ORDER_TYPE_BUY if signal.direction == SignalDirection.LONG else mt5.ORDER_TYPE_SELL

        # Get current price
        tick = mt5.symbol_info_tick(signal.symbol)
        if tick is None:
            return OrderResult(success=False, error="Cannot get tick data")

        price = tick.ask if signal.direction == SignalDirection.LONG else tick.bid

        # ↓↓↓ AJOUTE ICI ↓↓↓
        MAX_SLIPPAGE_PIPS = 20
        pip_size = self._settings.pip_value(signal.symbol)
        if signal.direction == SignalDirection.LONG:
            signal_price = risk.stop_loss + risk.sl_pips * pip_size  # SL sous l'entrée
        else:
            signal_price = risk.stop_loss - risk.sl_pips * pip_size  # SL au-dessus de l'entrée
        current_slippage = abs(price - signal_price) / pip_size
        if current_slippage > MAX_SLIPPAGE_PIPS:
            logger.warning(
                "SLIPPAGE TOO HIGH: {:.1f} pips > max {} — order aborted",
                current_slippage, MAX_SLIPPAGE_PIPS,
            )
            return OrderResult(success=False, error=f"Slippage {current_slippage:.1f} pips > max {MAX_SLIPPAGE_PIPS}")
        # ↑↑↑ FIN ↑↑↑


        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": risk.lot_size,
            "type": order_type,
            "price": price,
            "sl": risk.stop_loss,
            "tp": risk.tp2,  # MT5 native TP set to TP2 (we manage TP1 ourselves)
            "deviation": 10,  # max slippage in points
            "magic": 123456,
            "comment": f"ScalpBot|{signal.strategy_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Retry with exponential backoff
        for attempt in range(3):
            try:
                result = await asyncio.to_thread(mt5.order_send, request)

                if result is None:
                    error = mt5.last_error()
                    logger.warning("Order send returned None, attempt {}/3: {}", attempt + 1, error)
                    await asyncio.sleep(2 ** attempt)
                    continue

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    fill_price = result.price
                    slippage = abs(fill_price - price) / self._settings.pip_value(signal.symbol)

                    trade = self._create_trade_record(signal, risk, result.order, fill_price, price, slippage)
                    self._db.save_trade(trade)

                    logger.info(
                        "ORDER FILLED: {} {} {} lots @ {} (slip: {:.1f} pips)",
                        signal.symbol, signal.direction.value, risk.lot_size,
                        fill_price, slippage,
                    )

                    return OrderResult(
                        success=True,
                        ticket=result.order,
                        fill_price=fill_price,
                        slippage=slippage,
                        trade_record=trade,
                    )

                logger.warning(
                    "Order rejected ({}): {} — attempt {}/3",
                    result.retcode, result.comment, attempt + 1,
                )
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.error("Order execution error: {} — attempt {}/3", e, attempt + 1)
                await asyncio.sleep(2 ** attempt)

        return OrderResult(success=False, error="Failed after 3 attempts")

    async def close_position(self, symbol: str, ticket: int, volume: float, direction: SignalDirection) -> OrderResult:
        """Close a position (full or partial)."""
        if self._settings.DEMO_MODE:
            logger.info("[DEMO] Close {} {} lots ticket={}", symbol, volume, ticket)
            return OrderResult(success=True, ticket=ticket)

        close_type = mt5.ORDER_TYPE_SELL if direction == SignalDirection.LONG else mt5.ORDER_TYPE_BUY

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, error="Cannot get tick for close")

        price = tick.bid if direction == SignalDirection.LONG else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "ScalpBot|Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        for attempt in range(3):
            try:
                result = await asyncio.to_thread(mt5.order_send, request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info("Position closed: {} {} lots @ {}", symbol, volume, result.price)
                    return OrderResult(success=True, ticket=ticket, fill_price=result.price)
                logger.warning("Close order failed: {}", result.comment if result else "None")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error("Close error: {}", e)
                await asyncio.sleep(2 ** attempt)

        return OrderResult(success=False, error="Close failed after 3 attempts")

    async def modify_sl(self, symbol: str, ticket: int, new_sl: float, tp: float) -> bool:
        """Modify the stop loss of an open position."""
        if self._settings.DEMO_MODE:
            logger.info("[DEMO] Modify SL: {} ticket={} new_sl={}", symbol, ticket, new_sl)
            return True

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": tp,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        try:
            result = await asyncio.to_thread(mt5.order_send, request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("SL modified: {} ticket={} new_sl={}", symbol, ticket, new_sl)
                return True
            logger.warning("SL modify failed: {}", result.comment if result else "None")
        except Exception as e:
            logger.error("SL modify error: {}", e)
        return False

    async def close_all_positions(self) -> int:
        """Emergency close all open positions."""
        if self._settings.DEMO_MODE:
            logger.info("[DEMO] Close all positions")
            return 0

        positions = mt5.positions_get()
        if not positions:
            return 0

        closed = 0
        for pos in positions:
            if pos.magic == 123456:
                direction = SignalDirection.LONG if pos.type == mt5.ORDER_TYPE_BUY else SignalDirection.SHORT
                result = await self.close_position(pos.symbol, pos.ticket, pos.volume, direction)
                if result.success:
                    closed += 1
        return closed

    # ------------------------------------------------------------------
    # Demo Mode
    # ------------------------------------------------------------------

    async def _demo_entry(self, signal: Signal, risk: RiskDecision) -> OrderResult:
        """Simulate order in demo mode."""
        price = self._get_current_price(signal.symbol, signal.direction)
        if price is None:
            return OrderResult(success=False, error="[DEMO] Cannot get price")

        trade = self._create_trade_record(signal, risk, ticket=0, fill_price=price, requested_price=price, slippage=0)
        self._db.save_trade(trade)

        logger.info(
            "[DEMO] ORDER: {} {} {} lots @ {} SL={} TP1={} TP2={}",
            signal.symbol, signal.direction.value, risk.lot_size,
            price, risk.stop_loss, risk.tp1, risk.tp2,
        )
        return OrderResult(success=True, ticket=0, fill_price=price, trade_record=trade)

    def _get_current_price(self, symbol: str, direction: SignalDirection) -> Optional[float]:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return tick.ask if direction == SignalDirection.LONG else tick.bid
        return None

    def _create_trade_record(
        self,
        signal: Signal,
        risk: RiskDecision,
        ticket: int,
        fill_price: float,
        requested_price: float,
        slippage: float,
    ) -> Trade:
        return Trade(
            ticket=ticket,
            symbol=signal.symbol,
            direction=TradeDirection(signal.direction.value),
            state=TradeState.OPEN,
            strategy_name=signal.strategy_name,
            signal_strength=signal.strength,
            signal_confidence=signal.confidence,
            entry_reason=signal.entry_reason,
            entry_price=fill_price,
            stop_loss=risk.stop_loss,
            tp1_price=risk.tp1,
            tp2_price=risk.tp2,
            tp3_price=risk.tp3,
            lot_size=risk.lot_size,
            risk_amount=risk.risk_amount,
            sl_pips=risk.sl_pips,
            requested_price=requested_price,
            fill_price=fill_price,
            slippage_pips=slippage,
        )
