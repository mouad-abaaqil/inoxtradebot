"""Risk engine — position sizing, kill switch, and hard stop rules."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
from loguru import logger

from config.settings import Settings
from data.market_data import MarketDataModule, Timeframe
from storage.db import Database
from strategies.base_strategy import Signal, SignalDirection


@dataclass
class RiskDecision:
    """Result of risk evaluation."""
    approved: bool
    lot_size: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    risk_amount: float = 0.0
    sl_pips: float = 0.0
    rr_ratio: float = 0.0
    rejection_reason: str = ""


class RiskEngine:
    """Position sizing, kill switch, and all hard stop rules."""

    def __init__(
        self,
        settings: Settings,
        market_data: MarketDataModule,
        database: Database,
    ):
        self._settings = settings
        self._market_data = market_data
        self._db = database

        # Thread-safe kill switch
        self._kill_switch = threading.Event()
        self._paused_until: Optional[datetime] = None
        self._trading_enabled = True

        # Daily loss tracking (5% drawdown kill switch)
        self._start_of_day_balance: Optional[float] = None
        self._daily_loss_reached = False

    # ------------------------------------------------------------------
    # Kill Switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self, reason: str) -> None:
        self._kill_switch.set()
        self._trading_enabled = False
        logger.critical("KILL SWITCH ACTIVATED: {}", reason)

    def deactivate_kill_switch(self) -> None:
        self._kill_switch.clear()
        self._trading_enabled = True
        logger.info("Kill switch deactivated")

    def initialize_daily(self) -> None:
        """Initialize daily balance tracking at bot startup."""
        balance = self._get_balance()
        if balance > 0:
            self._start_of_day_balance = balance
            logger.info("Daily balance tracking initialized: ${:.2f}", balance)
        else:
            logger.warning("Cannot initialize daily balance — invalid balance")

    def reset_daily(self) -> None:
        """Reset at midnight: clear daily loss flag and reinitialize balance."""
        self.deactivate_kill_switch()
        self._paused_until = None
        self._daily_loss_reached = False
        # Reinitialize daily balance for new day
        self.initialize_daily()
        logger.info("Daily risk reset complete")

    @property
    def is_trading_enabled(self) -> bool:
        if self._kill_switch.is_set():
            return False
        if self._paused_until and datetime.utcnow() < self._paused_until:
            return False
        return self._trading_enabled

    def pause_trading(self, minutes: int) -> None:
        self._paused_until = datetime.utcnow() + timedelta(minutes=minutes)
        logger.warning("Trading paused for {} minutes", minutes)

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss has exceeded 5% of start-of-day balance.
        Returns True if limit is reached (trading should stop).
        """
        if self._daily_loss_reached:
            return True  # Already hit limit, stay halted

        if self._start_of_day_balance is None:
            return False  # Not initialized yet

        current_balance = self._get_balance()
        if current_balance <= 0:
            return False  # Invalid balance

        daily_loss_pct = (self._start_of_day_balance - current_balance) / self._start_of_day_balance * 100.0

        if daily_loss_pct >= 5.0:
            self._daily_loss_reached = True
            logger.critical(
                "DAILY LOSS LIMIT REACHED: {:.2f}% (${:.2f} loss) — trading halted until tomorrow",
                daily_loss_pct, self._start_of_day_balance - current_balance,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Evaluate Signal
    # ------------------------------------------------------------------

    def evaluate(self, signal: Signal) -> RiskDecision:
        """Full risk evaluation. Returns RiskDecision with lot size and levels."""

        # Kill switch check
        if not self.is_trading_enabled:
            return RiskDecision(approved=False, rejection_reason="Trading disabled (kill switch or pause)")

        symbol = signal.symbol

        # --- Hard stop checks ---

        # Daily 5% loss limit (new kill switch)
        if self.check_daily_loss_limit():
            return RiskDecision(approved=False, rejection_reason="Daily 5% loss limit reached — trading halted")

        balance = self._get_balance()
        if balance <= 0:
            return RiskDecision(approved=False, rejection_reason="Cannot determine account balance")

        # Log LIVE balance for position sizing transparency
        logger.debug(
            "LIVE BALANCE: ${:.2f} | will calculate risk_amount and dynamic lot_size from this",
            balance,
        )

        # Consecutive losses
        consec_losses = self._db.get_consecutive_losses()
        if consec_losses >= self._settings.CONSECUTIVE_LOSS_PAUSE:
            self.pause_trading(self._settings.PAUSE_DURATION_MINUTES)
            return RiskDecision(
                approved=False,
                rejection_reason=f"{consec_losses} consecutive losses — pausing {self._settings.PAUSE_DURATION_MINUTES}min"
            )

        # Max open trades
        open_trades = self._db.get_open_trades()
        if len(open_trades) >= self._settings.MAX_OPEN_TRADES:
            return RiskDecision(approved=False, rejection_reason=f"Max open trades ({self._settings.MAX_OPEN_TRADES}) reached")

        # No duplicate position on same symbol
        for t in open_trades:
            if t.symbol == symbol:
                return RiskDecision(approved=False, rejection_reason=f"Already have position on {symbol}")

        # Signal strength
        if signal.strength < self._settings.MIN_SIGNAL_STRENGTH:
            return RiskDecision(
                approved=False,
                rejection_reason=f"Signal strength {signal.strength:.2f} < {self._settings.MIN_SIGNAL_STRENGTH}"
            )

        # Market hours
        if not self._market_data.is_market_open(symbol):
            return RiskDecision(approved=False, rejection_reason=f"Market closed for {symbol}")

        # Spread check
        spread = self._market_data.get_spread(symbol)
        avg_spread = self._market_data.get_average_spread(symbol)
        if spread is not None and avg_spread > 0:
            if spread > avg_spread * self._settings.MAX_SPREAD_MULTIPLIER:
                return RiskDecision(
                    approved=False,
                    rejection_reason=f"Spread too wide: {spread:.5f} > {avg_spread * self._settings.MAX_SPREAD_MULTIPLIER:.5f}"
                )

        # Volatility assessment — 3-tier system for Gold
        # Normal: ATR <= 2x average
        # High volatility: 2x < ATR <= 3x (reduce size 50%, tighten SL)
        # Extreme: ATR > 3x (reject)
        df = self._market_data.get_dataframe(symbol, Timeframe.M1)
        volatility_mode = "normal"
        volatility_multiplier = 1.0  # position size multiplier
        sl_multiplier_override = None  # if set, override ATR_SL_MULTIPLIER

        if df is not None and "atr_14" in df.columns:
            current_atr = df["atr_14"].iloc[-1]
            atr_sma20 = df["atr_14"].tail(20 * 24 * 60).mean() if len(df) > 100 else df["atr_14"].mean()
            if not np.isnan(current_atr) and not np.isnan(atr_sma20) and atr_sma20 > 0:
                atr_ratio = current_atr / atr_sma20
                if atr_ratio > 3.0:
                    volatility_mode = "extreme"
                    return RiskDecision(
                        approved=False,
                        rejection_reason=f"Extreme volatility: ATR {current_atr:.5f} ({atr_ratio:.2f}x avg {atr_sma20:.5f})"
                    )
                elif atr_ratio > 2.0:
                    volatility_mode = "high_volatility"
                    volatility_multiplier = 0.5  # reduce position size by 50%
                    sl_multiplier_override = 0.7  # tighten SL from 1.0 to 0.7
                    logger.info(
                        "HIGH VOLATILITY MODE {}: ATR {:.5f} ({:.2f}x avg), reducing size 50%, SL 0.7x",
                        symbol, current_atr, atr_ratio,
                    )
                else:
                    logger.debug(
                        "NORMAL VOLATILITY {}: ATR {:.5f} ({:.2f}x avg)",
                        symbol, current_atr, atr_ratio,
                    )

        # Volume check
        if df is not None and "vol_ratio" in df.columns:
            vol_ratio = df["vol_ratio"].iloc[-1]
            if not np.isnan(vol_ratio) and vol_ratio < 0.5:
                return RiskDecision(approved=False, rejection_reason="Dead volume — vol_ratio < 0.5")

        # --- Calculate position sizing ---
        atr = self._get_atr(symbol)
        if atr is None or atr <= 0:
            return RiskDecision(approved=False, rejection_reason="Cannot determine ATR for stop calculation")

        entry_price = self._market_data.get_latest_price(symbol)
        if entry_price is None:
            return RiskDecision(approved=False, rejection_reason="Cannot determine entry price")

        # Apply SL multiplier override if in high volatility mode, else use strategy-specific or default
        # For Gold_Scalper HFT: use 0.8x ATR (ultra-tight stop for maximum scalping control)
        if "Scalper" in signal.strategy_name:
            base_sl_mult = 0.4  # HFT ultra-tight stop
            # En high volatility, on resserre encore le SL du scalper
            atr_sl_mult = base_sl_mult * (sl_multiplier_override / self._settings.ATR_SL_MULTIPLIER) \
                if sl_multiplier_override is not None else base_sl_mult
        elif sl_multiplier_override is not None:
            atr_sl_mult = sl_multiplier_override
        else:
            atr_sl_mult = self._settings.ATR_SL_MULTIPLIER




        sl_distance = atr * atr_sl_mult

        # Use strategy-provided SL for position sizing if available
        if getattr(signal, 'sl', 0) > 0 and entry_price:
            _strategy_sl_dist = abs(entry_price - signal.sl)
            if _strategy_sl_dist > 0:
                sl_distance = _strategy_sl_dist
                logger.info("Using strategy structural SL distance: {:.4f}", sl_distance)
        
        # Garantit que SL >= 2x spread courant (évite erreur 10016)
        sym_info = mt5.symbol_info(symbol)
        if sym_info:
            spread_distance = sym_info.spread * sym_info.point * 2.0
            symbol_key = symbol.upper().replace("/", "")
        
            broker_min = self._settings.MIN_SL_DISTANCE_XAUUSD if "XAU" in symbol_key else 0.0
            atr_min = atr * 0.8  # SL minimum = 0.8x ATR (jamais inférieur au mouvement naturel)
            min_sl_distance = max(spread_distance, broker_min, atr_min)
            logger.info(
                "SL minimum: spread={:.4f} broker={:.4f} atr_min={:.4f} → using {:.4f}",
                spread_distance, broker_min, atr_min, min_sl_distance,
            )

            if sl_distance < min_sl_distance:
                logger.info(
                    "SL distance {:.4f} < min {:.4f} (spread={:.4f} broker_min={:.4f}) — ajusté",
                    sl_distance, min_sl_distance, spread_distance, broker_min,
                )
                sl_distance = min_sl_distance




        pip_size = self._settings.pip_value(symbol)
        sl_pips = sl_distance / pip_size

        # Risk amount
        risk_amount = balance * self._settings.RISK_PER_TRADE

        # Lot size calculation using contract size from MT5
        # Formula: lot_size = risk_amount / (sl_distance_in_price * contract_size)
        # For Gold: contract_size = 100 (oz), so risk_per_lot = $6.50 * 100 = $650
        # For Forex: contract_size = 100,000, sl in price is tiny, math works out
        contract_size = self._get_contract_size(symbol)
        if contract_size <= 0:
            return RiskDecision(approved=False, rejection_reason="Cannot determine contract size")

        raw_lot_size = risk_amount / (sl_distance * contract_size)

        # Apply volatility position size multiplier
        adjusted_lot_size = raw_lot_size * volatility_multiplier

        logger.info(
            "POSITION SIZING {}: risk_amount=${:.2f} / (sl_distance={:.4f} * contract_size={:.0f}) = raw_lots={:.6f} [{} mode, mult={}]",
            symbol, risk_amount, sl_distance, contract_size, raw_lot_size, volatility_mode, volatility_multiplier,
        )
        lot_size = self._normalize_lots(symbol, adjusted_lot_size)
        logger.info("POSITION SIZING {}: after normalize -> lots={} (mode={})", symbol, lot_size, volatility_mode)

        # Log LIVE account balance + risk + lot sizing for transparency
        logger.info(
            "LIVE BALANCE: ${:.2f} | risk_amount=${:.2f} | lot_size={:.2f}",
            balance, risk_amount, lot_size,
        )


        # ↓↓↓ AJOUTE ICI ↓↓↓
        # --- Ajustement automatique au max lots affordable ---
        account = mt5.account_info()
        if account and account.margin_free > 0:
            margin_per_lot = mt5.order_calc_margin(
                mt5.ORDER_TYPE_BUY if signal.direction == SignalDirection.LONG else mt5.ORDER_TYPE_SELL,
                symbol, 1.0, entry_price
            )
            if margin_per_lot and margin_per_lot > 0:
                max_affordable_lots = (account.margin_free * 0.8) / margin_per_lot
                if lot_size > max_affordable_lots:
                    adjusted = self._normalize_lots(symbol, max_affordable_lots)
                    logger.warning(
                        "MARGIN LIMIT {}: lots={} -> {} (free=${:.2f} | margin/lot=${:.2f} | leverage 1:{})",
                        symbol, lot_size, adjusted,
                        account.margin_free, margin_per_lot, account.leverage,
                    )
                    lot_size = adjusted
                if lot_size <= 0:
                    return RiskDecision(
                        approved=False,
                        rejection_reason=f"Insufficient margin: need ${margin_per_lot:.2f}/lot | free ${account.margin_free:.2f} | leverage 1:{account.leverage}"
                    )
                logger.info(
                    "MARGIN OK {}: lots={} | need=${:.2f} | free=${:.2f}",
                    symbol, lot_size, margin_per_lot * lot_size, account.margin_free,
                )
        # ↑↑↑ FIN DU BLOC ↑↑↑


        # Check single trade risk doesn't exceed max
        actual_risk = lot_size * sl_distance * contract_size
        if actual_risk / balance > self._settings.MAX_SINGLE_TRADE_RISK:
            return RiskDecision(
                approved=False,
                rejection_reason=f"Single trade risk {actual_risk / balance:.2%} > {self._settings.MAX_SINGLE_TRADE_RISK:.0%}"
            )

        # --- Commission & Breakeven Analysis (IC Markets) ---
        # Round-trip commission: $7/lot * 2 = $14 per lot
        commission_total = self._settings.COMMISSION_PER_LOT * lot_size * 2
        pip_size = self._settings.pip_value(symbol)

        # Breakeven in pips: (commission_total_USD) / (lot_size * contract_size) / pip_size_in_usd
        # For Gold: $14 / (0.01 * 100) = $14 / $1 = 14 pips minimum to break even
        breakeven_pips = (commission_total / (lot_size * contract_size)) / pip_size
        min_tp1_pips = breakeven_pips * 2.0  # TP1 must be at least 2x breakeven

        logger.info(
            "COMMISSION {}: lot_size={} | commission_total=${:.2f} | breakeven={:.1f} pips | min_TP1={:.1f} pips",
            symbol, lot_size, commission_total, breakeven_pips, min_tp1_pips,
        )

        # Calculate TP levels (2-tier: TP1 and TP2)
        # For HFT on Gold_Scalper: use tight ATR-based multipliers
        # TP1 = 0.6x ATR (close 60% quickly), TP2 = 1.2x ATR (close remaining 40%)
        
        
        # TP basé sur sl_distance réel (déjà ajusté au spread minimum)
        # R:R garanti = 1.5:1 minimum, TP2 = 2x TP1
        tp1_distance = sl_distance * 1.5   # R:R exact 1.5:1
        tp2_distance = sl_distance * 3.0   # R:R 3:1

        if signal.direction == SignalDirection.LONG:
            sl  = entry_price - sl_distance
            tp1 = entry_price + tp1_distance
            tp2 = entry_price + tp2_distance
        else:
            sl  = entry_price + sl_distance
            tp1 = entry_price - tp1_distance
            tp2 = entry_price - tp2_distance



        # Override SL/TP with strategy-provided structural levels
        if getattr(signal, 'sl', 0) > 0:
            sl = signal.sl
        if getattr(signal, 'tp1', 0) > 0:
            tp1 = signal.tp1
        if getattr(signal, 'tp2', 0) > 0:
            tp2 = signal.tp2

        # Garantit que TP1/TP2 >= 2x spread courant (évite erreur 10016)
        if sym_info:
            min_tp_distance = sym_info.spread * sym_info.point * 2.0
            if signal.direction == SignalDirection.LONG:
                tp1 = max(tp1, entry_price + min_tp_distance)
                tp2 = max(tp2, entry_price + min_tp_distance * 2)
            else:
                tp1 = min(tp1, entry_price - min_tp_distance)
                tp2 = min(tp2, entry_price - min_tp_distance * 2)

        # Check TP1 covers commission breakeven (must be 2x breakeven_pips away from entry)
        tp1_distance_pips = abs(tp1 - entry_price) / pip_size




        if tp1_distance_pips < min_tp1_pips:
            return RiskDecision(
                approved=False,
                rejection_reason=f"TP1 ({tp1_distance_pips:.1f} pips) < 2x breakeven ({min_tp1_pips:.1f} pips) | commission too high"
            )

        # --- Check Minimum R:R after commission ---
        # R:R = TP1_distance / SL_distance
        sl_distance_pips = abs(sl - entry_price) / pip_size
        rr = tp1_distance_pips / sl_distance_pips if sl_distance_pips > 0 else 0

        # Strategy-provided levels are pre-validated; use 1.0:1 floor.
        # ATR-computed levels use stricter 1.5:1.
        has_strategy_levels = getattr(signal, 'sl', 0) > 0
        min_rr = 1.0 if has_strategy_levels else 1.5

        logger.info(
            "R:R ANALYSIS {}: SL={:.1f} pips | TP1={:.1f} pips | R:R={:.2f}:1 (min required={:.2f}:1{})",
            symbol, sl_distance_pips, tp1_distance_pips, rr, min_rr,
            " [strategy SL/TP]" if has_strategy_levels else "",
        )

        if rr < min_rr - 1e-9:
            return RiskDecision(
                approved=False,
                rejection_reason=f"R:R {rr:.2f}:1 < minimum {min_rr:.1f}:1 | SL too wide or TP1 too close"
            )
        digits = self._settings.pip_digits(symbol)

        return RiskDecision(
            approved=True,
            lot_size=lot_size,
            stop_loss=round(sl, digits),
            tp1=round(tp1, digits),
            tp2=round(tp2, digits),
            tp3=0.0,  # unused in 2-tier system
            risk_amount=round(risk_amount, 2),
            sl_pips=round(sl_pips, 1),
            rr_ratio=round(rr, 2),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_balance(self) -> float:
        try:
            info = mt5.account_info()
            if info:
                return float(info.balance)
        except Exception:
            pass
        # Fallback to last snapshot
        snap = self._db.get_latest_snapshot()
        return snap.balance if snap else 0.0

    def _get_atr(self, symbol: str) -> Optional[float]:
        df = self._market_data.get_dataframe(symbol, Timeframe.M1)
        if df is not None and "atr_14" in df.columns:
            val = df["atr_14"].iloc[-1]
            return float(val) if not np.isnan(val) else None
        return None

    def _get_contract_size(self, symbol: str) -> float:
        """Get the contract size from MT5 symbol info.

        For Gold (XAUUSD): 100 (ounces per lot).
        For Forex (EURUSD): 100,000 (units per lot).
        Falls back to hardcoded values if MT5 info unavailable.
        """
        info = mt5.symbol_info(symbol)
        if info is not None and info.trade_contract_size > 0:
            return float(info.trade_contract_size)
        # Fallback
        sym = symbol.upper().replace("/", "")
        if "XAU" in sym:
            return 100.0
        return 100_000.0

    def _normalize_lots(self, symbol: str, lots: float) -> float:
        """Round lot size to broker's allowed step."""
        info = mt5.symbol_info(symbol)
        if info:
            min_lot = info.volume_min
            max_lot = info.volume_max
            step = info.volume_step
            lots = max(min_lot, min(max_lot, lots))
            lots = round(lots / step) * step
            return round(lots, 2)
        # Fallback
        return round(max(0.01, lots), 2)
