"""Telegram message formatters for trade alerts and reports."""

from __future__ import annotations

from typing import Optional

from storage.models import Trade, TradeState


class TradeFormatter:
    """Formats trade data into Telegram-ready messages."""

    @staticmethod
    def entry_alert(
        trade: Trade,
        balance: float,
        daily_pnl: float,
        atr_pips: float = 0.0,
    ) -> str:
        emoji = "\U0001f7e2" if trade.direction.value == "LONG" else "\U0001f534"
        return (
            f"{emoji} TRADE OPENED  {trade.symbol}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Direction: {trade.direction.value}\n"
            f"Strategy:  {trade.strategy_name}\n"
            f"Reason:    {trade.entry_reason or 'N/A'}\n"
            f"Signal:    str={trade.signal_strength:.2f} conf={trade.signal_confidence:.2f}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Entry:     {trade.entry_price}\n"
            f"Stop Loss: {trade.stop_loss}  ({trade.sl_pips:.1f} pips)\n"
            f"TP1:       {trade.tp1_price}  (1R | closes 50%)\n"
            f"TP2:       {trade.tp2_price}  (2R | closes 50%)\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Lot Size:  {trade.lot_size}\n"
            f"Risk:      ${trade.risk_amount:.2f} ({trade.risk_amount / balance * 100:.1f}% of balance)\n"
            f"ATR:       {atr_pips:.1f} pips\n"
            f"Balance:   ${balance:.2f}\n"
            f"Daily P&L: {daily_pnl:+.2f}"
        )

    @staticmethod
    def tp_hit_alert(trade: Trade, tp_level: str, price: float) -> str:
        return (
            f"\U0001f3af {tp_level} HIT  {trade.symbol}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Strategy: {trade.strategy_name}\n"
            f"Entry: {trade.entry_price} \u2192 Current: {price}\n"
            f"SL moved to: {trade.stop_loss}"
        )

    @staticmethod
    def trade_closed_alert(trade: Trade, exit_price: float, pnl: float) -> str:
        emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
        state_text = "TP2" if trade.state == TradeState.CLOSED else "SL"
        return (
            f"{emoji} TRADE CLOSED  {trade.symbol}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Strategy: {trade.strategy_name}\n"
            f"Exit: {state_text} @ {exit_price}\n"
            f"Entry: {trade.entry_price} \u2192 Exit: {exit_price}\n"
            f"P&L: ${pnl:+.2f} ({trade.pnl_pips or 0:+.1f} pips)"
        )

    @staticmethod
    def kill_switch_alert(reason: str) -> str:
        return (
            f"\U0001f6a8 KILL SWITCH ACTIVATED\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Reason: {reason}\n"
            f"All trading halted until manual reset or next market open."
        )

    @staticmethod
    def daily_report(data: dict, balance: float) -> str:
        return (
            f"\U0001f4c8 DAILY REPORT\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Trades: {data['total_trades']} ({data['closed_trades']} closed)\n"
            f"Wins: {data['wins']} | Losses: {data['losses']}\n"
            f"Win Rate: {data['win_rate']:.1%}\n"
            f"P&L: ${data['total_pnl']:+.2f}\n"
            f"Best: ${data['best_trade']:+.2f}\n"
            f"Worst: ${data['worst_trade']:+.2f}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Balance: ${balance:.2f}"
        )

    @staticmethod
    def status_message(
        trading_enabled: bool,
        open_trades: int,
        daily_pnl: float,
        balance: float,
        active_strategies: dict,
    ) -> str:
        status = "\U00002705 RUNNING" if trading_enabled else "\U000026d4 PAUSED"
        strats = "\n".join(
            f"  {sym}: {s.name}" for sym, s in active_strategies.items()
        ) or "  None"
        return (
            f"\U0001f916 BOT STATUS: {status}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"Open Trades: {open_trades}\n"
            f"Daily P&L: ${daily_pnl:+.2f}\n"
            f"Balance: ${balance:.2f}\n"
            f"Strategies:\n{strats}"
        )
