#!/usr/bin/env python
"""INOXTRADE real-time terminal dashboard using Rich.

Standalone launch:
    python ui/dashboard.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Force UTF-8 on Windows so Rich box-drawing characters render correctly
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

REFRESH_INTERVAL = 5  # seconds

# ---------------------------------------------------------------------------
# ASCII Header
# ---------------------------------------------------------------------------
HEADER_ART = r"""
 ___ _   _  _____  _______ ____      _    ____  _____
|_ _| \ | |/ _ \ \/ /_   _|  _ \    / \  |  _ \| ____|
 | ||  \| | | | \  /  | | | |_) |  / _ \ | | | |  _|
 | || |\  | |_| /  \  | | |  _ <  / ___ \| |_| | |___
|___|_| \_|\___/_/\_\ |_| |_| \_\/_/   \_\____/|_____|
"""


# ---------------------------------------------------------------------------
# Data fetchers (each catches its own errors)
# ---------------------------------------------------------------------------
def _get_mt5_account() -> Optional[Dict[str, Any]]:
    """Fetch MT5 account info. Returns None if MT5 unavailable."""
    try:
        import MetaTrader5 as mt5

        if not mt5.terminal_info():
            from config.settings import get_settings

            settings = get_settings()
            kwargs = {}
            if settings.MT5_PATH:
                kwargs["path"] = settings.MT5_PATH
            if not mt5.initialize(**kwargs):
                return None
            mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER,
            )

        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
        }
    except Exception:
        return None


def _get_mt5_tick() -> Optional[float]:
    """Fetch live XAUUSD bid price."""
    try:
        import MetaTrader5 as mt5

        tick = mt5.symbol_info_tick("XAUUSD")
        if tick is None:
            return None
        return tick.bid
    except Exception:
        return None


def _get_db_data() -> Dict[str, Any]:
    """Fetch today's trades and daily report from SQLite."""
    try:
        from storage import get_database

        db = get_database()
        trades = db.get_today_trades()
        report = db.get_daily_report_data()
        return {"trades": trades, "report": report}
    except Exception:
        return {"trades": [], "report": None}


def _get_session(hour: int) -> str:
    """Determine active session from UTC hour."""
    if 7 <= hour < 12:
        return "London"
    elif 13 <= hour < 18:
        return "New York"
    return "Hors session"


def _get_daily_bias() -> str:
    """Try to compute daily bias via the strategy helper."""
    try:
        import MetaTrader5 as mt5
        import talib
        import numpy as np

        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_D1, 0, 60)
        if rates is None or len(rates) < 55:
            return "N/A"
        import pandas as pd

        df = pd.DataFrame(rates)
        close = df["close"].values.astype(float)
        ema20 = talib.EMA(close, timeperiod=20)
        ema50 = talib.EMA(close, timeperiod=50)
        c = close[-1]
        e20 = ema20[-1]
        e50 = ema50[-1]
        if np.isnan(e20) or np.isnan(e50):
            return "N/A"
        if c > e20 and e20 > e50:
            return "BULLISH"
        elif c < e20 and e20 < e50:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "N/A"


def _get_m1_atr() -> Optional[float]:
    """Fetch current M1 ATR(14)."""
    try:
        import MetaTrader5 as mt5
        import talib

        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 30)
        if rates is None or len(rates) < 20:
            return None
        import pandas as pd

        df = pd.DataFrame(rates)
        atr = talib.ATR(
            df["high"].values.astype(float),
            df["low"].values.astype(float),
            df["close"].values.astype(float),
            timeperiod=14,
        )
        val = float(atr[-1])
        import numpy as np

        return None if np.isnan(val) else val
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------
def _build_header() -> Panel:
    now_utc = datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")
    content = Text.from_ansi(f"[bold cyan]{HEADER_ART}[/]")
    content = Text()
    content.append(HEADER_ART.strip(), style="bold cyan")
    content.append("\n")
    content.append("Gold Algorithmic Trading System", style="bold white")
    content.append("  |  ", style="dim")
    content.append("XAUUSD", style="bold yellow")
    content.append(" . ", style="dim")
    content.append("IC Markets ECN", style="white")
    content.append(" . ", style="dim")
    content.append("MT5", style="white")
    content.append("  |  ", style="dim")
    content.append(now_utc, style="bold green")
    return Panel(Align.center(content), border_style="cyan", padding=(0, 1))


def _build_account_panel(mt5_acct: Optional[Dict], db_data: Dict) -> Panel:
    report = db_data.get("report")
    grid = Table.grid(padding=(0, 3))
    grid.add_column(justify="right", style="dim", min_width=18)
    grid.add_column(justify="left", min_width=22)

    if mt5_acct is None:
        grid.add_row("Statut MT5", Text("MT5 DECONNECTE", style="bold red"))
        grid.add_row("Balance", Text("---", style="dim"))
        grid.add_row("P&L du jour", Text("---", style="dim"))
        grid.add_row("Drawdown", Text("---", style="dim"))
    else:
        balance = mt5_acct["balance"]
        grid.add_row("Balance", Text(f"${balance:,.2f}", style="bold white"))

        daily_pnl = 0.0
        if report:
            daily_pnl = report.get("total_pnl", 0.0)
        pnl_pct = (daily_pnl / balance * 100) if balance > 0 else 0.0
        pnl_style = "bold green" if daily_pnl >= 0 else "bold red"
        pnl_sign = "+" if daily_pnl >= 0 else ""
        grid.add_row(
            "P&L du jour",
            Text(f"{pnl_sign}${daily_pnl:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)", style=pnl_style),
        )

        # Bot status heuristic
        dd_pct = abs(pnl_pct) if daily_pnl < 0 else 0.0
        if dd_pct >= 5.0:
            status = Text("KILL SWITCH", style="bold red blink")
        elif dd_pct >= 3.0:
            status = Text("EN PAUSE", style="bold yellow")
        else:
            status = Text("ACTIF", style="bold green")
        grid.add_row("Statut bot", status)

        dd_text = Text(f"{dd_pct:.2f}%", style="bold red" if dd_pct >= 3 else "white")
        dd_text.append(" / 5.00%", style="dim")
        grid.add_row("Drawdown", dd_text)

    return Panel(grid, title="[bold white]Compte[/]", border_style="blue", padding=(1, 2))


def _build_market_panel(tick: Optional[float], bias: str, atr: Optional[float]) -> Panel:
    grid = Table.grid(padding=(0, 3))
    grid.add_column(justify="right", style="dim", min_width=18)
    grid.add_column(justify="left", min_width=22)

    # Daily bias
    bias_styles = {"BULLISH": "bold green", "BEARISH": "bold red", "NEUTRAL": "bold yellow", "N/A": "dim"}
    grid.add_row("Daily Bias", Text(bias, style=bias_styles.get(bias, "dim")))

    # ATR M1
    if atr is not None:
        atr_style = "bold green" if atr >= 0.40 else "bold red"
        grid.add_row("ATR M1 (14)", Text(f"{atr:.4f}", style=atr_style))
    else:
        grid.add_row("ATR M1 (14)", Text("---", style="dim"))

    # Session
    hour = datetime.utcnow().hour
    session = _get_session(hour)
    sess_styles = {"London": "bold cyan", "New York": "bold magenta", "Hors session": "dim italic"}
    grid.add_row("Session", Text(session, style=sess_styles.get(session, "dim")))

    # Live price
    if tick is not None:
        grid.add_row("XAUUSD", Text(f"${tick:,.2f}", style="bold yellow"))
    else:
        grid.add_row("XAUUSD", Text("---", style="dim"))

    return Panel(grid, title="[bold white]Regime Marche[/]", border_style="magenta", padding=(1, 2))


def _build_last_trade_panel(trades: List) -> Panel:
    if not trades:
        return Panel(
            Align.center(Text("Aucun trade aujourd'hui", style="dim italic")),
            title="[bold white]Dernier Trade[/]",
            border_style="yellow",
            padding=(1, 2),
        )

    # Pick the most recent trade
    last = trades[-1]
    grid = Table.grid(padding=(0, 3))
    grid.add_column(justify="right", style="dim", min_width=14)
    grid.add_column(justify="left", min_width=20)

    dir_style = "bold green" if last.direction.value == "LONG" else "bold red"
    grid.add_row("Direction", Text(last.direction.value, style=dir_style))
    grid.add_row("Entree", Text(f"${last.entry_price:,.2f}", style="white"))

    if last.exit_price is not None:
        grid.add_row("Sortie", Text(f"${last.exit_price:,.2f}", style="white"))
    else:
        grid.add_row("Sortie", Text("EN COURS", style="bold cyan blink"))

    if last.pnl is not None:
        pnl_style = "bold green" if last.pnl >= 0 else "bold red"
        sign = "+" if last.pnl >= 0 else ""
        grid.add_row("P&L", Text(f"{sign}${last.pnl:,.2f}", style=pnl_style))
    else:
        grid.add_row("P&L", Text("---", style="dim"))

    reason = last.state.value if last.state else "---"
    grid.add_row("Raison sortie", Text(reason, style="white"))

    return Panel(grid, title="[bold white]Dernier Trade[/]", border_style="yellow", padding=(1, 2))


def _build_trades_table(trades: List) -> Panel:
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="dim",
        expand=True,
        row_styles=["", "dim"],
    )
    table.add_column("Heure", style="cyan", width=8)
    table.add_column("Dir", width=6)
    table.add_column("Entree", justify="right", width=10)
    table.add_column("Sortie", justify="right", width=10)
    table.add_column("Raison", width=10)
    table.add_column("P&L", justify="right", width=12)

    if not trades:
        table.add_row("--", "--", "--", "--", "--", "--")
    else:
        for i, t in enumerate(trades):
            hour_str = t.opened_at.strftime("%H:%M") if t.opened_at else "--"
            dir_style = "green" if t.direction.value == "LONG" else "red"
            entry_str = f"${t.entry_price:,.2f}"
            exit_str = f"${t.exit_price:,.2f}" if t.exit_price else "[cyan]OPEN[/]"
            reason = t.state.value if t.state else "---"

            if t.pnl is not None:
                pnl_style = "bold green" if t.pnl >= 0 else "bold red"
                sign = "+" if t.pnl >= 0 else ""
                pnl_str = f"[{pnl_style}]{sign}${t.pnl:,.2f}[/]"
            else:
                pnl_str = "[dim]---[/]"

            # Highlight last row if trade is still open
            is_open = t.state and t.state.value in ("OPEN", "TP1_HIT", "TP2_HIT")
            is_last = i == len(trades) - 1

            end_section = False
            style = ""
            if is_last and is_open:
                style = "bold on grey23"

            table.add_row(
                hour_str,
                f"[{dir_style}]{t.direction.value}[/]",
                entry_str,
                exit_str,
                reason,
                pnl_str,
                style=style,
                end_section=end_section,
            )

    return Panel(table, title="[bold white]Trades du Jour[/]", border_style="green", padding=(0, 1))


# ---------------------------------------------------------------------------
# Layout assembly
# ---------------------------------------------------------------------------
def _build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=11),
        Layout(name="body", size=10),
        Layout(name="trades"),
    )
    layout["body"].split_row(
        Layout(name="account", ratio=1),
        Layout(name="market", ratio=1),
        Layout(name="last_trade", ratio=1),
    )
    return layout


def _refresh(layout: Layout) -> None:
    """Fetch all data and update layout panels."""
    mt5_acct = _get_mt5_account()
    tick = _get_mt5_tick()
    db_data = _get_db_data()
    bias = _get_daily_bias()
    atr = _get_m1_atr()
    trades = db_data.get("trades", [])

    layout["header"].update(_build_header())
    layout["account"].update(_build_account_panel(mt5_acct, db_data))
    layout["market"].update(_build_market_panel(tick, bias, atr))
    layout["last_trade"].update(_build_last_trade_panel(trades))
    layout["trades"].update(_build_trades_table(trades))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    console = Console()
    layout = _build_layout()

    console.clear()
    console.print("[bold cyan]INOXTRADE Dashboard[/] starting...\n")

    try:
        with Live(layout, console=console, refresh_per_second=1, screen=True):
            while True:
                _refresh(layout)
                time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Dashboard stopped.[/]")


if __name__ == "__main__":
    main()
