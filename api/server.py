"""FastAPI control server for INOXTRADE bot.

Runs on localhost:8765 only (not exposed to internet).
All external access goes through SSH tunnel.

Launch:
    python api/server.py
    uvicorn api.server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from api.bot_process import BotProcess

app = FastAPI(
    title="INOXTRADE Control API",
    version="1.0.0",
    docs_url="/docs",
)


# ---------------------------------------------------------------------------
# Helpers — each catches its own errors so the API never crashes
# ---------------------------------------------------------------------------

def _get_mt5_data() -> Dict[str, Any]:
    """Fetch account info and live tick from MT5. Returns empty dict on failure."""
    try:
        import MetaTrader5 as mt5

        if not mt5.terminal_info():
            from config.settings import get_settings
            settings = get_settings()
            kwargs = {}
            if settings.MT5_PATH:
                kwargs["path"] = settings.MT5_PATH
            if not mt5.initialize(**kwargs):
                return {}
            mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER,
            )

        result: Dict[str, Any] = {}

        info = mt5.account_info()
        if info:
            result["balance"] = info.balance
            result["equity"] = info.equity
            result["profit"] = info.profit

        tick = mt5.symbol_info_tick("XAUUSD")
        if tick:
            result["bid"] = tick.bid
            result["ask"] = tick.ask

        return result
    except Exception:
        return {}


def _get_daily_bias() -> str:
    """Compute daily bias from D1 data."""
    try:
        import MetaTrader5 as mt5
        import numpy as np
        import pandas as pd
        import talib

        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_D1, 0, 60)
        if rates is None or len(rates) < 55:
            return "N/A"
        df = pd.DataFrame(rates)
        close = df["close"].values.astype(float)
        ema20 = talib.EMA(close, timeperiod=20)
        ema50 = talib.EMA(close, timeperiod=50)
        c, e20, e50 = close[-1], ema20[-1], ema50[-1]
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
        import numpy as np
        import pandas as pd
        import talib

        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 30)
        if rates is None or len(rates) < 20:
            return None
        df = pd.DataFrame(rates)
        atr = talib.ATR(
            df["high"].values.astype(float),
            df["low"].values.astype(float),
            df["close"].values.astype(float),
            timeperiod=14,
        )
        val = float(atr[-1])
        return None if np.isnan(val) else round(val, 4)
    except Exception:
        return None


def _get_session() -> str:
    """Current trading session from UTC hour."""
    hour = datetime.utcnow().hour
    if 7 <= hour < 12:
        return "London"
    elif 13 <= hour < 18:
        return "New York"
    return "Hors session"


def _get_db_trades() -> List[Dict]:
    """Fetch today's trades from SQLite."""
    try:
        from storage import get_database
        db = get_database()
        trades = db.get_today_trades()
        result = []
        for t in trades:
            result.append({
                "id": t.id,
                "ticket": t.ticket,
                "symbol": t.symbol,
                "direction": t.direction.value if t.direction else None,
                "state": t.state.value if t.state else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "tp1_price": t.tp1_price,
                "tp2_price": t.tp2_price,
                "lot_size": t.lot_size,
                "pnl": t.pnl,
                "pnl_pips": t.pnl_pips,
                "opened_at": t.opened_at.isoformat() if t.opened_at else None,
                "closed_at": t.closed_at.isoformat() if t.closed_at else None,
                "entry_reason": t.entry_reason,
                "strategy_name": t.strategy_name,
                "signal_strength": t.signal_strength,
            })
        return result
    except Exception:
        return []


def _get_db_daily_pnl() -> float:
    """Sum of today's realized P&L from SQLite."""
    try:
        from storage import get_database
        db = get_database()
        return db.get_daily_pnl()
    except Exception:
        return 0.0


def _get_db_report() -> Dict[str, Any]:
    """Daily report data from SQLite."""
    try:
        from storage import get_database
        db = get_database()
        return db.get_daily_report_data()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status():
    """Full bot status: process, account, market, trades summary."""
    bot = BotProcess.status()
    mt5_data = _get_mt5_data()
    report = _get_db_report()
    daily_pnl = _get_db_daily_pnl()

    balance = mt5_data.get("balance")
    daily_pnl_pct = (daily_pnl / balance * 100) if balance and balance > 0 else 0.0

    return {
        "bot": {
            "running": bot["running"],
            "pid": bot["pid"],
            "uptime_seconds": bot["uptime_seconds"],
        },
        "account": {
            "balance": balance,
            "equity": mt5_data.get("equity"),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "mt5_connected": bool(mt5_data),
        },
        "market": {
            "bid": mt5_data.get("bid"),
            "ask": mt5_data.get("ask"),
            "daily_bias": _get_daily_bias(),
            "atr_m1": _get_m1_atr(),
            "session": _get_session(),
        },
        "trades": {
            "total_today": report.get("total_trades", 0),
            "closed_today": report.get("closed_trades", 0),
            "wins": report.get("wins", 0),
            "losses": report.get("losses", 0),
            "win_rate": round(report.get("win_rate", 0.0) * 100, 1),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/start")
def start_bot():
    """Start the trading bot (main.py) as a background process."""
    result = BotProcess.start()
    return result


@app.post("/stop")
def stop_bot():
    """Stop the trading bot gracefully."""
    result = BotProcess.stop()
    return result


@app.get("/trades")
def get_trades():
    """Today's trades from SQLite."""
    trades = _get_db_trades()
    return {
        "count": len(trades),
        "trades": trades,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/logs", response_class=PlainTextResponse)
def get_logs(lines: int = 50):
    """Last N lines of the bot log file."""
    return BotProcess.tail_log(lines=min(lines, 500))


@app.get("/ping")
def ping():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Standalone launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("Starting INOXTRADE API on http://127.0.0.1:8765")
    print("Docs: http://127.0.0.1:8765/docs")
    uvicorn.run(
        "api.server:app",
        host="127.0.0.1",
        port=8765,
        log_level="info",
        reload=False,
    )
