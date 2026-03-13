"""FastAPI control server for INOXTRADE bot.

Runs on localhost:8765 only (not exposed to internet).
All external access goes through SSH tunnel.

Launch:
    python api/server.py
    uvicorn api.server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse

from api.bot_process import BotProcess

app = FastAPI(
    title="INOXTRADE Control API",
    version="2.0.0",
    docs_url="/docs",
)

STRATEGY_FILE = Path(PROJECT_ROOT) / "config" / "active_strategy.txt"
BACKTEST_SCRIPT = Path(PROJECT_ROOT) / "scripts" / "backtest.py"
BACKTEST_REPORT = Path(PROJECT_ROOT) / "results" / "backtest_report.txt"
ENV_FILE = Path(PROJECT_ROOT) / ".env"

# ---------------------------------------------------------------------------
# Backtest job state (in-memory, single job at a time)
# ---------------------------------------------------------------------------
_backtest_lock = threading.Lock()
_backtest_job: Dict[str, Any] = {"running": False, "done": False, "result": None}


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
            result["login"] = info.login
            result["server"] = info.server

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
    """Fetch current M1 ATR(14) — raw dollar value."""
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
        if np.isnan(val):
            return None
        return round(val, 4)
    except Exception:
        return None


def _get_ema20_slope() -> Optional[float]:
    """Compute EMA20 slope on M5 (last 2 candles difference)."""
    try:
        import MetaTrader5 as mt5
        import numpy as np
        import pandas as pd
        import talib

        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 30)
        if rates is None or len(rates) < 25:
            return None
        df = pd.DataFrame(rates)
        close = df["close"].values.astype(float)
        ema20 = talib.EMA(close, timeperiod=20)
        if np.isnan(ema20[-1]) or np.isnan(ema20[-2]):
            return None
        slope = round(ema20[-1] - ema20[-2], 4)
        return slope
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


def _get_active_strategy() -> str:
    """Read active strategy from file, default DTF."""
    try:
        if STRATEGY_FILE.exists():
            return STRATEGY_FILE.read_text(encoding="utf-8").strip() or "DTF"
    except Exception:
        pass
    return "DTF"


def _get_db_trades(days: Optional[int] = None) -> List[Dict]:
    """Fetch trades from SQLite. If days is None, return today only."""
    try:
        from storage import get_database
        db = get_database()

        if days is None or days <= 1:
            trades = db.get_today_trades()
        else:
            # Query trades from last N days
            from sqlalchemy import select as sa_select
            from storage.models import Trade
            cutoff = datetime.utcnow() - timedelta(days=days)
            with db.session() as s:
                stmt = sa_select(Trade).where(Trade.opened_at >= cutoff)
                trades = list(s.scalars(stmt).all())

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

@app.get("/ping")
def ping():
    """Health check."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


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
            "strategy": _get_active_strategy(),
        },
        "account": {
            "balance": balance,
            "equity": mt5_data.get("equity"),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 2),
            "mt5_connected": bool(mt5_data),
            "login": mt5_data.get("login"),
            "server": mt5_data.get("server"),
        },
        "market": {
            "bid": mt5_data.get("bid"),
            "ask": mt5_data.get("ask"),
            "daily_bias": _get_daily_bias(),
            "atr_m1": _get_m1_atr(),
            "ema20_slope": _get_ema20_slope(),
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
    return BotProcess.start()


@app.post("/stop")
def stop_bot():
    """Stop the trading bot gracefully."""
    return BotProcess.stop()


@app.get("/trades")
def get_trades(
    days: Optional[int] = Query(None, description="Number of days to fetch"),
    format: Optional[str] = Query(None, description="Response format: json or csv"),
):
    """Trades from SQLite. Supports ?days=N and ?format=csv."""
    trades = _get_db_trades(days=days)

    if format == "csv":
        if not trades:
            return PlainTextResponse("No trades found", media_type="text/csv")
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=trades[0].keys())
        writer.writeheader()
        writer.writerows(trades)
        return PlainTextResponse(output.getvalue(), media_type="text/csv")

    return {
        "count": len(trades),
        "trades": trades,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/logs", response_class=PlainTextResponse)
def get_logs(lines: int = 50):
    """Last N lines of the bot log file."""
    return BotProcess.tail_log(lines=min(lines, 500))


# ---------------------------------------------------------------------------
# POST /strategy — change active strategy
# ---------------------------------------------------------------------------

@app.post("/strategy")
def set_strategy(body: dict):
    """Set the active trading strategy. Body: {"name": "DTF"|"ERM"|"VRS"}."""
    name = body.get("name", "").strip().upper()
    valid = {"DTF", "ERM", "VRS"}
    if name not in valid:
        return {"success": False, "message": f"Unknown strategy: {name}. Valid: {', '.join(sorted(valid))}"}

    try:
        STRATEGY_FILE.parent.mkdir(parents=True, exist_ok=True)
        STRATEGY_FILE.write_text(name, encoding="utf-8")
        return {"success": True, "message": f"Strategy set to {name}", "strategy": name}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ---------------------------------------------------------------------------
# POST /backtest — launch async backtest
# GET  /backtest/status — poll job status
# GET  /backtest/result — get parsed results
# ---------------------------------------------------------------------------

def _run_backtest(strategy: str, days: int, lot: float):
    """Background thread that runs backtest.py and parses results."""
    global _backtest_job
    try:
        cmd = [
            sys.executable,
            str(BACKTEST_SCRIPT),
            "--days", str(days),
            "--lot", str(lot),
        ]
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )

        parsed = _parse_backtest_output(result.stdout)

        with _backtest_lock:
            _backtest_job["running"] = False
            _backtest_job["done"] = True
            _backtest_job["result"] = parsed
            _backtest_job["exit_code"] = result.returncode
            _backtest_job["stderr"] = result.stderr[-500:] if result.stderr else ""

    except subprocess.TimeoutExpired:
        with _backtest_lock:
            _backtest_job["running"] = False
            _backtest_job["done"] = True
            _backtest_job["result"] = {"error": "Backtest timed out (10 min)"}
    except Exception as e:
        with _backtest_lock:
            _backtest_job["running"] = False
            _backtest_job["done"] = True
            _backtest_job["result"] = {"error": str(e)}


def _parse_backtest_output(stdout: str) -> Dict[str, Any]:
    """Parse backtest stdout for key metrics."""
    result: Dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        low = line.lower()
        # Look for common metric patterns like "Total Trades : 42"
        if "total trades" in low:
            result["total_trades"] = _extract_number(line)
        elif "win rate" in low:
            val = _extract_number(line)
            result["win_rate"] = val
        elif "profit factor" in low:
            result["profit_factor"] = _extract_float(line)
        elif "net p&l" in low or "net pnl" in low or "total pnl" in low:
            result["pnl"] = _extract_float(line)
        elif "max drawdown" in low or "max dd" in low:
            result["max_drawdown"] = _extract_float(line)

    # Also try to read the report file
    if not result and BACKTEST_REPORT.exists():
        try:
            text = BACKTEST_REPORT.read_text(encoding="utf-8")
            for line in text.splitlines():
                low = line.strip().lower()
                if "total trades" in low:
                    result["total_trades"] = _extract_number(line)
                elif "win rate" in low:
                    result["win_rate"] = _extract_number(line)
                elif "profit factor" in low:
                    result["profit_factor"] = _extract_float(line)
                elif "net p&l" in low or "net pnl" in low or "total pnl" in low:
                    result["pnl"] = _extract_float(line)
                elif "max drawdown" in low or "max dd" in low:
                    result["max_drawdown"] = _extract_float(line)
        except Exception:
            pass

    return result


def _extract_number(line: str) -> Optional[int]:
    """Extract first integer from a line."""
    import re
    m = re.search(r"[\d,]+", line.split(":")[-1] if ":" in line else line)
    if m:
        return int(m.group().replace(",", ""))
    return None


def _extract_float(line: str) -> Optional[float]:
    """Extract first float from a line."""
    import re
    m = re.search(r"-?[\d,]+\.?\d*", line.split(":")[-1] if ":" in line else line)
    if m:
        return float(m.group().replace(",", ""))
    return None


@app.post("/backtest")
def launch_backtest(body: dict):
    """Launch a backtest asynchronously. Body: {"strategy": "DTF", "days": 30, "lot": 0.02}."""
    strategy = body.get("strategy", "DTF")
    days = body.get("days", 30)
    lot = body.get("lot", 0.02)

    with _backtest_lock:
        if _backtest_job["running"]:
            return {"success": False, "message": "A backtest is already running", "async": False}
        _backtest_job["running"] = True
        _backtest_job["done"] = False
        _backtest_job["result"] = None

    t = threading.Thread(target=_run_backtest, args=(strategy, days, lot), daemon=True)
    t.start()

    return {"success": True, "message": f"Backtest {strategy} {days}j started", "async": True}


@app.get("/backtest/status")
def backtest_status():
    """Poll backtest job status."""
    with _backtest_lock:
        return {
            "running": _backtest_job["running"],
            "done": _backtest_job["done"],
        }


@app.get("/backtest/result")
def backtest_result():
    """Get the last backtest result."""
    with _backtest_lock:
        if not _backtest_job["done"]:
            return {"error": "No backtest result available"}
        return _backtest_job["result"] or {}


# ---------------------------------------------------------------------------
# POST /mt5/credentials — update .env with new MT5 credentials
# ---------------------------------------------------------------------------

@app.post("/mt5/credentials")
def set_mt5_credentials(body: dict):
    """Update MT5 credentials in .env file. Body: {"login": ..., "password": ..., "server": ...}."""
    login = body.get("login", "")
    password = body.get("password", "")
    server = body.get("server", "")

    if not login or not password or not server:
        return {"success": False, "message": "login, password, and server are required"}

    try:
        # Read existing .env
        env_lines = []
        if ENV_FILE.exists():
            env_lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

        # Update or add MT5 keys
        updates = {
            "MT5_LOGIN": str(login),
            "MT5_PASSWORD": str(password),
            "MT5_SERVER": str(server),
        }
        found_keys = set()
        new_lines = []
        for line in env_lines:
            key = line.split("=", 1)[0].strip() if "=" in line else ""
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                found_keys.add(key)
            else:
                new_lines.append(line)

        # Add missing keys
        for key, val in updates.items():
            if key not in found_keys:
                new_lines.append(f"{key}={val}")

        ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        # Clear cached settings so next import picks up new values
        try:
            from config.settings import get_settings
            get_settings.cache_clear()
        except Exception:
            pass

        return {"success": True, "message": "MT5 credentials updated. Restart bot to apply."}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ---------------------------------------------------------------------------
# GET  /positions — live MT5 open positions
# POST /positions/close_all — close all positions
# POST /positions/close — close one position by ticket
# ---------------------------------------------------------------------------

@app.get("/positions")
def get_positions():
    """Get all open positions from MT5."""
    try:
        import MetaTrader5 as mt5

        if not mt5.terminal_info():
            from config.settings import get_settings
            settings = get_settings()
            kwargs = {}
            if settings.MT5_PATH:
                kwargs["path"] = settings.MT5_PATH
            if not mt5.initialize(**kwargs):
                return {"positions": [], "error": "MT5 not initialized"}
            mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER,
            )

        positions = mt5.positions_get()
        if positions is None:
            return {"positions": []}

        result = []
        for p in positions:
            direction = "LONG" if p.type == 0 else "SHORT"
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "direction": direction,
                "volume": p.volume,
                "entry_price": p.price_open,
                "current_price": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "pnl": p.profit,
                "swap": p.swap,
                "comment": p.comment,
            })

        return {"positions": result, "count": len(result)}
    except Exception as e:
        return {"positions": [], "error": str(e)}


@app.post("/positions/close_all")
def close_all_positions():
    """Close all open MT5 positions."""
    try:
        import MetaTrader5 as mt5

        if not mt5.terminal_info():
            from config.settings import get_settings
            settings = get_settings()
            kwargs = {}
            if settings.MT5_PATH:
                kwargs["path"] = settings.MT5_PATH
            if not mt5.initialize(**kwargs):
                return {"success": False, "message": "MT5 not initialized"}
            mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER,
            )

        positions = mt5.positions_get()
        if not positions:
            return {"success": True, "message": "No positions to close", "closed": 0}

        closed = 0
        errors = []
        for p in positions:
            close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(p.symbol)
            price = tick.bid if p.type == 0 else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": close_type,
                "position": p.ticket,
                "price": price,
                "deviation": 20,
                "magic": 0,
                "comment": "INOXTRADE close_all",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
            else:
                retcode = result.retcode if result else "no result"
                errors.append(f"Ticket {p.ticket}: {retcode}")

        msg = f"{closed}/{len(positions)} positions closed"
        if errors:
            msg += f" | Errors: {'; '.join(errors)}"

        return {"success": closed > 0, "message": msg, "closed": closed}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/positions/close")
def close_position(body: dict):
    """Close one position by ticket. Body: {"ticket": 123456}."""
    ticket = body.get("ticket")
    if not ticket:
        return {"success": False, "message": "ticket is required"}

    try:
        import MetaTrader5 as mt5

        if not mt5.terminal_info():
            from config.settings import get_settings
            settings = get_settings()
            kwargs = {}
            if settings.MT5_PATH:
                kwargs["path"] = settings.MT5_PATH
            if not mt5.initialize(**kwargs):
                return {"success": False, "message": "MT5 not initialized"}
            mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER,
            )

        positions = mt5.positions_get(ticket=int(ticket))
        if not positions:
            return {"success": False, "message": f"Position {ticket} not found"}

        p = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(p.symbol)
        price = tick.bid if p.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_type,
            "position": p.ticket,
            "price": price,
            "deviation": 20,
            "magic": 0,
            "comment": "INOXTRADE close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "message": f"Position {ticket} closed"}
        else:
            retcode = result.retcode if result else "no result"
            comment = result.comment if result else ""
            return {"success": False, "message": f"Failed: {retcode} {comment}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


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
