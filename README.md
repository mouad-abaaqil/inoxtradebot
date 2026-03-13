# INOXTRADE

Gold Algorithmic Trading Bot — XAUUSD · IC Markets ECN · MT5

## Stack

- **Python 3.11** + MetaTrader 5
- **Strategy** : Daily Trend Follower (DTF) — trades only in Daily trend direction with M1 pullback entries
- **Broker** : IC Markets ECN (spread 0, commission $7/lot)
- **Infrastructure** : Windows VPS + MT5 terminal

## Structure

```
main.py                        # asyncio entry point
├── config/
│   └── settings.py            # Pydantic-based configuration (.env)
├── data/
│   ├── market_data.py         # MT5 connection + candle streaming
│   └── feature_engine.py      # TA-Lib indicator computation
├── strategies/
│   ├── base_strategy.py       # Abstract base + Signal dataclass
│   ├── gold_scalper.py        # Daily Trend Follower (primary)
│   ├── rsi_reversal.py        # RSI extreme reversal
│   ├── macd_crossover.py      # MACD histogram zero-cross
│   ├── bb_squeeze.py          # Bollinger Band squeeze breakout
│   ├── vwap_bounce.py         # VWAP touch with stochastic
│   ├── ema_pullback.py        # Trend pullback to EMA8
│   ├── volume_surge.py        # High-volume momentum breakout
│   └── registry.py            # Strategy store + hot-swap
├── execution/
│   ├── risk_engine.py         # Position sizing + 5% daily kill switch
│   ├── decision_engine.py     # Pre-trade checklist
│   ├── order_executor.py      # MT5 order placement
│   └── trade_manager.py       # TP ladder state machine
├── optimization/
│   ├── backtester.py          # vectorbt wrapper
│   └── optimizer.py           # Optuna study
├── scripts/
│   └── backtest.py            # Standalone backtest on MT5 historical data
├── ui/
│   └── dashboard.py           # Rich terminal dashboard (real-time)
├── notifications/
│   ├── telegram_bot.py        # Async bot + commands
│   └── formatters.py          # Message templates
├── sentiment/
│   └── sentiment_analyzer.py  # News sentiment (stub)
└── storage/
    ├── models.py              # SQLAlchemy ORM (Trade, AccountSnapshot)
    └── db.py                  # Session management + queries
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your MT5 credentials and Telegram bot token
python main.py
```

## Backtest

```bash
python scripts/backtest.py --days 30 --lot 0.02
python scripts/backtest.py --days 90 --lot 0.04
```

## Dashboard

```bash
python ui/dashboard.py
```

## Risk Management

- 1% risk per trade (ATR-based stop loss)
- 5% daily loss kill switch
- Strategy-provided SL/TP passthrough
- TP1 partial close (60%) + SL-to-breakeven
- Max 5 concurrent trades
- Adaptive cooldown between signals

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Bot status, open trades, daily P&L |
| `/strategy` | Active strategies per symbol |
| `/pause` | Pause trading for 60 minutes |
| `/resume` | Resume trading (clear kill switch) |
| `/report` | Today's performance report |
| `/close_all` | Emergency close all positions |
