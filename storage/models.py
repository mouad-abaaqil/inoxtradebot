"""SQLAlchemy ORM models for the trading bot."""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    Text,
    Enum,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class TradeDirection(str, PyEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeState(str, PyEnum):
    OPEN = "OPEN"
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    CLOSED = "CLOSED"
    SL_HIT = "SL_HIT"


class Trade(Base):
    """Append-only trade log — full audit trail."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(Integer, index=True, nullable=True)  # MT5 ticket number
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(Enum(TradeDirection), nullable=False)
    state = Column(Enum(TradeState), default=TradeState.OPEN)

    # Strategy info
    strategy_name = Column(String(100), nullable=False)
    signal_strength = Column(Float, nullable=False)
    signal_confidence = Column(Float, nullable=True)
    entry_reason = Column(Text, nullable=True)

    # Prices
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    tp1_price = Column(Float, nullable=True)
    tp2_price = Column(Float, nullable=True)
    tp3_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)

    # Position sizing
    lot_size = Column(Float, nullable=False)
    risk_amount = Column(Float, nullable=False)  # dollar risk
    sl_pips = Column(Float, nullable=True)

    # Slippage tracking
    requested_price = Column(Float, nullable=True)
    fill_price = Column(Float, nullable=True)
    slippage_pips = Column(Float, nullable=True)

    # P&L
    pnl = Column(Float, nullable=True)  # realized
    pnl_pips = Column(Float, nullable=True)

    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    closed_at = Column(DateTime, nullable=True)
    tp1_hit_at = Column(DateTime, nullable=True)
    tp2_hit_at = Column(DateTime, nullable=True)

    # Indicators snapshot (JSON)
    indicators_snapshot = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Trade {self.id} {self.symbol} {self.direction.value} {self.state.value}>"


class AccountSnapshot(Base):
    """Periodic balance/equity snapshots."""

    __tablename__ = "account_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    margin = Column(Float, nullable=True)
    free_margin = Column(Float, nullable=True)
    daily_pnl = Column(Float, nullable=True)
    daily_pnl_pct = Column(Float, nullable=True)
    open_trades_count = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<Snapshot {self.timestamp} bal={self.balance}>"


class StrategyRecord(Base):
    """Optimization results and strategy performance records."""

    __tablename__ = "strategy_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    params_json = Column(Text, nullable=False)  # JSON serialized params

    # Metrics
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    trade_count = Column(Integer, nullable=True)
    total_return = Column(Float, nullable=True)

    # Validation
    is_oos = Column(Boolean, default=False)  # out-of-sample result
    is_active = Column(Boolean, default=False)
    validated_at = Column(DateTime, default=datetime.utcnow)
    train_start = Column(DateTime, nullable=True)
    train_end = Column(DateTime, nullable=True)
    test_start = Column(DateTime, nullable=True)
    test_end = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<StrategyRecord {self.strategy_name} sharpe={self.sharpe_ratio}>"


class SignalLog(Base):
    """Audit trail for every signal generated, including rejected ones."""

    __tablename__ = "signal_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False)
    direction = Column(String(10), nullable=False)
    strength = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    entry_reason = Column(Text, nullable=True)

    # Decision
    accepted = Column(Boolean, default=False)
    rejection_reason = Column(Text, nullable=True)

    # Indicator values at signal time
    indicators_json = Column(Text, nullable=True)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<Signal {self.symbol} {self.direction} str={self.strength} {'OK' if self.accepted else 'REJ'}>"
