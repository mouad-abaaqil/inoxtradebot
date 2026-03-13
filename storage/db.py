"""Database session management, queries, and backup utilities."""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from loguru import logger
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from storage.models import (
    Base,
    Trade,
    TradeState,
    AccountSnapshot,
    StrategyRecord,
    SignalLog,
)


class Database:
    def __init__(self, url: Optional[str] = None):
        self._url = url or get_settings().DATABASE_URL
        self._engine = create_engine(self._url, echo=False)
        self._SessionFactory = sessionmaker(bind=self._engine)
        Base.metadata.create_all(self._engine)
        logger.info("Database initialized: {}", self._url)

    def session(self) -> Session:
        return self._SessionFactory()

    # ---------- Trade Queries ----------

    def save_trade(self, trade: Trade) -> Trade:
        with self.session() as s:
            s.add(trade)
            s.commit()
            s.refresh(trade)
            return trade

    def update_trade(self, trade_id: int, **kwargs) -> Optional[Trade]:
        with self.session() as s:
            trade = s.get(Trade, trade_id)
            if not trade:
                return None
            for k, v in kwargs.items():
                setattr(trade, k, v)
            s.commit()
            s.refresh(trade)
            return trade

    def get_open_trades(self) -> List[Trade]:
        with self.session() as s:
            stmt = select(Trade).where(
                Trade.state.in_([TradeState.OPEN, TradeState.TP1_HIT, TradeState.TP2_HIT])
            )
            return list(s.scalars(stmt).all())

    def get_trade_by_ticket(self, ticket: int) -> Optional[Trade]:
        with self.session() as s:
            stmt = select(Trade).where(Trade.ticket == ticket)
            return s.scalars(stmt).first()

    def get_today_trades(self) -> List[Trade]:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with self.session() as s:
            stmt = select(Trade).where(Trade.opened_at >= today_start)
            return list(s.scalars(stmt).all())

    def get_daily_pnl(self) -> float:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with self.session() as s:
            result = s.execute(
                select(func.coalesce(func.sum(Trade.pnl), 0.0)).where(
                    Trade.opened_at >= today_start,
                    Trade.pnl.isnot(None),
                )
            ).scalar()
            return float(result)

    def get_consecutive_losses(self) -> int:
        """Count consecutive losses from the most recent trade backward."""
        with self.session() as s:
            stmt = (
                select(Trade)
                .where(Trade.state.in_([TradeState.CLOSED, TradeState.SL_HIT]))
                .order_by(Trade.closed_at.desc())
                .limit(20)
            )
            trades = list(s.scalars(stmt).all())
            count = 0
            for t in trades:
                if t.pnl is not None and t.pnl < 0:
                    count += 1
                else:
                    break
            return count

    # ---------- Account Snapshots ----------

    def save_snapshot(self, snapshot: AccountSnapshot) -> None:
        with self.session() as s:
            s.add(snapshot)
            s.commit()

    def get_latest_snapshot(self) -> Optional[AccountSnapshot]:
        with self.session() as s:
            stmt = select(AccountSnapshot).order_by(AccountSnapshot.timestamp.desc()).limit(1)
            return s.scalars(stmt).first()

    # ---------- Strategy Records ----------

    def save_strategy_record(self, record: StrategyRecord) -> None:
        with self.session() as s:
            s.add(record)
            s.commit()

    def get_active_strategy(self, symbol: str) -> Optional[StrategyRecord]:
        with self.session() as s:
            stmt = (
                select(StrategyRecord)
                .where(StrategyRecord.symbol == symbol, StrategyRecord.is_active == True)
                .order_by(StrategyRecord.validated_at.desc())
                .limit(1)
            )
            return s.scalars(stmt).first()

    def deactivate_strategies(self, symbol: str, strategy_name: str) -> None:
        with self.session() as s:
            stmt = (
                select(StrategyRecord)
                .where(
                    StrategyRecord.symbol == symbol,
                    StrategyRecord.strategy_name == strategy_name,
                    StrategyRecord.is_active == True,
                )
            )
            for rec in s.scalars(stmt).all():
                rec.is_active = False
            s.commit()

    # ---------- Signal Logs ----------

    def log_signal(self, signal_log: SignalLog) -> None:
        with self.session() as s:
            s.add(signal_log)
            s.commit()

    # ---------- Reports ----------

    def get_daily_report_data(self) -> dict:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with self.session() as s:
            trades_today = list(
                s.scalars(select(Trade).where(Trade.opened_at >= today_start)).all()
            )
            closed = [t for t in trades_today if t.pnl is not None]
            total_pnl = sum(t.pnl for t in closed)
            wins = [t for t in closed if t.pnl > 0]
            losses = [t for t in closed if t.pnl <= 0]
            return {
                "total_trades": len(trades_today),
                "closed_trades": len(closed),
                "total_pnl": total_pnl,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(closed) if closed else 0.0,
                "best_trade": max((t.pnl for t in closed), default=0.0),
                "worst_trade": min((t.pnl for t in closed), default=0.0),
            }

    # ---------- Backup ----------

    def backup(self, backup_dir: str = "backups") -> Optional[str]:
        """Copy the SQLite database file as a daily backup."""
        if not self._url.startswith("sqlite"):
            logger.warning("Backup only supported for SQLite databases")
            return None
        db_path = self._url.replace("sqlite:///", "")
        if not Path(db_path).exists():
            logger.warning("DB file not found for backup: {}", db_path)
            return None
        bak_dir = Path(backup_dir)
        bak_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest = bak_dir / f"scalping_bot_{stamp}.db"
        shutil.copy2(db_path, dest)
        logger.info("Database backed up to {}", dest)
        return str(dest)


_db_instance: Optional[Database] = None


def get_database() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
