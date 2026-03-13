"""Pydantic-based configuration with type-safe validation."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # --- MT5 Credentials ---
    MT5_LOGIN: int = 0
    MT5_PASSWORD: str = ""
    MT5_SERVER: str = ""
    MT5_PATH: str = ""  # path to terminal64.exe if needed

    # --- Telegram ---
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: int = 0

    # --- Trading Symbols ---
    SYMBOLS: List[str] = Field(default=["XAUUSD"])

    # --- Broker Settings (IC Markets) ---
    COMMISSION_PER_LOT: float = 7.0  # USD per lot (both sides = $14 round trip)
    SWAP_FREE: bool = True

    # --- Risk Management ---
    RISK_PER_TRADE: float = 0.01
    MAX_DAILY_LOSS: float = 0.05
    MIN_RR_RATIO: float = 1.0
    MAX_SPREAD_MULTIPLIER: float = 3.0
    ATR_SL_MULTIPLIER: float = 1.0
    MIN_SL_DISTANCE_XAUUSD: float = 1.50  # plancher SL en $ pour Gold (évite lots absurdes)
    TP1_RR: float = 1.0
    TP2_RR: float = 2.0
    TP1_CLOSE_PCT: float = 0.60
    TP2_CLOSE_PCT: float = 0.40
    MAX_OPEN_TRADES: int = 5
    CONSECUTIVE_LOSS_PAUSE: int = 100
    PAUSE_DURATION_MINUTES: int = 30
    MAX_SINGLE_TRADE_RISK: float = 0.02
    MIN_SIGNAL_STRENGTH: float = 0.6

    # --- Trading Control ---
    DEMO_MODE: bool = False
    TRADE_POLL_INTERVAL: int = 5  # seconds
    CANDLE_BUFFER_SIZE: int = 500
    HISTORICAL_DAYS_1M: int = 90
    HISTORICAL_DAYS_DAILY: int = 730  # ~2 years

    # --- Optimization ---
    OPTUNA_TRIALS: int = 200
    OPTIMIZATION_TRAIN_DAYS: int = 30
    OPTIMIZATION_TEST_DAYS: int = 10
    MIN_QUALIFYING_TRADES: int = 50
    MIN_OOS_SHARPE: float = 1.0
    MIN_PROFIT_FACTOR: float = 1.3
    MAX_DRAWDOWN_PCT: float = 0.15
    MIN_WIN_RATE: float = 0.35
    PARAM_SENSITIVITY_THRESHOLD: float = 0.20  # max 20% Sharpe drop

    # --- Database ---
    DATABASE_URL: str = "sqlite:///scalping_bot.db"

    # --- DTF Strategy (Daily Trend Follower) ---
    DTF_MIN_ATR: float = 0.40
    DTF_RSI_PULLBACK_LONG_MIN: int = 35
    DTF_RSI_PULLBACK_LONG_MAX: int = 55
    DTF_RSI_PULLBACK_SHORT_MIN: int = 45
    DTF_RSI_PULLBACK_SHORT_MAX: int = 65
    DTF_BODY_RATIO_MIN: float = 0.40
    DTF_MIN_TP1_PIPS: int = 10
    DTF_MAX_SL_FACTOR: float = 1.5
    DTF_SL_SWING_BARS: int = 5
    DTF_TP1_RR: float = 1.0
    DTF_TP2_RR: float = 2.0
    DTF_COOLDOWN_BASE: int = 180
    DTF_SESSION_WINDOWS: str = "7-12,13-18"  # UTC hour ranges

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"

    # --- Symbol Config Helpers ---
    def pip_value(self, symbol: str) -> float:
        """Return the pip size for a given symbol."""
        sym = symbol.upper().replace("/", "")
        if "JPY" in sym:
            return 0.01
        if "XAU" in sym:
            return 0.01
        return 0.0001

    def pip_digits(self, symbol: str) -> int:
        sym = symbol.upper().replace("/", "")
        if "JPY" in sym:
            return 3
        if "XAU" in sym:
            return 2
        return 5


@lru_cache()
def get_settings() -> Settings:
    return Settings()
