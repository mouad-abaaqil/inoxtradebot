"""Sentiment analyzer with hardcoded 2026 high-impact event schedule.

Blocks trades 15 minutes before and 30 minutes after known events:
NFP, CPI, PPI, FOMC, ECB, GDP releases.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

from loguru import logger


# 2026 High-Impact Economic Events (month, day, hour_utc, minute_utc, name)
# Times are approximate typical release times.
# NFP: first Friday of month, 13:30 UTC
# CPI: ~10th-15th of month, 13:30 UTC
# PPI: day after CPI typically, 13:30 UTC
# FOMC: 8 meetings per year, 19:00 UTC (statement + press conf)
# ECB: 8 meetings per year, 13:15 UTC (rate decision)
# GDP: end of month, 13:30 UTC (advance/preliminary/final)
_EVENTS_2026: List[Tuple[int, int, int, int, str]] = [
    # January
    (1, 9, 13, 30, "NFP"),
    (1, 14, 13, 30, "CPI"),
    (1, 15, 13, 30, "PPI"),
    (1, 22, 13, 15, "ECB"),
    (1, 28, 19, 0, "FOMC"),
    (1, 29, 13, 30, "GDP"),
    # February
    (2, 6, 13, 30, "NFP"),
    (2, 11, 13, 30, "CPI"),
    (2, 12, 13, 30, "PPI"),
    (2, 26, 13, 30, "GDP"),
    # March
    (3, 6, 13, 30, "NFP"),
    (3, 11, 13, 30, "CPI"),
    (3, 12, 13, 30, "PPI"),
    (3, 12, 13, 15, "ECB"),
    (3, 18, 19, 0, "FOMC"),
    (3, 26, 13, 30, "GDP"),
    # April
    (4, 3, 13, 30, "NFP"),
    (4, 14, 13, 30, "CPI"),
    (4, 15, 13, 30, "PPI"),
    (4, 16, 13, 15, "ECB"),
    (4, 29, 13, 30, "GDP"),
    # May
    (5, 1, 13, 30, "NFP"),
    (5, 13, 13, 30, "CPI"),
    (5, 14, 13, 30, "PPI"),
    (5, 6, 19, 0, "FOMC"),
    (5, 28, 13, 30, "GDP"),
    # June
    (6, 5, 13, 30, "NFP"),
    (6, 10, 13, 30, "CPI"),
    (6, 11, 13, 30, "PPI"),
    (6, 4, 13, 15, "ECB"),
    (6, 17, 19, 0, "FOMC"),
    (6, 25, 13, 30, "GDP"),
    # July
    (7, 2, 13, 30, "NFP"),
    (7, 15, 13, 30, "CPI"),
    (7, 16, 13, 30, "PPI"),
    (7, 16, 13, 15, "ECB"),
    (7, 29, 19, 0, "FOMC"),
    (7, 30, 13, 30, "GDP"),
    # August
    (8, 7, 13, 30, "NFP"),
    (8, 12, 13, 30, "CPI"),
    (8, 13, 13, 30, "PPI"),
    (8, 27, 13, 30, "GDP"),
    # September
    (9, 4, 13, 30, "NFP"),
    (9, 10, 13, 30, "CPI"),
    (9, 11, 13, 30, "PPI"),
    (9, 10, 13, 15, "ECB"),
    (9, 16, 19, 0, "FOMC"),
    (9, 24, 13, 30, "GDP"),
    # October
    (10, 2, 13, 30, "NFP"),
    (10, 14, 13, 30, "CPI"),
    (10, 15, 13, 30, "PPI"),
    (10, 22, 13, 15, "ECB"),
    (10, 29, 13, 30, "GDP"),
    # November
    (11, 6, 13, 30, "NFP"),
    (11, 12, 13, 30, "CPI"),
    (11, 13, 13, 30, "PPI"),
    (11, 4, 19, 0, "FOMC"),
    (11, 25, 13, 30, "GDP"),
    # December
    (12, 4, 13, 30, "NFP"),
    (12, 10, 13, 30, "CPI"),
    (12, 11, 13, 30, "PPI"),
    (12, 10, 13, 15, "ECB"),
    (12, 16, 19, 0, "FOMC"),
    (12, 23, 13, 30, "GDP"),
]


class SentimentAnalyzer:
    """News event avoidance module with hardcoded 2026 schedule."""

    def __init__(self):
        # Pre-build event datetime list for fast lookup
        self._events: List[Tuple[datetime, str]] = []
        year = 2026
        for month, day, hour, minute, name in _EVENTS_2026:
            try:
                dt = datetime(year, month, day, hour, minute)
                self._events.append((dt, name))
            except ValueError:
                logger.warning("Invalid event date: {}/{}/{} {}:{}", year, month, day, hour, minute)
        self._events.sort(key=lambda x: x[0])
        logger.info("SentimentAnalyzer initialized with {} high-impact events for 2026", len(self._events))

    def get_sentiment(self, symbol: str) -> float:
        """Get sentiment score for a symbol. Returns 0.0 (neutral)."""
        return 0.0

    def is_high_impact_event_near(self) -> bool:
        """Check if a high-impact news event is within the blackout window.

        Blackout: 15 minutes before to 30 minutes after event time.
        """
        now = datetime.utcnow()
        for event_time, event_name in self._events:
            window_start = event_time - timedelta(minutes=15)
            window_end = event_time + timedelta(minutes=30)
            if window_start <= now <= window_end:
                logger.info("High-impact event blackout: {} at {}", event_name, event_time)
                return True
            # Optimization: if event is more than 1 hour in the future, stop checking
            if event_time > now + timedelta(hours=1):
                break
        return False

    def sentiment_allows(self, direction: str) -> bool:
        """Check if sentiment allows trading in the given direction. Returns True."""
        return True
