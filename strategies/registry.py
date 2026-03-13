"""Strategy registry — stores active strategy, supports hot-swap and shadow mode."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional, Type

from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal


class StrategyRegistry:
    """Manages strategy lifecycle: registration, activation, hot-swap, and shadow logging."""

    def __init__(self):
        self._registered: Dict[str, Type[BaseStrategy]] = {}
        self._active: Dict[str, BaseStrategy] = {}  # symbol -> active strategy instance
        self._shadow: Dict[str, List[BaseStrategy]] = {}  # symbol -> shadow strategies
        self._default_strategy_name: Optional[str] = None

    def register(self, strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class by its name."""
        self._registered[strategy_class.name] = strategy_class
        logger.info("Strategy registered: {}", strategy_class.name)

    def register_all(self, strategies: List[Type[BaseStrategy]]) -> None:
        for s in strategies:
            self.register(s)

    def set_active(
        self, symbol: str, strategy_name: str, params: Optional[Dict] = None
    ) -> BaseStrategy:
        """Activate a strategy for a symbol. Hot-swaps if one is already active."""
        if strategy_name not in self._registered:
            raise ValueError(f"Strategy '{strategy_name}' not registered")

        cls = self._registered[strategy_name]
        instance = cls(params=params)

        old = self._active.get(symbol)
        if old:
            logger.info(
                "Hot-swapping {} -> {} for {}",
                old.name, strategy_name, symbol,
            )
        self._active[symbol] = instance
        return instance

    def get_active(self, symbol: str) -> Optional[BaseStrategy]:
        return self._active.get(symbol)

    def get_all_active(self) -> Dict[str, BaseStrategy]:
        return dict(self._active)

    def set_default(self, strategy_name: str) -> None:
        """Set the default strategy to use when no optimizer result is available."""
        if strategy_name not in self._registered:
            raise ValueError(f"Strategy '{strategy_name}' not registered")
        self._default_strategy_name = strategy_name
        logger.info("Default strategy set: {}", strategy_name)

    def ensure_active(self, symbol: str) -> BaseStrategy:
        """Ensure a strategy is active for a symbol, using default if needed."""
        active = self.get_active(symbol)
        if active:
            return active
        if self._default_strategy_name:
            return self.set_active(symbol, self._default_strategy_name)
        raise RuntimeError(f"No active or default strategy for {symbol}")

    # --- Shadow Mode ---

    def add_shadow(self, symbol: str, strategy_name: str, params: Optional[Dict] = None) -> None:
        """Add a candidate strategy in shadow mode for comparison."""
        if strategy_name not in self._registered:
            raise ValueError(f"Strategy '{strategy_name}' not registered")
        cls = self._registered[strategy_name]
        instance = cls(params=params)
        self._shadow.setdefault(symbol, []).append(instance)
        logger.info("Shadow strategy added: {} for {}", strategy_name, symbol)

    def get_shadows(self, symbol: str) -> List[BaseStrategy]:
        return self._shadow.get(symbol, [])

    def clear_shadows(self, symbol: str) -> None:
        self._shadow.pop(symbol, None)

    # --- Info ---

    @property
    def registered_names(self) -> List[str]:
        return list(self._registered.keys())

    def get_strategy_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        return self._registered.get(name)

    def status_report(self) -> str:
        lines = ["=== Strategy Registry ==="]
        lines.append(f"Registered: {', '.join(self.registered_names)}")
        for sym, strat in self._active.items():
            shadows = [s.name for s in self.get_shadows(sym)]
            line = f"  {sym}: {strat.name} (active)"
            if shadows:
                line += f" | shadows: {', '.join(shadows)}"
            lines.append(line)
        return "\n".join(lines)
