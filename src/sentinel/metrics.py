from __future__ import annotations

from collections import defaultdict
from threading import RLock


class Metrics:
    """Dependency-free Prometheus text exporter used by the loop and FastAPI."""

    def __init__(self) -> None:
        self._gauges: dict[str, float] = {}
        self._counters: dict[str, float] = defaultdict(float)
        self._lock = RLock()

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name, value in sorted(self._gauges.items()):
                lines.extend((f"# TYPE {name} gauge", f"{name} {value}"))
            for name, value in sorted(self._counters.items()):
                lines.extend((f"# TYPE {name} counter", f"{name} {value}"))
            return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {"gauges": dict(self._gauges), "counters": dict(self._counters)}
