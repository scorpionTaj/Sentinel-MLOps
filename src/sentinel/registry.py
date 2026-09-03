from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock

from sentinel.model import RidgeModel


class FileModelRegistry:
    """Atomically persists models, aliases, metrics, and release events."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        if not self.path.exists():
            self._write({"next_version": 1, "aliases": {}, "versions": [], "events": []})

    def register(self, model: RidgeModel, metrics: dict[str, float], state: str) -> int:
        with self._lock:
            data = self._read()
            version = int(data["next_version"])
            data["next_version"] = version + 1
            data["versions"].append(
                {"version": version, "state": state, "metrics": metrics, "model": model.to_dict()}
            )
            self._event(data, "registered", version)
            self._write(data)
            return version

    def deploy_initial(self, version: int) -> None:
        with self._lock:
            data = self._read()
            data["aliases"]["production"] = version
            self._set_state(data, version, "production")
            self._event(data, "initial_deploy", version)
            self._write(data)

    def start_canary(self, version: int) -> None:
        with self._lock:
            data = self._read()
            data["aliases"]["canary"] = version
            self._set_state(data, version, "canary")
            self._event(data, "canary_started", version)
            self._write(data)

    def promote(self) -> int:
        with self._lock:
            data = self._read()
            version = int(data["aliases"]["canary"])
            previous = data["aliases"].get("production")
            if previous is not None:
                self._set_state(data, int(previous), "superseded")
            data["aliases"]["production"] = version
            data["aliases"].pop("canary", None)
            self._set_state(data, version, "production")
            self._event(data, "promoted", version)
            self._write(data)
            return version

    def rollback(self) -> int:
        with self._lock:
            data = self._read()
            version = int(data["aliases"].pop("canary"))
            self._set_state(data, version, "rolled_back")
            self._event(data, "rolled_back", version)
            self._write(data)
            return version

    def reject(self, version: int) -> None:
        with self._lock:
            data = self._read()
            self._set_state(data, version, "rejected")
            self._event(data, "rejected", version)
            self._write(data)

    def replace_model(self, alias: str, model: RidgeModel, event_kind: str) -> int:
        """Fault-injection seam used to verify release safety in demos and tests."""
        with self._lock:
            data = self._read()
            version = data["aliases"].get(alias)
            if version is None:
                raise LookupError(f"alias {alias!r} is not assigned")
            record = next(item for item in data["versions"] if item["version"] == version)
            record["model"] = model.to_dict()
            self._event(data, event_kind, int(version))
            self._write(data)
            return int(version)

    def version(self, alias: str) -> int | None:
        value = self._read()["aliases"].get(alias)
        return int(value) if value is not None else None

    def model(self, alias: str) -> RidgeModel:
        data = self._read()
        version = data["aliases"].get(alias)
        if version is None:
            raise LookupError(f"alias {alias!r} is not assigned")
        record = next(item for item in data["versions"] if item["version"] == version)
        return RidgeModel.from_dict(record["model"])

    def snapshot(self) -> dict[str, object]:
        return self._read()

    def reset(self) -> None:
        """Clear only this registry's demo state."""
        with self._lock:
            self._write({"next_version": 1, "aliases": {}, "versions": [], "events": []})

    @staticmethod
    def _set_state(data: dict[str, object], version: int, state: str) -> None:
        versions = data["versions"]
        for record in versions:  # type: ignore[union-attr]
            if record["version"] == version:
                record["state"] = state
                return
        raise LookupError(f"model version {version} does not exist")

    @staticmethod
    def _event(data: dict[str, object], kind: str, version: int) -> None:
        data["events"].append({"kind": kind, "version": version})  # type: ignore[union-attr]

    def _read(self) -> dict[str, object]:
        with self._lock:
            return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, object]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)
