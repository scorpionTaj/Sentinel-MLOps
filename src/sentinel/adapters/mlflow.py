from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import RLock
from typing import Any

from sentinel.model import RidgeModel
from sentinel.registry import FileModelRegistry

logger = logging.getLogger("sentinel.adapters.mlflow")


class MLflowModelRegistry:
    """MLflow Model Registry adapter.

    Implements ModelRegistryProtocol. When MLflow is installed and configured,
    it synchronizes registered model versions, tags, metrics, and aliases
    (production, canary) with MLflow's tracking and registry backend.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "sentinel_turbofan_rul",
        model_name: str = "sentinel_ridge_rul",
        local_store_path: Path | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlflow")
        self.experiment_name = experiment_name
        self.model_name = model_name
        self._lock = RLock()

        # Local backing store ensures atomic fallback and local persistence
        store_path = local_store_path or Path("/tmp/sentinel_mlflow_local.json")
        self._local_registry = FileModelRegistry(store_path)
        self._mlflow_available = False

        try:
            import mlflow  # noqa: F401

            self._mlflow_available = True
        except ImportError:
            logger.info("MLflow library not installed. Running in local-emulated registry mode.")

    @property
    def is_mlflow_connected(self) -> bool:
        return self._mlflow_available

    def register(self, model: RidgeModel, metrics: dict[str, float], state: str) -> int:
        with self._lock:
            version = self._local_registry.register(model, metrics, state)
            if self._mlflow_available:
                self._sync_mlflow_register(version, model, metrics, state)
            return version

    def deploy_initial(self, version: int) -> None:
        with self._lock:
            self._local_registry.deploy_initial(version)
            if self._mlflow_available:
                self._sync_mlflow_alias("production", version)

    def start_canary(self, version: int) -> None:
        with self._lock:
            self._local_registry.start_canary(version)
            if self._mlflow_available:
                self._sync_mlflow_alias("canary", version)

    def promote(self) -> int:
        with self._lock:
            promoted_ver = self._local_registry.promote()
            if self._mlflow_available:
                self._sync_mlflow_alias("production", promoted_ver)
                self._sync_mlflow_alias("canary", None)
            return promoted_ver

    def rollback(self) -> int:
        with self._lock:
            active_prod = self._local_registry.rollback()
            if self._mlflow_available:
                self._sync_mlflow_alias("canary", None)
            return active_prod

    def model(self, alias: str) -> RidgeModel:
        return self._local_registry.model(alias)

    def version(self, alias: str) -> int | None:
        return self._local_registry.version(alias)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snap = self._local_registry.snapshot()
            snap["adapter"] = "mlflow"
            snap["mlflow_tracking_uri"] = self.tracking_uri
            snap["mlflow_model_name"] = self.model_name
            snap["mlflow_connected"] = self._mlflow_available
            return snap

    def reset(self) -> None:
        with self._lock:
            self._local_registry.reset()

    def _sync_mlflow_register(
        self, version: int, model: RidgeModel, metrics: dict[str, float], state: str
    ) -> None:
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=f"v{version}-{state}"):
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))
                mlflow.set_tags(
                    {
                        "sentinel.version": str(version),
                        "sentinel.state": state,
                        "sentinel.features": json.dumps(model.feature_names),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to synchronize model v{version} to MLflow: {exc}")

    def _sync_mlflow_alias(self, alias: str, version: int | None) -> None:
        try:
            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            if version is not None:
                client.set_registered_model_alias(self.model_name, alias, str(version))
            else:
                try:
                    client.delete_registered_model_alias(self.model_name, alias)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"Alias {alias} delete ignored: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to set alias {alias} -> v{version} in MLflow: {exc}")
