from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sentinel.features import FEATURE_NAMES
from sentinel.loop import HealingConfig, HealingLoop
from sentinel.scenarios import all_scenarios, drift_scenario_ids
from sentinel.synthetic import generate_telemetry


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=len(FEATURE_NAMES), max_length=len(FEATURE_NAMES))
    request_id: str | None = None


class PredictionResponse(BaseModel):
    prediction: float
    model_version: int
    route: str


def create_app(state_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Sentinel Self-Healing MLOps", version="0.1.0")
    runtime_dir = state_dir or Path(os.getenv("SENTINEL_STATE_DIR", "data/runtime"))
    config = HealingConfig(
        canary_min_observations=int(os.getenv("SENTINEL_CANARY_OBSERVATIONS", "150"))
    )
    web_dir = Path(__file__).with_name("web")

    def new_loop() -> HealingLoop:
        created = HealingLoop(runtime_dir, config)
        created.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
        return created

    def loop() -> HealingLoop:
        return app.state.healing_loop

    app.state.healing_loop = new_loop()
    app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/styles.css", include_in_schema=False)
    def dashboard_styles() -> FileResponse:
        return FileResponse(web_dir / "styles.css")

    @app.get("/app.js", include_in_schema=False)
    def dashboard_script() -> FileResponse:
        return FileResponse(web_dir / "app.js")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict[str, object]:
        return loop().status()

    @app.get("/datasets")
    def datasets() -> dict[str, object]:
        return {
            "baseline": "FD001",
            "datasets": [scenario.to_dict() for scenario in all_scenarios()],
        }

    @app.get("/metrics", response_class=Response)
    def metrics() -> Response:
        return Response(loop().metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        try:
            result = loop().predict(tuple(request.features), request.request_id or str(uuid4()))
        except (ValueError, LookupError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return PredictionResponse(
            prediction=result.value, model_version=result.model_version, route=result.route
        )

    drift_pattern = f"^({'|'.join(drift_scenario_ids())})$"

    @app.post("/simulate-drift")
    def simulate_drift(
        domain: str = Query(default="FD002", pattern=drift_pattern)
    ) -> dict[str, object]:
        seeds = {
            "FD002": 99,
            "FD003": 199,
            "FD004": 299,
            "SENSOR_BIAS": 399,
            "SENSOR_DROPOUT": 499,
        }
        seed = seeds.get(domain, 99)
        offset = 500 + int(loop().status()["gold_rows"])
        rows = generate_telemetry(domain, engines=3, cycles=35, seed=seed, engine_offset=offset)
        return loop().process(rows).to_dict()

    @app.post("/simulate-regression")
    def simulate_regression() -> dict[str, object]:
        if loop().registry.version("canary") is None:
            raise HTTPException(status_code=409, detail="start a canary with /simulate-drift first")
        loop().inject_canary_regression()
        offset = 500 + int(loop().status()["gold_rows"])
        rows = generate_telemetry("FD002", engines=3, cycles=35, seed=100, engine_offset=offset)
        return loop().process(rows).to_dict()

    @app.post("/demo/reset")
    def reset_demo() -> dict[str, object]:
        loop().registry.reset()
        app.state.healing_loop = new_loop()
        return loop().status()

    return app


app = create_app()
