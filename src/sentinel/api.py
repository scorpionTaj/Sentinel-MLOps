from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

# Ensure src/ is in sys.path when executed in serverless environments (e.g. Vercel)
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sentinel.features import FEATURE_NAMES
from sentinel.loop import HealingConfig, HealingLoop
from sentinel.scenarios import all_scenarios, drift_scenario_ids, get_scenario
from sentinel.synthetic import generate_telemetry

DEMO_ENGINES = 6


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=len(FEATURE_NAMES), max_length=len(FEATURE_NAMES))
    request_id: str | None = None


class PredictionResponse(BaseModel):
    prediction: float
    model_version: int
    route: str


def _resolve_runtime_dir(state_dir: Path | None = None) -> Path:
    if state_dir is not None:
        return state_dir
    if "SENTINEL_STATE_DIR" in os.environ:
        return Path(os.environ["SENTINEL_STATE_DIR"])
    # Serverless runtimes (e.g. Vercel, AWS Lambda) have read-only filesystems; only /tmp is writable
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        tmp_dir = Path(tempfile.gettempdir()) / "sentinel" / "runtime"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir
    local_dir = Path("data/runtime")
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        test_file = local_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        return local_dir
    except (OSError, PermissionError):
        tmp_dir = Path(tempfile.gettempdir()) / "sentinel" / "runtime"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir


def create_app(state_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Sentinel Self-Healing MLOps", version="0.1.0")
    runtime_dir = _resolve_runtime_dir(state_dir)
    # Demo batches hold DEMO_ENGINES * 35 = 210 rows, so a 300-observation canary spans two
    # batches: the first click starts it, the second (or /simulate-regression) decides it.
    config = HealingConfig(
        canary_min_observations=int(os.getenv("SENTINEL_CANARY_OBSERVATIONS", "300"))
    )
    web_dir = Path(__file__).resolve().parent / "web"
    if not web_dir.exists():
        candidate = Path("src/sentinel/web")
        if candidate.exists():
            web_dir = candidate.resolve()

    def new_loop() -> HealingLoop:
        created = HealingLoop(runtime_dir, config)
        created.bootstrap(generate_telemetry("FD001", engines=8, cycles=45, seed=10))
        return created

    def loop() -> HealingLoop:
        return app.state.healing_loop

    app.state.healing_loop = new_loop()
    if web_dir.exists():
        app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def dashboard() -> Response:
        index_file = web_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return Response("Sentinel Self-Healing MLOps API Online", media_type="text/plain")

    @app.get("/styles.css", include_in_schema=False)
    def dashboard_styles() -> Response:
        styles_file = web_dir / "styles.css"
        if styles_file.exists():
            return FileResponse(styles_file)
        raise HTTPException(status_code=404, detail="styles.css not found")

    @app.get("/app.js", include_in_schema=False)
    def dashboard_script() -> Response:
        script_file = web_dir / "app.js"
        if script_file.exists():
            return FileResponse(script_file)
        raise HTTPException(status_code=404, detail="app.js not found")

    @app.get("/logo.svg", include_in_schema=False)
    def dashboard_logo() -> Response:
        logo_file = web_dir / "logo.svg"
        if logo_file.exists():
            return FileResponse(logo_file, media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="logo.svg not found")

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    def dashboard_favicon() -> Response:
        favicon_file = web_dir / "favicon.svg"
        if favicon_file.exists():
            return FileResponse(favicon_file, media_type="image/svg+xml")
        logo_file = web_dir / "logo.svg"
        if logo_file.exists():
            return FileResponse(logo_file, media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="favicon not found")

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
        # The demo stays on one data source (the synthetic generator the loop was bootstrapped
        # on). Real C-MAPSS rows have different sensor units, so mixing them in would compare a
        # synthetic-scale model against real-scale targets. Real data is evaluated offline.
        scenario = get_scenario(domain.upper())
        return loop().process(_demo_batch(scenario.id, scenario.seed)).to_dict()

    @app.post("/simulate-regression")
    def simulate_regression() -> dict[str, object]:
        if loop().registry.version("canary") is None:
            raise HTTPException(status_code=409, detail="start a canary with /simulate-drift first")
        loop().inject_canary_regression()
        return loop().process(_demo_batch("FD002", 100)).to_dict()

    def _demo_batch(domain: str, seed: int) -> list:
        # A fleet cross-section of DEMO_ENGINES whole trajectories. The seed advances with every
        # batch so repeated clicks sample new engines instead of replaying identical rows.
        batch = int(loop().status()["metrics"]["counters"].get("sentinel_batches_total", 0))
        offset = 500 + int(loop().status()["gold_rows"])
        return generate_telemetry(
            domain, engines=DEMO_ENGINES, cycles=35, seed=seed + 1000 * batch, engine_offset=offset
        )

    @app.post("/demo/reset")
    def reset_demo() -> dict[str, object]:
        loop().registry.reset()
        app.state.healing_loop = new_loop()
        return loop().status()

    return app


app = create_app()
