const ui = {
  connection: document.querySelector("#connection"),
  production: document.querySelector("#production-version"),
  productionState: document.querySelector("#production-state"),
  canary: document.querySelector("#canary-version"),
  canaryState: document.querySelector("#canary-state"),
  drift: document.querySelector("#drift-score"),
  driftMeter: document.querySelector("#drift-meter"),
  gold: document.querySelector("#gold-rows"),
  timeline: document.querySelector("#timeline"),
  versions: document.querySelector("#versions"),
  retrains: document.querySelector("#retrain-count"),
  promotions: document.querySelector("#promotion-count"),
  rollbacks: document.querySelector("#rollback-count"),
  driftAction: document.querySelector("#drift-action"),
  regressionAction: document.querySelector("#regression-action"),
  resetAction: document.querySelector("#reset-action"),
  toast: document.querySelector("#toast"),
  dataset: document.querySelector("#dataset"),
};

let toastTimer;

function showToast(message, isError = false) {
  clearTimeout(toastTimer);
  ui.toast.textContent = message;
  ui.toast.className = `toast visible${isError ? " error" : ""}`;
  toastTimer = setTimeout(() => (ui.toast.className = "toast"), 3200);
}

function setBusy(busy) {
  ui.driftAction.disabled = busy;
  ui.resetAction.disabled = busy;
  if (busy) ui.regressionAction.disabled = true;
}

function formatMetric(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "0.000";
}

function renderTimeline(events) {
  if (!events.length) {
    ui.timeline.innerHTML = '<li class="empty">No release events yet.<br>Inject drift to begin.</li>';
    return;
  }
  ui.timeline.innerHTML = [...events].reverse().map((event, index) => `
    <li>
      <time>${String(events.length - index).padStart(2, "0")}</time>
      <span class="dot"></span>
      <div><b>${event.kind.replaceAll("_", " ")}</b><p>Model version ${event.version}</p></div>
    </li>`).join("");
}

function renderVersions(versions) {
  ui.versions.innerHTML = [...versions].reverse().map((model) => {
    const metric = model.metrics.validation_mae ?? model.metrics.training_mae;
    return `
      <div class="version">
        <span class="version-id">v${model.version}</span>
        <div><b>RUL predictor</b><small>MAE ${formatMetric(metric)}</small></div>
        <span class="state ${model.state}">${model.state.replaceAll("_", " ")}</span>
      </div>`;
  }).join("");
}

function render(status) {
  const production = status.aliases.production;
  const canary = status.aliases.canary;
  const gauges = status.metrics.gauges;
  const counters = status.metrics.counters;
  const psi = gauges.sentinel_drift_psi ?? 0;

  ui.connection.className = "connection online";
  ui.connection.innerHTML = "<span></span> Pipeline online";
  ui.production.textContent = production ? `v${production}` : "—";
  ui.productionState.textContent = production ? "Serving 100% baseline traffic" : "No model deployed";
  ui.canary.textContent = canary ? `v${canary}` : "—";
  ui.canaryState.textContent = canary ? "Shadow evaluation active" : "No candidate deployed";
  ui.drift.textContent = formatMetric(psi);
  ui.driftMeter.style.width = `${Math.min(100, psi / 1.0 * 100)}%`;
  ui.gold.textContent = Number(status.gold_rows).toLocaleString();
  ui.regressionAction.disabled = !canary;
  ui.retrains.textContent = Math.trunc(counters.sentinel_retrains_total ?? 0);
  ui.promotions.textContent = Math.trunc(counters.sentinel_promotions_total ?? 0);
  ui.rollbacks.textContent = Math.trunc(counters.sentinel_rollbacks_total ?? 0);
  renderTimeline(status.events);
  renderVersions(status.versions);
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? `Request failed (${response.status})`);
  return payload;
}

async function refresh() {
  try {
    render(await request("/status"));
  } catch (error) {
    ui.connection.className = "connection";
    ui.connection.innerHTML = "<span></span> Pipeline offline";
    showToast(error.message, true);
  }
}

async function act(path, successMessage) {
  setBusy(true);
  try {
    await request(path, { method: "POST" });
    await refresh();
    showToast(successMessage);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false);
  }
}

ui.dataset.addEventListener("change", () => {
  ui.driftAction.lastChild.textContent = ` Inject ${ui.dataset.value} drift`;
});
ui.driftAction.addEventListener("click", () => {
  const domain = ui.dataset.value || "FD002";
  act(`/simulate-drift?domain=${encodeURIComponent(domain)}`, `${domain} batch processed. Release evidence updated.`);
});
ui.regressionAction.addEventListener("click", () => act("/simulate-regression", "Regression caught. Production alias protected."));
ui.resetAction.addEventListener("click", () => act("/demo/reset", "Demo reset to production v1."));

if (location.protocol === "file:") {
  setBusy(true);
  ui.resetAction.disabled = true;
  ui.connection.className = "connection";
  ui.connection.textContent = "Open via http://localhost:8000";
  showToast("This dashboard needs FastAPI. Open http://localhost:8000 instead.", true);
} else {
  refresh();
}
