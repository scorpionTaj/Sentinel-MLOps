const API_BASE = (
  (typeof window !== "undefined" &&
    window.location?.search &&
    new URLSearchParams(window.location.search).get("api")) ||
  (typeof window !== "undefined" && window.SENTINEL_API_BASE) ||
  (typeof window !== "undefined" && window.__ENV__?.SENTINEL_API_BASE) ||
  (typeof document !== "undefined" &&
    document.querySelector?.('meta[name="sentinel-api-base"]')?.getAttribute?.("content")) ||
  ""
).replace(/\/$/, "");

const PREVIEW_SNAPSHOT = {
  bootstrapped: true,
  aliases: {
    production: 2,
    canary: null,
  },
  metrics: {
    gauges: {
      sentinel_drift_psi: 0.284,
      sentinel_model_version: 2.0,
      sentinel_training_mae: 1.392,
    },
    counters: {
      sentinel_bootstrap_total: 1,
      sentinel_drift_breaches_total: 2,
      sentinel_retrains_total: 2,
      sentinel_promotions_total: 1,
      sentinel_rollbacks_total: 1,
    },
  },
  gold_rows: 840,
  versions: [
    {
      version: 1,
      state: "superseded",
      metrics: {
        training_mae: 1.392,
      },
    },
    {
      version: 2,
      state: "production",
      metrics: {
        validation_mae: 0.738,
        production_mae: 4.092,
      },
    },
    {
      version: 3,
      state: "rolled_back",
      metrics: {
        validation_mae: 1.845,
        production_mae: 0.738,
      },
    },
  ],
  events: [
    {
      sequence: 1,
      kind: "drift_detected",
      version: 1,
      detail: "PSI threshold breached (0.312)",
    },
    {
      sequence: 2,
      kind: "canary_started",
      version: 2,
      detail: "version 2 passed offline gate",
    },
    {
      sequence: 3,
      kind: "promoted",
      version: 2,
      detail: "canary outperformed production",
    },
    {
      sequence: 4,
      kind: "drift_detected",
      version: 2,
      detail: "PSI threshold breached (0.284)",
    },
    {
      sequence: 5,
      kind: "canary_started",
      version: 3,
      detail: "version 3 passed offline gate",
    },
    {
      sequence: 6,
      kind: "rolled_back",
      version: 3,
      detail: "canary regression detected",
    },
  ],
};

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
  previewBanner: document.querySelector("#preview-banner"),
};

let toastTimer;
let isPreviewMode = false;

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
  if (!events || !events.length) {
    ui.timeline.innerHTML = '<li class="empty">No release events yet.<br>Inject drift to begin.</li>';
    return;
  }
  ui.timeline.innerHTML = [...events].reverse().map((event, index) => `
    <li>
      <time>${String(events.length - index).padStart(2, "0")}</time>
      <span class="dot"></span>
      <div><b>${event.kind.replaceAll("_", " ")}</b><p>Model version ${event.version ?? event.model_version ?? "—"}</p></div>
    </li>`).join("");
}

function renderVersions(versions) {
  if (!versions || !versions.length) {
    ui.versions.innerHTML = "";
    return;
  }
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

function render(status, isPreview = false) {
  isPreviewMode = isPreview;
  const production = status.aliases?.production;
  const canary = status.aliases?.canary;
  const gauges = status.metrics?.gauges || {};
  const counters = status.metrics?.counters || {};
  const psi = gauges.sentinel_drift_psi ?? 0;

  if (isPreview) {
    ui.connection.className = "connection preview";
    ui.connection.innerHTML = "<span></span> Preview mode";
    if (ui.previewBanner) ui.previewBanner.className = "preview-banner";
  } else {
    ui.connection.className = "connection online";
    ui.connection.innerHTML = "<span></span> Pipeline online";
    if (ui.previewBanner) ui.previewBanner.className = "preview-banner hidden";
  }

  ui.production.textContent = production ? `v${production}` : "—";
  ui.productionState.textContent = production ? "Serving 100% baseline traffic" : "No model deployed";
  ui.canary.textContent = canary ? `v${canary}` : "—";
  ui.canaryState.textContent = canary ? "Shadow evaluation active" : "No candidate deployed";
  ui.drift.textContent = formatMetric(psi);
  ui.driftMeter.style.width = `${Math.min(100, (psi / 1.0) * 100)}%`;
  ui.gold.textContent = Number(status.gold_rows ?? 0).toLocaleString();
  ui.regressionAction.disabled = !canary;
  ui.retrains.textContent = Math.trunc(counters.sentinel_retrains_total ?? 0);
  ui.promotions.textContent = Math.trunc(counters.sentinel_promotions_total ?? 0);
  ui.rollbacks.textContent = Math.trunc(counters.sentinel_rollbacks_total ?? 0);
  renderTimeline(status.events || []);
  renderVersions(status.versions || []);
}

async function request(path, options = {}, timeoutMs = 2500) {
  let timerId;
  let signal;
  if (typeof AbortController !== "undefined") {
    const controller = new AbortController();
    timerId = setTimeout(() => controller.abort(), timeoutMs);
    signal = controller.signal;
  }
  try {
    const url = `${API_BASE}${path}`;
    const response = await fetch(url, { ...options, signal });
    if (timerId) clearTimeout(timerId);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail ?? `Request failed (${response.status})`);
    return payload;
  } catch (error) {
    if (timerId) clearTimeout(timerId);
    throw error;
  }
}

let datasetsLoaded = false;

async function refresh() {
  try {
    const status = await request("/status", {}, 2500);
    render(status, false);
    if (!datasetsLoaded) {
      loadDatasets();
    }
  } catch {
    // If live API cannot be reached, fallback to pre-seeded preview snapshot
    render(PREVIEW_SNAPSHOT, true);
  }
}

async function loadDatasets() {
  if (datasetsLoaded) return;
  try {
    const data = await request("/datasets", {}, 2000);
    if (data && Array.isArray(data.datasets) && ui.dataset) {
      const currentVal = ui.dataset.value;
      const existingVals = new Set(Array.from(ui.dataset.options).map((o) => o.value));
      const injectables = data.datasets.filter((d) => d.injectable);
      const isMissingAny = injectables.some((d) => !existingVals.has(d.id));

      if (isMissingAny) {
        const options = injectables
          .map((d) => `<option value="${d.id}">${d.id} · ${d.name || d.id}</option>`)
          .join("");
        ui.dataset.innerHTML = options;
        if (currentVal && Array.from(ui.dataset.options).some((o) => o.value === currentVal)) {
          ui.dataset.value = currentVal;
        }
      }
      datasetsLoaded = true;
    }
  } catch {
    // Keep default options if dataset discovery is unavailable
  }
}

async function act(path, successMessage) {
  setBusy(true);
  try {
    await request(path, { method: "POST" }, 5000);
    await refresh();
    showToast(successMessage);
  } catch (error) {
    if (isPreviewMode) {
      showToast("Preview mode — clone repo & run make demo for live loop", true);
    } else {
      showToast(error.message, true);
    }
  } finally {
    setBusy(false);
  }
}

if (ui.dataset) {
  // Prevent mouse wheel from inadvertently flipping the select value while scrolling/hovering
  ui.dataset.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
    },
    { passive: false },
  );

  ui.dataset.addEventListener("change", () => {
    if (ui.driftAction && ui.driftAction.lastChild) {
      ui.driftAction.lastChild.textContent = ` Inject ${ui.dataset.value} drift`;
    }
  });
}
if (ui.driftAction) {
  ui.driftAction.addEventListener("click", () => {
    const domain = (ui.dataset && ui.dataset.value) || "FD002";
    act(
      `/simulate-drift?domain=${encodeURIComponent(domain)}`,
      `${domain} batch processed. Release evidence updated.`,
    );
  });
}
if (ui.regressionAction) {
  ui.regressionAction.addEventListener("click", () =>
    act("/simulate-regression", "Regression caught. Production alias protected."),
  );
}
if (ui.resetAction) {
  ui.resetAction.addEventListener("click", () => act("/demo/reset", "Demo reset to production v1."));
}

if (typeof location !== "undefined" && location.protocol === "file:") {
  setBusy(true);
  if (ui.resetAction) ui.resetAction.disabled = true;
  ui.connection.className = "connection";
  ui.connection.textContent = "Open via http://localhost:8000";
  showToast("This dashboard needs FastAPI. Open http://localhost:8000 instead.", true);
} else {
  refresh();
}
