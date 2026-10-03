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

const POLL_MS = 4000;
const MONITORED = [
  "operating_condition", "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
  "sensor_1_rolling_mean", "sensor_3_rolling_mean",
];

// Static snapshot shown when the API is unreachable (e.g. a static deployment of the page).
const PREVIEW_SNAPSHOT = {
  bootstrapped: true,
  aliases: { production: 2 },
  gold_rows: 780,
  metrics: {
    gauges: { sentinel_drift_psi: 0.142, sentinel_model_version: 2 },
    counters: {
      sentinel_batches_total: 3, sentinel_retrains_total: 1,
      sentinel_promotions_total: 1, sentinel_rollbacks_total: 0,
    },
  },
  versions: [
    { version: 1, state: "superseded", metrics: { training_mae: 1.392, holdout_mae: 1.982 } },
    { version: 2, state: "production", metrics: { validation_mae: 2.412, production_mae: 7.903 } },
  ],
  canary_progress: null,
  last_canary_evidence: {
    observations: 300, production_mae: 8.06, canary_mae: 2.61, mean_difference: -5.45,
    ci_lower: -5.92, ci_upper: -4.98, margin: 0.242, reason: "promoted",
  },
  last_drift: {
    detected: false, aggregate_psi: 0.142, threshold: 0.2, noise_floor: 0.245, warning: false,
    feature_psi: {
      operating_condition: 0.142, sensor_1: 0.061, sensor_2: 0.048, sensor_3: 0.052,
      sensor_4: 0.071, sensor_5: 0.055, sensor_1_rolling_mean: 0.039, sensor_3_rolling_mean: 0.044,
    },
    feature_shift: {
      operating_condition: 0.12, sensor_1: 0.05, sensor_2: 0.04, sensor_3: 0.06,
      sensor_4: 0.09, sensor_5: 0.03, sensor_1_rolling_mean: 0.04, sensor_3_rolling_mean: 0.05,
    },
  },
  drift_history: [
    { batch: 1, domain: "FD002", rows: 210, psi: 4.511, alert_line: 0.245, verdict: "detected", top_feature: "operating_condition" },
    { batch: 2, domain: "FD002", rows: 210, psi: 0.188, alert_line: 0.257, verdict: "warning", top_feature: "operating_condition" },
    { batch: 3, domain: "FD002", rows: 210, psi: 0.142, alert_line: 0.245, verdict: "stable", top_feature: "operating_condition" },
  ],
  pipeline_events: [
    { sequence: 1, kind: "bootstrapped", version: 1, batch: 0, domain: "", detail: "trained on 360 rows · train MAE 1.392, engine-holdout MAE 1.982" },
    { sequence: 2, kind: "drift_scored", batch: 1, domain: "FD002", detail: "210 rows · PSI 4.511 vs alert line 0.245 · detected" },
    { sequence: 3, kind: "drift_detected", batch: 1, domain: "FD002", detail: "PSI 4.511 ≥ 0.245; largest shift on operating_condition (29.90σ)" },
    { sequence: 4, kind: "canary_started", version: 2, batch: 1, domain: "FD002", detail: "offline gate passed: validation MAE 2.412 vs production 7.903 on 70 held-out rows" },
    { sequence: 5, kind: "drift_scored", batch: 2, domain: "FD002", detail: "210 rows · PSI 0.188 vs alert line 0.257 · warning" },
    { sequence: 6, kind: "promoted", version: 2, batch: 2, domain: "FD002", detail: "promoted: canary MAE 2.610 vs production 8.060, paired diff CI [-5.920, -4.980] margin 0.242, n=300" },
    { sequence: 7, kind: "drift_scored", batch: 3, domain: "FD002", detail: "210 rows · PSI 0.142 vs alert line 0.245 · stable" },
  ],
};

const EVENT_STYLE = {
  bootstrapped: { tone: "good", icon: "◆", label: "Bootstrapped" },
  restored: { tone: "neutral", icon: "◇", label: "Restored" },
  drift_scored: { tone: "neutral", icon: "·", label: "Batch scored" },
  drift_warning: { tone: "warning", icon: "!", label: "Drift warning" },
  drift_detected: { tone: "warning", icon: "▲", label: "Drift detected" },
  performance_degraded: { tone: "warning", icon: "▼", label: "Performance degraded" },
  retrain_skipped: { tone: "neutral", icon: "–", label: "Retrain skipped" },
  candidate_rejected: { tone: "critical", icon: "✕", label: "Candidate rejected" },
  canary_started: { tone: "good", icon: "▶", label: "Canary started" },
  promoted: { tone: "good", icon: "✓", label: "Promoted" },
  rolled_back: { tone: "critical", icon: "↺", label: "Rolled back" },
};
const VERDICT = {
  detected: { tone: "warning", icon: "▲", label: "Drift detected" },
  warning: { tone: "warning", icon: "!", label: "Warning" },
  stable: { tone: "good", icon: "✓", label: "Stable" },
  concept: { tone: "warning", icon: "▼", label: "Concept drift" },
};

// A batch can have stable inputs yet break the model; the error trigger overrides "stable".
function batchVerdict(entry, trigger) {
  if (!entry) return null;
  if (entry.verdict !== "detected" && trigger && (entry.error_ratio ?? 0) >= trigger) return "concept";
  return entry.verdict;
}

const $ = (selector) => document.querySelector(selector);
const ui = {
  connection: $("#connection"),
  production: $("#production-version"),
  productionState: $("#production-state"),
  canary: $("#canary-version"),
  canaryState: $("#canary-state"),
  canaryMeter: $("#canary-meter"),
  drift: $("#drift-score"),
  driftVerdict: $("#drift-verdict"),
  driftMeter: $("#drift-meter"),
  driftTick: $("#drift-tick"),
  driftDetail: $("#drift-detail"),
  gold: $("#gold-rows"),
  batchCount: $("#batch-count"),
  psiChart: $("#psi-chart"),
  featureBars: $("#feature-bars"),
  driftTable: $("#drift-table"),
  canaryChart: $("#canary-chart"),
  canaryFacts: $("#canary-facts"),
  timeline: $("#timeline"),
  livePill: $("#live-pill"),
  versions: $("#versions"),
  retrains: $("#retrain-count"),
  promotions: $("#promotion-count"),
  rollbacks: $("#rollback-count"),
  driftAction: $("#drift-action"),
  regressionAction: $("#regression-action"),
  regressionHint: $("#regression-hint"),
  resetAction: $("#reset-action"),
  nextStep: $("#next-step"),
  tooltip: $("#tooltip"),
  dataset: $("#dataset"),
  datasetDescription: $("#dataset-description"),
  driftActionLabel: $("#drift-action-label"),
  previewBanner: $("#preview-banner"),
};

let isPreviewMode = false;
let busy = false;
let lastSnapshot = "";
let pollTimer;
let messageTimer;

const fmt = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : "—");
const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const humanize = (name) => String(name).replaceAll("_", " ");

function announce(message, isError = false) {
  if (!ui.nextStep) return;
  clearTimeout(messageTimer);
  ui.nextStep.textContent = message;
  ui.nextStep.className = `next-step${isError ? " error" : " flash"}`;
  messageTimer = setTimeout(() => {
    ui.nextStep.className = "next-step";
    if (lastStatus) ui.nextStep.textContent = nextStepText(lastStatus);
  }, 5000);
}

function setBusy(next, activeButton) {
  busy = next;
  for (const button of [ui.driftAction, ui.resetAction, ui.regressionAction]) {
    if (!button) continue;
    if (next) button.disabled = true;
    button.setAttribute?.("aria-busy", next && button === activeButton ? "true" : "false");
  }
  if (activeButton?.classList) activeButton.classList.toggle("working", next);
  if (!next && lastStatus) updateControls(lastStatus);
}

// ----- signal cards ---------------------------------------------------------------------------

function alertLine(drift) {
  return Math.max(drift?.threshold ?? 0.2, drift?.noise_floor ?? 0);
}

function renderSignals(status) {
  const production = status.aliases?.production;
  const productionRecord = (status.versions || []).find((v) => v.version === production);
  const progress = status.canary_progress;
  const evidence = status.last_canary_evidence;
  const counters = status.metrics?.counters || {};

  ui.production.textContent = production ? `v${production}` : "—";
  const canaryShare = progress ? Math.round(100 * (progress.traffic_weight ?? 0.1)) : 0;
  ui.productionState.textContent = productionRecord
    ? `Serving ${100 - canaryShare}% of traffic · ${metricSummary(productionRecord.metrics)}`
    : "No model deployed";

  if (progress) {
    const share = Math.min(100, (100 * progress.observations) / progress.min_observations);
    ui.canary.textContent = `v${progress.version}`;
    ui.canaryMeter.style.width = `${share}%`;
    ui.canaryState.textContent = `Serving ${canaryShare}% · ${progress.observations} / ${progress.min_observations} shadow observations before a decision`;
  } else {
    ui.canary.textContent = "—";
    ui.canaryMeter.style.width = "0%";
    ui.canaryState.textContent = evidence
      ? `Last decision: ${humanize(evidence.reason)} · canary MAE ${fmt(evidence.canary_mae, 2)} vs ${fmt(evidence.production_mae, 2)}`
      : "No candidate deployed";
  }

  const drift = status.last_drift;
  if (drift) {
    const line = alertLine(drift);
    const lastBatch = status.drift_history?.at?.(-1);
    const verdictKey = batchVerdict(lastBatch, status.performance_trigger)
      ?? (drift.detected ? "detected" : drift.warning ? "warning" : "stable");
    const verdict = VERDICT[verdictKey];
    const scale = Math.max(1, line * 2);
    const offScale = drift.aggregate_psi > scale;
    ui.drift.textContent = fmt(drift.aggregate_psi);
    ui.driftVerdict.className = `verdict ${verdict.tone}`;
    ui.driftVerdict.textContent = `${verdict.icon} ${verdict.label}`;
    ui.driftMeter.style.width = `${Math.min(100, (100 * drift.aggregate_psi) / scale)}%`;
    ui.driftMeter.className = offScale ? "off-scale" : "";
    ui.driftTick.style.left = `${(100 * line) / scale}%`;
    const reason = drift.noise_floor > drift.threshold
      ? `raised from ${fmt(drift.threshold, 2)} by the sampling-noise floor for this batch size`
      : `noise floor ${fmt(drift.noise_floor)} is below it`;
    ui.driftDetail.textContent = verdictKey === "concept"
      ? `Inputs within the ${fmt(line)} alert line, but production error is ${lastBatch.error_ratio.toFixed(1)}× its validation MAE`
      : `Alert line ${fmt(line)}: ${reason}${offScale ? " · bar off scale" : ""}`;
  } else {
    ui.drift.textContent = "—";
    ui.driftVerdict.className = "verdict";
    ui.driftVerdict.textContent = "";
    ui.driftMeter.style.width = "0%";
    ui.driftTick.style.left = "20%";
    ui.driftDetail.textContent = "No batch scored yet";
  }

  ui.gold.textContent = Number(status.gold_rows ?? 0).toLocaleString();
  const batches = Math.trunc(counters.sentinel_batches_total ?? 0);
  ui.batchCount.textContent = `${batches} batch${batches === 1 ? "" : "es"} processed after bootstrap`;
  ui.retrains.textContent = Math.trunc(counters.sentinel_retrains_total ?? 0);
  ui.promotions.textContent = Math.trunc(counters.sentinel_promotions_total ?? 0);
  ui.rollbacks.textContent = Math.trunc(counters.sentinel_rollbacks_total ?? 0);
}

function metricSummary(metrics = {}) {
  if (metrics.validation_mae !== undefined) {
    return `validation MAE ${fmt(metrics.validation_mae, 2)} vs prod ${fmt(metrics.production_mae, 2)}`;
  }
  if (metrics.holdout_mae !== undefined) {
    return `holdout MAE ${fmt(metrics.holdout_mae, 2)} · train ${fmt(metrics.training_mae, 2)}`;
  }
  if (metrics.training_mae !== undefined) return `train MAE ${fmt(metrics.training_mae, 2)}`;
  return "no metrics";
}

// ----- drift evidence -------------------------------------------------------------------------

// Charts are drawn at the container's real pixel width so text never scales below its size.
const chartWidth = (element) => Math.max(260, Math.round(element?.clientWidth || 640));

function renderPsiChart(history, trigger) {
  if (!history?.length) {
    ui.psiChart.innerHTML = '<p class="chart-empty">Inject a batch to start the PSI history.</p>';
    return;
  }
  const width = chartWidth(ui.psiChart);
  const height = 190;
  const pad = { left: 44, right: 16, top: 12, bottom: 26 };
  // Log scale: PSI ranges from ~0.05 (noise) to ~5 (regime change); a linear axis hides one end.
  const values = history.flatMap((d) => [d.psi, d.alert_line]);
  const lo = Math.log10(Math.max(0.01, Math.min(...values) * 0.7));
  const hi = Math.log10(Math.max(...values) * 1.4);
  const n = history.length;
  const x = (i) => pad.left + (n === 1 ? (width - pad.left - pad.right) / 2 : (i * (width - pad.left - pad.right)) / (n - 1));
  const y = (v) => pad.top + ((hi - Math.log10(Math.max(v, 0.01))) * (height - pad.top - pad.bottom)) / (hi - lo);
  let ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].filter((t) => Math.log10(t) >= lo && Math.log10(t) <= hi);
  if (ticks.length > 6) {
    const anchor = Math.max(0, ticks.indexOf(0.2));
    ticks = ticks.filter((t, i) => i % 2 === anchor % 2);
  }
  const line = history.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(d.psi).toFixed(1)}`).join("");
  // The alert line is a step series: each batch's line spans its own slot.
  const half = n === 1 ? (width - pad.left - pad.right) / 2 : (x(1) - x(0)) / 2;
  const alert = history.map((d, i) => {
    const left = Math.max(pad.left, x(i) - half);
    const right = Math.min(width - pad.right, x(i) + half);
    return `${i ? "L" : "M"}${left.toFixed(1)},${y(d.alert_line).toFixed(1)}L${right.toFixed(1)},${y(d.alert_line).toFixed(1)}`;
  }).join("");
  const step = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(width / 70))));
  ui.psiChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">
      ${ticks.map((t) => `<line class="grid" x1="${pad.left}" x2="${width - pad.right}" y1="${y(t)}" y2="${y(t)}"/><text class="axis" x="${pad.left - 8}" y="${y(t) + 3}" text-anchor="end">${t}</text>`).join("")}
      ${history.map((d, i) => (i % step === 0 || i === n - 1 ? `<text class="axis" x="${x(i)}" y="${height - 8}" text-anchor="middle">#${d.batch}</text>` : "")).join("")}
      <path class="alert-line" d="${alert}"/>
      <path class="series" d="${line}"/>
      ${history.map((d, i) => `<circle class="point ${batchVerdict(d, trigger)}" cx="${x(i)}" cy="${y(d.psi)}" r="4.5"/>`).join("")}
      ${history.map((d, i) => `<rect class="hit" data-index="${i}" x="${x(i) - 14}" y="${pad.top}" width="28" height="${height - pad.top - pad.bottom}"/>`).join("")}
    </svg>`;
  ui.psiChart.querySelectorAll?.(".hit").forEach((rect) => {
    const d = history[Number(rect.dataset.index)];
    attachTooltip(rect, () =>
      `<b>Batch #${d.batch} · ${escapeHtml(d.domain)}</b>PSI ${fmt(d.psi)} vs alert line ${fmt(d.alert_line)}<br>${escapeHtml(VERDICT[batchVerdict(d, trigger)]?.label ?? d.verdict)} · top feature ${escapeHtml(humanize(d.top_feature))}<br>${d.rows} rows${Number.isFinite(d.error_ratio) ? ` · production error ${d.error_ratio.toFixed(1)}× validation` : ""}`);
  });
}

function renderFeatureBars(drift) {
  if (!drift?.feature_psi) {
    ui.featureBars.innerHTML = '<p class="chart-empty">Per-feature PSI appears after the first batch.</p>';
    ui.driftTable.innerHTML = "";
    return;
  }
  const line = alertLine(drift);
  const rows = Object.entries(drift.feature_psi).sort((a, b) => b[1] - a[1]);
  const scale = Math.max(line * 1.5, ...rows.map(([, v]) => v));
  ui.featureBars.innerHTML = `
    <div class="bars-scale"><span style="left:${(100 * line) / scale}%">alert ${fmt(line, 2)}</span></div>
    ${rows.map(([name, psi]) => `
      <div class="bar-row${psi >= line ? " over" : ""}" data-name="${escapeHtml(name)}" tabindex="0">
        <span class="bar-label">${escapeHtml(humanize(name))}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${Math.max(1, (100 * psi) / scale)}%"></span><i style="left:${(100 * line) / scale}%"></i></span>
        <span class="bar-value">${fmt(psi, 2)}</span>
        <span class="bar-shift">${fmt(drift.feature_shift?.[name], 2)}σ</span>
      </div>`).join("")}`;
  ui.featureBars.querySelectorAll?.(".bar-row").forEach((row) => {
    const name = row.dataset.name;
    attachTooltip(row, () =>
      `<b>${escapeHtml(humanize(name))}</b>PSI ${fmt(drift.feature_psi[name])} (${drift.feature_psi[name] >= line ? "over" : "under"} the ${fmt(line)} alert line)<br>Mean shift ${fmt(drift.feature_shift?.[name], 2)} reference σ`);
  });
  ui.driftTable.innerHTML = `<table><thead><tr><th>Feature</th><th>PSI</th><th>Shift (σ)</th></tr></thead><tbody>${rows
    .map(([name, psi]) => `<tr><td>${escapeHtml(humanize(name))}</td><td>${fmt(psi)}</td><td>${fmt(drift.feature_shift?.[name], 2)}</td></tr>`)
    .join("")}</tbody></table>`;
}

// ----- canary evidence ------------------------------------------------------------------------

function renderCanary(status) {
  const evidence = status.last_canary_evidence;
  const progress = status.canary_progress;
  const facts = [];
  if (progress) {
    facts.push(["Collecting", `v${progress.version}: ${progress.observations} / ${progress.min_observations} observations (decides by ${progress.max_observations})`]);
  }
  if (!evidence) {
    ui.canaryChart.innerHTML = `<p class="chart-empty">${progress ? "The confidence interval appears once the canary reaches its minimum evidence." : "No canary decision yet. Inject drift to train and deploy a candidate."}</p>`;
    ui.canaryFacts.innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join("");
    return;
  }
  const width = chartWidth(ui.canaryChart);
  const height = 128;
  const pad = 24;
  const points = [evidence.ci_lower, evidence.ci_upper, evidence.mean_difference, evidence.margin, 0];
  let lo = Math.min(...points);
  let hi = Math.max(...points);
  const span = hi - lo || 1;
  lo -= span * 0.12;
  hi += span * 0.12;
  const x = (v) => pad + ((v - lo) * (width - 2 * pad)) / (hi - lo);
  // Labels sit on separate rows and open away from the nearer edge, so none collide or clip.
  const label = (value) => {
    const px = x(value);
    return px > width / 2
      ? { x: Math.min(px, width - 4) - 6, anchor: "end" }
      : { x: Math.max(px, 4) + 6, anchor: "start" };
  };
  const zeroLabel = label(0);
  const marginLabel = label(evidence.margin);
  const meanLabel = label(evidence.mean_difference);
  const verdict = evidence.reason === "promoted" ? "good" : "critical";
  ui.canaryChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">
      <rect class="zone-promote" x="${pad}" y="22" width="${Math.max(0, x(evidence.margin) - pad)}" height="56"/>
      <line class="zero" x1="${x(0)}" x2="${x(0)}" y1="22" y2="86"/>
      <text class="axis" x="${zeroLabel.x}" y="102" text-anchor="${zeroLabel.anchor}">0 (equal)</text>
      <line class="margin" x1="${x(evidence.margin)}" x2="${x(evidence.margin)}" y1="22" y2="86"/>
      <text class="axis" x="${marginLabel.x}" y="118" text-anchor="${marginLabel.anchor}">margin ${fmt(evidence.margin, 2)}</text>
      <text class="axis" x="${meanLabel.x}" y="36" text-anchor="${meanLabel.anchor}">mean d ${fmt(evidence.mean_difference, 2)}</text>
      <line class="ci ${verdict}" x1="${x(evidence.ci_lower)}" x2="${x(evidence.ci_upper)}" y1="50" y2="50"/>
      <line class="ci ${verdict}" x1="${x(evidence.ci_lower)}" x2="${x(evidence.ci_lower)}" y1="42" y2="58"/>
      <line class="ci ${verdict}" x1="${x(evidence.ci_upper)}" x2="${x(evidence.ci_upper)}" y1="42" y2="58"/>
      <circle class="estimate ${verdict}" cx="${x(evidence.mean_difference)}" cy="50" r="6"/>
      <text class="axis strong" x="${pad}" y="12">← canary better</text>
      <text class="axis strong" x="${width - pad}" y="12" text-anchor="end">canary worse →</text>
      <rect class="hit" x="${x(evidence.ci_lower) - 10}" y="30" width="${x(evidence.ci_upper) - x(evidence.ci_lower) + 20}" height="36"/>
    </svg>`;
  ui.canaryChart.querySelector?.(".hit") &&
    attachTooltip(ui.canaryChart.querySelector(".hit"), () =>
      `<b>${escapeHtml(humanize(evidence.reason))}</b>mean d ${fmt(evidence.mean_difference)}<br>95% CI [${fmt(evidence.ci_lower)}, ${fmt(evidence.ci_upper)}]<br>margin ${fmt(evidence.margin)}`);
  if (progress) {
    ui.canaryChart.insertAdjacentHTML?.("afterbegin", '<p class="chart-caption">Previous canary decision, shown while the new canary collects evidence:</p>');
  }
  facts.push(
    [progress ? "Previous decision" : "Decision", humanize(evidence.reason)],
    ["Shadow observations", String(evidence.observations)],
    ["Canary MAE", fmt(evidence.canary_mae)],
    ["Production MAE", fmt(evidence.production_mae)],
    ["Mean d (95% CI)", `${fmt(evidence.mean_difference)} [${fmt(evidence.ci_lower)}, ${fmt(evidence.ci_upper)}]`],
  );
  ui.canaryFacts.innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join("");
}

// ----- timeline and ledger --------------------------------------------------------------------

function renderTimeline(status) {
  const events = status.pipeline_events?.length ? status.pipeline_events : status.events || [];
  if (!events.length) {
    ui.timeline.innerHTML = '<li class="empty">No release events yet.<br>Inject drift to begin.</li>';
    return;
  }
  ui.timeline.innerHTML = [...events].reverse().map((event) => {
    const style = EVENT_STYLE[event.kind] || { tone: "neutral", icon: "·", label: humanize(event.kind) };
    const meta = [
      event.version ? `v${event.version}` : null,
      event.batch ? `batch #${event.batch}` : event.batch === 0 ? "bootstrap" : null,
      event.domain || null,
    ].filter(Boolean).join(" · ");
    return `
      <li class="${style.tone}">
        <time>${String(event.sequence ?? "").padStart(2, "0")}</time>
        <span class="dot" aria-hidden="true">${style.icon}</span>
        <div><b>${escapeHtml(style.label)}</b>${meta ? `<em>${escapeHtml(meta)}</em>` : ""}<p>${escapeHtml(event.detail ?? `Model version ${event.version ?? "—"}`)}</p></div>
      </li>`;
  }).join("");
}

function renderVersions(versions) {
  if (!versions?.length) {
    ui.versions.innerHTML = "";
    return;
  }
  ui.versions.innerHTML = [...versions].reverse().map((model) => `
      <div class="version">
        <span class="version-id">v${model.version}</span>
        <div><b>${model.metrics.validation_mae !== undefined ? "Retrained candidate" : "Bootstrap model"}</b><small>${escapeHtml(metricSummary(model.metrics))}</small></div>
        <span class="state ${model.state}">${escapeHtml(humanize(model.state))}</span>
      </div>`).join("");
}

// ----- controls -------------------------------------------------------------------------------

let lastStatus;

function nextStepText(status) {
  const progress = status.canary_progress;
  if (isPreviewMode) return "Preview data. Run the API locally to drive the loop.";
  if (progress) {
    return `Canary v${progress.version} has ${progress.observations}/${progress.min_observations} shadow observations. Inject another batch to decide it, or corrupt it to watch a rollback.`;
  }
  const last = status.drift_history?.at?.(-1);
  if (!last) return "Step 1: pick a scenario and inject a drifted batch.";
  if (["detected", "concept"].includes(batchVerdict(last, status.performance_trigger))) return "Drift detected but no canary is running (retrain skipped or rejected). Inject again or reset.";
  return "Loop is stable. Inject another scenario, or reset to start over.";
}

function updateControls(status) {
  const hasCanary = Boolean(status.canary_progress ?? status.aliases?.canary);
  ui.regressionAction.disabled = busy || !hasCanary || isPreviewMode;
  ui.driftAction.disabled = busy || isPreviewMode;
  ui.resetAction.disabled = busy || isPreviewMode;
  if (ui.regressionHint) {
    ui.regressionHint.textContent = hasCanary
      ? "Damages the canary in place; live evidence should roll it back."
      : "Available while a canary is collecting shadow evidence. Inject drift first.";
  }
  if (ui.nextStep && !ui.nextStep.className.includes("flash") && !ui.nextStep.className.includes("error")) {
    ui.nextStep.textContent = nextStepText(status);
  }
}

function render(status, isPreview = false) {
  isPreviewMode = isPreview;
  lastStatus = status;
  if (isPreview) {
    ui.connection.className = "connection preview";
    ui.connection.innerHTML = "<span></span> Preview mode";
    if (ui.previewBanner) ui.previewBanner.className = "preview-banner";
    if (ui.livePill) ui.livePill.textContent = "PREVIEW";
  } else {
    ui.connection.className = "connection online";
    ui.connection.innerHTML = "<span></span> Pipeline online";
    if (ui.previewBanner) ui.previewBanner.className = "preview-banner hidden";
    if (ui.livePill) ui.livePill.textContent = `LIVE · ${POLL_MS / 1000}s`;
  }
  renderSignals(status);
  renderPsiChart(status.drift_history || [], status.performance_trigger);
  renderFeatureBars(status.last_drift);
  renderCanary(status);
  renderTimeline(status);
  renderVersions(status.versions || []);
  updateControls(status);
}

// ----- tooltip --------------------------------------------------------------------------------

function attachTooltip(target, content) {
  if (!target?.addEventListener || !ui.tooltip) return;
  const show = (event) => {
    ui.tooltip.innerHTML = content();
    ui.tooltip.hidden = false;
    const rect = target.getBoundingClientRect();
    const x = event?.clientX ?? rect.left + rect.width / 2;
    const y = event?.clientY ?? rect.top;
    const box = ui.tooltip.getBoundingClientRect();
    ui.tooltip.style.left = `${Math.min(window.innerWidth - box.width - 12, Math.max(12, x + 14))}px`;
    ui.tooltip.style.top = `${Math.max(12, y - box.height - 12)}px`;
  };
  const hide = () => { ui.tooltip.hidden = true; };
  target.addEventListener("mousemove", show);
  target.addEventListener("focus", show);
  target.addEventListener("mouseleave", hide);
  target.addEventListener("blur", hide);
}

// ----- data -----------------------------------------------------------------------------------

async function request(path, options = {}, timeoutMs = 2500) {
  let timerId;
  let signal;
  if (typeof AbortController !== "undefined") {
    const controller = new AbortController();
    timerId = setTimeout(() => controller.abort(), timeoutMs);
    signal = controller.signal;
  }
  try {
    const response = await fetch(`${API_BASE}${path}`, { ...options, signal });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail ?? `Request failed (${response.status})`);
    return payload;
  } finally {
    if (timerId) clearTimeout(timerId);
  }
}

let datasetsLoaded = false;
let datasetCatalog = new Map();

function updateDatasetContext() {
  const selected = datasetCatalog.get(ui.dataset?.value);
  if (ui.datasetDescription && selected) ui.datasetDescription.textContent = selected.description;
  if (ui.driftActionLabel) ui.driftActionLabel.textContent = `Inject ${ui.dataset.value} drift`;
}

async function refresh() {
  try {
    const status = await request("/status", {}, 2500);
    const snapshot = JSON.stringify(status);
    if (snapshot !== lastSnapshot || isPreviewMode) {
      lastSnapshot = snapshot;
      render(status, false);
    }
    if (!datasetsLoaded) loadDatasets();
  } catch {
    if (!isPreviewMode) render(PREVIEW_SNAPSHOT, true);
  }
}

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    if (!busy && document.visibilityState !== "hidden") refresh();
  }, POLL_MS);
}

async function loadDatasets() {
  if (datasetsLoaded) return;
  try {
    const data = await request("/datasets", {}, 2000);
    if (data && Array.isArray(data.datasets) && ui.dataset) {
      const currentVal = ui.dataset.value;
      const existingVals = new Set(Array.from(ui.dataset.options).map((o) => o.value));
      const injectables = data.datasets.filter((d) => d.injectable);
      datasetCatalog = new Map(injectables.map((dataset) => [dataset.id, dataset]));
      if (injectables.some((d) => !existingVals.has(d.id))) {
        ui.dataset.innerHTML = injectables
          .map((d) => `<option value="${escapeHtml(d.id)}">${escapeHtml(d.name || d.id)}</option>`)
          .join("");
        if (currentVal && Array.from(ui.dataset.options).some((o) => o.value === currentVal)) {
          ui.dataset.value = currentVal;
        }
      }
      datasetsLoaded = true;
      updateDatasetContext();
    }
  } catch {
    // Keep the built-in options if dataset discovery is unavailable.
  }
}

async function act(path, describe, button) {
  if (isPreviewMode) {
    announce("Preview mode: clone the repo and run `make api` for the live loop.", true);
    return;
  }
  setBusy(true, button);
  try {
    const result = await request(path, { method: "POST" }, 8000);
    lastSnapshot = "";
    await refresh();
    announce(describe(result));
  } catch (error) {
    announce(error.message, true);
  } finally {
    setBusy(false, button);
  }
}

function describeBatch(result) {
  const kinds = (result?.events || []).map((e) => e.kind);
  const psi = fmt(result?.drift?.aggregate_psi);
  if (kinds.includes("promoted")) return `PSI ${psi}. Canary promoted on live evidence.`;
  if (kinds.includes("rolled_back")) return `PSI ${psi}. Canary rolled back; production kept.`;
  if (kinds.includes("canary_started") && kinds.includes("performance_degraded")) return `PSI ${psi} looks stable, but production error jumped (concept drift). Candidate trained and deployed as canary.`;
  if (kinds.includes("canary_started")) return `PSI ${psi}: drift detected, candidate trained and deployed as canary.`;
  if (kinds.includes("candidate_rejected")) return `PSI ${psi}: drift detected, but the candidate failed the offline gate.`;
  if (kinds.includes("drift_detected")) return `PSI ${psi}: drift detected.`;
  if (kinds.includes("performance_degraded")) return `PSI ${psi} looks stable, but production error jumped: concept drift, retraining.`;
  if (kinds.includes("drift_warning")) return `PSI ${psi}: moderate shift logged, no retrain.`;
  return `PSI ${psi}: batch is stable.`;
}

if (ui.dataset) {
  // Stop the mouse wheel from flipping the select while the page scrolls.
  ui.dataset.addEventListener("wheel", (e) => e.preventDefault(), { passive: false });
  ui.dataset.addEventListener("change", updateDatasetContext);
}
ui.driftAction?.addEventListener("click", () => {
  const domain = ui.dataset?.value || "FD002";
  act(`/simulate-drift?domain=${encodeURIComponent(domain)}`, describeBatch, ui.driftAction);
});
ui.regressionAction?.addEventListener("click", () => act("/simulate-regression", describeBatch, ui.regressionAction));
ui.resetAction?.addEventListener("click", () => act("/demo/reset", () => "Demo reset to production v1.", ui.resetAction));

if (typeof location !== "undefined" && location.protocol === "file:") {
  setBusy(true);
  ui.connection.className = "connection";
  ui.connection.textContent = "Open via http://localhost:8000";
  if (ui.nextStep) {
    ui.nextStep.textContent = "This dashboard needs FastAPI. Open http://localhost:8000 instead.";
    ui.nextStep.className = "next-step error";
  }
} else {
  refresh();
  startPolling();
  let resizeTimer;
  window.addEventListener?.("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => lastStatus && render(lastStatus, isPreviewMode), 150);
  });
  document.addEventListener?.("visibilitychange", () => {
    if (document.visibilityState === "visible") refresh();
  });
}
