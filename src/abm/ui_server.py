from __future__ import annotations

from dataclasses import fields, replace
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import numpy as np

from abm.agents import initialise_agents
from abm.mobility import rank_correlation, shorrocks_index, transition_matrix
from abm.parameters import ModelParams
from abm.return_presets import RETURN_PRESETS, apply_return_preset
from abm.scenarios import SCENARIOS, calibrate_scenarios
from abm.simulation import Simulation
from abm.transfer_policies import TRANSFER_POLICIES, apply_transfer_policy


PARAM_FIELDS = {field.name: field for field in fields(ModelParams)}
DEFAULT_SCENARIOS = ["flat", "swedish_baseline_progressivity", "swedish_high_progressivity"]


def run_ui_server(host: str = "127.0.0.1", port: int = 8050) -> None:
    server = ThreadingHTTPServer((host, port), UIRequestHandler)
    print(f"ABM UI running at http://{host}:{port}")
    server.serve_forever()


def run_ui_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    params = _params_from_payload(payload.get("params", {}))
    return_preset_name = payload.get("return_preset", "baseline_return_heterogeneity")
    transfer_policy_name = payload.get("transfer_policy", "universal_plus_safety_floor")
    selected_scenarios = payload.get("scenarios") or DEFAULT_SCENARIOS

    params = apply_return_preset(params, RETURN_PRESETS[return_preset_name])
    params = apply_transfer_policy(params, TRANSFER_POLICIES[transfer_policy_name])
    params = _apply_toggles(params, payload.get("toggles", {}))

    scenario_configs = {name: SCENARIOS[name] for name in selected_scenarios}
    initial_agents = initialise_agents(params)
    calibrated = calibrate_scenarios(params, initial_agents, scenario_configs)

    yearly: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []

    for scenario_name, scenario in calibrated.items():
        sim = Simulation(
            scenario.params,
            tax_system=scenario.config.tax_system,
            initial_agents=initial_agents,
        )
        results = sim.run()
        for record in results.to_dict(orient="records"):
            record["tax_scenario"] = scenario_name
            yearly.append(_json_ready(record))

        matrix = transition_matrix(sim.initial_wealth, sim.agents["wealth"], n_quantiles=5)
        final = results.iloc[-1]
        revenue_difference = scenario.first_year_labour_revenue - scenario.target_labour_revenue

        calibration.append(
            {
                "tax_scenario": scenario_name,
                "tax_system": scenario.config.tax_system,
                "calibrated_rate": scenario.calibrated_rate,
                "target_labour_revenue": scenario.target_labour_revenue,
                "first_year_labour_revenue": scenario.first_year_labour_revenue,
                "first_year_revenue_difference": revenue_difference,
            }
        )
        summary.append(
            {
                "tax_scenario": scenario_name,
                "final_wealth_gini": final["wealth_gini"],
                "final_disposable_income_gini": final["disposable_income_gini"],
                "final_top_10_share": final["top_10_share"],
                "final_top_1_share": final["top_1_share"],
                "capital_income_share": final["capital_income_share"],
                "transfer_spending_share": final["transfer_spending_share"],
                "means_tested_recipient_share": final["means_tested_recipient_share"],
                "unemployment_rate": final["unemployment_rate"],
                "shorrocks_index": shorrocks_index(matrix),
                "rank_correlation": rank_correlation(sim.initial_wealth, sim.agents["wealth"]),
                "top_20_persistence": matrix[4, 4],
            }
        )

    return {
        "settings": {
            "N": params.N,
            "years": params.years,
            "seed": params.seed,
            "return_preset": return_preset_name,
            "transfer_policy": transfer_policy_name,
            "tax_scenarios": selected_scenarios,
        },
        "calibration": _json_ready(calibration),
        "summary": _json_ready(summary),
        "yearly": _json_ready(yearly),
    }


def _params_from_payload(raw_params: dict[str, Any]) -> ModelParams:
    values: dict[str, Any] = {}
    for name, value in raw_params.items():
        if name not in PARAM_FIELDS:
            continue
        default = getattr(ModelParams(), name)
        if isinstance(default, bool):
            values[name] = bool(value)
        elif isinstance(default, int) and not isinstance(default, bool):
            values[name] = int(value)
        elif isinstance(default, float):
            values[name] = float(value)
        elif isinstance(default, str):
            values[name] = str(value)
    return replace(ModelParams(), **values)


def _apply_toggles(params: ModelParams, toggles: dict[str, Any]) -> ModelParams:
    allowed = {
        "homogeneous_returns",
        "homogeneous_saving_rates",
        "no_capital_tax",
        "no_transfers",
    }
    values = {name: bool(toggles.get(name, False)) for name in allowed}
    return replace(params, **values)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value


class UIRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_text(INDEX_HTML, "text/html")
        elif path == "/app.js":
            self._send_text(APP_JS, "application/javascript")
        elif path == "/styles.css":
            self._send_text(STYLES_CSS, "text/css")
        elif path == "/api/options":
            self._send_json(
                {
                    "scenarios": list(SCENARIOS),
                    "return_presets": list(RETURN_PRESETS),
                    "transfer_policies": list(TRANSFER_POLICIES),
                    "defaults": _json_ready(ModelParams().__dict__),
                    "default_scenarios": DEFAULT_SCENARIOS,
                }
            )
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/run":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            result = run_ui_simulation(payload)
        except Exception as exc:  # pragma: no cover - returned to browser
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(result)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ABM Calibration UI</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <header>
    <h1>ABM calibration and analysis</h1>
    <button id="run-button">Run</button>
  </header>
  <main>
    <aside>
      <section>
        <h2>Core inputs</h2>
        <label>N <input id="N" type="number" min="100" step="100"></label>
        <label>Years <input id="years" type="number" min="1" step="1"></label>
        <label>Seed <input id="seed" type="number" step="1"></label>
        <label>Base income <input id="base_income" type="number" step="10000"></label>
        <label>Flat tax rate <input id="flat_tax_rate" type="number" min="0" max="1" step="0.01"></label>
      </section>
      <section>
        <h2>Tax scenarios</h2>
        <div id="scenario-list" class="check-list"></div>
      </section>
      <section>
        <h2>Policy settings</h2>
        <label>Return preset <select id="return_preset"></select></label>
        <label>Transfer policy <select id="transfer_policy"></select></label>
      </section>
      <section>
        <h2>Decomposition toggles</h2>
        <label><input id="homogeneous_returns" type="checkbox"> Homogeneous returns</label>
        <label><input id="homogeneous_saving_rates" type="checkbox"> Homogeneous saving rates</label>
        <label><input id="no_capital_tax" type="checkbox"> No capital tax</label>
        <label><input id="no_transfers" type="checkbox"> No transfers</label>
      </section>
      <section>
        <h2>Transfer inputs</h2>
        <label>Safety floor <input id="safety_floor" type="number" step="10000"></label>
        <label>Universal share <input id="universal_transfer_share" type="number" min="0" max="1" step="0.05"></label>
        <label>Unemployment probability <input id="unemployment_probability" type="number" min="0" max="1" step="0.01"></label>
      </section>
    </aside>
    <section class="workspace">
      <div id="status" class="status">Ready</div>
      <div class="grid">
        <article>
          <h2>Final summary</h2>
          <div id="summary-table"></div>
        </article>
        <article>
          <h2>Calibration</h2>
          <div id="calibration-table"></div>
        </article>
      </div>
      <article>
        <h2>Wealth Gini over time</h2>
        <canvas id="wealth_gini"></canvas>
      </article>
      <article>
        <h2>Disposable-income Gini over time</h2>
        <canvas id="disposable_income_gini"></canvas>
      </article>
      <article>
        <h2>Top 1% wealth share over time</h2>
        <canvas id="top_1_share"></canvas>
      </article>
    </section>
  </main>
  <script src="/app.js"></script>
</body>
</html>
"""


STYLES_CSS = """
:root { color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }
body { margin: 0; background: #f5f7fa; color: #17202a; }
header { height: 64px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; background: #ffffff; border-bottom: 1px solid #d9e0e7; }
h1 { font-size: 20px; margin: 0; font-weight: 650; }
h2 { font-size: 14px; margin: 0 0 12px; }
main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 65px); }
aside { background: #ffffff; border-right: 1px solid #d9e0e7; padding: 16px; overflow: auto; }
section, article { border-bottom: 1px solid #edf1f5; padding: 14px 0; }
article { background: #ffffff; border: 1px solid #d9e0e7; border-radius: 6px; padding: 16px; }
label { display: grid; gap: 5px; margin: 10px 0; font-size: 12px; color: #415064; }
input, select, button { height: 34px; border: 1px solid #c8d2dc; border-radius: 5px; padding: 0 10px; background: #fff; color: #17202a; }
input[type="checkbox"] { height: auto; }
button { background: #1f6feb; color: #fff; border-color: #1f6feb; font-weight: 650; min-width: 90px; cursor: pointer; }
button:disabled { opacity: .6; cursor: default; }
.workspace { padding: 18px; display: grid; gap: 16px; align-content: start; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.status { font-size: 13px; color: #415064; }
.check-list label { grid-template-columns: 18px 1fr; align-items: center; display: grid; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { text-align: right; padding: 7px 8px; border-bottom: 1px solid #edf1f5; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
canvas { width: 100%; height: 280px; display: block; }
@media (max-width: 1000px) {
  main { grid-template-columns: 1fr; }
  aside { border-right: none; border-bottom: 1px solid #d9e0e7; }
  .grid { grid-template-columns: 1fr; }
}
"""


APP_JS = """
let options = null;
let lastResult = null;

const numericInputs = [
  "N", "years", "seed", "base_income", "flat_tax_rate", "safety_floor",
  "universal_transfer_share", "unemployment_probability"
];

async function init() {
  options = await fetch("/api/options").then(r => r.json());
  for (const id of numericInputs) {
    document.getElementById(id).value = options.defaults[id];
  }
  fillSelect("return_preset", options.return_presets, "baseline_return_heterogeneity");
  fillSelect("transfer_policy", options.transfer_policies, "universal_plus_safety_floor");
  const scenarioList = document.getElementById("scenario-list");
  scenarioList.innerHTML = options.scenarios.map(name => `
    <label><input type="checkbox" value="${name}" ${options.default_scenarios.includes(name) ? "checked" : ""}> ${name}</label>
  `).join("");
  document.getElementById("run-button").addEventListener("click", runModel);
  await runModel();
}

function fillSelect(id, values, selected) {
  const el = document.getElementById(id);
  el.innerHTML = values.map(value => `<option value="${value}">${value}</option>`).join("");
  el.value = selected;
}

function payload() {
  const params = {};
  for (const id of numericInputs) {
    params[id] = Number(document.getElementById(id).value);
  }
  return {
    params,
    return_preset: document.getElementById("return_preset").value,
    transfer_policy: document.getElementById("transfer_policy").value,
    scenarios: [...document.querySelectorAll("#scenario-list input:checked")].map(el => el.value),
    toggles: {
      homogeneous_returns: document.getElementById("homogeneous_returns").checked,
      homogeneous_saving_rates: document.getElementById("homogeneous_saving_rates").checked,
      no_capital_tax: document.getElementById("no_capital_tax").checked,
      no_transfers: document.getElementById("no_transfers").checked
    }
  };
}

async function runModel() {
  const button = document.getElementById("run-button");
  const status = document.getElementById("status");
  button.disabled = true;
  status.textContent = "Running simulation...";
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload())
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Run failed");
    lastResult = result;
    render(result);
    status.textContent = `Completed ${result.summary.length} scenario run(s) for N=${result.settings.N}, years=${result.settings.years}`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function render(result) {
  table("summary-table", result.summary, [
    "tax_scenario", "final_wealth_gini", "final_disposable_income_gini",
    "final_top_10_share", "final_top_1_share", "shorrocks_index",
    "top_20_persistence", "transfer_spending_share"
  ]);
  table("calibration-table", result.calibration, [
    "tax_scenario", "calibrated_rate", "target_labour_revenue",
    "first_year_labour_revenue", "first_year_revenue_difference"
  ]);
  lineChart("wealth_gini", result.yearly, "wealth_gini");
  lineChart("disposable_income_gini", result.yearly, "disposable_income_gini");
  lineChart("top_1_share", result.yearly, "top_1_share");
}

function table(id, rows, columns) {
  const format = value => typeof value === "number" ? value.toFixed(Math.abs(value) >= 1000 ? 0 : 4) : value;
  document.getElementById(id).innerHTML = `
    <table>
      <thead><tr>${columns.map(c => `<th>${c}</th>`).join("")}</tr></thead>
      <tbody>${rows.map(row => `<tr>${columns.map(c => `<td>${format(row[c])}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  `;
}

function lineChart(id, rows, metric) {
  const canvas = document.getElementById(id);
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = rect.width * scale;
  canvas.height = rect.height * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const pad = 38;
  const width = rect.width - pad * 2;
  const height = rect.height - pad * 2;
  const scenarios = [...new Set(rows.map(r => r.tax_scenario))];
  const values = rows.map(r => Number(r[metric]));
  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const maxYear = Math.max(...rows.map(r => Number(r.year)));
  const colors = ["#1f6feb", "#d97706", "#059669", "#7c3aed", "#dc2626"];

  ctx.strokeStyle = "#c8d2dc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, pad + height);
  ctx.lineTo(pad + width, pad + height);
  ctx.stroke();
  ctx.fillStyle = "#415064";
  ctx.font = "12px system-ui";
  ctx.fillText(maxY.toFixed(3), 6, pad + 4);
  ctx.fillText(minY.toFixed(3), 6, pad + height);

  scenarios.forEach((scenario, index) => {
    const data = rows.filter(r => r.tax_scenario === scenario);
    ctx.strokeStyle = colors[index % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((row, i) => {
      const x = pad + ((Number(row.year) - 1) / Math.max(1, maxYear - 1)) * width;
      const y = pad + height - ((Number(row[metric]) - minY) / Math.max(0.000001, maxY - minY)) * height;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = colors[index % colors.length];
    ctx.fillText(scenario, pad + 8, pad + 16 + index * 16);
  });
}

window.addEventListener("resize", () => { if (lastResult) render(lastResult); });
init();
"""
