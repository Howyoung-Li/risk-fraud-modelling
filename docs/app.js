const state = {
  charts: [
    ["pr_curve", "PR Curve"],
    ["topk_capture", "Top-K Capture"],
    ["policy_cost", "Policy Cost"],
    ["score_drift", "Score Drift"],
    ["shap_summary", "SHAP Feature Effects"]
  ]
};

function evidenceTags(items) {
  return `<div class="tags">${items.map(item => `<span class="tag">${item}</span>`).join("")}</div>`;
}

function renderStatus(data) {
  const panel = document.querySelector("#status-panel");
  const passRate = Math.round((data.eval.pass_rate || 0) * 100);
  panel.innerHTML = `
    <strong>${passRate}%</strong>
    <span>eval harness pass rate across ${data.eval.total_cases} core routing and evidence checks</span>
  `;
}

function renderAgentCards(data) {
  const grid = document.querySelector("#agent");
  grid.innerHTML = data.agent_responses.map(response => `
    <article class="metric-card">
      <p class="eyebrow">${response.intent.replaceAll("_", " ")}</p>
      <h3>${response.question}</h3>
      <p>${response.answer.split("Evidence:")[0]}</p>
      ${evidenceTags(response.evidence)}
    </article>
  `).join("");
}

function renderFlow(data) {
  const flow = document.querySelector("#flow");
  flow.innerHTML = data.architecture.map((step, index) => `
    <div class="flow-step">${index + 1}. ${step}</div>
  `).join("");
}

function renderCharts(data) {
  const charts = document.querySelector("#charts");
  charts.innerHTML = state.charts
    .filter(([key]) => data.assets[key])
    .map(([key, title]) => `
      <article class="chart-card">
        <h3>${title}</h3>
        <img src="${data.assets[key]}" alt="${title}">
      </article>
    `).join("");
}

function renderEval(data) {
  const body = document.querySelector("#eval-table tbody");
  body.innerHTML = data.eval.cases.map(row => `
    <tr>
      <td>${row.name}</td>
      <td>${row.expected_intent}</td>
      <td>${row.actual_intent}</td>
      <td>${row.required_evidence}</td>
      <td class="${row.passed ? "ok" : "fail"}">${row.passed ? "Pass" : "Review"}</td>
    </tr>
  `).join("");
}

function loadJson(url) {
  if (typeof fetch === "function") {
    return fetch(url).then(response => response.json());
  }
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(JSON.parse(request.responseText));
      } else {
        reject(new Error(`HTTP ${request.status}`));
      }
    };
    request.onerror = () => reject(new Error("Network request failed"));
    request.send();
  });
}

async function boot() {
  const data = await loadJson("data/workbench.json");
  renderStatus(data);
  renderAgentCards(data);
  renderFlow(data);
  renderCharts(data);
  renderEval(data);
}

boot().catch(error => {
  document.querySelector("#status-panel").innerHTML = `<span>${error.message}</span>`;
});
