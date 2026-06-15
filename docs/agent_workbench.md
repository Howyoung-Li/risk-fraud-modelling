# Fraud Risk Agent Workbench

## What This Adds

This workbench upgrades the fraud model pipeline into an evidence-bound risk analyst agent.

The agent does not replace the fraud model. It sits above the model artifacts as an orchestration layer:

```text
user question
-> intent router
-> deterministic risk tool
-> model, policy, SHAP, or monitoring artifact
-> cited answer
-> eval harness
```

## Agent Capabilities

- `model_performance`: reads OOT metrics and Top-K review results.
- `review_capacity`: answers review capacity questions such as Top 1% or Top 3%.
- `policy_simulation`: reads threshold, cost, and approve/manual_review/decline policy outputs.
- `explain_case`: reads SHAP reason codes for high-risk transaction examples.
- `monitor_drift`: reads PSI and score drift monitoring outputs.
- `generate_risk_brief`: composes a concise risk report from the other tools.

Each response returns:

- `intent`
- `tools_called`
- `evidence`
- `trace`
- `answer`

## Local Interactive Demo

Run:

```bash
.venv/bin/python -m scripts.serve_agent_workbench
```

Open:

```text
http://127.0.0.1:4173/
```

Example API call:

```bash
curl 'http://127.0.0.1:4173/api/agent?question=At%20top%203%25%20review%20capacity%2C%20what%20are%20precision%20and%20recall%3F'
```

## Why This Matches Industry Agent Workflows

- It uses tool calling over existing business artifacts instead of free-form model guessing.
- It keeps model scoring, policy simulation, explanation, and monitoring as deterministic systems.
- It cites evidence files so answers are auditable.
- It exposes trace steps for debugging and review.
- It uses an eval harness to test routing, tool selection, and citation discipline.

## Current Boundary

The public GitHub Pages site is static and shows precomputed agent outputs. The local server provides the live interactive agent API.

MCP is intentionally not included in this portfolio version. The same tools could later be exposed through an MCP server as `score_transaction`, `explain_case`, `simulate_policy`, and `monitor_drift`.
