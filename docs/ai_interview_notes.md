# AI Interview Notes For Risk Analytics Roles

## Transformer

- Transformer is a neural architecture built around attention rather than recurrent sequence processing.
- The interview-safe explanation: token embeddings plus positional information go through stacked attention and feed-forward blocks, with residual connections and normalization for stable training.
- For risk work, the practical link is behavioral sequence modeling: login events, device switches, browsing, payment attempts, address changes, and claim/fraud histories can be treated as event sequences.
- You do not need to claim that Transformer replaces tabular GBDT in every fraud task. A stronger answer is that LightGBM remains a strong tabular baseline, while Transformer-style models help when event order, text, or multimodal signals matter.

## Attention

- Attention lets a model weigh which prior tokens or events matter for the current representation.
- Scaled dot-product attention is commonly written as `softmax(QK^T / sqrt(d_k))V`.
- `Q` is the query, `K` is the key being matched against, and `V` is the information retrieved after weights are computed.
- The `sqrt(d_k)` scaling keeps dot products from becoming too large as dimensionality grows.
- Multi-head attention learns several matching patterns in parallel, such as entity behavior, device behavior, amount spikes, or merchant/category context.
- Masks control what the model is allowed to see. Padding masks ignore empty positions; causal masks prevent looking at future tokens.

## Harness

- In interviews, harness usually means the controlled wrapper used to run, test, and evaluate a model or agent.
- An eval harness for an LLM agent should define cases, expected tool calls, required evidence, scoring rules, logs, and regression checks.
- For this project, the harness checks whether the agent routes questions to the right deterministic risk tool and cites the correct artifact.
- A strong answer: "I do not only evaluate final text quality. I evaluate tool selection, evidence grounding, and unsafe unsupported claims."

## MCP

- Model Context Protocol standardizes how AI applications connect to external tools, resources, and prompts.
- The key concepts to remember are host, client, server, tools, resources, prompts, schemas, permissions, and audit boundaries.
- In a risk setting, MCP could expose `score_transaction`, `explain_case`, `simulate_policy`, and `monitor_drift` as tools while keeping raw sensitive data behind controlled interfaces.
- For this portfolio project, MCP is intentionally left out because the current goal is to show agent reasoning, tool use, and harness evaluation without adding infrastructure noise.

## Agent Answer Pattern

Use this shape when asked about AI agents in risk:

1. Clarify the business task: case investigation, policy thresholding, monitoring, or reporting.
2. Route the task to deterministic tools rather than letting the LLM improvise.
3. Retrieve structured evidence: metrics, SHAP reason codes, policy tables, PSI reports.
4. Generate a concise answer with citations and guardrails.
5. Run an eval harness that checks routing, evidence, and safety behavior.
