# Hosted agent pilot — planned, disabled

Do not execute this pilot during benchmark preflight. The proposed cap is **$10
total**, including retries and failed requests, for a later explicitly resumed
pilot. The benchmark tools currently contain no API client or paid execution path.

Prerequisites: certified tasks, tested scorer/negative controls, complete visual
evidence, working sustained shared-agent loop, offline provider contract tests,
and a frozen local baseline.

## Provider integration before spending

The existing `src/emet/app/run_agent.py` already accepts `openai` and
`openai@http://host:port/v1[#model]`. Its client in
`src/emet/llms/openai_client.py` accepts a model ID, configurable endpoint and
credentials, images, and native tool schemas. Extend that shared client and loop
with the agent owner; the benchmark should not introduce a second policy loop.

Before connecting a paid endpoint, test the following using recorded responses
or a fake transport:

- Preserve provider tool-call IDs and the complete assistant/tool continuation
  across multiple actions, including malformed arguments and tool failures.
- Pass explicit generation limits and request timeouts; expose usage, finish
  reason, latency and model identity to the recorder. The current completion
  call does not pass these limits or return usage to the benchmark.
- Keep request retries separate from robot execution and give each action one
  durable command ID. Test timeout, rate-limit, cancellation and budget exhaustion.
- Validate image ordering against observation IDs; record exactly which image
  and memory evidence the policy received. Keep evaluator goals and coordinates
  private.
- Make paid execution an explicit configuration mode with a required nonzero
  budget. Fail before dispatch if pricing or a conservative cost bound is missing.

Select one image/tool-capable hosted model when this gate opens, and verify its
current API capabilities and prices then. No model ranking or dated price estimate
is needed to validate the benchmark today.

## Proposed tiny pilot

| Stage | Maximum allocation | Work |
| --- | ---: | --- |
| API smoke | $1 | Image input, native tool call and result continuation |
| Recorded decisions | $3 | Six fixed search, ambiguity, planning and stale-memory decisions |
| Simulator pilot | $6 | Two episodes, at most $3 each: delivery and moved-object recovery |

Use one hosted policy model; keep perception, memory, scene, seed, skills and
assistance fixed against the local baseline. Record exact model/version, pricing
snapshot, reasoning/output limits, images, tokens, latency and costs.

Reserve conservative maximum request cost before dispatch, including image and
reasoning tokens. Serialize calls, count every retry, retain unknown timeout
charges as reservations, and stop before either stage or total cap could be
exceeded. Budget exhaustion is an incomplete result. Never replay robot actions
to retry an API request; preserve command IDs and results.

After reviewing the tiny pilot, document a separate larger comparison. Two
episodes provide diagnostic evidence, not a claim of model superiority.
