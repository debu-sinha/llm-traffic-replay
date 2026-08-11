# Testing Databricks-hosted GLM 5.2

**Procedure status:** reviewed against this repository and the live Databricks
documentation on 2026-08-10. Provider availability, limits, model revisions,
and accepted request controls can change. Recheck the linked Databricks pages
and the target workspace immediately before paid traffic.

This guide explains how to test the Databricks-hosted
`databricks-glm-5-2` pay-per-token endpoint with `llm-traffic-replay`. It
separates three questions that must not be collapsed into one green result:

1. Does the load instrument correctly speak the endpoint protocol?
2. Does the exact customer workload meet its pre-agreed targets at the exact
   tested load?
3. Is the resulting evidence strong enough to support a narrowly worded
   capacity statement?

The replay, scheduling, evidence, and reporting core is model-independent
within an installed endpoint-protocol adapter contract. Model identity, route,
provider controls, and workload stay in configuration, but a different wire
dialect still needs its own adapter and conformance evidence. This document is
GLM 5.2-specific because it records the currently tested direct route, the
route-specific reasoning control, and the dated quota scope.

## 1. Supported boundary

The implemented and regression-tested harness boundary in this release is
deliberately narrow. It is not a claim that every deployment using this
protocol is production-ready:

| Dimension | Current harness boundary |
|---|---|
| Provider route | Exact standard workspace route `/serving-endpoints/databricks-glm-5-2/invocations` |
| Endpoint binding | At run time, the direct endpoint must be `READY`, non-route-optimized, and bound by active-entity metadata to the configured endpoint and foundation model; the target workspace used for the live canary reported `databricks-glm-5-2` and `system.ai.databricks-glm-5-2`, which must be re-resolved rather than assumed for another workspace or revision |
| API dialect | Streamed OpenAI-compatible Chat Completions |
| Adapter | `openai.chat_completions.sse/v1` |
| Request | Text `messages`, `max_tokens`, optional `temperature`, `stream: true`, optional streamed usage, and reviewed provider controls |
| Response | `text/event-stream` SSE with Chat Completions deltas and a terminal `[DONE]` or `finish_reason` |
| Authentication | Origin-matched Databricks PAT, CLI user-to-machine profile, or workspace OAuth machine-to-machine profile |
| Workload inputs | Text-only synthetic profile or approved text prompt replay |
| Load axis | Open-loop offered requests per second; concurrency is observed, not held |

Three similar-looking identifiers have different evidence and must not be
interchanged:

- `databricks-glm-5-2` is the public Databricks pay-per-token endpoint name;
- `system.ai.glm-5-2` is the Unity AI Gateway model-service name confirmed by
  serving engineering for this engagement. Public documentation establishes that
  Databricks provides a `system.ai` service for each Databricks-served
  foundation model, but does not currently enumerate this exact GLM FQN; and
- `system.ai.databricks-glm-5-2` is foundation-model metadata observed on the
  target workspace's direct endpoint. It is a control-plane observation, not
  the Gateway request-body model name or a universal constant.

List the installed adapter contract before a run:

```bash
python3 -m traffic_replay adapters --format json
```

The command must list `openai.chat_completions.sse/v1`. Unknown adapter IDs
fail before credentials or network access. Select the adapter explicitly in
customer commands even though it is currently the default:

```text
--endpoint-adapter openai.chat_completions.sse/v1
```

Do not add `--model` for the direct endpoint above. That flag is for a shared
Chat Completions route that requires a request-body model. The current quota
and endpoint-binding gate does not bind that shared route to one stable
destination and quota scope.

### Boundaries outside the current harness contract

- Unity AI Gateway Chat Completions can be exercised only as a protocol
  diagnostic in this release. The harness does not bind a model service to
  its destinations, routing and fallback state, or the intersection of
  Gateway and downstream quotas. It therefore supports no Gateway quota or
  capacity conclusion.
- Databricks documents non-streaming Chat responses and an Open Responses API
  that is available across Databricks-hosted open models, with
  features enabled per model. Those provider surfaces are not implemented by
  this release's streamed Chat/SSE adapter, and the public pages do not provide
  a complete GLM-specific Responses compatibility table. A changed URL is not
  evidence of equivalent reasoning, tool, stream, usage, quota, or timing
  behavior.
- Route-optimized endpoints, native provider APIs, and multimodal payloads are
  also outside the current adapter boundary. GLM 5.2 is publicly documented as
  text-input only.
- Direct SGLang may use a compatible Chat/SSE envelope, but its reasoning
  control is not the managed Databricks control documented below. It requires
  a separately validated endpoint contract and quota model.
- This guide does not establish provisioned-throughput support or performance
  for GLM 5.2. Do not apply generic provisioned-throughput statements to this
  pay-per-token endpoint.
- Structural answer and tool-call checks are not semantic quality scoring,
  factuality evaluation, safety evaluation, or business-task completion.

## 2. Reasoning configuration: public fact versus serving-engineering confirmation

The public Databricks reasoning-model guide currently:

- lists `databricks-glm-5-2` as a reasoning-only model;
- says models in that row always use internal reasoning;
- names the `reasoning_effort` request field; and
- does **not** enumerate GLM 5.2-specific accepted values or its omitted
  default.

For this engagement, Databricks serving engineering separately confirmed the
managed request contract:

- top-level `{"reasoning_effort":"none"}` disables reasoning; and
- omitting `reasoning_effort` selects maximum reasoning.

That confirmation covers the managed direct endpoint and the Unity AI
Gateway model service `system.ai.glm-5-2` for the route/revision discussed in
this engagement. Neither the exact FQN nor the GLM-specific `"none"` and
omission semantics are independently enumerated by the current public guide;
the guide's “always use internal reasoning” wording also makes the thinking-off
behavior an explicit public-documentation gap. Preserve the chosen value in
the exact run configuration and revalidate it for the target route and model
revision. An HTTP 200 proves only that the request was accepted; it does not by
itself prove that reasoning was disabled or that the field changed behavior.

For the managed thinking-off run, use:

```text
--extra-body '{"reasoning_effort":"none"}'
```

Do not translate controls between stacks:

- managed Databricks GLM 5.2: `{"reasoning_effort":"none"}`;
- direct SGLang thinking-off: a separately documented
  `{"chat_template_kwargs":{"enable_thinking":false}}` control; and
- Z.ai-hosted APIs: their own provider-native request contract.

To study reasoning impact, run default/unset and thinking-off as separate,
sealed configurations against the same frozen workload and authorized load.
Treat the comparison as directional if any other field or endpoint condition
differs. Inspect first-visible latency, reasoning-channel observations,
reasoning-token coverage when reported, completion status, visible-answer
rate, output-budget exhaustion, and task quality outside this harness.

### 2.1 Independent live diagnostics from the direct route

An authorized live diagnostic on 2026-08-10 exercised the exact direct route
`/serving-endpoints/databricks-glm-5-2/invocations` with the same shipped
canary profile in two separate request configurations:

- with explicit top-level `{"reasoning_effort":"none"}`, the 320-token and
  480-token preflight representatives produced user-visible answers and the
  harness observed no reasoning-channel deltas; and
- with `reasoning_effort` omitted, the 320-token and 480-token representatives
  produced reasoning-channel content but no visible answer, ended with
  `finish_reason: "length"`, and the preflight correctly stopped before
  calibration or measured replay.

This is narrow, route-specific behavioral evidence. It does not prove that the
request field alone caused the difference, establish every accepted value or
the default for another model revision, qualify Unity AI Gateway or another
workspace, measure answer quality, or support any latency, throughput, quota-
headroom, or capacity conclusion. These are separate diagnostics, not a
cryptographically paired controlled experiment or a released benchmark
result. Preserve both request configurations and raw evidence with the
engagement record; repeat the behavioral preflight
after any route, model revision, template, or output-budget change.

### 2.2 Function calling and adjacent public capability boundaries

Databricks currently lists GLM 5.2 function calling on pay-per-token endpoints.
The public function-calling feature is in Public Preview and is optimized for
single-turn calls. Its current documented limits include:

- parallel function calling is not supported;
- at most 32 functions can be supplied;
- a function JSON schema can contain at most 16 keys;
- `pattern`, `anyOf`, `oneOf`, `allOf`, `prefixItems`, and `$ref` are not
  supported; and
- object and array length constraints are not enforced.

Use the customer's actual schema in an authorized serial preflight. This
harness can judge whether streamed tool-call fragments assemble into a
nonempty function name and JSON-object arguments; it does not execute the
function or establish appropriate tool selection, argument correctness,
workflow completion, or multi-turn recovery. Do not request or report parallel
GLM tool-call behavior under the current documented contract.

Other public serving facts remain separate from this adapter:

- GLM 5.2 is not in the current Priority pay-per-token supported-model list.
  Keep `service_tier` absent or exactly `"default"` for the bundled quota
  snapshot; do not infer priority availability from generic API syntax.
- The regional model matrix marks GLM with `⥂` in multiple regions, meaning
  support depends on GPU availability and requires cross-geography routing.
  Resolve the exact workspace region, policy, and current availability before
  traffic instead of describing GLM as universally available.
- The direct Chat API documentation does not promise a GLM-specific cached-
  token response field or document GLM cache warm-up, eviction, or hit
  semantics. Unity AI Gateway usage tracking has generic cache token details,
  but that system-table schema is not proof that a direct GLM response reports
  them. Missing direct-response cache evidence must remain `NOT REPORTED`.
- No static GLM price, maximum output-token value, model revision, hardware,
  or capacity guarantee is established by this guide. Resolve and date those
  inputs independently when they matter to the decision.

## 3. Required intake and authorization

Do not schedule provider traffic until the customer and test owners have
recorded all of the following.

### Business and ownership

- The decision this test informs: protocol diagnosis, target-load validation,
  comparative evaluation, or authorized capacity sweep.
- Named customer technical owner, Databricks owner, test operator, and stop
  authority.
- Approved workspace, cloud, region, endpoint, workspace tier, test window,
  expected spend boundary, and whether the endpoint serves other traffic.
- Written permission for synthetic, scrubbed, or raw customer prompts and the
  approved artifact retention and sharing boundary.

### Workload contract

- Average, target, peak, and burst request rates and durations.
- Input-token, output-token, and cache-eligible-prefix distributions. Prefer
  an empirical joint distribution when correlations matter.
- Approved prompt or conversation records when semantics, tools, or quality
  matter.
- The exact reasoning profile: omitted/default or explicit thinking-off.
- Stream mode, cancellation behavior, retry policy, and the production
  client's connection behavior.
- Customer-owned p50/p90/p95/p99 first-visible and full-generation targets,
  acceptable success fraction, and error budget. Do not create targets after
  seeing results.
- Quality, tool correctness, and workflow success evaluation performed outside
  this structural load harness.

### Capacity and safety

- Current live quota row, workspace tier, unrelated workspace traffic, and
  any account-specific overrides.
- Authorized fixed rates or sweep rungs, client worker bound, pending-request
  bound, maximum output budget, and abort conditions.
- Customer and platform telemetry owners who can identify impact during the
  run.

If any required value is missing, label the run diagnostic or exploratory. It
cannot support a production-capacity conclusion.

## 4. Input schemas and complete examples

Exactly one workload source is permitted: `profile_path` or `prompts_file`.

### 4.1 Profile JSON

A schema-v1 profile requires `name`, and exact `p50`/`p95` numeric objects for
`input_tokens`, `output_tokens`, and `cache_fraction`. Token anchors must be
positive with `p95 >= p50`; cache fractions must satisfy
`0 <= p50 <= p95 <= 1`. `provenance` and `label` should state exactly how the
numbers were obtained and what they can establish.

`cache_fraction` is the intended share of input tokens placed in a reusable
prefix. It is not a request cache-hit probability and is not proof that the
server used a cache.

The repository's complete canary profile is:

```json
{
  "name": "glm52_p2t_canary_illustrative",
  "input_tokens": {
    "p50": 1000,
    "p95": 2000
  },
  "output_tokens": {
    "p50": 320,
    "p95": 480
  },
  "cache_fraction": {
    "p50": 0.0,
    "p95": 0.0
  },
  "provenance": "Illustrative instrument-conformance workload shape created and quota-revalidated on 2026-08-10. This profile describes workload shape only: synthetic input, output, and reusable-prefix quantiles. Endpoint identity, adapter, route, API, reasoning controls, quota assertions, and model behavior belong to the run configuration and its dated evidence. When used unchanged by the documented one-row canary command, the 320/480-token output budgets derive a 720-token request cap and its complete 12-attempt fallback envelope remains below 80% of the separately supplied 20,000 output-token/minute quota snapshot. The profile alone does not enforce that request population or attempt envelope. These values are not measured customer output demand, a natural-answer-length estimate, or a Databricks performance recommendation.",
  "label": "ILLUSTRATIVE CANARY ONLY: use a quota-planned, low-rate preflight to test transport, answer completeness, usage, and timing; expect workload-fidelity caution and never quote this profile as customer demand, performance, or capacity."
}
```

This example is valid only for instrument conformance. Build the customer
profile from approved content-free request records instead:

```bash
python3 scripts/profile_from_logs.py \
  --input APPROVED_REQUEST_METRICS.jsonl \
  --name CUSTOMER_WORKLOAD_VERSION \
  --input-field input_tokens \
  --output-field output_tokens \
  --cached-field cached_input_tokens \
  --mode empirical-joint \
  --out configs/profile_measured.json

python3 -m traffic_replay sample \
  --profile configs/profile_measured.json \
  --n 50000 \
  --seed 7
```

Review extraction counts, dropped rows, provenance, empirical weights, and
recovered quantiles before use. The 50,000 draws above validate the sampler;
they do not create 50,000 endpoint requests.

### 4.2 Prompt replay JSONL

JSONL accepts one JSON value per nonblank line. Supported values are a
`messages` object, a `prompt` string, a `text` string, an inline
`role`/`content` object, or a bare JSON string. Every message requires a
nonempty string role and string content. Multimodal content arrays are
rejected.

A complete three-request example is:

```jsonl
{"messages":[{"role":"system","content":"You are a concise support agent."},{"role":"user","content":"A customer's order arrived two days late. Draft a short apology and offer a 10 percent credit."}]}
{"prompt":"Explain the difference between a provisioned throughput endpoint and a pay-per-token endpoint in two sentences."}
{"text":"Classify this ticket as billing, technical, or account, and give one reason: 'I was charged twice this month.'"}
```

These are repository examples, not a GLM quality set. Replace them with
approved customer prompts. The runner does not persist raw prompts or response
content in the run artifact, but the source prompt file still contains that
data and must remain inside the approved data boundary.

### 4.3 Databricks pay-per-token quota JSON

The rate-limit object is closed: unknown fields fail. At least one limit is
required. This repository currently ships the following complete snapshot:

```json
{
  "input_tokens_per_minute": 200000,
  "output_tokens_per_minute": 20000,
  "queries_per_hour": 7200,
  "queries_per_second": 200,
  "request_bytes_max": 4000000,
  "warning_utilization": 0.8,
  "source": "https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits",
  "as_of": "2026-08-07",
  "verified_at": "2026-08-10",
  "max_age_days": 7,
  "scope": "Published Enterprise workspace pay-per-token endpoint quota row plus Foundation Model API workspace QPS and per-request payload limits; this snapshot does not model other traffic that may consume the same limits",
  "provider": "databricks",
  "deployment_mode": "pay_per_token",
  "workspace_tier": "Enterprise",
  "model": "databricks-glm-5-2",
  "accounting_model": "databricks_fmapi_pay_per_token",
  "note": "Published default snapshot, not measured headroom. The source states a 4 MB payload limit without specifying a binary byte convention; the harness uses the conservative decimal ceiling of 4,000,000 serialized request bytes. Rechecked against the live source on 2026-08-10; the source page was last updated 2026-08-07. Recheck the live source, workspace tier, endpoint mode, and unrelated workspace traffic immediately before every paid run."
}
```

At the cited 2026-08-07 page revision, the Enterprise pay-per-token GLM 5.2
row is 200,000 input tokens/minute, 20,000 output tokens/minute, and 7,200
queries/hour. The same page states a workspace limit of 200 queries/second and
a 4 MB request payload. The harness conservatively represents 4 MB as
4,000,000 serialized request bytes. Its `warning_utilization` of 0.8 is a
harness safety boundary, not a Databricks-published quota.

Databricks states that actual input tokens are counted, offered `max_tokens`
is reserved before admission, unused output reservation is credited back, and
the most restrictive rolling limit applies. The harness plans conservatively
against the offered output budget and all possible physical attempts.

Before every paid run, reopen the live source and confirm:

- the GLM 5.2 row has not changed;
- the workspace is actually Enterprise tier;
- this remains a pay-per-token direct endpoint;
- no account-specific override changes the limit;
- unrelated workspace traffic leaves enough room; and
- `verified_at` truthfully records this new review.

Never refresh only the date. If the live values or scope change, create a new
snapshot and re-plan the complete run.

### 4.4 Complete direct-run JSON

The high-level `benchmark` command is preferred because it freezes inputs,
plans quota, and owns the two-request preflight. The following is a complete,
schema-valid direct-run representation of the instrument canary. Replace the
workspace host and profile name. Running it with `run` does **not** recreate
the CLI preflight, so use it only after the preflighted canary or use the exact
immutable rerun configuration emitted by `benchmark`.

```json
{
  "endpoint": {
    "base_url": "https://YOUR-WORKSPACE-HOST",
    "path": "/serving-endpoints/databricks-glm-5-2/invocations",
    "auth_profile": "YOUR-DATABRICKS-PROFILE",
    "adapter": "openai.chat_completions.sse/v1",
    "temperature": 0.0,
    "max_retries": 0,
    "include_usage": true,
    "extra_body": {
      "reasoning_effort": "none"
    }
  },
  "profile_path": "configs/profile_glm52_canary_illustrative.json",
  "duration_s": 12,
  "qps_base": 0.1,
  "qps_burst": 0.1,
  "qps_min": 0.1,
  "qps_max": 0.1,
  "rate_scale": 1.0,
  "max_concurrency": 1,
  "max_pending_requests": 1,
  "seed": 7,
  "cpt": 4.0,
  "calibrate_n": 1,
  "max_output_tokens_cap": 720,
  "ttft_definition": "first_visible",
  "capture_endpoint_metadata": true,
  "measure_network_path": true,
  "out_dir": "results/glm52-direct-config-canary",
  "title": "GLM 5.2 direct endpoint instrument canary",
  "label": "INSTRUMENT CONFORMANCE ONLY - NOT CUSTOMER DEMAND OR CAPACITY",
  "rate_limits": {
    "input_tokens_per_minute": 200000,
    "output_tokens_per_minute": 20000,
    "queries_per_hour": 7200,
    "queries_per_second": 200,
    "request_bytes_max": 4000000,
    "warning_utilization": 0.8,
    "source": "https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits",
    "as_of": "2026-08-07",
    "verified_at": "2026-08-10",
    "max_age_days": 7,
    "scope": "Published Enterprise workspace pay-per-token endpoint quota row plus Foundation Model API workspace QPS and per-request payload limits; this snapshot does not model other traffic that may consume the same limits",
    "provider": "databricks",
    "deployment_mode": "pay_per_token",
    "workspace_tier": "Enterprise",
    "model": "databricks-glm-5-2",
    "accounting_model": "databricks_fmapi_pay_per_token",
    "note": "Published default snapshot, not measured headroom. Recheck before paid traffic."
  }
}
```

Run JSON accepts exactly one of `profile_path` and `prompts_file`; unknown
top-level fields are rejected. The endpoint adapter owns `messages`,
`max_tokens`, `temperature`, `stream`, `model`, and `stream_options`, so those
keys cannot be placed in `extra_body`. Secrets also cannot be placed in
`extra_body` because the object is persisted as reproducibility evidence.

The commands below retain the current CLI's explicit numeric
`temperature: 0.0`, matching the retained direct-route canary. If preflight or
the exact route contract requires the field to be absent, add
`--omit-temperature` and treat that as a separate configuration; numeric zero
and omission must not be compared as though they were identical. Never put
`temperature` in `--extra-body` because the selected adapter owns it.

The immutable rerun config produced by `benchmark` additionally contains
`input_expectations` with the exact workload-file SHA-256 and byte count. A
rerun refuses if those external bytes changed.

## 5. Execution procedure

All provider-facing commands below send real, potentially billable requests.
Replace every uppercase placeholder and obtain authorization first.

### Step 1: local instrument gate

No provider traffic is generated by these commands:

```bash
python3 -m pytest
python3 -m traffic_replay validate --port 0 --format json
python3 -m traffic_replay adapters --format json
```

`validate` compares client-observed timing with a local known-latency mock. A
pass validates the local measurement path, not Databricks, GLM 5.2, the
workspace network, or production capacity.

### Step 2: validate the workload input

For profile mode, inspect deterministic sampler recovery:

```bash
python3 -m traffic_replay sample \
  --profile configs/profile_measured.json \
  --n 50000 \
  --seed 7
```

There is no separate public prompt-validation or preflight-only subcommand.
`benchmark` freezes and strictly validates the exact profile or prompt bytes
before credential lookup or endpoint traffic. Do not invent a standalone
`preflight` command.

### Step 3: minimal GLM 5.2 canary and preflight

The following shipped canary performs two representative preflight requests,
one calibration request, and one measured replay request. With the default
compatibility/fallback envelope it permits at most 12 physical POST attempts.
Its 720-token request cap and offline quota plan are instrument-safety facts,
not observed usage, customer demand, latency, throughput, or capacity.

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint databricks-glm-5-2 \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_glm52_canary_illustrative.json \
  --endpoint-adapter openai.chat_completions.sse/v1 \
  --extra-body '{"reasoning_effort":"none"}' \
  --fixed-rate 0.1 \
  --duration 12 \
  --calibrate-requests 1 \
  --ttft-definition first_visible \
  --rate-limits configs/rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json \
  --out-dir results/glm52-instrument-canary \
  --label "INSTRUMENT CONFORMANCE ONLY - NOT CUSTOMER DEMAND OR CAPACITY"
```

The preflight is passed only when both representatives are reachable and
produce a clean, completed, structurally acceptable answer: non-refusal
visible content or a structurally valid non-refusal tool call, terminal stream
evidence, and no unrecoverable parse errors. HTTP 200 alone is insufficient.

If this canary fails, stop. Do not use `--force` to obtain a green result.
`--force` only authorizes an explicitly invalid diagnostic after a reachable
but unreadable response; it never changes the preflight outcome to passed and
does not override a reachability failure.

The explicit calibration count makes this canary's traffic population
reviewable. Its one calibration request runs before replay and can warm
endpoint, model-worker, and exact-payload state; the report records exact body
hash overlap when available. `--calibrate-requests 0` suppresses harness
calibration but does not reset or prove cold cache. For a customer run, choose
the count before authorization and include it in the quota plan.

### Step 4: authorized customer fixed-rate run

Use a measured profile, a rate known before traffic starts, customer-owned
targets, explicit client bounds, and the revalidated quota snapshot:

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint databricks-glm-5-2 \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --endpoint-adapter openai.chat_completions.sse/v1 \
  --extra-body '{"reasoning_effort":"none"}' \
  --fixed-rate YOUR_AUTHORIZED_REQUESTS_PER_SECOND \
  --duration YOUR_AUTHORIZED_DURATION_SECONDS \
  --calibrate-requests YOUR_AUTHORIZED_CALIBRATION_REQUESTS \
  --max-concurrency YOUR_TESTED_WORKER_BOUND \
  --max-pending-requests YOUR_TESTED_PENDING_BOUND \
  --ttft-definition first_visible \
  --ttft-p95 YOUR_FIRST_VISIBLE_P95_TARGET_MS \
  --ttfg-p95 YOUR_FULL_GENERATION_P95_TARGET_MS \
  --success-rate YOUR_TARGET_FRACTION_STRICTLY_BETWEEN_ZERO_AND_ONE \
  --rate-limits configs/rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json \
  --fail-on caution \
  --out-dir results/customer-glm52-fixed-rate \
  --title "Customer GLM 5.2 fixed-rate evaluation" \
  --label "AUTHORIZED CUSTOMER WORKLOAD AND TEST WINDOW"
```

`--max-concurrency` is a worker safety bound, not the load axis.
`--max-pending-requests` bounds running plus queued client work. The report
records delivered rate, request-start lateness, observed in-flight
concurrency, and pending-limit drops.

The client opens a fresh HTTP/1.1 connection for every physical attempt. Add
the following flag only if the production application does exactly the same:

```text
--production-connection-policy fresh_http1_per_physical_attempt
```

The flag is an operator assertion, not an observation of production. Omit it
for pooled keep-alive, HTTP/2, or unknown clients; the report will appropriately
keep capacity inconclusive while retaining diagnostic measurements.

For approved prompt replay, replace `--profile` with `--prompts` and explicitly
provide the output distribution used to derive the safety cap:

```text
--prompts APPROVED_PROMPTS.jsonl --output-tokens CUSTOMER_P50,CUSTOMER_P95
```

The high-level command derives `max_output_tokens_cap` as
`ceil(1.5 * output p95)`. Confirm the resulting immutable configuration before
the paid run.

### Step 5: exact rerun

`benchmark` prints the immutable generated run-config path. Preserve that
path; do not substitute a hand-edited config. Rerun the same frozen contract
with:

```bash
python3 -m traffic_replay run \
  --config EXACT_IMMUTABLE_RUN_CONFIG_PATH \
  --fail-on caution
```

`run` does not add the high-level command's separate two-request preflight.
Use it only for an authorized rerun after the route and configuration have
already passed the required gate.

### Step 6: optional authorized rate sweep

Use a sweep only after the fixed-rate point is valid and each rung has been
authorized. A production conclusion requires customer targets; otherwise use
`--diagnostic-only`, which intentionally publishes no held-rate conclusion.

```bash
python3 -m traffic_replay sweep \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint databricks-glm-5-2 \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --endpoint-adapter openai.chat_completions.sse/v1 \
  --extra-body '{"reasoning_effort":"none"}' \
  --rate YOUR_AUTHORIZED_RUNG_1,YOUR_AUTHORIZED_RUNG_2,YOUR_AUTHORIZED_RUNG_3 \
  --duration YOUR_AUTHORIZED_SECONDS_PER_RUNG \
  --cooldown YOUR_AUTHORIZED_SPACING_SECONDS \
  --cpt YOUR_PREMEASURED_CHARACTERS_PER_TOKEN \
  --max-concurrency YOUR_TESTED_WORKER_BOUND \
  --max-pending-requests YOUR_TESTED_PENDING_BOUND \
  --ttft-definition first_visible \
  --ttft-p95 YOUR_FIRST_VISIBLE_P95_TARGET_MS \
  --ttfg-p95 YOUR_FULL_GENERATION_P95_TARGET_MS \
  --success-rate YOUR_TARGET_FRACTION_STRICTLY_BETWEEN_ZERO_AND_ONE \
  --rate-limits configs/rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json \
  --out-dir results/customer-glm52-sweep
```

Cooldown is spacing, not proof that rolling quota, cache, autoscaling, or
unrelated traffic reset. A held top rung is not an endpoint ceiling.

### Step 7: verify and view the report

The benchmark prints the exact timestamped run directory. Use that exact path:

```bash
python3 -m traffic_replay verify-run \
  EXACT_RUN_DIRECTORY \
  --out EXACT_RUN_DIRECTORY-verification \
  --format json
```

The verification output must be a sibling of the source run, not inside it.
For a sweep:

```bash
python3 -m traffic_replay verify-sweep \
  results/customer-glm52-sweep \
  --format json
```

There is no separate report-generation command. A successful run writes
`report.html` and `report.md`; `verify-run` writes separately sealed
`verified-report.html` and `verified-report.md`. Share the verified view with
the untouched source run and receipt. A browser-printed PDF is an unsealed
derivative, not the evidence artifact.

## 6. Input and output artifacts

### Inputs retained by the operator

```text
configs/profile_measured.json          # or approved prompts JSONL
configs/rate_limits_*.json             # dated, revalidated provider snapshot
immutable generated run-config.json    # exact rerun contract and input hashes
source extraction record               # provenance and dropped-row review
customer authorization and targets     # retained in the approved system
```

The measured run stores workload filenames, hashes, byte counts, and
construction metadata, not raw prompts or response text. Endpoint metadata,
request parameters, `extra_body`, error statuses, bounded error-body hashes,
and timing/usage measurements are evidence and must be reviewed before
external sharing.

### Setup-traffic artifact

With preflight enabled, `benchmark` first creates a sibling setup directory:

```text
OUT_DIR-setup-traffic/TIMESTAMP/
```

It durably records metadata-only preflight and explicit probe rows. A normal
pass or refusal seals the setup artifact with no performance, SLA, or capacity
result. A crash leaves diagnostic partial evidence.

### Completed measured run

```text
.traffic-replay-complete
start.json
requests.jsonl
summary.json
report.md
report.html
manifest.json
```

- `start.json` records the effective redacted configuration, schedule,
  endpoint binding, quota plan, and runtime setup state.
- `requests.jsonl` is the canonical content-free row journal for preflight,
  probes, sizing, calibration, and replay.
- `summary.json` contains the canonical aggregates and five-part decision.
- `report.md` and `report.html` render the same decision and measurements.
- `manifest.json` binds artifact hashes, sizes, row counts, source identity,
  workload identity, execution identity, and configuration evidence.
- `.traffic-replay-complete` binds the final manifest and request-row count.

An interrupted run retains `.traffic-replay-writing` and may contain
`requests.jsonl.partial` and `failure.json`. It is diagnostic only and must not
be presented as a completed benchmark.

### Verification receipt

```text
.traffic-replay-complete
verification.json
verified-report.md
verified-report.html
manifest.json
```

The receipt proves internal hash consistency and reconstructibility checks. It
is not a digital signature, trusted timestamp, proof of authorship, or proof
that the source repository remains available.

## 7. Reading the report

Read the five decision dimensions independently and in this order:

| Dimension | Codes | Meaning |
|---|---|---|
| Evidence integrity | `VERIFIED`, `VERIFY_REQUIRED`, `TAMPERED` | Whether the enclosing artifact chain was externally checked |
| Measurement validity | `VALID`, `CAUTION`, `INVALID` | Whether protocol, identity, delivery, workload, sample, and compatibility evidence support interpretation |
| Acceptance checks | `PASS`, `MISS`, `INCONCLUSIVE`, `NOT_EVALUATED` | Outcome against targets defined before the run |
| Quota state | `EXCEEDED`, `LOCAL_GUARD_REFUSED`, `NOT_OBSERVED`, `UNKNOWN`, `NOT_EVALUATED` | Observed HTTP 429 and local admission evidence |
| Endpoint capacity | `HELD_AT_TESTED_LOAD`, `NOT_HELD_AT_TESTED_LOAD`, `INCONCLUSIVE`, `NOT_EVALUATED` | Only what happened at the exact tested point |

A newly created source report says `VERIFY_REQUIRED` because it cannot verify
the manifest that will enclose it. Use the separate receipt before sharing.

### Timing definitions

- `ttfb_ms`: time from immediately before the final physical POST to the first
  nonempty bounded response-body chunk returned by the client. It is not the
  first socket byte.
- `ttse_ms`: time to the first complete framed event emitted by the selected
  response adapter parser. It is a protocol diagnostic, not a token clock: the
  event can be usage-only, terminal, or a content-free parse diagnostic.
- `ttft_ms`: the configured acceptance metric. `first_content` means first
  visible, reasoning, or refusal delta; `first_visible` waits for user-visible
  assistant content. Tool-call fragments do not trigger TTFT.
- `ttfr_ms`: first reasoning-channel delta, when distinguishable.
- `ttfv_ms`: first visible assistant-content delta.
- `e2e_ms`: complete streamed response time.
- `interchunk_max_ms`: widest gap between successive eligible nonempty SSE
  content events. It is not token-to-token latency.

For GLM 5.2 customer-facing latency, use `first_visible` unless the customer
explicitly owns a different definition. A fast first reasoning delta can
coexist with a late or absent visible answer.

### Coverage and validity

- Failed, refused, incomplete, malformed, and output-budget-exhausted requests
  remain in error and success denominators; they are not zero-latency wins.
- `finish_reason: length` is truncation evidence. It is not a completed
  customer answer merely because HTTP status was 200.
- `NOT REPORTED` cached tokens means missing recognized endpoint evidence, not
  zero cache reuse. Intended reusable-prefix fraction and endpoint-reported
  cached-token fraction are separate.
- Reasoning-token throughput is shown only when a recognized usage field is
  reported. Reasoning SSE delta counts are event counts, not token estimates.
- `completion_tokens`, all-completion throughput, and completion TPOT can
  include hidden reasoning. Visible-output throughput or TPOT is published
  only when an exact provider field supplies visible-token counts with complete
  clean-usage coverage; visible SSE chunks are events, not token counts.
- HTTP 429 proves a rate-limit rejection, not which component enforced it and
  not a compute-capacity ceiling. Zero observed 429s is `NOT_OBSERVED`, not
  provider headroom.
- `LOCAL_GUARD_REFUSED` means the harness suppressed a POST to protect the
  configured budget. It is not an endpoint 429.
- Stable latency among a shrinking survivor population is not stable service;
  read error rate and event coverage with every percentile.

The report marks latency quantiles indicative below these implementation
evidence floors:

| Quantile | Minimum structurally acceptable answer-latency observations |
|---|---:|
| p50 | 20 |
| p90 | 100 |
| p95 | 200 |
| p99 | 1000 |

Success and latency targets use both the observed compliance fraction and a
one-sided 95% Wilson lower confidence bound. A pass requires the lower bound to
meet the target, under the stated independent-request assumption.

`HELD_AT_TESTED_LOAD` never means an endpoint ceiling, unused provider
headroom, a future guarantee, or performance on a different route, workload,
reasoning profile, client, or capacity product.

## 8. Apples-to-apples review

Before comparing GLM 5.2 with an incumbent or another GLM run, answer every
question below in the report appendix. If any material answer is no or
unknown, label the comparison directional rather than equivalent.

- Same frozen prompt/messages and chat template?
- Same input, output-budget, and reasoning-token distributions?
- Same chronological order, conversation boundaries, cache warm-up, and
  reusable-prefix construction?
- Same exact reasoning field and value, including omission semantics?
- Same `first_visible` definition, stream mode, cancellation, timeout, and
  retry policy?
- Same route, endpoint mode, region, network path, and endpoint revision?
- Same sampling parameters and tool schemas?
- Same offered arrival schedule, client worker/pending bounds, and production
  connection behavior?
- Same customer quality set, judge or executable tests, and success rubric?
- Same pricing scope and effective date if cost is compared?
- Same status/usage coverage and no unresolved identity, quota, or endpoint
  metadata mismatch?

The current `compare` command preserves and checks source evidence; it cannot
turn configurations that differ into an equivalent controlled experiment.

## 9. Safety and stop conditions

- Use only the authorized workspace, endpoint, traffic window, workload, and
  maximum rate. Do not test a shared production path without explicit owner
  approval and monitoring.
- Recheck quota and workspace tier before each command. The local guard cannot
  observe other workspace consumers or provider-side bucket state.
- Keep credentials in `auth_profile` or `auth_token_env`, never prompts,
  labels, filenames, `extra_body`, or checked-in configuration.
- Default to `max_retries: 0`. A retried POST can duplicate provider work and
  billing; final-response usage does not account for earlier ambiguous
  attempts.
- Do not use `--skip-preflight` for customer evidence. Do not use `--force` to
  continue a failed gate except for a separately authorized invalid
  diagnostic.
- Stop on any local admission denial, HTTP 429, unexpected 5xx increase,
  timeout or parse-error increase, incomplete/visible-answer failure, pending
  drop, generator saturation, endpoint metadata change, model identity change,
  or observed customer impact.
- Preserve the completed source directory and sibling verification receipt.
  Do not edit sealed files or manually add a completion marker.
- Review endpoint names, metadata, labels, paths, hashes, and request controls
  before sharing. Treat browser-printed PDFs as unsealed derivatives.

## 10. Troubleshooting

| Symptom | Meaning and action |
|---|---|
| Authentication fails before traffic | Confirm the named profile exists and its configured host exactly matches the requested workspace origin. The built-in resolver does not mint route-optimized endpoint-scoped tokens. |
| Quota plan is stale or refused | Reopen the live limits page, verify the exact row/tier/scope, update `as_of` only if the provider fact changed, update `verified_at` only after a real review, and re-plan. Do not bypass the gate. |
| Endpoint binding refuses before inference | The exact direct path, public endpoint name, `READY` state, `route_optimized=false`, and served-entity identity must agree. The target canary workspace reported `system.ai.databricks-glm-5-2`; treat that value as observed binding evidence to re-resolve, not a public universal identifier. |
| Preflight returns HTTP 200 but no visible answer | Inspect reasoning deltas, `finish_reason`, output cap, stream completion, and parse errors. Use the serving-engineering-confirmed managed `reasoning_effort:none` only when that is the intended profile. HTTP 200 alone does not prove the control took effect. |
| A documented control is being explored | `--probe-extra-body` is diagnostic and runs only after an unreadable preflight. It never mutates the measured config. If a candidate produces a complete answer, rerun the full preflight with that exact object as `--extra-body`. |
| A probe returns HTTP 400 or 422 | Status alone is not evidence that the candidate was rejected. The selected adapter must allow that status and the bounded response wording must explicitly identify the candidate field or path. Otherwise disposition and effective behavior remain unknown. The response text is never persisted; only sample byte count, full-body SHA-256, and classification are retained. |
| Streamed usage option is rejected | The Chat adapter can retry without `stream_options.include_usage` only when the response specifically proves that optional field was rejected. The final config records the fallback; usage-dependent conclusions become unavailable or cautioned. |
| Cached tokens are absent | Report missing cache evidence. Do not convert missing to zero and do not call constructed prefixes server cache hits. |
| First-content is fast but first-visible is slow | Reasoning or refusal content arrived before user-visible text. Use `first_visible` for the customer-facing GLM metric and inspect the gap. |
| `finish_reason` is `length` | The offered output budget was exhausted. Increase it only inside a newly quota-planned and authorized run; do not relabel the truncated answer successful. |
| Response content type is rejected | The selected adapter accepts `text/event-stream`. Confirm the endpoint is streamed Chat Completions. Responses API and non-streaming JSON require a separately implemented and conformance-tested adapter even though Databricks documents those provider surfaces. |
| HTTP 429 occurs | Stop the capacity claim. Check the specific provider error, other workspace traffic, offered input and `max_tokens`, QPH/QPS, and the current limits. A 429 does not identify compute saturation. |
| Local guard refuses a POST | The configured safety budget would be crossed or admission evidence became uncertain. No denied POST was sent. Reduce or re-authorize the entire plan; do not call it provider throttling. |
| Dispatch/request-start lag or pending drops rise | The generator did not faithfully deliver the offered schedule. Reduce load, increase only the pre-approved client bounds, or shard with synchronized configuration; the endpoint result is invalid or inconclusive. |
| Report says `VERIFY_REQUIRED` | Create the sibling `verify-run` receipt. Do not edit the source report to change the label. |
| Immutable rerun refuses changed input | The source profile, prompts, or trace no longer matches its recorded SHA-256/byte count. Restore the exact input or create a new reviewed run identity. |

## 11. Claims this procedure does not prove

Even a clean, externally verified run does not by itself prove:

- semantic answer quality, factual correctness, safety, tool choice, or
  end-to-end business outcome;
- that `reasoning_effort:none` changed model behavior solely because the
  request returned HTTP 200;
- production equivalence when the real client pools connections or uses
  HTTP/2 while the harness uses fresh HTTP/1.1 connections;
- cache reuse when cached-token usage is not reported with sufficient
  coverage;
- provider quota headroom, the cause of a 429, or a model compute ceiling;
- GLM 5.2 provisioned-throughput availability or performance;
- Unity AI Gateway capacity, routing stability, or downstream quota
  intersection;
- performance for the Responses API, non-streaming requests, route-optimized
  serving, multimodal input, SGLang, quantized serving, or another adapter;
- current pricing, invoice cost, or cost per successful task without an exact
  dated pricing source, complete physical-attempt usage, and external quality
  scoring; or
- future performance after a model, route, runtime, quota, region, hardware,
  or endpoint configuration changes.

## 12. Sources

Provider sources, rechecked 2026-08-10:

- [Databricks-hosted foundation models](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models) - current endpoint name, text input, context, architecture, and intended workload description.
- [July 2026 release notes](https://docs.databricks.com/aws/en/release-notes/product/2026/july) - dated announcement of GLM 5.2 as a Databricks-hosted Foundation Model API model.
- [Foundation Model APIs limits and quotas](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits) - current dated P2T and workspace limit facts and provider admission accounting.
- [Supported foundation models on Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview) - current regional P2T matrix, the `⥂` availability qualifier, and provisioned-throughput architecture list.
- [Query reasoning models](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-reason-models) - public GLM reasoning classification and `reasoning_effort` field; it does not enumerate the GLM-specific `none` behavior separately confirmed by serving engineering.
- [Foundation Model API reference](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference) - Chat Completions request, streaming and non-streaming response forms, usage fields, and model-varying parameter caveats.
- [Function calling on Databricks](https://docs.databricks.com/aws/en/machine-learning/model-serving/function-calling) - current GLM P2T support and Public Preview function/schema limitations.
- [Query a model with the Open Responses API](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-open-responses-models) - generic Open Responses availability and per-model feature boundary; this release does not implement that wire contract.
- [Priority pay-per-token](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/priority-mode) - current supported-model list, which does not list GLM 5.2.
- [Query model APIs](https://docs.databricks.com/aws/en/ai-gateway/query-model-services) - Unity AI Gateway model-service and API routes.
- [Govern model APIs](https://docs.databricks.com/aws/en/ai-gateway/model-services) - model-service destination routing and fallback behavior that limits Gateway capacity claims here.
- [Track model usage](https://docs.databricks.com/aws/en/ai-gateway/usage-tracking) - Gateway system-table token-detail and cache-hit telemetry; it is not a direct GLM response-field guarantee.
- [Databricks CLI authentication commands](https://docs.databricks.com/aws/en/dev-tools/cli/reference/auth-commands) - CLI user-to-machine token behavior.
- [Route-optimized serving authentication](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-route-optimization) - separate scoped-authentication boundary not implemented by this resolver.

The managed `reasoning_effort:none` behavior and omitted maximum-reasoning
default are engagement-specific serving-engineering confirmations. The
independent live diagnostics above are direct-route observational evidence,
not public documentation or a global product contract. Preserve the
confirmation and raw diagnostic evidence
with the benchmark record, and reconfirm both if the endpoint route, model
revision, template, or output budget changes.
