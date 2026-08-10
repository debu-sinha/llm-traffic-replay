# Production testing runbook

This runbook is for an authorized load window against a production-like LLM
endpoint. It separates instrument validation, protocol validation, workload
validation, stepped load, and evidence review. Do not begin a capacity test
from an unreviewed maximum-rate config.

## 0. Define the experiment

Record these before sending traffic:

- endpoint origin, route, model, region, capacity product, and active serving
  configuration;
- the authorized time window and an operator who can stop the test;
- provider rate, token, and account quotas from provider telemetry or current
  documentation; for Databricks Foundation Model APIs, recheck the exact
  model/deployment/tier row in the official
  [limits and quotas](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits);
- a workload input with its source digest and known fidelity limits;
- request parameters, reasoning policy, tool schema behavior, and output cap;
- TTFT definition and customer-owned acceptance targets, including
  `targets_are` provenance, hard timeouts, and a success-rate target strictly
  between 0 and 1;
- the generator region, host size, process count, and expected network path;
- the operator-supplied pricing source, product/tier applicability, and
  effective date if diagnostic cost arithmetic will be reported;
- stop conditions for errors, latency, saturation, quota, cost, and production
  impact.

Do not assume a model is available on provisioned throughput because a
different model is. Confirm eligibility for the exact model and region before
creating a provisioned endpoint or making a provisioned-capacity claim.

## 1. Prove the instrument on the generator host

```bash
python3 -m pytest
python3 -m traffic_replay validate --port 0 --format json
python3 -m traffic_replay adapters --format json
```

The test suite is the implementation regression gate. `validate` checks this
host's end-to-end measurement path against the localhost oracle. Preserve the
JSON result with the run record. A passing mock check does not validate a
provider dialect, production network, or workload.

## 2. Validate authentication and protocol at minimal load

Use the one-command path with a measured profile or a small real-prompt set.
Keep the sizing hint small and retain the default preflight:

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --endpoint-adapter openai.chat_completions.sse/v1 \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_validation_small.json \
  --sizing-concurrency 2 \
  --duration 60 \
  --calibrate-requests 0 \
  --ttft-definition first_visible \
  --out-dir results/protocol-smoke \
  --fail-on none
```

This sends real inference traffic. Review:

- both representative requests reached the intended route;
- the response is valid SSE for this client;
- non-refusal visible content or a structurally valid non-refusal tool call completed cleanly;
- `request_attempts` is understood, especially usage fallback or auth refresh;
- prompt, completion, cached, and reasoning usage fields are either present
  with named source paths or explicitly absent;
- model, route, and request controls in the manifest match the intended target;
- response-model identity is consistent and bound where reported;
- the recorded fresh-HTTP/1.1-per-attempt transport is either the real
  application's exact connection behavior or explicitly treated as a
  diagnostic mismatch; do not use a transport-mismatched or unqualified run
  for capacity;
- endpoint metadata was captured both before runner-owned sizing, calibration,
  and replay traffic and after response drain, or its incomplete coverage is
  explained;
- no secret or response body content appears in artifacts.

This stage validates mechanics only. Its latency is not a capacity result.
`--calibrate-requests 0` keeps this protocol smoke from adding the high-level
command's default pre-replay calibration population. It does not clear or
prove cold endpoint, cache, worker, or autoscaling state. For a later synthetic
workload-fidelity run, choose an authorized count explicitly; clean usage can
recalibrate characters/token, and every positive count is real, potentially
billable warm-up traffic. Prompt replay is never retokenized but would still
send those configured calibration rows.

The harness does not pool connections and does not use HTTP/2. Production
capacity therefore remains inconclusive by default even when protocol and SLA
checks pass. Only add
`--production-connection-policy fresh_http1_per_physical_attempt` when the real
application is known to use that exact policy. This flag records an operator
assertion; it does not inspect production traffic. For pooled or HTTP/2 clients,
retain the qualification until a future matching transport adapter exists.

Before credentials, endpoint metadata, network diagnostics, or this preflight,
the command copies the workload and optional trace to private temporary bytes,
strictly parses that exact view, and constructs representative bodies. A
fixed-rate or trace-driven run also materializes its complete schedule at this
boundary. The sizing example above is the explicit exception: its paid sizing
sample must derive a rate before that schedule can exist. The runner repeats
input capture and validation before its own traffic boundary.

Before the first preflight or probe `POST`, the CLI claims
`results/protocol-smoke-setup-traffic/TIMESTAMP` and fsyncs each completed
metadata-only row. A normal pass or refusal seals it as an explicit
non-performance/non-SLA/non-capacity artifact; a crash leaves incomplete
diagnostic evidence. A forced unreadable preflight is sealed as
`preflight_forced_unreadable`, never `preflight_passed`; force permits only an
INVALID diagnostic run. Whenever the command proceeds past this gate, the same
rows are attached once to the measured run's complete quota population without
request or response content.

For a Databricks pay-per-token smoke test, do not combine the quota gate with
`--sizing-concurrency`: paid sizing traffic is required to derive that rate,
so the schedule cannot be bounded in advance. Replace the sizing flag with a
small, authorized `--fixed-rate` and add `--rate-limits RATE_LIMITS.json` as
described in section 4. The implemented snapshot gate is specific to direct
Databricks pay-per-token endpoints; it is not a generic provider or
provisioned-throughput quota mechanism.

Databricks recommends service-principal OAuth machine-to-machine (M2M) for
unattended automation. The named-profile resolver supports PAT `token`
profiles (`auth_type` omitted or `pat`), CLI-cached U2M profiles
(`auth_type=databricks-cli`), and workspace M2M `client_id` / `client_secret`
profiles (`auth_type` omitted or `oauth-m2m`). The M2M path exchanges the
credentials directly at the origin-bound workspace `/oidc/v1/token` endpoint
with `scope=all-apis`; the CLI path remains U2M-only.

Protect and rotate credentials under the workspace policy. Never place a PAT
or client secret in a run config, command argument, or report label. Mixed,
incomplete, unsupported, or host-mismatched profiles fail closed. The direct
M2M path supports only standard workspace-origin invocation and does not mint
the endpoint-scoped `authorization_details` token required by a
route-optimized serving URL. See Databricks'
[OAuth M2M guide](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m),
[`auth token` reference](https://docs.databricks.com/aws/en/dev-tools/cli/reference/auth-commands#databricks-auth-token),
and
[route-optimized authentication guide](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-route-optimization).
Treat PAT authentication as legacy and use it only where its scope, storage,
lifetime, and rotation have been approved.

### Reasoning and tool-call endpoints

For a reasoning model, choose `ttft_definition=first_content` when the
configured latency target starts at the first visible, reasoning, or refusal
delta, or
`ttft_definition=first_visible` when it starts at meaningful visible assistant
content. Those are the only two selectable TTFT definitions. First reasoning
content (`ttfr_ms`) and first tool-call fragment (`ttf_tool_call_ms`) are
reported separately but cannot currently be selected as the scored TTFT
basis.

All final-attempt latency clocks begin immediately before `conn.request` on an
already-established connection and therefore include request upload. TTFB is
the first nonempty bounded response-body chunk returned by the client read,
not the first socket byte or first parsed SSE line. `ttse_ms` is the later
first complete framed event emitted by the selected adapter parser. It can be
usage-only, terminal, or a content-free parse diagnostic and must not be
called a token or content latency. Tool-call fragments do not trigger TTFT;
first-visible and first-tool-call timings remain separate.

`interchunk_max_ms` is the widest elapsed gap between successive SSE events
with a nonempty visible, reasoning, or refusal delta. It is not token-level
inter-token latency: events can batch tokens, heartbeats and usage-only events
do not advance it, and tool-call-only fragments are excluded. Fewer than two
qualifying events leaves a protocol-clean row unmeasured; any such row makes a
configured interchunk check inconclusive.

Reasoning controls are model-specific request contract. Use the provider's
documented field and supported value for the exact model. Do not copy a
reasoning or template parameter from a different endpoint without a successful
protocol check. An accepted unknown field can still be ignored.

The harness does not guess these controls. A repeatable
`--probe-extra-body '{...}'` explicitly sends one additional real preflight
request per supplied candidate after an unreadable answer. Use it only for
model-documented candidates in an authorized probe. Metadata-only result rows
for those calls are first fsynced and sealed in the setup-traffic artifact and,
after a pass, are also included once in the measured run's quota evidence
without request or response content. The sealed gate and report retain
credential-free canonical candidate JSON and digest, disposition/evidence
method, explicitly limited effective-behavior status, request ID, logical
request-body hash, and physical-attempt hashes linked to `requests.jsonl`.
Retain stdout and stderr as supplemental operator context. A candidate is not
copied into the measured config; rerun with the selected object as
`--extra-body`. Acceptance never proves that the provider applied a control.

The measured `extra_body` is persisted as reproducibility evidence.
Secret-like keys and credential-shaped values are rejected recursively before
config output or traffic. Command arguments can still be visible locally.
Keep authentication in the endpoint profile or token environment variable.

A larger output budget can reveal whether the model eventually produces a
visible answer, but it changes work and cost. It is diagnosis, not a neutral
fix. The measured run must use the product's actual budget and must report
global-cap truncation separately from the sampled output target.

Tool-call-only responses are acceptable only when the stream assembles at
least one nonempty function name whose arguments decode to a JSON object and
terminates without parse errors. This validates structure, not tool choice or
argument semantics. Use an application evaluator for semantic correctness.

## 3. Validate workload fidelity

Prefer one of:

- real prompts from an approved test dataset; or
- a schema-v2 `empirical_joint` profile built from complete token/cache
  triples; or
- a schema-v2 `quantile_cdf` profile with measured marginal knots.

Use schema v1 only when p50/p95 marginals are all that exists, and label the
unobserved tails and dependence as assumptions.

For a log-derived profile:

```bash
python3 scripts/profile_from_logs.py \
  --input request_metrics.jsonl \
  --name measured_workload \
  --mode empirical-joint \
  --out configs/profile_measured.json

python3 -m traffic_replay sample \
  --profile configs/profile_measured.json --n 50000 --seed 7
```

Check the source byte count and SHA-256, extraction counts, dropped incomplete
rows, recovered quantiles, unique empirical rows and cycle weight. The
extractor emits no
prompt text or arbitrary source fields, but the input export remains sensitive
and must follow its data policy.

For prompt replay, quantify how many scheduled requests will be repeats. Cache
reuse caused by cycling a short prompt list is a property of the experiment.

The one-command path saves the durable input paths plus
`input_expectations`: exactly one lowercase SHA-256 and byte count for the
profile or prompts file and, when configured, the trace. A rerun captures those
external bytes and refuses before credential or network access if they changed.
The saved config and sealed evidence do not contain raw prompts; the source
dataset must be retained and governed separately if exact reruns are required.

For either mode, run a small endpoint sample and verify:

- endpoint-reported prompt-token size versus intended size;
- endpoint-reported completion size and finish reasons;
- global-cap truncation rate;
- cached-token coverage and paired achieved-versus-intended error;
- acceptable outcome coverage, including tool-call-only results;
- provider tokenizer and request formatting differences.

If these do not match, do not compensate by relabeling the requested profile as
achieved workload. Fix the materialization or use real prompts.

## 4. Establish a guarded fixed-rate ladder

`sweep` directly controls open-loop arrival rate. Start below the expected
knee, use a bounded worker and pending queue, and stop on any unqualified rung.
The example below includes the supported Databricks pay-per-token gate; for a
provisioned endpoint or another provider, do not reuse that snapshot schema and
enforce the applicable limits outside this tool:

```bash
python3 -m traffic_replay sweep \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --endpoint-adapter openai.chat_completions.sse/v1 \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --rate 1,2,4,8 \
  --duration 120 \
  --cooldown 60 \
  --cpt YOUR_PREMEASURED_CHARACTERS_PER_TOKEN \
  --max-concurrency 256 \
  --max-pending-requests 512 \
  --ttft-definition first_visible \
  --ttft-p95 YOUR_TTFT_MS \
  --ttfg-p95 YOUR_TTFG_MS \
  --success-rate YOUR_FRACTION_STRICTLY_BETWEEN_0_AND_1 \
  --rate-limits RATE_LIMITS.json \
  --out-dir results/rate-sweep
```

The example rates are only a low starting ladder, not a recommended capacity
for an unknown endpoint. Pick authorized rungs from known traffic and quota
limits. Measure characters/token once before the ladder; the sweep fixes that
value and sends zero per-rung calibration requests. The 60-second default is
spacing after preflight and between rungs. It does not prove that a provider's
token, request, account quota, burst state, or cache state reset. The sweep is
sequential and therefore stateful.

For Databricks pay-per-token traffic, build `RATE_LIMITS.json` from the current
official
[Foundation Model APIs limits and quotas](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits).
Record the provider fact date as `as_of`, the operator's actual recheck date as
`verified_at`, and a positive `max_age_days`. The gate refuses before paid
inference when review evidence is missing, invalid, future-dated, or older than
that window. Age exactly equal to `max_age_days` remains fresh. It also refuses
when any required demand is unknown or the whole-ladder forecast reaches
`warning_utilization`.

Before authentication or network access, sweep validation constructs the exact
schedule and representative workload for every requested rung. Planning then
includes their union, the two preflight requests, explicitly configured probes,
calibration, offered `max_tokens` reservations, and worst-case physical
attempts. Cooldown is not credited as a quota reset.

Input demand uses a tokenizer-independent engineering bound of one token per
UTF-8 byte of the complete serialized request JSON, plus a harness-defined
64-token allowance for every message and one more 64-token request-level
allowance. The 64-token constants are conservative harness assumptions, not a
Databricks-published tokenizer or chat-framing contract. Roles, message
metadata, model, tools, provider controls, and JSON syntax are included.
Synthetic replay uses the larger of configured characters/token and the
calibration hard ceiling of 12. This supports both frozen prompt messages and
synthetic profiles without trusting intended token counts as an admission
bound.

If the offline budget passes, the command requires control-plane evidence that
the direct route names the configured model, `route_optimized` is exactly
false, every active served entity has the configured name, and each positively
identifies `foundation_model.name=system.ai.<rate_limits.model>`. Merely lacking
provisioned fields is insufficient. The standard quota snapshot also requires
request `service_tier` to be absent or exactly `"default"`; an observed
non-default response tier invalidates the standard-quota comparison. Workspace
tier remains a configured assertion, and unrelated workspace traffic is not
included. A gate pass is harness-budget evidence only, never provider-headroom
proof; a refusal exits with code 3.

After that offline plan and endpoint binding pass, one non-waiting runtime
guard spans preflight/probes, every physical fallback or retry, replay, and all
sweep rungs. Immediately before each `conn.request`, it atomically reserves
exact serialized request bytes, the conservative input bound, offered
`max_tokens`, and one query. Rolling dimensions remain strictly below
`limit * warning_utilization`; `request_bytes_max` is an inclusive per-request
ceiling. A denial or accounting uncertainty permanently trips the guard rather
than waiting for a reset. Reservations are released only when no `POST` could
have begun; otherwise they are conservatively committed at response headers or
an ambiguous transport outcome. Persisted guard scope, per-attempt transitions,
and run-local baseline/final snapshots must reconcile with physical attempts.
This covers only traffic from this command, not other workspace traffic.

For the managed `databricks-glm-5-2` P2T endpoint, the live page last updated
2026-08-07 currently gives 200,000 ITPM, 20,000 OTPM, and 7,200 QPH for an
Enterprise workspace. It pre-reserves requested `max_tokens`; short observed
answers do not retroactively make an unsafe offered load safe. The current
provisioned-throughput architecture list does not list GLM 5.2, so generic PT
limits must not be presented as this endpoint's capacity.

The same current limits page publishes 200 queries/second per Foundation Model
API workspace and a 4 MB request limit. The bundled dated snapshot uses 200
QPS and a conservative 4,000,000-byte serialized-request ceiling because the
page does not specify a decimal or binary MB convention. Recheck both workspace
limits and the GLM-specific row immediately before use.

The illustrative GLM 5.2 canary has a fully disclosed worst-case plan: two
preflight rows, one calibration row, and one measured replay row; no probes are
configured. The default fallback envelope allows at most 12 physical `POST`
attempts. Its illustrative output-budget p50/p95 is 320/480 tokens, which
derives a 720-token request cap. It explicitly selects the managed no-reasoning
path with `{"reasoning_effort":"none"}`. The offline gate reserves a peak
89,142 input tokens/minute and 4,320 output tokens/minute. These are
conservative planned admission values, not observed usage, customer demand, or
a performance/capacity result.

For managed Databricks GLM 5.2, serving-engineering-confirmed evidence establishes
that top-level `{"reasoning_effort":"none"}` disables reasoning and omission
selects maximum reasoning. That behavior is confirmed for both Unity AI
Gateway model service `system.ai.glm-5-2` and the direct managed endpoint. The
current public Databricks
[reasoning-model guide](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-reason-models)
still classifies `databricks-glm-5-2` as reasoning-only and names
`reasoning_effort` without enumerating accepted GLM-specific values. Treat the
off value as serving-engineering-confirmed managed behavior, not as a value independently
enumerated by that guide. Preserve it in `extra_body`, inspect reasoning and
visible-answer evidence, and repeat preflight; HTTP acceptance alone does not
prove that a behavioral control was applied.

The high-level CLI sends numeric `temperature: 0.0` by default. A route that
requires absence must use `--omit-temperature`; omission and zero are distinct
request contracts and must not be pooled or compared as identical.

Databricks documents the Unity AI Gateway
[model-service query API](https://docs.databricks.com/aws/en/ai-gateway/query-model-services)
and its
[destination routing and fallback model](https://docs.databricks.com/aws/en/ai-gateway/model-services).
The harness can serialize the same Chat/SSE protocol to the Gateway route, but
the retained live conformance evidence covers only an exact direct
`/serving-endpoints/.../invocations` route. Gateway is protocol-diagnostic
only: requested FQN-to-destination identity,
routing/fallback pre/post state, and the intersection of Gateway plus
downstream quotas are not bound. Such a run supports no tool quota or capacity
conclusion, and the shipped GLM quota snapshot refuses that route.

Direct SGLang hosting uses the nested native switch documented by its
[GLM-5.2 serving guide](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2):
`{"chat_template_kwargs":{"enable_thinking":false}}`. Thinking is the
SGLang default when that control is absent. With thinking enabled, nested
`reasoning_effort` unset maps to `Max`, `"high"` maps to `High`, and other
values fall through to `Max`; it is not the off switch. Z.ai's hosted
[Chat Completion API](https://docs.z.ai/api-reference/llm/chat-completion)
uses `{"thinking":{"type":"disabled"}}`. Do not copy any request shape across
serving adapters. The illustrative canary above explicitly selects the managed
no-reasoning path. Removing the field selects maximum reasoning and requires a
separately planned workload.

At each rung, inspect external endpoint and quota telemetry as well as the
harness report. Stop when any configured operational guard is breached,
including:

- production impact or operator stop request;
- HTTP error or 429 increase;
- pending-limit drops, queue wait, delivered-rate shortfall, or generator CPU
  and network saturation;
- unacceptable answer outcomes or stream parse errors;
- caller-experienced acceptance-target miss or hard timeout;
- achieved workload drift, missing usage, or global-cap truncation;
- cost or token budget exhaustion;
- endpoint autoscaling or cache state that makes the rung incomparable.

For an operator stop or `KeyboardInterrupt`, the runner signals all workers
and best-effort shuts down tracked active sockets before cancelling queued
work. This wakes blocked reads promptly. It does not cross-thread close the
`HTTPConnection`, because clearing its socket could let a racing request
auto-connect again; the owning worker closes it in `finally`. Clients check the
signal at entry, immediately before a first `POST`, before every retry, and
after transport I/O wakes; a cancellation-induced I/O error is not retried.
Work that has not started a `POST` is suppressed and is best-effort recorded
with zero request attempts. Already-issued traffic cannot be recalled, and its
provider outcome and billing remain ambiguous. The directory remains an
unsealed diagnostic artifact; do not turn it into a completed run manually.

DNS is part of these bounds even though the standard socket timeout does not
bound `getaddrinfo`: a daemon-only, DNS-only single-flight helper stops each
caller at its deadline and cannot make a late connection or `POST`. The
inference deadline covers the full stream. Endpoint metadata and workspace
OAuth M2M additionally use an absolute connection watchdog, so periodic body
bytes cannot keep those operations alive past their configured timeout.

After the ladder finishes, run
`python3 -m traffic_replay verify-sweep results/rate-sweep`. Do not quote a
ceiling if verification reports an incomplete source, unknown request attempt,
per-rung calibration, or a higher pass after a lower failure.

HTTP 429 accounting is row-exact and phase-aware. Only an integer terminal
HTTP `status` of 429 counts; the tool does not infer it from error text or a
redacted body digest. The denominator is every supplied request-operation row
across preflight, probes, sizing, calibration, and replay, and the summary also
records status coverage and per-phase counts. A row may contain multiple
physical attempts, so this is not an attempt-by-attempt status counter;
per-attempt runtime-admission events and `request_attempts` are separate. Any
429 means quota
`EXCEEDED`, measurement `INVALID`, and endpoint capacity `INCONCLUSIVE`. It
still cannot identify whether input tokens, output reservations, queries,
account policy, an edge component, or another limit caused the rejection.
Determine that from provider telemetry. Zero 429s means only “not observed,”
not quota headroom.

## 5. Run a long confirmation at one fixed condition

After a valid ladder identifies a candidate rate, run one configuration long
enough to observe multiple 60-second stability windows. Five minutes produces
five nominal windows, but setup and drain remain outside the schedule.

Copy an example config, replace all placeholders, set one measured workload,
and retain conservative client bounds. In the filename
`configs/run_pt_full.json`, `pt` means provisioned throughput, not
pay-per-token. The template starts at `rate_scale=0.1`,
`max_concurrency=256`, and
`max_pending_requests=512`; it is not permission to raise `rate_scale` to 1.0
on one process. Size client resources from measured mean occupancy and validate
delivery at every step. The template has client bounds but deliberately has no
`rate_limits` object or provider-capacity guard; enforce the applicable
provisioned allocation and account limits externally.

Little's Law uses mean occupancy:

```text
mean in flight approximately arrival rate * mean end-to-end service time
```

Do not substitute p95 latency into this equality. Tail latency is useful for
headroom and timeout analysis, but it does not calculate mean concurrency.
Also include connection setup, retries, fallbacks, response drain, and local
resource limits in the client capacity plan.

Run the lower-level config path when every setting must be explicit:

```bash
python3 -m traffic_replay run \
  --config configs/run_pt_full.json \
  --format json
```

The run exits nonzero on a default miss/invalid gate. Save stdout, stderr, the
exact config, source commit, and sealed output directory.

## 6. Shard only when one generator is proven insufficient

Sharding adds clock, identity, and aggregation failure modes. First demonstrate
with external host telemetry and harness queue/delivery evidence that one
generator cannot safely deliver the authorized rate.

Every shard needs the same future `start_at_unix`, `run_id`, workload, seed,
request parameters, and `shard_total`, plus a unique zero-based `shard_index`.
Synchronize clocks. The shared start must allow enough setup time for config
validation, target evidence, optional sizing, schedule generation, and
calibration.

Do not combine independent sizing passes into a sharded production claim. For
a reproducible shard test, establish the fixed global rate first, put it in the
shared config, split the schedule, and divide client resources deliberately.

After all shards seal successfully:

```bash
python3 -m traffic_replay merge results/merged \
  results/shard-0/RUN_DIR \
  results/shard-1/RUN_DIR
```

The merge must prove a complete nonoverlapping global-index set. Never use
`--force` to turn missing coverage into a capacity result; forced output is
explicitly INVALID. Exact monotonic caller durations present in source rows
are pooled and can drive merged acceptance-target scoring. Legacy
schedule/send timestamps
are not reconstructed across different run epochs; when exact caller clocks
are absent, the merge labels acceptance scoring as service-time only. Read each shard
for stability, wire lateness, and in-flight concurrency because those time axes
are not pooled.

## 7. Review and sign off evidence

The customer-facing files intentionally present five independent decision
dimensions rather than one combined verdict:

| Dimension | Decision codes |
|---|---|
| Evidence integrity | `VERIFIED`, `VERIFY_REQUIRED`, `TAMPERED` |
| Measurement validity | `VALID`, `CAUTION`, `INVALID` |
| Acceptance checks | `PASS`, `MISS`, `INCONCLUSIVE`, `NOT_EVALUATED` |
| Quota state | `EXCEEDED`, `LOCAL_GUARD_REFUSED`, `NOT_OBSERVED`, `UNKNOWN`, `NOT_EVALUATED` |
| Endpoint capacity | `HELD_AT_TESTED_LOAD`, `NOT_HELD_AT_TESTED_LOAD`, `INCONCLUSIVE`, `NOT_EVALUATED` |

`summary.json` is canonical; `report.html` and `report.md` render the same
codes, labels, reasons, and tested-load facts. A freshly written report says
`VERIFY_REQUIRED` because it cannot authenticate the manifest that encloses
it. Verify the completed marker/manifest chain externally; do not edit the
sealed report to manufacture `VERIFIED`. A qualified or invalid measurement
can retain the observed acceptance-check result, but a retained pass is
explicitly not a clean acceptance pass. `HELD_AT_TESTED_LOAD` never means
endpoint ceiling or provider headroom.

Response identity, the normalized pre-run/post-drain endpoint-stability
comparison, and runtime-admission reconciliation are evidence gates, not three
additional decision dimensions. Identity and stability feed measurement
validity; runtime admission feeds quota state and measurement validity. The
canonical report remains exactly the five dimensions above.

Treat `LOCAL_GUARD_REFUSED` separately from HTTP 429. It means the local
command-scoped guard suppressed a physical `POST`, so the requested load was
not delivered; it is not endpoint-capacity evidence. A clean quota review must
also reconcile guard scope, every per-attempt admission transition, the
run-local baseline/final snapshots, and physical-attempt counts. Inconsistent
guard evidence makes quota unknown and measurement invalid.

Review response and control-plane identity before latency. Multiple response
models invalidate a single-model benchmark. One stable response-model name
that differs from an explicit request-body model produces an
unverified-identity caution; alias or revision naming can differ, so the
string mismatch alone is not proof that another model served the request. The
endpoint name is not an expected OpenAI response-model value for a custom/PT
route. Bind each Databricks `served-model-name` response header to an active
served entity captured from the control plane; an unexpected entity or route
contradiction invalidates the result and incomplete binding is a caution.
Response IDs are hashed; response objects and system fingerprints are bounded
context. With endpoint capture enabled, compare
the normalized metadata summary captured before runner-owned sizing,
calibration, and replay traffic to the summary captured only after every
response drains. A change invalidates the single-configuration result; missing
either capture is explicit uncertainty. The compared summary is a selected
subset: endpoint name, task, `route_optimized`, READY state, and selected active
served-entity identity, foundation-model, workload/provisioning, version, and
scale-to-zero fields. It does not cover every control-plane field or prove an
undocumented data-plane revision stayed fixed.

Stability is computed after drain from persisted replay rows. Window latency
uses the same acceptable-outcome population as headline latency; failures and
unacceptable outcomes remain separate window errors. A failure-only window has
zero event coverage rather than a percentile. Do not accept a stable survivor
p95 while error rate rises or event coverage falls.

Create that external evidence as a separate sibling receipt. Replace the
example timestamp with the exact completed run directory:

```bash
python3 -m traffic_replay verify-run \
  results/benchmark/20260807-183532 \
  --out results/benchmark/20260807-183532-verification
```

The command refuses an output inside the run or outside the run's parent. It
never modifies the source run, never follows a symlink in place of a required
regular artifact, and never overwrites an existing receipt; a collision gets a
unique sibling suffix. It verifies the v3 completion/manifest chain, all
manifest-declared artifacts, strict start/summary/request JSON, and the
summary-to-journal counts plus replay schedule/index identities. It then
rereads the source around generation of
`verified-report.html` and `verified-report.md` and binds those files with
`verification.json` in a separately completed v3 receipt.

Exit 0 means the completed receipt passed its own manifest/completion reopen.
Exit 2 means the source or receipt failed verification or receipt creation;
do not use a directory that lacks `.traffic-replay-complete`. A failed write
may retain `.traffic-replay-writing` as diagnostic evidence. Use `--format
json` when automation needs the receipt path and decision object.

At the top of each verified receipt report, read the states independently:

- `Integrity: VERIFIED` means the internal SHA-256 byte bindings and semantic
  cross-checks agreed;
- `Source reproducibility: PASS/FAILED` states whether the recorded source was
  clean, complete, and internally consistent;
- `Verifier reproducibility: PASS/FAILED` states whether the external verifier
  itself ran from a clean recorded source identity.

A reproducibility failure is shown explicitly and prevents a positive
held-capacity conclusion; it does not erase the separately observed load and
latency facts. The receipt is not a digital signature and proves neither
authorship nor trusted time. It does not fetch a recorded commit to establish
availability and cannot prevent later mutation. Preserve the immutable source
run and sibling receipt together. A printed or exported PDF remains a
derivative; use the receipt manifest and completion marker as the evidence
chain.

The HTML is a self-contained, no-script/no-remote-asset view with responsive
cards, locally scrollable dense tables, chart text equivalents, expandable
decision reasons, and browser print rules for ordinary A4/Letter output.
Printing can still add browser-controlled headers, footers, margins, and page
breaks. Markdown is the portable textual view. Review both from the sealed
directory, not an exported screenshot alone.

The print view is stamped `UNSEALED PRINT/PDF DERIVATIVE`. Browser PDF is not
part of the source manifest and internal hashes are not a digital signature.

Accept a result only if all of the following are true:

- `.traffic-replay-complete` exists and `.traffic-replay-writing` does not;
- manifest schema is 3 and every bound artifact hash, byte count, and request
  row count verifies;
- the completion marker's artifact ID, manifest digest and byte count, and
  request-row count match the manifest-bound summary and journal;
- source state is clean and the commit is retained;
- endpoint, workload, request, schedule, execution, and artifact identities
  match the experiment record;
- runtime quota admission is either not configured or fully reconciled with no
  denial, invariant error, or unexplained physical attempt;
- response model is consistently bound where reported, and pre/post-drain
  endpoint metadata is stable or any missing coverage is explicitly resolved;
- scheduled and delivered load match within the documented acceptance policy;
- no pending-limit drops or unexplained physical duplicate attempts exist;
- acceptable answer/tool outcome and usage coverage are sufficient;
- achieved token and cached-token distributions match the intended workload;
- exact caller timing covers the scored population;
- the quoted quantile meets its sample floor: p50 20, p90 100, p95 200, p99
  1000 acceptable outcomes;
- the observed success fraction meets its target and the one-sided 95 percent
  Wilson lower confidence bound also meets it, under the stated independent
  outcome assumption;
- stability is established over the acceptable-outcome population while
  window error rate and event coverage remain healthy;
- cost is explicitly labeled unverified operator-supplied arithmetic over
  replay rows only, and aggregate per-token totals plus provisioned effective
  rates are withheld for missing/invalid usage, corrupt or incomplete streams,
  unknown attempt counts, or any ambiguous retry/multiple-POST row;
- external endpoint, quota, and generator telemetry agree with the conclusion.

An interrupted directory can retain useful newline-complete rows in
`requests.jsonl.partial`, but it is diagnostic evidence only. Do not rename the
journal or manufacture a completion marker.

Pricing is never fetched or bound to a provider contract by this tool.
Preflight, probe, sizing, and calibration traffic is outside the per-token cost
block. A replay aggregate or provisioned effective rate is available only when
every row is either known unsent or has one clean, complete, sane-usage result
from exactly one physical attempt with no retry marker. Otherwise only the
valid measured subset is shown; total, per-1,000-request, per-minute,
cache-savings, effective-rate, and token-throughput-denominator figures are
withheld. Reconcile the full bill, including ambiguous attempts and setup
traffic, with provider billing telemetry.

## 8. Compare providers or serving products cautiously

Keep the real prompts or workload profile, schedule identity, seed, output cap,
TTFT definition, acceptance policy, and load mode fixed. Then verify achieved
prompt, output, and cached-token distributions rather than assuming identical
request JSON creates identical work.

Provider routes can differ in tokenizer, chat template, reasoning controls,
tool-call framing, cache accounting, usage coverage, fallback behavior,
autoscaling, and quotas. `compare` requires sealed, integrity-verified inputs
and checks configured compatibility. A mismatch produces a sealed but
prominently INVALID diagnostic comparison rather than a winner. The comparison
directory contains both `comparison.html` and `comparison.md`; manifest schema
v3 binds both rendered files and the exact source manifest/summary identities.
The writer promotes its completion marker last and verifies the completed
artifact before returning, but the current CLI has no standalone
`verify-comparison` command. The source runs remain the underlying benchmark
evidence. Even a valid comparison cannot certify semantic or protocol
equivalence.

Input order defines the baseline: the first run is the baseline and every
later run is a candidate. HTML absolute delta is candidate minus baseline;
percent delta divides by the absolute baseline and is undefined when the
baseline is zero. A valid comparison labels arithmetic direction only; it
does not claim improvement/regression without repeat-run uncertainty and a
practical-effect threshold. Measurement warnings make the comparison
`QUALIFIED` and diagnostic-only. Compatibility/source-validity failures make
it `INVALID`. Both retain neutral diagnostic numbers and must not rank
endpoints.
Markdown carries portable side-by-side absolute values and warnings; HTML adds
the explicit delta matrix and is self-contained, responsive, and printable in
landscape with scripts and remote requests blocked.

Comparison rescans each manifest-bound request journal for 429s across every
phase. Any 429, summary/journal disagreement, explicitly invalid source, or
compatibility failure makes the comparison diagnostic-only even though its
bytes can still be correctly sealed.

Provisioned and pay-per-token products answer different capacity and cost
questions. Compare them only when the exact model is supported by both and the
claim states the product, allocated capacity, utilization, and current pricing
basis.
