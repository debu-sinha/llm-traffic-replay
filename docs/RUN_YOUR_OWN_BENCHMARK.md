# Benchmark your own endpoint

This is the shortest defensible path from an endpoint to a sealed report. For
an authorized production test, also follow
[Production testing](PRODUCTION_TESTING.md).

## 1. Install and validate

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest
python3 -m traffic_replay validate --port 0 --format json
```

The test suite is required for the full regression check. `validate` tests the
local timing pipeline against a bundled oracle; it does not test a provider.

## 2. Choose authentication

Named Databricks profile:

```bash
databricks auth profiles
```

The selected profile must contain a host matching the endpoint origin. A
legacy PAT profile uses `token` and may set `auth_type=pat`. A CLI-cached OAuth
U2M profile must set `auth_type=databricks-cli`; the harness then invokes
`databricks auth token -p PROFILE`. A workspace OAuth M2M profile uses
`client_id` and `client_secret` and may omit `auth_type` (the official profile
shape) or set it to `oauth-m2m`. The harness exchanges those credentials at
the matching workspace's `/oidc/v1/token` endpoint with `scope=all-apis`.

For example, an unattended standard workspace-origin run can select:

```ini
[load-test-m2m]
host = https://YOUR-WORKSPACE-HOST
client_id = YOUR-SERVICE-PRINCIPAL-CLIENT-ID
client_secret = YOUR-SERVICE-PRINCIPAL-OAUTH-SECRET
```

Protect that file and rotate the secret under your organization's credential
policy. Never place the client secret in a run config, command argument, or
report label. Profile resolution rejects mixed, incomplete, unsupported, or
host-mismatched credentials and never falls back to environment credentials.

The direct M2M implementation is for the standard workspace-origin invocation
route. It does not create the endpoint-scoped `authorization_details` token
required by route-optimized serving URLs. See Databricks'
[OAuth M2M guide](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m),
[`auth token` reference](https://docs.databricks.com/aws/en/dev-tools/cli/reference/auth-commands#databricks-auth-token),
and
[route-optimized authentication guide](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-route-optimization).
Treat PAT authentication as legacy.

Environment token:

```bash
export DATABRICKS_TOKEN='...'
```

Do not put a token in JSON or in a report label. Remote bearer-authenticated
HTTP is rejected; use HTTPS. Explicit loopback HTTP is reserved for local
tests.

The measured `extra_body` is persisted as reproducibility evidence, and probe
candidates/outcomes are reported with displayed values and errors
credential-redacted. Secret-like keys and credential-shaped values are
rejected recursively before config output or traffic. Command arguments can
still be visible locally, so keep authentication in the profile or token
environment variable.

## 3. Choose real prompts or a measured profile

Real prompts give the strongest content fidelity:

```jsonl
{"messages":[{"role":"user","content":"Your approved test prompt"}]}
```

Save one JSON value per line, then pass `--prompts prompts.jsonl`.

If only token/cache logs are available, generate a profile. Joint mode retains
observed combinations and frequencies without emitting prompt content:

```bash
python3 scripts/profile_from_logs.py \
  --input request_metrics.jsonl \
  --name measured_workload \
  --mode empirical-joint \
  --out configs/profile_measured.json

python3 -m traffic_replay sample \
  --profile configs/profile_measured.json --n 50000 --seed 7
```

The output records extraction counts plus the byte count and SHA-256 of the
exact frozen source bytes it parsed. Review dropped rows and recovered anchors.
If you use legacy `--mode quantiles`, the output is a v1
p50/p95 marginal model and does not preserve p90/p99 or cross-field
dependence.

When constructing a placeholder profile directly from CLI size flags, use
`--cache-fraction P50,P95` for the intended reusable-prefix share of prompt
tokens. This is not a request cache-hit probability. `--cache-hit-rate` remains
only as a compatibility alias with the same token-share meaning.

Always pass both values for production-oriented CLI profiles. A single input
or output value is treated as p50 and derives p95 as 2.4 times p50. A single
interior cache fraction derives p95 as `p50 + 0.65 * (1 - p50)` (exact 0 or 1
stays constant), and the high-level command derives the output cap as
`ceil(output_p95 * 1.5)`. Those are convenience assumptions, not measured
tails, targets, or provider recommendations.

## 4. Run preflight and a small benchmark

## Guided workload and customer SLA setup

Use `traffic_replay init-config` to turn a common Databricks/OpenAI telemetry
export and plain-language service expectations into separately owned
`workload-profile.json`, `customer-sla.json`, a timestamp schedule, and
`run-config.json`. Omit values in a terminal to be prompted, or supply every
flag for automation. Use `--provider custom` with `--input-field`,
`--output-field`, and `--cached-field` for another export schema.

The generated configuration selects `first_visible`: "response starts" means
the customer sees assistant answer content. Cache fraction means the share of
prompt tokens intended for reuse, not the percentage of requests with any
cache hit. The SLA file records the required customer-owned source and hard
abandonment limits. The run config references that file. If an expert also
adds inline `acceptance_targets`, the inline object explicitly overrides the
referenced SLA and both `run` and `check-config` print that precedence.

`traffic_replay check-config --config PATH` validates and explains the setup
without credentials, endpoint discovery, or inference traffic. It prints
modeled p50/p90/p95/p99 token values, extraction/drop counts, modeling choice,
the SLA in ordinary language, exact request-phase counts, estimated tokens,
and either a cost status or `COST NOT CALCULATED`.

## Run with an existing profile

With a profile:

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --endpoint-adapter openai.chat_completions.sse/v1 \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --sizing-concurrency 2 \
  --duration 60 \
  --calibrate-requests YOUR_AUTHORIZED_CALIBRATION_REQUESTS \
  --ttft-definition first_visible \
  --ttft-p95 YOUR_TTFT_MS \
  --ttfg-p95 YOUR_TTFG_MS \
  --success-rate YOUR_FRACTION_STRICTLY_BETWEEN_0_AND_1 \
  --out-dir results/first-run
```

With prompts, replace `--profile ...` with `--prompts prompts.jsonl`.

`--calibrate-requests` accepts an integer from 0 through 10,000; the actual
count is capped by the schedule. These are separate, billable pre-replay
requests. With clean usage, profile mode can use them to correct its
characters-per-token estimate. Prompt mode does not retokenize the supplied
messages but still sends the configured calibration rows. Any positive count
warms endpoint/model state and may repeat an exact replay payload; zero avoids
harness calibration only and is not proof of a cold cache or cold worker.

Before resolving credentials or making a network connection, the command
copies the workload and optional trace to private temporary bytes, strictly
parses that exact view, and constructs representative bodies. A fixed-rate or
trace-driven run also materializes its complete schedule at this boundary. The
sizing example above is the explicit exception: the paid sizing sample must
derive a rate before the schedule can exist. `sweep` constructs and validates
every exact requested rung before endpoint access.

Preflight sends two representative inference requests. After both reach HTTP
200, if either lacks an acceptable outcome, explicitly supplied
`--probe-extra-body` candidates can each send one additional request. The
harness does not guess provider controls. All are real traffic. An acceptable
outcome has no refusal marker and contains visible content or a
structurally valid tool call with a nonempty function name and arguments that
decode to a JSON object, plus clean stream completion and no parse errors; it
is not a correctness grade.

Before the first preflight or probe `POST`, the CLI claims a separate sibling
artifact at `results/first-run-setup-traffic/TIMESTAMP` and fsyncs every
completed metadata-only row. A normal pass or refusal seals it as an explicit
non-performance/non-SLA/non-capacity result; a crash leaves incomplete
diagnostic evidence. When preflight passes, those rows are also attached once
to the measured run journal without request or response content. With
`--force`, an unreadable gate is instead recorded as
`preflight_forced_unreadable`; it is never called passed, its rows are still
attached, and the resulting run or sweep is INVALID diagnostic evidence. They
participate in its quota population but not replay acceptance percentiles.
Capture stdout and stderr as well when the human-readable gate decision needs
an audit trail.

`--sizing-concurrency 2` measures unloaded service time and derives one fixed
open-loop rate plus a worker pool. When the worker cap is omitted, the derived
pool is capped at the default 256-thread safety ceiling; an explicit positive
cap replaces that ceiling. It does not hold two requests in flight. The
measured report shows the concurrency that resulted.

### Databricks pay-per-token: enable the paid-run gate

Rate limits change by model, deployment mode, and workspace tier. Recheck the
official
[Foundation Model APIs limits and quotas](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits),
write the exact current facts and source URL to an owned `RATE_LIMITS.json`,
and set both `verified_at` and `max_age_days`. `as_of` is the date of the
provider fact; `verified_at` is the date you actually rechecked it.

A quota-aware benchmark cannot use `--sizing-concurrency`, because the paid
sizing requests would occur before the schedule is known. Use a small,
authorized fixed rate instead:

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-PAY-PER-TOKEN-ENDPOINT \
  --endpoint-adapter openai.chat_completions.sse/v1 \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --fixed-rate YOUR_AUTHORIZED_REQUESTS_PER_SECOND \
  --duration 60 \
  --calibrate-requests YOUR_AUTHORIZED_CALIBRATION_REQUESTS \
  --ttft-definition first_visible \
  --rate-limits RATE_LIMITS.json \
  --out-dir results/p2t-smoke
```

Before paid inference, the gate requires a fresh snapshot and every configured
limit dimension to be bounded below `warning_utilization`. Input is bounded at
one token per UTF-8 byte of the complete serialized request JSON plus a
harness-defined 64-token allowance for each message and one additional
64-token request allowance. Those 64-token constants are conservative
engineering assumptions, not a Databricks-published tokenizer or framing
contract.
Roles, message metadata, model, tools, provider controls, and JSON syntax are
included. Synthetic replay uses the larger of configured characters/token and
the calibration hard ceiling of 12. Prompt mode is therefore bounded from the
complete request built from its exact frozen messages without trusting the
provider tokenizer or intended profile token counts.

The exact offered `max_tokens` reservation is budgeted, as are worst-case
physical attempts, preflight/probes, calibration, and replay. Control-plane
binding requires the direct route, `route_optimized=false`, matching endpoint
and served-entity names, and positive
`foundation_model.name=system.ai.<rate_limits.model>` identity for every active
entity. Absence of provisioned fields is not enough by itself. Request
`service_tier` must be absent or exactly `"default"` for this standard quota
model; an observed non-default response tier invalidates the comparison.
Missing or stale freshness, an unbounded configured dimension, threshold
contact, or incomplete endpoint binding refuses with exit code 3. The workspace
tier remains an operator assertion and unrelated workspace traffic is not
visible, so a pass is not proof of provider headroom.

After the offline plan and endpoint binding pass, one non-waiting runtime guard
spans the entire command: preflight/probes, every physical fallback or retry,
replay, and every sweep rung. Immediately before each `conn.request`, it
atomically reserves exact serialized request bytes, the conservative input
bound, offered `max_tokens`, and one query. Rolling dimensions remain strictly
below `limit * warning_utilization`; `request_bytes_max` is an inclusive
per-request ceiling. A denial or accounting uncertainty permanently trips the
guard instead of sleeping for a reset. Reservations are released only when no
`POST` could have begun; otherwise they are conservatively committed at
response headers or an ambiguous transport outcome. Persisted per-attempt
events and run-local baseline/final snapshots must reconcile with physical
attempt counts. This protects only traffic from this command and cannot see
other workspace traffic or prove provider headroom.

For `databricks-glm-5-2`, the live official limits page last updated
2026-08-07 currently states 200,000 ITPM, 20,000 OTPM, and 7,200 QPH for an
Enterprise pay-per-token workspace. Use the live row rather than a cached
search snippet. Databricks reserves requested `max_tokens` for admission and
later credits unused output reservation back. Do not label this managed P2T
endpoint as provisioned throughput: the current PT architecture list does not
list GLM 5.2.

The same current limits page publishes 200 queries/second per Foundation Model
API workspace and a 4 MB request limit. The bundled dated snapshot uses 200
QPS and a conservative 4,000,000-byte serialized-request ceiling because the
page does not specify a decimal or binary MB convention. Recheck both workspace
limits and the GLM-specific row immediately before use.

The illustrative GLM 5.2 canary's worst-case plan is exactly two preflight
rows, one calibration row, and one measured replay row, with no probes. The
default fallback envelope permits at most 12 physical `POST` attempts. The
illustrative output-budget p50/p95 is 320/480 tokens, which derives a 720-token
request cap. The command explicitly selects the managed no-reasoning path with
`{"reasoning_effort":"none"}`. The offline plan reserves a peak 89,142 input
tokens/minute and 4,320 output tokens/minute. These are conservative planned
admission values, not observed usage, customer demand, performance, or capacity.

## 5. Handle reasoning controls explicitly

When the endpoint emits reasoning before visible content, decide what the
configured latency target means:

- `first_content`: first visible, reasoning, or refusal delta;
- `first_visible`: first meaningful visible assistant content.

Tool-call-only outcomes have a separate time-to-first-tool-call metric.
Final-attempt clocks begin immediately before `conn.request`, include request
upload, and exclude connection setup. TTFB means the first nonempty bounded
response-body chunk returned by the client read, not the first socket byte or
first parsed SSE line. `ttse_ms` separately records the first complete framed
event emitted by the selected adapter parser. That event is not necessarily a
token or content: it can be usage-only, terminal, or a content-free parse
diagnostic. A tool-call fragment does not trigger TTFT;
first-visible and first-tool-call timings stay separate.

`interchunk_max_ms` is the widest elapsed gap between successive SSE events
that contain a nonempty visible, reasoning, or refusal delta. It is not
token-level inter-token latency: an event can batch tokens, heartbeats and
usage-only events are excluded, and tool-call-only fragments are excluded. A
protocol-clean request with fewer than two qualifying events has no interchunk
measurement; if an interchunk target is configured, any such row makes that
check inconclusive.

For managed Databricks GLM 5.2, serving-engineering-confirmed evidence establishes
that top-level `reasoning_effort="none"` disables reasoning. Leaving the field
unset selects maximum reasoning. This behavior is confirmed for both Unity AI
Gateway model service `system.ai.glm-5-2` and the direct managed endpoint. Seal
the selected behavior in the measured configuration:

```bash
--extra-body '{"reasoning_effort":"none"}'
```

The current Databricks
[reasoning-model guide](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-reason-models)
classifies `databricks-glm-5-2` as reasoning-only and names
`reasoning_effort`, but does not enumerate the accepted GLM-specific values.
Consequently, `"none"` is serving-engineering-confirmed managed behavior rather than a value
independently enumerated by the public guide. A successful HTTP status still
does not prove the requested behavior was applied. Require a completed answer,
inspect reasoning evidence, and pass the full two-representative preflight
with the selected control.

High-level commands send numeric `temperature: 0.0` by default. Use
`--omit-temperature` only when the exact route requires omission and preserve
that choice as a distinct run configuration; `temperature` is adapter-owned
and cannot be supplied through `--extra-body`.

Request compatibility is broader than this tool's production qualification.
Databricks documents the Unity AI Gateway
[model-service query API](https://docs.databricks.com/aws/en/ai-gateway/query-model-services)
and that
[model services can route and fall back across destinations](https://docs.databricks.com/aws/en/ai-gateway/model-services).
The harness can POST the Gateway protocol, but this release qualifies only the
exact direct `/serving-endpoints/.../invocations` route. Gateway is
protocol-diagnostic only: requested model-service FQN, destinations, routing/fallback,
pre/post state, and the intersection of Gateway and downstream quotas are not
bound. A Gateway run supports no tool quota or capacity conclusion, and the
shipped GLM rate-limit snapshot refuses that route.

Direct SGLang hosting uses a different native switch. The
[SGLang GLM-5.2 guide](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2)
documents thinking as the default and turns it off with:

```bash
--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

With SGLang thinking enabled, nested `reasoning_effort` unset means `Max` and
`"high"` means `High`; `"low"`, `"medium"`, and other values fall through to
`Max`. `reasoning_effort` is not SGLang's thinking-off switch.

Z.ai's hosted
[Chat Completion API](https://docs.z.ai/api-reference/llm/chat-completion)
has another request shape, `{"thinking":{"type":"disabled"}}`. Do not
transfer controls across serving adapters. The illustrative canary above
explicitly selects the managed no-reasoning path. Removing the field selects
maximum reasoning and changes the
workload plus quota plan.

To test multiple documented candidates after an unreadable preflight, repeat
`--probe-extra-body '{...}'`. Each candidate is another real request before
the sealed runner begins, so use this only as an authorized, explicit probe.
Candidates do not change the measured config. Rerun with a successful object
as `--extra-body`.

Do not solve a reasoning-heavy result by silently raising the output cap. A
larger cap changes cost and work. If the product requires it, update the
workload contract and rerun from preflight.

## 6. Read the sealed evidence

Open `report.html` for navigation, but treat `summary.json`,
`requests.jsonl`, `start.json`, and manifest schema v3 as the audit record.
`summary.json` is also the canonical decision source. `report.html` and
`report.md` render the same codes, labels, reasons, and tested-load facts for
five independent dimensions:

| Dimension | Codes |
|---|---|
| Evidence integrity | `VERIFIED`, `VERIFY_REQUIRED`, `TAMPERED` |
| Measurement validity | `VALID`, `CAUTION`, `INVALID` |
| Acceptance checks | `PASS`, `MISS`, `INCONCLUSIVE`, `NOT_EVALUATED` |
| Quota state | `EXCEEDED`, `LOCAL_GUARD_REFUSED`, `NOT_OBSERVED`, `UNKNOWN`, `NOT_EVALUATED` |
| Endpoint capacity | `HELD_AT_TESTED_LOAD`, `NOT_HELD_AT_TESTED_LOAD`, `INCONCLUSIVE`, `NOT_EVALUATED` |

Do not collapse those dimensions into one verdict. A 429 can coexist with an
observed acceptance-check outcome, but it makes measurement validity `INVALID`
and capacity `INCONCLUSIVE`; a retained acceptance `PASS` is visibly qualified.
Even `HELD_AT_TESTED_LOAD` is not an endpoint ceiling or a provider-headroom
claim.

Response identity, the normalized pre-run/post-drain endpoint-stability
comparison, and runtime-admission reconciliation are evidence gates rather
than three additional decisions. Identity and stability feed measurement
validity; runtime admission feeds quota state and measurement validity. The
canonical report remains exactly the five dimensions above.

The freshly written files say `VERIFY_REQUIRED` because they cannot verify the
manifest that encloses them. Preserve and verify the complete directory rather
than editing that state in place.

The HTML is self-contained: inline CSS/SVG, no JavaScript, remote fonts,
assets, or network fetches. It adapts cards and charts for narrow screens,
keeps dense tables within horizontal scrollers, exposes full decision reasons,
and provides print rules for ordinary A4/Letter browser output. Browser-added
headers, footers, margins, and pagination choices remain external. Markdown is
the portable text view; its layout differs but its decision semantics do not.
Browser-print/PDF output is stamped `UNSEALED PRINT/PDF DERIVATIVE`; it is a
convenience rendering, not a bound artifact or digital signature. Verify the
source directory and manifest.

HTTP 429 is counted from the numeric terminal status on each supplied
request-operation row used for quota evidence, including preflight, probes,
sizing, calibration, and replay. The summary carries the exact count, row
denominator, status coverage, and phase breakdown. One row can contain multiple
physical attempts, so this is not an attempt-by-attempt status counter;
per-attempt runtime-admission events and `request_attempts` are separate. A 429
proves a rejection but not its quota dimension, enforcing component, or the
endpoint compute ceiling. Zero with complete coverage means only
`NOT_OBSERVED`; incomplete coverage means `UNKNOWN`.

`LOCAL_GUARD_REFUSED` is different from HTTP 429: the command-local admission
guard suppressed a physical `POST`, so the requested load was not delivered.
It is a harness safety stop, not endpoint-capacity evidence. Missing or
inconsistent per-attempt guard events, scope/baseline/final snapshots, or
physical-attempt reconciliation makes quota state unknown and measurement
invalid.

For eligible streamed responses, inspect response-model and routed-entity
coverage. Multiple response-model values invalidate a single-model benchmark.
One stable response-model name that differs from an explicit request-body
model produces an unverified-identity caution; alias or revision naming can
differ, so the mismatch alone is not proof that another model served the
request. Do not compare the OpenAI response `model` to the serving endpoint
name: custom/PT endpoints can legitimately return an underlying model name.
The Databricks `served-model-name` response header is checked against active
served entities captured from the control plane; an unexpected entity or
route contradiction invalidates the result and incomplete binding is a
caution. Response IDs are stored only as SHA-256; `object` and
`system_fingerprint` are bounded context.

When endpoint metadata capture is enabled, the runner compares a snapshot from
before its own sizing, calibration, and replay traffic with a second snapshot
taken only after response drain. A change in the normalized metadata summary
invalidates a single-configuration benchmark; missing either capture is
explicit uncertainty. That summary is a selected subset: endpoint name, task,
`route_optimized`, READY state, and selected active served-entity identity,
foundation-model, workload/provisioning, version, and scale-to-zero fields. The
two reads do not compare every control-plane field or prove that an
undocumented data-plane revision stayed fixed between them.

Read in this order:

1. completion and artifact integrity;
2. runtime admission, HTTP statuses, response identity, endpoint metadata
   stability, acceptable outcomes, failures, and physical request attempts;
3. delivered rate, pending-limit drops, queue wait, and measured concurrency;
4. achieved prompt/output sizes and cached prompt-token fraction;
5. caller-experienced acceptance metrics and coverage;
6. sample-size and stability cautions;
7. final-attempt request-path clocks, usage throughput, and cost.

Stability windows are derived after drain from persisted replay rows. Their
latency population is the same acceptable-outcome population as the headline
latency; failures and unacceptable outcomes remain separate window errors. A
failure-only window has zero event coverage, not a latency percentile. Do not
call survivor p95 stable while error rate rises or event coverage falls.

Minimum answer-latency sample floors are 20 for p50, 100 for p90, 200 for p95,
and 1000 for p99. Smaller samples remain diagnostic only. Clean success-rate
and latency-target verdicts also require the one-sided 95 percent Wilson lower
confidence bound on their compliance fraction, not just the observed point
estimate, to meet the target. That inference assumes independent outcomes.

`NOT REPORTED` for cached tokens means missing endpoint evidence, not a zero
cache rate. Reasoning stream deltas are counts of SSE deltas, not tokens.

An incomplete or parse-corrupt current stream is a failed request even after
HTTP 200 or an earlier content delta. It stays in error/success denominators but
is excluded from answer latency, calibration, token throughput, cache fidelity,
and cost arithmetic.

Cost is unverified operator-supplied rate arithmetic, never fetched pricing or
an invoice. The per-token block covers replay rows only. Aggregate per-token
totals and provisioned effective rates are withheld if any replay row has
ambiguous retries or multiple physical `POST`s, unknown attempt accounting,
corrupt/incomplete streaming, or missing/invalid usage. A provisioned rate's
token-throughput denominator is exact only under that same physical-attempt
gate. Preflight, probes, sizing, and calibration require separate billing
reconciliation.

The network card is a best-effort diagnostic. After resolving DNS, it records
`tcp_connect_min_ms` and `tcp_connect_median_ms`. Neither is an exact RTT or
endpoint time, and neither should be subtracted from TTFT.

DNS resolution itself is bounded even when the system resolver stalls.
Concurrent identical lookups share one daemon resolver operation, active
unique lookups are capped, and a caller timeout or cancellation cannot cause a
late socket connection or `POST`. The inference, endpoint-metadata, and OAuth
M2M paths also enforce an absolute operation deadline across DNS, connect,
headers, and the response body; repeated tiny response chunks do not reset
that deadline. A deadline failure is harness/network evidence, not an endpoint
capacity result.

## 7. Rerun the exact config

The one-command path writes:

```text
results/.traffic-replay-configs/profiles/SHA256/profile.json
                                      immutable generated profile, when needed
results/.traffic-replay-configs/runs/SHA256/run-config.json
                                      immutable rerunnable config
results/first-run-setup-traffic/RUN_DIR/
                                      sealed preflight/probe non-result, when enabled
results/first-run/profile.json        first-run compatibility copy, never replaced
results/first-run/run-config.json     first-run compatibility copy, never replaced
results/first-run/RUN_DIR/            sealed measured artifacts
```

Use the exact immutable path printed by `benchmark`, for example:

```bash
python3 -m traffic_replay run \
  --config results/.traffic-replay-configs/runs/SHA256/run-config.json \
  --format json
```

Do not assume the compatibility filename belongs to the latest invocation;
the command preserves an earlier copy rather than racing or overwriting it.

The immutable run config retains the original durable workload/trace paths plus
an `input_expectations` map containing only SHA-256 and byte count for each
configured input. It does not embed raw prompt content. A rerun snapshots the
external bytes and refuses before credential or network access if either the
digest or byte count changed; intentionally changed data needs a newly
generated config. A new execution is still a new experiment: the same seed
reproduces the client plan, not endpoint, network, autoscaling, or cache state.

## 8. Identify the highest held tested rung

After the small run is valid:

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
  --rate-limits RATE_LIMITS.json
```

Choose authorized rates based on known traffic and quota, not the example.
The command stops on the first non-OK rung by default. A 429 indicates rate
limiting but cannot identify the quota dimension. Confirm the cause with
provider telemetry. The current gate supports only the documented Databricks
pay-per-token accounting mode, so do not attach this snapshot schema to another
product. For a supported paid run, omitting `--rate-limits` removes the
pre-inference budget protection and is not a production-safe substitute.

The highest rate submitted is not the capacity. The report names the highest
tested rung with an unqualified OK verdict, and even that claim is limited to
the observed workload, endpoint state, generator path, and test window.
Measure characters/token once before the ladder and pass that fixed value with
`--cpt`; the sweep sends zero per-rung calibration requests. The 60-second
default is spacing after preflight and between rungs, not evidence that QPH,
provider burst state, or cache state reset. Verify the finished aggregate with
`python3 -m traffic_replay verify-sweep PATH` before quoting its conclusion.
Every requested rung is frozen and prevalidated before authentication,
endpoint metadata access, or preflight; a later-rung local error cannot be
discovered only after earlier paid traffic.

## 9. Compare runs with an explicit baseline

`compare` accepts only sealed, integrity-verified run inputs. Input order is
part of the interpretation contract: the first run is always the baseline and
every later run is a candidate.

```bash
python3 -m traffic_replay compare results/comparison \
  BASELINE_RUN CANDIDATE_RUN [CANDIDATE_RUN...]
```

The command creates a fresh, sealed comparison directory containing
`comparison.html`, `comparison.md`, `manifest.json`, and
`.traffic-replay-complete`. The manifest binds both rendered files plus the
exact source manifest and summary identities. Preserve the
source runs; the comparison is an index over their evidence, not a replacement
for it.

In HTML, absolute delta is candidate minus the first-input baseline. Percent
delta divides by the absolute baseline and is undefined when the baseline is
zero. Missing values stay unavailable. A `VALID` comparison labels only the
arithmetic direction (`numerically preferred` or `numerically adverse`); it
does not claim improvement/regression without repeat-run uncertainty and a
practical-effect threshold. Measurement warnings produce a `QUALIFIED`,
diagnostic-only comparison. Compatibility/source-validity failures produce an
`INVALID` comparison. Both suppress direction labels and ranking. Markdown
carries the manifest-bound side-by-side absolute values, warnings, and
invalidity reasons; HTML adds the delta matrix. The HTML is
self-contained, responsive, printable in landscape, and blocks scripts and
remote requests with a restrictive content-security policy.

Any manifest-bound 429 in any source phase, inconsistent 429 summary/journal
evidence, an explicitly invalid source measurement, or a compatibility failure
makes the comparison diagnostic-only. A sealed artifact can therefore be an
`INVALID COMPARISON`; sealing proves identity and integrity, not comparability.
`compare` verifies its output before returning, but there is no standalone
`verify-comparison` CLI in the current interface.

## 10. Understand retries and duplicates

The built-in transport opens a fresh HTTP/1.1 connection for every physical
attempt. That is useful for measuring setup pressure, but it is not a pooled
keep-alive or HTTP/2 application client. The report leaves measurement validity
at `CAUTION` and capacity `INCONCLUSIVE` unless the operator explicitly declares
the exact same production policy with
`--production-connection-policy fresh_http1_per_physical_attempt` (or the
equivalent endpoint-config field). Do not use that declaration for an unknown,
pooled, or HTTP/2 client. It is recorded operator evidence, not independent
observation of production.

Transport retries default to zero. When enabled, a failure after `POST` can
duplicate inference and billing. Usage-option rejection and one credential
refresh can also cause a second physical `POST` even with zero configured
retries. Inspect `request_attempts`, `connection_attempts`, and `retry_reasons`.

The client is at-least-possibly-once under ambiguous transport failure, not
exactly once.

Any multi-attempt or retry-marked replay row makes aggregate per-token cost
unavailable because the journal does not observe usage for every potentially
billed attempt. The report retains only explicitly incomplete subset arithmetic.

On operator cancellation, workers receive a cooperative stop signal and the
runner best-effort shuts down tracked active sockets before cancelling queued
futures. Blocked reads wake promptly. The cancellation thread does not close
the `HTTPConnection`, because clearing its socket could let a racing request
auto-connect again; the owning worker closes it in `finally`. The client checks
cancellation before the first `POST`, before each retry, and after transport
I/O wakes, so a cancellation-induced I/O error is not retried. A `POST` already
on the wire cannot be recalled, and its provider outcome and billing remain
ambiguous.

## 11. Recover diagnostics, not a benchmark

An interrupted run retains `.traffic-replay-writing`, `start.json`, and
`requests.jsonl.partial`. Newline-complete rows can be inspected; a truncated
last fragment may be ignored. Do not rename the journal or create a completion
marker. This also applies to operator-cancelled runs. `merge` and `compare`
correctly reject unsealed evidence.

For a completed input, aggregate readers verify the completion marker's
artifact ID, manifest digest and byte count, and request-row count against the
manifest-bound summary and journal.
