# llm-traffic-replay

`llm-traffic-replay` is an open-loop load generator and measurement harness
for streamed Chat Completions-style LLM endpoints. It can replay real prompts
or construct synthetic text from a token and cache-shape profile. It records
the offered schedule, what the client actually delivered, final-attempt
request-path clocks, caller-experienced clocks, response outcomes, usage
coverage, and immutable run evidence.

The tool measures a configured experiment. It does not prove that synthetic
text behaves like production text, that two provider dialects are equivalent,
or that an HTTP 200 response is semantically correct.

## Start safely

Requirements:

- Python 3.10 or newer
- NumPy 1.24 or newer
- pytest 7 or newer for the full test suite
- the Databricks CLI only when a named OAuth user-to-machine (U2M) profile
  must mint or refresh a token

Set up an isolated environment and run the tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest
```

Prove the timing instrument against the bundled localhost oracle before using
an endpoint:

```bash
python3 -m traffic_replay validate --port 0 --format json
```

`validate` exercises sampling, scheduling, streaming, journaling, reporting,
and a mock server with known TTFT and end-to-end timings. The JSON result is a
measurement-error check for that machine. It is not provider performance
evidence.

## Run one endpoint

This command performs a real preflight and then a measured run:

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --sizing-concurrency 10 \
  --duration 300 \
  --ttft-p95 YOUR_TTFT_P95_MS \
  --ttfg-p95 YOUR_TTFG_P95_MS \
  --success-rate YOUR_SUCCESS_RATE \
  --out-dir results/benchmark
```

Important operational facts:

- Before credential lookup, endpoint metadata access, network diagnostics, or
  inference, `benchmark` freezes each local workload/trace input into a private
  temporary snapshot and validates that exact byte view. Validation includes
  strict parsing, representative body construction, and the complete fixed
  schedule. `sweep` does this for every exact requested rung, not only the
  first rung or a generic base config.
- Preflight sends two representative inference requests. After both reach
  HTTP 200, if either lacks an acceptable answer, explicitly supplied
  `--probe-extra-body` candidates can each send one additional real request.
  These calls can consume quota and incur cost. They occur before
  the measured runner claims its artifact directory; metadata-only result rows
  are then appended to the sealed `requests.jsonl` as `preflight` and `probe`
  phases without request or response content. Preserve command output too if
  the human-readable probe decisions must be audited.
- `--sizing-concurrency 10` does not hold ten concurrent requests. An unloaded
  sizing pass derives one fixed open-loop arrival rate from measured service
  time and derives the worker pool. When `--max-concurrency` is omitted, the
  derived pool is capped by the default 256-thread safety ceiling. An explicit
  positive value replaces that ceiling and caps the derived pool. Concurrency
  during replay is an observed outcome.
- The default `benchmark` duration is 300 seconds. Preflight, sizing,
  calibration, response drain, and artifact finalization are outside that
  offered-load schedule and add wall-clock time.
- `benchmark` saves its effective rerun configuration under a read-only,
  content-addressed `.traffic-replay-configs/runs/.../run-config.json` path
  before the measured run. It creates `OUT_DIR/run-config.json` once for
  backward compatibility and never overwrites an older run's copy. The saved
  config keeps the durable external input paths plus `input_expectations`
  containing only SHA-256 and byte count for each configured input. It does not
  embed raw prompt content, and a rerun refuses before credential or network
  access if those external bytes changed.
- The default gate exits nonzero for a miss or invalid result. Use
  `--fail-on none` only when a non-gating diagnostic run is intentional.

To use a token environment variable instead of a named profile:

```bash
export DATABRICKS_TOKEN='...'
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --profile configs/profile_measured.json \
  --sizing-concurrency 10
```

Never place a bearer token in a run config. The harness refuses to send a
bearer credential over cleartext HTTP except to an explicit loopback host.
Named profiles support three explicit, origin-bound modes:

- a PAT `token`, with `auth_type` omitted or set to `pat`;
- CLI-cached OAuth U2M, selected with `auth_type=databricks-cli`; or
- workspace OAuth M2M `client_id` and `client_secret`, with `auth_type` omitted
  or set to `oauth-m2m`.

The M2M path follows Databricks' documented workspace client-credentials
exchange at `/oidc/v1/token` with `scope=all-apis`; it refreshes through the
same named profile when the endpoint reports an expired credential. The
profile host must exactly match the requested workspace origin. Mixed,
incomplete, unsupported, or host-mismatched credentials fail closed without
falling back to environment credentials.

Databricks recommends OAuth M2M with a service principal for unattended
automation. Protect the `.databrickscfg` file and rotate its OAuth secret under
your organization's credential policy; do not put the client secret in a run
config, command argument, or report label. This implementation supports the
standard workspace-origin invocation route only. It does not mint the
endpoint-scoped `authorization_details` token required by a route-optimized
serving URL. See Databricks'
[OAuth M2M guide](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m),
[`auth token` reference](https://docs.databricks.com/aws/en/dev-tools/cli/reference/auth-commands#databricks-auth-token),
and
[route-optimized authentication guide](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-route-optimization).
Treat PAT authentication as legacy and follow the workspace's scope, storage,
lifetime, and rotation policy.

### Gate Databricks pay-per-token traffic before inference

`--rate-limits RATE_LIMITS.json` enables a fail-closed, pre-inference budget
gate for Databricks Foundation Model API pay-per-token endpoints. Provider
limits are mutable and vary by model, deployment mode, and workspace tier;
recheck the exact row in the official
[Foundation Model APIs limits and quotas](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits)
immediately before a paid run. Do not treat a checked-in snapshot as a
timeless default.

A quota-planned benchmark must use `--fixed-rate` rather than an unloaded
sizing pass, because sizing would spend inference traffic before the replay
schedule was known:

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-PAY-PER-TOKEN-ENDPOINT \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --fixed-rate YOUR_AUTHORIZED_REQUESTS_PER_SECOND \
  --duration 300 \
  --rate-limits RATE_LIMITS.json
```

The snapshot distinguishes the provider fact date (`as_of`) from the date an
operator rechecked it (`verified_at`). `max_age_days` is a positive review
window. A missing, invalid, future-dated, or stale verification refuses the
run; age equal to `max_age_days` is still accepted. The planner also refuses
when a configured limit dimension cannot be bounded or reaches the configured
`warning_utilization`. It budgets the complete harness schedule, setup and
calibration traffic, worst-case physical attempts, a tokenizer-independent
input bound, and offered `max_tokens` reservations. The input bound is one
token per UTF-8 byte of the complete serialized request JSON plus 64 tokens for
every message and one more 64-token request-level block. It therefore includes
roles, message metadata, model, tools, provider controls, and JSON syntax, not
only message content. Synthetic replay uses the larger of configured
characters/token and the calibration hard ceiling of 12, so calibration cannot
expand a request beyond the pre-traffic authorization. For a sweep, the tool
constructs and validates every exact requested rung and budgets their union
before credential or network access and before preflight, even if early stop
would normally omit later rungs.

If that offline plan passes, the tool reads serving-endpoint metadata and
requires the direct route, `route_optimized=false`, the exact endpoint and
served-entity names, and positive `foundation_model.name` identity for every
active entity. The expected identity is
`system.ai.<rate_limits.model>`; absence of provisioned-throughput fields alone
is not accepted as pay-per-token proof. The standard quota model also requires
request `service_tier` to be absent or the exact string `"default"`. An
observed non-default response tier invalidates the run's standard-quota
comparison. The workspace tier is still a configured assertion, and unrelated
workspace traffic is invisible to the harness. A passing plan therefore means
only that this harness's worst-case forecast is below its configured warning
budget; it never proves provider quota headroom. A refusal exits with code 3.

The `rate_limits` object has a closed schema:

| Fields | Contract |
|---|---|
| `input_tokens_per_minute`, `output_tokens_per_minute`, `queries_per_hour` | At least one positive finite limit is required; only configured dimensions are enforced |
| `warning_utilization` | Positive finite fraction no greater than 1; contact with the threshold refuses |
| `source`, `as_of` | HTTPS source URL and non-future `YYYY-MM-DD` provider-fact date |
| `verified_at`, `max_age_days` | Paired non-future `YYYY-MM-DD` operator-review date and positive integer freshness window; both are required to pass the paid-run gate |
| `provider`, `deployment_mode`, `accounting_model` | Exactly `databricks`, `pay_per_token`, and `databricks_fmapi_pay_per_token` |
| `model`, `workspace_tier`, `scope` | Nonempty configured assertions; `model` must match the direct serving-endpoint name |
| `note` | Optional nonempty operator note |

See
[`configs/rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json`](https://github.com/debu-sinha/llm-traffic-replay/blob/main/configs/rate_limits_databricks_glm_5_2_enterprise_p2t_2026-08-07.json)
for structure only; its dated values must pass the freshness gate and still be
rechecked against the official source. Prompt replay is bounded from the exact
complete request built from the frozen messages with the conservative
UTF-8-byte and framing rule above; it does not rely on the provider tokenizer
or intended profile token counts.

#### Current GLM 5.2 boundary

The live Databricks limits page was last updated on 2026-08-07. At that
revision, the Enterprise pay-per-token row for `databricks-glm-5-2` is
200,000 input tokens/minute, 20,000 reserved output tokens/minute, and 7,200
queries/hour. Search snippets have shown older values; use the live page.
Databricks counts actual prompt tokens, reserves the offered `max_tokens`
before admission, credits unused reservation back after completion, and
applies the most restrictive rolling limit. These are quota controls, not a
measurement of model-serving capacity.

The managed `databricks-glm-5-2` endpoint is documented as pay-per-token.
Current Databricks provisioned-throughput architecture lists do not list GLM
5.2, so generic PT statements such as "no TPM limits" must not be attached to
this endpoint. Recheck the current
[supported-model matrix](https://docs.databricks.com/aws/en/machine-learning/model-serving/foundation-model-overview)
before testing any separately provisioned deployment.

GLM-5.2 itself does support thinking off. Z.ai's current
[Thinking Mode guide](https://docs.z.ai/guides/capabilities/thinking-mode)
documents `{"thinking":{"type":"disabled"}}`, and its
[Chat Completion reference](https://docs.z.ai/api-reference/llm/chat-completion)
also documents `reasoning_effort` values `"none"` and `"minimal"` as
thinking-skipping controls for the Z.ai API.

That vendor behavior does not establish the request contract of the
Databricks-managed adapter. The current Databricks
[reasoning-model guide](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-reason-models)
was last updated on 2026-08-07 and classifies `databricks-glm-5-2` as
reasoning-only, says it always uses internal reasoning, and gives no accepted
GLM-specific `reasoning_effort` values. Treating those documents as distinct
is important: this repository does not claim that GLM-5.2 lacks thinking-off;
it declines to promise that a Z.ai or self-hosted control survives the
Databricks adapter. Treat any candidate on that managed endpoint as an
authorized runtime experiment. HTTP 200 alone is insufficient; require a
completed visible/tool-call answer, reasoning-token/content evidence, and a
repeat preflight with the selected control. This repository deliberately
ships no Databricks GLM thinking-off default.

Likewise, repeated-prefix construction is only cache-eligible traffic. Call it
a server-observed cache hit only when the exact endpoint response reports a
cache-token field with sufficient coverage. The report keeps intended prefix
reuse, endpoint-reported cache usage, and missing cache evidence separate.

## Preflight and model behavior

The `benchmark` and `sweep` commands run a two-request gate unless
`--skip-preflight` is supplied. The gate checks reachability, streamed usage,
cached-token reporting, reasoning-channel output, clean completion, visible
content, and structurally valid tool calls. The harness does not guess
provider controls. A repeatable `--probe-extra-body '{...}'` is an explicit
opt-in to one extra request per candidate after an unreadable preflight; use
only controls documented for the exact target and authorized for the test.
Candidates are diagnostic and never mutate the measured run configuration. If
one works, rerun with that object as `--extra-body` before starting load.

An acceptable outcome for the gate and primary answer-latency population is:

1. visible assistant content or at least one structurally valid tool call with
   a nonempty function name and arguments that decode to a JSON object;
2. a completed stream; and
3. no unrecoverable stream parse errors.

This is structural validity, not semantic correctness. The tool does not grade
the factual answer or whether the selected tool was appropriate.

For current artifacts, an incomplete or parse-corrupt stream is a failed
request even if HTTP status was 200 or content arrived before the failure. It
is excluded from answer-latency, token-throughput, cache-fidelity, calibration,
and cost populations. The failure still remains in the request denominator and
error evidence; it is never silently treated as a zero-token success.

Reasoning controls are provider and model specific. Pass only a control that
the target endpoint documents and accepts:

```bash
--extra-body '{"reasoning_effort":"low"}'
```

`extra_body` is a passthrough object. The harness always owns and overwrites
`messages`, `max_tokens`, `temperature`, `stream`, `model`, and
`stream_options`. A control that works for one serving stack is not evidence
that another stack supports it.

`extra_body` is persisted across the rerun config and sealed reproducibility
evidence; probe candidates and outcomes are reported in preflight text, with
displayed values and errors passed through credential redaction. Preflight
result metadata is sealed into the run journal without
request or response content, and participates in quota-window evidence. The
selected probe object is diagnostic only and is not silently copied into the
measured configuration. The endpoint config and both CLI flags recursively
reject secret-like keys and
credential-shaped values before writing derived profile/config output or
sending traffic. Command arguments can still be visible to local process
inspection. Keep credentials in `auth_profile` or `auth_token_env`.

## Workload inputs

Exactly one of `profile_path` and `prompts_file` is required.

### Profile mode

Profile mode constructs synthetic text with the requested token-count and
prefix shape. This is useful for exercising mechanics such as prefill size,
decode budget, arrival shape, and potential prefix reuse. Synthetic content
does not reproduce production semantics, tool selection, safety paths,
reasoning difficulty, tokenizer behavior, or provider routing. Use real
prompts when conclusions depend on those properties.

The term `cache_fraction` means the intended share of prompt tokens placed in
a reusable prefix. It is not a request cache-hit probability. The achieved
metric is endpoint-reported cached prompt tokens divided by endpoint-reported
prompt tokens, per response.

#### Profile schema v1

When `schema_version` is absent, the file is schema v1. It contains exactly
p50 and p95 anchors for each marginal. All profile numbers below are schema
examples, not a recommended production workload:

```json
{
  "name": "example_v1",
  "input_tokens": {"p50": 10000, "p95": 24000},
  "output_tokens": {"p50": 200, "p95": 480},
  "cache_fraction": {"p50": 0.3, "p95": 0.7},
  "provenance": "replace with the source and extraction method",
  "label": "State whether this shape is measured or assumed."
}
```

V1 samples input and output counts from independent log-normal marginals.
Cache fractions use a logit-normal marginal, with explicit handling for
constant or boundary distributions. A p50/p95 profile has no evidence about
p90, p99, or cross-field dependence; those properties are model assumptions.

#### Profile schema v2: quantile CDF

Use `quantile_cdf` when several measured marginal quantiles are available:

```json
{
  "schema_version": 2,
  "name": "example_quantiles",
  "input_tokens": {"p50": 10000, "p95": 24000},
  "output_tokens": {"p50": 200, "p95": 480},
  "cache_fraction": {"p50": 0.3, "p95": 0.7},
  "sampling": {
    "mode": "quantile_cdf",
    "probabilities": [0.5, 0.9, 0.95, 0.99],
    "input_tokens": [10000, 19000, 24000, 42000],
    "output_tokens": [200, 390, 480, 900],
    "cache_fraction": [0.3, 0.6, 0.7, 0.85]
  }
}
```

The probabilities must be finite, strictly increasing values in `(0, 1)` and
must include exact 0.5 and 0.95 knots. Each value array has the same length and
is nondecreasing. Token values are positive; cache values are in `[0, 1]`.
The 0.5 and 0.95 ladder values must exactly match the legacy anchors.

Token ladders interpolate in log space, cache ladders interpolate linearly,
and values beyond the first and last knots clamp to those end knots. For a
finite draw, one stratified rank grid `(i + 0.5) / n` is independently shuffled
for input, output, and cache. This bounds marginal finite-sample drift without
inventing cross-field rank correlation. Sampling metadata reports
`dependence=independent_marginals`,
`rank_sampling=independently_shuffled_stratified`, and
`tail_policy=clamp_to_end_knots`.

#### Profile schema v2: empirical joint

Use `empirical_joint` to preserve observed input, output, and cache-fraction
combinations without storing prompt text:

```json
{
  "schema_version": 2,
  "name": "example_joint",
  "input_tokens": {"p50": 100, "p95": 500},
  "output_tokens": {"p50": 20, "p95": 80},
  "cache_fraction": {"p50": 0.2, "p95": 0.8},
  "sampling": {
    "mode": "empirical_joint",
    "rows": [
      {"input_tokens": 100, "output_tokens": 20,
       "cache_fraction": 0.2, "weight": 18},
      {"input_tokens": 500, "output_tokens": 80,
       "cache_fraction": 0.8, "weight": 2}
    ]
  }
}
```

Rows are unique triples with positive integer token counts, a finite cache
fraction in `[0, 1]`, and a positive integer weight. The total cycle weight is
limited to 5,000,000. The p50 and p95 anchors must equal the weighted empirical
inverse-CDF anchors. Rows are canonically sorted, then sampled in deterministic
fixed-seed shuffled weighted cycles. Sampling metadata reports
`dependence=observed_joint_triples` and
`sampling=balanced_weighted_cycles`, with
`quantile_method=inverted_cdf`. The `sample` report uses that same discrete
inverse-CDF method, so its anchors are observed row values rather than linear
interpolations between rows.

Build a content-free profile from JSONL or CSV request records:

```bash
python3 scripts/profile_from_logs.py \
  --input request_metrics.jsonl \
  --name measured_workload \
  --mode empirical-joint \
  --out configs/profile_measured.json

python3 -m traffic_replay sample \
  --profile configs/profile_measured.json \
  --n 50000 --seed 7
```

The command-line `--mode` spelling is `empirical-joint`; the JSON schema mode
is `empirical_joint`. The default `--mode quantiles` emits legacy v1 p50/p95
marginals. The generated profile records extraction counts, byte count, and
SHA-256 of the exact frozen source bytes it parsed. It emits token/cache
statistics, weights, and selected
provenance only; it does not copy prompt text, arbitrary source fields, or the
source path. The input file itself still contains whatever its owner exported
and must be handled under the applicable data policy.

`sample` prints p50 and p95 for v1. For v2 `quantile_cdf`, it prints every
configured ladder knot. For `empirical_joint`, it reports the recovered
profile anchors using the validated inverted-CDF contract.

### Prompts mode

Prompts mode accepts `.jsonl`, `.ndjson`, `.json`, or `.txt`:

```jsonl
{"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Explain the result."}]}
{"prompt":"A single user prompt"}
"A bare JSON string"
```

A `.json` file is an array of the same item forms. A `.txt` file is one user
prompt per nonblank line. Only string content is accepted; multimodal parts
are outside the current contract.

When the schedule has more requests than prompts, the tool cycles the prompt
list. Those repeats can warm an endpoint cache and make achieved cache reuse a
property of the replay. The report identifies this condition. Do not present
that achieved cache fraction as production behavior without a matching repeat
pattern in production.

### Arrival trace

Set `timestamps_file` to a text file containing one finite timestamp per line,
or JSONL objects with a finite `t` field:

```text
1710000000.120
1710000000.145
1710000000.900
```

Values may be epoch-like or already relative. The loader sorts them and shifts
the earliest value to zero. `duration_s` becomes an inclusive cap on shifted
timestamps, so rows after the cap are omitted. Blank lines are ignored. The
number of replay requests is therefore the number of retained trace rows, not
the original file line count.

Without a trace, the scheduler generates a seeded two-state modulated Poisson
arrival process. `rate_scale` deterministically thins the generated arrivals
for a fixed seed. A seed fixes the client plan; it cannot make endpoint timing,
autoscaling, caching, or network conditions deterministic.

## Timing and outcome definitions

The harness records two related timing families:

| Metric | Start and end |
|---|---|
| `connect_ms` | DNS, TCP, and TLS setup for the final attempt |
| `ttfb_ms` | immediately before final-attempt `conn.request` to the first iterated response-body/SSE line (not necessarily the first response byte) |
| `ttft_ms` | immediately before final-attempt `conn.request` to the first nonempty visible or reasoning content delta; tool-call fragments do not trigger it |
| `ttfr_ms` | immediately before final-attempt `conn.request` to the first reasoning delta |
| `ttfv_ms` | immediately before final-attempt `conn.request` to the first meaningful visible content |
| `ttf_tool_call_ms` | immediately before final-attempt `conn.request` to the first tool-call fragment |
| `e2e_ms` | immediately before final-attempt `conn.request` through `[DONE]`, or response EOF when `[DONE]` is absent |
| `caller_*` | scheduled monotonic target through the corresponding event |

The final-attempt clocks begin after connection establishment but still include
request transmission, network transit, serving-edge behavior, endpoint work,
and response transit. They are not pure server compute time.

Exact caller clocks include worker queueing, connection setup, usage-option
fallback, credential refresh, configured transport retries, and the attempt
that returns the result. Reports expose caller-experienced tables with
`*_corrected_ms` names for compatibility, and acceptance-target evaluation
prefers them when
coverage is available. Legacy artifacts without exact fields may be
reconstructed as final-attempt request-path time plus queue wait and are
labeled separately.

`ttft_definition` controls acceptance-target scoring:

- `first_content` scores the first visible or reasoning content delta.
- `first_visible` scores the first meaningful visible assistant content.

Tool-call latency is reported separately. A tool-call-only answer can be an
acceptable outcome even though it has no first-visible-content timing.

The optional network-path probe resolves the endpoint, then times several TCP
connect attempts with DNS outside that probe timer. It records
`tcp_connect_min_ms` and `tcp_connect_median_ms`. They are diagnostic path
indicators, not an exact RTT, not endpoint processing time, and not numbers to
subtract from TTFT. If the probe fails, the benchmark continues without that
evidence.

## Retries and physical requests

`endpoint.max_retries` defaults to zero. When enabled, it applies to transport
failures. A transport failure after `POST` may mean the endpoint received and
billed the request even though the client retries it. The journal records
connection attempts, request attempts, retry count, and retry reasons.

Two compatibility paths can create a second physical `POST` even when
`max_retries` is zero:

- a 400 response that explicitly rejects `stream_options.include_usage` can
  be retried without that optional field;
- a qualifying 401 or token-expiry 403 can refresh a configured credential
  once and retry.

Treat `request_attempts > 1` as possible duplicate inference work. The tool
does not provide exactly-once delivery or exactly-once billing.

On operator cancellation, the runner sets a cooperative cancellation event,
then best-effort shuts down the client's tracked active sockets so a blocked
read wakes promptly, and then cancels queued futures. It deliberately does not
cross-thread close the `HTTPConnection`: that could clear its socket and let a
racing request auto-connect again. The owning worker performs the final close
in `finally`. A worker checks the event at entry, immediately before its first
`POST`, and before every retry. An I/O error caused by socket shutdown is
therefore returned as cancelled rather than retried. Best-effort rows for
cancelled queued work record `request_attempts=0`. A `POST` already on the wire
cannot be recalled; its provider outcome and billing remain ambiguous. The run
remains unsealed diagnostic evidence rather than being finalized as a
completed benchmark.

Non-200 response bodies are not persisted. Error evidence contains the status,
sampled body length, and a truncated SHA-256 digest.

## Run artifacts and recovery

A runner-owned output directory is claimed before authentication, endpoint
discovery, sizing, calibration, or replay traffic inside the measured runner.
The higher-level benchmark preflight described above happens before this
boundary, but only after the CLI has frozen and prevalidated the exact local
inputs. Its metadata-only result rows are passed through a private API and
appended to the sealed run journal as `preflight` and `probe` phases before
runner-owned target traffic. Request and response content is not carried
forward. The runner makes its own private snapshot before parsing. Persisted
input evidence contains names, SHA-256 digests, byte counts, and construction
metadata, not raw prompt text. `start.json` is updated atomically as target,
schedule, and calibration facts become available.

During traffic, every completed row is appended to
`requests.jsonl.partial`. The journal is fsynced every 16 rows by default,
after measured replay drains before it is reread, and during normal
finalization or exception cleanup. Sizing and calibration transitions do not
force their own sync. A crash can therefore lose rows completed since the last
successful sync, but it does not turn a partial run into a completed one.

A normal completed directory contains:

```text
.traffic-replay-complete
start.json
requests.jsonl
summary.json
report.md
report.html
manifest.json
```

Manifest schema v3 records workload, execution, and artifact identities;
effective redacted configuration; immutable input and schedule identities;
source state; endpoint and request metadata; and SHA-256, byte count, and row
count integrity declarations for bound artifacts. The completion marker is
promoted only after `manifest.json` is durable.

The completion marker binds the artifact ID, manifest digest and byte count,
and manifest-bound request-row count. Aggregate readers parse and verify those
values against the manifest and request journal before accepting an input.

An interrupted directory intentionally retains
`.traffic-replay-writing`, `start.json`, and `requests.jsonl.partial`, and may
also contain `failure.json` or some unsealed report files. Every valid
newline-terminated JSON object in the partial journal is recoverable; at most
one truncated final fragment may be ignored. Such a directory is diagnostic
evidence only. `merge` and `compare` reject it, missing completion markers,
unsupported manifest schemas, symlinked/nonregular artifacts, missing
integrity declarations, and hash, size, or row-count mismatches.

Do not manually add a completion marker or edit a sealed run. That destroys
the evidence contract.

### Create an external verification receipt

After a run is complete, verify it from outside its sealed directory and
create a sibling receipt:

```bash
python3 -m traffic_replay verify-run \
  results/benchmark/20260807-183532 \
  --out results/benchmark/20260807-183532-verification
```

The `--out` path must be a sibling of the source run, never inside it. The
command opens the source only for strict, no-follow reads; it does not edit the
source `summary.json`, reports, manifest, or completion marker. If the exact
output name already exists, the command claims a unique suffixed sibling
rather than overwriting it. A complete receipt contains:

```text
.traffic-replay-complete
verification.json
verified-report.md
verified-report.html
manifest.json
```

The receipt binds the exact source manifest, completion marker, start,
summary, request journal evidence, and every source artifact declared by the
manifest. It strictly parses the canonical JSON/JSONL evidence, cross-checks
summary counts plus replay schedule/index identities against `requests.jsonl`,
rereads the source before and after rendering, and self-seals all three receipt
artifacts. The source reports stay
`VERIFY_REQUIRED`; the receipt reports are separately labeled `EXTERNAL
VERIFIED VIEW` and show independent states for integrity, source
reproducibility, and verifier reproducibility. A dirty, missing, or
inconsistent source/verifier identity cannot produce a positive
`HELD_AT_TESTED_LOAD` conclusion even when the artifact hashes agree.

Exit code 0 means the receipt was completed and reopened through its own
manifest/completion chain. Exit code 2 means verification or receipt creation
failed; no completed receipt is valid, although a
`.traffic-replay-writing` directory may remain as failure evidence. Add
`--format json` for machine-readable stdout.

This is an internal-consistency receipt, not an authenticity proof. SHA-256
bindings detect byte disagreement; they are not a digital signature, do not
prove authorship or trusted time, do not establish that a Git commit is still
available, and do not prevent later mutation. Preserve the immutable source
run and its sibling receipt together. Browser print/PDF output remains a
derivative and carries its own derivative stamp; rely on the receipt directory
and its manifest for verification.

## Reading a result

### Five independent decision dimensions

Every completed run carries one canonical decision object in `summary.json`.
`report.md` and `report.html` render the same five codes, labels, reasons, and
tested-load facts from that object:

| Decision dimension | Possible codes | Question answered |
|---|---|---|
| Evidence integrity | `VERIFIED`, `VERIFY_REQUIRED`, `TAMPERED` | Has the enclosing seal been checked? |
| Measurement validity | `VALID`, `CAUTION`, `INVALID` | Are the measurement, coverage, compatibility, and workload-fidelity gates usable? |
| Acceptance checks | `PASS`, `MISS`, `INCONCLUSIVE`, `NOT_EVALUATED` | What happened against explicitly configured performance targets? These are not called a contractual SLA unless their provenance establishes that. |
| Quota state | `EXCEEDED`, `NOT_OBSERVED`, `UNKNOWN`, `NOT_EVALUATED` | What does captured HTTP-status evidence show? |
| Endpoint capacity | `HELD_AT_TESTED_LOAD`, `NOT_HELD_AT_TESTED_LOAD`, `INCONCLUSIVE`, `NOT_EVALUATED` | Did this verified, bound test point hold? |

These dimensions are deliberately not collapsed into one green/red verdict.
For example, an acceptance check can retain its observed result while an HTTP 429
makes the measurement invalid and endpoint capacity inconclusive. A retained
acceptance `PASS` is explicitly qualified when measurement validity is not `VALID`.
`HELD_AT_TESTED_LOAD` is only a statement about the observed point; every
capacity state keeps `endpoint_ceiling_established=false` and
`provider_headroom_established=false`.

The files inside a newly written run say `VERIFY_REQUIRED` because a report
cannot authenticate the manifest that will enclose it. That is not a claim
that the bytes are corrupt. Preserve the whole completed directory and verify
the marker/manifest chain before relying on it; aggregate readers do this
before accepting a source run. Do not edit a sealed file to change the
embedded integrity state.

`report.html` is a standalone presentation: its CSS and charts are inline,
and it contains no JavaScript, remote fonts, remote assets, or network fetches.
It includes responsive layouts for narrow screens, horizontally scrollable
dense tables, text equivalents for charts, and print rules that retain the
decision and measurement-evidence text while removing navigation. The print
styles target ordinary browser A4/Letter output; a browser's own headers,
footers, margins, and pagination settings remain outside the artifact.
Every browser-print/PDF view carries an `UNSEALED PRINT/PDF DERIVATIVE` stamp.
The PDF is a convenience rendering, not a manifest-bound artifact; use the
source HTML and manifest for evidence verification. Internal hashes are not a
digital signature.
`report.md` is the dependency-free textual alternative. Layout differs, but
decision semantics do not.

### Interpret HTTP 429 exactly

HTTP 429 is counted only from an integer HTTP `status` field, not by parsing
an error string or body digest. `summary.json` records the exact count,
denominator, status-coverage count, and phase breakdown over all sealed logical
request rows supplied to quota accounting: preflight, explicit probes,
sizing, calibration, and replay. Different redacted response-body digests do
not fragment the aggregate; detailed rows remain distinct while the failure
summary uses one stable `http 429 (rate limited)` key.

Any captured 429 produces quota `EXCEEDED`, measurement `INVALID`, and
endpoint-capacity `INCONCLUSIVE`. It proves a rate-limit or quota rejection,
not which quota dimension or component enforced it and not the endpoint's
compute ceiling. Conversely, zero observed 429s with complete status coverage
produces `NOT_OBSERVED`, not a provider-headroom claim. Missing status coverage
produces `UNKNOWN`; internally inconsistent 429 aliases fail closed.

Read the evidence in this order:

1. completion marker and manifest integrity;
2. acceptable outcomes, HTTP status coverage, and failures;
3. delivered arrival rate, queue/wire lateness, pending-limit drops, and
   measured concurrency;
4. achieved token and cached-token coverage versus intended workload;
5. exact caller-experienced acceptance metrics and their coverage;
6. stability windows and only then service-time diagnostics;
7. pricing coverage and cost.

The primary latency population includes only structurally acceptable outcomes
when answer observability is available. Failed and unacceptable requests do
not disappear: they affect error and success-rate evidence. Percentiles alone
must never be used to hide shed or malformed requests.

The sample gate uses roughly ten observations beyond a quantile:

| Quantile | Minimum acceptable answer-latency observations |
|---|---:|
| p50 | 20 |
| p90 | 100 |
| p95 | 200 |
| p99 | 1000 |

Below a threshold, the percentile is still printed for diagnosis but is
marked indicative only. Success-rate scoring reports both the observed
fraction and a one-sided 95 percent Wilson lower confidence bound. A clean
verdict requires the lower bound, not only the observed fraction, to meet the
target. This calculation assumes independent request outcomes. For scale, an
all-success sample needs roughly 2,704 independent attempts before its lower
bound can substantiate a 0.999 target. The latency floors above are evidence
rules, not confidence intervals.

Cached-token coverage is explicit. `NOT REPORTED` means the endpoint did not
provide a recognized usage field; it does not mean zero cache reuse. When the
workload has an intended cache fraction, the report compares paired achieved
and intended fractions and cautions on missing coverage, invalid values, or an
absolute p50/p95 error above 0.10.

Reasoning-token throughput is shown only when the endpoint reports a recognized
reasoning-token usage field. If it does not, the harness can count
`reasoning_content` SSE deltas. Those are labeled stream-delta counts, not
token estimates.

## Cost

Pricing is never fetched or verified automatically. Any pricing object is
operator-supplied arithmetic, not a current provider price, invoice, or
commercial-product binding. Supply rates that apply to the exact provider,
model, capacity product, cloud, region, service tier, contract, and effective
date, and retain that source outside the run if auditability requires it.

Per-token mode requires uncached input and output rates. Cache-read is optional
and defaults to the input rate. The numbers below are arbitrary arithmetic
examples, not current pricing for any model:

```json
{
  "pricing": {
    "mode": "per_token",
    "input_dbu_per_m": 20.0,
    "output_dbu_per_m": 60.0,
    "cache_read_dbu_per_m": 5.0,
    "usd_per_dbu": 0.07
  }
}
```

For each fully measured row:

```text
DBU = uncached_input_tokens / 1,000,000 * input_rate
    + cached_input_tokens / 1,000,000 * cache_read_rate
    + output_tokens / 1,000,000 * output_rate
```

Per-token arithmetic covers measured replay rows only. Preflight, probes,
sizing, and calibration are outside it. An aggregate total is available only
when every replay row is either a known unsent row with exactly zero request
attempts or one clean, complete, internally sane usage response from exactly
one physical `POST`. If any row has multiple or retry-marked physical `POST`s,
unknown attempt accounting, missing/invalid usage, or an incomplete/corrupt
stream, aggregate total, per-1,000-request, per-minute, and cache-savings
figures are unavailable because earlier billed-attempt usage is not observed.
The valid measured subset remains diagnostic and is labeled incomplete.

Provisioned mode requires capacity DBU per hour. This is also an arbitrary
arithmetic example:

```json
{"pricing":{"mode":"provisioned","dbu_per_hour":100.0}}
```

When every replay row is either known unsent or has one clean, complete usage
response from exactly one physical `POST` with no retry marker:

```text
effective DBU per 1M tokens = dbu_per_hour * 1,000,000 / tokens_per_hour
```

This is utilization-dependent effective cost, not a per-token tariff. The same
physical-attempt completeness gate used for per-token totals applies here. If
any row has multiple or retry-marked physical `POST`s, unknown attempt
accounting, or missing/invalid usage, the effective DBU and USD rates and their
token-throughput denominator are unavailable. Final-response tokens may remain
as an explicitly labeled measured-subset diagnostic, but cannot establish all
provider work or billing.

## Full run configuration reference

Run JSON accepts the following top-level fields. Unknown top-level fields are
rejected by `RunConfig` construction.

| Field | Default | Contract |
|---|---:|---|
| `endpoint` | required | Object documented below |
| `profile_path` | `null` | Synthetic profile; exactly one workload input is required |
| `prompts_file` | `null` | Real text prompts; exactly one workload input is required |
| `duration_s` | `300` | Positive integer schedule seconds; trace cap when a trace is used |
| `qps_base` | `25.0` | Positive finite base-state rate |
| `qps_burst` | `350.0` | Positive finite burst-state rate |
| `qps_min` | `10.0` | Positive finite scheduler floor |
| `qps_max` | `500.0` | Positive finite scheduler ceiling |
| `rate_scale` | `1.0` | Deterministic thinning fraction in `(0, 1]` |
| `max_concurrency` | `null` | Positive worker-thread safety ceiling when explicit. Fixed-rate omission normalizes to 256. Sizing derives a pool but caps it at 256 when omitted; an explicit value replaces that ceiling |
| `max_pending_requests` | `null` | Running plus queued-work bound; runtime default is `max(2 * max_concurrency, max_concurrency + 1)` |
| `sizing_concurrency` | `null` | Positive unloaded sizing hint; derives one fixed rate and a safety-capped pool, does not hold concurrency |
| `concurrency` | `null` | Legacy alias for `sizing_concurrency`; do not set both |
| `seed` | `7` | Non-negative deterministic client-plan seed |
| `cpt` | `4.0` | Positive initial characters-per-token estimate in profile mode |
| `calibrate_n` | `12` | Non-negative extra calibration requests; actual count is capped by the unsharded schedule count |
| `shard_index` | `0` | Zero-based shard index |
| `shard_total` | `1` | Positive shard count with `0 <= shard_index < shard_total` |
| `run_id` | `null` | Shared nonempty logical ID, required when `shard_total > 1` |
| `start_at_unix` | `null` | Shared finite wall-clock start, required when `shard_total > 1` |
| `start_tolerance_s` | `0.5` | Non-negative stale-start tolerance |
| `timestamps_file` | `null` | Arrival trace replacing the synthetic schedule |
| `pool_docs_per_bucket` | `40` | Positive reusable-prefix documents per size bucket in profile mode |
| `pool_zipf_s` | `1.1` | Positive Zipf popularity exponent in profile mode |
| `out_dir` | `"results"` | Parent for timestamped runner output |
| `title` | `"traffic replay"` | Report title; control characters and credential patterns are sanitized |
| `label` | `""` | Operator context rendered in the report |
| `max_output_tokens_cap` | `512` | Positive safety cap; per request budget is the smaller of the sampled output and this cap |
| `acceptance_targets` | `null` | Strict performance-target object documented below |
| `pricing` | `null` | Strict pricing object documented above |
| `rate_limits` | `null` | Databricks pay-per-token quota snapshot; enables the conservative freshness, schedule-budget, and endpoint-binding gate described above |
| `input_expectations` | `null` | Closed map for each configured `profile`, `prompts`, and optional `timestamps` input containing exact lowercase `sha256` and non-negative `bytes`; generated rerun configs use it to refuse changed external input bytes |
| `capture_endpoint_metadata` | `true` | Best-effort Databricks serving-config lookup before inference traffic |
| `measure_network_path` | `true` | Best-effort TCP-connect diagnostic before inference traffic |
| `ttft_definition` | `"first_content"` | `first_content` or `first_visible` for acceptance-target scoring |

Endpoint fields:

| Field | Default | Contract |
|---|---:|---|
| `base_url` | required | HTTP(S) origin only; no path, query, fragment, or userinfo |
| `path` | required | One absolute request path beginning with `/`, never `//` |
| `auth_token_env` | `"DATABRICKS_TOKEN"` | Environment variable read when no named profile is set |
| `auth_profile` | `null` | Named Databricks config profile; takes precedence and fails closed |
| `model` | `null` | Included only when a shared Chat Completions route requires it |
| `connect_timeout_s` | `10.0` | Positive finite setup timeout per attempt |
| `read_timeout_s` | `120.0` | Positive finite idle timeout for each response read |
| `total_timeout_s` | `180.0` | Positive finite absolute deadline for the whole request/stream; heartbeats cannot extend it |
| `temperature` | `0.0` | Finite sampling temperature |
| `max_retries` | `0` | Non-negative transport retry count; duplicate POST risk applies |
| `include_usage` | `true` | Request streamed usage and allow the explicit unsupported-field fallback |
| `extra_body` | `null` | Credential-free finite JSON object merged below harness-owned keys; secret-like keys and credential-shaped values are rejected because request parameters are persisted as evidence. Output-budget aliases are rejected because the harness owns `max_tokens`; with `rate_limits`, `service_tier` must be absent or exactly `"default"` |

A named auth profile is origin-bound. Its configured host must normalize to the
same scheme, host, and port as `base_url`. A PAT profile has a `token` and may
set `auth_type=pat`. U2M must set `auth_type=databricks-cli` and invokes
`databricks auth token -p NAME`; Databricks documents that command as U2M-only.
A workspace M2M profile has `client_id` and `client_secret` and may set
`auth_type=oauth-m2m`; the harness exchanges those credentials directly at the
profile host's `/oidc/v1/token` endpoint with `scope=all-apis`. Official
workspace M2M profiles that omit `auth_type` are accepted. This path does not
support a route-optimized serving URL, whose token request requires
endpoint-scoped `authorization_details`. A missing profile, host mismatch,
mixed or incomplete credentials, CLI/token-exchange failure, or invalid token
fails closed; it does not fall
back to `auth_token_env`.

Accepted `acceptance_targets` fields are `ttft_ms`, `ttfg_ms`,
`hard_timeouts`, `success_rate`, `interchunk_ms`, `targets_are`, `priority`,
and `note`. Latency target objects accept only `p50`, `p90`, `p95`, and `p99`
with positive finite milliseconds. Hard timeouts accept positive `ttft_s` and
`ttfg_s` plus an optional note. Success rate is in `(0, 1]`.

### High-level command defaults

The convenience commands add these defaults before constructing the strict run
config:

| Command | Default behavior |
|---|---|
| `sample` | 50,000 draws, seed 7; profile is required |
| `schedule` | 300 seconds and `rate_scale=1.0`, using the scheduler defaults in the run-config table |
| `benchmark` | 300 seconds, sizing hint 10, `results/benchmark`, two-request preflight, no provider-control candidates unless explicitly supplied, `fail-on=miss`, text output |
| `sweep` | six geometric rungs from 1 through 32 requests/second, 120 seconds per rung, 60-second spacing after preflight and between rungs, fixed `cpt=4.0`, zero per-rung calibration requests, 256 workers, `results/sweep`, two-request preflight and early stop enabled, no provider-control candidates unless explicitly supplied |
| `quickstart` | 240 seconds, `results/quickstart`, output config `configs/quickstart.json`; profile and sizing hint are required |
| `run` | `fail-on=miss`, text output; run config is required |
| `validate` | OS-assigned port 0, 25-second schedule, `results/validation`, 60 ms oracle-error tolerance, text output |
| `merge` / `compare` | output path plus at least two complete input run directories |

When `benchmark` or `sweep` receives neither `--profile` nor `--prompts`, it
builds an explicitly stated schema-v1 placeholder profile: input p50 10,000 and
p95 24,000 tokens; output p50 200 and p95 480 tokens; cache fraction p50 0.3
and p95 0.7. The derived output safety cap is 720 tokens. These are CLI
defaults, not measured workload facts, and should be replaced before a
production conclusion.

`--cache-fraction` is the preferred public flag. It sets profile
`cache_fraction`: the intended fraction of prompt tokens placed in a reusable
prefix, not a request cache-hit probability. The old `--cache-hit-rate` spelling
is retained only as a compatibility alias with identical semantics.

## Sharding, merge, and compare

Shards must be created from the same immutable input and configuration. Every
process uses:

- the same `run_id`;
- the same future `start_at_unix`;
- the same `shard_total`; and
- one distinct `shard_index` from zero through `shard_total - 1`.

The full schedule is generated before a deterministic round-robin split, and
each request retains its global index. Synchronize host clocks and choose a
start far enough in the future for validation, endpoint metadata, network
probing, optional sizing, schedule generation, and calibration. A stale start
beyond `start_tolerance_s` aborts rather than silently desynchronizing shards.

Merge only a complete shard set:

```bash
python3 -m traffic_replay merge results/merged \
  results/shard-0/RUN_DIR results/shard-1/RUN_DIR
```

`merge` verifies manifest schema v3, artifact integrity, identities, shared
logical run and start time, distinct shard and global indices, complete
coverage, endpoint identity, workload identity, code provenance, request
parameters, and schedule identity. `--force` can preserve a compatibility or
coverage failure only as an explicitly INVALID diagnostic aggregate. It does
not override corrupt identities, duplicate evidence, or unsealed artifacts.

Merged throughput is meaningful as an aggregate rate only when shards ran
concurrently. Exact monotonic caller durations already carried by source rows
are validly pooled and can drive merged acceptance-target scoring. Legacy artifacts without
those exact fields are not reconstructed from schedule/send timestamps across
different run epochs; their merged acceptance scoring is explicitly labeled
service-time only. Stability windows, wire lateness, and in-flight concurrency are not
pooled because their cross-process time axes are not comparable.

Compare sealed runs side by side:

```bash
python3 -m traffic_replay compare results/comparison RUN_A RUN_B
```

Comparison permits different endpoints and requires sealed, integrity-verified
inputs. When workload, schedule, request parameters, harness version, latency
basis, source commit, or clean source state is not compatible, it retains the
tables only inside an explicitly INVALID diagnostic comparison. It cannot make
unlike provider request dialects, tokenizers, cache semantics, or quota states
comparable. Validate each route, pin effective request bodies, and compare
achieved workload and usage coverage, not just the requested config.

Input order is semantic: the first input is the baseline and every later input
is a candidate. In `comparison.html`, absolute delta is candidate minus
baseline; percent delta divides that difference by the absolute baseline.
Percent delta is undefined when the baseline is zero, and missing values remain
unavailable. A `VALID` comparison may label only the arithmetic direction as
`numerically preferred` or `numerically adverse`. It does not call a change an
improvement or regression because the tool has no configured repeat-run
uncertainty model or practical-effect threshold. A `QUALIFIED` comparison has
compatible inputs but measurement warnings and is diagnostic-only; an
`INVALID` comparison has compatibility or source-validity failures. Both keep
neutral deltas and suppress arithmetic preference labels and performance
ranking. `comparison.md` presents the same manifest-bound runs, compatibility
failures, warnings, and absolute values in
a portable side-by-side form; the richer HTML adds the explicit baseline and
delta table.

Comparison output is a separate sealed manifest-v3 diagnostic artifact with
`artifact_type=comparison`. A fresh directory contains `comparison.md`,
`comparison.html`, `manifest.json`, and `.traffic-replay-complete`; the
manifest binds both rendered files and the exact source manifest and
summary identities. The HTML is responsive, print-oriented in
landscape, dependency-free, and protected from scripts and remote requests by
a restrictive content-security policy. Browser PDF output is explicitly
stamped as an unsealed derivative. Output verification is performed
before `compare` returns, but there is currently no standalone
`verify-comparison` CLI. Preserve the comparison directory and all source runs:
a completed comparison does not replace its evidence or turn an INVALID
compatibility result into a valid one.

## Rate sweeps

Use a fixed-rate ladder to find the highest tested rung that remains valid:

```bash
python3 -m traffic_replay sweep \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --rate 1:32:6 \
  --duration 120 \
  --cooldown 60 \
  --cpt YOUR_PREMEASURED_CHARACTERS_PER_TOKEN
```

The rate axis is open-loop requests per second. `--rate 1:32:6` is a six-rung
geometric ladder; a comma list selects exact rungs. Default duration is 120
seconds per rung, spacing is 60 seconds after preflight and between rungs, and
the default worker bound is 256. Measure characters/token once in a separate
benchmark and pass it with `--cpt`; a sweep fixes `calibrate_n=0` so calibration
traffic cannot vary the workload between rungs. Cooldown is operational
spacing in a sequential, stateful experiment. It proves neither QPH recovery
nor provider burst or cache reset. By default the sweep stops when a rung is
not unqualified OK. An incomplete rung, per-rung calibration, unknown request
attempt, or higher pass after a lower failure invalidates the sweep and removes
the capacity conclusion. The highest scheduled or submitted rung is not
automatically a capacity claim.

Verify a copied or archived aggregate before quoting it:

```bash
python3 -m traffic_replay verify-sweep results/sweep
```

The verifier checks each source run's internal hashes and bindings, then
re-derives the report, traffic counts, validity, exit status, and highest held
rate from the sealed evidence. Internal hashes are not a digital signature.

## Production use checklist

Before treating a run as production evidence:

- obtain authorization and a load window for the target endpoint;
- confirm provider quotas and provisioned capacity outside this tool;
- use the real client region and network path;
- run `validate` on the generator host;
- validate the provider's exact streaming dialect and usage fields;
- pin endpoint identity, model, request parameters, reasoning controls, and
  output cap;
- use real prompts or a measured profile with explicit limitations;
- set your own acceptance targets and current pricing;
- bound workers and pending requests, then increase load in guarded steps;
- monitor endpoint, client, network, quota, and cost telemetry externally;
- retain only sealed manifest-v3 directories as benchmark evidence.

An HTTP 429 proves that a request was rate limited. This tool does not infer
which quota dimension caused it, such as input tokens, output tokens, requests,
or an account-level policy. Diagnose that with provider telemetry and the
current official
[Databricks limits and quotas](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits).

## Architecture and repository layout

See [Architecture](https://github.com/debu-sinha/llm-traffic-replay/blob/main/docs/ARCHITECTURE.md)
for component and clock boundaries and
[Production testing](https://github.com/debu-sinha/llm-traffic-replay/blob/main/docs/PRODUCTION_TESTING.md)
for a staged runbook. The
editable and rendered diagrams are in `docs/diagrams/`.

```text
traffic_replay/                  package implementation
traffic_replay/data/             packaged validation profile
tests/                           pytest suite
configs/                         example profiles and run configs
scripts/profile_from_logs.py     profile extractor for token/cache logs
docs/                            runbooks and diagrams
notebooks/                       workspace packaging and smoke notebook
```

The checked-in report screenshots are historical format illustrations only.
Trust the fields in the current `summary.json`, manifest schema v3, and reports
generated by the current tested commit.
