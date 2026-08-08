# Changelog

This file records behavior changes. It does not certify benchmark numbers from
older artifacts. Compare only sealed runs whose manifests prove compatible
code, workload, request, schedule, and timing definitions.

## 0.6.0 - 2026-08-08

### Pre-traffic production safety

- Freeze workload and trace inputs to private temporary bytes. Fixed-rate and
  trace runs construct their exact schedule before credential or network
  access; unloaded sizing mode prevalidates its workload first but necessarily
  derives the measured schedule only after the authorized sizing requests.
  Prevalidate every requested sweep rung before the first preflight request.
  Generated rerun configs retain only the original paths plus closed-schema
  SHA-256/byte-count `input_expectations`; changed external bytes refuse a
  rerun, and raw prompt content is not copied into the config or run evidence.
- Add origin-bound workspace OAuth M2M profiles with direct
  `/oidc/v1/token` client-credentials exchange and fail-closed credential
  selection. This supports standard workspace-origin routes, not the
  endpoint-scoped token required by route-optimized serving URLs.
- Add an optional fail-closed Databricks pay-per-token quota gate that checks
  dated source freshness and conservatively budgets preflight, probes,
  calibration, replay, retries, input-token targets, and offered output-token
  reservations before inference begins.
- Bind a passing plan to live serving-endpoint metadata, including the direct
  route, `route_optimized=false`, exact endpoint and served-entity names, and
  positive `system.ai.<model>` foundation-model identity for every active
  entity. Standard quota accounting accepts only absent/default request
  `service_tier` and is invalidated by an observed non-default response tier.
  Workspace tier and unrelated workspace traffic remain outside the tool's
  independent knowledge, so a pass is not provider-headroom proof.
- Bound planned input demand at one token per UTF-8 byte of the complete
  serialized request JSON plus 64 framing tokens per message and per request,
  including roles, message metadata, model, tools, provider controls, and JSON
  syntax. Synthetic workloads use the larger of configured characters/token
  and the calibration hard ceiling of 12; prompt mode is bounded from its
  exact frozen messages.
- Preserve preflight and explicit probe outcomes as content-free request rows
  in the sealed run journal, with their physical attempts included in quota
  evidence.

### Auditable decisions and artifacts

- Add one canonical decision object with independent evidence-integrity,
  measurement-validity, acceptance-check, quota, and tested-capacity states.
  Reports no longer collapse these different questions into one verdict.
- Add `verify-run`, which creates a separately sealed, sibling verification
  receipt after re-reading and cross-checking a completed run's canonical
  evidence. The receipt proves internal consistency, not authorship or trusted
  time.
- Seal sweep output as a manifest-v3 artifact and add `verify-sweep`. Harden
  merge and comparison inputs against incomplete, nonregular, duplicated,
  identity-inconsistent, or byte-inconsistent evidence.
- Qualify run comparisons when endpoint identity, HTTP-status coverage,
  request-journal coverage, or replay coverage is missing; invalid comparisons
  do not present directional deltas as conclusions.

### Measurement and reporting correctness

- Separate final-attempt request-path clocks from exact caller-experienced
  clocks, retain HTTP status by execution phase, and keep intended cache reuse
  distinct from endpoint-reported cached tokens.
- Prefer the accurately named `--cache-fraction` flag for reusable-prefix token
  share; retain `--cache-hit-rate` only as a compatibility alias, never as a
  request-level hit-probability claim.
- Treat incomplete or parse-corrupt streams as failures and exclude them from
  answer latency, usage throughput, cache fidelity, calibration, and cost.
- Withhold aggregate per-token cost and provisioned effective rates when any
  replay row has ambiguous retries, multiple physical posts, unknown attempt
  accounting, invalid usage, or an incomplete stream. Provisioned token
  throughput is subject to the same physical-attempt completeness gate. Cost
  output is explicitly unverified operator-supplied rate arithmetic over replay
  rows only, not a fetched price or invoice.
- On operator cancellation, signal workers and best-effort shut down tracked
  active sockets before cancelling queued futures. Blocked reads wake promptly,
  clients do not retry after cancellation, already-issued `POST`s remain
  ambiguous, and the interrupted directory remains unsealed diagnostics.
- Add strict JSON and JSONL parsing, duplicate-key and nonfinite-number
  rejection, safer immutable configuration snapshots, and bounded
  content-free stream-parse diagnostics.
- Replace the report and comparison layouts with responsive, print-aware HTML
  views that surface evidence limitations and decision blockers before
  latency tables.

### Databricks GLM 5.2 reference data

- Add an explicitly illustrative GLM 5.2 canary profile and a dated Enterprise
  pay-per-token quota snapshot sourced from the live Databricks limits page.
  These inputs do not certify customer demand, endpoint capacity, latency,
  throughput, or provider quota headroom.
- Document the provider boundary for thinking controls: Z.ai documents
  thinking-off controls for its GLM-5.2 API, while current Databricks managed
  endpoint documentation describes `databricks-glm-5-2` as reasoning-only and
  does not publish accepted GLM-specific off values.

## 0.5.1 - 2026-08-07

- Bound every request with an absolute stream deadline and persist exact
  completion times for both successful and failed requests.
- Include timed-out and failed requests in observation windows and occupancy,
  preventing throughput inflation and false low-concurrency reports.
- Gate verdicts on paired input/output token fidelity and on whether acceptance
  targets are explicitly illustrative.
- Reject ambiguous multi-choice streams, duplicate JSON keys, persisted request
  credentials, unsafe endpoint query secrets, and unbounded local allocations.

## 0.5.0 - 2026-08-06

### Crash-safe benchmark evidence

- The runner claims and fsyncs an output directory before target traffic.
- Completed outcomes are appended to `requests.jsonl.partial`; interrupted
  runs retain the writing marker and recoverable newline-complete rows.
- Final artifacts use atomic same-directory replacement. The manifest is
  written last and `.traffic-replay-complete` is promoted only after it is
  durable.
- Manifest schema v3 binds request, summary, Markdown, HTML, and start
  artifacts by SHA-256 and byte count, plus request row count.
- Merge and compare require sealed schema-v3 evidence and verify identity,
  regular-file, hash, size, and row-count invariants, including the completion
  marker's binding to its manifest and request journal.
- Comparison output is its own manifest-v3 artifact. Its completion marker
  binds the comparison manifest, which binds the rendered report and exact
  source manifest and manifest-bound summary identities.

### Workload fidelity

- Profile schema v2 adds `quantile_cdf` with explicit marginal knots,
  independently shuffled stratified ranks, log token interpolation, linear
  cache interpolation, and clamped end-knot tails.
- Profile schema v2 adds `empirical_joint` with canonically sorted unique
  token/cache triples and deterministic balanced weighted cycles.
- Empirical-joint sample reports use the same inverted-CDF quantile method as
  profile validation, so they never invent interpolated values between
  observed rows.
- The log extractor can emit empirical-joint profiles, records structured
  extraction counts, and binds the exact source bytes by SHA-256 without
  copying prompt text, arbitrary source fields, or the source path.
- Schema v1 remains the default when `schema_version` is absent and retains
  its p50/p95 independent-marginal behavior.

### Clocks, outcomes, and physical requests

- Request rows carry exact monotonic caller clocks from the scheduled target
  through the first response-body line, content, reasoning, visible-content,
  tool-call, and completion events.
- Caller clocks include queueing, connection setup, usage fallback,
  credential refresh, and transport retries. Acceptance-target scoring prefers these clocks
  when coverage exists.
- Structurally valid tool-call-only responses can be acceptable outcomes and
  have their own first-tool-call timing.
- Rows distinguish connection attempts from possible physical POST attempts
  and record retry reasons. Transport retries default to zero; usage fallback
  and credential refresh can still create an additional physical POST.
- Error artifacts retain status, sampled body length, and a body digest rather
  than response text.

### Validation and aggregation

- Configuration, policy, workload, transport, endpoint-origin, and profile
  schemas reject malformed or nonfinite values before measured traffic.
- Preflight does not guess or automatically send provider reasoning controls.
  Repeatable `--probe-extra-body` values opt in to one user-supplied, finite
  JSON-object candidate per extra real request after an unreadable preflight.
- Persisted `extra_body` and reported probe candidates reject secret-like keys
  and credential-shaped values recursively before derived output or traffic.
- A clean success-rate verdict requires both the observed fraction and its
  one-sided 95 percent Wilson lower confidence bound to meet the target; the
  report states the independent-outcome assumption.
- Named authentication profiles are bound to the configured endpoint origin
  and fail closed.
- The active dispatcher bounds running plus queued work and journals an unsent
  failure when the pending bound is full.
- Merge validates a complete, nonoverlapping global shard index set. Forced
  compatibility or coverage failures remain explicitly INVALID.
- Merged reports pool exact monotonic caller durations and may score acceptance targets from
  them. They do not reconstruct legacy caller latency from timestamps across
  different run epochs; the service-only basis is labeled when exact fields
  are absent.

## 0.4.1

- Added `benchmark`, a preflighted endpoint-to-report workflow that saves a
  rerun config.
- Added caller-experienced latency summaries, stricter evidence-based verdicts,
  usage coverage, richer provenance, and corrected throughput and arrival-rate
  arithmetic.
- Calibration requests remain a separate phase and do not consume scheduled
  replay arrivals. The actual calibration count is
  `min(calibrate_n, global_schedule_count)`.
- Exact in-flight concurrency uses an event sweep rather than a sampled peak.

## 0.4.0

- Separated transport/content success from readable-answer evidence.
- Added first-visible-token coverage, answer outcome accounting, global-cap
  truncation, unloaded concurrency sizing, `quickstart`, named Databricks
  profiles, and acceptance-target provenance.
- The legacy `concurrency` field became a sizing hint. It is now named
  `sizing_concurrency` and derives one fixed open-loop rate; it does not hold
  concurrency.

## 0.3.0

- Moved DNS/TCP/TLS setup out of final-attempt TTFT, TTFB, and end-to-end
  clocks into `connect_ms`.
- Added delivery lateness, sample-size cautions, stability windows, best-effort
  endpoint metadata, prompt-repeat cautions, cost blocks, HTML reports,
  prompts mode, request-body passthrough, and reasoning observability.
- Later releases supersede the original sample-size guidance. Current floors
  are p50 20, p90 100, p95 200, and p99 1000 acceptable answer-latency rows.

## 0.2.0

- Added sharded schedules, merge, provider comparison, acceptance-target scoring,
  interchunk tracking, and believability context.

## 0.1.0

- Added profile-driven synthetic workloads, reusable-prefix pools, bursty
  open-loop scheduling, streamed latency capture, and mock-server validation.
