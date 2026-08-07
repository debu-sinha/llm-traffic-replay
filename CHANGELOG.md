# Changelog

This file records behavior changes. It does not certify benchmark numbers from
older artifacts. Compare only sealed runs whose manifests prove compatible
code, workload, request, schedule, and timing definitions.

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
  source manifest and authenticated summary identities.

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
  credential refresh, and transport retries. SLA scoring prefers these clocks
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
- Merged reports pool exact monotonic caller durations and may score SLA from
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

- Added sharded schedules, merge, provider comparison, SLA scoring,
  interchunk tracking, and believability context.

## 0.1.0

- Added profile-driven synthetic workloads, reusable-prefix pools, bursty
  open-loop scheduling, streamed latency capture, and mock-server validation.
