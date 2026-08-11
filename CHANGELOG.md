# Changelog

## Guided customer configuration

- Default guided telemetry recovery to correlation-preserving empirical-joint
  sampling; the independent p50/p95 model is now an explicit lossy fallback.
- Make `check-config` reuse the exact frozen workload draw, clearly define
  calibration, and print one quoted real-endpoint command plus report-opening
  guidance.
- Require one-sided 95% Wilson confidence for latency-target compliance as
  well as success-rate acceptance, and render point-estimate passes without
  enough evidence as `NOT PROVEN`.
- Move the field glossary to the report appendix and put the exact request
  count and load window in the first customer-facing scope statement.
- Add interactive and noninteractive `init-config` setup for common
  Databricks/OpenAI telemetry exports, custom column mappings, customer-owned
  SLA files, exact schedules, and referenced run configs.
- Add zero-traffic `check-config` validation with plain-language workload,
  SLA, request-count, token, cost, dropped-row, modeling, and assumption
  output.
- Make customer reports distinguish configured rate-curve intensity, logical
  replay-window average, exact all-phase physical POST peaks, and peak
  in-flight concurrency so quota-spaced bursts cannot be presented as steady
  traffic.
- Surface declared-versus-observed token-shape mismatch, output-cap coverage,
  and percentile sample floors before the engineering appendix; p99 is never
  published as supported below 1,000 eligible successful observations.
- Keep stability language attribution-neutral when the artifact has only
  caller-path evidence; client, network, gateway, and backend causes require
  correlated telemetry.

This file records behavior changes. It does not certify benchmark numbers from
older artifacts. Compare only sealed runs whose manifests prove compatible
code, workload, request, schedule, and timing definitions.

## 0.7.0 - 2026-08-10

### Versioned endpoint adapter foundation

- Extract the streamed Chat Completions wire behavior behind the versioned
  `openai.chat_completions.sse/v1` endpoint adapter while preserving request
  bytes for configurations that were already valid. The client now delegates
  serialization, media types, framing, event folding, finalization, usage
  normalization, and usage-control fallback to the selected adapter.
- Add a fail-closed adapter registry and `traffic-replay adapters` catalog.
  Adapter IDs, response modes, media contracts, usage modes, implementation
  identities, and runtime-derived implementation SHA-256 fingerprints are
  inspectable and mutation-checked within the current Python runtime;
  duplicate, mutable, malformed, or post-registration-modified contracts are
  rejected.
- Bind adapter identity into workload and run identities, request comparison
  parameters, every request result, and transport evidence. Record the exact
  serialized SHA-256 of every physical POST body so an adapter fallback cannot
  diverge invisibly from logical request evidence.
- Reject adapter-owned fields in `extra_body` instead of silently discarding
  conflicting values. The Chat/SSE adapter owns `messages`, `max_tokens`,
  `temperature`, `stream`, `model`, and `stream_options`. Allow
  `temperature=null` in a run config to omit that field deliberately.
- Add an integrity-checked, immutable capability-profile/catalog library with
  exact provider/model/route/API resolution. It is a data-contract foundation,
  not yet a CLI-loaded or runtime-enforced capability control plane, and its
  digest proves byte consistency rather than source trust.
- Stop labeling a single accepted reasoning-control probe as `ignored` when it
  does not produce a visible answer. HTTP 200 now proves only acceptance; the
  effective behavior remains unknown until stronger paired evidence exists.
- Do not classify HTTP 400 or 422 alone as proof that a probe candidate was
  rejected. Rejection requires the selected adapter's status contract and a
  bounded response sample that explicitly identifies the candidate field or
  path. Persist only the sample byte count, full-body SHA-256, and
  classification; never persist the response text.
- Fail closed when multiple recognized cache-token or reasoning-token aliases
  appear in one usage object with conflicting values.

### Measurement and evidence corrections

- Add `ttse_ms` and exact caller `caller_ttse_ms` for the first complete
  framed event emitted by the selected response adapter parser. Keep this
  protocol diagnostic separate from TTFB, reasoning, visible content, tool
  calls, and end-to-end time; reports and verification explicitly forbid
  interpreting an SSE event or response-body chunk as a model token.
- Treat a missing or deliberately skipped preflight as exploratory caution,
  never a held benchmark or capacity result. Bind every carried preflight to
  the exact origin, route, adapter implementation, model, request controls,
  workload, representative logical/physical body hashes, and sealed setup
  artifact; every sweep rung inherits that proof without duplicating setup
  traffic.
- Reclassify one stable request/response model alias mismatch as unverified
  caution. Mixed response-model values or a contradiction with the bound
  Databricks served-model/control-plane identity remain invalid.
- Separate attempted, on-wire, and successful prompt populations, and label
  completion throughput and TPOT as all-completion measurements. Hidden
  reasoning can be included; visible-output rates remain unavailable without
  exact, source-labeled visible-token accounting and complete clean coverage.
- Make unloaded calibration count explicit and persist warm-state and exact
  payload-overlap limitations. Disabling harness calibration does not prove a
  globally cold endpoint or cache.

### Customer workflow and field clarity

- Add `benchmark --requests N` for an exact deterministic measured replay
  population, with a durable content-addressed timestamp trace and separate
  pretraffic disclosure of replay, setup/preflight, calibration, and logical
  total counts.
- Add `init-databricks --auth-profile PROFILE` to resolve the profile-bound
  workspace host, discover READY chat/foundation endpoints through the
  Databricks CLI, and write a bounded three-request starter configuration.
- Add `--verify-after-run` to preserve the immutable source run, create a
  sibling verification receipt, and print authoritative HTML/Markdown paths,
  captured traffic counts, and all five canonical decision codes.
- Embed a report field glossary in both HTML and Markdown. Define calibration
  explicitly as real paid unloaded traffic used only for synthetic
  characters-per-token estimation - not warm-up exclusion, quality, latency,
  capacity, or quota-reservation evidence - and add a complete output data
  dictionary for summaries, request rows, manifests, and receipts.

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
- Add a command-scoped, non-waiting runtime quota guard for every physical
  inference `POST`, including preflight, probes, compatibility/auth fallbacks,
  transport retries, replay, and all sweep rungs. It enforces strict rolling
  warning budgets plus exact serialized request-byte ceilings, permanently
  trips on denial or accounting uncertainty, statically partitions shard
  budgets, and persists per-attempt admission transitions for reconciliation.
- Bind a passing plan to live serving-endpoint metadata, including the direct
  route, `route_optimized=false`, exact endpoint and served-entity names, and
  positive `system.ai.<model>` foundation-model identity for every active
  entity. Standard quota accounting accepts only absent/default request
  `service_tier` and is invalidated by an observed non-default response tier.
  Workspace tier and unrelated workspace traffic remain outside the tool's
  independent knowledge, so a pass is not provider-headroom proof.
- Bound planned input demand at one token per UTF-8 byte of the complete
  serialized request JSON plus harness-defined conservative 64-token allowances
  per message and per request; these allowances are engineering assumptions,
  not a provider tokenizer contract. Include roles, message metadata, model,
  tools, provider controls, and JSON syntax. Synthetic workloads use the larger
  of configured characters/token
  and the calibration hard ceiling of 12; prompt mode is bounded from its
  exact frozen messages.
- Preserve preflight and explicit probe outcomes as content-free request rows
  in the sealed run journal, with their physical attempts included in quota
  evidence.
- Claim and fsync a separate setup-traffic artifact before CLI preflight/probe
  inference. Seal normal pass/refusal evidence as an explicit non-performance,
  non-SLA, non-capacity result; leave crashes incomplete, and attach passing
  metadata-only rows once to the measured artifact's complete quota population.

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
- Bind clean wheel and source-distribution builds to provenance schema v2,
  covering every shipped package Python file and instrument-owned JSON input.
  Git-less installs reject missing, dirty, malformed, or source-mismatched
  provenance; the deterministic build ID is an integrity checksum, not a
  signature or trusted-time assertion.
- Generate the self-contained Databricks diagnostic notebook from a clean
  tracked source inventory and make it verify its payload, source provenance,
  and exact collected test count before endpoint access. Generate the
  five-page customer field-guide PDF from clean HTML with visible source
  commit/hash stamps, semantic page checks, and a hash sidecar. Neither
  derivative is benchmark evidence.

### Measurement and reporting correctness

- Separate final-attempt request-path clocks from exact caller-experienced
  clocks, retain HTTP status by execution phase, and keep intended cache reuse
  distinct from endpoint-reported cached tokens.
- Capture bounded response model/object/fingerprint identity and hashed response
  IDs. Conflicting stream identity is a protocol error; mixed or unexpected
  response models invalidate a single-model benchmark, while fingerprint
  rotation remains context.
- Re-read and normalize serving-endpoint metadata only after response drain and
  compare it with the normalized pre-run summary. Changed summarized metadata
  invalidates a single-configuration benchmark; incomplete capture remains
  explicit uncertainty.
- Define interchunk latency exactly as the widest gap between successive
  visible/reasoning/refusal-bearing SSE events, excluding heartbeats,
  usage-only events, and tool-call-only fragments. Stability windows now use
  the headline acceptable-outcome population and retain failures separately.
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
- Record the built-in fresh-HTTP/1.1-per-physical-attempt transport as a closed
  machine contract. Capacity is inconclusive unless an operator explicitly
  declares that exact production connection policy; the report preserves that
  declaration as an assertion rather than claiming production observation.

### Databricks GLM 5.2 reference data

- Add an explicitly illustrative GLM 5.2 canary profile and a dated Enterprise
  pay-per-token quota snapshot sourced from the live Databricks limits page.
  These inputs do not certify customer demand, endpoint capacity, latency,
  throughput, or provider quota headroom.
- Include the current Foundation Model API workspace ceilings of 200 QPS and
  4 MB/request in the dated snapshot. Because the source does not state a
  decimal/binary MB convention, enforce a conservative 4,000,000 serialized
  request bytes and require a live pre-run recheck.
- Relabel the bundled blended and validation profiles and the provisioned
  template as unverified illustrative inputs. The provisioned template has
  client bounds but no built-in provider-capacity or quota guard. Make the
  bundled smoke, provisioned, and prompts templates score visible answer onset
  explicitly rather than inheriting reasoning-inclusive `first_content`.
- Record the serving-engineering-confirmed managed Databricks GLM 5.2 request contract:
  top-level `{"reasoning_effort":"none"}` disables reasoning, while omission
  selects maximum reasoning. The public Databricks guide classifies the model
  as reasoning-only and names the field but does not enumerate GLM-specific
  accepted values. Keep this managed control distinct from direct SGLang's
  nested `{"chat_template_kwargs":{"enable_thinking":false}}` switch and
  Z.ai's hosted API request shape. Serving-engineering confirmation covers the
  engagement's discussed managed direct route and `system.ai.glm-5-2` Gateway
  service. Conformance and behavioral validation remain route- and
  revision-specific, and HTTP acceptance alone does not prove effect;
  production qualification remains direct-only until Gateway routing,
  identity, and combined quotas are bound.

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
