# Architecture

The harness is an open-loop dispatcher, a bounded pool of blocking streaming
clients, and a crash-safe evidence writer. It deliberately keeps workload
intent, physical request attempts, final-attempt request-path clocks, and
caller-experienced clocks separate.

![Component and evidence flow](diagrams/architecture.svg)

## Run lifecycle

The runner follows this order:

1. Detach and strictly revalidate the run config and exactly one workload
   input, including any closed-schema `input_expectations`.
2. Snapshot package source identity and copy workload and trace files once to
   private temporary bytes. Enforce each expected SHA-256 and byte count against
   that captured view.
3. Parse the private workload, construct representative and sampled request
   bodies, and materialize the complete fixed schedule or timestamp trace. A
   sizing-derived schedule is the only schedule that cannot yet exist; its
   representative workload is still constructed here.
4. When `rate_limits` is configured, compute a conservative budget from those
   same parsed objects and enforce snapshot freshness. Refuse an unbounded,
   threshold-reaching, missing, invalid, or stale plan before claiming a run
   directory or resolving credentials.
5. Exclusively claim the output directory and durably write
   `.traffic-replay-writing`, `start.json`, and an empty
   `requests.jsonl.partial` before target traffic.
6. Append metadata-only rows for any CLI preflight and explicitly authorized
   probe traffic supplied through the private runner API, then sync them into
   the partial journal. Request and response content is not carried forward.
7. Resolve credentials and construct the client.
8. Capture network-path and endpoint metadata evidence. Metadata is best effort
   for an ordinary run; a quota-aware run binds the direct route, configured
   model, `route_optimized=false`, exact served-entity names, and positive
   `system.ai.<model>` identity for every active foundation-model entity. It
   fails closed before inference when that evidence is incomplete.
9. If requested, send an unloaded sizing sample and derive one fixed
   open-loop rate and worker bound. The derived worker bound is capped by an
   explicit `max_concurrency`, or by the default 256-thread safety ceiling
   when that field is omitted.
10. Use the already validated fixed schedule, or generate the sizing-derived
    schedule, compute its binary identity, select the shard's globally indexed
    subset, and update `start.json`.
11. Send calibration requests. In profile mode, clean, complete,
    endpoint-reported prompt usage can adjust the characters-per-token estimate.
12. Dispatch measured arrivals against monotonic schedule targets. The
    executor and pending-future map are bounded. A full pending bound creates a
    journaled unsent failure instead of unbounded memory growth.
13. Append each observed outcome to the durable partial journal as collection
    sees it. Rows can be in completion order; `global_index` retains workload
    order.
14. Drain outstanding work and sync the journal.
15. When endpoint metadata capture is enabled, take a second normalized
    control-plane summary only after response drain and compare it with the
    pre-run summary. Record changed or incomplete stability evidence together
    with the final runtime-quota snapshot.
16. Summarize persisted replay rows for acceptance and stability metrics while
    using all sealed phases for quota-window and admission evidence.
17. Atomically promote `requests.jsonl`, write summary and reports, write the
    manifest last, and promote the writing marker to
    `.traffic-replay-complete` only after the manifest is durable.

The higher-level `benchmark` and `sweep` path has an earlier local gate. It
freezes workload/trace bytes, parses them, constructs representative bodies,
and, for fixed-rate or trace-driven work, materializes the exact schedule
before credential or network access. Unloaded sizing validates the frozen
workload and representatives at this gate but cannot construct its schedule
until the authorized sizing requests derive a rate. For a sweep, every
requested rung is separately constructed and checked while sharing the same
frozen workload source. Before its two representative preflight requests or
any explicitly supplied model-control candidate probes, the CLI claims a
separate `OUT_DIR-setup-traffic/TIMESTAMP` artifact. Every completed
metadata-only row is fsynced. A normal pass or refusal seals that artifact as
an explicit non-performance/non-SLA/non-capacity result; a crash leaves an
incomplete diagnostic journal. If `--force` permits continuation after both
representatives were reachable but an answer was unreadable, the gate outcome
is `preflight_forced_unreadable`, never `preflight_passed`; the measured run
or sweep remains explicitly INVALID diagnostic evidence. Force does not
override an unreachable or failed transport preflight. On a pass, the same
rows are passed through a
private API and included once in the first measured run's journal as
`preflight` and `probe` phases. Request and response content is not included.
They participate in quota-window evidence but not replay acceptance
percentiles. A sweep attaches them only to its first rung. The tool does not
guess provider controls.

For `benchmark` and `sweep`, the offline quota gate runs after exact local
prevalidation and before credential lookup. It budgets setup requests and the
complete measured schedule, then resolves credentials and uses a control-plane
endpoint read to bind a passing plan before the first inference `POST`. The
passing high-level plan also constructs one command-scoped runtime guard that
is shared by preflight, probes, automatic physical fallbacks/retries, replay,
and all sweep rungs. The lower-level `run` repeats input capture, exact
prevalidation, and the offline
plan before claiming its directory; it performs endpoint binding inside that
directory before sizing, calibration, or replay. A control-plane read or TCP
diagnostic can still make a network connection after the local gate; the
guarantee is that a local or quota-plan refusal happens before paid inference.

The public rerun config keeps the operator's durable input paths plus
`input_expectations` containing only SHA-256 and byte count for the configured
profile/prompts and optional trace. A rerun captures those external bytes and
refuses before credential or network access if either value changed. Private
snapshots may contain prompts while the process runs, but raw prompt content is
not copied into the saved config, request journal, summary, report, or manifest.

An editable high-level overview is
[architecture.excalidraw](diagrams/architecture.excalidraw). The more detailed
SVG used above is [architecture.svg](diagrams/architecture.svg) and is
maintained separately; architecture changes must keep both views consistent.

## Workload construction

![Open-loop load model](diagrams/load-model.svg)

Profile mode and prompts mode share the scheduler and client but have different
fidelity boundaries.

Profile mode samples token and intended cached-prefix shapes, constructs text,
and selects shared-prefix documents from a deterministic Zipf pool. The pool
creates an opportunity for prefix reuse. Only endpoint-reported
`cached_tokens / prompt_tokens` is an achieved cached prompt-token fraction.
The intended fraction is not a request cache-hit rate.

Prompts mode replays supplied text. If requests outnumber prompts, it cycles
the list; repeated prompts can warm an endpoint cache. That is replay behavior
unless the production workload has the same repeat pattern.

The synthetic scheduler is a two-state modulated Poisson process followed by
seeded thinning. A timestamp trace replaces it, is sorted, shifted to zero,
and capped by `duration_s`. The runner creates all global indices before shard
selection so schedule and coverage identities are auditable.

## Request sequence and clocks

![Physical request and timing sequence](diagrams/request-sequence.svg)

For one logical replay row:

1. The dispatcher waits until the scheduled monotonic target.
2. It submits to the worker pool if the pending bound has room.
3. The worker records exact queue wait, opens a fresh connection, and records
   `connect_ms` across DNS, TCP, and TLS setup.
4. When `rate_limits` is configured, the command-scoped guard atomically
   reserves exact serialized bytes, the conservative input-token bound,
   offered `max_tokens`, and one query for this physical attempt. Admission
   happens after connection setup and immediately before the last safe point
   preceding `conn.request`; a refusal sends no `POST` for that attempt.
5. The final-attempt request-path clock begins immediately before the blocking
   `conn.request` call. It includes request upload; it does not claim to begin
   when the first or last request byte reaches the socket or provider.
6. Streaming events update TTFB, first content, first reasoning, first visible
   content, first tool-call fragment, interchunk gaps, and end-to-end time.
   TTFB ends at the first nonempty bounded response-body chunk returned by the
   client read, not the first socket byte or first parsed SSE line. TTFT under
   `first_content` ends at a nonempty visible, reasoning, or refusal delta,
   never a tool-call fragment. End-to-end stops at
   `[DONE]`, or at response EOF when `[DONE]` is absent; a `finish_reason`
   records completion semantics but does not itself stop the response reader.
   The interchunk maximum is the widest elapsed gap between successive SSE
   events with a nonempty visible, reasoning, or refusal delta. It is not
   token-level inter-token latency; heartbeat, usage-only, and tool-call-only
   events do not advance it.
7. Tool-call fragments are assembled only long enough to verify a nonempty
   function name and arguments that decode to a JSON object. Argument content
   is not persisted.
8. The caller clocks measure from the scheduled target to the same observed
   events. They include queueing, connection setup, fallback requests,
   credential refresh, and configured transport retries.

The client opens a fresh HTTP/1.1 connection for every physical attempt. The
sealed run records this machine-readable contract and an optional closed
operator declaration of the real application's connection policy. If the
production policy is absent or differs, the canonical decision marks the
measurement `CAUTION` and capacity `INCONCLUSIVE`; a pooled keep-alive or HTTP/2
client can have materially different edge and connection pressure. The only
accepted matching declaration is `fresh_http1_per_physical_attempt`. It records
an operator assertion and does not claim the harness observed production.

The final-attempt metrics exclude connection setup but include request upload,
network and edge transit, endpoint work, and response transit. They must not be
labeled pure server compute time. The separate network probe resolves DNS
outside its timer and records `tcp_connect_min_ms` and
`tcp_connect_median_ms`. Those fields are not exact RTT and cannot be
subtracted to recover endpoint time.

Every runtime hostname resolution is deadline-bounded. Resolver helpers are
daemon-only and single-flight identical concurrent lookups, with a hard cap on
active unique lookups; a caller timeout or cancellation cannot later open a
socket or send a `POST`. Inference, endpoint-metadata, and OAuth M2M transports
also apply one absolute watchdog across DNS, connect, response headers, and
body consumption, so a peer that continually dribbles bytes cannot extend an
operation forever. These are client safety bounds, not endpoint latency
measurements.

## Outcome populations

These populations are not interchangeable:

- HTTP status describes transport response status when observed.
- A content stream means visible or reasoning content arrived.
- An acceptable outcome has no refusal marker and contains visible content or
  a structurally valid tool call, clean stream completion, and no parse errors.
- Semantic correctness is not measured.

Primary answer-latency percentiles use acceptable outcomes when current answer
observability fields exist. Errors and unacceptable outcomes remain in failure
and success-rate accounting. `finish_reason=length` is reported but is not by
itself an unacceptable outcome; truncation by the global cap is called out
separately because it means the requested workload was not reproduced.

A current row with an incomplete stream or any unrecoverable parse error is a
failure even when HTTP status is 200 or content arrived earlier. Such rows are
excluded from answer latency, calibration, endpoint-token throughput,
cache-fidelity, and cost arithmetic. They remain in attempted-request and error
denominators, so malformed survivors cannot make the run look faster or
cheaper.

Tool-call-only outcomes can be acceptable without first-visible-content
timing. Their first tool-call timing is reported separately.

The SSE parser retains bounded response `model`, `object`, and
`system_fingerprint` fields and SHA-256 of the response ID; the HTTP layer also
retains bounded Databricks `served-model-name`. Conflicting values inside one
stream are protocol errors. Across eligible HTTP 200 rows, multiple response
models or a response model outside an explicit request-body model invalidates a
single-model benchmark. Endpoint names are not expected OpenAI model values.
Instead, `served-model-name` is bound to active control-plane served entities;
an unexpected entity invalidates the result and incomplete binding is caution.

After response drain, stability windows are computed from persisted replay
rows using the same acceptable-outcome population as headline latency.
Failures and unacceptable outcomes remain separate per-window errors. A
failure-only window has zero event coverage and no latency percentile, so
survivor p95 cannot hide shedding. Endpoint configuration stability is a
separate pre-runner-target versus post-drain normalized control-plane-summary
comparison.
That summary is a selected subset: endpoint name, task, `route_optimized`,
READY state, and selected active served-entity identity, foundation-model,
workload/provisioning, version, and scale-to-zero fields. Changed subset
metadata invalidates the single-configuration result; incomplete capture
remains explicit uncertainty. Omitted control-plane fields and undocumented
data-plane revisions are outside this comparison.

HTTP 429 evidence uses only the valid integer terminal `status` captured on
each supplied request-operation row. It does not infer status from error text.
Preflight, explicit probe, sizing, calibration, and replay rows all contribute
to the exact row count, denominator, status-coverage count, and phase
breakdown. A row can contain multiple physical attempts, so this is not an
attempt-by-attempt HTTP-status counter; attempt admission events and
`request_attempts` remain separate evidence. Redacted response-body digests do
not split the stable 429 failure aggregate. Any 429 makes measurement validity
invalid and endpoint capacity inconclusive; no 429 with complete coverage
means only that a rejection was not observed, not that provider headroom
exists.

Local runtime-admission evidence is separate from HTTP 429. A guard denial
sends no physical `POST` for that attempt, permanently trips the command, and
makes the requested-load measurement invalid and capacity inconclusive. Guard
IDs, scope, sequence, reservations, transitions, run-local baseline/final
snapshots, and physical-attempt counters must reconcile. Missing or conflicting
evidence fails closed even if no HTTP 429 was captured.

## Pay-per-token planning boundary

The implemented `rate_limits` model is intentionally narrow: Databricks
Foundation Model API pay-per-token traffic on a direct
`/serving-endpoints/<name>/invocations` route. It is not a generic provider
quota engine and cannot be combined with provisioned-throughput pricing or an
unknown sizing-derived rate.

Databricks limits change independently of the harness. Operators must source
the exact current model, deployment-mode, and workspace-tier facts from the
official
[Foundation Model APIs limits and quotas](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits).
The snapshot records both provider-fact date `as_of` and operator review date
`verified_at`; `max_age_days` defines the positive freshness window. Runtime
age greater than that window is stale, while equal age is accepted. Missing,
future, invalid, or stale review evidence refuses with the same fail-closed
policy as an unsafe schedule.

The planner forecasts the harness in isolation. The complete serialized JSON
request uses an engineering bound of one token per UTF-8 byte plus a
harness-defined 64-token allowance per message and one additional 64-token
request allowance. Those constants are conservative harness assumptions, not a
Databricks-published tokenizer or chat-framing contract. The bound includes
roles, message metadata, model, tools, provider controls, and JSON syntax.
Synthetic content also uses the larger of configured characters/token and the
calibration hard maximum of 12, so post-authorization calibration cannot
enlarge planned input demand. Output is the offered `max_tokens` reservation.
Planning includes worst-case physical
attempts (transport retries plus usage-option and credential-refresh
fallbacks), setup traffic, calibration, and replay. Sweep planning constructs
and checks every exact requested rung and does not treat cooldown as proof of a
quota-window reset. A peak at or above `warning_utilization`, or a required
dimension that cannot be bounded, refuses paid inference.

The closed schema supports rolling
`input_tokens_per_minute`, `output_tokens_per_minute`, `queries_per_hour`, and
`queries_per_second` dimensions plus an inclusive integer
`request_bytes_max` ceiling over the exact serialized body of each physical
`POST`. The current Databricks limits page publishes a Foundation Model API
workspace limit of 200 QPS and 4 MB per request. The bundled snapshot uses a
conservative 4,000,000-byte ceiling because the source does not state a decimal
or binary MB convention; these facts must still be rechecked live.

Endpoint binding verifies only facts present in the captured normalized
serving-endpoint summary. The direct request route must name the configured
model,
`route_optimized` must be exactly false, each active served entity must have the
same name, and each must positively identify
`foundation_model.name=system.ai.<rate_limits.model>`. Absence of provisioned
fields is an additional check, not the positive identity proof. The standard
pay-per-token model accepts only absent/default request `service_tier`; an
observed non-default response tier invalidates its comparison. Workspace tier
and unrelated workspace traffic remain outside that evidence. Consequently,
`may_start=true` never sets provider-headroom proof.

The runtime guard enforces the same configured contract without waiting. Each
physical attempt reserves query count, offered output, conservative input, and
exact bytes atomically. Rolling totals must stay strictly below
`limit * warning_utilization`; request bytes may equal `request_bytes_max`.
Reservations remain provisional until the client can prove the `POST` did not
start (release) or observes response headers/ambiguous transport outcome
(conservative commit). A denial or internal state/clock uncertainty permanently
trips the guard. Shards receive deterministic static partitions of each
integer warning budget. The guard covers one harness command only and cannot
observe unrelated workspace traffic or provider burst state.

## Retry model

The configured transport retry count is an integer from 0 through 2 and
defaults to zero. Connection and request attempt counters show whether a
physical `POST` was attempted. When a
configured transport retry occurs, `retry_reasons` distinguishes a connection
failure before `POST` from a transport error after a possible `POST`; the
latter can duplicate inference and billing. A final failure that is not
retried remains in `error` and does not add a retry reason.

The streamed-usage compatibility fallback and one credential-refresh retry can
also create a second physical `POST` independently of the configured transport
retry count. Every row records connection attempts, physical request attempts,
and retry reasons. The system offers no exactly-once guarantee.

Operator cancellation is cooperative. The runner sets a shared cancellation
event, then best-effort shuts down all tracked active client sockets to wake
blocked I/O, before cancelling queued futures. It does not cross-thread close
the `HTTPConnection`, because clearing its socket can let a racing request
auto-connect again; the owning worker closes it in `finally`. Clients check the
event at worker entry, immediately before the first `POST`, before each retry,
and after transport I/O wakes. A shutdown-induced I/O error is returned as
cancelled and never retried. Queued work that is successfully cancelled is
best-effort journaled as unsent; a `POST` already on the wire cannot be recalled
and its provider outcome and billing remain ambiguous. The exception path syncs
recoverable rows where possible and leaves an unsealed writing artifact rather
than a completed benchmark.

Cost accounting is intentionally narrower than token reporting. Rates are
unverified operator input, never fetched pricing. Per-token arithmetic covers
replay rows only. Aggregate per-token totals and provisioned effective rates
are withheld unless every row is either known unsent or has one clean,
complete, sane-usage response from exactly one physical attempt with no retry
marker. Ambiguous retries, multiple posts, unknown attempt counts,
corrupt/incomplete streams, or usage gaps leave only a labeled measured-subset
diagnostic because billed usage and the token-throughput denominator across all
attempts are not known.

## Evidence model

### Exact-analysis resource envelope

The current analyzer computes exact percentiles from materialized request
records, so one run is limited to **50,000 logical rows total** across replay,
calibration, concurrency-sizing probes, and carried command setup traffic. A
sweep applies the same 50,000-row limit to the cumulative population of every
rung, and merge applies it to the combined input population. Fixed schedules,
timestamp traces, and sizing ceilings are counted before credential lookup,
control-plane access, preflight, or inference traffic. Exceeding the limit is
a refusal, not sampling. Raising it requires bounded-memory streaming
statistics and new resource tests.

Generated CLI sizing configs reserve setup, calibration, and sizing-probe rows
first. Their QPS ceiling uses the remaining replay budget with eight Poisson
standard deviations of headroom; prevalidation then counts the actual seeded
schedule and still refuses any overage. Shorter duration or an explicit
fixed-rate workload within the envelope is the supported escape path.

Input and artifact bounds are part of that contract: profiles and timestamp
traces are at most 16 MiB, prompt inputs at most 64 MiB, one decoded prompt or
prompt-JSONL line at most 4 MiB, and one timestamp line at most 64 KiB.
Manifest-bound metadata artifacts are at most 16 MiB. A request journal is at
most 256 MiB, 50,000 rows, and 256 KiB per JSONL row. These readers require a
stable regular file, reject symlinks and special files, and detect replacement,
growth, or truncation while reading.

`start.json` is the pre-traffic and in-progress provenance record. It includes
redacted effective configuration, workload/input digests, source identity,
logical/execution/artifact IDs, target evidence when available, exact schedule
and shard identities, calibration results, runtime-guard baseline/final state,
and pre/post-drain endpoint metadata stability when captured.

High-level preflight/probe traffic has its own standard manifest-v3 setup
artifact. Its active journal syncs every completed row rather than every 16.
The sealed summary explicitly sets performance, SLA, and capacity result flags
false. A passing command duplicates those metadata-only rows into the measured
artifact exactly once so that artifact's quota population is complete; the
setup artifact remains the crash/refusal-visible evidence boundary.

Generated rerun configs retain durable external input paths and a closed
`input_expectations` map of SHA-256 plus byte count. The manifest-bound run
evidence records captured input identity and size, not prompt content. This
detects changed rerun inputs but is not a self-contained copy of a customer
prompt dataset.

`requests.jsonl.partial` is the active journal. It is synced every 16 appended
rows by default, after measured replay drains before the rows are reread, and
during finalization or exception cleanup. Sizing and calibration transitions
do not force an additional journal sync. An interrupted final fragment can be
ignored, but a directory retaining `.traffic-replay-writing` is never a
completed benchmark.

On success, manifest schema v3 binds `requests.jsonl`, `summary.json`,
`report.md`, `report.html`, and `start.json` by SHA-256 and byte count, plus a
row count for the request journal. The writer stores the artifact identity,
manifest digest and size, and request-row count in the completion marker and
promotes that marker last. Aggregate readers verify the entire marker,
manifest, and request-journal chain before accepting an input.

Aggregate readers reject incomplete runs, unsupported schemas, nonregular or
symlinked artifacts, integrity mismatches, duplicate artifacts, and malformed
identity data. `--force` can produce an explicitly INVALID diagnostic merge
for compatibility or incomplete shard coverage; it cannot turn corrupt or
unsealed evidence into valid evidence.

### Canonical report decisions and presentation

The writer redacts one canonical summary, adds a deterministic
`decision_schema_version=1` object, and serializes that object to
`summary.json`. Both human renderers consume the same summary, so
`report.html` and `report.md` carry the same codes, labels, reasons, and
tested-load facts for five independent dimensions:

1. evidence integrity;
2. measurement validity;
3. configured acceptance checks;
4. quota state; and
5. endpoint capacity at the tested load.

Independence prevents a clean latency check from hiding a quota rejection or
an invalid measurement from erasing the observed acceptance outcome. A
retained acceptance pass is qualified when measurement validity is not
`VALID`. Endpoint capacity
can say only held/not-held at a verified, bound test point; the model never
sets endpoint-ceiling or provider-headroom proof.

Response identity, the normalized pre-run/post-drain endpoint-stability
comparison, and runtime-admission reconciliation are evidence gates rather
than additional canonical decisions. Identity and stability feed measurement
validity; admission feeds quota state and measurement validity. The canonical
object remains exactly the five dimensions above.

Stored run reports use evidence state `VERIFY_REQUIRED`. A summary cannot
authenticate the future manifest that will contain its own bytes, so integrity
can become `VERIFIED` only in an external verification context. This is
separate from measurement validity and is not rewritten into a sealed report.

The run HTML contains inline CSS and SVG only: no JavaScript, remote assets,
remote fonts, or network fetches. Responsive rules stack decision cards and
charts, keep dense tables locally scrollable, and retain full reasons in an
expandable evidence block. Print rules remove navigation, repeat table
headers, avoid splitting key cards where practical, and substitute visible
measurement-evidence text for interactive details. Browser-specific print
headers, margins, and pagination remain outside the artifact contract.

The print view carries an `UNSEALED PRINT/PDF DERIVATIVE` stamp. Browser PDF is
not manifest-bound; internal hashes are not a digital signature.

### Comparison artifact and baseline

`compare` first verifies each source against its internal completion-chain
hashes and bindings. It rescans the manifest-bound journal for 429s across all
phases; a 429, a summary/journal disagreement, an explicitly invalid source,
or a compatibility mismatch makes the comparison diagnostic-only.

The first positional input is always the baseline. Every later input is a
candidate. HTML absolute delta is candidate minus baseline; percent delta is
that difference divided by the absolute baseline and is undefined for a zero
baseline. A `VALID` comparison can label only arithmetic direction; without
repeat-run uncertainty and a practical-effect threshold it never calls a
change an improvement or regression. Measurement warnings produce a
`QUALIFIED`, diagnostic-only comparison. Compatibility/source-validity
failures produce an `INVALID` comparison. Both keep neutral diagnostic values.

The comparison writer claims a fresh directory, writes `comparison.md` and
the responsive/printable/self-contained `comparison.html`, binds both in a
manifest-v3 `artifact_type=comparison`, promotes the completion marker last,
and verifies the completed output before returning. The manifest records the
exact source manifest and summary identities. Markdown provides
portable side-by-side absolute values and warnings; HTML additionally provides
the baseline/delta matrix. Source runs remain the underlying evidence, and the
current CLI has no standalone `verify-comparison` command.

## Boundedness and backpressure

`max_concurrency` bounds worker threads. `max_pending_requests` bounds running
plus queued futures; when omitted it is computed as
`max(2 * max_concurrency, max_concurrency + 1)`. The journal avoids a
run-sized in-memory result list while traffic is active. However, the complete
unsharded schedule and profile-mode sampled workload arrays scale with the
global request count. Exact percentile summarization also rereads persisted
replay rows after the workload drains. The tool is therefore bounded in active
client work, not constant-memory in total run size.

A pending-limit drop is a client-side failure and no inference request is sent
for that row. A saturated pool appears in exact queue wait and caller latency;
the dispatcher lag alone cannot detect executor queueing.

## Security boundaries

- Bearer credentials are allowed only over HTTPS or explicit loopback HTTP.
- `base_url` is an origin, while the request path is configured separately.
- Named Databricks profiles are bound to the same normalized origin and fail
  closed on mismatch or token-resolution failure.
- `extra_body` is persisted as request evidence, so secret-like keys and
  credential-shaped values are rejected recursively before target traffic.
- Artifact configuration and strings pass through semantic secret redaction.
- Non-200 bodies are represented only by status, sampled byte length, and a
  truncated body digest.
- Input prompts exist only in the operator source and private temporary
  snapshots needed for replay. Saved configs and run artifacts retain path or
  basename plus SHA-256/byte-count identity as applicable, not prompt content.
  Tool argument content is likewise not persisted.

## Portability boundary

The client implements a tested subset of streamed Chat Completions behavior.
An endpoint described as compatible can still differ in route, authentication,
model selection, request controls, SSE framing, tool-call deltas, usage fields,
tokenizer, cache semantics, retry safety, and quotas. Provider comparison
requires conformance testing and achieved-workload evidence; changing only the
URL is not a portability proof.
