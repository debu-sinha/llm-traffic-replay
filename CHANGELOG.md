# Changelog

## 0.4.1

### One command from an endpoint URL to a report

`benchmark` takes a host, an endpoint name and rough token sizes. Sizes take
`p50` or `p50,p95`, so `--input-tokens 10000,24000` replaces authoring a
profile JSON. It writes the derived profile and the exact run config next to
the results.

Before any load it sends two requests and says what the endpoint does:
whether auth works, whether token usage and cached tokens are reported, and
whether this is a reasoning model. On a reasoning model that produced no
visible answer inside 512 tokens it says so, warns that the configured
output budget will produce none either, and scores TTFT on the first visible
token. Verified against a live reasoning endpoint.

### Latency as the caller experienced it

The latency clock starts when a worker sends, so a request that waited in
the client queue reported only what the endpoint took once it went out. That
is how a saturated load generator reports a healthy tail. Measured against
an endpoint taking 200 ms every time with the client 10 seconds behind: e2e
p95 read 200 ms while a caller asking on schedule waited 10,200 ms.

Every request carries `queue_wait_ms`, and the summary carries
`ttft_corrected_ms` and `e2e_corrected_ms` measured from intended arrival.
Both views are reported. Service time belongs to the endpoint, corrected
belongs to the user, and a run where they disagree was not driving the load
it claimed.

### A green verdict now requires evidence

Enumerating failure modes kept leaving doors open. A run with an 8 percent
error rate and one passing latency target printed "meets every acceptance
target". Errors, a client-saturation warning, a concurrency warning, an
unstable stability verdict, missing coverage on the scored metric, or a
sample too small to support the quantile a target is scored on, each
downgrade the verdict to a caution naming the reason. Both reports render the same verdict from one function, having
previously disagreed.

### Numbers that were wrong

- Throughput divided token totals by the send window while counting
  generations that finished during the drain. About 61 percent high for a 99
  second window with 60 second generations. It now divides by the
  observation interval, and the arrival rate stays on the send span.
- Achieved arrival rate used `n/span` while the offered rate used
  `(n-1)/span`, so the two disagreed on an identical schedule.
- Token-targeting error was `abs(median(ratio) - 1)`, which cancels
  symmetric error: half the requests at 0.5x and half at 1.5x reported 0
  percent. It is the median absolute error now.
- `answer_rate` was labelled "of attempted" while computed over requests
  that returned HTTP 200, so a run losing 8 of 100 printed "100.0% of
  attempted".
- Concurrency anchored its window on completion times, so one straggler
  stretched the span into its own drain and collapsed the reported
  concurrency to 1. The window is bounded by send times, the peak is a true
  peak over the whole run, and a retried row spans its whole occupancy.
- `calibrate_n` at or above the schedule length consumed every arrival and
  reported "0 requests" on a run that really sent some. It raises now.
- A shard's schedule block reported the whole run's request count.
- Missing token usage was silently summed as zero. Coverage is reported and
  warns below 99 percent.

### Provenance

Each run writes `manifest.json`: harness version and latency basis, git
commit and dirty flag, profile name, path, provenance and content hash,
seed, endpoint base URL, model and metadata, request parameters, schedule,
and Python, numpy and platform versions. Run metadata previously omitted the
base URL and model, which let `compare` and `merge` pool two providers
sitting behind the same route.

### Also

- `complete_answers` renamed `answered`. Most generations stop at the
  requested output length, so the old name told anyone reading
  `summary.json` that truncated answers were complete.
- `truncated_by_global_cap` separates truncation at a request's own sampled
  target, which means the replay worked, from truncation by the global cap,
  which means the run never reproduced the profile's output distribution.
- The report no longer tells profile-mode users to raise
  `max_output_tokens_cap`, which cannot work because the per-request budget
  is the smaller of that and the sampled value.
- Tests bind ephemeral ports, so both runners can run at once.

## 0.4.0

### Transport success is no longer reported as answer success

A reasoning model that spends its whole token budget thinking returns
HTTP 200, a well formed stream, a finish reason, and nothing a user could
read. Every one of those counted as a success, so a run where not a single
request produced a readable answer reported a 100 percent success rate.

Measured on a real reasoning endpoint: 187 requests, zero errors, and 132 of
them returned 200 with no visible content. The old report showed a green
"meets every acceptance target" banner over it.

Each request now records `visible_content_seen`, `stream_complete`,
`reasoning_seen`, `truncated`, `parse_errors` and `max_tokens_requested`.
The summary gains an `answers` block separating attempted, transport-ok, and
answered, and both reports lead with it.

- A response that returns 200 with no visible content counts against the
  success rate.
- Truncation does NOT count as a failure. This harness caps `max_tokens` at
  the sampled output size on purpose, so ending on "length" is the expected
  way to hit a target output length. It is reported as its own rate instead.
- A target with no measurement behind it is no longer scored as a pass. The
  green banner is downgraded when any scored target is unmeasured, and a run
  where nothing was answered renders INVALID.
- `report.md` now carries a verdict line. It had none, and it is the file
  that gets pasted into email.
- The SLA table renders "not measured" rather than a literal `None`.

### TTFV percentiles say what they leave out

When the SLA scores the first visible token, requests that never produced one
carry no `ttfv_ms` and silently dropped out of the percentile. The report
printed a clean p50 computed over the fastest subset. Percentile tables now
carry `missing` and `of`, the reasoning note names the subset, and the
scorecard cautions when more than 5 percent of successful requests never
produced the token being scored.

### Concurrency

- `concurrency` is a config knob. A short sizing pass measures service time
  and derives the arrival rate and pool size.
- Achieved concurrency is measured by interval overlap and printed next to
  what was asked for.
- The rate is derived from service time measured WITHOUT load, and service
  time rises under load, so this design overshoots. Both directions now warn.
  Previously only under-delivery did, which let a run labeled "30 concurrent"
  that actually held 65 go out clean.
- Achieved concurrency is measured by an exact event sweep. An earlier
  version of this entry described a 41-sample approximation, which 0.4.1
  replaced.
- Sharded runs compare against their own share of the concurrency rather than
  the unsharded total, which used to report every shard as falling short.

### Added

- `quickstart` subcommand: writes a runnable config from a host, an endpoint
  and a profile, so getting to a first run does not mean hand-editing JSON.
- `endpoint.auth_profile` reads `~/.databrickscfg`. PAT tokens are used
  directly, OAuth profiles shell out to `databricks auth token`.
- SLA targets say where they came from, and bundled illustrative targets are
  flagged so they do not read as agreed numbers.

## 0.3.0

### Changed what the latency numbers mean (read this before comparing runs)

DNS, TCP and TLS setup is no longer inside `ttft_ms`, `ttfb_ms` or
`e2e_ms`. Connection setup is timed separately and reported as `connect_ms`.

Up to 0.2.x the connection was established lazily inside the timed region, so
every request paid a fresh handshake and that cost sat inside TTFT. This
harness opens one connection per request, while production clients pool
connections and pay the handshake once, so the old numbers overstated
per-request TTFT by roughly a handshake, which measured about 280 ms from
where we ran the client. That figure is client-to-region distance, not
endpoint behavior.

The new basis is the more representative one, but it isn't the old one. A
0.2.x TTFT and a 0.3.x TTFT are different measurements:

- Don't put them in the same table. `compare` now warns when the runs it is
  given come from different harness versions.
- Re-run any baseline you plan to compare against.
- Every `summary.json` now carries `harness_version` and `latency_basis`, and
  both reports print the basis, so a saved report says which convention
  produced it.

### Fixed: client saturation was invisible

The generator is open loop: a dispatcher thread submits into a bounded thread
pool without waiting for responses. `dispatch_lag_ms` was documented as the
signal that the client had fallen behind, but `ThreadPoolExecutor.submit()`
queues rather than blocking, and the lag is stamped before the submit, so it
could never see a saturated pool.

Measured against a deliberately slow endpoint, 8 requests/second offered into
a pool of 2: dispatch lag p95 reported 5 ms while wire lateness p95 was
76.9 seconds and the run delivered 2.0 of the 8.5 requests/second asked for. Endpoint latency stayed clean throughout, so the
old claim that saturation is not blended into latency held. The claim that it
surfaces as dispatch lag did not.

Reports now carry `wire_lateness_ms`, computed from `first_send_unix` (the
moment a request's FIRST attempt went out, so a retry cannot charge the
endpoint's delay to the client) against when the schedule wanted it, and print
a client-saturation caution above the tables when the run-average rate falls
more than 20 percent below the schedule, or when wire lateness passes a
second. Each case gets its own conclusion: a rate shortfall says the run
delivered fewer requests per second than asked for, while a run that held its
average but arrived late says the load arrived reshaped. Neither names a
cause, because a full pool looks the same whether the client could not keep
up or the endpoint back-pressured it, and the stability card is what
separates them.

Verified on a healthy run too, against an endpoint recording its own receive
times so the arrival rate is checked server-side: 20 requests/second offered,
20.7 observed by the endpoint, 1252 received against 1248 replay rows plus 4
calibration. A rate ladder tracked the target within 1 percent (50 to 50.2,
100 to 100.4, 200 to 200.6) and stayed silent, then bent at 400 where it
delivered 272.6 and the caution fired.

Re-run end to end after the measurement moved onto `first_send_unix`: a
healthy run delivered 19.0 of 20 requests/second with wire lateness p95 of
18 ms and no caution, and a saturated one (8 offered into a pool of 2)
delivered 2.0 with wire lateness p95 of 76.9 seconds and the caution fired.
Both `requests.jsonl` carry the new field, and it sits a handshake ahead of
`t_send_unix` as intended.

### Added

- **Sample size gate.** The report counts the requests behind the percentiles
  and cautions under 100 (p99 unstable) and under 30 (whole tail indicative
  only). Zero successful requests says so rather than describing a p99.
- **Stability over time.** Requests are bucketed into 60-second windows with
  per-window TTFT and E2E p95, plus a per-window error count. A run is
  reported as `failing`, decided on error rate rather than the 1.3x rule,
  when one window lost more than 5 percent of its requests while the others
  held, or when every window is losing more than 10 percent. Latency
  percentiles only cover requests that came back, and a collapsing
  endpoint's survivors are the fast ones, so the error rule is asked first
  and sizes its floor on attempted requests rather than successful ones. A
  run is unstable when the worst counted window
  is more than 1.3x the best in either direction, reported as `warming`,
  `degrading`, `spike` or `variable`. Windows too small to support a p95 are
  printed but excluded from the verdict, and a direction is only named with at
  least three counted windows, because two points can't separate a trend from
  noise. Merged runs do not get a stability verdict, since pooled shards would
  span the gaps between them.
- **Connection setup timing.** `connect_ms` percentiles, labeled as excluded
  from TTFT and as an upper bound on network distance rather than the
  per-request network cost a pooled client pays.
- **Endpoint under test.** The run reads the serving endpoint's own config and
  prints what was measured (task, route-optimized, ready state, and served
  entity workload type and size where the endpoint has a provisioned entity).
  Best effort: an HTTP error or exception logs one line to stderr naming the
  failure class, never the token or body, and the run continues without the
  card. A path with no serving-endpoints segment is skipped silently, since
  that is the normal third-party-provider case. The
  endpoint name is parsed from the configured path, so custom endpoint names
  work. New setting `capture_endpoint_metadata`, default true.
- **Prompt replay caution.** Prompts mode cycles the supplied prompts across
  the schedule, so any run with more requests than prompts sends verbatim
  repeats that the endpoint prompt cache serves. The report now states how
  many requests were repeats and cautions that the achieved cache fraction
  describes the replay rather than production traffic. Measured: 10 prompts
  over 100 requests went from 0% achieved cache on the first pass to 92% on
  the repeats.
- **Cost in DBUs.** Per-request, per-1000-request and per-minute DBU cost from
  user-supplied rates, plus the DBUs the prompt cache saved.
- Self-contained HTML report, prompts mode, `extra_body` passthrough, and
  reasoning-token reporting with a stream-counted fallback.

## 0.2.0

Sharded runs and `merge`, provider `compare`, SLA scorecard, interchunk
stall tracking, believability block.

## 0.1.0

Profile-driven load generation, prefix pool for constructed cache reuse,
bursty arrival model, streaming client with TTFT capture, mock-server
validation.
