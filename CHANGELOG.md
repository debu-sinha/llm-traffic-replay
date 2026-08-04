# Changelog

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
