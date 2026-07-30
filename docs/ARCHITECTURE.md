# Architecture

## Components and data flow

```mermaid
flowchart TB
    subgraph inputs [Inputs]
        PJ[profile JSON\nstated quantiles + provenance label]
        RJ[run config JSON\nendpoint, schedule, scale, caps]
    end

    PJ --> PR[profile.py\nlognormal / logit-normal closed-form fits\nper-request: input, output, cache target]
    PR --> PP[prefix_pool.py\nbucketed docs, Zipf popularity\nper-request: doc_id, prefix tokens]
    RJ --> SC[schedule.py\ntwo-state modulated Poisson\nabsolute timestamps, rate_scale, shard]

    PP --> TG[textgen.py\ndeterministic doc text, unique suffixes\ncpt calibrated against endpoint truth]
    TG --> RU[runner.py\ncalibration pass, paced dispatch\nbounded ThreadPoolExecutor]
    SC --> RU

    RU --> CL[client.py\nhttp.client streaming POST\nmonotonic TTFB/TTFT/E2E\nusage extraction via sse.py]
    CL --> EP[(endpoint\nreal PT / pay-per-token / mock)]
    EP --> CL
    CL --> ME[metrics.py\npercentiles, achieved cache,\ntoken targeting, dispatch lag]
    ME --> OUT[(results dir\nrequests.jsonl / summary.json / report.md)]
```

## Per-request sequence

```mermaid
sequenceDiagram
    participant R as runner (dispatcher)
    participant W as worker thread
    participant E as endpoint
    R->>R: sleep until scheduled timestamp
    R->>W: submit(messages, max_tokens, request_id)
    Note over R: lateness recorded as dispatch_lag_ms
    W->>E: POST chat completion, stream=true (t_send)
    E-->>W: SSE role-only chunk (TTFB; NOT TTFT)
    E-->>W: SSE first content delta (TTFT)
    E-->>W: ... content chunks ...
    E-->>W: final chunk with usage + [DONE] (E2E)
    W->>W: extract usage: prompt / completion / cached tokens
    W-->>R: RequestResult (all timings + intended vs reported)
```

## Design decisions and their tradeoffs

**Standard library HTTP client, threads not asyncio.** One dependency
(numpy) instead of an async stack. Blocking socket reads release the GIL,
so hundreds of concurrent streams are fine. The cost is thread overhead at
extreme concurrency; the mitigation is honest measurement (dispatch lag is
reported per request) plus `shard` support to split a schedule across
processes or machines when a single client saturates. For a single-node
endpoint evaluation, the endpoint saturates long before the client does.

**TTFT keys on first content delta, not first byte.** Real servers send a
role-only chunk immediately on connection; timing that would flatter the
endpoint. The bundled mock deliberately reproduces this trap and the test
suite asserts the client does not fall into it.

**Cache is constructed, never asserted.** The pool guarantees the traffic
STRUCTURE (shared leading text, popularity skew, cold first-uses). Whether
a request actually hits is the endpoint's business, read back from the
usage block, with the exact field name recorded. When an endpoint does not
report cached tokens, the report says NOT REPORTED rather than guessing.

**Token counts are targeted, reported, and corrected.** Text is generated
to a characters-per-token budget, cpt is recalibrated during the warmup
pass from endpoint-reported prompt_tokens, and the residual targeting
error is printed in every report. Endpoint-reported counts are the source
of truth everywhere.

**Determinism by seed.** Profiles, pool assignment, schedules, documents
and suffixes are all seeded. Two runs with the same config are the same
experiment except for the endpoint's behavior, which is the variable under
test.

**Instrument validated before use.** `python -m traffic_replay validate`
runs the whole pipeline against the bundled mock, whose latency model is
known by construction and whose server-side truth log is joined back to
client measurements by request id. The suite asserts error bounds; the
validate command prints them. Numbers from an unvalidated instrument are
not numbers.

## Failure modes handled

- Endpoint rejects `stream_options.include_usage`: detected on first 400,
  learned, retried without it, remembered for the rest of the run.
- Connection errors: one retry, counted and reported per request.
- Stream ends with no content: recorded as a failure with reason, never a
  silent zero.
- Non-200: status and first 300 chars of the body land in the result row.
- Client saturation: visible as dispatch lag percentiles, not blended into
  endpoint latency.
- Cold cache at run start: warmup/calibration phase is logged separately
  (`phase` field) and excluded from the replay summary.
