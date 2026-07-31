# Architecture

## Components and data flow

![components and data flow](diagrams/architecture.svg)

## Per-request sequence

![request sequence](diagrams/request-sequence.svg)

## Design decisions and their tradeoffs

**Standard library HTTP client, threads not asyncio.** One dependency
(numpy) instead of an async stack. Blocking socket reads release the GIL,
so hundreds of concurrent streams are fine. The cost is thread overhead at
extreme concurrency. The mitigation is honest measurement (dispatch lag is
reported per request) plus `shard` support to split a schedule across
processes or machines when a single client saturates. For a single-node
endpoint evaluation, the endpoint saturates long before the client does.

**TTFT split: first token, reasoning, first visible.** Real servers send a
role-only chunk immediately on connection, and timing off that byte would
flatter the endpoint. The bundled mock reproduces this trap and the suite
asserts the client keys on the first content delta, not the first byte. For
thinking models that stream a reasoning channel before visible output the
client records three timestamps per request: `ttft` (first token of either
kind, kept for back compat), `ttfr` (first reasoning delta) and `ttfv`
(first visible content delta). The run config's `ttft_definition`
(`first_content` or `first_visible`) chooses which one the SLA scorecard
scores, and the report flags when they diverge so the definition is agreed
before a number is quoted.

**Cache is constructed, never asserted.** The pool guarantees the traffic
STRUCTURE (shared leading text, popularity skew, cold first-uses). Whether
a request actually hits is the endpoint's business, read back from the
usage block, with the exact field name recorded. When an endpoint doesn't
report cached tokens, the report says NOT REPORTED rather than guessing.

**Synthetic text: shape, not content.** Request text is generated to hit
the token sizes and cache structure, not to mean anything. Serving latency
and throughput depend on token counts, cache hits, and arrival timing, so
the numbers transfer. Content-dependent behavior (quality, guardrail
triggers, semantic routing) is out of scope and needs real prompts.

**Token counts are targeted, reported, and corrected.** Text is generated
to a characters-per-token budget, cpt is recalibrated during the warmup
pass from endpoint-reported prompt_tokens, and the residual targeting
error is printed in every report. Endpoint-reported counts are the source
of truth everywhere.

**Determinism by seed.** Profiles, pool assignment, schedules, documents
and suffixes are all seeded. Two runs with the same config are the same
experiment except for the endpoint's behavior, which is the variable under
test.

**Instrument validated before use.** `python3 -m traffic_replay validate`
runs the whole pipeline against the bundled mock, whose latency model is
known by construction and whose server-side truth log is joined back to
client measurements by request id. The suite asserts error bounds, and the
validate command prints them. Numbers from an unvalidated instrument aren't
numbers.

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
- Mid-stream stalls: the widest interchunk gap is recorded per request; when
  the profile sets `acceptance_targets.interchunk_ms`, requests over it count
  as SLA breaches against the success rate.
