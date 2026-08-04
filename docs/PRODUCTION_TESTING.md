# Production testing, step by step

Three stages, in order. Each stage has an explicit purpose and an explicit
list of what its numbers may and may not be used for. Do not skip stages.

## Stage 0: prove the instrument (no endpoint, 2 minutes)

```bash
python3 -m pytest                      # the full suite
python3 -m traffic_replay validate     # end-to-end vs bundled mock
```

PASS criteria: all tests green. Validate reports TTFT error p95 under the
tolerance (default 60 ms, typical laptop result is under 5 ms). If this
fails on the machine you plan to run from, fix that first, nothing
downstream is meaningful.

## Stage 1: smoke test on shared capacity (client correctness only)

Purpose: prove auth, streaming, TTFT capture, usage parsing and the cache
field name against a REAL serving stack, cheaply, before the dedicated
endpoint exists or without spending its capacity.

1. Copy `configs/run_smoke.json`, set:
   - `endpoint.base_url`: your workspace host, e.g.
     `https://<workspace>.cloud.databricks.com`
   - `endpoint.path`: `/serving-endpoints/<pay-per-token-endpoint>/invocations`
2. Auth: `export DATABRICKS_TOKEN=<your PAT>` on the machine running the
   harness. Inside a Databricks notebook you can instead read the ambient
   context token. See `notebooks/` if present.
3. Run: `python3 -m traffic_replay run --config configs/run_smoke.json`
   (about 60 seconds at 1 to 6 QPS, small prompts, `max_output_tokens_cap`
   32, cost is negligible).
4. Open `results/smoke/<ts>/report.html` in a browser (or read `report.md`)
   and check, in order:
   - error rate 0%, or every failure explained
   - TTFT populated for every ok request (role-only chunks not counted)
   - `prompt_tokens` reported, token targeting error under ~15% after
     calibration
   - cached tokens: reported or NOT REPORTED, and if reported, WHICH field
     (this tells you what the PT run will be able to show)
   - wire lateness p95 in the tens of ms and no client-saturation caution
     (the client is loafing at 6 QPS). Real healthy runs land between about
     20 and 140 ms depending on prompt size, so read the caution rather than
     a fixed number.

**The latency numbers from stage 1 are not evidence of anything.** Shared
pay-per-token capacity says nothing about a dedicated endpoint. The run
config's label says this and the label prints in the report. Leave it in.

## Stage 1b: is this a reasoning model? (do this before Stage 2)

Skip this and every number downstream can be wrong in the same direction.

A reasoning model emits its thinking on a separate channel before the answer.
Those tokens are billed as output tokens and they count against `max_tokens`.
If the budget runs out mid-thought, the endpoint returns HTTP 200, a well
formed stream, and a finish reason, with nothing a user could read. That is a
successful request by every transport measure and a failed one by every
measure that matters.

Run a short probe first:

```bash
python3 -m traffic_replay quickstart \
  --host https://<workspace>.cloud.databricks.com \
  --endpoint <endpoint> --profile <your profile> \
  --concurrency 2 --duration 60 --out configs/probe.json
python3 -m traffic_replay run --config configs/probe.json
```

Then read the `answers` block in `report.md`, not the latency table:

- `produced a readable answer` well below `returned HTTP 200` means the model
  is spending your output budget on reasoning. Stop. The latency numbers
  describe only the requests that finished thinking, which are the fastest
  ones, and the report says so.
- `reasoning model detected` in the report means TTFT and TTFV are different
  numbers. TTFT is the first token of any kind, including thinking. TTFV is
  the first token a user could see. A user-facing SLA describes TTFV, so set
  `"ttft_definition": "first_visible"` in the run config or the scorecard
  scores the wrong one.

### Turning thinking off

If production runs without thinking, the request has to say so, and the flag
that works is model-specific. Put it in `endpoint.extra_body`:

```json
"extra_body": {"reasoning_effort": "none"}
```

Verify it took effect rather than assuming. Some endpoints accept a flag and
ignore it silently, which is worse than rejecting it, because the config
looks correct. On one Databricks-hosted reasoning model, measured with a real
10,000 token prompt:

| setting | reasoning emitted | visible output |
|---|---|---|
| `reasoning_effort: "none"` | none | yes |
| `thinking: {"type": "disabled"}` | more than the default | yes |
| `enable_thinking: false` | unchanged | none |

Two of the three flags most people reach for first did nothing. The check
that catches it is the `answers` block plus `reasoning tokens` in the report,
not the absence of an error.

### Raising the budget is not the fix

Giving the model more room buys a longer think, not a faster answer. Measured
on the same endpoint and prompt shape:

| output budget | produced an answer | end-to-end p50 |
|---|---|---|
| 40 / 90 | 0 of 535 | 1.5 s |
| 1,200 / 2,000 | 55 of 187 | 13.6 s |
| 4,000 / 8,000 | 31 of 72 | 28.1 s |

If a sub-second first-token target matters, the decision is the model or the
mode, not the token budget.

## Stage 2: provisioned throughput endpoint, stepped load

Purpose: the real measurement. Requires the dedicated endpoint and an
agreed profile config (until the customer's exact dataset lands, the
bundled profile carries its ASSUMPTION label and so does every report).

1. Copy `configs/run_pt_full.json`, set base_url and the PT endpoint path.
2. Agree the acceptance targets in writing before the first run, per
   workload class and held across the burst schedule rather than on average.
   Ask for the interchunk-stall threshold too. The moment they give a number, put it
   in the profile as `acceptance_targets.interchunk_ms` and the scorecard
   counts breaches against the success rate. The targets in the bundled
   profile are illustrative, so replace them with yours before the run.
3. Step the load, one knob, in this order, reading the believability block
   between steps:
   - `rate_scale`: 0.1 -> 0.25 -> 0.5 -> 1.0
   - at each step: error rate first, then wire lateness and the achieved
     rate against the schedule (is the client still delivering the load: the
     report raises a caution at a 20 percent run-average shortfall or a wire
     lateness p95 over one second),
     then achieved cache fraction, THEN latency percentiles.
4. Two workload classes = two profile configs = two runs. Do not blend
   them into one table.
5. If wire lateness p95 passes ~1 s at high rate_scale, or the report
   raises the client-saturation caution,
   split the schedule across processes: `shard_index`/`shard_total` in two
   run configs on two machines, then pool their output dirs with
   `python3 -m traffic_replay merge`. To put Databricks PT next to another
   provider on identical measurement, run each and
   `python3 -m traffic_replay compare` them. It warns when their achieved
   cache rates differ enough to make the latency comparison unfair.
6. Warm vs cold: the summary excludes the calibration phase. Per-document
   first uses remain in the data (`doc_id` per row) so cold-start behavior
   can be reported separately rather than averaged away.

Record for every run, alongside the results dir: endpoint config (GPU
type, node count), model, date/time, profile config hash, and who ran it.
A number that cannot be tied to its configuration does not go in front of
anyone.

## Stage 3: customer-dataset replay

When the exact production profile lands (as distributions, or as a request
log reduced to per-request token counts and repeat structure):

1. Convert it to a profile JSON (same four fields, set `provenance` to the
   dataset name and date, clear the ASSUMPTION label).
2. Re-run stage 2 with the new profile. Config swap only. The harness,
   schedule and measurement do not change, so stage-2 and stage-3 results
   are directly comparable.
3. If the customer prefers to generate load from their side: they run the
   generator half (profile + pool + schedule + textgen produce the request
   stream), both sides measure, and the readout compares notes on an
   agreed instrument.

## Interpreting the readout honestly

- Transport success is not answer success. Read the `answers` block before
  the latency table. A run can be 100 percent HTTP 200 and 0 percent useful,
  and that is the shape a reasoning model fails in.
- The harness hits a target output length by setting `max_tokens` to the
  sampled value, so every request ends on `length` by construction and
  `completion_tokens` equals what was asked for. That is deliberate, it is
  how output size is controlled, and it has one consequence worth stating in
  any deck: you measured the time to generate N tokens, where N came from the
  profile, NOT the model's natural answer length. If the model would have
  produced more, end-to-end understates production. Check the natural length
  once with a generous cap and a small sample, and if it sits well above the
  profile's p50, say so next to the number.
- Comparing two model configurations needs the cache controlled. The prefix
  pool is seeded, so two runs with the same `seed` send the same leading
  tokens, and the second one inherits a prompt cache the first one warmed.
  Either use a different seed per arm, or run the pair in both orders and
  check the conclusion survives. Quote the achieved cache fraction for each
  arm either way, since that is the number that shows whether it happened.
- `concurrency` overrides `rate_scale`, so the stepping procedure above and
  the concurrency knob are alternatives, not a pair. Step one or the other.
- Concurrency is derived from service time measured without load, and service
  time rises under load, so a run tends to hold more than the number on the
  label. Read the measured in-flight figure, not the flag you passed. The
  report cautions in both directions.
- Quote latency WITH its achieved cache fraction and arrival rate. The
  believability block exists so those travel together. Keep them together
  in any slide that quotes the number.
- Cold-start and steady-state are different claims. Report both, labeled.
- If the test hardware differs from production hardware, say so in the
  same sentence as the number. Bounding the delta is the readout's job,
  not the footnote's.
- Anything built to stated-but-unverified figures carries the assumption
  label all the way into the final deck. The label comes off when the
  dataset does, never before.
- Check the sample size before quoting a tail number. The report cautions
  under 100 requests because p99 is unstable there. If the caution is
  printed, either run longer or quote p50 and p95 only.
- Read the stability-over-time card before treating one number as steady
  state. When the card says unstable it also says which shape.

  `failing` means a window lost a large share of its requests. This is the
  one to stop on. The surviving latency numbers in that window describe what
  the endpoint could still serve, not what it was asked for, so they look
  better than reality. That is your breaking point: record the rate_scale
  that produced it and step back down, do not run it for longer.

  The other four are latency shapes, and for all of them the answer is a
  longer or repeated run rather than an average across the change. `warming`
  means the early windows are cold start and you should quote the later ones.
  `degrading` means the endpoint slowed under sustained load and the run is
  the story. `spike` means something transient hit mid-run and needs
  explaining before any number ships. `variable` means the windows are just
  noisy, which on shared capacity is common and still disqualifies the run as
  a steady-state number. A run under two minutes
  can't answer this at all.
- Don't subtract the connection-setup line from TTFT. Since 0.3.0 the
  handshake is already excluded from TTFT, TTFB and TTFG, and the line is
  printed so you can see how far the client sat from the endpoint. A
  handshake is several round trips, so read it as an upper bound on network
  distance, not as the per-request network cost a pooled production client
  pays. Run the client from where production traffic actually originates or
  the number is yours and not the customer's.
- Confirm the endpoint-under-test card matches the endpoint you meant to
  measure, including workload size and route-optimized state. Endpoint
  configs change between runs, and a report without that card pinned to it
  is not reproducible evidence.
