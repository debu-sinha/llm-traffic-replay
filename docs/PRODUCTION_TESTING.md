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
   - dispatch lag p95 in single-digit ms (the client is loafing at 6 QPS).

**The latency numbers from stage 1 are not evidence of anything.** Shared
pay-per-token capacity says nothing about a dedicated endpoint. The run
config's label says this and the label prints in the report. Leave it in.

## Stage 2: provisioned throughput endpoint, stepped load

Purpose: the real measurement. Requires the dedicated endpoint and an
agreed profile config (until the customer's exact dataset lands, the
bundled profile carries its ASSUMPTION label and so does every report).

1. Copy `configs/run_pt_full.json`, set base_url and the PT endpoint path.
2. Agree the acceptance targets in writing before the first run (for this
   engagement: TTFT p50 500 ms / p95 900 ms, full generation p50 700 ms /
   p95 1500 ms, held across the burst schedule, per workload class). Ask for
   the interchunk-stall threshold too. The moment they give a number, put it
   in the profile as `acceptance_targets.interchunk_ms` and the scorecard
   counts breaches against the success rate.
3. Step the load, one knob, in this order, reading the believability block
   between steps:
   - `rate_scale`: 0.1 -> 0.25 -> 0.5 -> 1.0
   - at each step: error rate first, then dispatch lag (client health),
     then achieved cache fraction, THEN latency percentiles.
4. Two workload classes = two profile configs = two runs. Do not blend
   them into one table.
5. If the client's dispatch lag p95 grows past ~100 ms at high rate_scale,
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
