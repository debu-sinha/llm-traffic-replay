# llm-traffic-replay

[![tests](https://github.com/debu-sinha/llm-traffic-replay/actions/workflows/tests.yml/badge.svg)](https://github.com/debu-sinha/llm-traffic-replay/actions/workflows/tests.yml)

Replay **your production traffic shape** against an LLM serving endpoint,
instead of testing with generic synthetic load.

Flat-rate load tests with uniform prompts produce latency numbers that
don't transfer to production. Real agent traffic has three properties that
drive serving behavior, and this harness reproduces all three:

1. **Heavy-tailed prompt sizes**, sampled from distributions fitted to your
   stated quantiles (e.g. P50 10K input tokens, P95 24K, outputs 40 to 90).
2. **High, variable prompt-cache hit ratio** (e.g. P50 60%, P95 87%). You
   can't ask an endpoint for a hit rate, so the harness **constructs** it by
   making requests share long leading context the way production traffic
   actually repeats.
3. **Bursty arrivals**: spikes between 10 and 500 QPS, not a steady drip.
   You get synthetic bursts by default, and your own production arrival
   trace drops in via `timestamps_file` and replaces the synthetic schedule
   entirely.

It works against any OpenAI-compatible streaming chat endpoint. That
includes Databricks provisioned throughput and pay-per-token serving as
well as other hosted providers, so the same instrument that measures your
candidate endpoint also measures the alternatives on an identical basis.

## What it measures, and what it doesn't

The harness reproduces the *shape* of your traffic: prompt sizes, cache
structure, and arrival timing. It fills that shape with synthetic text. An
endpoint's latency and throughput depend on token counts, cache hits, and
arrival rate, not on what the words say, so the numbers transfer to
production even though the prompts themselves are gibberish.

It doesn't measure anything content-dependent. Response quality, guardrail
and safety triggers, and semantic routing all need real prompts. This is a
load-shape benchmark, not a quality eval.

## Requirements

Python 3.10 or newer and `numpy`. That's the whole list; the HTTP client is
standard library. `pytest` is optional (a bundled zero-dependency runner runs
the same tests without it).

```bash
python3 -m pip install numpy      # add pytest for nicer test output, optional
```

Commands below use `python3`, which is what a stock macOS or Linux box has.
If your `python` already points at 3.10+ (a venv or conda env), `python`
works too.

## Quickstart (no endpoint needed, about 60 seconds)

Run every command from the repository root (the directory holding this file).
Don't `cd` into `traffic_replay/`; that is the package, and both
`python3 -m traffic_replay` and the relative `configs/` paths resolve from
the root.

```bash
git clone https://github.com/debu-sinha/llm-traffic-replay.git
cd llm-traffic-replay

# 1. Run the test suite (either runner; they run the same tests)
python3 -m pytest                 # no pytest? python3 scripts/run_tests_stdlib.py

# 2. Self-test the instrument against the bundled mock (known latency model)
python3 -m traffic_replay validate
```

`validate` runs the entire pipeline (sampler, prefix pool, burst schedule,
streaming client, measurement) against a local mock server that KNOWS its
own true latency per request, then reports client-measured minus
server-true error. Current calibration on a laptop-class machine: TTFT
error p50 ~2 ms, p95 < 5 ms. If it doesn't PASS on your machine, don't
trust any number the harness produces there.

## Run against a real endpoint

Open `configs/run_smoke.json` and fill in the two `YOUR-...` placeholders,
your workspace host and the endpoint path:

```json
"endpoint": {
  "base_url": "https://your-workspace.cloud.databricks.com",
  "path": "/serving-endpoints/your-endpoint-name/invocations",
  "auth_token_env": "DATABRICKS_TOKEN"
}
```

Export the token that `auth_token_env` names, then run:

```bash
export DATABRICKS_TOKEN=<your PAT, or any bearer token the endpoint accepts>
python3 -m traffic_replay run --config configs/run_smoke.json
```

Outputs land in `results/<timestamp>/`:

- `requests.jsonl`: every request: TTFT/TTFB/E2E ms, endpoint-reported
  prompt/completion/cached tokens, intended sizes, document id, dispatch
  lag, errors.
- `summary.json`: percentile tables plus the believability block.
- `report.md`: the human-readable readout.

Then follow `docs/PRODUCTION_TESTING.md` for the staged plan: smoke test on
shared capacity (client correctness only), then the provisioned throughput
endpoint at stepped rate scales, then customer-dataset replay.

## Bring your own data

Two inputs plug in: your traffic SHAPE (a token-count export) and, optionally,
your arrival TIMING (an arrival-time export). The prompt text stays synthetic
on purpose (see "What it measures"), so no real prompt content ever moves.

### 1. Traffic shape, from your logs

Export one row per request with three token counts. JSONL, one object per
line:

```json
{"input_tokens": 10231, "output_tokens": 42, "cached_tokens": 6100}
{"input_tokens": 8977,  "output_tokens": 55, "cached_tokens": 8004}
{"input_tokens": 24310, "output_tokens": 88, "cached_tokens": 15220}
```

or CSV with a header row:

```
input_tokens,output_tokens,cached_tokens
10231,42,6100
8977,55,8004
24310,88,15220
```

Only those three numbers per row are read; no prompt text is needed or
touched. Build a profile from the export and check it:

```bash
python3 scripts/profile_from_logs.py \
  --input your_logs.jsonl --name decagon_real \
  --out configs/profile_decagon_real.json

# confirm the recovered quantiles match your data
python3 -m traffic_replay sample --profile configs/profile_decagon_real.json
```

That writes a profile like this (the P50/P95 the harness will reproduce):

```json
{
  "name": "decagon_real",
  "input_tokens":   {"p50": 10126, "p95": 23193},
  "output_tokens":  {"p50": 45,    "p95": 86},
  "cache_fraction": {"p50": 0.618, "p95": 0.832},
  "provenance": "Computed from 400 request records.",
  "label": "Built from a real dataset. ..."
}
```

If your columns have other names, pass `--input-field` / `--output-field` /
`--cached-field`, or `--cache-fraction-field` if you already have a
per-request cache fraction instead of a cached-token count.

### 2. Arrival timing, from your trace (optional)

Export your request arrival times, one epoch-second value per line:

```
0.0
0.4
0.9
1.2
2.1
```

or JSONL with a `t` field per line:

```json
{"t": 0.0}
{"t": 0.4}
{"t": 0.9}
```

Times are shifted to start at zero and sorted for you, and the line count is
the request count. Leave this out and the harness uses its synthetic bursty
schedule instead.

### 3. Point a run config at both

Copy `configs/run_smoke.json` to `configs/run_byod.json`, set `profile_path`
to your new profile, add `timestamps_file` if you have a trace, and fill in
the endpoint. A minimal config:

```json
{
  "profile_path": "configs/profile_decagon_real.json",
  "timestamps_file": "your_arrivals.txt",
  "endpoint": {
    "base_url": "https://your-workspace.cloud.databricks.com",
    "path": "/serving-endpoints/your-endpoint-name/invocations",
    "auth_token_env": "DATABRICKS_TOKEN"
  },
  "duration_s": 300,
  "out_dir": "results/decagon",
  "title": "decagon real-data run"
}
```

```bash
export DATABRICKS_TOKEN=<your token>
python3 -m traffic_replay run --config configs/run_byod.json
```

Every field you leave out falls back to the default in the settings reference
below. The report names the trace it ran from and carries the profile's
label, so a reader can see exactly what shaped the run.

## Where to run it

The harness measures client side, so the machine it runs on is part of the
experiment. Match the machine to the stage:

**Tests and `validate` (stage 0):** anywhere with Python 3.10+ and numpy.
A laptop is fine. The mock is local, so no network's involved.

**Smoke test (stage 1):** a Databricks notebook is the easiest path and is
what `notebooks/smoke_test_e2e_demo.ipynb` does: serverless or classic
compute, ambient workspace auth, no token handling. A laptop or VM with a
PAT works equally well at smoke rates (a few QPS).

**Measured PT runs (stage 2):** use a dedicated machine in the same cloud
region as the endpoint's workspace, and nothing else running on it. Two
good options: a single-node cloud VM (8 or more vCPUs, network-optimized),
or a single-node Databricks cluster in that workspace with the run started
from its driver (web terminal or a notebook `%sh` cell). Avoid laptops for
measured runs: Wi-Fi jitter, VPNs, sleep states and cross-region paths all
land in your TTFT tail and are indistinguishable from endpoint behavior
after the fact. The believability block will expose a struggling client as
dispatch lag, but the better plan is not to generate that noise at all.
If dispatch lag p95 grows past ~100 ms at full rate_scale, split the
schedule across two machines with `shard_index`/`shard_total` and pool the
`requests.jsonl` files.

## Pooling and comparing runs

Sharded across machines? Pool their output dirs into one summary:

```bash
python3 -m traffic_replay merge results/pooled \
  results/m1/2026* results/m2/2026* results/m3/2026*
```

Comparing providers (the cost-parity motion) means running each on this
same instrument, then:

```bash
python3 -m traffic_replay compare results/compare \
  results/dbx-pt results/together results/baseten
```

`compare` writes `comparison.md`, one column per run, and warns in bold when
the runs' achieved cache p50 differ by more than 0.10, because comparing
latency at different cache rates is not a comparison. Pass `--profile` to
`merge` to score the pooled run against acceptance targets, and `--force` to
merge runs whose endpoint paths differ.

## Reading results: the honesty rules

Every latency table ships with the context that decides whether it can be
believed, and the report prints them together:

- **Achieved cache fraction** (endpoint-reported, with the exact usage
  field named). A good p50 at an unrealistic hit rate is a fake result.
- **Constructed vs achieved**: what the traffic intended vs what the
  endpoint served. Cold first-uses are included and visible per document.
- **Token targeting error**: text is generated through a calibrated
  characters-per-token ratio. Endpoint-reported token counts are the source
  of truth, and the residual error is printed.
- **Dispatch lag**: how late the client fired versus the schedule. If the
  client saturates, that's reported as client lag, not silently blended
  into endpoint latency.
- **Profile label**: runs built to stated (spoken) figures carry that
  label until the exact production dataset replaces the profile config.
- **Interchunk max**: the widest gap between streamed content chunks per
  request, so an SLA sensitive to mid-stream stalls has a number to read.
- **Output token targeting**: reported completion tokens vs intended, with
  the finish_reason mix (stop vs length), so a compressed output
  distribution is visible next to the calibrated input side.
- **SLA scorecard**: when the profile carries `acceptance_targets`, the
  report prints a pass/fail scorecard (TTFT, TTFG, hard-timeout and
  interchunk breaches, success rate). A missed target prints NO, not a dash.
- **Reasoning split**: for thinking models the report gives `ttft` (first
  token of either kind), `ttfr` (first reasoning) and `ttfv` (first visible)
  and flags when they diverge. `ttft_definition` picks which the scorecard
  scores.

## Settings reference

Run configs are plain JSON deserialized into `RunConfig`
(`traffic_replay/runner.py`). Every field, with defaults:

| field | default | meaning |
|---|---|---|
| `profile_path` | required | path to a profile JSON (see below) |
| `endpoint.base_url` | required | `https://<workspace-host>`, no trailing slash |
| `endpoint.path` | required | e.g. `/serving-endpoints/<name>/invocations` |
| `endpoint.auth_token_env` | `DATABRICKS_TOKEN` | env var read for the bearer token |
| `endpoint.model` | null | set only for shared `/chat/completions` routes |
| `endpoint.connect_timeout_s` | 10 | TCP/TLS connect timeout |
| `endpoint.read_timeout_s` | 120 | per-read socket timeout during streaming |
| `endpoint.temperature` | 0.0 | request temperature (keep 0 for benchmarks) |
| `endpoint.max_retries` | 1 | connection-error retries. The retried count prints in the report |
| `duration_s` | 300 | schedule length in seconds |
| `qps_base` / `qps_burst` | 25 / 350 | mean rates of the two arrival states |
| `qps_min` / `qps_max` | 10 / 500 | hard clamp on the rate curve |
| `rate_scale` | 1.0 | uniform thinning that preserves shape. Step 0.1 to 1.0 per the run plan |
| `timestamps_file` | null | real arrival trace (text or JSONL `{"t": s}`) that replaces the synthetic schedule |
| `max_concurrency` | 256 | thread pool bound. Excess arrivals show up as dispatch lag |
| `seed` | 7 | root seed. Same config plus same seed is the same experiment |
| `cpt` | 4.0 | starting characters-per-token guess, recalibrated during warmup |
| `calibrate_n` | 12 | sequential warmup requests used to calibrate cpt |
| `shard_index` / `shard_total` | 0 / 1 | deterministic 1-of-n schedule split across client machines |
| `pool_docs_per_bucket` | 40 | shared-prefix documents per size bucket |
| `pool_zipf_s` | 1.1 | popularity skew. Higher concentrates traffic on hot documents |
| `out_dir` | `results` | output root. Each run writes a timestamped subdirectory |
| `title` / `label` | "" | report title and the provenance label printed at the bottom |
| `max_output_tokens_cap` | 512 | safety cap on max_tokens per request (32 in the smoke config) |
| `ttft_definition` | `first_content` | `first_content` (first token of either kind) or `first_visible` (skip reasoning-channel deltas). The SLA scorecard scores whichever is set |

Profile JSON fields: `name`, then `input_tokens`, `output_tokens`, and
`cache_fraction` (each `{"p50": .., "p95": ..}`, cache in (0,1)).
`provenance` records where the numbers came from, and `label` is printed on
every report built from this profile. An optional `acceptance_targets`
object (ttft_ms, ttfg_ms, hard_timeouts, success_rate, interchunk_ms) drives
the SLA scorecard; without it, no scorecard is printed. Auth is never stored in any config.
The token comes from the environment variable at run time or, in a
notebook, from the ambient workspace context.

## Architecture

![architecture](docs/diagrams/architecture.svg)

The per-request sequence and the validation design are in
`docs/ARCHITECTURE.md`.

## Repository layout

```
traffic_replay/          the package (profile, prefix_pool, schedule,
                         textgen, sse, client, metrics, runner,
                         mock_server, aggregate, cli)
configs/                 profiles and run configs (JSON)
tests/                   pytest suite (unit + end-to-end)
scripts/run_tests_stdlib.py   zero-dependency test runner
scripts/profile_from_logs.py  build a profile JSON from real request logs
notebooks/               self-contained Databricks workspace smoke notebook
                         (embeds this repo, ambient auth, smoke labels)
docs/ARCHITECTURE.md     diagrams and design decisions
docs/PRODUCTION_TESTING.md   step-by-step run plan
```

## Provenance and labels

The bundled `configs/profile_decagon_20260723.json` is built to
customer-stated figures from the 2026-07-23 call and says so in its label.
When the exact production dataset lands, it replaces that config file and
the label comes off. Nothing else in the harness changes.
