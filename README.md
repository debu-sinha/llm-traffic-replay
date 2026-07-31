# llm-traffic-replay

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

Python 3.10+, `numpy`. That's the whole list. The HTTP client is standard
library. Tests run with `pytest` if you have it, or with the bundled
zero-dependency runner if you don't.

## Quickstart (no endpoint needed, ~60 seconds)

Every command in this README runs from the repository root (the directory
containing this file). Don't cd into `traffic_replay/`. That's the package,
and `python -m traffic_replay` plus the relative `configs/` paths both
resolve from the root.

```bash
cd llm-traffic-replay

# 1. Full test suite
python -m pytest                          # or: python3 scripts/run_tests_stdlib.py

# 2. Instrument self-test against the bundled mock (known latency model)
python -m traffic_replay validate
```

`validate` runs the entire pipeline (sampler, prefix pool, burst schedule,
streaming client, measurement) against a local mock server that KNOWS its
own true latency per request, then reports client-measured minus
server-true error. Current calibration on a laptop-class machine: TTFT
error p50 ≈ 2 ms, p95 < 5 ms. If it doesn't PASS on your machine, don't
trust any number the harness produces there.

## Run against a real endpoint

```bash
export DATABRICKS_TOKEN=...   # or any bearer token your endpoint accepts
# edit configs/run_smoke.json: base_url, path
python -m traffic_replay run --config configs/run_smoke.json
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

The bundled profile is built to spoken figures. When you have real numbers,
two inputs plug in, and the prompt text stays synthetic on purpose.

**Traffic shape, from your logs.** Export per-request token counts (input
tokens, output tokens, and cached prompt tokens) as JSONL or CSV, then build
a profile:

```bash
python3 scripts/profile_from_logs.py \
  --input your_logs.jsonl --name decagon_real \
  --out configs/profile_decagon_real.json

# check the recovered quantiles match your data
python -m traffic_replay sample --profile configs/profile_decagon_real.json
```

The defaults read `input_tokens`, `output_tokens`, and `cached_tokens`.
Override with `--input-field`, `--output-field`, `--cached-field`, or pass
`--cache-fraction-field` if you already have a per-request fraction. Only the
distribution is read, no prompt text, so a token-count export is enough and
no customer content moves. Point your run config's `profile_path` at the new
file.

**Arrival timing, from your trace.** Write your request arrival times to a
file, one epoch-second value per line (or JSONL `{"t": <seconds>}`), and set
`"timestamps_file"` in the run config. It replaces the synthetic burst
schedule: the line count is the request count, the timing is yours, and the
report names the trace it ran from.

**Prompt text stays synthetic** (see "What it measures"). Exact-prompt replay
isn't supported today.

A full real-data run uses both: your quantiles in the profile, your
timestamps in `timestamps_file`.

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

Profile JSON fields: `name`, then `input_tokens`, `output_tokens`, and
`cache_fraction` (each `{"p50": .., "p95": ..}`, cache in (0,1)).
`provenance` records where the numbers came from, and `label` is printed on
every report built from this profile. Auth is never stored in any config.
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
                         mock_server, cli)
configs/                 profiles and run configs (JSON)
tests/                   pytest suite (33 tests)
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
