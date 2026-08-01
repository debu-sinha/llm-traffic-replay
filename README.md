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

## How the load model works

Three properties of your production traffic get rebuilt, each from settings
you control. The picture below is the whole mental model: you set the shape on
the left, the harness fills it with meaningless synthetic text, and the only
thing left to measure is the endpoint on the right.

![how the load is shaped, and which settings control it](docs/diagrams/load-model.svg)

| Traffic property | How it is built | Settings that control it |
| --- | --- | --- |
| Prompt sizes (heavy-tailed) | Fit a lognormal to your p50 and p95, then draw every request's input and output size from that fit. | Profile: `input_tokens` p50/p95, `output_tokens` p50/p95 |
| Cache reuse (shared prefixes) | A pool of shared-prefix documents, picked by Zipf popularity, gives requests a common leading block so the endpoint's prompt cache can engage. Your number is the target hit rate. | Profile: `cache_fraction` p50/p95. Config: `pool_zipf_s`, `pool_docs_per_bucket` |
| Arrival timing (bursty) | A two-state on/off model (MMPP) makes quiet stretches broken by spikes, or you replay your real arrival trace exactly. | Config: `qps_base`, `qps_burst`, `qps_min`, `qps_max`, `rate_scale`, or `timestamps_file` |

You can look at either shape before spending a single endpoint call.

`sample` draws from a profile and prints the sizes and cache fraction it
recovered, so you can confirm the profile matches your traffic:

```bash
python3 -m traffic_replay sample --profile configs/profile_decagon_20260723.json
```
```json
"recovered": {
  "input_tokens":  {"p50": 9967.0,  "p95": 23854.1},
  "output_tokens": {"p50": 40.0,    "p95": 90.0},
  "cache_fraction":{"p50": 0.60,    "p95": 0.87}
}
```

`schedule` builds an arrival curve and prints how bursty it is, so you know
what a config produces before you point it at anything:

```bash
python3 -m traffic_replay schedule --duration 300
```
```json
{
  "seconds": 300,
  "requests": 36928,
  "rate_p50": 38.2,
  "rate_p95": 438.9,
  "rate_max": 500.0,
  "spiky": true
}
```

`spiky` is true when peak QPS is at least 8x the trough. That is the bursty
regime this arrival model is built for. A flat load test won't surface the
queueing behavior that bursts do.

## Requirements

Python 3.10 or newer and `numpy`. That's the whole list, the HTTP client is
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
Don't `cd` into `traffic_replay/`, that is the package, and both
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
- `report.html`: the readout to open in a browser or share. Stat cards, a
  color-coded SLA scorecard, units on every metric, and the believability
  block as a callout. Self-contained, no internet needed.
- `report.md`: the same readout in plain text, for terminals and diffs.

Then follow `docs/PRODUCTION_TESTING.md` for the staged plan: smoke test on
shared capacity (client correctness only), then the provisioned throughput
endpoint at stepped rate scales, then customer-dataset replay.

## Bring your own data

There are two ways to feed the harness, pick the one that matches what you
have:

- **Token-count logs** (no prompt text): build a statistical profile and the
  harness sends synthetic text shaped to it. This section. Real prompt content
  never leaves your side.
- **The actual prompts you test with**: replay them verbatim. See
  [Bring your own prompts](#bring-your-own-prompts-real-text-no-profile) below.

Arrival timing (a trace) is optional and plugs into either path.

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

Only those three numbers per row are read. No prompt text is needed or
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

## Bring your own prompts (real text, no profile)

If you don't have token-count logs but you do have the prompts you actually
test with, skip the profile entirely and replay those prompts verbatim. The
harness sends your real text and measures the endpoint on it. Sizes and any
cache reuse are whatever your prompts already are, so there's nothing to
construct or calibrate.

Put your prompts in a file. JSONL is the most flexible, one prompt per line,
each line any of these shapes:

```json
{"messages": [{"role": "system", "content": "You are a concise support agent."}, {"role": "user", "content": "A customer's order arrived two days late. Draft a short apology."}]}
{"prompt": "Explain provisioned throughput vs pay-per-token in two sentences."}
{"text": "Classify this ticket as billing, technical, or account: 'I was charged twice.'"}
```

`messages` is a full chat turn, sent as-is. `prompt` and `text` are shorthand
for a single user message. A plain `.txt` file works too (one prompt per
line), and a `.json` file may hold an array of any of the shapes above.

Point a run config at the file with `prompts_file` instead of `profile_path`.
Arrival timing still applies: leave it synthetic, or add `timestamps_file` for
your real trace. Your prompts cycle across the scheduled arrivals. Acceptance
targets are optional and score the same SLA scorecard:

```json
{
  "prompts_file": "your_prompts.jsonl",
  "endpoint": {
    "base_url": "https://your-workspace.cloud.databricks.com",
    "path": "/serving-endpoints/your-endpoint-name/invocations",
    "auth_token_env": "DATABRICKS_TOKEN"
  },
  "duration_s": 120,
  "max_output_tokens_cap": 300,
  "acceptance_targets": {"ttft_ms": {"p50": 1500, "p95": 3000}, "success_rate": 0.99},
  "out_dir": "results/decagon_prompts",
  "title": "decagon prompts-mode run"
}
```

```bash
export DATABRICKS_TOKEN=<your token>
python3 -m traffic_replay run --config configs/run_prompts.json
```

The report's believability block tells the reader this was real text, not a
constructed shape: it prints "real prompts replayed verbatim" and marks token
targeting not applicable, while still reporting achieved cache, throughput,
arrival honesty, and finish reasons. Set either `profile_path` or
`prompts_file`, not both.

## Steering model behavior (extra_body)

`endpoint.extra_body` is a dict merged into every request body, so you can
pass any parameter the endpoint accepts: sampling knobs, structured output,
and provider-specific thinking control. The harness-owned keys (`messages`,
`max_tokens`, `temperature`, `stream`, `stream_options`, `model`) always win,
so a run stays measurable no matter what you put here. Use the `temperature`
field for temperature, not `extra_body`.

Sampling and output shape, provider-independent:

```json
"endpoint": {
  "base_url": "https://your-workspace.cloud.databricks.com",
  "path": "/serving-endpoints/your-endpoint/invocations",
  "extra_body": {"top_p": 0.95, "stop": ["\n\n"], "seed": 42}
}
```

Turning thinking on or off depends on the model. Common shapes:

```json
"extra_body": {"reasoning_effort": "low"}
```
```json
"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 2000}}
```
```json
"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}
```

The first is OpenAI o-series style, the second is Anthropic extended thinking,
the third is how Qwen and GLM thinking models toggle on a vLLM-backed
Databricks endpoint. Check your endpoint's docs for the exact key, since a
param the endpoint doesn't recognize is usually ignored, and some reject it
with a 400 that the report shows as a failed request.

Every run echoes its parameters in the report so a reader knows what produced
the numbers:

```
request params: temperature 0.0, max_tokens cap 220, extra_body {"top_p": 0.95}
```

When the endpoint reports a thinking-token count
(`completion_tokens_details.reasoning_tokens`), the believability block adds a
line for it. Endpoints that stream a reasoning channel but don't report the
count leave the line out rather than guessing. To measure what thinking costs
on an endpoint, run it once with thinking on and once off, then `compare` the
two: the reasoning-token, TTFT, and end-to-end differences land side by side.

## Cost in Databricks DBUs

The report can turn the tokens it measured into cost, using rates you supply
from the [Databricks pricing page](https://www.databricks.com/product/pricing/foundation-model-serving).
The tool never fetches or hardcodes prices, so the report states the
arithmetic and the numbers you gave it. Databricks denominates in DBUs, and
the dollar conversion (`usd_per_dbu`) comes from your own plan or commit, so
it is optional.

Pay-per-token endpoints bill input, output, and cache-read tokens separately
(three DBU/M rates on the pricing page). The tool already measures those three
token counts, so the cost is exact:

```json
"pricing": {"mode": "per_token",
  "input_dbu_per_m": 20.0, "output_dbu_per_m": 62.857,
  "cache_read_dbu_per_m": 2.0, "usd_per_dbu": 0.070}
```

The report then shows DBU per request (p50/p95), DBU per 1,000 requests, DBU
per minute, and the DBUs the prompt cache saved. Leave `usd_per_dbu` out to
stay in DBUs.

Provisioned throughput bills capacity by the hour, not per token, so the
useful figure is effective cost per 1M tokens at the load you measured:

```json
"pricing": {"mode": "provisioned", "dbu_per_hour": 85.714, "usd_per_dbu": 0.070}
```

```
effective DBU per 1M tokens = dbu_per_hour / (tokens served per hour at the measured throughput)
```

That figure improves as you fill the endpoint, and it is what a
cost-per-throughput comparison against another provider actually needs.

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
- **Reasoning tokens**: when the endpoint reports a thinking-token count, the
  report prints total and per-request reasoning tokens with the usage field it
  came from. When it streams a reasoning channel but reports no count (GLM on
  dogfood does this), the report falls back to counting the reasoning deltas
  and labels the number a stream-counted estimate.
- **Cost**: when you supply DBU rates, the report shows cost per request, per
  1,000 requests, and per minute, plus the DBUs the cache saved, all traceable
  to the tokens measured and the rates you gave. See [Cost](#cost-in-databricks-dbus).
- **Request params**: the report echoes the temperature, max-tokens cap, and
  any `extra_body`, so a reader knows exactly what request parameters produced
  the numbers.

## Settings reference

Run configs are plain JSON deserialized into `RunConfig`
(`traffic_replay/runner.py`). Every field, with defaults:

| field | default | example | what it does, and when to change it |
|---|---|---|---|
| `profile_path` | null | `"configs/profile_decagon_real.json"` | path to a profile JSON (shape mode). Set this or `prompts_file`, not both |
| `prompts_file` | null | `"your_prompts.jsonl"` | path to a real-prompts file, `.jsonl` / `.txt` / `.json` (prompts mode). Set this or `profile_path` |
| `endpoint.base_url` | required | `"https://your-ws.cloud.databricks.com"` | workspace host, no trailing slash |
| `endpoint.path` | required | `"/serving-endpoints/my-ep/invocations"` | the serving endpoint route. Model-serving uses `.../invocations` |
| `endpoint.auth_token_env` | `DATABRICKS_TOKEN` | `"DATABRICKS_TOKEN"` | env var the bearer token is read from. Never put the token in the config |
| `endpoint.model` | null | `"databricks-meta-llama-3-3-70b-instruct"` | set only for shared `/chat/completions` routes that need a model field; leave null for a dedicated `.../invocations` endpoint |
| `endpoint.connect_timeout_s` | 10 | `10` | TCP/TLS connect timeout. Raise on a slow/cold endpoint |
| `endpoint.read_timeout_s` | 120 | `120` | per-read socket timeout while streaming. Raise for long generations |
| `endpoint.temperature` | 0.0 | `0.0` | request temperature. Keep 0 for repeatable benchmarks |
| `endpoint.max_retries` | 1 | `1` | connection-error retries only. The retried count prints in the report |
| `endpoint.extra_body` | null | `{"top_p": 0.95}` | passthrough request params (top_p, stop, response_format, thinking control). Harness-owned keys always win. See [Steering model behavior](#steering-model-behavior-extra_body) |
| `duration_s` | 300 | `300` | how long the schedule runs, in seconds |
| `qps_base` / `qps_burst` | 25 / 350 | `2` / `8` | mean request rates of the two arrival states (quiet and burst). Set both low for a gentle probe, high for a stress run |
| `qps_min` / `qps_max` | 10 / 500 | `1` / `12` | hard floor and ceiling clamped on the rate curve |
| `rate_scale` | 1.0 | `0.5` | uniform thinning that keeps the shape. Step it 0.1 to 1.0 to find where the endpoint bends |
| `timestamps_file` | null | `"your_arrivals.txt"` | real arrival trace (one epoch/line, or JSONL `{"t": s}`) that replaces the synthetic schedule |
| `max_concurrency` | 256 | `64` | in-flight request bound. Excess arrivals surface as dispatch lag, not fake latency |
| `seed` | 7 | `7` | root RNG seed. Same config plus same seed is the same experiment |
| `cpt` | 4.0 | `4.0` | starting characters-per-token guess, recalibrated during warmup (profile mode only) |
| `calibrate_n` | 12 | `8` | warmup requests run before the schedule. Also primes the endpoint cache |
| `shard_index` / `shard_total` | 0 / 1 | `0` / `3` | deterministic 1-of-n schedule split, one per client machine, then `merge` the outputs |
| `pool_docs_per_bucket` | 40 | `40` | shared-prefix documents per size bucket (profile-mode cache structure) |
| `pool_zipf_s` | 1.1 | `1.1` | document popularity skew. Higher concentrates traffic on hot documents |
| `out_dir` | `results` | `"results/decagon"` | output root. Each run writes a timestamped subdirectory under it |
| `title` / `label` | "" | `"decagon PT 250K"` | report title and the provenance label printed at the bottom of the report |
| `max_output_tokens_cap` | 512 | `300` | ceiling on `max_tokens` per request. Reasoning models spend this budget thinking first |
| `acceptance_targets` | null | `{"ttft_ms": {"p95": 900}}` | inline SLA targets (`ttft_ms`, `ttfg_ms`, `hard_timeouts`, `success_rate`, `interchunk_ms`). In prompts mode this is how the scorecard gets its targets |
| `ttft_definition` | `first_content` | `"first_visible"` | `first_content` (first token of either kind) or `first_visible` (skip reasoning deltas). The SLA scorecard scores whichever is set |
| `pricing` | null | `{"mode": "per_token", "input_dbu_per_m": 20, "output_dbu_per_m": 62.857}` | DBU cost rates you supply from the pricing page. The report turns measured tokens into cost. See [Cost](#cost-in-databricks-dbus) |

Profile JSON fields: `name`, then `input_tokens`, `output_tokens`, and
`cache_fraction` (each `{"p50": .., "p95": ..}`, cache in (0,1)).
`provenance` records where the numbers came from, and `label` is printed on
every report built from this profile. An optional `acceptance_targets`
object (ttft_ms, ttfg_ms, hard_timeouts, success_rate, interchunk_ms) drives
the SLA scorecard. Without it, no scorecard is printed. Auth is never stored in any config.
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
