# Customer load-testing guide

This tool answers a practical question: **at the load and workload you asked
for, how quickly and reliably did the endpoint answer, and what did that
measured replay cost?** It produces a standardized HTML report automatically.

Use real prompts or a profile made from your own logs before making a buying or
production decision. The small examples below are workflow checks, not capacity
claims.

## What you need

1. Python 3.10+, this repository, and `pip install -e '.[dev]'` in a virtual
   environment.
2. A workspace host, serving endpoint name, and either a Databricks CLI profile
   or the name of an environment variable containing a token.
3. Your expected workload: real prompts, a measured profile, or explicit token
   sizes.
4. Your acceptance targets: response-start latency, response-finish latency,
   and required success rate.
5. A recently checked quota snapshot. Quotas are workspace-wide; the harness
   cannot see unrelated traffic.

First prove the local measuring instrument:

```bash
python3 -m traffic_replay validate --port 0
```

## Recommended path: fixed rate with quota protection

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE \
  --endpoint YOUR-ENDPOINT \
  --auth-profile YOUR-PROFILE \
  --profile configs/profile_measured.json \
  --fixed-rate YOUR_REQUESTS_PER_SECOND \
  --duration 300 \
  --ttft-definition first_visible \
  --ttft-p95 YOUR_RESPONSE_START_P95_MS \
  --ttfg-p95 YOUR_RESPONSE_FINISH_P95_MS \
  --success-rate YOUR_REQUIRED_FRACTION \
  --rate-limits configs/rate_limits_checked_today.json \
  --out-dir results/customer-run
```

For managed Databricks GLM 5.2, add the documented request control that matches
your production application, for example `--extra-body
'{"reasoning_effort":"none"}'` when production disables reasoning. Do not add
it merely to improve a benchmark.

## Every supported workflow

| Need | Command/path | What it proves |
|---|---|---|
| Check the local instrument | `validate` | Sampling, scheduling, streaming, timing, journaling, and reporting work against a known local oracle. |
| Inspect a profile | `sample --profile FILE --n 100` | The profile produces the intended token and cache shape. No endpoint traffic. |
| Inspect the default synthetic schedule | `schedule --duration 300 --rate-scale 1` | The arrival generator can materialize a schedule. No endpoint traffic. |
| Replay measured workload shape | `benchmark --profile FILE --fixed-rate RPS ...` | Real endpoint behavior for that profile and rate. |
| Replay real prompts | Replace `--profile FILE` with `--prompts PROMPTS.jsonl` | Real prompt semantics and sizes; prompt content is sent to the endpoint. |
| Generate synthetic prompts | Use `--input-tokens P50,P95 --output-tokens P50,P95 --cache-fraction P50,P95` | Instrument or controlled-shape testing; weaker workload fidelity than real prompts. |
| Create a runnable config | `quickstart ... --fixed-rate RPS --out configs/run.json`, then `run --config configs/run.json` | The same fixed-rate run through a reviewable, repeatable config. |
| Replay an exact arrival trace | Add `"timestamps_file": "trace.txt"` to a run config | Behavior for exact offsets, one finite seconds-from-start value per line. |
| Derive a rate from unloaded concurrency | `benchmark --sizing-concurrency N ...` | A short sizing pass derives one open-loop rate. It does **not** hold concurrency and cannot be quota-planned before paid sizing traffic. |
| Explore several rates | `sweep --rate 1,2,4 ...` | A sequential rate ladder. Supply customer-owned latency and reliability targets for a capacity conclusion; otherwise use `--diagnostic-only`. |
| Run synchronized shards | Set the same `run_id`, future `start_at_unix`, trace, and `shard_total`; give each process a unique zero-based `shard_index` | Distributed traffic generation. Run all shards, then merge. |
| Merge shards | `merge OUT SHARD_RUN_1 SHARD_RUN_2 ...` | One aggregate only when source identity and compatibility checks pass. `--force` intentionally produces an invalid diagnostic aggregate. |
| Compare runs | `compare OUT RUN_A RUN_B ...` | Side-by-side standardized evidence; it does not make unlike workloads comparable. |
| Verify a run | `verify-run RUN_DIR --out SIBLING_RECEIPT_DIR` | Internal hashes and completion chain match. This is not a signature or trusted timestamp. |
| Verify a sweep | `verify-sweep SWEEP_DIR` | Sweep report, rungs, traffic counts, and conclusion agree. |
| Build a profile from logs | `python3 scripts/profile_from_logs.py --input usage.jsonl --out configs/profile_measured.json` | A content-free token/cache distribution from your records. |

`quickstart` accepts `--fixed-rate`, `--rate-limits`, `--extra-body`, SLA flags,
and `--ttft-definition`. Use `--sizing-concurrency` only when a pre-run rate is
unknown. A quota snapshot and sizing cannot be combined because the exact paid
schedule is unknowable until sizing finishes; the command explains this before
endpoint traffic.

## The report is automatic

Each successful run writes `report.html`, `report.md`, `summary.json`,
`requests.jsonl`, `start.json`, and `manifest.json`. Open `report.html` in a
browser. No separate report command is needed. A sweep writes `sweep.html` and
`sweep.md`; compare writes `comparison.html` and `comparison.md`.

At the top of a run report you should see:

- the exact endpoint and workload mode;
- scheduled and achieved requests per second;
- request count, duration, and observed in-flight load;
- input and output tokens per second;
- separate same-scale charts for response started and response finished;
- exact p50/p90/p95/p99 latency values;
- success, errors, quota state, and customer acceptance result;
- prompt-cache evidence; and
- measured cost, or an explicit statement that pricing was not supplied.

**Response finished must be at least as late as response started for the same
request.** The report keeps these charts separate for readability but uses the
same vertical scale. Exact values are printed below them.

## Cache and cost, in plain language

“Cached prompt-token share” is cached prompt tokens divided by all prompt
tokens. It is **not** request hit rate. “Prompt cache used: No” means the
endpoint reported zero cached prompt tokens for the measured eligible rows.
“Not reported” means the endpoint did not provide enough evidence; it does not
mean zero.

The tool never guesses prices. Add a pricing block to the run config only after
checking the exact model, cloud, region, product, tier, contract, and effective
date. The report then separates uncached input, cached input, and output cost.
Without applicable pricing it says “Cost not calculated,” never “free.” Provider
billing remains authoritative.

## Before making a decision

- Replace example profiles and SLAs with customer-owned inputs.
- Run for at least several one-minute windows; tiny smoke runs cannot establish
  tails, stability, reliability, quota headroom, or endpoint capacity.
- Match production reasoning controls and connection behavior.
- Check live quotas immediately before paid traffic and account for other
  workspace users.
- Verify the artifact before quoting it.
- Treat “inconclusive” as a result, not a pass.

For delegated execution, use the guarded prompts in
[AI-assisted load testing](AI_ASSISTED_LOAD_TESTING.md).
