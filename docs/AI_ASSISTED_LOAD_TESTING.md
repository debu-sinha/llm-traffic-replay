# AI-assisted load testing with Claude Code or Codex

Claude Code and Codex can prepare commands, inspect artifacts, and explain a
report. They must not invent credentials, prices, quotas, SLAs, or workload
shape. Give the agent the repository path and use one of the prompts below.

Replace every `YOUR_...` value. Ask the agent to stop before paid traffic if a
required value cannot be discovered safely.

## One fixed-rate profile run

> In this repository, validate the local instrument, inspect
> `YOUR_PROFILE.json` and `YOUR_RATE_LIMITS.json`, then run a fixed-rate load
> test against `YOUR_HOST` / `YOUR_ENDPOINT` using Databricks profile
> `YOUR_AUTH_PROFILE`, `YOUR_RPS` for `YOUR_DURATION_SECONDS`, response-start
> p95 target `YOUR_TTFT_MS`, response-finish p95 target `YOUR_TTFG_MS`, and
> success target `YOUR_SUCCESS_FRACTION`. Use `first_visible`. Do not print
> credentials. Confirm the quota snapshot is current and applicable before
> paid traffic. Verify the completed artifact, visually inspect desktop and
> mobile HTML, and explain latency, throughput, cache, cost, errors, SLA, and
> limitations in plain language. Do not call a smoke run a capacity result.

## Real prompts

> Run the same guarded fixed-rate workflow using `--prompts YOUR_PROMPTS.jsonl`
> instead of a profile. Confirm that I authorize sending the prompt contents to
> the endpoint, report prompt count and usage coverage, and do not expose prompt
> text in your response.

## Synthetic token/cache shape

> Run a diagnostic fixed-rate test with input tokens `YOUR_INPUT_P50,P95`,
> output tokens `YOUR_OUTPUT_P50,P95`, and cached prompt-token fraction
> `YOUR_CACHE_P50,P95`. Label it synthetic and do not describe intended cache
> fraction as request hit probability. Compare intended and endpoint-reported
> cached-token share.

## Quickstart and repeatable config

> Use `traffic_replay quickstart --fixed-rate` to create a reviewable config for
> my endpoint, profile, quota snapshot, request controls, and SLA targets. Show
> me the config with credentials redacted, run it with `traffic_replay run`,
> verify the artifact, and explain where the automatic report was written.

## Exact timestamp trace

> Validate `YOUR_TRACE.txt` as finite seconds-from-start offsets, create a run
> config using `timestamps_file`, preserve my profile and SLA targets, run the
> real endpoint with quota protection, and verify that scheduled request count
> and trace identity match the report.

## Sizing-derived rate

> Run the sizing path with `--sizing-concurrency YOUR_N`. Explain before traffic
> that this sends paid sizing requests, derives one fixed open-loop rate, does
> not hold concurrency, and cannot prove a quota-safe schedule in advance.
> Keep the test bounded and label the conclusion diagnostic unless the evidence
> supports more.

## Rate sweep

> Run a sequential sweep over `YOUR_RATES` with my customer-owned TTFT, TTFG,
> and success targets. Check the full ladder against the quota snapshot before
> traffic. Stop on invalid or quota-limited evidence. Use `--diagnostic-only`
> if the targets are examples. Run `verify-sweep` and explain the highest tested
> passing rate only if the verified conclusion permits it.

## Synchronized shards and merge

> Prepare `YOUR_SHARD_COUNT` configs with the same future start time, run ID,
> exact trace, workload, and endpoint; assign unique zero-based shard indexes.
> Start them concurrently, verify every source, and merge without `--force`.
> If compatibility or source-reproducibility checks fail, stop and explain; do
> not turn an invalid diagnostic merge into a capacity claim.

## Compare runs

> Compare `RUN_A`, `RUN_B`, and any other supplied verified run directories.
> First check endpoint, model, workload, request controls, load, transport,
> duration, and SLA compatibility. Generate the automatic comparison report,
> visually inspect it, and explain only defensible differences.

## Profile from logs

> Inspect only the schema—not sensitive content—of `YOUR_USAGE_LOG`. Run
> `scripts/profile_from_logs.py` with the correct input, output, cached-token or
> cache-fraction fields. Prefer empirical-joint mode when preserving observed
> correlations matters. Validate and sample the resulting content-free profile,
> then show the recovered p50/p95 workload shape.

## Report-only review

> Do not send traffic. Verify `RUN_DIR`, open its HTML report, inspect it at
> desktop and mobile widths, and answer as a customer rather than an infra
> specialist: what was tested, endpoint, RPS, tokens/second, response-start and
> response-finish latency, cache evidence, cost, success/errors, SLA result,
> capacity conclusion, and what remains unknown. Flag any overlap, ambiguous
> legend, clipped label, contradictory number, or unsupported claim.
