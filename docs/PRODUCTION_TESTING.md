# Production testing runbook

This runbook is for an authorized load window against a production-like LLM
endpoint. It separates instrument validation, protocol validation, workload
validation, stepped load, and evidence review. Do not begin a capacity test
from an unreviewed maximum-rate config.

## 0. Define the experiment

Record these before sending traffic:

- endpoint origin, route, model, region, capacity product, and active serving
  configuration;
- the authorized time window and an operator who can stop the test;
- provider rate, token, and account quotas from provider telemetry or current
  documentation;
- a workload input with its source digest and known fidelity limits;
- request parameters, reasoning policy, tool schema behavior, and output cap;
- TTFT definition, acceptance targets, hard timeouts, and success-rate target;
- the generator region, host size, process count, and expected network path;
- current pricing source and effective date if cost will be reported;
- stop conditions for errors, latency, saturation, quota, cost, and production
  impact.

Do not assume a model is available on provisioned throughput because a
different model is. Confirm eligibility for the exact model and region before
creating a provisioned endpoint or making a provisioned-capacity claim.

## 1. Prove the instrument on the generator host

```bash
python3 -m pytest
python3 -m traffic_replay validate --port 0 --format json
```

The test suite is the implementation regression gate. `validate` checks this
host's end-to-end measurement path against the localhost oracle. Preserve the
JSON result with the run record. A passing mock check does not validate a
provider dialect, production network, or workload.

## 2. Validate authentication and protocol at minimal load

Use the one-command path with a measured profile or a small real-prompt set.
Keep the sizing hint small and retain the default preflight:

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_validation_small.json \
  --sizing-concurrency 2 \
  --duration 60 \
  --out-dir results/protocol-smoke \
  --fail-on none
```

This sends real inference traffic. Review:

- both representative requests reached the intended route;
- the response is valid SSE for this client;
- visible content or a structurally valid tool call completed cleanly;
- `request_attempts` is understood, especially usage fallback or auth refresh;
- prompt, completion, cached, and reasoning usage fields are either present
  with named source paths or explicitly absent;
- model, route, and request controls in the manifest match the intended target;
- endpoint metadata was captured, or its absence is explained;
- no secret or response body content appears in artifacts.

This stage validates mechanics only. Its latency is not a capacity result.

For an unattended production run, prefer a service-principal OAuth
machine-to-machine profile. Use a PAT only for development or a controlled
test where its scope, storage, lifetime, and rotation have been approved.

### Reasoning and tool-call endpoints

For a reasoning model, choose `ttft_definition=first_content` when the SLA
starts at the first visible-or-reasoning content delta, or
`ttft_definition=first_visible` when it starts at meaningful visible assistant
content. Those are the only two selectable TTFT definitions. First reasoning
content (`ttfr_ms`) and first tool-call fragment (`ttf_tool_call_ms`) are
reported separately but cannot currently be selected as the scored TTFT
basis.

Reasoning controls are model-specific request contract. Use the provider's
documented field and supported value for the exact model. Do not copy a
reasoning or template parameter from a different endpoint without a successful
protocol check. An accepted unknown field can still be ignored.

The harness does not guess these controls. A repeatable
`--probe-extra-body '{...}'` explicitly sends one additional real preflight
request per supplied candidate after an unreadable answer. Use it only for
model-documented candidates in an authorized probe; retain its stdout and
stderr because those calls precede the sealed runner artifact. A candidate is
not copied into the measured config; rerun with the selected object as
`--extra-body`.

The measured `extra_body` is persisted as reproducibility evidence, while
probe candidates and outcomes are reported in preflight text with displayed
values and errors credential-redacted. Secret-like keys and credential-shaped
values are rejected recursively before config output or traffic. Command
arguments can still be visible locally. Keep authentication in the endpoint
profile or token environment variable.

A larger output budget can reveal whether the model eventually produces a
visible answer, but it changes work and cost. It is diagnosis, not a neutral
fix. The measured run must use the product's actual budget and must report
global-cap truncation separately from the sampled output target.

Tool-call-only responses are acceptable only when the stream assembles at
least one nonempty function name whose arguments decode to a JSON object and
terminates without parse errors. This validates structure, not tool choice or
argument semantics. Use an application evaluator for semantic correctness.

## 3. Validate workload fidelity

Prefer one of:

- real prompts from an approved test dataset; or
- a schema-v2 `empirical_joint` profile built from complete token/cache
  triples; or
- a schema-v2 `quantile_cdf` profile with measured marginal knots.

Use schema v1 only when p50/p95 marginals are all that exists, and label the
unobserved tails and dependence as assumptions.

For a log-derived profile:

```bash
python3 scripts/profile_from_logs.py \
  --input request_metrics.jsonl \
  --name measured_workload \
  --mode empirical-joint \
  --out configs/profile_measured.json

python3 -m traffic_replay sample \
  --profile configs/profile_measured.json --n 50000 --seed 7
```

Check the source SHA-256, extraction counts, dropped incomplete rows, recovered
quantiles, unique empirical rows and cycle weight. The extractor emits no
prompt text or arbitrary source fields, but the input export remains sensitive
and must follow its data policy.

For prompt replay, quantify how many scheduled requests will be repeats. Cache
reuse caused by cycling a short prompt list is a property of the experiment.

For either mode, run a small endpoint sample and verify:

- endpoint-reported prompt-token size versus intended size;
- endpoint-reported completion size and finish reasons;
- global-cap truncation rate;
- cached-token coverage and paired achieved-versus-intended error;
- acceptable outcome coverage, including tool-call-only results;
- provider tokenizer and request formatting differences.

If these do not match, do not compensate by relabeling the requested profile as
achieved workload. Fix the materialization or use real prompts.

## 4. Establish a guarded fixed-rate ladder

`sweep` directly controls open-loop arrival rate. Start below the expected
knee, use a bounded worker and pending queue, and stop on any unqualified rung:

```bash
python3 -m traffic_replay sweep \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --rate 1,2,4,8 \
  --duration 120 \
  --cooldown 30 \
  --max-concurrency 256 \
  --max-pending-requests 512 \
  --ttft-p95 YOUR_TTFT_MS \
  --ttfg-p95 YOUR_TTFG_MS \
  --success-rate YOUR_RATE \
  --out-dir results/rate-sweep
```

The example rates are only a low starting ladder, not a recommended capacity
for an unknown endpoint. Pick authorized rungs from known traffic and quota
limits. A 30-second cooldown is the command default; it does not prove that a
provider's token, request, or account quota window reset.

At each rung, inspect external endpoint and quota telemetry as well as the
harness report. Stop when any configured operational guard is breached,
including:

- production impact or operator stop request;
- HTTP error or 429 increase;
- pending-limit drops, queue wait, delivered-rate shortfall, or generator CPU
  and network saturation;
- unacceptable answer outcomes or stream parse errors;
- caller-experienced SLA miss or hard timeout;
- achieved workload drift, missing usage, or global-cap truncation;
- cost or token budget exhaustion;
- endpoint autoscaling or cache state that makes the rung incomparable.

An HTTP 429 identifies rate limiting but not the quota dimension. Determine
whether input tokens, output tokens, requests, hourly policy, or another limit
caused it from provider telemetry.

## 5. Run a long confirmation at one fixed condition

After a valid ladder identifies a candidate rate, run one configuration long
enough to observe multiple 60-second stability windows. Five minutes produces
five nominal windows, but setup and drain remain outside the schedule.

Copy an example config, replace all placeholders, set one measured workload,
and retain conservative client bounds. `configs/run_pt_full.json` starts at
`rate_scale=0.1`, `max_concurrency=256`, and
`max_pending_requests=512`; it is not permission to raise `rate_scale` to 1.0
on one process. Size client resources from measured mean occupancy and validate
delivery at every step.

Little's Law uses mean occupancy:

```text
mean in flight approximately arrival rate * mean end-to-end service time
```

Do not substitute p95 latency into this equality. Tail latency is useful for
headroom and timeout analysis, but it does not calculate mean concurrency.
Also include connection setup, retries, fallbacks, response drain, and local
resource limits in the client capacity plan.

Run the lower-level config path when every setting must be explicit:

```bash
python3 -m traffic_replay run \
  --config configs/run_pt_full.json \
  --format json
```

The run exits nonzero on a default miss/invalid gate. Save stdout, stderr, the
exact config, source commit, and sealed output directory.

## 6. Shard only when one generator is proven insufficient

Sharding adds clock, identity, and aggregation failure modes. First demonstrate
with external host telemetry and harness queue/delivery evidence that one
generator cannot safely deliver the authorized rate.

Every shard needs the same future `start_at_unix`, `run_id`, workload, seed,
request parameters, and `shard_total`, plus a unique zero-based `shard_index`.
Synchronize clocks. The shared start must allow enough setup time for config
validation, target evidence, optional sizing, schedule generation, and
calibration.

Do not combine independent sizing passes into a sharded production claim. For
a reproducible shard test, establish the fixed global rate first, put it in the
shared config, split the schedule, and divide client resources deliberately.

After all shards seal successfully:

```bash
python3 -m traffic_replay merge results/merged \
  results/shard-0/RUN_DIR \
  results/shard-1/RUN_DIR
```

The merge must prove a complete nonoverlapping global-index set. Never use
`--force` to turn missing coverage into a capacity result; forced output is
explicitly INVALID. Exact monotonic caller durations present in source rows
are pooled and can drive merged SLA scoring. Legacy schedule/send timestamps
are not reconstructed across different run epochs; when exact caller clocks
are absent, the merge labels SLA scoring as service-time only. Read each shard
for stability, wire lateness, and in-flight concurrency because those time axes
are not pooled.

## 7. Review and sign off evidence

Accept a result only if all of the following are true:

- `.traffic-replay-complete` exists and `.traffic-replay-writing` does not;
- manifest schema is 3 and every bound artifact hash, byte count, and request
  row count verifies;
- the completion marker's artifact ID, manifest digest and byte count, and
  request-row count match the authenticated manifest and journal;
- source state is clean and the commit is retained;
- endpoint, workload, request, schedule, execution, and artifact identities
  match the experiment record;
- scheduled and delivered load match within the documented acceptance policy;
- no pending-limit drops or unexplained physical duplicate attempts exist;
- acceptable answer/tool outcome and usage coverage are sufficient;
- achieved token and cached-token distributions match the intended workload;
- exact caller timing covers the scored population;
- the quoted quantile meets its sample floor: p50 20, p90 100, p95 200, p99
  1000 acceptable outcomes;
- the observed success fraction meets its target and the one-sided 95 percent
  Wilson lower confidence bound also meets it, under the stated independent
  outcome assumption;
- stability is established for the run itself;
- cost is complete or explicitly withheld due to missing usage;
- external endpoint, quota, and generator telemetry agree with the conclusion.

An interrupted directory can retain useful newline-complete rows in
`requests.jsonl.partial`, but it is diagnostic evidence only. Do not rename the
journal or manufacture a completion marker.

## 8. Compare providers or serving products cautiously

Keep the real prompts or workload profile, schedule identity, seed, output cap,
TTFT definition, acceptance policy, and load mode fixed. Then verify achieved
prompt, output, and cached-token distributions rather than assuming identical
request JSON creates identical work.

Provider routes can differ in tokenizer, chat template, reasoning controls,
tool-call framing, cache accounting, usage coverage, fallback behavior,
autoscaling, and quotas. `compare` requires sealed, integrity-verified inputs
and checks configured compatibility. A mismatch produces a sealed but
prominently INVALID diagnostic comparison rather than a winner. The comparison
manifest binds its rendered report and exact source manifest/summary
identities, but the source runs remain the underlying benchmark evidence. Even
a valid comparison cannot certify semantic or protocol equivalence.

Provisioned and pay-per-token products answer different capacity and cost
questions. Compare them only when the exact model is supported by both and the
claim states the product, allocated capacity, utilization, and current pricing
basis.
