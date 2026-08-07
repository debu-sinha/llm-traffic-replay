# Benchmark your own endpoint

This is the shortest defensible path from an endpoint to a sealed report. For
an authorized production test, also follow
[Production testing](PRODUCTION_TESTING.md).

## 1. Install and validate

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest
python3 -m traffic_replay validate --port 0 --format json
```

The test suite is required for the full regression check. `validate` tests the
local timing pipeline against a bundled oracle; it does not test a provider.

## 2. Choose authentication

Named Databricks profile:

```bash
databricks auth profiles
```

The selected profile must contain a host matching the endpoint origin. PAT
profiles use the configured token. OAuth profiles invoke
`databricks auth token -p PROFILE`. Profile resolution fails closed on a
missing profile, host mismatch, or token failure.

For unattended production, prefer a service-principal OAuth
machine-to-machine profile. Use PAT profiles for development or controlled
testing under the workspace's credential policy.

Environment token:

```bash
export DATABRICKS_TOKEN='...'
```

Do not put a token in JSON or in a report label. Remote bearer-authenticated
HTTP is rejected; use HTTPS. Explicit loopback HTTP is reserved for local
tests.

The measured `extra_body` is persisted as reproducibility evidence, and probe
candidates/outcomes are reported with displayed values and errors
credential-redacted. Secret-like keys and credential-shaped values are
rejected recursively before config output or traffic. Command arguments can
still be visible locally, so keep authentication in the profile or token
environment variable.

## 3. Choose real prompts or a measured profile

Real prompts give the strongest content fidelity:

```jsonl
{"messages":[{"role":"user","content":"Your approved test prompt"}]}
```

Save one JSON value per line, then pass `--prompts prompts.jsonl`.

If only token/cache logs are available, generate a profile. Joint mode retains
observed combinations and frequencies without emitting prompt content:

```bash
python3 scripts/profile_from_logs.py \
  --input request_metrics.jsonl \
  --name measured_workload \
  --mode empirical-joint \
  --out configs/profile_measured.json

python3 -m traffic_replay sample \
  --profile configs/profile_measured.json --n 50000 --seed 7
```

The output records extraction counts and source SHA-256. Review dropped rows
and recovered anchors. If you use legacy `--mode quantiles`, the output is a v1
p50/p95 marginal model and does not preserve p90/p99 or cross-field
dependence.

## 4. Run preflight and a small benchmark

With a profile:

```bash
python3 -m traffic_replay benchmark \
  --host https://YOUR-WORKSPACE-HOST \
  --endpoint YOUR-ENDPOINT-NAME \
  --auth-profile YOUR-DATABRICKS-PROFILE \
  --profile configs/profile_measured.json \
  --sizing-concurrency 2 \
  --duration 60 \
  --ttft-p95 YOUR_TTFT_MS \
  --ttfg-p95 YOUR_TTFG_MS \
  --success-rate YOUR_RATE \
  --out-dir results/first-run
```

With prompts, replace `--profile ...` with `--prompts prompts.jsonl`.

Preflight sends two representative inference requests. After both reach HTTP
200, if either lacks an acceptable outcome, explicitly supplied
`--probe-extra-body` candidates can each send one additional request. The
harness does not guess provider controls. All are real traffic. An acceptable
outcome is visible content or a
structurally valid tool call with a nonempty function name and arguments that
decode to a JSON object, plus clean stream completion and no parse errors; it
is not a correctness grade.

Preflight runs before the measured runner claims its output directory. Its
probe rows are not in the sealed request journal or manifest. Capture stdout
and stderr separately when those decisions need an audit trail.

`--sizing-concurrency 2` measures unloaded service time and derives one fixed
open-loop rate plus a worker pool. When the worker cap is omitted, the derived
pool is capped at the default 256-thread safety ceiling; an explicit positive
cap replaces that ceiling. It does not hold two requests in flight. The
measured report shows the concurrency that resulted.

## 5. Handle reasoning controls explicitly

When the endpoint emits reasoning before visible content, decide what your SLA
means:

- `first_content`: first visible or reasoning content delta;
- `first_visible`: first meaningful visible assistant content.

Tool-call-only outcomes have a separate time-to-first-tool-call metric.

Pass only a model-documented control, for example:

```bash
--extra-body '{"reasoning_effort":"low"}'
```

The accepted field and values depend on the exact model and provider. A
successful HTTP status does not prove an unknown control changed behavior.
Use preflight output and a small run to verify it before load.

To test multiple documented candidates after an unreadable preflight, repeat
`--probe-extra-body '{...}'`. Each candidate is another real request before
the sealed runner begins, so use this only as an authorized, explicit probe.
Candidates do not change the measured config. Rerun with a successful object
as `--extra-body`.

Do not solve a reasoning-only result by silently raising the output cap. A
larger cap changes cost and work. If the product requires it, update the
workload contract and rerun from preflight.

## 6. Read the sealed evidence

Open `report.html` for navigation, but treat `summary.json`,
`requests.jsonl`, `start.json`, and manifest schema v3 as the audit record.
Read in this order:

1. completion and artifact integrity;
2. acceptable outcomes, HTTP statuses, failures, and physical request attempts;
3. delivered rate, pending-limit drops, queue wait, and measured concurrency;
4. achieved prompt/output sizes and cached prompt-token fraction;
5. caller-experienced SLA metrics and coverage;
6. sample-size and stability cautions;
7. service clocks, usage throughput, and cost.

Minimum answer-latency sample floors are 20 for p50, 100 for p90, 200 for p95,
and 1000 for p99. Smaller samples remain diagnostic only. A clean success-rate
verdict also requires the one-sided 95 percent Wilson lower confidence bound,
not just the observed fraction, to meet the target. That inference assumes
independent request outcomes.

`NOT REPORTED` for cached tokens means missing endpoint evidence, not a zero
cache rate. Reasoning stream deltas are counts of SSE deltas, not tokens.

The network card is a best-effort diagnostic. After resolving DNS, it records
`tcp_connect_min_ms` and `tcp_connect_median_ms`. Neither is an exact RTT or
endpoint time, and neither should be subtracted from TTFT.

## 7. Rerun the exact config

The one-command path writes:

```text
results/first-run/profile.json       only when token-size flags built a profile
results/first-run/run-config.json    rerunnable config
results/first-run/RUN_DIR/           sealed measured artifacts
```

Rerun the config directly:

```bash
python3 -m traffic_replay run \
  --config results/first-run/run-config.json \
  --format json
```

The runner snapshots the input bytes before traffic, but a new execution is a
new experiment. The same seed reproduces the client plan, not endpoint,
network, autoscaling, or cache state.

## 8. Find a ceiling with fixed-rate rungs

After the small run is valid:

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
  --success-rate YOUR_RATE
```

Choose authorized rates based on known traffic and quota, not the example.
The command stops on the first non-OK rung by default. A 429 indicates rate
limiting but cannot identify the quota dimension. Confirm the cause with
provider telemetry.

The highest rate submitted is not the capacity. The report names the highest
tested rung with an unqualified OK verdict, and even that claim is limited to
the observed workload, endpoint state, generator path, and test window.

## 9. Understand retries and duplicates

Transport retries default to zero. When enabled, a failure after `POST` can
duplicate inference and billing. Usage-option rejection and one credential
refresh can also cause a second physical `POST` even with zero configured
retries. Inspect `request_attempts`, `connection_attempts`, and `retry_reasons`.

The client is at-least-possibly-once under ambiguous transport failure, not
exactly once.

## 10. Recover diagnostics, not a benchmark

An interrupted run retains `.traffic-replay-writing`, `start.json`, and
`requests.jsonl.partial`. Newline-complete rows can be inspected; a truncated
last fragment may be ignored. Do not rename the journal or create a completion
marker. `merge` and `compare` correctly reject unsealed evidence.

For a completed input, aggregate readers verify the completion marker's
artifact ID, manifest digest and byte count, and request-row count against the
authenticated manifest and journal.
