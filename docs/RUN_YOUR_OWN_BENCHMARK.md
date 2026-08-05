# Benchmark your own endpoint

Measures what your traffic will actually experience: time to first token,
end to end, achieved prompt-cache rate, and whether the endpoint holds up at
the rate you need. Python 3.10 and numpy are the whole dependency list.

## 1. Setup

```bash
git clone https://github.com/debu-sinha/llm-traffic-replay.git
cd llm-traffic-replay
python3 -m traffic_replay validate     # self-test, no endpoint needed
```

`validate` runs the whole measurement path against a bundled mock server
with a known latency model and reports its own error. If that doesn't say
PASS, nothing downstream is worth reading.

Auth is either an environment variable or a `~/.databrickscfg` profile:

```bash
export DATABRICKS_TOKEN=<your PAT>
# or add --auth-profile <profile-name> to any command below
```

## 2. Run it

One command. Replace the host, the endpoint name, and your token sizes.

```bash
python3 -m traffic_replay benchmark \
  --host https://<your-workspace>.cloud.databricks.com \
  --endpoint <your-endpoint-name> \
  --input-tokens 10000,24000 \
  --output-tokens 40,90 \
  --cache-hit-rate 0.6,0.87 \
  --concurrency 10 --duration 300 \
  --ttft-p50 500 --ttft-p95 900 \
  --ttfg-p50 700 --ttfg-p95 1500 \
  --success-rate 0.99
```

Each size takes `p50` or `p50,p95`. Pass one number and the p95 is inferred.
The targets are yours. The report scores against them and the process
exits non-zero if they're missed, so this works as a CI check.

Open `results/benchmark/<timestamp>/report.html` when it finishes.

## 3. Use your own prompts instead of synthetic text

Swap the three size flags for a JSONL file, one request per line. Any of
these three shapes work:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]}
{"prompt": "..."}
{"text": "..."}
```

```bash
  --prompts your_traffic.jsonl
```

If you have request logs with token counts, build a traffic profile from
them instead and keep the synthetic generator:

```bash
python3 scripts/profile_from_logs.py --input your_logs.jsonl \
  --name prod --out configs/prod.json
# then pass --profile configs/prod.json
```

## 4. The one thing that will bite you: reasoning models

Several models on Databricks emit reasoning tokens before the answer, and
those count against `max_tokens`. With a small output budget the model can
spend the entire budget thinking and return HTTP 200 with nothing readable.
Every request looks successful and no answer ever arrives.

**You don't need to know which models these are, and you won't waste a run
finding out.** Before any load starts, the tool sends two requests. If the
endpoint can't answer at your budget it tries each known reasoning control,
tells you which one works, and then stops rather than spending your time and
tokens on a run it has already shown will produce nothing:

```
[preflight] this is a REASONING model. it emits thinking tokens before the
            answer, and they count against max_tokens.
[preflight] trying the reasoning controls this endpoint might accept:
[preflight]   reasoning_effort=none    WORKS   answered, finish stop, 109 tokens
[preflight]   thinking.type=disabled   ignored accepted, still no visible answer
[preflight] use this: --extra-body '{"reasoning_effort": "none"}'

[preflight] STOPPING before the load starts. this run would have produced no
readable answers, so it would cost you time and tokens for a verdict we can
already give you.

  re-run with the control that worked:

    --extra-body '{"reasoning_effort": "none"}'
```

Add that flag to the command from step 2 and run it again. The flag differs
by model, and some models accept one and silently ignore it, which is why the
tool tests rather than assumes. If nothing works it says so and tells you the
choice is a bigger output budget or a different model. Pass `--force` to run
anyway.

## 5. Reading the result

The report leads with a verdict. It's green only when the run met your
targets **and** nothing undermines the measurement. Anything that doesn't
gets named: errors, an unstable window, a sample too small for the quantile
you scored, missing token usage, or the client sitting too far from the
endpoint.

Read the **answers** block before the latency table. `started a readable
answer` well below `returned HTTP 200` means the latency figures describe
only the requests that answered, and the report will say so.

## 6. Two things worth setting up before you trust a number

**Run the generator in the same region as the endpoint.** Every latency
figure contains one network round trip. The report measures and prints that
distance, and subtracts it for you, but removing it is better than
subtracting it. From across the US it's roughly 80 ms of every TTFT.

**Use a provisioned throughput endpoint for capacity questions.** On
pay-per-token, workspace rate limits usually bind before the model does, so
you end up measuring a quota rather than the endpoint. The report tells you
when that happened. Pay-per-token is fine for checking the setup works and
your prompts flow correctly.

## 7. Finding your ceiling

When the question is "how much can this take", climb a rate ladder in one
command instead of guessing a concurrency:

```bash
python3 -m traffic_replay sweep \
  --host https://<your-workspace>.cloud.databricks.com \
  --endpoint <your-endpoint-name> \
  --rate 2:32 --duration 120 \
  --ttft-p95 900 --ttfg-p95 1500 --success-rate 0.99
```

It stops at the first rung that fails and reports the highest rate that
held, plus the concurrency that rate turned out to carry.

---

Questions, or a result that looks wrong: open an issue on the repo, or come
back to us with the `results/<run>/` directory. It has the exact config, every
request, and a manifest pinning the code version, so a number can always be
traced back to what produced it.
