# Follow-up work

These are known product limits, not completed features.

## Add an explicit interrupted-run recovery command

Interrupted runs preserve newline-complete rows in
`requests.jsonl.partial`, but recovery is currently an inspection procedure,
not a CLI workflow. A future command should:

- verify the writing marker and regular-file boundaries;
- copy, never mutate, the interrupted evidence;
- ignore at most one truncated final fragment;
- report recovered and lost/unknown coverage;
- emit an explicitly diagnostic artifact that cannot be mistaken for a sealed
  benchmark or merged as valid evidence.

## Add semantic evaluators

The current answer policy validates visible content or tool-call structure and
clean stream completion. It cannot judge factual correctness, instruction
following, tool choice, argument semantics, safety behavior, or task success.
Production qualification needs application-specific evaluators with their own
versioned inputs and provenance.

## Expand provider conformance fixtures

The streamed Chat Completions subset needs captured, scrubbed fixtures for more
documented provider dialects, including SSE framing, usage-only terminal
chunks, reasoning channels, tool-call deltas, cached-token fields, and explicit
unsupported-field errors. Conformance should be proven before a provider is
described as supported.

## Add first-class quota evidence

HTTP 429 is observable, but the harness cannot determine which token, request,
hourly, account, or provider policy caused it. A future provider adapter could
attach quota headers and control-plane telemetry without persisting secrets or
response content. Until then, quota diagnosis remains external.

## Add scalable online quantiles

Active client work and futures are bounded by workers and pending requests,
but the complete global schedule and profile-mode sampled workload arrays are
proportional to the global request count. Final exact percentile calculation
also rereads all persisted replay rows. Very large runs need streaming schedule
and workload construction plus a versioned approximate-quantile mode with
explicit error bounds and a report label that prevents mixing exact and
approximate summaries.

## Add pricing-source provenance

Pricing values are user supplied and the manifest records the effective rates,
but not a source URL, region, model SKU, or effective date. A strict optional
pricing provenance object would make cost evidence easier to audit while
keeping automatic price retrieval out of the critical run path.

## Improve target configuration evidence

Endpoint metadata capture is best effort and specific to recognized serving
routes. It does not prevent a target from changing after the pre-traffic
snapshot. Provider-specific immutable deployment revision identifiers, when
available, should be captured and compared with post-run state.
