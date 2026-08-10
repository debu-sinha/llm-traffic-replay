# Follow-up work

These are known product limits, not completed features.

## Choose and publish the repository license

The public repository currently has no license file or package license
expression. Copyright therefore remains reserved by default, which prevents
downstream teams from assuming reuse, modification, or redistribution rights.
The repository owner and the appropriate legal reviewer must select the
license; the build must then ship the exact license text and matching package
metadata. This is an ownership decision, not a value the tool can infer.

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

## Add a production-qualified Unity AI Gateway adapter

Gateway Chat Completions is protocol-diagnostic only today. A production-
qualified adapter must:

- bind the requested fully qualified model-service name to destination identity
  reported in every response;
- capture and compare Unity Catalog model-service destinations, routing, and
  fallback configuration before traffic and after response drain;
- enforce the conservative intersection of Gateway and downstream quotas;
- cover routing changes, fallbacks, quota responses, and HTTP 429 attribution
  with Gateway-specific mocks.

Until these are sealed as evidence, a Gateway run must not publish a supported
quota or capacity conclusion.

## Add provider-side quota attribution and shared-workspace telemetry

The current runtime guard gives exact local admission evidence for configured
token/query windows and serialized request-byte ceilings, and HTTP 429 remains
observable across every captured phase. Neither can determine which provider
dimension or component caused a 429, see unrelated workspace traffic, or prove
provider burst/headroom state. A future provider adapter could attach documented
quota headers and control-plane telemetry without persisting secrets or response
content. Until then, provider-side diagnosis and shared-workspace reconciliation
remain external.

## Add production transport adapters

The current client opens a fresh HTTP/1.1 connection for each physical attempt.
Reports correctly qualify capacity unless an operator declares that production
uses the exact same policy. Add bounded pooled keep-alive and HTTP/2 adapters,
make their pool/stream limits explicit in sealed evidence, and prove connection
reuse, retry, cancellation, deadline, and quota-admission behavior under load.
Adapter selection must remain a controlled experimental variable and must not
silently change between sweep rungs.

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
routes. The runner now compares pre-runner-target and post-drain normalized
endpoint summaries, but two captures cannot prove that an undocumented
data-plane revision stayed fixed between them. Provider-specific immutable
deployment revision identifiers, when available, should be captured and bound
to every response or compared with both control-plane captures.
