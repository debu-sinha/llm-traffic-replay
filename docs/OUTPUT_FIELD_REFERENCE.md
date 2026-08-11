# Output and report field reference

This is the data dictionary for a completed run and its verification receipt.
The HTML and Markdown reports embed the same human-facing metric glossary.
Fields marked unavailable, unknown, `null`, or `NOT REPORTED` are missing
evidence; they never mean zero or success.

## Traffic phases and calibration

| Term | Exact meaning |
|---|---|
| `preflight` | Real setup requests that prove representative request/stream compatibility before replay. Excluded from replay performance; included in complete traffic/quota evidence. |
| `probe` | Optional real setup request for an explicitly supplied provider control after an unreadable preflight. Acceptance proves only that the request was accepted, not that the control changed behavior. |
| `sizing` | Unloaded real requests used to convert `sizing_concurrency` into one fixed open-loop rate. Not replay performance. |
| `calibration` | Real, paid, unloaded requests used only to estimate the synthetic generator's characters per endpoint-reported prompt token. They are not warm-up exclusions, quality checks, latency/capacity samples, or quota reservations. They are excluded from replay performance and included in complete traffic/quota evidence. Any positive count can warm routing, workers, model state, and caches. `calibrate_n=0` disables them; actual count is `min(calibrate_n, replay rows)`. Calibration affects synthetic input construction only; it does not make the requested output length more likely. |
| `replay` | Logical workload rows scheduled inside the measured open-loop load window. Headline performance and acceptance populations come from eligible replay rows only. |

One logical row can make multiple physical POST attempts because of retries.
`request_attempts` counts sends; `connection_attempts` also includes failures
before a request could be sent. Final-attempt request-path latency excludes
earlier attempts. `caller_*` latency includes the caller's total scheduled wait.

## `summary.json` top-level fields

Quantile blocks (`*_ms`, intended/achieved fractions, and similar objects) use
`p50`, `p90`, `p95`, and `p99` for observed percentiles, `n` for the eligible
row count, and optional coverage/note fields to name the population and source.

| Field | Meaning |
|---|---|
| `harness_version` | Package version that summarized the run. |
| `run` | Frozen run identity, endpoint/workload/configuration metadata, bindings, and provenance; detailed below. |
| `decision` | Five independent canonical decisions plus tested-load facts; detailed below. An unverified source report intentionally says `VERIFY_REQUIRED`. |
| `requests_total`, `requests_ok`, `requests_failed` | Measured replay logical rows; harness-successful rows; and failed rows. Harness success proves protocol completion, not answer acceptability. |
| `requests_retried`, `physical_post_attempts` | Retry-marked logical rows and reconciled logical-row/physical-attempt counts and trigger coverage. |
| `error_rate` | `requests_failed / requests_total`; not an HTTP-only error rate. |
| `failures_by_error`, `failures_by_http_status` | Stable error categories and terminal HTTP-status counts for failed rows. |
| `http_429`, `http_429_count`, `http_429_rate`, `quota_limited` | Captured row-level HTTP 429 evidence, denominator/coverage/phase details, compatibility aliases, and whether quota rejection was observed. Zero does not establish provider headroom. |
| `runtime_quota_admission` | Command-local per-attempt admission reconciliation: configured limits, admitted/denied attempts, baselines, snapshots, and invariant errors. It excludes unrelated workspace traffic. |
| `rate_limits`, `observed_rate_windows` | Reviewed provider snapshot evaluation and observed rolling QPS/QPH/token/request-byte windows, including coverage and phase scope. |
| `sample` | Evidence thresholds for percentiles and which quantiles are indicative because `n` is too small. |
| `answers` | Readable/visible/refusal/tool-call/finish-reason outcome counts and answer-judgment coverage. |
| `latency_population` | Exact eligibility rule and row count used for headline latency. Failed/unacceptable rows remain in error evidence. |
| `latency_basis` | Plain-language clock boundary for headline request-path latency. |
| `stream_event_definition` | Exact parser/content boundaries for first event, reasoning, refusal, visible content, tool calls, and terminal completion. |
| `latency_correction_note`, `latency_correction_provenance` | Explanation and source clocks for exact caller-experienced latency; this is not a numerical post-hoc subtraction. |
| `ttfb_ms` | Final-attempt send to first response byte; connection setup excluded. |
| `ttse_ms`, `ttse_corrected_ms` | Final-attempt and exact-caller time to first parsed SSE event. This is a protocol event, not necessarily visible content. |
| `ttft_definition` | Whether scored response start is `first_content` or `first_visible`. |
| `ttft_ms`, `ttft_corrected_ms` | Final-attempt and exact-caller configured response-start latency. `first_content` accepts visible, reasoning, or refusal onset; `first_visible` requires visible assistant content. |
| `ttfv_ms`, `ttfv_corrected_ms` | Final-attempt and exact-caller time to first visible assistant content. |
| `ttf_tool_call_ms` | Time to the first structurally valid complete tool call. |
| `e2e_ms`, `e2e_corrected_ms` | Final-attempt and exact-caller time through terminal response/stream completion. |
| `connect_ms` | Fresh DNS/TCP/TLS setup diagnostic. Excluded from request-path metrics and not representative of pooled/HTTP2 clients. |
| `interchunk_max_ms` | Per-row maximum gap between parsed stream events, summarized over eligible rows. |
| `tpot_ms`, `completion_tpot_ms`, `tpot_scope`, `tpot_note` | Time-per-token diagnostics and their exact token/population definition. Endpoint completion tokens can include hidden reasoning; these are not visible-output TPOT without exact visible-token accounting. |
| `throughput` | Achieved request and endpoint-reported token rates, usage coverage, reasoning/visible-token qualifications, and warnings. |
| `schedule` | Planned replay count, seconds, schedule source/digest, rate shape, and shard context. |
| `arrivals` | Achieved client request-start rate, dispatch lag, request-start lateness, pending drops, and clock basis. These are not provider receipt timestamps. |
| `drift` | Windowed stability/error/event-coverage evidence and drift classification after response drain. |
| `network_path` | Best-effort DNS/TCP-connect floor from the client location. It is not exact RTT or endpoint processing time and must not be subtracted. |
| `response_identity` | Observed response model/object/service-tier/fingerprint consistency and expected-identity comparison. |
| `intended_cache_fraction` | Synthetic workload's constructed reusable-prefix token share. Not a cache-hit probability. |
| `achieved_cache_fraction` | Endpoint-reported cached/prompt token share with source-field and coverage evidence. Missing means unknown, not zero. |
| `cache_fidelity` | Paired achieved-versus-intended cache fraction comparison, coverage, error, and warning. |
| `token_targeting` | Endpoint-reported prompt/completion tokens versus intended synthetic targets, usage coverage, errors, and finish reasons. `max_tokens` is only a cap. |
| `calibration_warmth` | Calibration row count, body-hash coverage, exact calibration/replay payload overlap, replay overlap share, and warm-state warning. It does not assert a cache hit. |
| `sla` | Customer-owned target configuration and scored evidence. Latency rows contain the required and observed meeting fractions, nearest-rank point estimate, one-sided 95% Wilson lower bound, and `statistically_demonstrated`; a point-estimate hit without confidence is `NOT PROVEN`, not an acceptance pass. Success-rate evidence uses the same confidence rule. |

## `run` identity and provenance

| Field group | Meaning |
|---|---|
| `artifact_id`, `logical_run_id`, `execution_id`, `workload_id`, `run_id` | Unique sealed artifact, logical experiment, execution, workload, and optional caller-supplied/shard identifiers. |
| `title`, `label`, `harness_version` | Sanitized presentation context and implementation version. |
| `endpoint_base_url`, `endpoint_path`, `endpoint_model` | Exact public target origin/path and optional request model. No endpoint-name inference is implied. |
| `endpoint_binding`, `invocation_binding` | Pretraffic control-plane and request-route/body binding evidence. |
| `endpoint_metadata`, `endpoint_metadata_after`, `endpoint_metadata_stability`, `endpoint_metadata_warning` | Normalized before/after control-plane snapshots and comparison. |
| `request_params` | Adapter, streaming mode, temperature, output cap, and reviewed provider controls used to serialize requests. |
| `transport` | Actual fresh-connection HTTP policy and any declared production comparability. |
| `input_mode`, `profile*`, `prompts_file` | Workload source, measured/illustrative provenance, and frozen source references. |
| `schedule_identity`, `index_identity`, `global_index_*`, `shard` | Exact schedule/index digests, range, and shard identity. |
| `seed`, `cpt_final`, `derived_qps`, `sizing_concurrency_*`, `max_pending_requests`, `ttft_definition` | Effective generator/sizing/scheduling controls. `cpt_final` is only the synthetic construction estimate. |
| `preflight_gate`, `setup_artifact_reference` | Compatibility result and immutable sibling setup-traffic linkage. |
| `quota_plan`, `runtime_quota_guard`, `runtime_quota_guard_baseline` | Pretraffic budget and command-local runtime admission configuration/baseline. |
| `network_path`, `start_at_unix` | Client-location diagnostic and requested coordinated start. |

## Canonical `decision` fields

Every decision state has `code`, `label`, `severity`, `reason`, `reason_codes`,
and `reason_details`. The code is the stable machine value; the reason explains
this run. The five independent dimensions are:

| Field | Question |
|---|---|
| `evidence_integrity` | Has the enclosing manifest/completion chain been externally checked? |
| `measurement_validity` | Are protocol, identity, delivery, workload, usage, sample, and stability evidence usable? |
| `customer_sla` | What happened against explicitly supplied acceptance targets? |
| `quota_state` | What do captured 429 and local admission evidence show? |
| `endpoint_capacity` | Did a valid verified test point hold? This never establishes an endpoint ceiling. |
| `tested_load` | Observed replay count/outcomes, achieved rate, schedule source/window, and complete captured traffic count; facts, not a sixth decision. |

## `requests.jsonl` fields

Each line is one logical request-operation row. Prompt and response content are
intentionally omitted.

| Fields | Meaning |
|---|---|
| `request_id`, `phase`, `global_index`, `sample_index`, `prompt_index`, `body_request_id` | Stable row and workload-plan identity. |
| `scheduled_s`, `dispatch_lag_ms`, `queue_wait_ms` | Planned offset, dispatcher delay, and client pool/queue wait. |
| `first_attempt_unix`, `first_send_unix`, `t_send_unix`, `finished_unix` | First connection attempt, first physical send, final-attempt send, and logical completion wall clocks. |
| `connection_attempts`, `request_attempts`, `retries`, `retry_reasons` | Connection tries, physical HTTP sends, retry count, and bounded trigger categories. |
| `connect_ms` | Final-attempt fresh DNS/TCP/TLS setup time. |
| `ttfb_ms`, `ttse_ms`, `ttft_ms`, `ttfr_ms`, `ttfv_ms`, `ttf_tool_call_ms`, `e2e_ms`, `interchunk_max_ms` | Final-attempt request-path latencies: first byte, parsed event, configured first content, refusal, visible content, valid tool call, completion, and largest event gap. |
| `caller_send_ms`, `caller_ttfb_ms`, `caller_ttse_ms`, `caller_ttft_ms`, `caller_ttfr_ms`, `caller_ttfv_ms`, `caller_ttf_tool_call_ms`, `caller_e2e_ms` | Exact caller-experienced latency from scheduled time, including queueing/retries, for the corresponding milestones. |
| `status`, `ok`, `error` | Terminal HTTP status, harness protocol success, and stable redacted error category. |
| `response_content_type`, `response_mode`, `endpoint_adapter` | Observed media type and exact versioned wire contract. |
| `content_chunks`, `reasoning_chunks`, `refusal_chunks`, `tool_call_chunks` | Parsed SSE delta/event counts. They are not token counts. |
| `visible_content_seen`, `reasoning_seen`, `refusal_seen`, `tool_call_seen`, `valid_tool_calls` | Observed semantic channel presence and structurally valid tool-call count. |
| `stream_complete`, `finish_reason`, `truncated`, `parse_errors`, `parse_error_details` | Terminal/framing completeness, provider finish reason, cap-driven truncation, and bounded parse diagnostics. |
| `prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens` | Endpoint-reported usage. Completion can include hidden reasoning. Missing cached/reasoning usage means unknown. |
| `cached_tokens_source`, `reasoning_tokens_source` | Exact recognized response path that supplied usage. |
| `intended_input_tokens`, `intended_output_tokens`, `intended_cache_fraction`, `chars_sent`, `doc_id`, `max_tokens_requested` | Constructed workload intent and actual character/budget controls. Output intent is not a promise of generated length. |
| `served_model_name`, `response_model`, `response_object`, `system_fingerprint`, `service_tier` | Redacted response identity/serving metadata. |
| `response_id_sha256` | SHA-256 of response ID rather than the raw identifier. |
| `request_body_sha256`, `physical_request_body_sha256s` | Logical representative and per-attempt serialized request-body bindings without retaining body content. |
| `quota_guard_id`, `quota_guard_denied`, `quota_guard_events` | Command-local per-attempt admission linkage and bounded decisions. |
| `http_error_body_sample_bytes`, `http_error_body_sample_sha256` | Size and digest of bounded error-body evidence; no raw error body. |
| `reasoning_control_probe`, `probe_candidate_rejected`, `setup_request_binding`, `setup_request_binding_sha256` | Setup/probe disposition and exact request-plan/body binding evidence. |

## Other sealed files

| File | Meaning |
|---|---|
| `start.json` | Pretraffic frozen configuration, input hashes, schedule/workload identity, source state, endpoint binding, and claimed artifact/execution IDs. |
| `manifest.json` | Canonical artifact inventory with byte counts and SHA-256 bindings plus run/result identity. |
| `.traffic-replay-complete` | Final completion marker bound into the manifest chain. Presence alone is insufficient; verify the chain. |
| `report.md`, `report.html` | Human views derived from `summary.json`. The HTML is standalone and contains the embedded field glossary. |
| verification `verification.json` | External receipt identity, source artifact bindings, independent decisions, reconstructibility of source and verifier, assurance limits, and verifier version/time. |
| `verified-report.md`, `verified-report.html` | Authoritative human views inside the completed verification receipt. Internal SHA-256 consistency is not a digital signature, authorship proof, or trusted timestamp. |
