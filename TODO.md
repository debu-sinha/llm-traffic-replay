# Follow-up work

## Scrub the customer name from the repo (do before any wider release)

The customer name appears throughout the repo from earlier commits. It is
already public on `origin/main`, so this is a cleanup rather than a leak, but
it should land before the project is pointed at a wider audience.

Where it currently appears:

- `configs/profile_decagon_20260723.json`, `configs/profile_decagon_poc_doc_20260727.json`
  (filenames and `provenance` / `label` text)
- `configs/run_prompts.json`, `configs/run_pt_full.json` (profile paths)
- `README.md` (several config path references and one example `out_dir`)
- `scripts/profile_from_logs.py` (docstring example)
- `tests/test_report_accuracy.py` (profile path)

Both profile JSONs are also base64-embedded in the notebook payload by
`scripts/pack_notebook.py`, so the customer's stated traffic figures ship
inside the notebook. Rename the profiles to something neutral, update every
reference, and re-run `python3 scripts/pack_notebook.py`.

## Bucket failed requests in the stability check

`_drift_block` buckets successful requests only, so an endpoint that degrades
into errors can still read `stable`: the survivors stay fast and the window's
ok-count drops. Bucketing failures changes the meaning of the `n` column and
the p95 basis, so it wants its own change with its own tests. Pair it with the
error-rate warning `compare` already emits.

## Record which serving config was read

`endpoint_meta._summarize` reads the active `config` only. During an endpoint
update the pending config carries the new workload shape. Recording a
`config_source` field and rendering it would let the card say which one it
described.

## Document the dispatch-lag and connect_ms interaction

Dispatch lag is stamped when the request is submitted, and the latency clock
starts after the handshake, so offered arrivals reach the endpoint roughly
`connect_ms` after the reported dispatch lag. Worth one sentence in the
connection-setup line.
