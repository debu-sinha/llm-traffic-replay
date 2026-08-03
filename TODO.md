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

## Record which serving config was read

`endpoint_meta._summarize` reads the active `config` only. During an endpoint
update the pending config carries the new workload shape. Recording a
`config_source` field and rendering it would let the card say which one it
described. Severity checked: with the `pending_config` fallback dropped, a
mid-update endpoint reports NO served-entity rows at all and shows
`ready: UPDATING`, so the card cannot make a false capacity claim. Cosmetic.

## Document the dispatch-lag and connect_ms interaction

Dispatch lag is stamped when the request is submitted, and the latency clock
starts after the handshake, so offered arrivals reach the endpoint roughly
`connect_ms` after the reported dispatch lag. Worth one sentence in the
connection-setup line. Severity checked: no wrong number is printed, both
figures are individually correct and labeled. Documentation only.
