# Follow-up work

## Decide what to do about the customer name in git history

The working tree is scrubbed. Earlier history is not, and it carries
customer-identifying and commercially sensitive material that the working
tree never should have held. Neither the specifics nor where to find them
are recorded here: this file is tracked and public, and a description
precise enough to be useful is also precise enough to be a map. The owner
knows what it is.

That history is on a public remote. Anyone running `git log -p`, opening a
commit permalink, or cloning the repo sees all of it, so the working-tree
scrub reduces exposure very little on its own.

This is a decision for the repo owner, not something to do quietly. Removing
it means `git filter-repo` plus a force push, plus a GitHub Support request to
purge cached blob views, plus checking the fork network, since forks keep the
old objects reachable after a rewrite.

Until that decision is made and carried out, do not describe this repo as
scrubbed.

## Record which serving config was read

`endpoint_meta._summarize` reads the active `config` only. During an endpoint
update the pending config carries the new workload shape. Recording a
`config_source` field and rendering it would let the card say which one it
described. Severity checked: with the `pending_config` fallback dropped, a
mid-update endpoint reports NO served-entity rows at all and shows
`ready: UPDATING`, so the card cannot make a false capacity claim. Cosmetic.

## Document the dispatch-lag and connect_ms interaction

Mostly answered by the wire-lateness work: `first_send_unix` is stamped
before the handshake, so wire lateness covers dispatcher-to-wire. The
residual is `connect_ms`, since arrival at the endpoint is roughly
wire lateness plus the handshake. Both numbers are printed and labeled, so
this is a wording question rather than a missing measurement.
