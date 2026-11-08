# Retained synthetic timing corpus

`client/requests.json` contains the actual synthetic requests used in the final
confirmation, unchanged (SHA256
`907cc125ed2988868cfc0d7e9e88c589624c09f7e40e0ebbcd1b632b2da09a45`).
The public client's endpoint allowlist replaces the private site's LAN address
with localhost. Request bytes, timing arithmetic and parser logic are unchanged.
The manifest records the public client hashes; `qualification-manifest.json`
retains the original input manifest for provenance.

The client does not mutate model, registry, transport, fabric or service state.
An actual run makes up to 258 inference requests; `--dry-run` sends none.
Use a private backend/localhost forward and an isolated admission window.
See [the method](../docs/validation.md) for all wave counts and limits.

Public `results/confirmation.json` retains all wave summaries. It deliberately
omits private host inventories and full response/event payloads. Timing figures
can be independently reduced; exact-output outcomes are recorded receipts,
not a published complete token/logit trace.
