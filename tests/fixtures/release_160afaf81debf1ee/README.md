# Release 160afaf81debf1ee parity fixtures

This directory contains only safe, derived fixture metadata. It does not
contain raw sensor data, labels, subject identifiers, or competition inputs.
The five hashes in `fixture_manifest.json` are explicitly **aggregate evidence
aliases** to existing promoted evidence (diagnostics, schema, summary, and
attestation). They are not direct raw-session span, feature, candidate,
admission, or event fixtures and must not be interpreted as such. Task 3 must
replace or supplement these aliases with direct real-session layer hashes
before canonical feature extraction is accepted.

Any future fixture must be generated from a legally distributable source and
recorded in `fixture_manifest.json` with its schema and SHA-256 hashes. Do not
copy raw data into this repository.

## Task 3 stop-gate record

The manifest now records one legal raw/session-cache/slide-cache pairing.  Its
raw parser arrays match the paired session cache, and the legacy
`scripts/slide_features.py:_process_session` output matches the selected slide
rows exactly after canonical window-row serialization.  The evidence is 278
rows by **62** `float32` columns.

Task 3 requires a canonical macro producer with a 63-column contract.  The
extra promoted column is the frozen time prior added later by
`runner._with_time_prior`, not an output of `slide_features.py`.  Therefore the
direct golden evidence cannot prove a 63-D extraction without changing the
layer boundary.  Per the delivery plan's stop gate, no macro code has been
extracted or changed.  A later explicitly approved plan must first define and
test the 62-D raw feature batch plus the separate frozen 1-D time-prior model
adapter, or revise the producer ABI without changing the release.

Generation command (metadata only):

```text
python scripts/bootstrap_event_stack_diagnostics.py --run-key 160afaf81debf1ee
```
