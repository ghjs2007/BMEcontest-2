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

## Task 3 macro parity record

The manifest now records one legal raw/session-cache/slide-cache pairing.  Its
raw parser arrays match the paired session cache, and the legacy
`scripts/slide_features.py:_process_session` output matches the selected slide
rows exactly after canonical window-row serialization.  The evidence is 278
rows by **62** `float32` columns.

The approved ABI resolves the two layers explicitly: the canonical raw macro
producer returns the historic 62-D matrix, and `add_time_prior` appends the
separate frozen one-column model adapter.  The manifest records hashes for
both matrices and schemas, as well as the adapter source.  Three-way parity
must remain exact: legacy 62-D equals canonical 62-D; the adapter equals the
legacy runner boundary; and adapted 63-D equals the frozen model-facing matrix.

Generation command (metadata only):

```text
python scripts/bootstrap_event_stack_diagnostics.py --run-key 160afaf81debf1ee
```
