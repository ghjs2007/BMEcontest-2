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

Generation command (metadata only):

```text
python scripts/bootstrap_event_stack_diagnostics.py --run-key 160afaf81debf1ee
```
