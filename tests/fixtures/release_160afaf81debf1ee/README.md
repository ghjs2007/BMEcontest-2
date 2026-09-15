# Release 160afaf81debf1ee parity fixtures

This directory contains only safe, derived fixture metadata. It does not
contain raw sensor data, labels, subject identifiers, or competition inputs.
The five hashes in `fixture_manifest.json` are anchors to existing promoted
evidence (diagnostics, schema, summary, and attestation); they are not copied
feature arrays or raw sessions.

Any future fixture must be generated from a legally distributable source and
recorded in `fixture_manifest.json` with its schema and SHA-256 hashes. Do not
copy raw data into this repository.

Generation command (metadata only):

```text
python scripts/bootstrap_event_stack_diagnostics.py --run-key 160afaf81debf1ee
```
