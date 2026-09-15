# Release 160afaf81debf1ee parity fixtures

This directory contains only safe, synthetic or derived fixture metadata. It
does not contain raw sensor data, labels, subject identifiers, or competition
inputs. The promoted release evidence remains under `models/event_stack/` and
is validated by its attestation.

Any future fixture must be generated from a legally distributable source and
recorded in `fixture_manifest.json` with its schema and SHA-256 hashes. Do not
copy raw data into this repository.

Generation command (metadata only):

```text
python scripts/bootstrap_event_stack_diagnostics.py --run-key 160afaf81debf1ee
```
