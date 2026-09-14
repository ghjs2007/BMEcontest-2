# Task 3 Diagnostics and Attestation Report

## RED / GREEN evidence

- RED: `test_subject_diagnostics_zero_and_undefined_contract` failed with `ModuleNotFoundError: src.pipeline.diagnostics`.
- GREEN: the diagnostics contract test and paired-union test passed after the canonical implementation.
- RED: the peak-RSS aggregate test failed because `FoldResult` did not accept `runtime_diagnostics`.
- GREEN: the aggregate reports the maximum worker peak (307), not the sum.
- RED: the CLI sibling test failed with the expected missing diagnostics file.
- GREEN: the CLI writes the canonical `fold<k>_<hash>.diagnostics.json` payload.
- RED: the promotion tamper test exposed diagnostics as an unbound file.
- GREEN: changed fold diagnostics now fail promotion-attestation verification, and registered promotion rejects a sibling differing from cached fold evidence.
- GREEN: bootstrap input-fingerprint tampering is refused before the incumbent directory changes.

## Verification

`D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/pipeline/test_runner.py tests/pipeline/test_artifacts.py -q`

Result: `141 passed, 1 skipped`.

`D:/Anaconda3/envs/bme/python.exe scripts/bootstrap_event_stack_diagnostics.py --run-key 035644cf0889a5dd`

Result: five diagnostics files under `models/event_stack/035644cf0889a5dd/diagnostics/`; promotion attestation verification returned `()`.

## Incumbent gate

The replay gate retained exactly TP `109`, truths `153`, predictions `237`, and F1 `0.558974358974359` before writing diagnostics. It also reproduced the frozen v5 cache identity while handling the incumbent's legacy omission of the null Context-v1 field.

## Commit

`feat: attest subject-level experiment diagnostics` (final committed change)

## Concerns

- Windows RSS uses `GetProcessMemoryInfo`; non-Windows records an explicit unavailable reason and leaves CUDA/SSL runtime fields null.
- The permanent bootstrap is intentionally restricted to run key `035644cf0889a5dd`; already attested diagnostics are verified and returned without overwriting immutable evidence.
