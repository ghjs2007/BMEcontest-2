# Context-v1 Implementation Plan

> **Historical execution record:** this plan was written while `035644cf0889a5dd` was the incumbent.
> It was superseded by the promoted Context-v1 release `160afaf81debf1ee`; the retired model directory
> was deliberately removed under the release-retention policy. References below describe that historical
> bootstrap workflow and are not current reproduction commands.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the one-shot deterministic 60-column Context-v1 verifier block, attested per-subject diagnostics, and an incumbent-aware release gate, then run one strict five-fold experiment against F1 `0.5589743590`.

**Architecture:** A pure `context_features` module computes an ordered, versioned block from same-session macro/micro probability streams. `RunConfig` gates the block so the legacy 56-column path remains unchanged; runner cache keys, model widths, artifact schemas and diagnostics change together. Promotion compares candidates to a tracked incumbent before packaging can replace `dist/`.

**Tech Stack:** Python 3.11, NumPy, scikit-learn pipelines, LightGBM, pytest, JSON/SHA-256 artifact manifests.

**Spec:** `docs/superpowers/specs/2026-09-14-context-ssl-sequence-encoder-design.md`

## Global Constraints

- Context-v1 uses exactly 60 ordered columns and produces a 116-column verifier when enabled.
- Fixed thresholds are macro `0.28838`, micro `0.20`; context is 20 minutes; run gap is 60 seconds; minimum run length is two windows.
- Features never cross `sid`, never aggregate by subject, and never use labels or selected policies.
- Empty statistics are `NaN`; coverage and neighbor count/total duration are zero; `Inf` is rejected.
- The disabled branch must remain bit-for-bit compatible with the current 56-column output.
- All selection remains four-way subject-disjoint inner OOF; outer validation is scored once.
- Technical promotion requires F1 strictly above the incumbent `0.5589743590`; `delta F1 <0.005` is reported as a scientific tie.
- Failed experiment outputs and run-specific caches are removed; protected data and current caches are retained.

---

### Task 1: Pure Context-v1 Feature Contract

**Files:**
- Create: `src/pipeline/context_features.py`
- Create: `tests/pipeline/test_context_features.py`

**Interfaces:**
- Consumes: `EventRef`, `Mapping[str, Sequence[tuple[int, int, float]]]` macro/micro streams.
- Produces: `CONTEXT_V1_COLUMNS: tuple[str, ...]`, `CONTEXT_V1_SCHEMA_HASH: str`, and `context_v1_features(candidates, macro_windows_by_sid, micro_windows_by_sid, session_bounds_by_sid) -> np.ndarray` with shape `(n, 60)`.

- [ ] **Step 1: Write failing tests for the ordered schema and region statistics**

```python
def test_context_v1_schema_is_fixed_and_hashed():
    assert len(CONTEXT_V1_COLUMNS) == 60
    assert CONTEXT_V1_COLUMNS[:4] == (
        "macro_pre_mean", "macro_pre_max", "macro_pre_std",
        "macro_pre_above_fraction",
    )
    assert CONTEXT_V1_COLUMNS[30] == "micro_pre_mean"
    expected = hashlib.sha256(
        json.dumps(CONTEXT_V1_COLUMNS, separators=(",", ":")).encode()
    ).hexdigest()
    assert CONTEXT_V1_SCHEMA_HASH == expected

def test_context_v1_uses_same_session_clipped_real_time_regions():
    candidate = EventRef("s1", 1_200_000, 1_800_000)
    macro = {
        "s1": [(0, 30_000, .1), (1_300_000, 1_330_000, .8), (1_900_000, 1_930_000, .2)],
        "s2": [(1_300_000, 1_330_000, 1.0)],
    }
    row = context_v1_features([candidate], macro, {}, {"s1": (0, 3_000_000)})[0]
    assert row[CONTEXT_V1_COLUMNS.index("macro_candidate_mean")] == pytest.approx(.8)
    assert row[CONTEXT_V1_COLUMNS.index("macro_post_mean")] == pytest.approx(.2)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -p no:cacheprovider tests/pipeline/test_context_features.py -q`  
Expected: collection fails because `src.pipeline.context_features` does not exist.

- [ ] **Step 3: Implement the constants, region selection and 30-column per-scale builder**

```python
CONTEXT_V1_CONTEXT_MS = 1_200_000
CONTEXT_V1_RUN_GAP_MS = 60_000
CONTEXT_V1_MIN_RUN_WINDOWS = 2
MACRO_STRIDE_MS, MACRO_WINDOW_MS = 15_000, 240_000
MICRO_STRIDE_MS, MICRO_WINDOW_MS = 7_500, 15_000
_SCALES = (
    ("macro", 0.28838, MACRO_STRIDE_MS, MACRO_WINDOW_MS),
    ("micro", 0.20, MICRO_STRIDE_MS, MICRO_WINDOW_MS),
)

def context_v1_features(candidates, macro_windows_by_sid, micro_windows_by_sid, session_bounds_by_sid):
    rows = []
    for candidate in candidates:
        event = getattr(candidate, "event", candidate)
        blocks = [
            _scale_context(
                event, streams.get(event.sid, ()), session_bounds_by_sid[event.sid],
                threshold, stride_ms, window_ms,
            )
            for _, threshold, stride_ms, window_ms, streams in (
                (*_SCALES[0], macro_windows_by_sid),
                (*_SCALES[1], micro_windows_by_sid),
            )
        ]
        rows.append(np.concatenate(blocks))
    result = np.asarray(rows, dtype=np.float64).reshape((-1, 60))
    if np.isinf(result).any():
        raise ValueError("Context-v1 features must not contain infinity")
    return result
```

- [ ] **Step 4: Add failing edge-case tests, then implement each case**

Tests must assert: empty-scale NaN/zero contract; session-start/end clipping; a score gap does not redefine
the session boundary; fixed above fractions; two-window/60-second run joining; own-run exclusion;
no-neighbor NaNs; center-half mass; first-minus-second mean; output stability under input ordering; named
window/stride constants; and rejection of `Inf`. Run the test after adding each assertion and confirm it
fails for the missing behavior before implementing `_regions`, `_summaries`, `_candidate_like_runs` and
`_scale_context`.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest -p no:cacheprovider tests/pipeline/test_context_features.py -q`  
Expected: PASS.

```bash
git add src/pipeline/context_features.py tests/pipeline/test_context_features.py
git commit -m "feat: add deterministic context v1 features"
```

---

### Task 2: Runner, Cache-Key and Schema Integration

**Files:**
- Modify: `src/pipeline/event_stack.py:970`
- Modify: `src/pipeline/runner.py:157` (RunConfig, feature construction, dimensions and serialization)
- Modify: `scripts/crossfit_event_stack.py` (CLI flag)
- Modify: `scripts/promote_event_stack.py` (dynamic verifier width/schema)
- Modify: `src/pipeline/artifacts.py` (versioned schema object and compatibility loader)
- Modify: `scripts/package_event_stack.py` and `scripts/predict_event_stack.py` (versioned runtime schema)
- Modify: `tests/pipeline/test_event_stack.py`
- Modify: `tests/pipeline/test_runner.py`
- Modify: `tests/pipeline/test_artifacts.py`
- Modify: `tests/pipeline/test_multiscale_runner.py`
- Modify: `tests/pipeline/test_event_stack_dist.py`

**Interfaces:**
- Consumes: Task 1 `context_v1_features` and schema constants.
- Produces: `RunConfig.context_features_version: str | None`, `multiscale_verifier_features(..., context_features_version=None, session_bounds_by_sid=None)`, `expected_feature_dimensions(config)`, and a versioned schema `{schema_version, widths, context}`.

- [ ] **Step 1: Write failing legacy-parity and enabled-width tests**

```python
def test_multiscale_context_disabled_is_identical_to_legacy_fixture():
    actual = multiscale_verifier_features(candidates, macro_windows, micro_windows)
    assert np.array_equal(actual, expected_legacy_56)

def test_multiscale_context_v1_appends_ordered_block_and_preserves_nan():
    actual = multiscale_verifier_features(
        candidates, macro_windows, micro_windows, context_features_version="v1"
    )
    assert actual.shape == (len(candidates), 116)
    assert np.isnan(actual[:, 56:]).any()
    assert not np.isinf(actual).any()
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -p no:cacheprovider tests/pipeline/test_event_stack.py -q`  
Expected: FAIL because the new keyword is unsupported.

- [ ] **Step 3: Implement the opt-in append without touching the legacy path**

```python
def multiscale_verifier_features(..., context_features_version: str | None = None):
    legacy = _legacy_multiscale_verifier_features(...)
    if context_features_version is None:
        return legacy
    if context_features_version != "v1":
        raise ValueError("context_features_version must be None or 'v1'")
    context = context_v1_features(candidates, macro_windows_by_sid, micro_windows_by_sid)
    return np.concatenate((legacy, context), axis=1)
```

- [ ] **Step 4: Write failing configuration/cache/schema tests**

```python
def test_context_version_changes_experiment_and_fold_cache_keys():
    base = RunConfig(outer_fold=0, micro_enabled=True, candidate_control_enabled=True)
    context = replace(base, context_features_version="v1")
    assert experiment_key([replace(base, outer_fold=i) for i in range(5)]) != experiment_key(
        [replace(context, outer_fold=i) for i in range(5)]
    )
    assert cache_key(base, (63, 56, 47), files) != cache_key(context, (63, 116, 47), files)

def test_promoted_context_schema_records_columns_and_hash():
    schema = _validated_feature_schema(context_fit, context_features_version="v1")
    assert schema["schema_version"] == 2
    assert schema["widths"]["verifier"] == 116
    assert schema["context"]["version"] == "v1"
    assert schema["context"]["columns"] == list(CONTEXT_V1_COLUMNS)
    assert schema["context"]["schema_hash"] == CONTEXT_V1_SCHEMA_HASH
```

- [ ] **Step 5: Implement configuration plumbing and CLI validation**

Add `context_features_version: str | None = None` to `RunConfig`; accept only `None`/`"v1"`; pass it
and real session bounds to every inner, outer and full-target call; derive all cache, fitted-model and
promotion widths through one `expected_feature_dimensions(config)` helper; add `--context-features {v1}`;
include column metadata in bundle schemas. Add `--summary-alias PATH`, which atomically writes a canonical
copy of the aggregate summary after all requested folds succeed and refuses paths outside
`outputs/crossfit/`. Replace the registered key and fixed `(63,56,47)` checks in
`_REGISTERED_EXPERIMENT_KEY`, `_validate_current_cache_bindings` and `_validated_feature_schema` with
registered-summary identity plus that helper. Evidence refit, full-target fit, manifests and packaging
must consume the same schema object. Do not add a tunable context grid.

Upgrade `EventStackBundle.feature_schema` from `Mapping[str, int]` to a versioned JSON object. Schema v1
loads existing `{"macro":63,"micro":47,"verifier":56}` bundles unchanged; schema v2 requires
`{"schema_version":2,"widths":...,"context":...}` and validates ordered columns/hash. Add repository,
package, dist-runtime and old-bundle round-trip tests; unknown versions fail explicitly.

- [ ] **Step 6: Run integration tests and commit**

Run: `python -m pytest -p no:cacheprovider tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py tests/pipeline/test_multiscale_runner.py tests/pipeline/test_artifacts.py tests/pipeline/test_event_stack_dist.py -q`  
Expected: PASS.

```bash
git add src/pipeline/event_stack.py src/pipeline/runner.py src/pipeline/artifacts.py scripts/crossfit_event_stack.py scripts/promote_event_stack.py scripts/package_event_stack.py scripts/predict_event_stack.py tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py tests/pipeline/test_multiscale_runner.py tests/pipeline/test_artifacts.py tests/pipeline/test_event_stack_dist.py
git commit -m "feat: integrate context v1 into event stack"
```

---

### Task 3: Per-Subject Diagnostics and Attestation

**Files:**
- Create: `src/pipeline/diagnostics.py`
- Create: `scripts/bootstrap_event_stack_diagnostics.py`
- Modify: `src/pipeline/runner.py` (FoldResult and aggregate output)
- Modify: `src/pipeline/artifacts.py` (diagnostic hashes)
- Modify: `tests/pipeline/test_runner.py`
- Modify: `tests/pipeline/test_artifacts.py`

**Interfaces:**
- Produces: `subject_diagnostics(predictions, truths, subjects) -> dict`, one canonical `diagnostics.json` per fold, immutable incumbent baseline diagnostics, and aggregate paired distribution fields bound by promotion attestation.

- [ ] **Step 1: Write failing subject-metric tests**

```python
def test_subject_diagnostics_zero_and_undefined_contract():
    result = subject_diagnostics(predictions, truths, sid_to_subject)
    assert result["subjects"]["with_truth_no_match"]["f1"] == 0.0
    assert result["subjects"]["negative_with_fp"]["f1"] == 0.0
    assert result["subjects"]["empty"]["f1"] is None
    assert result["distribution"]["excluded_empty"] == 1

def test_paired_subject_counts_use_union_without_changing_promotion_score():
    paired = paired_subject_diagnostics(incumbent, candidate)
    assert paired["improved"] + paired["unchanged"] + paired["worsened"] == 3
```

- [ ] **Step 2: Verify RED, then implement canonical diagnostics**

Run the two named tests; expect import failure. Implement session-to-subject grouping, official event metrics per subject, linear-interpolated percentile calculation, paired counts and canonical JSON serialization. Runtime fields not used by Stage A are explicit `null`, never fabricated zeros.

- [ ] **Step 3: Write failing attestation-tamper test, then bind hashes**

```python
def test_promotion_attestation_rejects_modified_fold_diagnostics(tmp_path):
    root = promoted_fixture(tmp_path, diagnostics=True)
    path = root / "outer-fold-0" / "diagnostics.json"
    path.write_text("{}", encoding="utf-8")
    assert "diagnostics" in " ".join(verify_promotion_attestation(root))
```

`FoldResult` carries canonical subject rows and runtime diagnostics; `fold_result_to_dict` stores them in
the content-addressed fold JSON so cache hits restore exactly the same evidence. The CLI writes a sibling
`fold<k>_<hash>.diagnostics.json` derived from that result. Promotion verifies equality with the cached
record, copies the same canonical file into `outer-fold-k/diagnostics.json`, includes it in the bundle
manifest, and binds all five hashes in the run attestation. Summary aggregation recomputes distribution
values from the five diagnostic files rather than trusting self-reported summary fields.

Implement `peak_working_set_bytes()` on Windows with `ctypes` and `GetProcessMemoryInfo`'s
`PeakWorkingSetSize`; return `None` only when the platform API is unavailable and store an explicit
`"unavailable_reason"`. Each fold worker records its own peak at completion. Aggregate peak RSS is the
maximum worker peak, never the sum. Add a dependency-injected test provider that returns known fold
peaks and assert aggregate `max` behavior; Stage A may leave CUDA/SSL-specific fields null but not RSS.

- [ ] **Step 4: Bootstrap immutable diagnostics for the current incumbent**

Add a permanent command that loads the verified incumbent summary and its five evidence records, replays
each frozen outer fold through the same runner without changing any selection, and writes canonical
per-subject diagnostics under the then-incumbent model diagnostics directory. It must assert the replayed
aggregate remains exactly TP `109`, true `153`, predictions `237`, F1 `0.558974358974359`; otherwise it
refuses all writes. Rebuild the incumbent attestation so it binds the five diagnostic hashes. Add a test
that a changed replay metric or input fingerprint leaves the incumbent directory untouched.

Run:

```bash
python scripts/bootstrap_event_stack_diagnostics.py --run-key 035644cf0889a5dd
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -p no:cacheprovider tests/pipeline/test_runner.py tests/pipeline/test_artifacts.py -q`  
Expected: PASS.

```bash
git add src/pipeline/diagnostics.py src/pipeline/runner.py src/pipeline/artifacts.py scripts/bootstrap_event_stack_diagnostics.py models/event_stack/<then-incumbent> tests/pipeline/test_runner.py tests/pipeline/test_artifacts.py
git commit -m "feat: attest subject-level experiment diagnostics"
```

---

### Task 4: Incumbent-Aware Atomic Release Gate

**Files:**
- Create: `release/event_stack_incumbent.json`
- Create: `scripts/release_event_stack.py`
- Modify: `src/pipeline/artifacts.py`
- Modify: `scripts/promote_event_stack.py`
- Modify: `scripts/package_event_stack.py`
- Modify: `tests/pipeline/test_artifacts.py`
- Modify: `tests/pipeline/test_event_stack_dist.py`

**Interfaces:**
- Produces: `load_incumbent_registry(path)`, candidate-versus-incumbent validation, a durable release journal, crash recovery, and one release orchestrator.

- [ ] **Step 1: Write failing registry and regression tests**

```python
def test_incumbent_registry_is_bound_to_current_release():
    registry = load_incumbent_registry(Path("release/event_stack_incumbent.json"))
    assert registry["run_key"] == "035644cf0889a5dd"
    assert registry["f1"] == pytest.approx(0.558974358974359)

def test_candidate_below_incumbent_cannot_replace_dist(tmp_path):
    dist_before = tree_hash(tmp_path / "dist")
    with pytest.raises(PromotionContractError, match="incumbent"):
        release_candidate(candidate_f1=.55, incumbent_f1=.56)
    assert tree_hash(tmp_path / "dist") == dist_before
```

- [ ] **Step 2: Verify RED and add the canonical registry**

Run the named tests; expect missing registry APIs. Create canonical JSON containing schema version 1,
current run key/F1, summary SHA-256, attestation SHA-256 and incumbent diagnostic-set SHA-256 derived from
verified files—not manually copied hashes.

- [ ] **Step 3: Implement comparison, temporary packaging and rollback**

Candidate validation uses `candidate_f1 > incumbent_f1`. `scripts/release_event_stack.py` is the only
command allowed to update the active release. It builds/verifies the model bundle and temporary dist,
then writes `release/.event-stack-transaction.json` containing candidate/previous registry bytes, staged
dist hash, backup paths and phase. Each `os.replace` transition fsyncs the journal first. Phases are
`prepared`, `dist_replaced`, `registry_replaced`, `verified`; startup resumes or rolls back any non-verified
phase based on hashes. Direct package/promote entrypoints refuse active-release destinations unless given
the orchestrator's one-use transaction token. Keep `PROMOTION_F1_FLOOR` only for old artifact-format
validation, never release replacement.

- [ ] **Step 4: Add failure-injection tests and commit**

Cover invalid registry/diagnostic hash, candidate equality, package failure, process interruption after
each journal phase, registry write failure and post-copy manifest failure. Invoke recovery in a new
process for every interruption case and assert it reaches either the completely old or completely new
verified release, never a mixed pair.

Run: `python -m pytest -p no:cacheprovider tests/pipeline/test_artifacts.py tests/pipeline/test_event_stack_dist.py -q`  
Expected: PASS.

```bash
git add release/event_stack_incumbent.json src/pipeline/artifacts.py scripts/promote_event_stack.py scripts/package_event_stack.py scripts/release_event_stack.py tests/pipeline/test_artifacts.py tests/pipeline/test_event_stack_dist.py
git commit -m "fix: gate releases against attested incumbent"
```

---

### Task 5: Smoke, Locked Five-Fold Experiment and Release Decision

**Files:**
- Modify after measured result: `README.md`
- Modify after measured result: `docs/三阶段重构设计.md`
- Modify only after promotion: `dist/README.md`, `dist/event_stack/**`, `models/event_stack/**`, `outputs/crossfit/**`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: one registered Context-v1 run and either a promoted release or a cleaned failed-experiment record.

- [ ] **Step 1: Run focused and complete verification before training**

```bash
python -m pytest -p no:cacheprovider tests/pipeline/test_context_features.py tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py tests/pipeline/test_artifacts.py tests/pipeline/test_event_stack_dist.py -q
python -m pytest -p no:cacheprovider -q
```

Set `PYTHONDONTWRITEBYTECODE=1` and `LOKY_MAX_CPU_COUNT=16`. Expected: all tests pass, with only documented Windows symlink skips.

- [ ] **Step 2: Run one outer-fold smoke**

```bash
python scripts/crossfit_event_stack.py --fold 0 --inner-splits 4 --no-tcn --workers 1 --micro-enabled --candidate-control-enabled --admission-minimum-recall 0.80 --context-features v1
```

Verify the config hash is new, verifier width is 116, subject isolation assertions pass, diagnostics exist, and no `Inf` appears. This smoke is operational only; do not tune Context-v1 from its F1.

- [ ] **Step 3: Run the single registered five-fold experiment**

```bash
python scripts/crossfit_event_stack.py --fold all --inner-splits 4 --no-tcn --workers 5 --micro-enabled --candidate-control-enabled --admission-minimum-recall 0.80 --context-features v1 --summary-alias outputs/crossfit/context_v1_summary.json
```

Record aggregate F1/TP/pred/FP/PPV/recall, short-meal and candidate recall, fold scores, per-subject median/IQR/p10 and paired improved/unchanged/worsened counts, wall time and peak RSS.

- [ ] **Step 4: Apply the pre-registered decision**

If F1 is not strictly greater than the incumbent, delete only the exact Context-v1 run-key outputs and caches, add `Context-v1 failed` plus metrics to README, run tests, and commit documentation. Do not alter thresholds/features and do not package models.

If F1 is greater, run the only incumbent-aware release entrypoint:

```bash
python scripts/release_event_stack.py --summary outputs/crossfit/context_v1_summary.json
```

The command verifies the alias content and embedded experiment key against the content-addressed fold
evidence, then promotes, packages, commits the journal transaction and verifies 5+1 manifests,
attestation, diagnostics, repository/dist parity, CPU/auto behavior and CUDA rejection. Remove the
superseded tracked release only after the new dist and incumbent registry verify.

- [ ] **Step 5: Update documentation, clean, verify and commit**

Update exact architecture widths, run key, metrics, device status, limitations and reproduction commands. Remove run-specific temporary files, bytecode and failed caches; retain protected caches. Run the full suite, `git diff --check`, bundle verification, `git count-objects -vH` and `git status --short`.

```bash
git add README.md docs/三阶段重构设计.md dist models/event_stack outputs/crossfit
git commit -m "feat: evaluate context v1 event verifier"
```
