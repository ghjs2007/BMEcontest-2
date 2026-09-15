# Competition Delivery Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Freeze release 160afaf81debf1ee and deliver canonical raw-session inference, standalone inference/submission bundles, and a visualization contract without changing any promoted decision.

**Architecture:** Extract raw loading, timeline/window assembly and features into canonical src/pipeline functions used by both training and Predictor. Distribution packages are generated copies of canonical runtime source and verified artifacts; no manually maintained dist feature code is allowed. Parity gates must pass before cleanup.

**Tech Stack:** Python 3.11.15, NumPy 2.4.6, SciPy, joblib 1.5.3, scikit-learn 1.9.0, LightGBM 4.7.0, JSON Schema, pytest.

**Spec:** docs/BMEcontest-2 竞赛交付与仓库收口重构任务书.md; audit: docs/repository_cleanup_audit.md.

## Global Constraints

- Current release is immutable: run 160afaf81debf1ee, F1 0.6514285714285715, TP 114, FP 83, predictions 197, truths 153.
- Preserve five fold records, schema v2 macro 63, micro 47, verifier 116, Context-v1 hash, model hashes, fingerprints, imputation, candidate generation, admission, policy, thresholds, matching and decoder.
- No parameter selection, outer-fold experimentation, algorithm modification, feature change or official-format guessing.
- Raw loading retains timestamp/session/gap/coverage/gravity/NaN/Inf behavior and never crosses a gap or session.
- src/ is the only manually maintained algorithm source. dist/ is generated; it must run without parent repo, cache, outputs, split, training data or research scripts.
- Cleanup only after four-way parity plus clean-room tests, manifest and attestation verification. Never delete current model/evidence or manifest-fingerprinted cache.
- Run tests with PYTHONDONTWRITEBYTECODE=1 and -p no:cacheprovider.

## Planned file ownership

| Path | Responsibility |
|---|---|
| src/pipeline/io/raw_session.py | Raw file/folder discovery and side-effect-free SessionData loading. |
| src/pipeline/preprocessing/timeline.py | Time validation, spans, gaps and eligible windows. |
| src/pipeline/features/{macro,micro}.py | Canonical 63-D/47-D feature window production. |
| src/pipeline/inference/{predictor,schema,legacy_payload,competition_adapter}.py | Predictor API, public JSON, legacy compatibility, isolated adapter. |
| scripts/{build_inference_distribution,build_submission}.py | Deterministic distribution builders. |
| dist/{inference,visual,submission,examples,schema}/ | Generated inference, frontend contract, generated submission, safe example, schema. |

---

### Task 1: Lock the promoted baseline and legal parity fixtures

**Files:**
- Create: tests/fixtures/release_160afaf81debf1ee/README.md
- Create: tests/fixtures/release_160afaf81debf1ee/fixture_manifest.json
- Create: tests/parity/test_promoted_release_baseline.py
- Modify: src/pipeline/artifacts.py

**Interfaces:**
- Produces load_current_promoted_release(root: Path) -> Mapping[str, object] and verify_current_promoted_release(root: Path) -> tuple[str, ...].
- Fixture manifest stores safe fixture path names, schema, and canonical SHA-256 hashes for spans/features/candidates/admission/events; it stores no raw data.

- [ ] **Step 1: Write failing release-lock tests.**

    def test_current_promoted_release_is_attested(repo_root: Path):
        release = load_current_promoted_release(repo_root)
        assert release["run_key"] == "160afaf81debf1ee"
        assert release["aggregate"]["outer_metrics"]["n_tp"] == 114
        assert release["aggregate"]["outer_metrics"]["n_pred"] == 197
        assert release["aggregate"]["outer_metrics"]["n_true"] == 153
        assert release["aggregate"]["outer_metrics"]["f1"] == pytest.approx(.6514285714285715)
        assert verify_current_promoted_release(repo_root) == ()

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/parity/test_promoted_release_baseline.py -q

Expected: FAIL because release-lock APIs do not exist.

- [ ] **Step 3: Implement verification by reusing incumbent registry and attestation.**

    def load_current_promoted_release(root: Path) -> Mapping[str, object]:
        registry = load_incumbent_registry(root / "release" / "event_stack_incumbent.json")
        run_root = root / "models" / "event_stack" / str(registry["run_key"])
        problems = verify_promotion_attestation(run_root)
        if problems:
            raise PromotionContractError("; ".join(problems))
        return {"run_key": registry["run_key"],
                "aggregate": json.loads((run_root / "promotion_summary.json").read_text("utf-8"))}

Generate fixture hashes only from existing legal fixtures. Record source names and exact generation command in fixture README.

- [ ] **Step 4: Verify and commit.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/parity/test_promoted_release_baseline.py tests/pipeline/test_artifacts.py tests/pipeline/test_release_event_stack.py -q

    git add src/pipeline/artifacts.py tests/fixtures/release_160afaf81debf1ee tests/parity/test_promoted_release_baseline.py
    git commit -m "test: lock promoted release refactor baseline"

### Task 2: Extract raw IO and timeline primitives without changing training behavior

**Files:**
- Create: src/pipeline/io/__init__.py, src/pipeline/io/raw_session.py
- Create: src/pipeline/preprocessing/__init__.py, src/pipeline/preprocessing/timeline.py
- Modify: src/data/loader.py, src/pipeline/micro_cache.py
- Test: tests/unit/test_raw_session.py, tests/unit/test_timeline.py, tests/parity/test_raw_loader_parity.py

**Interfaces:**
- Produces RawSessionSource(path: Path, session_id: str, subject_id: str | None), discover_raw_sessions(path: Path) -> tuple[RawSessionSource, ...], load_raw_session(source) -> SessionData, valid_imu_spans(session) -> tuple[TimelineSpan, ...], window_starts(span, window_ms, stride_ms, coverage_min) -> np.ndarray.
- Legacy load_session_tsv becomes a compatibility wrapper; only legacy load_session retains cache write side effects.

- [ ] **Step 1: Write failing discovery/gap tests.**

    def test_discover_raw_session_folder(tmp_path):
        raw = write_collect_data(tmp_path / "S01" / "collect_data1_2_3.txt")
        assert discover_raw_sessions(raw.parent) == (RawSessionSource(raw, "S01", None),)

    def test_gap_creates_separate_spans():
        session = session_with_valid_timestamps([0, 50, 100, 10_000, 10_050])
        assert [(x.start_ms, x.end_ms) for x in valid_imu_spans(session)] == [(0, 100), (10_000, 10_050)]

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/unit/test_raw_session.py tests/unit/test_timeline.py -q

- [ ] **Step 3: Move parser rules verbatim and implement timeline helpers.**

    def load_raw_session(source: RawSessionSource) -> SessionData:
        if source.path.is_dir():
            source = discover_raw_sessions(source.path)[0]
        if source.path.suffix.lower() != ".txt":
            raise ValueError("supported raw session input is collect_data*.txt")
        return _parse_collect_data_tsv(source.path)

    def valid_imu_spans(session: SessionData) -> tuple[TimelineSpan, ...]:
        # Extract the existing cache/window discontinuity rule exactly; reject
        # unordered/nonfinite valid timestamps rather than repair them.
        return tuple(_split_valid_imu_rows(session))

Do not create a second parser. Preserve current 53-column handling, invalid masks, row-rate inference and decoding replacement.

- [ ] **Step 4: Add exact legacy parity before changing callers.**

    def test_legacy_and_canonical_reader_are_equal(raw_file):
        old = load_session_tsv(raw_file)
        new = load_raw_session(RawSessionSource(raw_file, "fixture", None))
        for name in ("acc", "gyro", "ppg", "t_acc", "t_ppg", "imu_valid", "ppg_valid"):
            assert np.array_equal(getattr(old, name), getattr(new, name))
        assert old.meta["row_rate"] == new.meta["row_rate"]

Also compare micro-cache fixture window IDs/bounds/labels/features exactly, excluding only recorded extraction seconds.

- [ ] **Step 5: Verify and commit.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/unit/test_raw_session.py tests/unit/test_timeline.py tests/parity/test_raw_loader_parity.py tests/pipeline/test_micro_cache.py -q

    git add src/pipeline/io src/pipeline/preprocessing src/data/loader.py src/pipeline/micro_cache.py tests/unit/test_raw_session.py tests/unit/test_timeline.py tests/parity/test_raw_loader_parity.py
    git commit -m "refactor: extract canonical raw session timeline"

### Task 3: Canonicalize macro/micro feature window producers

**Files:**
- Create: src/pipeline/features/__init__.py, src/pipeline/features/macro.py, src/pipeline/features/micro.py
- Modify: src/pipeline/runner.py, src/pipeline/imu_features.py, src/pipeline/micro_cache.py
- Test: tests/unit/test_macro_features.py, tests/parity/test_feature_producer_parity.py

**Interfaces:**
- Produces extract_macro_windows(session, *, session_id, config) -> MacroWindowBatch and extract_micro_windows(session, *, session_id, config: MicroFeatureConfig) -> MicroWindowBatch.
- Exact contracts: macro (n,63); micro (m,47); current numerical primitive remains extract_micro_features.

- [ ] **Step 1: Write failing dimension/no-cross-gap tests.**

    def test_macro_is_63d_and_never_crosses_gap():
        batch = extract_macro_windows(gapped_session, session_id="s1", config=MACRO_CONFIG)
        assert batch.features.shape[1] == 63
        assert all(w.end_ms <= 1_000 or w.start_ms >= 10_000 for w in batch.windows)

    def test_micro_delegates_to_frozen_47d_primitive(monkeypatch):
        monkeypatch.setattr(features_micro, "extract_micro_features", lambda a, g, hz: np.zeros(47, np.float32))
        assert extract_micro_windows(session, session_id="s1", config=MicroFeatureConfig()).features.shape[1] == 47

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/unit/test_macro_features.py tests/parity/test_feature_producer_parity.py -q

- [ ] **Step 3: Extract existing mathematics, then redirect cache/training callers.**

Port macro math from actual current producer verbatim; do not infer it from width. Keep WindowBatch ABI and adapt only at module boundary.

    def extract_micro_windows(session, *, session_id, config):
        rows, windows = [], []
        for span in valid_imu_spans(session):
            for start_ms in window_starts(span, config.window_ms, config.stride_ms, config.coverage_min):
                acc, gyro, hz = span.samples(start_ms, start_ms + config.window_ms)
                rows.append(extract_micro_features(acc, gyro, hz))
                windows.append(EventRef(session_id, int(start_ms), int(start_ms + config.window_ms)))
        return MicroWindowBatch(np.asarray(rows, dtype=np.float32).reshape((-1, 47)), tuple(windows))

- [ ] **Step 4: Write exact cache-feature parity test.**

    def test_canonical_features_match_promoted_fixture():
        legacy_macro, legacy_micro = load_promoted_feature_fixture()
        macro, micro = build_features_from_same_raw_fixture()
        assert_event_rows_equal(legacy_macro.windows, macro.windows)
        np.testing.assert_allclose(legacy_macro.features, macro.features, rtol=0, atol=0, equal_nan=True)
        assert_event_rows_equal(legacy_micro.windows, micro.windows)
        np.testing.assert_allclose(legacy_micro.features, micro.features, rtol=0, atol=0, equal_nan=True)

If legal raw source for macro cache cannot be reproduced, stop and document the missing fixture; do not approximate.

- [ ] **Step 5: Verify and commit.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/unit/test_macro_features.py tests/pipeline/test_imu_features.py tests/pipeline/test_micro_cache.py tests/parity/test_feature_producer_parity.py -q

    git add src/pipeline/features src/pipeline/runner.py src/pipeline/imu_features.py src/pipeline/micro_cache.py tests/unit/test_macro_features.py tests/parity/test_feature_producer_parity.py
    git commit -m "refactor: canonicalize macro and micro feature producers"

### Task 4: Add raw Predictor API and prediction schema

**Files:**
- Create: src/pipeline/inference/__init__.py, schema.py, predictor.py, legacy_payload.py
- Create: dist/schema/prediction.schema.json
- Modify: scripts/predict_event_stack.py
- Test: tests/unit/test_prediction_schema.py, tests/integration/test_predictor_raw.py

**Interfaces:**

    @dataclass(frozen=True)
    class PredictionOptions:
        include_timeline: bool = False
        include_candidates: bool = False
        device: Literal["auto", "cpu", "gpu", "cuda"] = "auto"

    class Predictor:
        @classmethod
        def from_bundle(cls, path: Path, *, device: str = "auto") -> "Predictor":
            return cls(_load_verified_bundle(path), device=device)
        def predict_file(self, path: Path, *, subject_id: str | None = None,
                         options: PredictionOptions = PredictionOptions()) -> dict[str, object]:
            return self.predict_sources((RawSessionSource(path, path.stem, subject_id),), options)
        def predict_folder(self, path: Path, *, subject_id: str | None = None,
                           options: PredictionOptions = PredictionOptions()) -> dict[str, object]:
            return self.predict_sources(discover_raw_sessions(path), options)

- [ ] **Step 1: Write failing public result tests.**

    def test_prediction_schema_minimum_result_is_valid():
        result = make_prediction_result(run_key="160afaf81debf1ee", source="fixture.txt", duration_seconds=12.5, events=[])
        validate_prediction(result)
        assert set(result) == {"schema_version", "model", "input", "events", "diagnostics"}

    def test_predictor_raw_file_needs_no_feature_payload(raw_file, bundle):
        result = Predictor.from_bundle(bundle).predict_file(raw_file, subject_id="fixture-subject")
        validate_prediction(result)
        assert result["model"]["run_key"] == "160afaf81debf1ee"

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/unit/test_prediction_schema.py tests/integration/test_predictor_raw.py -q

- [ ] **Step 3: Implement canonical graph using existing algorithms only.**

    def predict_sources(self, sources, options):
        batches = [self._window_batches(load_raw_session(source), source) for source in sources]
        macro_probs, micro_probs = self._score_windows(batches)
        candidates = self._generate_candidates(macro_probs, micro_probs)
        verifier = multiscale_verifier_features(candidates, macro_probs.by_sid, micro_probs.by_sid,
            context_features_version="v1", session_bounds_by_sid=self._bounds(batches))
        return self._prediction_result(self._apply_models_admission_policy(candidates, verifier), options)

Use bundle schema to assert 63/47/116 at each model boundary. Reuse existing candidate, Context-v1, admission and event-policy functions. Keep old serialized interface under legacy_payload.predict_feature_payload; do not delete it here.

- [ ] **Step 4: Thin CLI.**

Change scripts/predict_event_stack.py to parse INPUT, --bundle, --output, --subject-id, --include-timeline, --include-candidates, --device and call Predictor. Retain explicit --input-features mode routed to compatibility.

- [ ] **Step 5: Verify and commit.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/unit/test_prediction_schema.py tests/integration/test_predictor_raw.py tests/pipeline/test_event_stack_dist.py -q

    git add src/pipeline/inference dist/schema/prediction.schema.json scripts/predict_event_stack.py tests/unit/test_prediction_schema.py tests/integration/test_predictor_raw.py
    git commit -m "feat: add raw event-stack predictor API"

### Task 5: Require layer-by-layer canonical/Predictor parity

**Files:**
- Create: tests/parity/conftest.py, tests/parity/test_inference_four_way_parity.py
- Modify: src/pipeline/inference/predictor.py, legacy_payload.py
- Modify: tests/fixtures/release_160afaf81debf1ee/fixture_manifest.json

**Interfaces:**
- Produces internal InferenceTrace with spans, macro/micro/context matrices, probabilities, candidates, admitted rows, verifier scores and events.

- [ ] **Step 1: Write failing trace parity test.**

    def test_legacy_canonical_predictor_trace_is_identical(fixture, bundle):
        old, direct, predictor = legacy_trace(fixture, bundle), canonical_trace(fixture, bundle), Predictor.from_bundle(bundle).trace_file(fixture.raw)
        assert old.spans == direct.spans == predictor.spans
        for left, right in ((old.macro_features, predictor.macro_features), (old.micro_features, predictor.micro_features), (old.context_features, predictor.context_features)):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(old.verifier_scores, predictor.verifier_scores, rtol=1e-12, atol=1e-12)
        assert old.candidates == predictor.candidates
        assert old.admitted == predictor.admitted
        assert old.events == predictor.events

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/parity/test_inference_four_way_parity.py -q

- [ ] **Step 3: Implement trace-only observability.**

    @dataclass(frozen=True)
    class InferenceTrace:
        spans: tuple[tuple[str, int, int], ...]  # session ID, inclusive start, inclusive end
        macro_features: np.ndarray
        micro_features: np.ndarray
        context_features: np.ndarray
        macro_probabilities: np.ndarray
        micro_probabilities: np.ndarray
        candidates: tuple[dict[str, object], ...]  # canonical sorted candidate rows
        admitted: tuple[dict[str, object], ...]  # rows after frozen NMS/cap admission
        verifier_scores: np.ndarray
        events: tuple[dict[str, object], ...]  # final sorted decoder rows

Values more than 1e-12 from decision thresholds must match; boundary cases record score, threshold and the existing deterministic >= tie rule. Never skip them.

- [ ] **Step 4: Verify and commit.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/parity tests/integration/test_predictor_raw.py tests/pipeline/test_event_stack.py tests/pipeline/test_runner.py -q

    git add src/pipeline/inference tests/parity tests/fixtures/release_160afaf81debf1ee
    git commit -m "test: add canonical inference parity coverage"

### Task 6: Build standalone dist/inference and prove clean-room operation

**Files:**
- Create: scripts/build_inference_distribution.py
- Create: dist/inference/README.md
- Modify: src/pipeline/artifacts.py, tests/pipeline/test_event_stack_dist.py
- Create: tests/release/test_inference_clean_room.py

**Interfaces:**
- Produces build_inference_distribution(*, repository_root: Path, bundle_path: Path, destination: Path) -> Path.
- Generated directory contains exactly predict.py, event_stack/, models/, manifest.json, feature_schema.json, requirements.txt, README.md plus declared runtime dependency files.

- [ ] **Step 1: Write failing clean-room test.**

    def test_inference_runs_when_only_package_is_copied(tmp_path, raw_fixture):
        package = build_inference_distribution(repository_root=ROOT, bundle_path=DEPLOYMENT, destination=tmp_path / "inference")
        clean = tmp_path / "clean"; shutil.copytree(package, clean)
        done = subprocess.run([sys.executable, "-I", "predict.py", str(raw_fixture), "--output", "result.json"], cwd=clean, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        validate_prediction(json.loads((clean / "result.json").read_text()))

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/release/test_inference_clean_room.py -q

- [ ] **Step 3: Implement deterministic staging and manifest.**

    def build_inference_distribution(*, repository_root, bundle_path, destination):
        verify_current_promoted_release(repository_root)
        staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
        copy_canonical_runtime_source(repository_root / "src" / "pipeline", staging / "event_stack")
        copy_verified_deployment_bundle(bundle_path, staging / "models")
        write_predict_entrypoint(staging / "predict.py")
        write_exact_requirements(staging / "requirements.txt", bundle_path)
        write_distribution_manifest(staging, release_run_key="160afaf81debf1ee", prediction_schema_version="1.0")
        verify_distribution_manifest(staging)
        return atomic_replace_directory(staging, destination)

Manifest includes run key, model/schema versions, feature schema, dependency pins, sorted model/source files and SHA-256 hashes; rejects missing/extra/symlink/hash mismatch. Execute subprocess with python -I and no repository parent in import path.

- [ ] **Step 4: Verify and commit.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/release/test_inference_clean_room.py tests/pipeline/test_event_stack_dist.py tests/parity/test_inference_four_way_parity.py -q

    git add scripts/build_inference_distribution.py src/pipeline/artifacts.py dist/inference tests/release/test_inference_clean_room.py tests/pipeline/test_event_stack_dist.py
    git commit -m "build: create standalone inference distribution"

### Task 7: Publish visual schema and safe example

**Files:**
- Create: dist/visual/README.md, dist/examples/example_prediction.json
- Modify: dist/schema/prediction.schema.json, dist/README.md
- Create: tests/release/test_visual_contract.py

**Interfaces:**
- Required JSON: schema_version, model, input, events, diagnostics.
- Event fields: id, session_id, start_ms, end_ms, duration_s, confidence; optional timeline, candidates, gaps.

- [ ] **Step 1: Write failing validation/no-algorithm tests.**

    def test_visual_example_validates():
        jsonschema.validate(load_json(DIST/"examples/example_prediction.json"), load_json(DIST/"schema/prediction.schema.json"))
        assert load_json(DIST/"examples/example_prediction.json")["input"]["source"] == "synthetic-example"

    def test_visual_has_no_algorithm_python():
        assert not list((DIST/"visual").rglob("*.py"))

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/release/test_visual_contract.py -q

- [ ] **Step 3: Add schema, synthetic safe output and visual README.**

Document timeline keys timestamp_ms, macro_probability, micro_probability, valid, gap; candidates start_ms, end_ms, score, admitted; gaps start_ms, end_ms. Frontend only displays canonical output and cannot reimplement threshold/admission/fusion/decoder/features.

- [ ] **Step 4: Verify and commit.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/release/test_visual_contract.py tests/unit/test_prediction_schema.py -q

    git add dist/schema/prediction.schema.json dist/visual dist/examples/example_prediction.json dist/README.md tests/release/test_visual_contract.py
    git commit -m "docs: define visualization prediction schema"

### Task 8: Generate submission bundle with explicit unsupported official adapter

**Files:**
- Create: scripts/build_submission.py, src/pipeline/inference/competition_adapter.py, dist/submission/README.md
- Modify: tests/parity/test_inference_four_way_parity.py
- Create: tests/release/test_submission_clean_room.py

**Interfaces:**

    class CompetitionAdapter(Protocol):
        def load(self, path: Path) -> tuple[Path, str | None]:
            raise NotImplementedError
        def dump(self, prediction: Mapping[str, object], path: Path) -> None:
            raise NotImplementedError

    class UnsupportedCompetitionAdapter:
        def load(self, path: Path) -> tuple[Path, str | None]:
            raise NotImplementedError("official competition input/output adapter is not registered")

- [ ] **Step 1: Write failing submission and adapter tests.**

    def test_submission_raw_mode_matches_predictor(tmp_path, raw_fixture):
        package = build_submission(repository_root=ROOT, destination=tmp_path/"submission")
        assert final_events(run_submission_raw_mode(package, raw_fixture)) == final_events(Predictor.from_bundle(DEPLOYMENT).predict_file(raw_fixture))

    def test_official_adapter_refuses_unknown_contract(tmp_path):
        with pytest.raises(NotImplementedError, match="not registered"):
            UnsupportedCompetitionAdapter().load(tmp_path/"input")

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/release/test_submission_clean_room.py -q

- [ ] **Step 3: Implement build and raw verification mode.**

build_submission.py uses Task 6 staging/manifest routines, adds main.py, and atomically replaces only dist/submission. main.py --raw INPUT --output OUTPUT calls Predictor. --official-input/--official-output invokes only a registered adapter; default raises exact error above. README documents this known constraint and invents no wire fields.

- [ ] **Step 4: Make four-way parity mandatory.**

    def test_final_events_match_all_four_boundaries(raw_fixture):
        expected = legacy_promoted_events(raw_fixture)
        assert expected == predictor_events(raw_fixture)
        assert expected == run_dist_inference(raw_fixture)
        assert expected == run_dist_submission_raw_mode(raw_fixture)

- [ ] **Step 5: Verify and commit.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/release/test_submission_clean_room.py tests/release/test_inference_clean_room.py tests/parity/test_inference_four_way_parity.py -q

    git add scripts/build_submission.py src/pipeline/inference/competition_adapter.py dist/submission tests/release/test_submission_clean_room.py tests/parity/test_inference_four_way_parity.py
    git commit -m "build: add competition submission bundle"

### Task 9: Evidence-safe cleanup, wrappers and documentation

**Files:**
- Create: scripts/train_event_stack.py, scripts/evaluate_event_stack.py, scripts/reproduce_release.py
- Create: tests/fixtures/deletion_manifest.json, tests/release/test_repository_hygiene.py
- Modify: .gitignore, README.md, docs/三阶段重构设计.md, docs/数据处理说明.md, docs/repository_cleanup_audit.md
- Delete: only audit-proven paths in deletion manifest.

**Interfaces:**
- train_event_stack.py delegates legal full-target promotion/train; evaluate_event_stack.py delegates strict crossfit; reproduce_release.py --run-key 160afaf81debf1ee verifies registry, bundles, attestation and summary without retraining.
- Each deletion entry has path, imports, tests, reproduction, release, submission with all proof values false.

- [ ] **Step 1: Write failing reproduction/deletion-proof tests.**

    def test_reproduce_release_verifies_without_training():
        done = subprocess.run([PYTHON, "scripts/reproduce_release.py", "--run-key", "160afaf81debf1ee"], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "0.6514285714285715" in done.stdout

    def test_deleted_paths_have_no_consumers():
        for item in load_json(ROOT/"tests/fixtures/deletion_manifest.json")["deleted"]:
            assert not (ROOT/item["path"]).exists()
            assert rg_import_references(item["path"]) == []

- [ ] **Step 2: Run RED.**

Run: D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider tests/release/test_repository_hygiene.py -q

- [ ] **Step 3: Add wrappers and rewrite docs before deletion.**

Root README covers task, frozen F1, leakage-safe CV, architecture, raw quick inference, reproduce/evaluate/train, layout, submission, visual contract, limitations and compact historical table. dist README identifies inference/visual/submission/examples/schema; inference README gives install/predict command; submission README gives clean-room raw test and adapter constraint.

- [ ] **Step 4: Re-audit every deletion candidate immediately before deletion.**

For every candidate from audit (rank_events*, old ranker/slide/FD scripts, one-off diagnostics, legacy dist/predict*.py, dist/models, dist/slide_models, dist/src), record:

    rg -n --glob '!tests/fixtures/deletion_manifest.json' '<filename-or-module-stem>' src scripts tests README.md docs dist release
    D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider

If any import/test/reproduction/release/submission/docs command consumes it, change audit action to KEEP/REFACTOR; do not delete. Delete only all-false manifest entries. Preserve model root, registry, attestation, canonical evidence and manifest-referenced cache.

- [ ] **Step 5: Add ignore policy and clean rebuildable trash only.**

Add __pycache__/, .pytest_cache/, *.pyc, cache/ssl_checkpoints/, cache/embeddings/, outputs/tmp/, outputs/experiments/, .release-staging-*/. Never ignore model/evidence/release roots. Before removal, resolve each path and prove it lies inside D:\BMEtest.

- [ ] **Step 6: Full verification and two commits.**

Run:

    $env:PYTHONDONTWRITEBYTECODE='1'
    D:/Anaconda3/envs/bme/python.exe scripts/reproduce_release.py --run-key 160afaf81debf1ee
    D:/Anaconda3/envs/bme/python.exe scripts/build_inference_distribution.py
    D:/Anaconda3/envs/bme/python.exe scripts/build_submission.py
    D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider
    git status --short

Expected: TP 114, FP 83, predictions 197, F1 0.6514285714285715; four-way parity, clean rooms, schema validation, manifests/attestation pass and no temporary cache remains.

    git add -A
    git commit -m "chore: remove obsolete experiment scripts and artifacts"
    git add README.md docs dist .gitignore scripts tests
    git commit -m "docs: update competition release and reproduction guide"

## Final verification checklist

- [ ] Promoted metrics, evidence, model hashes, manifests and attestation remain valid.
- [ ] Raw file/folder prediction requires no manual feature payload/cache/fold/training data.
- [ ] Legacy promoted pipeline = canonical Predictor = dist/inference = dist/submission final events on fixtures.
- [ ] Both distribution directories pass isolated python -I clean-room execution.
- [ ] Visual example validates; visual has no algorithm implementation.
- [ ] Submission build is deterministic and its unknown official adapter fails explicitly.
- [ ] Deletion manifest proves every deletion; current models/evidence/caches remain.
- [ ] All README commands work; git status --short contains no accidental cache/temp output.

## Self-review against specification

| Spec sections | Covered by |
|---|---|
| 2 immutable release | Constraints, Tasks 1 and 5, final checklist |
| 3–5 canonical source/scripts | Ownership, Tasks 2, 3, 9 |
| 6–8 Predictor/raw/schema | Task 4 |
| 9–10 visual/standalone inference | Tasks 6–7 |
| 11–13 submission/parity/clean room | Tasks 5, 6, 8 |
| 14–17 scripts/models/outputs/cache cleanup | Task 9 |
| 18–20 explainability/docs/manifests | Tasks 6, 8, 9 |
| 21–26 exclusions/order/commits/failure/DoD | Constraints, task gates, checklist |

No official I/O placeholder is hidden: the explicit tested unsupported adapter is the required isolation boundary. All other implementation interfaces, files, tests, commands and commit boundaries are stated above.
