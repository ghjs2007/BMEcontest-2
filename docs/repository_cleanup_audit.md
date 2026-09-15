# Repository cleanup audit — Phase 1 (read-only)

Audit date: 2026-09-15.  Scope is the current `main` worktree only.  This
document deliberately makes no move, deletion, or algorithm change.  Its
purpose is to establish the immutable pre-refactor contract and the proof
obligations for later phases of the competition-delivery refactor.

## Immutable baseline: `CURRENT_PROMOTED_RELEASE`

`CURRENT_PROMOTED_RELEASE` is **`160afaf81debf1ee`**.  It is established by
the tracked incumbent registry `release/event_stack_incumbent.json`, whose
run key, aggregate F1, summary hash and attestation hash agree with
`models/event_stack/160afaf81debf1ee/`.

| Property | Locked value |
|---|---|
| Evaluation | strict subject-disjoint nested five-fold CV; four inner splits |
| Aggregate F1 | `0.6514285714285715` |
| TP / eligible GT / predictions | `114 / 153 / 197` |
| FP | `83` |
| PPV / recall | `0.5786802030456852` / `0.7450980392156863` |
| Candidate count / candidate recall | `230` / `0.7712418300653595` |
| Short-meal candidate recall / final recall | `0.5384615384615384` / `20/39 = 0.5128205128205128` |
| Feature schema | v2: macro `63`, micro `47`, verifier `116` (56 base + 60 Context-v1) |
| Deployment policy | admission threshold `.2`, NMS IoU `.3`, subject cap `6`, blend `.25`, decoder threshold `.6595272837359293` |
| Runtime ABI | Python `3.11.15`; numpy `2.4.6`; joblib `1.5.3`; scikit-learn `1.9.0`; lightgbm `4.7.0` |

Five immutable outer-evidence records:

| Fold | Config hash | TP / GT / pred | F1 |
|---:|---|---:|---:|
| 0 | `afcf9609a2f6925a` | 16 / 23 / 35 | .5517241379 |
| 1 | `8b3e27bd2796d038` | 30 / 31 / 46 | .7792207792 |
| 2 | `b463226db1e6073a` | 15 / 27 / 33 | .5000000000 |
| 3 | `3f26fcb2172cd882` | 25 / 32 / 42 | .6756756757 |
| 4 | `0199041f39a09f5d` | 28 / 40 / 41 | .6913580247 |

### Evidence and source anchors

| Artifact | SHA-256 |
|---|---|
| `release/event_stack_incumbent.json` | `27ee2f0ad4cd85cb8819e4c6fa71e985c8ec2d3666ec8eed8f14602faf0009b8` |
| `models/event_stack/160afaf81debf1ee/promotion_summary.json` | `6d435ebb829b06b16f50e99f360cda62a516dc291b6ccb0574ce4b9ed71945a9` |
| `models/event_stack/160afaf81debf1ee/promotion_attestation.json` | `2e8235c10ed1b02f67cc7c10124b85d0e85d6e9671f1d423a24d02bec79024b5` |
| deployment `manifest.json` | `71a7cdd7f181f3cc02309d35513c6fc65b9fed4338637b4e96dc4c4f182d7237` |
| `src/pipeline/event_stack.py` | `f04a2c83a4c7bb08f59908d4d3ad38cd6397435e6205e1df408013560c4c4248` |
| `src/pipeline/runner.py` | `094021edf6839edf50641e69f31e924c183db2f492217754a7c9bfee575f1b50` |
| `src/pipeline/imu_features.py` | `f6a3cf4e0e145cb8e105107eee77a7ae7bdd4ca4bb90457c7b91d65850fb1b4c` |
| `src/pipeline/context_features.py` | `4eca09cadc3f165595fe506e9a3d50bf7f2976e6a26c63716b444859722d9b28` |
| `src/pipeline/candidate_control.py` | `c65326fc1f61d23ebb5f9df8ae39bb2801584c002740b7542c8fb1da7ea7f373` |
| `scripts/predict_event_stack.py` and active dist copy | `4c7c8a5e01ea4d58969d4ce194e4f7ca244177cea5ec5d4d13f46d22a98765e9` |
| `dist/event_stack/runtime_manifest.json` | `253a9b616a77e4bde3f8b89a9e96d268a1ee6c16b15ad8f696426670b255b1f6` |

The deployment manifest has source fingerprints for the raw data, session,
slide, micro and split caches used to reproduce the release.  Some cache
entries are explicitly recorded as missing; that is evidence of the historic
build environment, not permission to delete more cache.  Phase 8 must not
remove any manifest-referenced cache until raw Predictor parity and a revised,
verified reproduction contract exist.

## Current dependency graph

```text
Data CSV + cache/sessions + cache/splits + cache/slide + cache/micro15
  -> FilesystemDataSource / src.data.{loader,manifests,splits}
  -> runner.build_*_dataset
       -> event_stack macro feature/window model/density candidate generation
       -> imu_features + micro_cache micro ACC+GYRO model/candidates
       -> candidate_control (same-sid NMS and subject admission)
       -> context_features (fixed Context-v1 60 columns)
       -> runner verifier LogisticRegression + constrained LightGBM blend
       -> frozen policy / event decoder / official matching in event_stack
  -> scripts/crossfit_event_stack.py (nested-CV evidence)
  -> scripts/promote_event_stack.py -> artifacts (5 evidence + deployment)
  -> promotion_summary + promotion_attestation + incumbent registry
  -> scripts/release_event_stack.py -> scripts/package_event_stack.py
  -> dist/event_stack/{predict_event_stack.py,bundle,runtime_manifest.json}

tests/pipeline/{test_runner,test_multiscale_runner,test_candidate_control,
test_context_features,test_imu_features,test_micro_cache,test_event_stack,
test_artifacts,test_event_stack_dist,test_release_event_stack}
  -> every bolded production/release layer above.
```

The canonical event-stack algorithm is already largely in `src/pipeline/`.
However, raw competition-file loading and end-to-end Predictor API are absent:
the current runtime CLI accepts serialized feature/candidate payloads, not raw
sensor files.  The duplicated `scripts/predict_event_stack.py` and
`dist/event_stack/predict_event_stack.py` have identical bytes today, but are
two maintained paths and must be canonicalized before release cleanup.

## Inventory and next-phase disposition

Action is a plan, not an authorization to delete now.  `DELETE-CANDIDATE`
means the stated import/reproduction/parity proof remains required.

| Path(s) | Current role | Promoted pipeline | Training reproduction | Inference | Action |
|---|---|:---:|:---:|:---:|---|
| `src/pipeline/{event_stack,runner}.py` | macro windows, candidates, verifier, decoder, orchestration | yes | yes | future | KEEP; split only behind parity tests |
| `src/pipeline/{imu_features,micro_cache}.py` | gravity-aligned 47-D ACC+GYRO and cache ABI | yes | yes | future | KEEP |
| `src/pipeline/{context_features,candidate_control,diagnostics,crossfit}.py` | Context-v1, admission, diagnostics, subject-safe OOF | yes | yes | future | KEEP |
| `src/pipeline/artifacts.py` | bundle manifests, attestation, incumbent contract | yes | yes | yes | KEEP |
| `src/data/{loader,manifests,splits}.py` | raw/session and split readers | yes | yes | future | REFACTOR into raw-inference path without behavior change |
| `src/eval/metrics.py` | official matching primitives | yes | yes | indirect | KEEP |
| `scripts/crossfit_event_stack.py` | current registered training/evaluation CLI | yes | yes | no | KEEP; make thin `evaluate_event_stack` wrapper later |
| `scripts/promote_event_stack.py` | legal full-target trainer | yes | yes | no | KEEP; make thin release-training wrapper later |
| `scripts/{release_event_stack,package_event_stack}.py` | active release transaction and dist build | yes | yes | yes | KEEP; package must become `build_submission` sibling later |
| `scripts/predict_event_stack.py` | active serialized-payload CLI; copied verbatim into dist | yes | no | yes | REFACTOR into `src/pipeline/inference`; preserve byte/event parity |
| `dist/event_stack/` | current standalone serialized-payload runtime, models and ABI evidence | yes | no | yes | KEEP until replacement clean-room runtime passes |
| `models/event_stack/160afaf81debf1ee/` | 5 outer bundles, deployment, summary, attestation | yes | yes | yes | KEEP immutable |
| `release/event_stack_incumbent.json` | canonical active release pointer | yes | yes | yes | KEEP immutable |
| `cache/{sessions,splits,slide,micro15}/` | manifest-fingerprinted reproduction inputs | yes | yes | no | KEEP pending post-parity cache policy |
| `outputs/crossfit/summary_160afaf81debf1ee.json` and registered aliases | canonical scientific evidence referenced by README/release | yes | yes | no | KEEP; move only if hash chain is rebuilt and verified |
| `scripts/build_micro_features.py` | current micro-cache build CLI; parser has direct test coverage | yes | yes | no | KEEP; thin wrapper candidate |
| `scripts/bootstrap_event_stack_diagnostics.py` | one-time incumbent diagnostic repair/bootstrap | no | evidence maintenance | no | REFACTOR/retire only after an immutable migration test replaces it |
| `scripts/{slide_features,slide_verifier}.py` | historical slide pipeline; parity test imports `slide_verifier` | no | historical comparison | no | KEEP until parity test and README history are migrated/retired |
| `scripts/{rank_events,rank_events_v2,official_iou_eval,train_ranker,tcn_slide_score,pretrain_fd,prep_fd,fd_slide_exp,loso_eval,slide_eval,train_deploy_slide}` | legacy/deep/proposal experiments named by README | no | no for current release | no | DELETE-CANDIDATE after import search, historical README rewrite and full tests |
| `scripts/{analyze_bands,analyze_label_offset,bag_eval,calib_decode,compress_sessions,compress_sessions2,diag_decode_split,diag_miss_attribution,diag_pri_subwin,diag_proposals,eval_global_thr,summarize_v2,predict}.py` | one-off analysis, historical cache prep or legacy inference | no | possibly raw-cache provenance | no | DELETE-CANDIDATE; first prove `compress_sessions2` behavior is extracted for raw Reader |
| `dist/{predict.py,predict_legacy.py,predict_slide.py,models/,slide_models/,src/}` | obsolete legacy distribution candidates separate from active `dist/event_stack` | no | no | unclear | HOLD; must inspect competition consumers and run clean-room tests before deletion |
| `README.md`, `docs/三阶段重构设计.md`, `docs/数据处理说明.md` | current/historical documentation and reproduction commands | yes | yes | yes | REFACTOR after canonical commands exist; preserve a compact history table |
| `tests/pipeline/` | release/algorithm regressions | yes | yes | yes | KEEP; reorganize only after coverage is preserved |

## Required parity boundaries for later phases

1. Existing registered nested-CV result must retain the five records and
   aggregate above; no threshold, admission, feature, imputation or matching
   change is allowed.
2. For legal existing session fixtures, compare preprocessing boundaries,
   macro `63`, micro `47`, Context-v1 `60`, verifier `116`, probabilities,
   admission and final decoded events between legacy current code, canonical
   `Predictor`, `dist/inference`, and `dist/submission`.
3. Verify every bundle manifest and promotion attestation before and after a
   structural commit.  The active runtime must still match the deployment
   manifest hash recorded by `dist/event_stack/runtime_manifest.json`.
4. Run tests with bytecode and pytest cache disabled; deletion may proceed only
   after the full suite and the clean-room inference/submission tests pass.

## Ambiguities and blockers to isolate, not guess

* **Task 3 macro producer parity record (2026-09-15):** a legal raw session was
  paired with `cache/sessions/` and `cache/slide/fold0_val.npz`.  The raw
  parser arrays equal the session cache and the legacy
  `scripts/slide_features.py:_process_session` output equals the selected 278
  slide rows, including finite float32 bytes and NaN representation.  The
  recorded fixture hashes and fingerprints are in
  `tests/fixtures/release_160afaf81debf1ee/fixture_manifest.json`.  However,
  the direct legacy producer/cache is **62-D**, while the frozen release's
  model-facing macro schema is **63-D** because `runner._with_time_prior`
  appends its one frozen column after cache loading.  The approved delivery
  plan resolves this as two immutable boundaries: a raw 62-D producer and the
  separate pure `add_time_prior` adapter.  Both matrices, schemas, the adapter
  fingerprint, windows and NaN-preserving bytes are locked by three-way parity
  tests; no model/policy behavior is implied by this structural split.

* **Official competition adapter:** `Resources/试题.txt` describes scoring,
  but this audit found no authoritative final machine input/output file schema
  or runner contract.  `dist/submission/main.py` must therefore isolate the
  adapter and declare a TODO; it must not invent the official format.
* **Raw input discovery:** existing data reading is organised around the
  repository's CSV/session-cache conventions.  A supported raw file/folder
  contract, including how multiple files form a session and how `sid` is
  supplied, must be documented and tested before `Predictor.predict_file()`
  can be claimed complete.
* **Current dist scope:** `dist/event_stack` is a validated standalone runtime
  only for serialized feature/candidate payloads.  It is not yet the raw-data
  inference workspace requested by the delivery specification.  Legacy files
  beside it cannot be deleted until their ownership is resolved.
* **Evidence location:** the current summary aliases are ignored runtime
  outputs while the immutable model-root summary/attestation are tracked.
  Do not mechanically move evidence to `outputs/release/`: first preserve or
  deliberately regenerate the attestation hash chain.

## Test baseline

Command launched from repository root with cache disabled:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/Anaconda3/envs/bme/python.exe -m pytest -p no:cacheprovider
```

Result: **299 passed, 2 skipped in 197.24s (0:03:17)**.  The command ran in a
separate retained process; its exit was observed after pytest printed the
complete summary.  This is the pre-refactor test baseline.  Deletion still
requires the same full suite plus the future raw Predictor and clean-room
distribution parity gates.
