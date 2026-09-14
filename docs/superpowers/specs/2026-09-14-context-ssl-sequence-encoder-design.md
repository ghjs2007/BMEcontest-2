# Context and Self-Supervised Sequence Encoder Design

Status: approved direction, implementation gated by this specification.  
Date: 2026-09-14.

## 1. Goal and success criteria

The project target remains leakage-safe subject-independent event F1 `>=0.65`. The current locked
development result is run `035644cf0889a5dd`: F1 `0.5589743590`, TP `109/153`, predictions `237`,
FP `128`. Work proceeds in three separately attributable stages:

1. deterministic hard-negative context features;
2. fold-contained self-supervised TCN embeddings;
3. a lightweight Transformer encoder control.

Each stage is a separate registered experiment. A stage can replace the current release only when
its five-fold aggregate F1 is strictly greater than `0.5589743590`, all five outer folds complete,
the evidence/manifest/attestation chain verifies, and the full test suite passes. Short-meal recall,
PPV, recall, candidate recall, FP count, time and peak memory are mandatory diagnostics, not hidden
promotion criteria. Results from repeated outer CV remain development evidence, not an untouched-test
generalization claim.

## 2. Fixed evaluation and leakage boundary

The eligible denominator remains 153 events, including 39 meals shorter than ten minutes. Event IoU
`>=0.25` defines a match. Outer validation subjects are excluded from every fit, normalization,
feature selection, self-supervised update, early-stopping decision, threshold search and policy search.

Within each outer training partition, all learned candidate scores used to train or select a verifier
come from subject-disjoint inner OOF predictions. Deterministic context extraction may run on an outer
validation session because it has no fitted state; any learned context normalization is fitted on the
corresponding training partition only. No subject's labels may influence features or thresholds used
to score that same subject.

For self-supervision, only raw IMU belonging to the current outer-train subjects is admissible. Using
outer-validation raw signals, even without labels, is transductive adaptation and is forbidden for the
reported subject-independent score. External pretraining data must have documented provenance,
license, sensor location, channels and sampling rate, and must be compared against both random
initialization and target-only self-supervision under identical downstream folds.

## 3. Stage A: hard-negative context features

### 3.1 Architecture

Add a focused context-feature module under `src/pipeline/`. It consumes one candidate, its session's
ordered macro/micro window timestamps and probabilities, and produces a fixed schema. It does not read
labels or fit a model. `multiscale_verifier_features` appends this block behind the existing 56 features
only when a versioned `context_features_enabled` run configuration flag is true.

The first registered feature block is deliberately small:

- pre/candidate/post macro and micro probability mean, maximum, standard deviation and above-threshold
  fraction;
- candidate-to-pre and candidate-to-post differences for those summaries;
- valid-window coverage for each region and scale;
- number, total duration and maximum duration of neighboring candidate-like runs;
- distance to the nearest preceding and following candidate-like run;
- within-candidate temporal concentration and first-half versus second-half probability difference.

Regions are clipped to the same session and use real timestamps. Empty regions yield `NaN` plus an
explicit coverage value; the existing imputer handles `NaN`, while `Inf` remains invalid. Features may
not cross `sid` boundaries and may not aggregate by subject. The schema order and count are constants,
stored in bundle `feature_schema.json`, included in cache keys and verified against fitted model widths.
The context builder preserves `NaN` for missing regions until the training pipeline imputer; the current
final `nan_to_num` behavior must not silently erase missingness before coverage features can explain it,
while `Inf` is rejected.

### 3.2 Experiment sequence

First run one deterministic unit/integration test suite and a single outer-fold smoke experiment.
Then run all five folds with the current run configuration plus context features as the only changed
factor. If F1 does not improve, delete generated fold results and feature caches for that run key and
retain only a concise README history entry. Do not tune the feature list against outer-fold errors.

## 4. Stage B: fold-contained self-supervised TCN

### 4.1 Representation task

Use wrist ACC+GYRO windows already produced by the micro data path. A compact 1D TCN encoder outputs a
fixed 128-dimensional embedding. Target-only self-supervision uses two augmented views of the same
window with time masking, small Gaussian noise, bounded amplitude scaling, channel dropout and small
3D rotations applied consistently to vector channels. The initial objective is augmentation-consistency
contrastive learning; no eating labels enter pretraining.

Pretraining is executed independently inside every inner/outer training partition. Checkpoints record
the exact training-subject hash, architecture, augmentations, seed, dependency versions and source-file
fingerprints. A checkpoint whose subject hash does not match the requested partition must be rejected.
Deterministic seeds and subject-balanced sampling are required.

### 4.2 Downstream use

For each event candidate, pool encoder outputs across intersecting micro windows using mean, standard
deviation and maximum. To control dimensionality, a projection fitted only on the relevant training
partition may reduce the pooled vector before it is appended to verifier features. The baseline,
randomly initialized encoder, self-supervised frozen encoder and self-supervised fine-tuned encoder are
distinct registered variants. The first implementation stops at a frozen encoder unless it improves
F1; supervised fine-tuning is not bundled into the same initial experiment.

Training supports `--device auto|cpu|cuda`; `auto` selects CUDA only when PyTorch reports a usable CUDA
device. Mixed precision is allowed on CUDA, but saved embeddings and event scores must be finite.
CPU/CUDA parity is tested with tolerances on embeddings and identical decoded events. A CUDA-trained
encoder is not shipped until the packaged runtime has an explicit audited CUDA component and the
current anti-fake adapter checks pass.

## 5. Stage C: lightweight Transformer control

The Transformer consumes the same windows, augmentations, 128-dimensional output contract and fold
boundaries as the TCN. It uses a small convolutional patch stem, positional encoding and at most four
encoder layers. Dataset, objective, optimizer budget, downstream pooling and evaluation are held fixed
against TCN. It is attempted only after Stage B infrastructure and leakage tests pass. It is promoted
only on measured event F1, not window AUC.

## 6. Interfaces, storage and repository hygiene

Run configuration fields, feature schema version, encoder type, embedding width and pretraining subject
hash flow into `experiment_key` and every cache key. `RunConfig`, feature-dimension calculation and the
current hard-coded `63/56/47` bundle validation are updated together; old 56-wide verifier bundles remain
valid only under their old schema version and are never silently padded. Learned encoder artifacts live under the promoted
`models/event_stack/<run-key>/` bundle, never in the repository root. Rebuildable embeddings and
checkpoints live under narrowly named ignored cache directories with per-run size reporting.

Only the current promoted five fold records, aggregate summary, 5+1 bundles, attestation and packaged
`dist/event_stack/` remain after promotion. Failed experiment outputs, temporary scripts, bytecode and
superseded caches are removed. Existing data, `cache/sessions`, `cache/slide`, `cache/micro15`, split
definitions and protected source datasets are not cleanup targets. Every successful algorithm update
is committed after tests and documentation updates.

The current precomputed-feature input contract remains valid during Stage A. Stage B extends the bundle
schema explicitly; older bundles must fail with a clear version/schema mismatch rather than silently
padding features. Raw-session inference is a separate adapter project and is not implicitly solved here.

## 7. Tests and evidence

Stage A tests cover timestamp clipping, session isolation, empty coverage, stable ordering, finite-value
validation, schema/cache-key changes, inner/outer subject isolation and unchanged legacy output when
disabled. Stages B/C add augmentation determinism, subject-hash rejection, train/validation exclusion,
checkpoint provenance, CPU execution, optional CUDA execution/parity, pooling and bundle round trips.

Before each promotion, run the focused tests, a single-fold smoke, the strict five-fold experiment, all
bundle/attestation/dist checks and the complete suite with bytecode and pytest cache disabled. README,
the architecture document and `dist/README.md` must state the exact run key, metrics, feature widths,
device behavior, limitations and reproduction command. Before promoting the first result from this
specification, update and test the promotion floor: the implementation currently records
`PROMOTION_F1_FLOOR=0.5432098765`, while the new release must strictly exceed the current
`0.5589743590` baseline.

## 8. Stop conditions

Stop a stage and clean its artifacts when it fails its registered gate, violates subject isolation,
exceeds available disk without a bounded cache policy, produces non-finite features, or cannot be
reproduced from its recorded configuration. Stage A receives one fixed feature schema and one five-fold
evaluation before redesign. Stage B receives target-only SSL and its controls before external data is
introduced. Stage C is not started merely because Stage B underperforms; the shared representation
pipeline must first be verified correct.
