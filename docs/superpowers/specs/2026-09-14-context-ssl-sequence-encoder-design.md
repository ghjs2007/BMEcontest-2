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

Each stage is a separate registered experiment. The locked scientific baseline for this specification
is `0.5589743590`; the release incumbent is read from an attested registry and advances after every
successful promotion. A stage can replace the current release only when its five-fold aggregate F1 is
strictly greater than the incumbent promoted aggregate F1, all five outer folds complete,
the evidence/manifest/attestation chain verifies, and the full test suite passes. Short-meal recall,
PPV, recall, candidate recall, FP count, time and peak memory are mandatory diagnostics, not hidden
promotion criteria. Results from repeated outer CV remain development evidence, not an untouched-test
generalization claim. Any strict aggregate improvement is technically promotion-eligible and, per the
project release rule, is solidified with models and `dist/`; improvements smaller than `0.005` are
reported as scientific ties rather than algorithmic breakthroughs. A result is marked recommended only
when aggregate `delta F1 >=0.005` and at least three of five outer folds are non-worse. These labels do
not change the pre-registered technical gate. B2 additionally must exceed its paired B0 and B1 results;
a later stage can never overwrite `dist/` with a score below the incumbent even if it beats the original
scientific baseline.

Every stage also reports per-subject event F1 median, interquartile range, 10th percentile, and counts
of subjects improved, unchanged and worsened against the paired current baseline. These are mandatory
cross-subject diagnostics, not additional hidden promotion criteria.

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

Context-v1 fixes its complete morphology before evaluation. Pre and post regions are each 20 minutes.
The above-threshold fractions use macro probability `>=0.28838` and micro probability `>=0.20`.
A candidate-like run is a maximal sequence above the corresponding fixed threshold, joined only when
the timestamp gap is `<=60 s`, with at least two windows. Neighboring-run features exclude the event
candidate's own intersecting run. These horizons, thresholds, gap and minimum-window count are part of
the context schema version; they never come from an inner-selected admission or decoder policy.

### 3.2 Mechanical Context-v1 schema

Window membership uses the real-time center: pre `[candidate_start-20 min, candidate_start)`, candidate
`[candidate_start, candidate_end)`, and post `[candidate_end, candidate_end+20 min)`, all clipped to the
same session. Macro nominal stride is 15 s and micro nominal stride is 7.5 s. Region coverage is
`finite_probability_window_count / max(1, ceil(clipped_region_duration / nominal_stride))`, clipped to
`[0,1]`. Statistics use finite probabilities only. An empty region has `NaN` mean/max/std/above-fraction,
zero coverage, and any difference involving it is `NaN`.

For each scale, candidate-like runs are built from all finite-probability window centers in the clipped
pre+candidate+post interval. A run span is `[first_center-window_length/2,
last_center+window_length/2)`; its duration is that span length. Runs whose span intersects the candidate
are the candidate's own runs and are excluded from neighbor summaries. If there are no neighboring runs,
count and total duration are zero, while maximum duration and preceding/following distance are `NaN`.
Center-half concentration is the probability mass in the middle half of the candidate interval divided
by all non-negative finite probability mass in the candidate; a zero denominator yields `NaN`. Half
difference is first-half mean minus second-half mean.

Context-v1 appends exactly 60 columns: all 30 `macro_` columns below, then all 30 `micro_` columns in
the same suffix order. This ordered list and its SHA-256 are stored in `feature_schema.json`:

```text
pre_mean, pre_max, pre_std, pre_above_fraction,
candidate_mean, candidate_max, candidate_std, candidate_above_fraction,
post_mean, post_max, post_std, post_above_fraction,
candidate_minus_pre_mean, candidate_minus_pre_max,
candidate_minus_pre_std, candidate_minus_pre_above_fraction,
candidate_minus_post_mean, candidate_minus_post_max,
candidate_minus_post_std, candidate_minus_post_above_fraction,
pre_coverage, candidate_coverage, post_coverage,
neighbor_run_count, neighbor_run_total_duration_s, neighbor_run_max_duration_s,
preceding_run_distance_s, following_run_distance_s,
candidate_center_half_mass_fraction, candidate_first_minus_second_mean
```

A macro-only, micro-only or fused candidate is evaluated against both complete session probability
streams. If a scale has no windows in the session, its statistics/differences/distances/concentration
are `NaN`, coverages and run count/total duration are zero. The disabled branch remains bit-for-bit on
the legacy 56 columns, including its existing `nan_to_num`; only the enabled 116-column branch preserves
context `NaN` for `SimpleImputer` and rejects `Inf`.

Regions are clipped to the same session and use real timestamps. Empty regions yield `NaN` plus an
explicit coverage value; the existing imputer handles `NaN`, while `Inf` remains invalid. Features may
not cross `sid` boundaries and may not aggregate by subject. The schema order and count are constants,
stored in bundle `feature_schema.json`, included in cache keys and verified against fitted model widths.
The context builder preserves `NaN` for missing regions until the training pipeline imputer; the current
final `nan_to_num` behavior must not silently erase missingness before coverage features can explain it,
while `Inf` is rejected.

### 3.3 Experiment sequence

First run one deterministic unit/integration test suite and a single outer-fold smoke experiment.
Then run all five folds with the current run configuration plus context features as the only changed
factor. If F1 does not improve, delete generated fold results and feature caches for that run key and
retain only a concise README history entry labelled `Context-v1 failed`. Do not change the 20-minute
horizon, fixed thresholds, run definition or feature list in response to those outer-fold errors; proceed
to the already registered Stage B.

## 4. Stage B: fold-contained self-supervised TCN

### 4.1 Representation task

Use wrist ACC+GYRO windows already produced by the micro data path. A compact 1D TCN encoder outputs a
fixed 128-dimensional embedding. Target-only self-supervision uses two augmented views of the same
window with time masking, small Gaussian noise, bounded amplitude scaling, channel dropout and small
3D rotations applied consistently to vector channels. ACC and GYRO receive the same rotation matrix.
Because the current micro input is gravity-aligned, rotation augmentation is restricted to a bounded
yaw-like rotation around the aligned gravity axis; arbitrary rotations that move gravity are forbidden.
No eating labels enter pretraining.

The first registered objective is VICReg-style augmentation-consistency learning, not NT-Xent: mean
squared invariance plus variance and covariance regularization with fixed weights `25/25/1`. It never
treats other windows as negatives, avoiding false negatives from adjacent instances of the same free-
living behavior. Later contrastive objectives require a new registration and must at minimum exclude
same-`sid` windows within 20 minutes and the same candidate-like temporal neighborhood from negatives.

Pretraining is executed independently inside every inner/outer training partition. SSL-v1 fixes
`inner_splits=4`. For each outer fold,
each inner-validation subject is embedded by an SSL encoder trained only on that inner fold's training
subjects. An encoder pretrained on all outer-train subjects may be used only for the final refit after
all inner verifier, threshold and policy selections are frozen; it may not generate inner OOF embeddings.
Thus one outer fold has four inner-train SSL fits plus one final outer-train SSL fit. Checkpoints record
the exact training-subject hash, architecture, augmentations, seed, dependency versions and source-file
fingerprints. A checkpoint whose subject hash does not match the requested partition must be rejected.
Deterministic seeds and subject-balanced sampling are required. SSL-v1 uses exactly 10,000 optimizer
steps per partition, effective batch size 256, and no early stopping. A smaller physical batch caused by
the available 8 GiB VRAM uses gradient accumulation to preserve an effective batch of 256; optimizer-
step count and augmentation schedule remain unchanged. The partition's inner-split count, train-subject
hash and validation-subject hash are recorded in checkpoint and cache provenance.

SSL-v1 resamples each valid 15 s ACC+GYRO window to 100 Hz (1,500 time steps), uses train-partition
per-channel mean/std normalization, and fixes this TCN: six-channel input; Conv1d stem with 64 channels,
kernel 7 and stride 2; four residual blocks with 128 channels, kernel 5 and dilations 1/2/4/8; GroupNorm,
GELU and global average pooling; linear 128-dimensional encoder output; and a 128-256-128 VICReg
projector used only during pretraining. AdamW uses learning rate `1e-3`, weight decay `1e-4`, cosine
decay over 10,000 steps, and no warm-up. Seed is `20260914 + outer_fold*100 + inner_fold`, with
`inner_fold=99` for the final outer-train fit. B1 uses the identical seeded initialization before any
SSL update. These values are stored in a tracked `ssl-v1` configuration artifact and are not searched.

### 4.2 Downstream use

Each 128-dimensional micro-window embedding is first projected to exactly 32 dimensions by PCA fitted
only on the corresponding training partition's micro windows. Inner OOF PCA sees inner-train subjects
only; the final outer model's PCA sees all outer-train subjects only. PCA provenance records its subject
hash, window count and raw-source fingerprint, and rejects mismatches. Candidate pooling then concatenates mean,
standard deviation and maximum into a fixed 96-dimensional block. Projection size is not searched.
Projecting after event pooling is forbidden because candidate-level sample size is too small.

The first Stage B comparison has exactly three variants with identical downstream admission, verifier
and decoder configuration: B0 current promoted representation without embeddings, B1 frozen randomly
initialized TCN embeddings, and B2 frozen target-only VICReg TCN embeddings. B2 must improve relative
to both B0 and B1 before the result is described as evidence for self-supervised transfer. Supervised
fine-tuning B3 is a later separately registered experiment.

Training supports `--device auto|cpu|cuda`; `auto` selects CUDA only when PyTorch reports a usable CUDA
device. Mixed precision is allowed on CUDA, but saved embeddings and event scores must be finite.
CPU/CUDA parity requires `allclose` embeddings and verifier scores with `rtol=1e-4` and `atol=1e-4`.
Decoded
events must be identical when every relevant score is more than `1e-5` from every decision threshold.
Boundary cases inside that epsilon are reported separately and follow a deterministic `score >= threshold`
tie policy rather than failing parity solely for floating-point noise. The packaged CPU path remains
deterministic. A CUDA-trained encoder is not shipped until the packaged runtime has an explicit audited
CUDA component and the current anti-fake adapter checks pass.

## 5. Stage C: lightweight Transformer control

The Transformer consumes the same windows, augmentations, 128-dimensional output contract and fold
boundaries as the TCN. It uses a small convolutional patch stem, positional encoding and at most four
encoder layers. Dataset, objective, optimizer budget, downstream pooling and evaluation are held fixed
against TCN. It is attempted only after Stage B infrastructure and leakage tests pass. It is promoted
only on measured event F1, not window AUC. Exact parameter equality is not required; every TCN and
Transformer result reports trainable parameters, MACs/FLOPs, peak VRAM, SSL wall time and embedding
throughput so a small F1 difference can be interpreted against its engineering cost.

## 6. Interfaces, storage and repository hygiene

Run configuration fields, feature schema version, encoder type, embedding width and pretraining subject
hash flow into `experiment_key` and every cache key. `RunConfig`, feature-dimension calculation and the
current hard-coded `63/56/47` bundle validation are updated together; old 56-wide verifier bundles remain
valid only under their old schema version and are never silently padded. Learned encoder artifacts live under the promoted
`models/event_stack/<run-key>/` bundle, never in the repository root. Rebuildable embeddings and
checkpoints live under narrowly named ignored cache directories with per-run size reporting.

The existing `cache/micro15` contains only 47-dimensional engineered features and is never used as raw
encoder input or modified by Stage B. Stage B adds a separate ignored raw-window cache, sharded by
session and source fingerprint, containing float ACC/GYRO sequences, window start/end timestamps,
`sid`, subject ID, source sampling rate, gravity-alignment state and valid coverage. Partition views and
embedding caches are keyed by the exact ordered subject hashes plus encoder/projection configuration;
the runner exposes separate inner-train, inner-validation, outer-train and outer-validation selectors.
Session arrays are stored once and windows are index views rather than duplicated overlapping tensors.
The combined SSL raw/embedding/checkpoint cache has a 25 GiB cap, reports bytes per run, and evicts only
unreferenced failed-run shards after resolving their absolute paths under the dedicated SSL cache root.

Only the current promoted five fold records, aggregate summary, 5+1 bundles, attestation and packaged
`dist/event_stack/` remain after promotion. Failed experiment outputs, temporary scripts, bytecode and
superseded caches are removed. Existing data, `cache/sessions`, `cache/slide`, `cache/micro15`, split
definitions and protected source datasets are not cleanup targets. Every successful algorithm update
is committed after tests and documentation updates.

A tracked incumbent registry under `models/event_stack/` stores the promoted run key, aggregate F1,
summary hash and attestation hash. Promotion first verifies the candidate against that registry, writes
the candidate bundle, packages and verifies a temporary `dist`, and only then replaces `dist/` and the
registry with rollback on failure. The registry is the release-replacement floor; the fixed scientific
baseline remains in this specification. This replaces reliance on a stale module-level promotion floor.

The current precomputed-feature input contract remains valid during Stage A. Stage B extends the bundle
schema explicitly; older bundles must fail with a clear version/schema mismatch rather than silently
padding features. Raw-session inference is a separate adapter project and is not implicitly solved here.

## 7. Tests and evidence

Stage A tests cover timestamp clipping, session isolation, empty coverage, stable ordering, finite-value
validation, schema/cache-key changes, inner/outer subject isolation and unchanged legacy output when
disabled. Stages B/C add augmentation determinism and shared ACC/GYRO rotation, subject-hash rejection,
proof that every inner-validation subject was absent from its encoder pretraining, checkpoint provenance,
fixed-step accounting, projection-before-pooling, CPU execution, optional CUDA threshold-neighborhood
parity, and bundle round trips.

Each fold writes an attested `diagnostics.json`; the aggregate summary binds its hashes. It contains
per-subject event counts and F1, paired improved/unchanged/worsened counts, median/IQR/10th percentile,
peak process RSS, peak CUDA allocated memory, SSL wall time, windows/second, embedding windows/second,
trainable parameter count and estimated MACs/FLOPs. Subject F1 is zero when it has truths or predictions
but no match, and is undefined/excluded only when it has neither truths nor predictions. Paired counts
use the union of subjects defined in either the incumbent or candidate diagnostic.

Before each promotion, run the focused tests, a single-fold smoke, the strict five-fold experiment, all
bundle/attestation/dist checks and the complete suite with bytecode and pytest cache disabled. README,
the architecture document and `dist/README.md` must state the exact run key, metrics, feature widths,
device behavior, limitations and reproduction command. Before the first experiment, bootstrap and test
the incumbent registry from the verified current run `035644cf0889a5dd` and F1 `0.5589743590`. The
implementation's older module-level `PROMOTION_F1_FLOOR=0.5432098765` may remain only as a backward-
compatible artifact-format minimum; it must not authorize replacing the current release.

## 8. Stop conditions

Stop a stage and clean its artifacts when it fails its registered gate, violates subject isolation,
exceeds available disk without a bounded cache policy, produces non-finite features, or cannot be
reproduced from its recorded configuration. Stage A receives one fixed feature schema and one five-fold
evaluation before redesign. Stage B receives target-only SSL and its controls before external data is
introduced. Stage C is not started merely because Stage B underperforms; the shared representation
pipeline must first be verified correct.
