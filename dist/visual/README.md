# Visualization Workspace Contract

This directory belongs to the team visualization application. It is empty on
purpose: frontend code developed here must consume canonical prediction output
and nothing else.

## What the frontend receives

The only supported interfaces are:

1. `dist/schema/prediction.schema.json` — the stable JSON contract;
2. `dist/examples/example_prediction.json` — a schema-valid safe example
   (`input.source == "synthetic-example"`, no real sensor data);
3. prediction JSON produced by `dist/inference` (`python predict.py ...`) or by
   the canonical `Predictor` API.

The frontend **must not reimplement** thresholding, candidate admission, event
fusion, decoding, or any feature extraction. If a decision is not present in the
prediction JSON, it is not the frontend's job to compute it.

## Prediction JSON structure

Required fields (always present):

| Field | Meaning |
|---|---|
| `schema_version` | Contract version, currently `"1.0"`. |
| `model.name` / `model.run_key` | Always `"event-stack"` and the frozen release key. |
| `input.source` / `input.duration_seconds` | Input descriptor and covered duration. |
| `events[]` | Final decoded eating episodes. |
| `diagnostics.coverage` / `diagnostics.warnings` / `diagnostics.resolved_device` | Coverage fraction, warnings, executor device. |

`events[]` records:

| Key | Meaning |
|---|---|
| `id` | Dense 0-based index; the array is the canonical order. |
| `session_id`, `start_ms`, `end_ms` | Episode interval in raw session timestamps. |
| `duration_s` | `(end_ms - start_ms) / 1000`, exact. |
| `confidence` | Final frozen decoder score in `[0, 1]`. |

Optional debug blocks (requested with `--include-timeline` /
`--include-candidates`; never required):

- `candidates[]`: `session_id`, `start_ms`, `end_ms`, `score`, `admitted` —
  every union candidate and whether admission kept it. Rejected candidates are
  part of the display contract (`admitted: false`), so the UI can show why an
  episode was or was not emitted.
- `gaps[]`: `session_id`, `start_ms`, `end_ms` — unmonitored intervals between
  valid spans. Render as "no data"; never interpolate across them.
- `timeline`: `session_ids`, `macro_windows`, `micro_windows` counts, plus an
  optional `series[]` of per-timestamp points:
  `timestamp_ms`, `macro_probability`, `micro_probability`, `valid`, `gap`.
  Series points with `gap: true` carry `valid: false` and zero probabilities;
  render them as "no data", not as a measured zero.

## Rendering suggestions

Suggested display layers (all driven by prediction JSON, nothing recomputed):

```text
raw activity (from the session file, for reference only)
macro probability          timeline.series[].macro_probability
micro probability          timeline.series[].micro_probability
candidate regions          candidates[] (admitted vs rejected)
verifier decisions         candidates[].admitted
final eating episodes      events[]
```

If `timeline.series` is absent (the default output omits it), degrade to the
event and candidate layers; do not synthesize probabilities.

## Development

- Frontend code lives here (`dist/visual/`); no Python algorithm files are
  allowed in this directory — the release suite enforces that.
- Validate fixtures against the schema:
  `python -m jsonschema -i dist/examples/example_prediction.json dist/schema/prediction.schema.json`
- Real prediction JSON can be produced with the standalone inference package:

```bash
cd dist/inference
python predict.py path/to/collect_data1_2_3.txt --output prediction.json --include-timeline --include-candidates
```
