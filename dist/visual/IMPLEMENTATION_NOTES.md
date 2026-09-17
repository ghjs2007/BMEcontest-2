# Visualization audit and implementation notes

## Current contracts (Phase 1)

- `dist/schema/prediction.schema.json` v1.0 contains final `events`, optional `candidates` with `admitted`, `gaps`, and `timeline` window counts. The promoted `dist/inference` output has no real probability samples in `timeline.series`.
- The canonical `Predictor` already computes macro/micro window probabilities before event decoding. This UI adds only optional, per-session time/value arrays to `timeline`; it does not change candidates, scores, event policy, or the promoted bundle.
- Original `collect_data*.txt` uses millisecond timestamps, `ACC_X/Y/Z`, `GYRO_X/Y/Z`, with 44 PPG columns between timestamps and IMU. The parser passes numeric IMU values through unchanged. The README calls these raw ADC values. `D:\BME_CONTEST\code\prepare_clemson_cad_external.py` documents the training ACC target scale as 4096 counts/g; no source in this repository establishes the watch gyroscope counts-per-deg/s or counts-per-rad/s factor, signed sensor-to-body axis transform, or handedness for a specific raw session.
- Model micro features may gravity-align windows internally. Visualization telemetry reads original IMU values and must not claim to show those transformed model features.
- Battery, live device connection, wear side, and current streaming status are absent from the prediction ABI. The UI displays offline/session state and omits unsupported device claims.

## Stack and files

React + TypeScript + Vite + Three.js. `prediction.json` remains the model contract. Optional `motion.json` v1 + `motion.bin` are visualization-only telemetry: little-endian records of one float64 timestamp (ms) and six float32 raw IMU channel values. A standalone Node CLI exports them from a raw TXT without importing model internals. Binary keeps full available sample resolution; the Canvas timeline downsamples only for drawing. The browser accepts these files independently of Python.

## Physical and coordinate limits

- Real raw ADC can be graphed as counts. Absolute acceleration in g requires explicit ACC calibration. Gyroscope integration requires an explicit gyro conversion to rad/s, signed axis mapping, and sensor frame definition. Without these, real Motion Replay is unavailable rather than fabricated.
- Visualization-only orientation uses a deterministic complementary gravity/gyro filter for calibrated telemetry. It estimates orientation, not absolute position or independent wrist-joint motion. Yaw can drift without a magnetometer. The whole forearm/wrist/hand/watch is one rigid body.
- Internal viewer coordinates: Three.js X = arm longitudinal direction toward the hand, Y = viewer/world up, Z = depth. Synthetic demo sensor axes and sensor-to-viewer mapping are declared in its fixture. A real sensor mapping must be supplied explicitly.
