export type Range = { start_ms: number; end_ms: number };
export type Event = Range & { id: number; session_id: string; duration_s: number; confidence: number };
export type Candidate = Range & { session_id: string; score: number; admitted: boolean };
export type Gap = Range & { session_id: string };
export type Stream = { timestamp_ms: number[]; probability: number[] };
export type SessionTimeline = Range & { session_id: string; macro: Stream; micro: Stream };
export type Prediction = {
  schema_version: string;
  model: { name: string; run_key: string };
  input: { source: string; duration_seconds: number };
  events: Event[];
  candidates?: Candidate[];
  gaps?: Gap[];
  diagnostics: { coverage: number; warnings: string[]; resolved_device?: string };
  timeline?: { session_ids: string[]; macro_windows: number; micro_windows: number; sessions?: SessionTimeline[]; series?: Array<{ session_id: string; timestamp_ms: number; macro_probability: number; micro_probability: number; valid: boolean; gap: boolean }> };
};
export type MotionManifest = {
  telemetry_version: '1.0'; session_id: string; record_format: 'f64_ms_6xf32_le';
  sample_count: number; start_ms: number; end_ms: number; binary_file: string;
  units: { acceleration: 'raw_adc' | 'g'; gyroscope: 'raw_adc' | 'rad/s' | 'deg/s' };
  calibration?: { acceleration_counts_per_g: number; gyroscope_counts_per_rad_s: number; viewer_from_sensor: number[] };
};
export type Imu = { t: Float64Array; ax: Float32Array; ay: Float32Array; az: Float32Array; gx: Float32Array; gy: Float32Array; gz: Float32Array };

export function parsePrediction(value: unknown): Prediction {
  if (!value || typeof value !== 'object') throw new Error('Prediction must be a JSON object.');
  const p = value as Prediction;
  if (p.schema_version !== '1.0' || !p.model?.run_key || !Array.isArray(p.events) || !p.diagnostics || !p.input)
    throw new Error('Unsupported prediction schema. Expected version 1.0.');
  for (const e of p.events) if (!Number.isFinite(e.start_ms) || !Number.isFinite(e.end_ms) || e.end_ms <= e.start_ms) throw new Error('Invalid event timestamps.');
  if (p.timeline && !p.timeline.sessions && p.timeline.series?.length) {
    const grouped = new Map<string, SessionTimeline>();
    for (const point of p.timeline.series) {
      if (!Number.isFinite(point.timestamp_ms) || !Number.isFinite(point.macro_probability) || !Number.isFinite(point.micro_probability) || point.macro_probability < 0 || point.macro_probability > 1 || point.micro_probability < 0 || point.micro_probability > 1) throw new Error('Invalid timeline series.');
      if (point.gap || !point.valid) continue;
      const current = grouped.get(point.session_id) || { session_id: point.session_id, start_ms: point.timestamp_ms, end_ms: point.timestamp_ms, macro: { timestamp_ms: [], probability: [] }, micro: { timestamp_ms: [], probability: [] } };
      current.start_ms = Math.min(current.start_ms, point.timestamp_ms); current.end_ms = Math.max(current.end_ms, point.timestamp_ms);
      current.macro.timestamp_ms.push(point.timestamp_ms); current.macro.probability.push(point.macro_probability);
      current.micro.timestamp_ms.push(point.timestamp_ms); current.micro.probability.push(point.micro_probability);
      grouped.set(point.session_id, current);
    }
    p.timeline.sessions = [...grouped.values()];
  }
  for (const session of p.timeline?.sessions || []) {
    if (!Number.isFinite(session.start_ms) || !Number.isFinite(session.end_ms) || session.end_ms <= session.start_ms) throw new Error('Invalid timeline timestamps.');
    for (const key of ['macro', 'micro'] as const) {
      const s = session[key];
      if (!s || s.timestamp_ms.length !== s.probability.length || s.timestamp_ms.some((t, i) => !Number.isFinite(t) || (i > 0 && t < s.timestamp_ms[i - 1])) || s.probability.some(x => !Number.isFinite(x) || x < 0 || x > 1)) throw new Error('Invalid timeline stream.');
    }
  }
  return p;
}
export function parseMotion(manifestValue: unknown, buffer: ArrayBuffer): { manifest: MotionManifest; imu: Imu } {
  const m = manifestValue as MotionManifest;
  if (!m || m.telemetry_version !== '1.0' || m.record_format !== 'f64_ms_6xf32_le' || !Number.isSafeInteger(m.sample_count) || m.sample_count < 0 || buffer.byteLength !== m.sample_count * 32) throw new Error('Invalid motion telemetry manifest or binary size.');
  const t = new Float64Array(m.sample_count), channels = Array.from({ length: 6 }, () => new Float32Array(m.sample_count));
  const view = new DataView(buffer);
  for (let i = 0; i < m.sample_count; i++) {
    t[i] = view.getFloat64(i * 32, true);
    if (!Number.isFinite(t[i]) || (i > 0 && t[i] < t[i - 1])) throw new Error('IMU timestamps must be ordered and finite.');
    for (let c = 0; c < 6; c++) {
      channels[c][i] = view.getFloat32(i * 32 + 8 + c * 4, true);
      if (!Number.isFinite(channels[c][i])) throw new Error('IMU values must be finite.');
    }
  }
  return { manifest: m, imu: { t, ax: channels[0], ay: channels[1], az: channels[2], gx: channels[3], gy: channels[4], gz: channels[5] } };
}
export function boundsFor(p: Prediction, sessionId: string): Range | null {
  const session = p.timeline?.sessions?.find(s => s.session_id === sessionId);
  if (session) return session;
  const rows = [...p.events, ...(p.candidates || []), ...(p.gaps || [])].filter(r => r.session_id === sessionId);
  if (!rows.length) return null;
  return { start_ms: Math.min(...rows.map(r => r.start_ms)), end_ms: Math.max(...rows.map(r => r.end_ms)) };
}
export function fmtTime(ms: number, detailed = false): string {
  const d = new Date(ms);
  return d.toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', second: detailed ? '2-digit' : undefined, fractionalSecondDigits: detailed ? 1 : undefined });
}
export function duration(ms: number): string {
  const s = Math.max(0, ms / 1000);
  return s < 60 ? `${s.toFixed(1)} s` : s < 3600 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${(s / 3600).toFixed(1)} h`;
}
