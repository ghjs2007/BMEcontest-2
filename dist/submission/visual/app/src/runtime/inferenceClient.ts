import { parsePrediction } from '../data/prediction';
import { parseMotion } from '../data/telemetry';
import type { MotionData, Prediction } from '../data/types';

export type AnalyzeOutcome = {
  prediction: Prediction;
  motions: Record<string, MotionData>;
  warnings: string[];
};

type SessionArtifact = {
  session_id: string;
  source_name?: string;
  sample_count?: number;
  motion?: { manifest: unknown; bin_url: string };
};

/**
 * Client for the local canonical-inference bridge (`dist/inference/serve.py`).
 * It only transports raw TXT files and returns the published prediction contract plus
 * motion telemetry — the browser never implements model logic.
 */
async function ensureOk(response: Response, fallback: string): Promise<any> {
  if (response.ok) return response.json();
  let detail = fallback;
  try {
    const body = await response.json();
    if (body?.error) detail = String(body.error);
  } catch {
    /* keep fallback */
  }
  throw new Error(detail);
}

export async function uploadSessionFile(file: File, relativePath: string | null): Promise<string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/octet-stream', 'X-Session-Name': file.name };
  if (relativePath) headers['X-Relative-Path'] = relativePath.slice(0, 400);
  const response = await fetch('./api/upload', { method: 'POST', headers, body: await file.arrayBuffer() });
  const body = await ensureOk(response, `Upload failed for ${file.name}.`);
  if (!body?.token) throw new Error(`Upload failed for ${file.name}.`);
  return String(body.token);
}

export async function analyzeTokens(tokens: string[], sessionId?: string): Promise<AnalyzeOutcome> {
  const response = await fetch('./api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tokens, session_id: sessionId ?? null }),
  });
  const body = await ensureOk(response, 'Local inference failed.');
  const prediction = parsePrediction(body.prediction);
  const warnings: string[] = Array.isArray(body.warnings) ? body.warnings.map(String) : [];
  const motions: Record<string, MotionData> = {};
  for (const session of (body.sessions || []) as SessionArtifact[]) {
    if (!session?.motion?.bin_url) continue;
    const bin = await fetch(`.${session.motion.bin_url}`);
    if (!bin.ok) { warnings.push(`Telemetry unavailable for ${session.session_id}.`); continue; }
    motions[session.session_id] = parseMotion(session.motion.manifest, await bin.arrayBuffer());
  }
  return { prediction, motions, warnings };
}

export type AnalyzeProgress = (phase: 'preparing' | 'running' | 'timeline', done: number, total: number) => void;

/** Upload raw TXT selections (file picker or folder picker) and run canonical inference. */
export async function analyzeRawSelection(files: File[], folder: boolean, onProgress?: AnalyzeProgress): Promise<AnalyzeOutcome> {
  const total = files.length;
  const tokens: string[] = [];
  for (let i = 0; i < files.length; i++) {
    onProgress?.('preparing', i, total);
    const file = files[i];
    const relative = folder ? (file as File & { webkitRelativePath?: string }).webkitRelativePath || null : null;
    tokens.push(await uploadSessionFile(file, relative));
  }
  onProgress?.('running', total, total);
  const outcome = await analyzeTokens(tokens);
  onProgress?.('timeline', total, total);
  return outcome;
}
