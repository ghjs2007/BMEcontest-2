export type Capabilities = {
  inference: boolean;
  service?: string;
  runKey?: string;
};

/**
 * Probe the optional local inference bridge (`GET ./api/health`). The static/demo
 * application must keep working when the bridge is absent: failures resolve to
 * `{ inference: false }` instead of throwing.
 */
export async function probeCapabilities(timeoutMs = 1500): Promise<Capabilities> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch('./api/health', { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) return { inference: false };
    const health = await response.json();
    if (health?.status !== 'ok') return { inference: false };
    return { inference: true, service: health.service, runKey: health.run_key };
  } catch {
    return { inference: false };
  }
}
