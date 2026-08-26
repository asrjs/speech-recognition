import type { BenchmarkMemorySnapshot } from './runtime/benchmark.js';

interface BrowserMemoryResultLike {
  readonly bytes?: unknown;
}

interface BrowserMemoryPerformanceLike {
  measureUserAgentSpecificMemory?: () => Promise<BrowserMemoryResultLike>;
}

export interface MeasureBrowserMemoryOptions {
  readonly performance?: BrowserMemoryPerformanceLike | null;
  readonly now?: () => string;
}

function unavailableSnapshot(
  capturedAt: string,
  reason: NonNullable<BenchmarkMemorySnapshot['reason']>,
): BenchmarkMemorySnapshot {
  return {
    capturedAt,
    source: 'unavailable',
    scope: 'unavailable',
    bytes: null,
    reason,
  };
}

/**
 * Measures browser process memory when the User-Agent Specific Memory API is
 * available. It deliberately does not fall back to deprecated JS heap
 * estimates, which are not comparable across browsers or runs.
 */
export async function measureBrowserMemory(
  options: MeasureBrowserMemoryOptions = {},
): Promise<BenchmarkMemorySnapshot> {
  const capturedAt = options.now?.() ?? new Date().toISOString();
  const browserPerformance =
    options.performance === undefined
      ? (globalThis.performance as BrowserMemoryPerformanceLike | undefined)
      : options.performance;
  const measure = browserPerformance?.measureUserAgentSpecificMemory;

  if (typeof measure !== 'function') {
    return unavailableSnapshot(capturedAt, 'unsupported');
  }

  try {
    const result = await measure.call(browserPerformance);
    const bytes = Number(result?.bytes);
    if (!Number.isFinite(bytes) || bytes < 0) {
      return unavailableSnapshot(capturedAt, 'invalid-result');
    }
    return {
      capturedAt,
      source: 'measure-user-agent-specific-memory',
      scope: 'process',
      bytes,
    };
  } catch {
    return unavailableSnapshot(capturedAt, 'measurement-failed');
  }
}
