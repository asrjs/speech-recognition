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
  /** Optional deadline for browser measurements that may wait for a future GC. */
  readonly timeoutMs?: number;
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

  const timeoutMs = options.timeoutMs;
  if (timeoutMs !== undefined && (!Number.isFinite(timeoutMs) || timeoutMs <= 0)) {
    throw new RangeError('Memory measurement timeoutMs must be positive and finite.');
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;
  try {
    const pending = measure.call(browserPerformance);
    const result = timeoutMs === undefined ? await pending : await Promise.race([
      pending,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => {
          timedOut = true;
          reject(new Error('Memory measurement timed out.'));
        }, timeoutMs);
      }),
    ]);
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
    return unavailableSnapshot(capturedAt, timedOut ? 'timeout' : 'measurement-failed');
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
