import type { BrowserRealtimeStarter, BrowserRealtimeStarterSnapshot } from './browser-realtime.js';
import type { StreamingSpeechDetectorEvent } from './streaming-detector.js';

export interface BrowserRealtimeMonitorOptions {
  readonly frameIntervalMs?: number;
}

export interface BrowserRealtimeMonitor {
  subscribe(listener: (snapshot: BrowserRealtimeStarterSnapshot) => void): () => void;
  getSnapshot(): BrowserRealtimeStarterSnapshot | null;
  flush(): void;
  dispose(): void;
}

const DEFAULT_FRAME_INTERVAL_MS = 33;

export function createBrowserRealtimeMonitor(
  source: Pick<BrowserRealtimeStarter, 'getSnapshot' | 'subscribe'>,
  options: BrowserRealtimeMonitorOptions = {},
): BrowserRealtimeMonitor {
  const listeners = new Set<(snapshot: BrowserRealtimeStarterSnapshot) => void>();
  const frameIntervalMs = options.frameIntervalMs ?? DEFAULT_FRAME_INTERVAL_MS;
  let latestSnapshot: BrowserRealtimeStarterSnapshot | null = source.getSnapshot();
  let pendingSnapshot: BrowserRealtimeStarterSnapshot | null = null;
  let hasPendingMetrics = false;
  let timerId = 0;

  const emit = (snapshot: BrowserRealtimeStarterSnapshot | null) => {
    if (!snapshot) {
      return;
    }
    latestSnapshot = snapshot;
    for (const listener of listeners) {
      listener(snapshot);
    }
  };

  const flush = () => {
    if (timerId) {
      clearTimeout(timerId);
      timerId = 0;
    }
    if (!pendingSnapshot && hasPendingMetrics) {
      pendingSnapshot = source.getSnapshot();
    }
    const snapshot = pendingSnapshot ?? latestSnapshot ?? source.getSnapshot();
    pendingSnapshot = null;
    hasPendingMetrics = false;
    emit(snapshot);
  };

  const schedule = () => {
    if (timerId) {
      return;
    }
    timerId = globalThis.setTimeout(flush, frameIntervalMs) as unknown as number;
  };

  const unsubscribeSource = source.subscribe((event: StreamingSpeechDetectorEvent) => {
    if (event.type === 'metrics') {
      hasPendingMetrics = true;
      schedule();
      return;
    }
    pendingSnapshot = source.getSnapshot();
    flush();
  });

  return {
    subscribe(listener: (snapshot: BrowserRealtimeStarterSnapshot) => void): () => void {
      listeners.add(listener);
      if (latestSnapshot) {
        listener(latestSnapshot);
      }
      return () => listeners.delete(listener);
    },
    getSnapshot(): BrowserRealtimeStarterSnapshot | null {
      if (pendingSnapshot) {
        return pendingSnapshot;
      }
      if (latestSnapshot && !hasPendingMetrics) {
        return latestSnapshot;
      }
      return source.getSnapshot();
    },
    flush,
    dispose(): void {
      if (timerId) {
        clearTimeout(timerId);
        timerId = 0;
      }
      pendingSnapshot = null;
      listeners.clear();
      unsubscribeSource();
    },
  };
}
