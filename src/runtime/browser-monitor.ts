import type {
  BrowserRealtimeStarter,
  BrowserRealtimeStarterSnapshot,
} from './browser-realtime.js';
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
  let pendingSnapshot: BrowserRealtimeStarterSnapshot | null = latestSnapshot;
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
    const snapshot = pendingSnapshot ?? source.getSnapshot();
    pendingSnapshot = null;
    emit(snapshot);
  };

  const schedule = () => {
    if (timerId) {
      return;
    }
    timerId = globalThis.setTimeout(flush, frameIntervalMs) as unknown as number;
  };

  const unsubscribeSource = source.subscribe((event: StreamingSpeechDetectorEvent) => {
    pendingSnapshot = source.getSnapshot();
    if (event.type === 'metrics') {
      schedule();
      return;
    }
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
      return latestSnapshot ?? pendingSnapshot ?? source.getSnapshot();
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
