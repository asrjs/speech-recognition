import type {
  StreamingTenVadLike,
  StreamingTenVadResultEvent,
  StreamingTenVadStatus,
} from './streaming-detector.js';
import {
  STREAMING_PROCESSING_SAMPLE_RATE,
  STREAMING_TIMELINE_CHUNK_FRAMES,
  framesToMilliseconds,
} from './audio-timeline.js';

export interface TenVadAdapterConfig {
  readonly sampleRate?: number;
  readonly hopSize?: number;
  readonly threshold?: number;
  readonly confirmationWindowMs?: number;
  readonly hangoverMs?: number;
  readonly minSpeechDurationMs?: number;
  readonly minSilenceDurationMs?: number;
  readonly speechPaddingMs?: number;
  readonly negativeThresholdOffset?: number;
  // Deprecated duration-unsafe overrides. Prefer duration-based settings above.
  readonly minSpeechHops?: number;
  readonly minSpeechRatio?: number;
  readonly minSilenceHops?: number;
  readonly assetBaseUrl?: string;
  readonly scriptUrl?: string;
  readonly wasmUrl?: string;
  readonly fallbackToBundledAssets?: boolean;
}

interface TenVadWorkerLike {
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
}

export interface TenVadAdapterOptions {
  readonly workerFactory?: () => TenVadWorkerLike;
  readonly now?: () => number;
}

export interface TenVadRecentResult {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly probability: number;
  readonly rawSpeaking: boolean;
  readonly speaking: boolean;
  readonly createdAt: number;
}

interface PendingRequest {
  readonly resolve: (value: unknown) => void;
  readonly reject: (reason?: unknown) => void;
}

const TEN_VAD_INIT_TIMEOUT_MS = 30_000;

const DEFAULT_TEN_VAD_CONFIG: Required<
  Omit<TenVadAdapterConfig, 'assetBaseUrl' | 'scriptUrl' | 'wasmUrl'>
> = {
  sampleRate: STREAMING_PROCESSING_SAMPLE_RATE,
  hopSize: STREAMING_TIMELINE_CHUNK_FRAMES,
  threshold: 0.5,
  confirmationWindowMs: 192,
  hangoverMs: 320,
  minSpeechDurationMs: 240,
  minSilenceDurationMs: 80,
  speechPaddingMs: 48,
  negativeThresholdOffset: 0.15,
  minSpeechHops: 0,
  minSpeechRatio: 0.5,
  minSilenceHops: 0,
  fallbackToBundledAssets: true,
};

export interface TenVadAssetUrls {
  readonly scriptUrl: string;
  readonly wasmUrl: string;
}

export interface ResolvedTenVadAssetUrls extends TenVadAssetUrls {
  readonly fallbackScriptUrl: string | null;
  readonly fallbackWasmUrl: string | null;
}

export function resolveDefaultTenVadAssetUrls(): TenVadAssetUrls {
  const scriptUrl = new URL('./assets/ten-vad/ten_vad.js', import.meta.url).href;
  const wasmUrl = new URL('./assets/ten-vad/ten_vad.wasm', import.meta.url).href;
  return {
    scriptUrl,
    wasmUrl,
  };
}

export function resolveTenVadAssetUrls(
  config: Pick<
    TenVadAdapterConfig,
    'assetBaseUrl' | 'scriptUrl' | 'wasmUrl' | 'fallbackToBundledAssets'
  > = {},
): ResolvedTenVadAssetUrls {
  const defaults = resolveDefaultTenVadAssetUrls();
  const assetBaseUrl = config.assetBaseUrl ?? null;
  const scriptUrl =
    config.scriptUrl ??
    (assetBaseUrl ? new URL('ten_vad.js', assetBaseUrl).href : defaults.scriptUrl);
  const wasmUrl =
    config.wasmUrl ??
    (assetBaseUrl ? new URL('ten_vad.wasm', assetBaseUrl).href : defaults.wasmUrl);
  const shouldFallback =
    (config.fallbackToBundledAssets ?? DEFAULT_TEN_VAD_CONFIG.fallbackToBundledAssets) &&
    (scriptUrl !== defaults.scriptUrl || wasmUrl !== defaults.wasmUrl);

  return {
    scriptUrl,
    wasmUrl,
    fallbackScriptUrl: shouldFallback ? defaults.scriptUrl : null,
    fallbackWasmUrl: shouldFallback ? defaults.wasmUrl : null,
  };
}

function defaultWorkerFactory(): TenVadWorkerLike {
  return new Worker(new URL('./ten-vad-worker.js', import.meta.url), { type: 'module' });
}

export class TenVadAdapter implements StreamingTenVadLike {
  private config: Required<TenVadAdapterConfig>;
  private readonly workerFactory: () => TenVadWorkerLike;
  private readonly now: () => number;
  private worker: TenVadWorkerLike | null = null;
  private messageId = 0;
  private pending = new Map<number, PendingRequest>();
  private listeners = new Set<(event: StreamingTenVadResultEvent) => void>();
  private status: StreamingTenVadStatus['state'] = 'idle';
  private lastError: Error | null = null;
  private recentResults: TenVadRecentResult[] = [];
  private latestProbability = 0;
  private latestSpeaking = false;
  private speechRunHops = 0;
  private silenceRunHops = 0;
  private smoothedSpeechActive = false;

  constructor(config: TenVadAdapterConfig = {}, options: TenVadAdapterOptions = {}) {
    const defaults = resolveDefaultTenVadAssetUrls();
    const assetBaseUrl = config.assetBaseUrl ?? null;
    const resolvedAssets = resolveTenVadAssetUrls(config);
    this.config = {
      ...DEFAULT_TEN_VAD_CONFIG,
      ...config,
      assetBaseUrl: assetBaseUrl ?? defaults.scriptUrl.replace(/ten_vad\.js$/, ''),
      scriptUrl: resolvedAssets.scriptUrl,
      wasmUrl: resolvedAssets.wasmUrl,
    };
    this.workerFactory = options.workerFactory ?? defaultWorkerFactory;
    this.now = options.now ?? (() => Date.now());
  }

  subscribe(listener: (event: StreamingTenVadResultEvent) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(payload: StreamingTenVadResultEvent): void {
    for (const listener of this.listeners) {
      listener(payload);
    }
  }

  async init(): Promise<void> {
    if (this.worker) {
      return;
    }

    this.status = 'initializing';
    this.worker = this.workerFactory();
    this.worker.onmessage = (event) => this.handleMessage(event.data);
    this.worker.onerror = (event) => {
      const message = event?.message || 'TEN-VAD worker error';
      this.fail(new Error(message));
    };

    try {
      const resolvedAssets = resolveTenVadAssetUrls(this.config);
      const initRequest = this.sendRequest('INIT', {
        hopSize: this.config.hopSize,
        threshold: this.config.threshold,
        scriptUrl: resolvedAssets.scriptUrl,
        wasmUrl: resolvedAssets.wasmUrl,
        fallbackScriptUrl: resolvedAssets.fallbackScriptUrl,
        fallbackWasmUrl: resolvedAssets.fallbackWasmUrl,
      });
      await this.waitWithTimeout(initRequest, TEN_VAD_INIT_TIMEOUT_MS, 'TEN-VAD init timed out.');
      this.status = 'ready';
      this.lastError = null;
    } catch (error) {
      this.fail(error);
      throw error;
    }
  }

  private fail(error: unknown): void {
    this.status = 'degraded';
    this.lastError = error instanceof Error ? error : new Error(String(error));
    for (const [, pending] of this.pending) {
      pending.reject(this.lastError);
    }
    this.pending.clear();
  }

  private handleMessage(message: any): void {
    if (message.type === 'RESULT') {
      this.recordResult(message.payload);
      this.emit({
        type: 'result',
        payload: message.payload,
      });
      return;
    }

    if (message.type === 'ERROR') {
      const pending = this.pending.get(message.id);
      if (pending) {
        this.pending.delete(message.id);
        pending.reject(new Error(message.payload));
      }
      this.fail(new Error(message.payload));
      return;
    }

    const pending = this.pending.get(message.id);
    if (pending) {
      this.pending.delete(message.id);
      pending.resolve(message.payload);
    }
  }

  private recordResult(result: any): void {
    const hopSize = this.config.hopSize;
    const { minSpeechHops, minSilenceHops, paddingFrames, negativeThreshold } =
      this.getDerivedTemporalConfig();

    for (let index = 0; index < result.hopCount; index += 1) {
      const startFrame = result.globalSampleOffset + index * hopSize;
      const endFrame = startFrame + hopSize;
      const probability = result.probabilities[index];
      const rawSpeaking = result.flags[index] === 1 || probability >= this.config.threshold;

      if (rawSpeaking) {
        this.speechRunHops += 1;
        this.silenceRunHops = 0;
      } else if (probability <= negativeThreshold) {
        this.silenceRunHops += 1;
        this.speechRunHops = 0;
      } else {
        this.speechRunHops = 0;
        this.silenceRunHops = 0;
      }

      if (!this.smoothedSpeechActive && this.speechRunHops >= minSpeechHops) {
        this.smoothedSpeechActive = true;
      } else if (
        this.smoothedSpeechActive &&
        !rawSpeaking &&
        this.silenceRunHops >= minSilenceHops
      ) {
        this.smoothedSpeechActive = false;
      }

      this.recentResults.push({
        startFrame,
        endFrame,
        probability,
        rawSpeaking,
        speaking: this.smoothedSpeechActive,
        createdAt: this.now(),
      });
      this.latestProbability = probability;
      this.latestSpeaking = this.smoothedSpeechActive;
    }

    const maxAgeMs = Math.max(this.config.hangoverMs * 8, 5000);
    const cutoff = this.now() - maxAgeMs;
    this.recentResults = this.recentResults.filter((entry) => entry.createdAt >= cutoff);

    if (paddingFrames > 0 && this.latestSpeaking) {
      for (
        let index = this.recentResults.length - 1;
        index >= 0 && index >= this.recentResults.length - paddingFrames;
        index -= 1
      ) {
        const entry = this.recentResults[index]!;
        this.recentResults[index] = {
          ...entry,
          speaking: true,
        };
      }
    }
  }

  process(samples: Float32Array, globalSampleOffset: number): boolean {
    if (this.status !== 'ready' || !this.worker) {
      return false;
    }

    const copy = new Float32Array(samples);
    this.worker.postMessage(
      {
        type: 'PROCESS',
        payload: {
          samples: copy,
          globalSampleOffset,
        },
      },
      [copy.buffer],
    );
    return true;
  }

  async reset(): Promise<void> {
    this.recentResults = [];
    this.latestProbability = 0;
    this.latestSpeaking = false;
    this.speechRunHops = 0;
    this.silenceRunHops = 0;
    this.smoothedSpeechActive = false;

    if (this.worker && this.status === 'ready') {
      await this.sendRequest('RESET', {});
    }
  }

  async dispose(): Promise<void> {
    if (this.worker && this.status === 'ready') {
      try {
        await this.sendRequest('DISPOSE', {});
      } catch {
        // ignore dispose failures
      }
    }
    this.worker?.terminate();
    this.worker = null;
    this.status = 'idle';
  }

  updateConfig(config: Record<string, unknown> = {}): void {
    this.config = {
      ...this.config,
      ...config,
    } as Required<TenVadAdapterConfig>;
    if (this.worker && this.status === 'ready') {
      void this.sendRequest('UPDATE_CONFIG', {
        hopSize: this.config.hopSize,
        threshold: this.config.threshold,
      }).catch((error) => {
        this.fail(error);
      });
    }
  }

  getStatus(): StreamingTenVadStatus {
    return {
      state: this.status,
      error: this.lastError?.message ?? null,
      probability: this.latestProbability,
      speaking: this.latestSpeaking,
      threshold: this.config.threshold,
    };
  }

  findFirstSpeechFrame(startFrame: number, endFrame: number): number | null {
    const { paddingFrames } = this.getDerivedTemporalConfig();
    const recent = this.recentResults.filter(
      (entry) => entry.endFrame >= startFrame && entry.startFrame <= endFrame,
    );
    for (const entry of recent) {
      if (entry.speaking) {
        return Math.max(startFrame, entry.startFrame - paddingFrames * this.config.hopSize);
      }
    }

    return null;
  }

  hasRecentSpeech(endFrame: number, windowMs: number, sampleRate: number): boolean {
    const { minSpeechHops } = this.getDerivedTemporalConfig();
    const summary = this.getWindowSummary(endFrame, windowMs, sampleRate);
    const speechRatio = summary.totalHops > 0 ? summary.speechHopCount / summary.totalHops : 0;
    return (
      summary.speechHopCount >= minSpeechHops &&
      summary.maxConsecutiveSpeech >= minSpeechHops &&
      summary.maxProbability >= this.config.threshold &&
      speechRatio >= this.config.minSpeechRatio
    );
  }

  hasRecentSilence(endFrame: number, windowMs: number, sampleRate: number): boolean {
    const { minSilenceHops } = this.getDerivedTemporalConfig();
    const summary = this.getWindowSummary(endFrame, windowMs, sampleRate);
    if (summary.totalHops < minSilenceHops) {
      return false;
    }
    return summary.nonSpeechHopCount >= minSilenceHops;
  }

  getWindowSummary(endFrame: number, windowMs: number, sampleRate: number) {
    const windowFrames = Math.ceil((windowMs / 1000) * sampleRate);
    const startFrame = Math.max(0, endFrame - windowFrames);
    const recent = this.recentResults.filter(
      (entry) => entry.endFrame >= startFrame && entry.startFrame <= endFrame,
    );

    let speechHopCount = 0;
    let nonSpeechHopCount = 0;
    let maxConsecutiveSpeech = 0;
    let consecutiveSpeech = 0;
    let maxProbability = 0;

    for (const entry of recent) {
      if (entry.speaking) {
        speechHopCount += 1;
        consecutiveSpeech += 1;
        maxConsecutiveSpeech = Math.max(maxConsecutiveSpeech, consecutiveSpeech);
      } else {
        nonSpeechHopCount += 1;
        consecutiveSpeech = 0;
      }
      maxProbability = Math.max(maxProbability, entry.probability ?? 0);
    }

    return {
      totalHops: recent.length,
      speechHopCount,
      nonSpeechHopCount,
      maxConsecutiveSpeech,
      maxProbability,
      recent,
    };
  }

  private sendRequest(type: string, payload: unknown): Promise<unknown> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error('TEN-VAD worker is not initialized.'));
        return;
      }
      const id = ++this.messageId;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type, payload, id });
    });
  }

  private waitWithTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(message));
      }, timeoutMs);

      promise.then(
        (value) => {
          clearTimeout(timeoutId);
          resolve(value);
        },
        (error) => {
          clearTimeout(timeoutId);
          reject(error);
        },
      );
    });
  }

  private getDerivedTemporalConfig() {
    const hopDurationMs = framesToMilliseconds(this.config.hopSize, this.config.sampleRate);
    const resolveHopCount = (durationMs: number, deprecatedHops: number) => {
      if (Number.isFinite(deprecatedHops) && deprecatedHops > 0) {
        return Math.max(1, Math.floor(deprecatedHops));
      }
      return Math.max(1, Math.ceil(durationMs / Math.max(1, hopDurationMs)));
    };

    return {
      minSpeechHops: resolveHopCount(this.config.minSpeechDurationMs, this.config.minSpeechHops),
      minSilenceHops: resolveHopCount(this.config.minSilenceDurationMs, this.config.minSilenceHops),
      paddingFrames: Math.max(
        0,
        Math.ceil(this.config.speechPaddingMs / Math.max(1, hopDurationMs)),
      ),
      negativeThreshold: Math.max(0, this.config.threshold - this.config.negativeThresholdOffset),
    };
  }
}
