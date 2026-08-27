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
import {
  AssetLoadAbortedError,
  isAssetLoadAbortedError,
  isDomAbortError,
  toFetchAbortSignal,
} from '../io/abort.js';

export interface FireRedVadAdapterConfig {
  readonly sampleRate?: number;
  readonly hopSize?: number;
  readonly threshold?: number;
  readonly modelDir?: string;
  readonly modelFilename?: string;
  readonly modelUrl?: string;
  readonly cmvnJsonUrl?: string;
  readonly wasmPaths?: string | Record<string, string>;
  readonly wasmNumThreads?: number;
  readonly cacheAssets?: boolean;
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

interface FireRedVadWorkerLike {
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
}

export interface FireRedVadAdapterOptions {
  readonly workerFactory?: () => FireRedVadWorkerLike;
  readonly now?: () => number;
  readonly signal?: { readonly aborted: boolean } | null;
}

export interface FireRedVadRecentResult {
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

const FIRERED_VAD_INIT_TIMEOUT_MS = 30_000;
const FIRERED_VAD_SUPPORTED_HOP_DURATIONS_MS = [10] as const;
const DEFAULT_FIRERED_MODEL_URL =
  'https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/pretrained_models/onnx_models/fireredvad_stream_vad_with_cache.onnx';

function resolveBundledFireRedCmvnJsonUrl(): string {
  return new URL('./firered-vad/assets/cmvn.json', import.meta.url).href;
}

const DEFAULT_FIRERED_VAD_CONFIG: Required<
  Omit<FireRedVadAdapterConfig, 'assetBaseUrl' | 'scriptUrl' | 'wasmUrl' | 'wasmPaths'>
> = {
  sampleRate: STREAMING_PROCESSING_SAMPLE_RATE,
  hopSize: Math.round((10 / 1000) * STREAMING_PROCESSING_SAMPLE_RATE),
  threshold: 0.5,
  modelDir: '',
  modelFilename: 'fireredvad_stream_vad_with_cache.onnx',
  modelUrl: DEFAULT_FIRERED_MODEL_URL,
  cmvnJsonUrl: resolveBundledFireRedCmvnJsonUrl(),
  wasmNumThreads: 1,
  cacheAssets: true,
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

export interface FireRedVadModelUrls {
  readonly scriptUrl: string;
  readonly wasmUrl: string;
}

export interface ResolvedFireRedVadModelUrls extends FireRedVadModelUrls {
  readonly fallbackScriptUrl: string | null;
  readonly fallbackWasmUrl: string | null;
}

export function resolveSupportedFireRedVadHopSize(
  sampleRate = STREAMING_PROCESSING_SAMPLE_RATE,
  preferredHopSize?: number,
): number {
  const safeSampleRate =
    Number.isFinite(sampleRate) && sampleRate > 0 ? sampleRate : STREAMING_PROCESSING_SAMPLE_RATE;
  const defaultPreferredHopSize = Math.max(
    1,
    Math.round(
      (STREAMING_TIMELINE_CHUNK_FRAMES * safeSampleRate) / STREAMING_PROCESSING_SAMPLE_RATE,
    ),
  );
  const safePreferredHopSize =
    typeof preferredHopSize === 'number' &&
    Number.isFinite(preferredHopSize) &&
    preferredHopSize > 0
      ? Math.round(preferredHopSize)
      : defaultPreferredHopSize;
  const supportedHopSizes = FIRERED_VAD_SUPPORTED_HOP_DURATIONS_MS.map((durationMs) =>
    Math.max(1, Math.round((durationMs / 1000) * safeSampleRate)),
  );

  return supportedHopSizes.reduce((best, candidate) => {
    const bestDistance = Math.abs(best - safePreferredHopSize);
    const candidateDistance = Math.abs(candidate - safePreferredHopSize);
    return candidateDistance < bestDistance ? candidate : best;
  }, supportedHopSizes[0]!);
}

export function resolveDefaultFireRedVadModelUrls(): FireRedVadModelUrls {
  const scriptUrl = DEFAULT_FIRERED_MODEL_URL;
  const wasmUrl = resolveBundledFireRedCmvnJsonUrl();
  return {
    scriptUrl,
    wasmUrl,
  };
}

export function resolveFireRedVadModelUrls(
  config: Pick<
    FireRedVadAdapterConfig,
    'assetBaseUrl' | 'scriptUrl' | 'wasmUrl' | 'fallbackToBundledAssets'
  > = {},
): ResolvedFireRedVadModelUrls {
  const defaults = resolveDefaultFireRedVadModelUrls();
  const assetBaseUrl = config.assetBaseUrl ?? null;
  const scriptUrl =
    config.scriptUrl ??
    (assetBaseUrl
      ? new URL('fireredvad_stream_vad_with_cache.onnx', assetBaseUrl).href
      : defaults.scriptUrl);
  const wasmUrl =
    config.wasmUrl ?? (assetBaseUrl ? new URL('cmvn.json', assetBaseUrl).href : defaults.wasmUrl);

  return {
    scriptUrl,
    wasmUrl,
    fallbackScriptUrl: null,
    fallbackWasmUrl: null,
  };
}

function defaultWorkerFactory(): FireRedVadWorkerLike {
  return new Worker(new URL('./firered-vad-worker.js', import.meta.url), { type: 'module' });
}

function createFireRedVadInitAbortedError(): AssetLoadAbortedError {
  return new AssetLoadAbortedError('firered-vad-init');
}

function isFireRedVadInitAborted(
  error: unknown,
  signal?: { readonly aborted: boolean } | null,
): boolean {
  return isAssetLoadAbortedError(error) || isDomAbortError(error) || Boolean(signal?.aborted);
}

export class FireRedVadAdapter implements StreamingTenVadLike {
  private config: Required<FireRedVadAdapterConfig>;
  private readonly workerFactory: () => FireRedVadWorkerLike;
  private readonly now: () => number;
  private readonly initSignal?: { readonly aborted: boolean } | null;
  private worker: FireRedVadWorkerLike | null = null;
  private messageId = 0;
  private pending = new Map<number, PendingRequest>();
  private listeners = new Set<(event: StreamingTenVadResultEvent) => void>();
  private status: StreamingTenVadStatus['state'] = 'idle';
  private lastError: Error | null = null;
  private recentResults: FireRedVadRecentResult[] = [];
  private latestProbability = 0;
  private latestSpeaking = false;
  private speechRunHops = 0;
  private silenceRunHops = 0;
  private smoothedSpeechActive = false;

  constructor(config: FireRedVadAdapterConfig = {}, options: FireRedVadAdapterOptions = {}) {
    const defaults = resolveDefaultFireRedVadModelUrls();
    const assetBaseUrl = config.assetBaseUrl ?? null;
    const resolvedAssets = resolveFireRedVadModelUrls(config);
    const resolvedWasmPaths = config.wasmPaths ?? '';
    const sampleRate =
      typeof config.sampleRate === 'number' &&
      Number.isFinite(config.sampleRate) &&
      config.sampleRate > 0
        ? config.sampleRate
        : DEFAULT_FIRERED_VAD_CONFIG.sampleRate;
    this.config = {
      ...DEFAULT_FIRERED_VAD_CONFIG,
      ...config,
      sampleRate,
      hopSize: resolveSupportedFireRedVadHopSize(sampleRate, config.hopSize),
      assetBaseUrl:
        assetBaseUrl ?? defaults.scriptUrl.replace(/fireredvad_stream_vad_with_cache\.onnx$/, ''),
      scriptUrl: resolvedAssets.scriptUrl,
      wasmUrl: resolvedAssets.wasmUrl,
      wasmPaths: resolvedWasmPaths,
    };
    this.workerFactory = options.workerFactory ?? defaultWorkerFactory;
    this.now = options.now ?? (() => Date.now());
    this.initSignal = options.signal;
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

  async init(signal?: { readonly aborted: boolean } | null): Promise<void> {
    const abortSignal = signal ?? this.initSignal;
    if (abortSignal?.aborted) {
      throw createFireRedVadInitAbortedError();
    }
    if (this.worker) {
      return;
    }

    this.status = 'initializing';
    this.worker = this.workerFactory();
    this.worker.onmessage = (event) => this.handleMessage(event.data);
    this.worker.onerror = (event) => {
      const message = event?.message || 'FireRed VAD worker error';
      this.fail(new Error(message));
    };

    try {
      if (abortSignal?.aborted) {
        throw createFireRedVadInitAbortedError();
      }
      const resolvedAssets = resolveFireRedVadModelUrls(this.config);
      const wasmPaths =
        typeof this.config.wasmPaths === 'string' && this.config.wasmPaths.length === 0
          ? undefined
          : this.config.wasmPaths;
      const initRequest = this.sendRequest('INIT', {
        hopSize: this.config.hopSize,
        threshold: this.config.threshold,
        modelDir: this.config.modelDir,
        modelFilename: this.config.modelFilename,
        modelUrl: this.config.modelUrl ?? resolvedAssets.scriptUrl,
        cmvnJsonUrl: this.config.cmvnJsonUrl ?? resolvedAssets.wasmUrl,
        wasmPaths,
        wasmNumThreads: this.config.wasmNumThreads,
        cacheAssets: this.config.cacheAssets,
      });
      await this.waitWithTimeout(
        initRequest,
        FIRERED_VAD_INIT_TIMEOUT_MS,
        'FireRed VAD init timed out.',
        abortSignal,
      );
      if (abortSignal?.aborted) {
        throw createFireRedVadInitAbortedError();
      }
      this.status = 'ready';
      this.lastError = null;
    } catch (error) {
      if (isFireRedVadInitAborted(error, abortSignal)) {
        await this.dispose();
        throw createFireRedVadInitAbortedError();
      }
      this.fail(error);
      throw error;
    }
  }

  private fail(error: unknown): void {
    if (!this.worker) {
      return;
    }
    this.status = 'degraded';
    this.lastError = error instanceof Error ? error : new Error(String(error));
    this.rejectPending(this.lastError);
  }

  private rejectPending(error: Error): void {
    for (const [, pending] of this.pending) {
      pending.reject(error);
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
    this.resetTemporalState();

    if (this.worker && this.status === 'ready') {
      await this.sendRequest('RESET', {});
    }
  }

  async dispose(): Promise<void> {
    const worker = this.worker;
    if (worker && this.status === 'ready') {
      try {
        await this.sendRequest('DISPOSE', {});
      } catch {
        // ignore dispose failures
      }
    }
    this.rejectPending(new Error('FireRed VAD adapter disposed.'));
    if (worker) {
      worker.onmessage = null;
      worker.onerror = null;
      worker.terminate();
    }
    this.worker = null;
    this.status = 'idle';
    this.resetTemporalState();
  }

  private resetTemporalState(): void {
    this.recentResults = [];
    this.latestProbability = 0;
    this.latestSpeaking = false;
    this.speechRunHops = 0;
    this.silenceRunHops = 0;
    this.smoothedSpeechActive = false;
  }

  updateConfig(config: Record<string, unknown> = {}): void {
    const previousHopSize = this.config.hopSize;
    const previousThreshold = this.config.threshold;
    const nextSampleRate =
      typeof config.sampleRate === 'number' &&
      Number.isFinite(config.sampleRate) &&
      config.sampleRate > 0
        ? config.sampleRate
        : this.config.sampleRate;
    this.config = {
      ...this.config,
      ...config,
      sampleRate: nextSampleRate,
      hopSize: resolveSupportedFireRedVadHopSize(
        nextSampleRate,
        typeof config.hopSize === 'number' ? config.hopSize : this.config.hopSize,
      ),
    } as Required<FireRedVadAdapterConfig>;
    const workerConfigChanged =
      this.config.hopSize !== previousHopSize || this.config.threshold !== previousThreshold;
    if (workerConfigChanged) {
      this.resetTemporalState();
    }
    if (this.worker && this.status === 'ready' && workerConfigChanged) {
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
        reject(new Error('FireRed VAD worker is not initialized.'));
        return;
      }
      const id = ++this.messageId;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ type, payload, id });
    });
  }

  private waitWithTimeout<T>(
    promise: Promise<T>,
    timeoutMs: number,
    message: string,
    signal?: { readonly aborted: boolean } | null,
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(createFireRedVadInitAbortedError());
        return;
      }

      const timeoutId = setTimeout(() => {
        reject(new Error(message));
      }, timeoutMs);
      const native = toFetchAbortSignal(signal);
      const onAbort = () => {
        clearTimeout(timeoutId);
        native?.removeEventListener('abort', onAbort);
        reject(createFireRedVadInitAbortedError());
      };
      native?.addEventListener('abort', onAbort);

      promise.then(
        (value) => {
          clearTimeout(timeoutId);
          native?.removeEventListener('abort', onAbort);
          if (signal?.aborted) {
            reject(createFireRedVadInitAbortedError());
            return;
          }
          resolve(value);
        },
        (error) => {
          clearTimeout(timeoutId);
          native?.removeEventListener('abort', onAbort);
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
