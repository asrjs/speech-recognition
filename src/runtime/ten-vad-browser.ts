import type { StreamingTenVadLike, StreamingTenVadResultEvent, StreamingTenVadStatus } from './streaming-detector.js';

const TEN_VAD_WORKER_SOURCE = `
let moduleInstance = null;
let vadHandle = 0;
let hopSize = 256;
let threshold = 0.5;
let audioPtr = 0;
let probPtr = 0;
let flagPtr = 0;
let handlePtr = 0;
let accumulator = null;
let accumulatorPos = 0;

self.onmessage = async (event) => {
  const message = event.data;

  try {
    switch (message.type) {
      case 'INIT':
        await handleInit(message.id, message.payload ?? {});
        break;
      case 'PROCESS':
        handleProcess(
          message.payload?.samples ?? new Float32Array(0),
          message.payload?.globalSampleOffset ?? 0,
        );
        break;
      case 'RESET':
        handleReset(message.id);
        break;
      case 'DISPOSE':
        handleDispose(message.id);
        break;
      default:
        break;
    }
  } catch (error) {
    respond({
      type: 'ERROR',
      id: message.id ?? 0,
      payload: String(error instanceof Error ? error.message : error),
    });
  }
};

async function handleInit(id, config) {
  hopSize = config.hopSize || 256;
  threshold = config.threshold || 0.5;
  const scriptUrl = config.scriptUrl;
  const wasmUrl = config.wasmUrl;
  if (!scriptUrl || !wasmUrl) {
    throw new Error('TEN-VAD init requires scriptUrl and wasmUrl.');
  }

  const response = await fetch(scriptUrl);
  if (!response.ok) {
    throw new Error(\`Failed to fetch TEN-VAD script: \${response.status}\`);
  }
  const jsText = await response.text();
  const blobUrl = URL.createObjectURL(
    new Blob([jsText], { type: 'application/javascript' }),
  );

  try {
    const { default: createVADModule } = await import(/* @vite-ignore */ blobUrl);
    moduleInstance = await createVADModule({
      locateFile(file) {
        if (file.endsWith('.wasm')) {
          return wasmUrl;
        }
        return file;
      },
    });
  } finally {
    URL.revokeObjectURL(blobUrl);
  }

  handlePtr = moduleInstance._malloc(4);
  audioPtr = moduleInstance._malloc(hopSize * 2);
  probPtr = moduleInstance._malloc(4);
  flagPtr = moduleInstance._malloc(4);

  const createStatus = moduleInstance._ten_vad_create(handlePtr, hopSize, threshold);
  if (createStatus !== 0) {
    throw new Error(\`ten_vad_create failed with code \${createStatus}\`);
  }
  vadHandle = moduleInstance.HEAP32[handlePtr >> 2];

  const versionPtr = moduleInstance._ten_vad_get_version();
  const version = moduleInstance.UTF8ToString
    ? moduleInstance.UTF8ToString(versionPtr)
    : \`ptr@\${versionPtr}\`;

  accumulator = new Float32Array(hopSize);
  accumulatorPos = 0;

  respond({ type: 'INIT', id, payload: { success: true, version } });
}

function handleProcess(samples, globalSampleOffset) {
  if (!moduleInstance || !vadHandle || !accumulator) return;

  const maxHops = Math.ceil((samples.length + accumulatorPos) / hopSize);
  const probabilities = new Float32Array(maxHops);
  const flags = new Uint8Array(maxHops);
  let hopCount = 0;
  let sampleIndex = 0;
  let firstResultOffset = globalSampleOffset;
  let resultStartSet = false;

  while (sampleIndex < samples.length) {
    while (accumulatorPos < hopSize && sampleIndex < samples.length) {
      accumulator[accumulatorPos++] = samples[sampleIndex++];
    }

    if (accumulatorPos >= hopSize) {
      if (!resultStartSet) {
        firstResultOffset = globalSampleOffset + sampleIndex - hopSize;
        resultStartSet = true;
      }

      for (let index = 0; index < hopSize; index += 1) {
        const clamped = Math.max(-1, Math.min(1, accumulator[index]));
        moduleInstance.HEAP16[(audioPtr >> 1) + index] = Math.round(clamped * 32767);
      }

      const processStatus = moduleInstance._ten_vad_process(
        vadHandle,
        audioPtr,
        hopSize,
        probPtr,
        flagPtr,
      );

      if (processStatus === 0) {
        probabilities[hopCount] = moduleInstance.HEAPF32[probPtr >> 2];
        flags[hopCount] = moduleInstance.HEAP32[flagPtr >> 2];
        hopCount += 1;
      }

      accumulatorPos = 0;
    }
  }

  if (hopCount > 0) {
    const trimmedProbabilities = probabilities.slice(0, hopCount);
    const trimmedFlags = flags.slice(0, hopCount);
    self.postMessage(
      {
        type: 'RESULT',
        payload: {
          probabilities: trimmedProbabilities,
          flags: trimmedFlags,
          globalSampleOffset: firstResultOffset,
          hopCount,
        },
      },
      [trimmedProbabilities.buffer, trimmedFlags.buffer],
    );
  }
}

function handleReset(id) {
  if (accumulator) {
    accumulator.fill(0);
    accumulatorPos = 0;
  }

  if (moduleInstance && vadHandle) {
    moduleInstance._ten_vad_destroy(vadHandle);
    const createStatus = moduleInstance._ten_vad_create(handlePtr, hopSize, threshold);
    if (createStatus === 0) {
      vadHandle = moduleInstance.HEAP32[handlePtr >> 2];
    }
  }

  respond({ type: 'RESET', id, payload: { success: true } });
}

function handleDispose(id) {
  if (moduleInstance && vadHandle) {
    moduleInstance._ten_vad_destroy(vadHandle);
    moduleInstance._free(audioPtr);
    moduleInstance._free(probPtr);
    moduleInstance._free(flagPtr);
    moduleInstance._free(handlePtr);
  }

  moduleInstance = null;
  vadHandle = 0;
  accumulator = null;
  respond({ type: 'DISPOSE', id, payload: { success: true } });
}

function respond(message) {
  self.postMessage(message);
}
`;

export interface TenVadAdapterConfig {
  readonly hopSize?: number;
  readonly threshold?: number;
  readonly confirmationWindowMs?: number;
  readonly hangoverMs?: number;
  readonly minSpeechHops?: number;
  readonly minSpeechRatio?: number;
  readonly minSilenceHops?: number;
  readonly assetBaseUrl?: string;
  readonly scriptUrl?: string;
  readonly wasmUrl?: string;
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
  readonly speaking: boolean;
  readonly createdAt: number;
}

interface PendingRequest {
  readonly resolve: (value: unknown) => void;
  readonly reject: (reason?: unknown) => void;
}

const DEFAULT_TEN_VAD_CONFIG: Required<
  Omit<TenVadAdapterConfig, 'assetBaseUrl' | 'scriptUrl' | 'wasmUrl'>
> = {
  hopSize: 256,
  threshold: 0.5,
  confirmationWindowMs: 192,
  hangoverMs: 320,
  minSpeechHops: 4,
  minSpeechRatio: 0.5,
  minSilenceHops: 5,
};

export function resolveDefaultTenVadAssetUrls(): {
  readonly scriptUrl: string;
  readonly wasmUrl: string;
} {
  const scriptUrl = new URL('./assets/ten-vad/ten_vad.js', import.meta.url).href;
  const wasmUrl = new URL('./assets/ten-vad/ten_vad.wasm', import.meta.url).href;
  return {
    scriptUrl,
    wasmUrl,
  };
}

function defaultWorkerFactory(): TenVadWorkerLike {
  const blobUrl = URL.createObjectURL(
    new Blob([TEN_VAD_WORKER_SOURCE], { type: 'text/javascript' }),
  );
  const worker = new Worker(blobUrl, { type: 'module' });
  URL.revokeObjectURL(blobUrl);
  return worker;
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

  constructor(config: TenVadAdapterConfig = {}, options: TenVadAdapterOptions = {}) {
    const defaults = resolveDefaultTenVadAssetUrls();
    const assetBaseUrl = config.assetBaseUrl ?? null;
    this.config = {
      ...DEFAULT_TEN_VAD_CONFIG,
      ...config,
      assetBaseUrl: assetBaseUrl ?? defaults.scriptUrl.replace(/ten_vad\.js$/, ''),
      scriptUrl: config.scriptUrl ?? (assetBaseUrl ? new URL('ten_vad.js', assetBaseUrl).href : defaults.scriptUrl),
      wasmUrl: config.wasmUrl ?? (assetBaseUrl ? new URL('ten_vad.wasm', assetBaseUrl).href : defaults.wasmUrl),
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
      await this.sendRequest('INIT', {
        hopSize: this.config.hopSize,
        threshold: this.config.threshold,
        scriptUrl: this.config.scriptUrl,
        wasmUrl: this.config.wasmUrl,
      });
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
    for (let index = 0; index < result.hopCount; index += 1) {
      const startFrame = result.globalSampleOffset + index * hopSize;
      const endFrame = startFrame + hopSize;
      const probability = result.probabilities[index];
      const speaking = result.flags[index] === 1;

      this.recentResults.push({
        startFrame,
        endFrame,
        probability,
        speaking,
        createdAt: this.now(),
      });
      this.latestProbability = probability;
      this.latestSpeaking = speaking;
    }

    const maxAgeMs = Math.max(this.config.hangoverMs * 8, 5000);
    const cutoff = this.now() - maxAgeMs;
    this.recentResults = this.recentResults.filter((entry) => entry.createdAt >= cutoff);
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
    const recent = this.recentResults.filter(
      (entry) => entry.endFrame >= startFrame && entry.startFrame <= endFrame,
    );
    let runLength = 0;
    let runStart: number | null = null;
    for (const entry of recent) {
      if (entry.speaking) {
        if (runStart === null) {
          runStart = entry.startFrame;
          runLength = 1;
        } else {
          runLength += 1;
        }

        if (runLength >= this.config.minSpeechHops) {
          return runStart;
        }
      } else {
        runStart = null;
        runLength = 0;
      }
    }

    return null;
  }

  hasRecentSpeech(endFrame: number, windowMs: number, sampleRate: number): boolean {
    const summary = this.getWindowSummary(endFrame, windowMs, sampleRate);
    const speechRatio =
      summary.totalHops > 0 ? summary.speechHopCount / summary.totalHops : 0;
    return (
      summary.speechHopCount >= this.config.minSpeechHops &&
      summary.maxConsecutiveSpeech >= this.config.minSpeechHops &&
      summary.maxProbability >= this.config.threshold &&
      speechRatio >= this.config.minSpeechRatio
    );
  }

  hasRecentSilence(endFrame: number, windowMs: number, sampleRate: number): boolean {
    const summary = this.getWindowSummary(endFrame, windowMs, sampleRate);
    if (summary.totalHops < this.config.minSilenceHops) {
      return false;
    }
    return summary.nonSpeechHopCount >= this.config.minSilenceHops;
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
}
