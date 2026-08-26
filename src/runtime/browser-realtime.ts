import {
  RealtimeTranscriptionController,
  type RealtimeTranscriptionControllerOptions,
} from './controller.js';
import type { RealtimeLatencySummary } from './realtime-latency.js';
import {
  StreamingSpeechDetector,
  type StreamingSpeechDetectorEvent,
  type StreamingSpeechDetectorOptions,
  type StreamingSpeechDetectorSnapshot,
} from './streaming-detector.js';
import {
  DEFAULT_STREAMING_DETECTOR_CONFIG,
  STREAMING_VAD_BACKENDS,
  mergeStreamingConfig,
  resolveStreamingProfileId,
  type StreamingDetectorConfig,
} from './streaming-config.js';
import {
  TenVadAdapter,
  resolveSupportedTenVadHopSize,
  type TenVadAdapterConfig,
  type TenVadAdapterOptions,
} from './ten-vad-browser.js';
import {
  FireRedVadAdapter,
  resolveSupportedFireRedVadHopSize,
  type FireRedVadAdapterConfig,
  type FireRedVadAdapterOptions,
} from './firered-vad-browser.js';
import {
  VoiceActivityProbabilityBuffer,
  type VoiceActivityProbabilityBufferOptions,
  type VoiceActivityProbabilityTimelinePoint,
  type VoiceActivityProbabilityBufferWindowSummary,
} from './vad.js';
import {
  framesToMilliseconds,
  resolveStreamingTimelineChunkFrames,
  STREAMING_PROCESSING_SAMPLE_RATE,
} from './audio-timeline.js';
import type { StreamingDetectorConfigOverrides } from './streaming-config.js';

export interface BrowserRealtimeStarterOptions extends StreamingSpeechDetectorOptions {
  readonly bufferDurationSeconds?: number;
  readonly fireRedVadConfig?: FireRedVadAdapterConfig;
  readonly tenVadConfig?: TenVadAdapterConfig;
  readonly fireRedVadOptions?: FireRedVadAdapterOptions;
  readonly tenVadOptions?: TenVadAdapterOptions;
  readonly controllerOptions?: Omit<
    RealtimeTranscriptionControllerOptions,
    'sampleRate' | 'bufferDurationSeconds' | 'vad'
  >;
  readonly transcribe?: RealtimeTranscriptionControllerOptions['transcribe'];
}

export interface BrowserRealtimeStarterSnapshot extends StreamingSpeechDetectorSnapshot {
  readonly vadBuffer: VoiceActivityProbabilityBufferWindowSummary & {
    readonly timeline: readonly VoiceActivityProbabilityTimelinePoint[];
  };
  readonly plot: BrowserRealtimePlot;
  /**
   * Realtime latency summary from the transcription controller, or null when
   * no controller exists or it was created without the `latency` option.
   */
  readonly latency: RealtimeLatencySummary | null;
}

export interface BrowserRealtimePlotColumn {
  readonly index: number;
  readonly hasData: boolean;
  readonly waveformMin: number;
  readonly waveformMax: number;
  readonly roughEnergy: number;
  readonly preVadRms: number;
  readonly roughSpeechRatio: number;
  readonly roughIsSpeech: boolean;
  readonly roughPass: boolean;
  readonly vadProbability: number;
  readonly vadSpeechRatio: number;
  readonly vadSpeaking: boolean;
  readonly tenVadPass: boolean;
  readonly detectorPass: boolean;
  readonly activeSegment: boolean;
  readonly recentSegment: boolean;
}

export interface BrowserRealtimePlot {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly chunkFrames: number;
  readonly chunkDurationMs: number;
  readonly pointCount: number;
  readonly filledPointCount: number;
  readonly padPoints: number;
  readonly livePointIndex: number;
  readonly gateMode: string;
  readonly columns: readonly BrowserRealtimePlotColumn[];
}

export interface BrowserRealtimeStarter {
  readonly detector: StreamingSpeechDetector;
  readonly tenVad: TenVadAdapter | FireRedVadAdapter;
  readonly vadBuffer: VoiceActivityProbabilityBuffer;
  readonly controller: RealtimeTranscriptionController | null;
  subscribe(listener: (event: StreamingSpeechDetectorEvent) => void): () => void;
  start(options?: { readonly sampleRate?: number }): Promise<void>;
  processChunk(
    chunk: Float32Array,
    meta?: { readonly startFrame?: number; readonly endFrame?: number },
  ): void;
  flush(reason?: string): ReturnType<StreamingSpeechDetector['flush']>;
  stop(options?: { readonly flush?: boolean }): ReturnType<StreamingSpeechDetector['stop']>;
  updateConfig(
    partial?: StreamingDetectorConfigOverrides & {
      readonly profileId?: string;
    },
  ): void;
  getSnapshot(): BrowserRealtimeStarterSnapshot;
  dispose(): Promise<void>;
}

function resolveTenVadConfig(
  resolvedConfig: StreamingDetectorConfig,
  options: BrowserRealtimeStarterOptions,
): TenVadAdapterConfig | FireRedVadAdapterConfig {
  const base =
    resolvedConfig.vadBackend === STREAMING_VAD_BACKENDS.FIRERED_VAD
      ? options.fireRedVadConfig ?? {}
      : options.tenVadConfig ?? {};
  const sampleRate = resolvedConfig.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE;
  const hopDurationMs = resolvedConfig.vadHopDurationMs;
  const preferredHopSize = resolveStreamingTimelineChunkFrames(sampleRate, hopDurationMs);
  const resolveSupportedHopSize =
    resolvedConfig.vadBackend === STREAMING_VAD_BACKENDS.FIRERED_VAD
      ? resolveSupportedFireRedVadHopSize
      : resolveSupportedTenVadHopSize;
  return {
    ...base,
    sampleRate,
    hopSize: resolveSupportedHopSize(sampleRate, base.hopSize ?? preferredHopSize),
    threshold: base.threshold ?? resolvedConfig.tenVadThreshold ?? 0.5,
    confirmationWindowMs:
      base.confirmationWindowMs ??
      resolvedConfig.tenVadConfirmationWindowMs ??
      192,
    hangoverMs: base.hangoverMs ?? resolvedConfig.tenVadHangoverMs ?? 320,
    minSpeechDurationMs:
      base.minSpeechDurationMs ??
      resolvedConfig.tenVadMinSpeechDurationMs ??
      240,
    minSilenceDurationMs:
      base.minSilenceDurationMs ??
      resolvedConfig.tenVadMinSilenceDurationMs ??
      80,
    speechPaddingMs:
      base.speechPaddingMs ??
      resolvedConfig.tenVadSpeechPaddingMs ??
      48,
  };
}

function createVadBuffer(
  resolvedConfig: StreamingDetectorConfig,
  tenVadConfig: ReturnType<typeof resolveTenVadConfig>,
): VoiceActivityProbabilityBuffer {
  const visualBucketFrames = resolveStreamingTimelineChunkFrames(
    resolvedConfig.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE,
    resolvedConfig.vadVisualBucketDurationMs,
  );
  const bufferOptions: VoiceActivityProbabilityBufferOptions = {
    sampleRate: resolvedConfig.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE,
    maxDurationSeconds:
      (resolvedConfig.ringBufferDurationMs ??
        DEFAULT_STREAMING_DETECTOR_CONFIG.ringBufferDurationMs) / 1000,
    hopFrames: visualBucketFrames,
    speechThreshold: tenVadConfig.threshold ?? 0.5,
  };
  return new VoiceActivityProbabilityBuffer(bufferOptions);
}

function appendAlignedVadProbabilities(
  vadBuffer: VoiceActivityProbabilityBuffer,
  bucketMaxima: Map<number, number>,
  probabilities: Float32Array | readonly number[],
  sourceStartFrame: number,
  sourceHopFrames: number,
): void {
  const targetBucketFrames = vadBuffer.hopFrames;
  const safeSourceHopFrames = Math.max(1, Math.floor(sourceHopFrames));

  for (let index = 0; index < probabilities.length; index += 1) {
    const sourceFrame = sourceStartFrame + index * safeSourceHopFrames;
    const bucketIndex = Math.max(0, Math.floor(sourceFrame / targetBucketFrames));
    const probability = Math.max(
      bucketMaxima.get(bucketIndex) ?? 0,
      probabilities[index] ?? 0,
    );
    bucketMaxima.set(bucketIndex, probability);
    vadBuffer.appendProbabilitiesAtFrame([probability], bucketIndex * targetBucketFrames);
  }

  const baseEntry = vadBuffer.getBaseEntry();
  for (const bucketIndex of bucketMaxima.keys()) {
    if (bucketIndex < baseEntry) {
      bucketMaxima.delete(bucketIndex);
    }
  }
}

function buildAlignedPlot(
  snapshot: StreamingSpeechDetectorSnapshot,
  vadTimeline: readonly VoiceActivityProbabilityTimelinePoint[],
): BrowserRealtimePlot {
  const chunkFrames = resolveStreamingTimelineChunkFrames(
    snapshot.sampleRate,
    snapshot.config.chunkDurationMs,
  );
  const pointCount = Math.max(1, Math.floor(snapshot.waveform.minMax.length / 2));
  const filledPointCount = Math.min(
    pointCount,
    Math.max(0, Math.ceil((snapshot.waveform.endFrame - snapshot.waveform.startFrame) / chunkFrames)),
  );
  const padPoints = Math.max(0, pointCount - filledPointCount);
  const displaySpanFrames = pointCount * chunkFrames;
  const startFrame = snapshot.waveform.endFrame - displaySpanFrames;
  const columnStartFrames = Array.from(
    { length: pointCount },
    (_, index) => startFrame + index * chunkFrames,
  );
  const columnOverlapsSegment = (
    columnIndex: number,
    segment: { readonly startFrame: number; readonly endFrame: number } | null | undefined,
  ): boolean => {
    if (!segment) {
      return false;
    }
    const columnStartFrame = columnStartFrames[columnIndex]!;
    const columnEndFrame = columnStartFrame + chunkFrames;
    return segment.endFrame > columnStartFrame && segment.startFrame < columnEndFrame;
  };
  const computeDetectorPass = (roughPass: boolean): boolean => {
    switch (snapshot.gate.effectiveMode) {
      case 'ten-vad-only':
        // In FireRed-first mode, hop-level VAD is only a segment proposal.
        // Accepted speech is shown through active/recent segment spans instead.
        return false;
      case 'rough-and-ten-vad':
        return false;
      default:
        return roughPass;
    }
  };
  const columns: BrowserRealtimePlotColumn[] = Array.from({ length: pointCount }, (_, index) => ({
    index,
    hasData: false,
    waveformMin: 0,
    waveformMax: 0,
    roughEnergy: 0,
    preVadRms: 0,
    roughSpeechRatio: 0,
    roughIsSpeech: false,
    roughPass: false,
    vadProbability: 0,
    vadSpeechRatio: 0,
    vadSpeaking: false,
    tenVadPass: false,
    detectorPass: false,
    activeSegment: false,
    recentSegment: false,
  }));

  for (let sourceIndex = 0; sourceIndex < filledPointCount; sourceIndex += 1) {
    const targetIndex = padPoints + sourceIndex;
    const column = columns[targetIndex]!;
    const roughPoint = snapshot.rough.timeline?.[sourceIndex];
    const vadPoint = vadTimeline[sourceIndex];
    columns[targetIndex] = {
      ...column,
      hasData: true,
      waveformMin: snapshot.waveform.minMax[sourceIndex * 2] ?? 0,
      waveformMax: snapshot.waveform.minMax[sourceIndex * 2 + 1] ?? 0,
      roughEnergy: roughPoint?.energy ?? 0,
      preVadRms: roughPoint?.rawEnergy ?? roughPoint?.energy ?? 0,
      roughSpeechRatio: roughPoint?.speechRatio ?? 0,
      roughIsSpeech: roughPoint?.isSpeech ?? false,
      roughPass: roughPoint?.isSpeech ?? false,
      vadProbability: vadPoint?.probability ?? 0,
      vadSpeechRatio: vadPoint?.speechRatio ?? 0,
      vadSpeaking: vadPoint?.speaking ?? false,
      tenVadPass: vadPoint?.speaking ?? false,
      detectorPass: computeDetectorPass(
        roughPoint?.isSpeech ?? false,
      ),
      activeSegment: columnOverlapsSegment(targetIndex, snapshot.activeSegment),
      recentSegment: snapshot.recentSegments.some((segment) =>
        columnOverlapsSegment(targetIndex, segment),
      ),
    };
  }

  return {
    startFrame,
    endFrame: startFrame + displaySpanFrames,
    chunkFrames,
    chunkDurationMs: framesToMilliseconds(chunkFrames, snapshot.sampleRate),
    pointCount,
    filledPointCount,
    padPoints,
    livePointIndex: filledPointCount > 0 ? padPoints + filledPointCount - 1 : -1,
    gateMode: snapshot.gate.effectiveMode,
    columns,
  };
}

export function createBrowserRealtimeStarter(
  options: BrowserRealtimeStarterOptions = {},
): BrowserRealtimeStarter {
  if (options.controllerOptions && !options.transcribe) {
    throw new Error(
      'createBrowserRealtimeStarter requires transcribe when controllerOptions are provided.',
    );
  }
  let activeProfileId = options.profileId ?? resolveStreamingProfileId(options.isRealtimeEouModel === true);
  let resolvedConfig = mergeStreamingConfig(
    activeProfileId,
    options.config,
  );
  let tenVadConfig = resolveTenVadConfig(resolvedConfig, options);
  let vadBuffer = createVadBuffer(resolvedConfig, tenVadConfig);
  let vadBucketMaxima = new Map<number, number>();
  let tenVadUnsubscribe: (() => void) | null = null;

  const resolveTenVadOptions = (
    config: StreamingDetectorConfig,
  ): TenVadAdapterOptions | FireRedVadAdapterOptions | undefined =>
    config.vadBackend === STREAMING_VAD_BACKENDS.FIRERED_VAD
      ? options.fireRedVadOptions
      : options.tenVadOptions;

  let tenVadOptions = resolveTenVadOptions(resolvedConfig);
  let tenVad: TenVadAdapter | FireRedVadAdapter;

  const subscribeTenVadBuffer = (
    adapter: TenVadAdapter | FireRedVadAdapter,
  ): void => {
    tenVadUnsubscribe?.();
    tenVadUnsubscribe = adapter.subscribe((event) => {
      if (event.type === 'result') {
        const payload = event.payload as
          | {
              readonly probabilities?: Float32Array | readonly number[];
              readonly globalSampleOffset?: number;
            }
          | null
          | undefined;
        const probabilities = payload?.probabilities;
        if (probabilities instanceof Float32Array || Array.isArray(probabilities)) {
          const sourceHopFrames =
            tenVadConfig.hopSize ??
            resolveStreamingTimelineChunkFrames(
              resolvedConfig.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE,
              resolvedConfig.vadHopDurationMs,
            );
          appendAlignedVadProbabilities(
            vadBuffer,
            vadBucketMaxima,
            probabilities,
            typeof payload?.globalSampleOffset === 'number'
              ? payload.globalSampleOffset
              : vadBuffer.getLatestFrame(),
            sourceHopFrames,
          );
        }
      }
    });
  };

  let tenVadBackend = resolvedConfig.vadBackend;

  const createTenVad = (config: Record<string, unknown> = {}): TenVadAdapter | FireRedVadAdapter => {
    const vadBackend =
      config.vadBackend === STREAMING_VAD_BACKENDS.TEN_VAD ||
      config.vadBackend === STREAMING_VAD_BACKENDS.FIRERED_VAD
        ? config.vadBackend
        : resolvedConfig.vadBackend;
    const adapterConfigSource = {
      ...resolvedConfig,
      vadBackend,
    };
    tenVadConfig = resolveTenVadConfig(adapterConfigSource, options);
    tenVadOptions = resolveTenVadOptions(adapterConfigSource);
    const adapter =
      vadBackend === STREAMING_VAD_BACKENDS.FIRERED_VAD
        ? new FireRedVadAdapter(tenVadConfig as FireRedVadAdapterConfig, tenVadOptions as FireRedVadAdapterOptions)
        : new TenVadAdapter(tenVadConfig as TenVadAdapterConfig, tenVadOptions as TenVadAdapterOptions);
    tenVad = adapter;
    tenVadBackend = vadBackend;
    subscribeTenVadBuffer(adapter);
    return adapter;
  };

  tenVad = createTenVad({ vadBackend: resolvedConfig.vadBackend });
  const detector = new StreamingSpeechDetector({
    profileId: options.profileId,
    config: resolvedConfig,
    isRealtimeEouModel: options.isRealtimeEouModel,
    tenVadFactory: (config) => {
      if (tenVad && config.vadBackend === resolvedConfig.vadBackend && tenVadBackend === config.vadBackend) {
        return tenVad;
      }
      return createTenVad(config);
    },
    tenVadOptions,
  });

  const controller = options.transcribe
    ? new RealtimeTranscriptionController({
        sampleRate: resolvedConfig.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE,
        bufferDurationSeconds: options.bufferDurationSeconds,
        transcribe: options.transcribe,
        ...options.controllerOptions,
      })
    : null;

  return {
    detector,
    get tenVad() {
      return tenVad;
    },
    get vadBuffer() {
      return vadBuffer;
    },
    controller,
    subscribe(listener: (event: StreamingSpeechDetectorEvent) => void): () => void {
      return detector.subscribe(listener);
    },
    async start({ sampleRate }: { readonly sampleRate?: number } = {}): Promise<void> {
      await detector.start({ sampleRate });
      if (controller) {
        controller.reset();
      }
      await tenVad.reset();
      vadBucketMaxima.clear();
      vadBuffer.reset();
    },
    processChunk(
      chunk: Float32Array,
      meta: { readonly startFrame?: number; readonly endFrame?: number } = {},
    ): void {
      detector.processChunk(chunk, meta);
    },
    flush(reason = 'manual') {
      return detector.flush(reason);
    },
    stop(options: { readonly flush?: boolean } = {}) {
      return detector.stop(options);
    },
    updateConfig(
      partial: StreamingDetectorConfigOverrides & {
        readonly profileId?: string;
      } = {},
    ): void {
      activeProfileId = partial.profileId ?? activeProfileId;
      const nextOverrides: Record<string, unknown> = {
        ...resolvedConfig,
        ...partial,
      };
      delete nextOverrides.profileId;
      if (
        Object.prototype.hasOwnProperty.call(partial, 'vadBackend')
        && !Object.prototype.hasOwnProperty.call(partial, 'vadHopDurationMs')
      ) {
        delete nextOverrides.vadHopDurationMs;
      }
      if (
        Object.prototype.hasOwnProperty.call(partial, 'vadBackend')
        && !Object.prototype.hasOwnProperty.call(partial, 'vadVisualBucketDurationMs')
      ) {
        delete nextOverrides.vadVisualBucketDurationMs;
      }
      const previousConfig = resolvedConfig;
      resolvedConfig = mergeStreamingConfig(
        activeProfileId,
        nextOverrides as StreamingDetectorConfigOverrides,
      );
      tenVadConfig = resolveTenVadConfig(resolvedConfig, options);
      tenVadOptions = resolveTenVadOptions(resolvedConfig);
      if (
        previousConfig.sampleRate !== resolvedConfig.sampleRate ||
        previousConfig.ringBufferDurationMs !== resolvedConfig.ringBufferDurationMs ||
        previousConfig.vadVisualBucketDurationMs !== resolvedConfig.vadVisualBucketDurationMs ||
        previousConfig.tenVadThreshold !== resolvedConfig.tenVadThreshold
      ) {
        vadBuffer = createVadBuffer(resolvedConfig, tenVadConfig);
        vadBucketMaxima = new Map<number, number>();
      }
      detector.updateConfig(partial);
      if (
        !resolvedConfig.tenVadEnabled
        && (previousConfig.tenVadEnabled || previousConfig.vadBackend !== resolvedConfig.vadBackend)
      ) {
        const previousTenVad = tenVad;
        createTenVad({ vadBackend: resolvedConfig.vadBackend });
        void previousTenVad.dispose().catch(() => undefined);
      }
    },
    getSnapshot(): BrowserRealtimeStarterSnapshot {
      const snapshot = detector.getSnapshot();
      const waveformPointCount = Math.max(1, Math.floor(snapshot.waveform.minMax.length / 2));
      const vadTimeline = vadBuffer.getTimeline(
        snapshot.waveform.startFrame,
        snapshot.waveform.endFrame,
        waveformPointCount,
      );
      return {
        ...snapshot,
        vadBuffer: {
          ...vadBuffer.getWindowSummary(
            vadBuffer.getLatestFrame(),
            snapshot.config.tenVadConfirmationWindowMs,
            snapshot.sampleRate,
          ),
          timeline: vadTimeline,
        },
        plot: buildAlignedPlot(snapshot, vadTimeline),
        latency: controller?.getState().latency ?? null,
      };
    },
    async dispose(): Promise<void> {
      tenVadUnsubscribe?.();
      await detector.dispose();
      if (controller) {
        controller.reset();
      }
    },
  };
}
