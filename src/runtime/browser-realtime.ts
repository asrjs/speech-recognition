import {
  RealtimeTranscriptionController,
  type RealtimeTranscriptionControllerOptions,
} from './controller.js';
import {
  StreamingSpeechDetector,
  type StreamingSpeechDetectorEvent,
  type StreamingSpeechDetectorOptions,
  type StreamingSpeechDetectorSnapshot,
} from './streaming-detector.js';
import { TenVadAdapter, type TenVadAdapterConfig, type TenVadAdapterOptions } from './ten-vad-browser.js';
import {
  VoiceActivityProbabilityBuffer,
  type VoiceActivityProbabilityBufferOptions,
  type VoiceActivityProbabilityTimelinePoint,
  type VoiceActivityProbabilityBufferWindowSummary,
} from './vad.js';
import {
  resolveStreamingTimelineChunkFrames,
  STREAMING_PROCESSING_SAMPLE_RATE,
} from './audio-timeline.js';

export interface BrowserRealtimeStarterOptions extends StreamingSpeechDetectorOptions {
  readonly bufferDurationSeconds?: number;
  readonly tenVadConfig?: TenVadAdapterConfig;
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
}

export interface BrowserRealtimeStarter {
  readonly detector: StreamingSpeechDetector;
  readonly tenVad: TenVadAdapter;
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
    partial?: Partial<BrowserRealtimeStarterOptions['config']> & {
      readonly profileId?: string;
    },
  ): void;
  getSnapshot(): BrowserRealtimeStarterSnapshot;
  dispose(): Promise<void>;
}

function resolveTenVadConfig(
  options: BrowserRealtimeStarterOptions,
): Required<Pick<TenVadAdapterConfig, 'hopSize' | 'threshold' | 'confirmationWindowMs' | 'hangoverMs'>> {
  const base = options.tenVadConfig ?? {};
  const sampleRate = options.config?.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE;
  return {
    hopSize: base.hopSize ?? resolveStreamingTimelineChunkFrames(sampleRate),
    threshold: base.threshold ?? options.config?.tenVadThreshold ?? 0.5,
    confirmationWindowMs: base.confirmationWindowMs ?? options.config?.tenVadConfirmationWindowMs ?? 192,
    hangoverMs: base.hangoverMs ?? options.config?.tenVadHangoverMs ?? 320,
  };
}

function createVadBuffer(
  options: BrowserRealtimeStarterOptions,
  tenVadConfig: ReturnType<typeof resolveTenVadConfig>,
): VoiceActivityProbabilityBuffer {
  const bufferOptions: VoiceActivityProbabilityBufferOptions = {
    sampleRate: options.config?.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE,
    maxDurationSeconds: (options.config?.ringBufferDurationMs ?? 8000) / 1000,
    hopFrames: tenVadConfig.hopSize,
    speechThreshold: tenVadConfig.threshold,
  };
  return new VoiceActivityProbabilityBuffer(bufferOptions);
}

export function createBrowserRealtimeStarter(
  options: BrowserRealtimeStarterOptions = {},
): BrowserRealtimeStarter {
  const tenVadConfig = resolveTenVadConfig(options);
  const tenVad = new TenVadAdapter(tenVadConfig, options.tenVadOptions);
  const vadBuffer = createVadBuffer(options, tenVadConfig);
  const detector = new StreamingSpeechDetector({
    profileId: options.profileId,
    config: options.config,
    isRealtimeEouModel: options.isRealtimeEouModel,
    tenVadFactory: () => tenVad,
    tenVadOptions: options.tenVadOptions,
  });

  const tenVadUnsubscribe = tenVad.subscribe((event) => {
    if (event.type === 'result') {
      const payload = event.payload as
        | { readonly probabilities?: Float32Array | readonly number[] }
        | null
        | undefined;
      const probabilities = payload?.probabilities;
      if (probabilities instanceof Float32Array || Array.isArray(probabilities)) {
        vadBuffer.appendProbabilities(probabilities);
      }
    }
  });

  const controller =
    options.transcribe || options.controllerOptions
      ? new RealtimeTranscriptionController({
          sampleRate: options.config?.sampleRate ?? STREAMING_PROCESSING_SAMPLE_RATE,
          bufferDurationSeconds: options.bufferDurationSeconds,
          transcribe: options.transcribe ?? (async () => ({
            text: '',
            warnings: [],
            meta: {
              detailLevel: 'text',
              isFinal: false,
            },
          })),
          ...options.controllerOptions,
        })
      : null;

  return {
    detector,
    tenVad,
    vadBuffer,
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
      partial: Partial<BrowserRealtimeStarterOptions['config']> & {
        readonly profileId?: string;
      } = {},
    ): void {
      detector.updateConfig(partial);
    },
    getSnapshot(): BrowserRealtimeStarterSnapshot {
      const snapshot = detector.getSnapshot();
      const waveformPointCount = Math.max(1, Math.floor(snapshot.waveform.minMax.length / 2));
      return {
        ...snapshot,
        vadBuffer: {
          ...vadBuffer.getWindowSummary(
            vadBuffer.getLatestFrame(),
            tenVadConfig.confirmationWindowMs,
            snapshot.sampleRate,
          ),
          timeline: vadBuffer.getTimeline(
            snapshot.waveform.startFrame,
            snapshot.waveform.endFrame,
            waveformPointCount,
          ),
        },
      };
    },
    async dispose(): Promise<void> {
      tenVadUnsubscribe();
      await detector.dispose();
      if (controller) {
        controller.reset();
      }
    },
  };
}
