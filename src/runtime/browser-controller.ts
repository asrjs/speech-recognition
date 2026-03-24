import {
  createBrowserRealtimeMonitor,
  type BrowserRealtimeMonitor,
  type BrowserRealtimeMonitorOptions,
} from './browser-monitor.js';
import {
  createBrowserRealtimeStarter,
  type BrowserRealtimeStarter,
  type BrowserRealtimeStarterOptions,
} from './browser-realtime.js';
import {
  startMicrophoneCapture,
  type BrowserMicrophoneCaptureHandle,
  type BrowserMicrophoneCaptureOptions,
  type MicrophoneAudioChunk,
} from './capture.js';
import {
  STREAMING_PROCESSING_SAMPLE_RATE,
  STREAMING_TIMELINE_CHUNK_FRAMES,
  STREAMING_TIMELINE_CHUNK_MS,
  resolveStreamingTimelineChunkFrames,
} from './audio-timeline.js';
import type {
  StreamingDetectorConfig,
  StreamingDetectorConfigOverrides,
} from './streaming-config.js';
import { mergeStreamingConfig, resolveStreamingProfileId } from './streaming-config.js';

export type BrowserRealtimeMicrophoneMode = 'manual' | 'speech-detect';

export interface BrowserRealtimeCaptureInfo {
  readonly inputSampleRate: number | null;
  readonly contextSampleRate: number | null;
  readonly processingSampleRate: number;
  readonly chunkFrames: number;
  readonly chunkDurationMs: number;
  readonly deviceLabel: string;
}

export interface BrowserRealtimeMicrophoneUtterance {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
  readonly reason: string;
  readonly durationSeconds: number;
  readonly sourceLabel: string;
  readonly metadata: unknown;
}

export interface BrowserRealtimeMicrophoneControllerState {
  readonly isMicActive: boolean;
  readonly micStatus: string;
  readonly micError: string;
  readonly micMode: BrowserRealtimeMicrophoneMode;
  readonly captureInfo: BrowserRealtimeCaptureInfo;
}

export interface BrowserRealtimeMicrophoneControllerOptions extends Omit<
  BrowserRealtimeStarterOptions,
  'profileId' | 'config'
> {
  readonly profileId?: string;
  readonly config?: StreamingDetectorConfigOverrides | StreamingDetectorConfig;
  readonly micMode?: BrowserRealtimeMicrophoneMode;
  readonly frameIntervalMs?: number;
  readonly microphoneConstraints?: MediaTrackConstraints;
  readonly onUtterance?: (utterance: BrowserRealtimeMicrophoneUtterance) => void;
  readonly onStatus?: (status: string) => void;
  readonly createStarter?: (options?: BrowserRealtimeStarterOptions) => BrowserRealtimeStarter;
  readonly createMonitor?: (
    source: Pick<BrowserRealtimeStarter, 'getSnapshot' | 'subscribe'>,
    options?: BrowserRealtimeMonitorOptions,
  ) => BrowserRealtimeMonitor;
  readonly startCapture?: (
    options: BrowserMicrophoneCaptureOptions,
  ) => Promise<BrowserMicrophoneCaptureHandle>;
}

export interface BrowserRealtimeMicrophoneController {
  readonly monitor: BrowserRealtimeMonitor;
  readonly starter: BrowserRealtimeStarter;
  subscribe(listener: (state: BrowserRealtimeMicrophoneControllerState) => void): () => void;
  getState(): BrowserRealtimeMicrophoneControllerState;
  configure(options?: {
    readonly profileId?: string;
    readonly config?: StreamingDetectorConfigOverrides | StreamingDetectorConfig;
  }): void;
  setMicMode(mode: BrowserRealtimeMicrophoneMode): void;
  start(): Promise<void>;
  stop(options?: { readonly flush?: boolean }): Promise<void>;
  flush(reason?: string): void;
  dispose(): Promise<void>;
}

const DEFAULT_FRAME_INTERVAL_MS = 33;

function concatFloat32Chunks(chunks: readonly Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function createInitialCaptureInfo(): BrowserRealtimeCaptureInfo {
  return {
    inputSampleRate: null,
    contextSampleRate: null,
    processingSampleRate: STREAMING_PROCESSING_SAMPLE_RATE,
    chunkFrames: STREAMING_TIMELINE_CHUNK_FRAMES,
    chunkDurationMs: STREAMING_TIMELINE_CHUNK_MS,
    deviceLabel: '',
  };
}

function createInitialState(
  micMode: BrowserRealtimeMicrophoneMode,
): BrowserRealtimeMicrophoneControllerState {
  return {
    isMicActive: false,
    micStatus: 'Microphone idle',
    micError: '',
    micMode,
    captureInfo: createInitialCaptureInfo(),
  };
}

export function createBrowserRealtimeMicrophoneController(
  options: BrowserRealtimeMicrophoneControllerOptions = {},
): BrowserRealtimeMicrophoneController {
  const createStarter = options.createStarter ?? createBrowserRealtimeStarter;
  const createMonitor = options.createMonitor ?? createBrowserRealtimeMonitor;
  const startCapture = options.startCapture ?? startMicrophoneCapture;
  const listeners = new Set<(state: BrowserRealtimeMicrophoneControllerState) => void>();
  const onUtterance = options.onUtterance ?? (() => undefined);
  const onStatus = options.onStatus ?? (() => undefined);
  let micMode: BrowserRealtimeMicrophoneMode = options.micMode ?? 'speech-detect';
  let profileId =
    options.profileId ?? resolveStreamingProfileId(options.isRealtimeEouModel === true);
  let resolvedConfig = mergeStreamingConfig(profileId, options.config);
  let captureHandle: BrowserMicrophoneCaptureHandle | null = null;
  let sampleRate = STREAMING_PROCESSING_SAMPLE_RATE;
  let manualChunks: Float32Array[] = [];
  let state = createInitialState(micMode);

  const starter = createStarter({
    ...options,
    profileId,
    config: resolvedConfig,
  });
  const monitor = createMonitor(starter, {
    frameIntervalMs: options.frameIntervalMs ?? DEFAULT_FRAME_INTERVAL_MS,
  });

  const emit = (): void => {
    for (const listener of listeners) {
      listener(state);
    }
  };

  const setState = (
    patch:
      | Partial<BrowserRealtimeMicrophoneControllerState>
      | ((
          current: BrowserRealtimeMicrophoneControllerState,
        ) => BrowserRealtimeMicrophoneControllerState),
  ): void => {
    state =
      typeof patch === 'function'
        ? patch(state)
        : {
            ...state,
            ...patch,
          };
    emit();
  };

  const updateStatus = (micStatus: string, appStatus = micStatus): void => {
    setState({ micStatus });
    onStatus(appStatus);
  };

  const updateCaptureInfo = (
    partial: Partial<BrowserRealtimeCaptureInfo> = {},
  ): BrowserRealtimeCaptureInfo => {
    const nextCaptureInfo = {
      ...state.captureInfo,
      ...partial,
    };
    setState({ captureInfo: nextCaptureInfo });
    return nextCaptureInfo;
  };

  const getConfiguredChunkDurationMs = (): number =>
    resolvedConfig.chunkDurationMs ?? STREAMING_TIMELINE_CHUNK_MS;

  const getConfiguredChunkFrames = (): number =>
    resolveStreamingTimelineChunkFrames(
      STREAMING_PROCESSING_SAMPLE_RATE,
      getConfiguredChunkDurationMs(),
    );

  const handleChunk = (input: MicrophoneAudioChunk): void => {
    const chunk = input?.pcm instanceof Float32Array ? input.pcm : null;
    if (!chunk?.length) {
      return;
    }

    const nextSampleRate = input.sampleRate || sampleRate || STREAMING_PROCESSING_SAMPLE_RATE;
    if (nextSampleRate !== sampleRate) {
      sampleRate = nextSampleRate;
      updateCaptureInfo({
        processingSampleRate: nextSampleRate,
      });
    } else {
      sampleRate = nextSampleRate;
    }

    if (micMode === 'manual') {
      manualChunks.push(chunk);
      updateStatus(`Recording microphone at ${sampleRate} Hz…`, 'Microphone active');
      return;
    }

    starter.processChunk(chunk, {
      startFrame: input.startFrame,
      endFrame: input.endFrame,
    });
  };

  const starterUnsubscribe = starter.subscribe((event) => {
    if (event.type === 'metrics') {
      const tenVadState = event.payload?.tenVad?.state;
      if (state.isMicActive && tenVadState === 'degraded') {
        setState({ micStatus: 'Listening with rough gate fallback…' });
      }
      return;
    }

    if (event.type === 'speech-start') {
      setState({ micStatus: 'Speech detected. Rewinding to onset…' });
      return;
    }
    if (event.type === 'speech-update') {
      setState({ micStatus: 'Speech active. Waiting for a natural pause…' });
      return;
    }
    if (event.type === 'speech-end') {
      setState({ micStatus: 'Detected a pause. Finalizing utterance…' });
      return;
    }
    if (event.type === 'segment-ready') {
      const pcm = event.payload.readPcm();
      onUtterance({
        pcm,
        sampleRate: event.payload.sampleRate,
        reason: event.payload.reason,
        durationSeconds: pcm.length / event.payload.sampleRate,
        sourceLabel: `microphone-${event.payload.reason}-${new Date().toLocaleTimeString()}`,
        metadata: event.payload.metadata,
      });
      return;
    }
    if (event.type === 'error') {
      setState({
        micError: event.payload?.message ?? String(event.payload),
      });
    }
  });

  const applyConfig = (): void => {
    starter.updateConfig({
      ...resolvedConfig,
      profileId,
    });
    if (!state.isMicActive) {
      updateCaptureInfo({
        processingSampleRate: STREAMING_PROCESSING_SAMPLE_RATE,
        chunkFrames: getConfiguredChunkFrames(),
        chunkDurationMs: getConfiguredChunkDurationMs(),
      });
    }
    monitor.flush();
  };

  applyConfig();

  return {
    monitor,
    starter,
    subscribe(listener: (state: BrowserRealtimeMicrophoneControllerState) => void): () => void {
      listeners.add(listener);
      listener(state);
      return () => {
        listeners.delete(listener);
      };
    },
    getState(): BrowserRealtimeMicrophoneControllerState {
      return state;
    },
    configure(
      next: {
        readonly profileId?: string;
        readonly config?: StreamingDetectorConfigOverrides | StreamingDetectorConfig;
      } = {},
    ): void {
      profileId = next.profileId ?? profileId;
      resolvedConfig = mergeStreamingConfig(profileId, next.config ?? resolvedConfig);
      applyConfig();
    },
    setMicMode(mode: BrowserRealtimeMicrophoneMode): void {
      if (mode === micMode) {
        return;
      }
      micMode = mode;
      setState({ micMode: mode });
    },
    async start(): Promise<void> {
      manualChunks = [];
      setState({ micError: '' });

      const handle = await startCapture({
        constraints: {
          audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            ...options.microphoneConstraints,
          },
        },
        chunkFrames: getConfiguredChunkFrames(),
        chunkDurationMs: getConfiguredChunkDurationMs(),
        targetSampleRate: STREAMING_PROCESSING_SAMPLE_RATE,
        stopTracksOnStop: true,
        onChunk: handleChunk,
        onError: (error) => {
          setState({
            micError: error instanceof Error ? error.message : String(error),
          });
        },
      });

      captureHandle = handle;
      sampleRate = handle.sampleRate || STREAMING_PROCESSING_SAMPLE_RATE;
      const audioTrack = handle.stream.getAudioTracks?.()[0] ?? null;
      updateCaptureInfo({
        inputSampleRate: handle.deviceSampleRate,
        contextSampleRate: handle.contextSampleRate,
        processingSampleRate: sampleRate,
        chunkFrames: handle.chunkFrames ?? STREAMING_TIMELINE_CHUNK_FRAMES,
        chunkDurationMs: handle.chunkDurationMs ?? STREAMING_TIMELINE_CHUNK_MS,
        deviceLabel: audioTrack?.label ?? '',
      });
      setState({ isMicActive: true });
      monitor.flush();

      if (micMode === 'speech-detect') {
        await starter.start({ sampleRate });
        const tenVadState = starter.getSnapshot()?.tenVad?.state;
        updateStatus(
          tenVadState === 'degraded'
            ? `Listening for speech at ${sampleRate} Hz with rough gate fallback…`
            : `Listening for speech at ${sampleRate} Hz with streaming detector…`,
          'Microphone active',
        );
        return;
      }

      updateStatus(
        `Recording microphone at ${sampleRate} Hz. Stop to transcribe the captured audio.`,
        'Microphone active',
      );
    },
    async stop({ flush = true }: { readonly flush?: boolean } = {}): Promise<void> {
      setState({ isMicActive: false });

      const handle = captureHandle;
      captureHandle = null;
      if (handle) {
        await handle.stop();
      }

      updateCaptureInfo({
        inputSampleRate: null,
        contextSampleRate: null,
        deviceLabel: '',
      });

      if (micMode === 'speech-detect') {
        await starter.stop({ flush });
      } else if (flush && manualChunks.length > 0) {
        const pcm = concatFloat32Chunks(manualChunks);
        manualChunks = [];
        onUtterance({
          pcm,
          sampleRate,
          reason: 'stop',
          durationSeconds: pcm.length / sampleRate,
          sourceLabel: `microphone-stop-${new Date().toLocaleTimeString()}`,
          metadata: null,
        });
      } else {
        manualChunks = [];
      }

      updateStatus('Microphone stopped', 'Model ready');
      monitor.flush();
    },
    flush(reason = 'manual'): void {
      if (micMode === 'speech-detect') {
        starter.flush(reason);
        return;
      }

      if (!manualChunks.length) {
        return;
      }
      const pcm = concatFloat32Chunks(manualChunks);
      manualChunks = [];
      onUtterance({
        pcm,
        sampleRate,
        reason,
        durationSeconds: pcm.length / sampleRate,
        sourceLabel: `microphone-${reason}-${new Date().toLocaleTimeString()}`,
        metadata: null,
      });
    },
    async dispose(): Promise<void> {
      if (captureHandle) {
        await captureHandle.stop();
        captureHandle = null;
      }
      starterUnsubscribe();
      monitor.dispose();
      await starter.dispose();
      listeners.clear();
    },
  };
}
