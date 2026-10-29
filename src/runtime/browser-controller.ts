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
import type { RealtimeLatencySummary } from './realtime-latency.js';
import type { AbortSignalLike, BaseTranscriptionOptions, PartialTranscript, StreamingTranscriber, TranscriptResult } from '../types/index.js';
import type { RealtimeTranscriptionContext, RealtimeTranscriptionUpdate } from './controller.js';
import { AudioRingBuffer, createCaptureTranscript } from './realtime.js';
import { PcmAudioBuffer } from '../audio/audio.js';
import { SpeechContextBuffer, type SpeechContextRevision } from './speech-context.js';
import type { LoadedSpeechModel } from './load.js';
import { resampleSinc } from '../audio/resample.js';
import { subscribeToAbortSignal } from '../io/abort.js';

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
  readonly context?: RealtimeTranscriptionContext;
}

export interface BrowserRealtimeTranscriptionContextOptions {
  readonly maxDurationSeconds?: number;
  readonly maxGapSeconds?: number;
  readonly prerollSeconds?: number;
}

export interface BrowserRealtimeMicrophoneControllerState {
  readonly isMicActive: boolean;
  readonly micStatus: string;
  readonly micError: string;
  readonly micMode: BrowserRealtimeMicrophoneMode;
  readonly captureInfo: BrowserRealtimeCaptureInfo;
  /** From the inner transcription controller when a transcribe callback is attached. */
  readonly latency: RealtimeLatencySummary | null;
  readonly transcript?: RealtimeTranscriptionUpdate | null;
  readonly transcriptText?: string;
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
  readonly streamingTranscriber?: StreamingTranscriber;
  readonly onPartialTranscript?: (partial: PartialTranscript) => void;
  readonly transcribeMonoPcm?: (
    pcm: Float32Array,
    sampleRate: number,
  ) => Promise<unknown>;
  readonly onSegment?: (segment: unknown) => void;
  /** Re-run each accepted VAD segment over the complete captured turn context. */
  readonly transcriptionContext?: BrowserRealtimeTranscriptionContextOptions;
  /** Borrowed loaded model used for microphone utterances. The caller owns disposal. */
  readonly model?: LoadedSpeechModel;
  readonly transcribeOptions?: BaseTranscriptionOptions;
  readonly onTranscript?: (update: RealtimeTranscriptionUpdate) => void;
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

function combineAbortSignals(
  signals: readonly (AbortSignalLike | null | undefined)[],
): { readonly signal: AbortSignal; dispose(): void } {
  const controller = new AbortController();
  if (signals.some((signal) => signal?.aborted)) controller.abort();
  const cleanups = controller.signal.aborted
    ? []
    : signals.map((signal) => subscribeToAbortSignal(signal, () => controller.abort()));
  return { signal: controller.signal, dispose: () => cleanups.forEach((cleanup) => cleanup()) };
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
    latency: null,
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
  if (options.model && (options.transcribe || options.transcribeMonoPcm || options.streamingTranscriber)) {
    throw new TypeError('model cannot be combined with transcribe, transcribeMonoPcm, or streamingTranscriber.');
  }
  let captureTranscript = options.model ? createCaptureTranscript() : null;
  let micMode: BrowserRealtimeMicrophoneMode = options.micMode ?? 'speech-detect';
  let profileId =
    options.profileId ?? resolveStreamingProfileId(options.isRealtimeEouModel === true);
  let resolvedConfig = mergeStreamingConfig(profileId, options.config);
  let captureHandle: BrowserMicrophoneCaptureHandle | null = null;
  let sampleRate = STREAMING_PROCESSING_SAMPLE_RATE;
  let manualChunks: Float32Array[] = [];
  let manualIngestFrame = 0;
  const transcriptionContextMaxDurationSeconds = options.transcriptionContext?.maxDurationSeconds ?? 30;
  const resolvedTranscriptionContext = options.transcriptionContext ?? (options.model ? {} : undefined);
  const transcriptionContext = resolvedTranscriptionContext
    ? new SpeechContextBuffer({
        sampleRate: STREAMING_PROCESSING_SAMPLE_RATE,
        maxDurationSeconds: resolvedTranscriptionContext.maxDurationSeconds ?? transcriptionContextMaxDurationSeconds,
        retentionDurationSeconds: (resolvedTranscriptionContext.maxDurationSeconds ?? transcriptionContextMaxDurationSeconds) + resolvedConfig.ringBufferDurationMs / 1000,
        maxGapSeconds: resolvedTranscriptionContext.maxGapSeconds ?? 3,
        // Neural VAD can report onset after the first ~300ms of speech. Keep
        // additional captured margin so contextual transcription retains it.
        prerollSeconds: resolvedTranscriptionContext.prerollSeconds ?? 0.6,
      })
    : null;
  let lifecycleGeneration = 0;
  let lifecycleOperationTail: Promise<void> = Promise.resolve();
  let disposed = false;
  let startAbort: AbortController | null = null;
  let state = createInitialState(micMode);
  const pendingTranscriptions = new Set<Promise<void>>();
  const transcriptionErrors: unknown[] = [];

  const drainTranscriptions = async (): Promise<void> => {
    while (pendingTranscriptions.size) {
      await Promise.all([...pendingTranscriptions]);
    }
    if (transcriptionErrors.length) {
      const errors = transcriptionErrors.splice(0);
      if (errors.length === 1) throw errors[0];
      throw new AggregateError(errors, 'Microphone transcription failed.');
    }
  };

  const modelTranscribe = options.model
    ? async (request: import('./controller.js').RealtimeTranscriptionRequest) => {
        const targetSampleRate = options.model!.model.info.inference?.sampleRate ?? request.sampleRate;
        const pcm = request.sampleRate === targetSampleRate
          ? request.pcm
          : resampleSinc(request.pcm, request.sampleRate, targetSampleRate);
        const combined = combineAbortSignals([request.signal, options.transcribeOptions?.signal]);
        try {
          const response = await options.model!.transcribeMonoPcm(pcm, targetSampleRate, {
            ...options.transcribeOptions,
            responseFlavor: 'canonical',
            signal: combined.signal,
          });
          return response as TranscriptResult;
        } finally {
          combined.dispose();
        }
      }
    : undefined;

  const starter = createStarter({
    ...options,
    transcribe: options.transcribe ?? modelTranscribe,
    profileId,
    config: resolvedConfig,
  });
  const monitor = createMonitor(starter, {
    frameIntervalMs: options.frameIntervalMs ?? DEFAULT_FRAME_INTERVAL_MS,
  });

  const withLatency = (
    current: BrowserRealtimeMicrophoneControllerState,
  ): BrowserRealtimeMicrophoneControllerState => ({
    ...current,
    latency: starter.getSnapshot().latency ?? null,
  });

  const emit = (): void => {
    const nextState = withLatency(state);
    for (const listener of listeners) {
      listener(nextState);
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

  const trackTranscription = (operation: Promise<unknown>): void => {
    const task = operation
      .then(() => undefined)
      .catch((error) => {
        transcriptionErrors.push(error);
        if (!disposed) setState({ micError: error instanceof Error ? error.message : String(error) });
      })
      .finally(() => {
        pendingTranscriptions.delete(task);
      });
    pendingTranscriptions.add(task);
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

  const streamingTranscriber = options.streamingTranscriber;
  const onPartialTranscript = options.onPartialTranscript;
  let isSpeechActive = false;
  let streamingOperationTail: Promise<void> = Promise.resolve();
  let streamingAudio: AudioRingBuffer | null = null;
  let streamingCursor: number | null = null;

  const enqueueStreamingChunk = (pcmChunk: Float32Array): void => {
    if (!streamingTranscriber) {
      return;
    }
    const audio = PcmAudioBuffer.fromMono(pcmChunk, sampleRate);
    const run = async (): Promise<void> => {
      if (disposed) {
        return;
      }
      try {
        const partial = await streamingTranscriber.pushAudio(audio);
        if (partial) {
          const text = partial.text?.trim() || partial.previewText?.trim();
          if (text && onPartialTranscript) {
            onPartialTranscript(partial);
          }
        }
      } catch (error) {
        setState({ micError: error instanceof Error ? error.message : String(error) });
      }
    };
    streamingOperationTail = streamingOperationTail.then(run, run);
  };

  const feedStreamingThrough = (endFrame: number): void => {
    if (!streamingAudio || streamingCursor === null || endFrame <= streamingCursor) return;
    // A lost onset must be observable, never silently truncated to the ring's base.
    const pcm = streamingAudio.read(streamingCursor, endFrame);
    streamingCursor = endFrame;
    enqueueStreamingChunk(pcm);
  };

  const finalizeStreaming = (): void => {
    if (!streamingTranscriber) return;
    const run = async (): Promise<void> => {
      if (disposed) return;
      try {
        const final = await streamingTranscriber.finalize();
        if (final.text?.trim() && onPartialTranscript) onPartialTranscript(final);
      }
      finally { await streamingTranscriber.reset(); }
    };
    streamingOperationTail = streamingOperationTail.then(run, run);
    // Observe a rejected background boundary immediately, while keeping the
    // original rejected tail available to stop()/dispose() callers.
    void streamingOperationTail.catch((error) => {
      setState({ micError: error instanceof Error ? error.message : String(error) });
    });
    streamingCursor = null;
  };

  const handleChunk = (input: MicrophoneAudioChunk, generation: number): void => {
    if (disposed || generation !== lifecycleGeneration || !state.isMicActive) {
      return;
    }
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

    if (transcriptionContext && micMode === 'speech-detect') {
      try {
        if (nextSampleRate !== STREAMING_PROCESSING_SAMPLE_RATE) {
          throw new RangeError(`Speech context requires ${STREAMING_PROCESSING_SAMPLE_RATE} Hz PCM, got ${nextSampleRate} Hz.`);
        }
        if (input.startFrame === undefined) {
          throw new RangeError('Speech context requires frame-addressed microphone chunks.');
        }
        transcriptionContext.write(input.startFrame, chunk);
      } catch (error) {
        setState({ micError: error instanceof Error ? error.message : String(error) });
        return;
      }
    }

    if (micMode === 'manual') {
      manualChunks.push(chunk);
      manualIngestFrame += chunk.length;
      starter.controller?.noteIngest(manualIngestFrame);
      if (streamingTranscriber) {
        enqueueStreamingChunk(chunk);
      }
      updateStatus(`Recording microphone at ${sampleRate} Hz…`, 'Microphone active');
      return;
    }

    if (streamingTranscriber) {
      streamingAudio ??= new AudioRingBuffer({ sampleRate, durationSeconds: resolvedConfig.ringBufferDurationMs / 1000 });
      streamingAudio.write(chunk);
    }
    starter.processChunk(chunk, {
      startFrame: input.startFrame,
      endFrame: input.endFrame,
    });

    if (streamingTranscriber && isSpeechActive && streamingAudio) {
      feedStreamingThrough(streamingAudio.getCurrentFrame());
    }
  };

  const enqueueLifecycleOperation = <T>(operation: () => Promise<T>): Promise<T> => {
    const result = lifecycleOperationTail.then(operation, operation);
    lifecycleOperationTail = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  };

  const publishCapturedAudio = (
    pcm: Float32Array,
    details: {
      readonly sampleRate: number;
      readonly reason: string;
      readonly startFrame: number;
      readonly endFrame: number;
      readonly metadata: unknown;
    },
    contextOverride?: SpeechContextRevision,
  ): void => {
    let contextualAudio: SpeechContextRevision | null = contextOverride ?? null;
    if (!contextualAudio && transcriptionContext && micMode === 'speech-detect') {
      const metadata = details.metadata as { logicalStartFrame?: number; logicalEndFrame?: number } | null;
      try {
        contextualAudio = transcriptionContext.snapshot(
          details.startFrame,
          details.endFrame,
          metadata?.logicalStartFrame ?? details.startFrame,
          metadata?.logicalEndFrame ?? details.endFrame,
        );
      } catch (error) {
        setState({ micError: error instanceof Error ? error.message : String(error) });
        return;
      }
    }
    const inferencePcm = contextualAudio?.pcm ?? pcm;
    const context = contextualAudio ? {
      id: contextualAudio.id,
      revision: contextualAudio.revision,
      startFrame: contextualAudio.startFrame,
      endFrame: contextualAudio.endFrame,
      sampleRate: contextualAudio.sampleRate,
      finalized: contextualAudio.finalized,
    } : undefined;
    onUtterance({
      pcm: inferencePcm,
      sampleRate: details.sampleRate,
      reason: details.reason,
      durationSeconds: inferencePcm.length / details.sampleRate,
      sourceLabel: `microphone-${details.reason}-${new Date().toLocaleTimeString()}`,
      metadata: details.metadata,
      context,
    });

    if (streamingTranscriber) {
      // Detection finalizes at speech-end, before a continuation's audio is queued.
      if (micMode === 'manual') finalizeStreaming();
    } else if (options.transcribeMonoPcm && pcm.length > 0) {
      trackTranscription(Promise.resolve().then(() => options.transcribeMonoPcm!(inferencePcm, details.sampleRate)));
    }

    const transcription = starter.controller;
    if (!transcription || pcm.length === 0) {
      return;
    }
    const task = transcription
      .transcribeUtterance(inferencePcm, {
        startFrame: context?.startFrame ?? details.startFrame,
        endFrame: context?.endFrame ?? details.endFrame,
        sampleRate: details.sampleRate,
        reason: details.reason,
        context,
      })
      .then((update) => {
        if (!update || disposed) return;
        if (captureTranscript) {
          const transcriptText = captureTranscript.update(update.canonical.text, update.context);
          setState({ transcript: update, transcriptText });
        }
        options.onTranscript?.(update);
      })
      .catch((error) => {
        transcriptionErrors.push(error);
        if (!disposed) setState({ micError: error instanceof Error ? error.message : String(error) });
      })
      .finally(() => {
        pendingTranscriptions.delete(task);
        if (!disposed) {
          emit();
        }
      });
    pendingTranscriptions.add(task);
  };

  const starterUnsubscribe = starter.subscribe((event) => {
    if (disposed) {
      return;
    }
    if (event.type === 'metrics') {
      const tenVadState = event.payload?.tenVad?.state;
      if (state.isMicActive && tenVadState === 'degraded') {
        setState({ micStatus: 'Listening with rough fallback…' });
      }
      return;
    }

    if (event.type === 'speech-start') {
      isSpeechActive = true;
      streamingCursor = event.payload.startFrame;
      setState({ micStatus: 'Speech detected. Rewinding to onset…' });
      return;
    }
    if (event.type === 'speech-update') {
      isSpeechActive = true;
      setState({ micStatus: 'Speech active. Waiting for a natural pause…' });
      return;
    }
    if (event.type === 'speech-end') {
      isSpeechActive = false;
      feedStreamingThrough(event.payload.endFrame);
      finalizeStreaming();
      setState({ micStatus: 'Detected a pause. Finalizing utterance…' });
      return;
    }
    if (event.type === 'segment-ready') {
      isSpeechActive = false;
      options.onSegment?.(event.payload);
      publishCapturedAudio(event.payload.readPcm(), {
        sampleRate: event.payload.sampleRate,
        reason: event.payload.reason,
        startFrame: event.payload.startFrame,
        endFrame: event.payload.endFrame,
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
      listener(withLatency(state));
      return () => {
        listeners.delete(listener);
      };
    },
    getState(): BrowserRealtimeMicrophoneControllerState {
      return withLatency(state);
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
      if (disposed) {
        throw new Error('BrowserRealtimeMicrophoneController has been disposed.');
      }
      const generation = ++lifecycleGeneration;
      return enqueueLifecycleOperation(async () => {
        if (generation !== lifecycleGeneration || disposed || state.isMicActive) {
          return;
        }
        await drainTranscriptions();
        manualChunks = [];
        manualIngestFrame = 0;
        transcriptionContext?.reset();
        if (captureTranscript) {
          captureTranscript = createCaptureTranscript();
          setState({ transcript: null, transcriptText: '' });
        }
        streamingAudio = null;
        streamingCursor = null;
        isSpeechActive = false;
        if (streamingTranscriber) {
          await streamingOperationTail;
          await streamingTranscriber.reset();
        }
        setState({ micError: '' });

        // Capture can deliver PCM before the async starter/model has reset.
        // Keep an owned, bounded prefix so all consumers see the same timeline.
        const startupChunks: MicrophoneAudioChunk[] = [];
        let startupFrames = 0;
        const startupFrameLimit = Math.floor(
          resolvedConfig.ringBufferDurationMs / 1000 * STREAMING_PROCESSING_SAMPLE_RATE,
        );
        let bufferingStartup = micMode === 'speech-detect';
        let startupError: Error | null = null;
        const releaseStartup = (replay: boolean): void => {
          bufferingStartup = false;
          try {
            if (replay) {
              for (const chunk of startupChunks) handleChunk(chunk, generation);
            }
          } finally {
            startupChunks.length = 0;
          }
        };

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
          onChunk: (input) => {
            if (disposed || generation !== lifecycleGeneration) return;
            if (!bufferingStartup) {
              handleChunk(input, generation);
              return;
            }
            if (startupError || !(input?.pcm instanceof Float32Array) || !input.pcm.length) return;
            startupFrames += input.pcm.length;
            if (startupFrames > startupFrameLimit) {
              startupError = new RangeError(
                'Microphone startup PCM exceeded ring-buffer duration while waiting for model readiness.',
              );
              startupChunks.length = 0;
              startAbort?.abort();
              return;
            }
            startupChunks.push({ ...input, pcm: input.pcm.slice() });
          },
          onError: (error) => {
            if (generation !== lifecycleGeneration || disposed) {
              return;
            }
            setState({
              micError: error instanceof Error ? error.message : String(error),
            });
          },
        }).catch((error: unknown) => {
          releaseStartup(false);
          throw error;
        });

        if (generation !== lifecycleGeneration || disposed) {
          releaseStartup(false);
          await handle.stop();
          return;
        }

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

        try {
          if (startupError) throw startupError;
          if (micMode === 'speech-detect') {
            const abort = new AbortController();
            startAbort = abort;
            try {
              await starter.start({ sampleRate, signal: abort.signal });
            } catch (error) {
              if (generation !== lifecycleGeneration || disposed) {
                releaseStartup(false);
                await starter.stop({ flush: false }).catch(() => undefined);
                await handle.stop().catch(() => undefined);
                captureHandle = null;
                setState({ isMicActive: false });
                return;
              }
              throw startupError ?? error;
            } finally {
              if (startAbort === abort) {
                startAbort = null;
              }
            }
            if (generation !== lifecycleGeneration || disposed) {
              releaseStartup(false);
              await starter.stop({ flush: false });
              await handle.stop();
              captureHandle = null;
              return;
            }
            if (startupError) throw startupError;
            releaseStartup(true);
            const snapshot = starter.getSnapshot();
            const tenVadState = snapshot?.tenVad?.state;
            const effectiveGateMode = snapshot?.gate?.effectiveMode ?? 'rough-only';
            const vadBackend = snapshot?.gate?.vadBackend;
            let segmenterLabel: string;
            if (tenVadState === 'degraded') {
              segmenterLabel = 'rough fallback';
            } else if (
              effectiveGateMode === 'rough-only' ||
              tenVadState === 'disabled' ||
              tenVadState === 'idle' ||
              !snapshot?.config?.tenVadEnabled
            ) {
              segmenterLabel = 'RoughSpeechGate';
            } else {
              segmenterLabel = vadBackend === 'ten-vad' ? 'TEN-VAD' : 'FireRed VAD';
            }
            updateStatus(
              `Listening for speech at ${sampleRate} Hz with ${segmenterLabel}…`,
              'Microphone active',
            );
            return;
          }

          releaseStartup(true);
          updateStatus(
            `Recording microphone at ${sampleRate} Hz. Stop to transcribe the captured audio.`,
            'Microphone active',
          );
        } catch (error) {
          releaseStartup(false);
          if (captureHandle === handle) {
            captureHandle = null;
          }
          await handle.stop().catch(() => undefined);
          if (generation === lifecycleGeneration && !disposed) {
            setState({
              isMicActive: false,
              micError: error instanceof Error ? error.message : String(error),
            });
          }
          throw error;
        }
      });
    },
    async stop({ flush = true }: { readonly flush?: boolean } = {}): Promise<void> {
      if (disposed) {
        return;
      }
      const generation = ++lifecycleGeneration;
      startAbort?.abort();
      setState({ isMicActive: false });
      return enqueueLifecycleOperation(async () => {
        if (
          generation !== lifecycleGeneration &&
          !captureHandle &&
          !manualChunks.length &&
          !pendingTranscriptions.size &&
          !transcriptionErrors.length
        ) {
          return;
        }
        const errors: unknown[] = [];
        const attempt = async (operation: () => void | Promise<void>): Promise<void> => {
          try {
            await operation();
          } catch (error) {
            errors.push(error);
          }
        };
        const handle = captureHandle;
        captureHandle = null;
        if (handle) {
          await attempt(() => handle.stop());
        }
        if (disposed) {
          manualChunks = [];
          manualIngestFrame = 0;
          transcriptionContext?.reset();
          if (errors.length) throw errors[0];
          return;
        }

        await attempt(() => {
          updateCaptureInfo({
            inputSampleRate: null,
            contextSampleRate: null,
            deviceLabel: '',
          });
        });

        if (micMode === 'speech-detect') {
          let detectorStopped = false;
          await attempt(async () => {
            await starter.stop({ flush: flush && !disposed });
            detectorStopped = true;
          });
          if (flush && !disposed && detectorStopped && transcriptionContext && !streamingTranscriber) {
            await attempt(() => {
              let finalized: SpeechContextRevision | null;
              try {
                finalized = transcriptionContext.finalize();
              } catch (error) {
                setState({ micError: error instanceof Error ? error.message : String(error) });
                throw error;
              }
              if (!finalized) return;
              publishCapturedAudio(finalized.pcm, {
                sampleRate: finalized.sampleRate,
                reason: 'stop',
                startFrame: finalized.startFrame,
                endFrame: finalized.endFrame,
                metadata: { finalized: true },
              }, finalized);
            });
          }
          await attempt(() => streamingOperationTail);
        } else {
          await attempt(async () => {
            if (flush && !disposed && manualChunks.length > 0) {
              const pcm = concatFloat32Chunks(manualChunks);
              const endFrame = manualIngestFrame;
              manualChunks = [];
              manualIngestFrame = 0;
              publishCapturedAudio(pcm, {
                sampleRate,
                reason: 'stop',
                startFrame: Math.max(0, endFrame - pcm.length),
                endFrame: Math.max(pcm.length, endFrame),
                metadata: null,
              });
            } else {
              manualChunks = [];
              manualIngestFrame = 0;
            }
            await streamingOperationTail;
          });
        }

        await attempt(drainTranscriptions);

        if (!disposed) {
          await attempt(() => updateStatus('Microphone stopped', 'Model ready'));
          if (!disposed) await attempt(() => monitor.flush());
        }
        if (errors.length === 1) throw errors[0];
        if (errors.length > 1)
          throw new AggregateError(errors, 'Microphone controller stop failed.');
        transcriptionContext?.reset();
      });
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
      const endFrame = manualIngestFrame;
      manualChunks = [];
      manualIngestFrame = 0;
      publishCapturedAudio(pcm, {
        sampleRate,
        reason,
        startFrame: Math.max(0, endFrame - pcm.length),
        endFrame: Math.max(pcm.length, endFrame),
        metadata: null,
      });
    },
    async dispose(): Promise<void> {
      if (disposed) {
        return;
      }
      disposed = true;
      ++lifecycleGeneration;
      startAbort?.abort();
      setState({ isMicActive: false });
      return enqueueLifecycleOperation(async () => {
        const errors: unknown[] = [];
        const attempt = async (release: () => void | Promise<void>): Promise<void> => {
          try {
            await release();
          } catch (error) {
            errors.push(error);
          }
        };
        const handle = captureHandle;
        captureHandle = null;
        if (handle) {
          await attempt(() => handle.stop());
        }
        manualChunks = [];
        manualIngestFrame = 0;
        await attempt(starterUnsubscribe);
        await attempt(() => monitor.dispose());
        await attempt(() => starter.dispose());
        await attempt(drainTranscriptions);
        await attempt(() => streamingOperationTail);
        listeners.clear();
        if (errors.length === 1) throw errors[0];
        if (errors.length > 1) {
          throw new AggregateError(errors, 'Microphone controller disposal failed.');
        }
      });
    },
  };
}
