import { PcmAudioBuffer } from '../audio/audio.js';
import { subscribeToAbortSignal } from '../io/abort.js';
import type {
  BaseTranscriptionOptions,
  PartialTranscript,
  RuntimeProgressEvent,
  StreamingSessionOptions,
  TranscriptResult,
  TranscriptWarning,
} from '../types/index.js';
import type { LoadSpeechModelOptions, LoadedSpeechModel } from './load.js';
import type { DecodeAudioSourceOptions } from './media.js';
import { createDefaultModelInferenceLimits } from '../pipeline/window-policy.js';
import type {
  BrowserRealtimeMicrophoneController,
  BrowserRealtimeMicrophoneControllerOptions,
  BrowserRealtimeMicrophoneControllerState,
} from './browser-controller.js';
import type { RealtimeTranscriptionUpdate } from './controller.js';

export type SpeechControllerStatus =
  | 'idle'
  | 'loading'
  | 'ready'
  | 'transcribing'
  | 'streaming'
  | 'error'
  | 'disposed';

export interface SpeechControllerSnapshot {
  readonly status: SpeechControllerStatus;
  readonly modelId: string | null;
  readonly progress: RuntimeProgressEvent | null;
  readonly loadWarnings?: readonly TranscriptWarning[];
  readonly result: TranscriptResult | null;
  readonly partial: PartialTranscript | null;
  readonly error: unknown | null;
  readonly supportsStreaming: boolean;
  readonly microphone?: BrowserRealtimeMicrophoneControllerState | null;
}

export interface SpeechController {
  getSnapshot(): SpeechControllerSnapshot;
  subscribe(listener: () => void): () => void;
  load(options: LoadSpeechModelOptions): Promise<void>;
  transcribe(
    pcm: Float32Array,
    sampleRate: number,
    options?: BaseTranscriptionOptions,
  ): Promise<TranscriptResult>;
  startStreaming(options?: StreamingSessionOptions): Promise<void>;
  pushAudio(pcm: Float32Array, sampleRate: number): Promise<PartialTranscript>;
  finalizeStreaming(): Promise<PartialTranscript>;
  dispose(): Promise<void>;
}

/** File/URL decoding capability returned by the built-in controller factory. */
export interface AudioSourceSpeechController extends SpeechController {
  transcribeSource(
    source: string | Blob,
    options?: BaseTranscriptionOptions,
    audioOptions?: Omit<DecodeAudioSourceOptions, 'targetSampleRate' | 'signal'>,
  ): Promise<TranscriptResult>;
}

export type SpeechMicrophoneOptions = Omit<
  BrowserRealtimeMicrophoneControllerOptions,
  'model' | 'transcribe' | 'transcribeMonoPcm' | 'streamingTranscriber'
>;

export interface MicrophoneSpeechController extends AudioSourceSpeechController {
  startMicrophone(options?: SpeechMicrophoneOptions): Promise<void>;
  stopMicrophone(): Promise<void>;
}

function createAbortError(operation: string): Error {
  const error = new Error(`Speech controller ${operation} was aborted.`);
  error.name = 'AbortError';
  return error;
}

/** Framework-neutral, lazy-loading state owner for app integrations. */
export function createSpeechController(): MicrophoneSpeechController {
  let snapshot: SpeechControllerSnapshot = Object.freeze({
    status: 'idle',
    modelId: null,
    progress: null,
    loadWarnings: Object.freeze([]),
    result: null,
    partial: null,
    error: null,
    supportsStreaming: false,
  });
  const listeners = new Set<() => void>();
  let loaded: LoadedSpeechModel | null = null;
  let streaming: Awaited<ReturnType<LoadedSpeechModel['createStreamingTranscriber']>> | null = null;
  let loadGeneration = 0;
  let loadAbort: AbortController | null = null;
  let loadTask: Promise<void> | null = null;
  const loadTasks = new Map<number, Promise<void>>();
  const inferenceAbort = new AbortController();
  let operation: 'transcribing' | 'streaming' | 'microphone' | null = null;
  let finalizing = false;
  let inferenceTask: Promise<unknown> | null = null;
  let startCleanupFailure: { readonly error: unknown } | null = null;
  let pushQueue: Promise<PartialTranscript> = Promise.resolve(null as never);
  let disposed = false;
  let disposeTask: Promise<void> | null = null;
  let microphone: BrowserRealtimeMicrophoneController | null = null;
  let unsubscribeMicrophone: (() => void) | null = null;
  let microphoneGeneration = 0;
  let microphoneStartGeneration = 0;
  let microphoneStartTask: Promise<void> | null = null;
  let microphoneStopTask: Promise<void> | null = null;
  let microphoneCleanupTask: Promise<void> | null = null;

  const publish = (patch: Partial<SpeechControllerSnapshot>): void => {
    snapshot = Object.freeze({ ...snapshot, ...patch });
    for (const listener of [...listeners]) {
      try {
        listener();
      } catch {
        /* A consumer callback cannot break model ownership. */
      }
    }
  };
  const assertOpen = (): void => {
    if (disposed) throw new Error('Speech controller is disposed.');
  };
  const assertReady = (): LoadedSpeechModel => {
    assertOpen();
    if (!loaded) throw new Error('Load a speech model before using the controller.');
    return loaded;
  };
  const closeStream = async (): Promise<void> => {
    const active = streaming;
    streaming = null;
    if (active) await active.dispose?.();
  };

  const cleanupMicrophone = (active: BrowserRealtimeMicrophoneController): Promise<void> => {
    if (microphoneCleanupTask) return microphoneCleanupTask;
    if (microphone !== active) return Promise.resolve();
    unsubscribeMicrophone?.();
    unsubscribeMicrophone = null;
    microphone = null;
    const cleanup = Promise.resolve().then(() => active.dispose());
    const task = cleanup.finally(() => {
      microphoneCleanupTask = null;
    });
    microphoneCleanupTask = task;
    return task;
  };

  async function runTranscription(
    input: { pcm: Float32Array; sampleRate: number } | { source: string | Blob },
    options: BaseTranscriptionOptions = {},
    audioOptions: Omit<DecodeAudioSourceOptions, 'targetSampleRate' | 'signal'> = {},
  ): Promise<TranscriptResult> {
    const model = assertReady();
    if (operation || loadTask) throw new Error('Another controller operation is active.');
    operation = 'transcribing';
    publish({ status: 'transcribing', result: null, partial: null, error: null, microphone: null });
    const callerSignal = options.signal;
    const signal = {
      get aborted(): boolean {
        return inferenceAbort.signal.aborted || callerSignal?.aborted === true;
      },
    };
    const task = Promise.resolve()
      .then(async () => {
        if (signal.aborted) throw createAbortError('transcription');
        let pcm: Float32Array;
        let rate: number;
        if ('pcm' in input) {
          pcm = input.pcm;
          rate = input.sampleRate;
        } else {
          const { decodeAudioSourceToMonoPcm } = await import('./media.js');
          const inference =
            model.model.info.inference ??
            createDefaultModelInferenceLimits({
              family: model.model.info.family,
              modelId: model.model.info.modelId,
            });
          const audio = await decodeAudioSourceToMonoPcm(input.source, {
            ...audioOptions,
            targetSampleRate: inference.sampleRate,
            signal,
          });
          pcm = audio.pcm;
          rate = audio.sampleRate;
        }
        if (signal.aborted) throw createAbortError('transcription');
        return model.transcribeMonoPcm(pcm, rate, {
          ...options,
          responseFlavor: 'canonical',
          signal,
        });
      })
      .then((result) => {
        if (signal.aborted) throw createAbortError('transcription');
        const transcript = result as TranscriptResult;
        if (!disposed) publish({ status: 'ready', result: transcript, partial: null });
        return transcript;
      })
      .catch((error: unknown) => {
        if (!disposed) publish({ status: 'error', error });
        throw error;
      })
      .finally(() => {
        operation = null;
        inferenceTask = null;
        if (!disposed && snapshot.status === 'transcribing') publish({ status: 'ready' });
      });
    inferenceTask = task;
    return task;
  }

  const api: MicrophoneSpeechController = {
    getSnapshot: () => snapshot,
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    async load(options) {
      assertOpen();
      if (operation) throw new Error(`Cannot load a model while ${operation} is active.`);
      const generation = ++loadGeneration;
      loadAbort?.abort();
      const aborter = new AbortController();
      loadAbort = aborter;
      const unsubscribeAbort = subscribeToAbortSignal(options.signal, () => aborter.abort());
      const current = (): boolean => !disposed && generation === loadGeneration;
      publish({
        status: 'loading',
        progress: null,
        loadWarnings: Object.freeze([]),
        result: null,
        partial: null,
        error: null,
        microphone: null,
      });
      const task = Promise.resolve().then(async () => {
        let candidate: LoadedSpeechModel | null = null;
        try {
          const { loadSpeechModel } = await import('./load.js');
          candidate = await loadSpeechModel({
            ...options,
            signal: aborter.signal,
            onProgress: (event) => {
              options.onProgress?.(event);
              if (current())
                publish({
                  progress: event,
                  ...(event.warning
                    ? {
                        loadWarnings: Object.freeze([
                          ...(snapshot.loadWarnings ?? []),
                          event.warning,
                        ]),
                      }
                    : {}),
                });
            },
          });
          if (aborter.signal.aborted) {
            const abandoned = candidate;
            candidate = null;
            const aborted = createAbortError('model loading');
            try {
              await abandoned.dispose();
            } catch (cleanupError) {
              throw new AggregateError(
                [aborted, cleanupError],
                'Aborted model load cleanup failed.',
                {
                  cause: aborted,
                },
              );
            }
            throw aborted;
          }
          if (!current()) {
            const abandoned = candidate;
            candidate = null;
            await abandoned.dispose();
            return;
          }
          const previous = loaded;
          loaded = null;
          if (previous) publish({ modelId: null, supportsStreaming: false });
          const retirementFailures: unknown[] = [];
          try {
            await closeStream();
          } catch (error) {
            retirementFailures.push(error);
          }
          if (previous) {
            try {
              await previous.dispose();
            } catch (error) {
              retirementFailures.push(error);
            }
          }
          if (retirementFailures.length === 1) throw retirementFailures[0];
          if (retirementFailures.length > 1) {
            throw new AggregateError(retirementFailures, 'Previous model cleanup failed.', {
              cause: retirementFailures[0],
            });
          }
          if (aborter.signal.aborted) {
            const abandoned = candidate;
            candidate = null;
            const aborted = createAbortError('model loading');
            try {
              await abandoned.dispose();
            } catch (cleanupError) {
              throw new AggregateError(
                [aborted, cleanupError],
                'Aborted model load cleanup failed.',
                { cause: aborted },
              );
            }
            throw aborted;
          }
          if (!current()) {
            const abandoned = candidate;
            candidate = null;
            await abandoned.dispose();
            return;
          }
          const accepted = candidate;
          loaded = accepted;
          candidate = null;
          publish({
            status: 'ready',
            modelId: accepted.model.info.modelId,
            supportsStreaming: accepted.supportsStreaming,
            progress: snapshot.progress,
            error: null,
          });
        } catch (error) {
          if (candidate && candidate !== loaded) {
            const abandoned = candidate;
            candidate = null;
            try {
              await abandoned.dispose();
            } catch (cleanupError) {
              error = new AggregateError([error, cleanupError], 'Model load and cleanup failed.', {
                cause: error,
              });
            }
          }
          if (aborter.signal.aborted && !current() && !(error instanceof AggregateError)) return;
          if (current()) publish({ status: 'error', error });
          throw error;
        } finally {
          unsubscribeAbort();
          if (loadAbort === aborter) loadAbort = null;
          loadTasks.delete(generation);
          if (current()) loadTask = null;
        }
      });
      loadTasks.set(generation, task);
      loadTask = task;
      await task;
    },
    async startMicrophone(options = {}) {
      const model = assertReady();
      if (operation || loadTask || microphone || microphoneCleanupTask)
        throw new Error('Another controller operation is active.');
      const generation = ++microphoneGeneration;
      microphoneStartGeneration = generation;
      operation = 'microphone';
      publish({ status: 'loading', result: null, partial: null, error: null, microphone: null });
      const current = (): boolean =>
        !disposed && generation === microphoneGeneration && operation === 'microphone';
      const task = (async () => {
        let active: BrowserRealtimeMicrophoneController | null = null;
        try {
          const { createBrowserRealtimeMicrophoneController } =
            await import('./browser-controller.js');
          if (!current()) throw createAbortError('microphone startup');
          active = createBrowserRealtimeMicrophoneController({
            ...options,
            model,
            onTranscript: (update: RealtimeTranscriptionUpdate) => {
              if (microphone !== active || disposed) return;
              publish({
                result: update.canonical,
                partial: update.partial,
                microphone: active?.getState() ?? snapshot.microphone ?? null,
              });
              try {
                options.onTranscript?.(update);
              } catch {
                /* Consumer callbacks cannot interrupt microphone ownership. */
              }
            },
          });
          microphone = active;
          unsubscribeMicrophone = active.subscribe((state) => {
            if (microphone === active && !disposed) publish({ microphone: state });
          });
          await active.start();
          if (!current()) throw createAbortError('microphone startup');
          publish({ status: 'streaming', microphone: active.getState(), error: null });
        } catch (error) {
          let failure = error;
          if (active) {
            try {
              await cleanupMicrophone(active);
            } catch (cleanupError) {
              failure = new AggregateError(
                [error, cleanupError],
                'Microphone startup and cleanup failed.',
                {
                  cause: error,
                },
              );
            }
          }
          if (operation === 'microphone' && generation === microphoneGeneration) {
            operation = null;
            if (!disposed) publish({ status: 'error', error: failure });
          }
          throw failure;
        } finally {
          if (microphoneStartGeneration === generation) microphoneStartTask = null;
        }
      })();
      microphoneStartTask = task;
      await task;
    },
    async stopMicrophone() {
      if (microphoneStopTask) return microphoneStopTask;
      if (operation !== 'microphone' && !microphoneStartTask && !microphone) return;
      const startupGeneration = microphoneGeneration;
      const generation = ++microphoneGeneration;
      const starting = microphoneStartTask;
      const active = microphone;
      const task = (async () => {
        let failure: { readonly error: unknown } | null = null;
        try {
          if (active) await active.stop({ flush: true });
        } catch (error) {
          failure = { error };
        }
        try {
          if (active) await cleanupMicrophone(active);
          if (starting) await starting;
        } catch (error) {
          const intentionalStartupAbort =
            error instanceof Error && error.name === 'AbortError' && startupGeneration < generation;
          if (!intentionalStartupAbort) {
            failure = failure
              ? {
                  error: new AggregateError(
                    [failure.error, error],
                    'Microphone stop and cleanup failed.',
                    { cause: failure.error },
                  ),
                }
              : { error };
          }
        } finally {
          if (operation === 'microphone') operation = null;
          if (!disposed && generation === microphoneGeneration) {
            const lastMicrophoneState = snapshot.microphone ?? (active ? active.getState() : null);
            publish({
              status: failure ? 'error' : 'ready',
              error: failure ? failure.error : null,
              microphone: lastMicrophoneState
                ? {
                    ...lastMicrophoneState,
                    isMicActive: false,
                    micStatus: failure ? 'Microphone error' : 'Microphone stopped',
                    micError:
                      failure?.error instanceof Error
                        ? failure.error.message
                        : lastMicrophoneState.micError,
                  }
                : null,
            });
          }
          microphoneStopTask = null;
        }
        if (failure) throw failure.error;
      })();
      microphoneStopTask = task;
      return task;
    },
    async transcribe(pcm, sampleRate, options = {}) {
      return runTranscription({ pcm, sampleRate }, options);
    },
    async transcribeSource(source, options, audioOptions) {
      return runTranscription({ source }, options, audioOptions);
    },
    async startStreaming(options = {}) {
      const model = assertReady();
      if (operation || loadTask) throw new Error('Another controller operation is active.');
      if (!model.supportsStreaming) {
        const error = new Error(`Model "${model.model.info.modelId}" does not support streaming.`);
        publish({ status: 'error', error });
        throw error;
      }
      operation = 'streaming';
      publish({ status: 'loading', result: null, partial: null, error: null, microphone: null });
      const task = Promise.resolve()
        .then(() => model.createStreamingTranscriber(options))
        .then(async (candidate) => {
          if (disposed) {
            try {
              await candidate.dispose?.();
            } catch (error) {
              startCleanupFailure = { error };
              throw error;
            }
            throw createAbortError('stream startup');
          }
          streaming = candidate;
          pushQueue = Promise.resolve(null as never);
          publish({ status: 'streaming', partial: null, error: null });
        })
        .catch((error: unknown) => {
          operation = null;
          if (!disposed) publish({ status: 'error', error });
          throw error;
        })
        .finally(() => {
          inferenceTask = null;
        });
      inferenceTask = task;
      return task;
    },
    pushAudio(pcm, sampleRate) {
      assertOpen();
      if (operation !== 'streaming' || !streaming || finalizing)
        return Promise.reject(new Error('Streaming has not been started.'));
      const transcriber = streaming;
      const task = pushQueue
        .then(async () => {
          if (disposed || streaming !== transcriber)
            throw new Error('Streaming session is no longer active.');
          const partial = await transcriber.pushAudio(PcmAudioBuffer.fromMono(pcm, sampleRate));
          if (!disposed) publish({ partial });
          return partial;
        })
        .catch((error: unknown) => {
          if (!disposed) publish({ status: 'error', error });
          throw error;
        });
      pushQueue = task;
      return task;
    },
    async finalizeStreaming() {
      assertOpen();
      if (operation !== 'streaming' || !streaming || finalizing)
        throw new Error('Streaming has not been started.');
      finalizing = true;
      const transcriber = streaming;
      let partial: PartialTranscript | undefined;
      const task = (async () => {
        let failure: { readonly error: unknown } | null = null;
        try {
          await pushQueue;
          partial = await transcriber.finalize();
        } catch (error) {
          failure = { error };
        }
        if (streaming === transcriber) {
          try {
            await closeStream();
          } catch (cleanupError) {
            failure = failure
              ? {
                  error: new AggregateError(
                    [failure.error, cleanupError],
                    'Streaming finalization and cleanup failed.',
                    { cause: failure.error },
                  ),
                }
              : { error: cleanupError };
          }
        }
        operation = null;
        if (failure) {
          if (!disposed) publish({ status: 'error', error: failure.error });
          throw failure.error;
        }
        if (!partial) throw new Error('Streaming finalization returned no transcript.');
        if (!disposed) publish({ status: 'ready', partial });
        return partial;
      })().finally(() => {
        finalizing = false;
        inferenceTask = null;
      });
      inferenceTask = task;
      return task;
    },
    async dispose() {
      if (disposeTask) return disposeTask;
      disposed = true;
      microphoneGeneration++;
      loadGeneration++;
      loadAbort?.abort();
      inferenceAbort.abort();
      const activeMicrophone = microphone;
      const microphoneCleanup = activeMicrophone
        ? cleanupMicrophone(activeMicrophone)
        : microphoneCleanupTask;
      publish({ status: 'disposed' });
      listeners.clear();
      disposeTask = (async () => {
        const failures: unknown[] = [];
        const loadResults = await Promise.allSettled([...loadTasks.values()]);
        for (const result of loadResults) {
          if (
            result.status === 'rejected' &&
            !(result.reason instanceof Error && result.reason.name === 'AbortError')
          ) {
            failures.push(result.reason);
          }
        }
        if (loadTask) await loadTask.catch(() => undefined);
        if (microphoneStartTask) await microphoneStartTask.catch(() => undefined);
        if (microphoneCleanup) {
          try {
            await microphoneCleanup;
          } catch (error) {
            failures.push(error);
          }
        }
        if (inferenceTask) await inferenceTask.catch(() => undefined);
        if (startCleanupFailure) {
          failures.push(startCleanupFailure.error);
          startCleanupFailure = null;
        }
        try {
          await pushQueue;
        } catch {
          /* The push caller observes its rejection. */
        }
        try {
          await closeStream();
        } catch (error) {
          failures.push(error);
        }
        const active = loaded;
        loaded = null;
        if (active)
          try {
            await active.dispose();
          } catch (error) {
            failures.push(error);
          }
        if (failures.length === 1) throw failures[0];
        if (failures.length > 1)
          throw new AggregateError(failures, 'Speech controller cleanup failed.', {
            cause: failures[0],
          });
      })();
      return disposeTask;
    },
  };
  return api;
}
