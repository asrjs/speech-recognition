import type {
  BaseTranscriptionOptions,
  MonoPcmInput,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../types/index.js';
import type { LoadSpeechModelOptions } from './load.js';
import type { LoadSpeechModelFromLocalEntriesOptions } from './local-browser.js';
import { AssetLoadAbortedError, subscribeToAbortSignal } from '../io/abort.js';

interface BrowserTranscriptionWorkerLike {
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
}

interface PendingWorkerRequest {
  readonly resolve: (value: unknown) => void;
  readonly reject: (reason?: unknown) => void;
  readonly cleanup: () => void;
}

type BrowserTranscriptionWorkerRequestType =
  | 'LOAD_BUILT_IN_MODEL'
  | 'LOAD_LOCAL_MODEL'
  | 'TRANSCRIBE_MONO_PCM'
  | 'DISPOSE_MODEL';

type BrowserTranscriptionWorkerState = 'idle' | 'loading' | 'ready' | 'error' | 'disposed';

type BrowserTranscriptionBuiltInLoadRequest<TLoadOptions = unknown> = Omit<
  LoadSpeechModelOptions<TLoadOptions>,
  'runtime' | 'hooks' | 'onProgress'
> & {
  readonly signal?: { readonly aborted: boolean } | null;
};

type BrowserTranscriptionLocalLoadRequest = Omit<
  LoadSpeechModelFromLocalEntriesOptions,
  'runtime' | 'hooks' | 'onProgress'
> & {
  readonly signal?: { readonly aborted: boolean } | null;
};

interface BrowserTranscriptionWorkerInfo {
  readonly modelId?: string;
  readonly source: 'built-in' | 'local';
  readonly selection?: unknown;
  readonly info?: unknown;
}

interface BrowserTranscriptionWorkerResponseMeta {
  readonly state?: BrowserTranscriptionWorkerState;
  readonly error?: string | null;
  readonly model?: BrowserTranscriptionWorkerInfo | null;
}

interface BrowserTranscriptionWorkerSuccessMessage {
  readonly id: number;
  readonly type: 'SUCCESS';
  readonly payload: unknown;
  readonly meta?: BrowserTranscriptionWorkerResponseMeta;
}

interface BrowserTranscriptionWorkerErrorMessage {
  readonly id: number;
  readonly type: 'ERROR';
  readonly payload: unknown;
  readonly meta?: BrowserTranscriptionWorkerResponseMeta;
}

type BrowserTranscriptionWorkerResponseMessage =
  | BrowserTranscriptionWorkerSuccessMessage
  | BrowserTranscriptionWorkerErrorMessage;

export interface BrowserTranscriptionWorkerClientStatus {
  readonly state: BrowserTranscriptionWorkerState;
  readonly error: string | null;
  readonly model: BrowserTranscriptionWorkerInfo | null;
}

export interface BrowserTranscriptionWorkerClientOptions {
  readonly workerFactory?: () => BrowserTranscriptionWorkerLike;
  readonly signal?: { readonly aborted: boolean } | null;
}

export interface BrowserTranscriptionWorkerClient {
  getStatus(): BrowserTranscriptionWorkerClientStatus;
  loadBuiltInModel<TLoadOptions = unknown>(
    request: BrowserTranscriptionBuiltInLoadRequest<TLoadOptions>,
  ): Promise<BrowserTranscriptionWorkerInfo>;
  loadLocalModel(
    request: BrowserTranscriptionLocalLoadRequest,
  ): Promise<BrowserTranscriptionWorkerInfo>;
  transcribeMonoPcm<
    TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
    TNative = unknown,
    TFlavor extends TranscriptResponseFlavor = 'canonical',
  >(
    pcm: MonoPcmInput,
    sampleRate: number,
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<TranscriptResponse<TNative, TFlavor>>;
  disposeModel(): Promise<void>;
  dispose(): Promise<void>;
}

function defaultWorkerFactory(): BrowserTranscriptionWorkerLike {
  return new Worker(new URL('./browser-transcription-worker-thread.js', import.meta.url), {
    type: 'module',
  });
}

function createBrowserTranscriptionAbortedError(
  stage: 'browser-transcription-load' | 'browser-transcription-transcribe',
): AssetLoadAbortedError {
  return new AssetLoadAbortedError(stage);
}

function abortStageFor(
  type: BrowserTranscriptionWorkerRequestType,
): 'browser-transcription-load' | 'browser-transcription-transcribe' {
  return type === 'TRANSCRIBE_MONO_PCM'
    ? 'browser-transcription-transcribe'
    : 'browser-transcription-load';
}

function omitSignal<T extends Record<string, unknown>>(value: T): Omit<T, 'signal'> {
  const { signal: _signal, ...rest } = value;
  return rest;
}

function clonePayloadForWorker(
  type: BrowserTranscriptionWorkerRequestType,
  payload: unknown,
): unknown {
  if (!payload || typeof payload !== 'object') {
    return payload;
  }
  if (type === 'TRANSCRIBE_MONO_PCM') {
    const request = payload as {
      pcm: Float32Array;
      sampleRate: number;
      options?: Record<string, unknown> | null;
    };
    const options = request.options;
    return {
      pcm: request.pcm,
      sampleRate: request.sampleRate,
      options:
        options && typeof options === 'object'
          ? omitSignal(options as Record<string, unknown>)
          : (options ?? null),
    };
  }
  return omitSignal(payload as Record<string, unknown>);
}

export function createBrowserTranscriptionWorkerClient(
  options: BrowserTranscriptionWorkerClientOptions = {},
): BrowserTranscriptionWorkerClient {
  const workerFactory = options.workerFactory ?? defaultWorkerFactory;
  const clientSignal = options.signal;
  let worker: BrowserTranscriptionWorkerLike | null = null;
  let requestId = 0;
  let disposed = false;
  let status: BrowserTranscriptionWorkerClientStatus = {
    state: 'idle',
    error: null,
    model: null,
  };
  const pending = new Map<number, PendingWorkerRequest>();

  const resetWorker = (): void => {
    if (!worker) {
      return;
    }
    try {
      worker.terminate();
    } finally {
      worker = null;
    }
  };

  const fail = (error: unknown): void => {
    const message = error instanceof Error ? error.message : String(error);
    status = {
      ...status,
      state: disposed ? 'disposed' : 'error',
      error: message,
    };
    for (const [, request] of pending) {
      request.cleanup();
      request.reject(new Error(message));
    }
    pending.clear();
  };

  const handleMessage = (message: BrowserTranscriptionWorkerResponseMessage): void => {
    const request = pending.get(message.id);
    if (!request) {
      return;
    }
    pending.delete(message.id);
    request.cleanup();

    if (message.type === 'ERROR') {
      const error = new Error(String(message.payload ?? 'Worker request failed.'));
      status = {
        ...status,
        state: disposed ? 'disposed' : 'error',
        error: error.message,
      };
      request.reject(error);
      return;
    }

    if (message.meta?.state) {
      status = {
        state: message.meta.state,
        error: message.meta.error ?? null,
        model: message.meta.model ?? null,
      };
    }
    request.resolve(message.payload);
  };

  const ensureWorker = (): BrowserTranscriptionWorkerLike => {
    if (disposed) {
      throw new Error('BrowserTranscriptionWorkerClient has been disposed.');
    }
    if (worker) {
      return worker;
    }
    worker = workerFactory();
    worker.onmessage = (event) => handleMessage(event.data);
    worker.onerror = (event) => {
      resetWorker();
      fail(new Error(event?.message || 'Browser transcription worker error.'));
    };
    return worker;
  };

  const sendRequest = (
    type: BrowserTranscriptionWorkerRequestType,
    payload: unknown,
    transfer: Transferable[] = [],
    signal?: { readonly aborted: boolean } | null,
  ): Promise<unknown> => {
    const abortSignal = signal ?? clientSignal;
    const stage = abortStageFor(type);
    if (abortSignal?.aborted) {
      throw createBrowserTranscriptionAbortedError(stage);
    }
    const activeWorker = ensureWorker();
    if (abortSignal?.aborted) {
      resetWorker();
      throw createBrowserTranscriptionAbortedError(stage);
    }
    const id = ++requestId;
    return new Promise((resolve, reject) => {
      let cleanupAbort = (): void => undefined;
      const onAbort = () => {
        const pendingRequest = pending.get(id);
        if (!pendingRequest) {
          cleanupAbort();
          return;
        }
        const error = createBrowserTranscriptionAbortedError(stage);

        // A transcription can be canceled cooperatively because the worker
        // owns the loaded model. Reject this caller immediately, then let the
        // worker abort its model-level decode without tearing down the model
        // or forcing the next request to reload it.
        if (type === 'TRANSCRIBE_MONO_PCM') {
          pending.delete(id);
          pendingRequest.cleanup();
          pendingRequest.reject(error);
          try {
            activeWorker.postMessage({
              id: ++requestId,
              type: 'CANCEL_TRANSCRIBE',
              payload: { requestId: id },
            });
            status = {
              ...status,
              error: null,
            };
          } catch {
            // A worker that cannot accept the cancellation command is no
            // longer safe to reuse. Preserve the old teardown fallback only
            // for this transport failure.
            resetWorker();
            status = {
              state: disposed ? 'disposed' : 'idle',
              error: null,
              model: null,
            };
          }
          return;
        }

        resetWorker();
        status = {
          state: disposed ? 'disposed' : 'idle',
          error: null,
          model: null,
        };
        for (const [, request] of pending) {
          request.cleanup();
          request.reject(error);
        }
        pending.clear();
      };
      pending.set(id, {
        resolve,
        reject,
        cleanup: () => cleanupAbort(),
      });
      cleanupAbort = subscribeToAbortSignal(abortSignal, onAbort);
      if (abortSignal?.aborted) {
        onAbort();
      }
      if (!pending.has(id)) {
        return;
      }
      try {
        activeWorker.postMessage(
          { id, type, payload: clonePayloadForWorker(type, payload) },
          transfer,
        );
      } catch (error) {
        pending.delete(id);
        cleanupAbort();
        resetWorker();
        fail(error);
        reject(error);
      }
    });
  };

  return {
    getStatus() {
      return status;
    },
    async loadBuiltInModel<TLoadOptions = unknown>(
      request: BrowserTranscriptionBuiltInLoadRequest<TLoadOptions>,
    ): Promise<BrowserTranscriptionWorkerInfo> {
      const abortSignal = request.signal ?? clientSignal;
      if (abortSignal?.aborted) {
        throw createBrowserTranscriptionAbortedError('browser-transcription-load');
      }
      status = {
        ...status,
        state: 'loading',
        error: null,
      };
      return (await sendRequest(
        'LOAD_BUILT_IN_MODEL',
        request,
        [],
        abortSignal,
      )) as BrowserTranscriptionWorkerInfo;
    },
    async loadLocalModel(
      request: BrowserTranscriptionLocalLoadRequest,
    ): Promise<BrowserTranscriptionWorkerInfo> {
      const abortSignal = request.signal ?? clientSignal;
      if (abortSignal?.aborted) {
        throw createBrowserTranscriptionAbortedError('browser-transcription-load');
      }
      status = {
        ...status,
        state: 'loading',
        error: null,
      };
      return (await sendRequest(
        'LOAD_LOCAL_MODEL',
        request,
        [],
        abortSignal,
      )) as BrowserTranscriptionWorkerInfo;
    },
    async transcribeMonoPcm<
      TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
      TNative = unknown,
      TFlavor extends TranscriptResponseFlavor = 'canonical',
    >(
      pcm: MonoPcmInput,
      sampleRate: number,
      options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
    ): Promise<TranscriptResponse<TNative, TFlavor>> {
      const abortSignal = options?.signal ?? clientSignal;
      if (abortSignal?.aborted) {
        throw createBrowserTranscriptionAbortedError('browser-transcription-transcribe');
      }
      const chunk = new Float32Array(pcm);
      return (await sendRequest(
        'TRANSCRIBE_MONO_PCM',
        {
          pcm: chunk,
          sampleRate,
          options: options ?? null,
        },
        [chunk.buffer],
        abortSignal,
      )) as TranscriptResponse<TNative, TFlavor>;
    },
    async disposeModel(): Promise<void> {
      if (!worker || disposed) {
        status = {
          state: disposed ? 'disposed' : 'idle',
          error: null,
          model: null,
        };
        return;
      }
      await sendRequest('DISPOSE_MODEL', null);
      status = {
        state: 'idle',
        error: null,
        model: null,
      };
    },
    async dispose(): Promise<void> {
      if (disposed) {
        return;
      }
      disposed = true;
      if (worker) {
        worker.terminate();
        worker = null;
      }
      fail(new Error('BrowserTranscriptionWorkerClient disposed.'));
      status = {
        state: 'disposed',
        error: null,
        model: null,
      };
    },
  };
}
