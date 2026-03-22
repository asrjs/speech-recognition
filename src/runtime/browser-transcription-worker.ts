import type {
  BaseTranscriptionOptions,
  MonoPcmInput,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../types/index.js';
import type { LoadSpeechModelOptions } from './load.js';
import type { LoadSpeechModelFromLocalEntriesOptions } from './local-browser.js';

interface BrowserTranscriptionWorkerLike {
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
}

interface PendingWorkerRequest {
  readonly resolve: (value: unknown) => void;
  readonly reject: (reason?: unknown) => void;
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
>;

type BrowserTranscriptionLocalLoadRequest = Omit<
  LoadSpeechModelFromLocalEntriesOptions,
  'runtime' | 'hooks' | 'onProgress'
>;

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

export function createBrowserTranscriptionWorkerClient(
  options: BrowserTranscriptionWorkerClientOptions = {},
): BrowserTranscriptionWorkerClient {
  const workerFactory = options.workerFactory ?? defaultWorkerFactory;
  let worker: BrowserTranscriptionWorkerLike | null = null;
  let requestId = 0;
  let disposed = false;
  let status: BrowserTranscriptionWorkerClientStatus = {
    state: 'idle',
    error: null,
    model: null,
  };
  const pending = new Map<number, PendingWorkerRequest>();

  const fail = (error: unknown): void => {
    const message = error instanceof Error ? error.message : String(error);
    status = {
      ...status,
      state: disposed ? 'disposed' : 'error',
      error: message,
    };
    for (const [, request] of pending) {
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
      fail(new Error(event?.message || 'Browser transcription worker error.'));
    };
    return worker;
  };

  const sendRequest = (
    type: BrowserTranscriptionWorkerRequestType,
    payload: unknown,
    transfer: Transferable[] = [],
  ): Promise<unknown> => {
    const activeWorker = ensureWorker();
    const id = ++requestId;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      activeWorker.postMessage({ id, type, payload }, transfer);
    });
  };

  return {
    getStatus() {
      return status;
    },
    async loadBuiltInModel<TLoadOptions = unknown>(
      request: BrowserTranscriptionBuiltInLoadRequest<TLoadOptions>,
    ): Promise<BrowserTranscriptionWorkerInfo> {
      status = {
        ...status,
        state: 'loading',
        error: null,
      };
      return (await sendRequest('LOAD_BUILT_IN_MODEL', request)) as BrowserTranscriptionWorkerInfo;
    },
    async loadLocalModel(
      request: BrowserTranscriptionLocalLoadRequest,
    ): Promise<BrowserTranscriptionWorkerInfo> {
      status = {
        ...status,
        state: 'loading',
        error: null,
      };
      return (await sendRequest('LOAD_LOCAL_MODEL', request)) as BrowserTranscriptionWorkerInfo;
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
      const chunk = new Float32Array(pcm);
      return (await sendRequest(
        'TRANSCRIBE_MONO_PCM',
        {
          pcm: chunk,
          sampleRate,
          options: options ?? null,
        },
        [chunk.buffer],
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
