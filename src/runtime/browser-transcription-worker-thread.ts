import { loadSpeechModel } from './load.js';
import { loadSpeechModelFromLocalEntries } from './local-browser.js';
import type { LoadSpeechModelOptions } from './load.js';
import type { LoadSpeechModelFromLocalEntriesOptions } from './local-browser.js';

interface WorkerScopeLike {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
}

type WorkerBuiltInLoadPayload = LoadSpeechModelOptions<unknown>;
type WorkerLocalLoadPayload = LoadSpeechModelFromLocalEntriesOptions;
type WorkerTranscribePayload = {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
  readonly options?: Record<string, unknown> | null;
};

type WorkerRequestMessage =
  | {
      readonly id: number;
      readonly type: 'LOAD_BUILT_IN_MODEL';
      readonly payload: WorkerBuiltInLoadPayload;
    }
  | {
      readonly id: number;
      readonly type: 'LOAD_LOCAL_MODEL';
      readonly payload: WorkerLocalLoadPayload;
    }
  | {
      readonly id: number;
      readonly type: 'TRANSCRIBE_MONO_PCM';
      readonly payload: WorkerTranscribePayload;
    }
  | {
      readonly id: number;
      readonly type: 'CANCEL_TRANSCRIBE';
      readonly payload: { readonly requestId: number };
    }
  | {
      readonly id: number;
      readonly type: 'DISPOSE_MODEL';
      readonly payload: null;
    };

interface WorkerSuccessMessage {
  readonly id: number;
  readonly type: 'SUCCESS';
  readonly payload: unknown;
  readonly meta: ReturnType<typeof getMeta>;
}

interface WorkerErrorMessage {
  readonly id: number;
  readonly type: 'ERROR';
  readonly payload: string;
  readonly meta: ReturnType<typeof getMeta>;
}

type LoadedModelLike = {
  readonly model?: { readonly info?: unknown; readonly id?: string };
  readonly selection?: unknown;
  transcribeMonoPcm(
    pcm: Float32Array,
    sampleRate: number,
    options?: Record<string, unknown> | null,
  ): Promise<unknown>;
  dispose(): Promise<void>;
};

const workerScope = self as unknown as WorkerScopeLike;
let loadedModel: LoadedModelLike | null = null;
let loadSource: 'built-in' | 'local' | null = null;
const activeTranscribes = new Map<number, AbortController>();
const queuedTranscribes = new Set<number>();
const canceledTranscribes = new Set<number>();

function assertNever(value: never): never {
  throw new Error(`Unknown browser transcription worker request: ${String(value)}`);
}

function sanitizeForPostMessage<T>(value: T): T | null {
  if (value == null) {
    return null;
  }
  try {
    if (typeof structuredClone === 'function') {
      return structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value)) as T;
  } catch {
    return null;
  }
}

function getMeta(error: Error | null = null) {
  return {
    state: loadedModel ? 'ready' : 'idle',
    error: error?.message ?? null,
    model: loadedModel
      ? {
          source: loadSource!,
          modelId: loadedModel.model?.id,
          info: sanitizeForPostMessage(loadedModel.model?.info ?? null),
          selection: sanitizeForPostMessage(loadedModel.selection ?? null),
        }
      : null,
  };
}

async function disposeLoadedModel(): Promise<void> {
  const modelToDispose = loadedModel;
  if (!modelToDispose) {
    return;
  }
  await modelToDispose.dispose();
  if (loadedModel === modelToDispose) {
    loadedModel = null;
    loadSource = null;
  }
}

async function handleRequest(message: WorkerRequestMessage): Promise<unknown> {
  switch (message.type) {
    case 'LOAD_BUILT_IN_MODEL': {
      await disposeLoadedModel();
      loadedModel = (await loadSpeechModel(message.payload)) as LoadedModelLike;
      loadSource = 'built-in';
      return getMeta().model;
    }
    case 'LOAD_LOCAL_MODEL': {
      await disposeLoadedModel();
      loadedModel = (await loadSpeechModelFromLocalEntries(message.payload)) as LoadedModelLike;
      loadSource = 'local';
      return getMeta().model;
    }
    case 'TRANSCRIBE_MONO_PCM': {
      queuedTranscribes.delete(message.id);
      if (canceledTranscribes.delete(message.id)) {
        throw new Error('Browser transcription request was canceled before decode began.');
      }
      if (!loadedModel) {
        throw new Error('No worker transcription model is loaded.');
      }
      const controller = new AbortController();
      activeTranscribes.set(message.id, controller);
      try {
        return await loadedModel.transcribeMonoPcm(
          message.payload.pcm,
          message.payload.sampleRate,
          {
            ...(message.payload.options ?? {}),
            signal: controller.signal,
          },
        );
      } finally {
        activeTranscribes.delete(message.id);
      }
    }
    case 'CANCEL_TRANSCRIBE': {
      activeTranscribes.get(message.payload.requestId)?.abort();
      return null;
    }
    case 'DISPOSE_MODEL': {
      await disposeLoadedModel();
      return null;
    }
  }
  return assertNever(message);
}

let requestChain = Promise.resolve();

workerScope.onmessage = (event: MessageEvent<unknown>) => {
  const rawMessage = event.data;

  // Cancellation must bypass requestChain: a transcribe request is awaited
  // there, so queueing the cancel behind it would never reach the active
  // AbortController until inference had already finished.
  if (
    typeof rawMessage === 'object' &&
    rawMessage !== null &&
    (rawMessage as { type?: unknown }).type === 'CANCEL_TRANSCRIBE' &&
    typeof (rawMessage as { payload?: unknown }).payload === 'object' &&
    (rawMessage as { payload?: { requestId?: unknown } }).payload !== null &&
    typeof (rawMessage as { payload: { requestId?: unknown } }).payload.requestId === 'number'
  ) {
    const requestId = (rawMessage as { payload: { requestId: number } }).payload.requestId;
    const active = activeTranscribes.get(requestId);
    if (active) {
      active.abort();
    } else if (queuedTranscribes.has(requestId)) {
      canceledTranscribes.add(requestId);
    }
    return;
  }

  const requestId =
    typeof rawMessage === 'object' &&
    rawMessage !== null &&
    'id' in rawMessage &&
    typeof (rawMessage as { id?: unknown }).id === 'number'
      ? (rawMessage as { id: number }).id
      : -1;
  if (
    typeof rawMessage === 'object' &&
    rawMessage !== null &&
    (rawMessage as { type?: unknown }).type === 'TRANSCRIBE_MONO_PCM' &&
    typeof (rawMessage as { id?: unknown }).id === 'number'
  ) {
    queuedTranscribes.add(requestId);
  }
  requestChain = requestChain
    .catch(() => undefined)
    .then(async () => {
      try {
        if (
          typeof rawMessage !== 'object' ||
          rawMessage === null ||
          typeof (rawMessage as { id?: unknown }).id !== 'number' ||
          typeof (rawMessage as { type?: unknown }).type !== 'string'
        ) {
          throw new Error('Invalid browser transcription worker request.');
        }

        const message = rawMessage as WorkerRequestMessage;
        const payload = await handleRequest(message);
        const response: WorkerSuccessMessage = {
          id: message.id,
          type: 'SUCCESS',
          payload,
          meta: getMeta(),
        };
        workerScope.postMessage(response);
      } catch (error) {
        const resolvedError = error instanceof Error ? error : new Error(String(error));
        const response: WorkerErrorMessage = {
          id: requestId,
          type: 'ERROR',
          payload: resolvedError.message,
          meta: getMeta(resolvedError),
        };
        workerScope.postMessage(response);
      }
    });
};
