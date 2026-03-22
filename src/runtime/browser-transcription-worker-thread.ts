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

function assertNever(value: never): never {
  throw new Error(`Unknown browser transcription worker request: ${String(value)}`);
}

function getMeta(error: Error | null = null) {
  return {
    state: loadedModel ? 'ready' : 'idle',
    error: error?.message ?? null,
    model: loadedModel
      ? {
          source: loadSource!,
          modelId: loadedModel.model?.id,
          info: loadedModel.model?.info ?? null,
          selection: loadedModel.selection ?? null,
        }
      : null,
  };
}

async function disposeLoadedModel(): Promise<void> {
  if (!loadedModel) {
    return;
  }
  try {
    await loadedModel.dispose();
  } finally {
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
      if (!loadedModel) {
        throw new Error('No worker transcription model is loaded.');
      }
      return await loadedModel.transcribeMonoPcm(
        message.payload.pcm,
        message.payload.sampleRate,
        message.payload.options ?? undefined,
      );
    }
    case 'DISPOSE_MODEL': {
      await disposeLoadedModel();
      return null;
    }
  }
  return assertNever(message);
}

let requestChain = Promise.resolve();

workerScope.onmessage = (event: MessageEvent<WorkerRequestMessage>) => {
  const message = event.data;
  requestChain = requestChain
    .catch(() => undefined)
    .then(async () => {
      try {
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
          id: message.id,
          type: 'ERROR',
          payload: resolvedError.message,
          meta: getMeta(resolvedError),
        };
        workerScope.postMessage(response);
      }
    });
};
