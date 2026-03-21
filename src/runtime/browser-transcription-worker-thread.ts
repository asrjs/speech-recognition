import { loadSpeechModel } from './load.js';
import { loadSpeechModelFromLocalEntries } from './local-browser.js';

interface WorkerScopeLike {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
}

interface WorkerRequestMessage {
  readonly id: number;
  readonly type: 'LOAD_BUILT_IN_MODEL' | 'LOAD_LOCAL_MODEL' | 'TRANSCRIBE_MONO_PCM' | 'DISPOSE_MODEL';
  readonly payload: any;
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

function getMeta(error: Error | null = null) {
  return {
    state: loadedModel ? 'ready' : 'idle',
    error: error?.message ?? null,
    model: loadedModel
      ? {
          source: loadSource ?? 'built-in',
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
    default: {
      throw new Error(`Unknown browser transcription worker request: ${String(message.type)}`);
    }
  }
}

workerScope.onmessage = async (event: MessageEvent<WorkerRequestMessage>) => {
  const message = event.data;
  try {
    const payload = await handleRequest(message);
    workerScope.postMessage({
      id: message.id,
      type: 'SUCCESS',
      payload,
      meta: getMeta(),
    });
  } catch (error) {
    const resolvedError = error instanceof Error ? error : new Error(String(error));
    workerScope.postMessage({
      id: message.id,
      type: 'ERROR',
      payload: resolvedError.message,
      meta: getMeta(resolvedError),
    });
  }
};
