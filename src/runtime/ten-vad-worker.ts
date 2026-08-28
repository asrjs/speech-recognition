import { STREAMING_TIMELINE_CHUNK_FRAMES } from './audio-timeline.js';
import type {
  TenVadInitPayload,
  TenVadResponseMessage,
  TenVadUpdateConfigPayload,
} from './ten-vad-protocol.js';
import { isTenVadWorkerRequest } from './ten-vad-protocol.js';

interface WorkerScopeLike {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  postMessage(message: TenVadResponseMessage, transfer?: Transferable[]): void;
  location: Location;
}

const workerScope = self as unknown as WorkerScopeLike;

let moduleInstance: any = null;
let vadHandle = 0;
let hopSize = STREAMING_TIMELINE_CHUNK_FRAMES;
let threshold = 0.5;
let audioPtr = 0;
let probPtr = 0;
let flagPtr = 0;
let handlePtr = 0;
let accumulator: Float32Array | null = null;
let accumulatorPos = 0;

workerScope.onmessage = async (event: MessageEvent<unknown>) => {
  const message = event.data;
  if (!isTenVadWorkerRequest(message)) {
    return;
  }

  try {
    switch (message.type) {
      case 'INIT':
        await handleInit(message.id, message.payload);
        break;
      case 'PROCESS':
        handleProcess(message.payload.samples, message.payload.globalSampleOffset);
        break;
      case 'RESET':
        handleReset(message.id);
        break;
      case 'UPDATE_CONFIG':
        handleUpdateConfig(message.id, message.payload);
        break;
      case 'DISPOSE':
        handleDispose(message.id);
        break;
      default:
        break;
    }
  } catch (error) {
    respond({
      type: 'ERROR',
      id: message.id ?? 0,
      payload: String(error instanceof Error ? error.message : error),
    });
  }
};

async function handleInit(id: number, config: TenVadInitPayload) {
  hopSize = config.hopSize;
  threshold = config.threshold;
  const { scriptUrl, wasmUrl, fallbackScriptUrl, fallbackWasmUrl } = config;

  try {
    moduleInstance = await loadTenVadModule(scriptUrl, wasmUrl);
  } catch (primaryError) {
    if (!fallbackScriptUrl || !fallbackWasmUrl) {
      throw primaryError;
    }
    try {
      moduleInstance = await loadTenVadModule(fallbackScriptUrl, fallbackWasmUrl);
    } catch (fallbackError) {
      throw new Error(
        formatTenVadLoadError(
          scriptUrl,
          wasmUrl,
          primaryError,
          fallbackScriptUrl,
          fallbackWasmUrl,
          fallbackError,
        ),
      );
    }
  }

  handlePtr = moduleInstance._malloc(4);
  audioPtr = moduleInstance._malloc(hopSize * 2);
  probPtr = moduleInstance._malloc(4);
  flagPtr = moduleInstance._malloc(4);

  const createStatus = moduleInstance._ten_vad_create(handlePtr, hopSize, threshold);
  if (createStatus !== 0) {
    throw new Error(`ten_vad_create failed with code ${createStatus}`);
  }
  vadHandle = moduleInstance.HEAP32[handlePtr >> 2];

  const versionPtr = moduleInstance._ten_vad_get_version();
  const version = moduleInstance.UTF8ToString
    ? moduleInstance.UTF8ToString(versionPtr)
    : `ptr@${versionPtr}`;

  accumulator = new Float32Array(hopSize);
  accumulatorPos = 0;

  respond({ type: 'INIT', id, payload: { success: true, version } });
}

function handleProcess(samples: Float32Array, globalSampleOffset: number) {
  if (!moduleInstance || !vadHandle || !accumulator) return;

  const maxHops = Math.ceil((samples.length + accumulatorPos) / hopSize);
  const probabilities = new Float32Array(maxHops);
  const flags = new Uint8Array(maxHops);
  let hopCount = 0;
  let sampleIndex = 0;
  let firstResultOffset = globalSampleOffset;
  let resultStartSet = false;

  while (sampleIndex < samples.length) {
    while (accumulatorPos < hopSize && sampleIndex < samples.length) {
      accumulator[accumulatorPos++] = samples[sampleIndex++] ?? 0;
    }

    if (accumulatorPos >= hopSize) {
      if (!resultStartSet) {
        firstResultOffset = globalSampleOffset + sampleIndex - hopSize;
        resultStartSet = true;
      }

      for (let index = 0; index < hopSize; index += 1) {
        const clamped = Math.max(-1, Math.min(1, accumulator[index] ?? 0));
        moduleInstance.HEAP16[(audioPtr >> 1) + index] = Math.round(clamped * 32767);
      }

      const processStatus = moduleInstance._ten_vad_process(
        vadHandle,
        audioPtr,
        hopSize,
        probPtr,
        flagPtr,
      );

      if (processStatus === 0) {
        probabilities[hopCount] = moduleInstance.HEAPF32[probPtr >> 2] ?? 0;
        flags[hopCount] = moduleInstance.HEAP32[flagPtr >> 2] ?? 0;
        hopCount += 1;
      }

      accumulatorPos = 0;
    }
  }

  if (hopCount > 0) {
    const trimmedProbabilities = probabilities.slice(0, hopCount);
    const trimmedFlags = flags.slice(0, hopCount);
    workerScope.postMessage(
      {
        type: 'RESULT',
        payload: {
          probabilities: trimmedProbabilities,
          flags: trimmedFlags,
          globalSampleOffset: firstResultOffset,
          hopCount,
        },
      },
      [trimmedProbabilities.buffer, trimmedFlags.buffer],
    );
  }
}

function handleReset(id: number) {
  if (accumulator) {
    accumulator.fill(0);
    accumulatorPos = 0;
  }

  if (moduleInstance && vadHandle) {
    moduleInstance._ten_vad_destroy(vadHandle);
    const createStatus = moduleInstance._ten_vad_create(handlePtr, hopSize, threshold);
    if (createStatus === 0) {
      vadHandle = moduleInstance.HEAP32[handlePtr >> 2];
    } else {
      vadHandle = 0;
      respond({
        type: 'ERROR',
        id,
        payload: `Reset failed: _ten_vad_create returned ${createStatus}`,
      });
      return;
    }
  }

  respond({ type: 'RESET', id, payload: { success: true } });
}

function handleUpdateConfig(id: number, config: TenVadUpdateConfigPayload) {
  const hopSizeChanged = config.hopSize !== hopSize;

  if (hopSizeChanged) {
    hopSize = Math.floor(config.hopSize);
  }
  threshold = config.threshold;

  if (!moduleInstance) {
    respond({ type: 'UPDATE_CONFIG', id, payload: { success: true } });
    return;
  }

  if (hopSizeChanged) {
    if (audioPtr) {
      moduleInstance._free(audioPtr);
    }
    audioPtr = moduleInstance._malloc(hopSize * 2);
    accumulator = new Float32Array(hopSize);
    accumulatorPos = 0;
  }

  handleReset(id);
}

function handleDispose(id: number) {
  if (moduleInstance && vadHandle) {
    moduleInstance._ten_vad_destroy(vadHandle);
    moduleInstance._free(audioPtr);
    moduleInstance._free(probPtr);
    moduleInstance._free(flagPtr);
    moduleInstance._free(handlePtr);
  }

  moduleInstance = null;
  vadHandle = 0;
  accumulator = null;
  respond({ type: 'DISPOSE', id, payload: { success: true } });
}

function respond(message: TenVadResponseMessage) {
  workerScope.postMessage(message);
}

async function loadTenVadModule(scriptUrl: string, wasmUrl: string) {
  const response = await fetch(scriptUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch TEN-VAD script: ${response.status}`);
  }

  const jsText = await response.text();
  const blobUrl = URL.createObjectURL(new Blob([jsText], { type: 'application/javascript' }));

  try {
    const moduleImport = await import(/* @vite-ignore */ blobUrl);
    const createVADModule = moduleImport?.default;
    if (typeof createVADModule !== 'function') {
      throw new Error(`TEN-VAD module at ${scriptUrl} did not export a default factory.`);
    }
    return createVADModule({
      locateFile(file: string) {
        if (file.endsWith('.wasm')) {
          return wasmUrl;
        }
        return file;
      },
    });
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

function formatTenVadLoadError(
  scriptUrl: string,
  wasmUrl: string,
  primaryError: unknown,
  fallbackScriptUrl: string,
  fallbackWasmUrl: string,
  fallbackError: unknown,
) {
  const primaryDetail = formatTenVadLoadDetail(scriptUrl, wasmUrl, primaryError);
  const fallbackDetail = formatTenVadLoadDetail(fallbackScriptUrl, fallbackWasmUrl, fallbackError);
  return `${primaryDetail} Fallback to packaged assets also failed. ${fallbackDetail}`;
}

function formatTenVadLoadDetail(scriptUrl: string, wasmUrl: string, error: unknown) {
  const detail = error instanceof Error ? error.message : String(error);
  const hint = isLikelyGitHubRawUrl(scriptUrl)
    ? ' Raw GitHub URLs are not a reliable runtime host for TEN-VAD browser assets. Prefer the packaged local assets or serve ten_vad.js and ten_vad.wasm from your own app.'
    : '';
  return `Failed to load TEN-VAD from ${scriptUrl} (wasm: ${wasmUrl}). ${detail}.${hint}`;
}

function isLikelyGitHubRawUrl(url: string) {
  return url.includes('raw.githubusercontent.com/') || url.includes('githubusercontent.com/');
}

export {};
