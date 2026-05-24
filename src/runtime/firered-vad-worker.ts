import { STREAMING_TIMELINE_CHUNK_FRAMES } from './audio-timeline.js';
import { FireredVadStreamPacked } from './firered-vad/api/classes.js';

interface WorkerScopeLike {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  location: Location;
}

const workerScope = self as unknown as WorkerScopeLike;

let vadEngine: FireredVadStreamPacked | null = null;
let hopSize = STREAMING_TIMELINE_CHUNK_FRAMES;
let threshold = 0.5;
let accumulator: Float32Array | null = null;
let accumulatorPos = 0;
let processQueue: Promise<void> = Promise.resolve();

workerScope.onmessage = async (event: MessageEvent) => {
  const message = event.data as any;

  try {
    switch (message.type) {
      case 'INIT':
        await handleInit(message.id, message.payload ?? {});
        break;
      case 'PROCESS':
        processQueue = processQueue.then(() =>
          handleProcess(
            message.payload?.samples ?? new Float32Array(0),
            message.payload?.globalSampleOffset ?? 0,
          ),
        );
        await processQueue;
        break;
      case 'RESET':
        handleReset(message.id);
        break;
      case 'UPDATE_CONFIG':
        handleUpdateConfig(message.id, message.payload ?? {});
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

function resolveModelUrl(config: Record<string, unknown>): string | undefined {
  if (typeof config.modelUrl === 'string' && config.modelUrl.length > 0) {
    return config.modelUrl;
  }
  if (typeof config.modelDir !== 'string' || config.modelDir.length === 0) {
    return undefined;
  }
  const modelDir = config.modelDir;
  const filename =
    typeof config.modelFilename === 'string' && config.modelFilename.length > 0
      ? config.modelFilename
      : 'fireredvad_stream_vad_with_cache.onnx';
  if (/^(https?:\/\/|file:\/\/)/i.test(modelDir)) {
    const base = modelDir.endsWith('/') ? modelDir : `${modelDir}/`;
    return new URL(filename, base).href;
  }
  const separator = modelDir.endsWith('/') || modelDir.endsWith('\\') ? '' : '/';
  return `${modelDir}${separator}${filename}`;
}

async function handleInit(id: number, config: Record<string, unknown>) {
  const requestedHopSize = Number(config.hopSize);
  hopSize =
    Number.isFinite(requestedHopSize) && requestedHopSize > 0
      ? Math.round(requestedHopSize)
      : STREAMING_TIMELINE_CHUNK_FRAMES;
  threshold =
    Number.isFinite(config.threshold) && Number(config.threshold) >= 0
      ? Number(config.threshold)
      : 0.5;
  const wasmNumThreads =
    Number.isFinite(config.wasmNumThreads) && Number(config.wasmNumThreads) > 0
      ? Math.floor(Number(config.wasmNumThreads))
      : undefined;
  const modelUrl = resolveModelUrl(config);
  const cmvnJsonUrl =
    typeof config.cmvnJsonUrl === 'string' && config.cmvnJsonUrl.length > 0
      ? config.cmvnJsonUrl
      : undefined;
  const wasmPaths =
    typeof config.wasmPaths === 'string' ||
    (typeof config.wasmPaths === 'object' && config.wasmPaths !== null)
      ? config.wasmPaths
      : undefined;

  if (vadEngine) {
    await vadEngine.dispose();
    vadEngine = null;
  }

  vadEngine = await FireredVadStreamPacked.create({
    modelUrl,
    cmvnJsonUrl,
    threshold,
    wasmPaths: wasmPaths as string | Record<string, string> | undefined,
    wasmNumThreads,
    cacheAssets: config.cacheAssets !== false,
  });

  accumulator = new Float32Array(hopSize);
  accumulatorPos = 0;

  respond({ type: 'INIT', id, payload: { success: true, version: 'firered-vad-web' } });
}

async function handleProcess(samples: Float32Array, globalSampleOffset: number) {
  if (!vadEngine || !accumulator) return;

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

      const pcm16 = new Int16Array(hopSize);
      for (let index = 0; index < hopSize; index += 1) {
        const clamped = Math.max(-1, Math.min(1, accumulator[index] ?? 0));
        pcm16[index] = Math.round(clamped * 32767);
      }

      const result = await vadEngine.process_stream(pcm16);
      probabilities[hopCount] = result.confidence;
      flags[hopCount] = result.is_speech ? 1 : 0;
      hopCount += 1;

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

  vadEngine?.reset();

  respond({ type: 'RESET', id, payload: { success: true } });
}

function handleUpdateConfig(id: number, config: Record<string, unknown>) {
  const nextHopSize = Number(config.hopSize);
  const nextThreshold = Number(config.threshold);
  const hopSizeChanged = Number.isFinite(nextHopSize) && nextHopSize > 0 && nextHopSize !== hopSize;

  if (hopSizeChanged) {
    hopSize = Math.floor(nextHopSize);
  }
  if (Number.isFinite(nextThreshold) && nextThreshold >= 0) {
    threshold = nextThreshold;
  }

  if (!vadEngine) {
    respond({ type: 'UPDATE_CONFIG', id, payload: { success: true } });
    return;
  }

  if (hopSizeChanged) {
    accumulator = new Float32Array(hopSize);
    accumulatorPos = 0;
  }

  vadEngine.reset();
  respond({ type: 'UPDATE_CONFIG', id, payload: { success: true } });
}

async function handleDispose(id: number) {
  if (vadEngine) {
    await vadEngine.dispose();
  }
  vadEngine = null;
  accumulator = null;
  respond({ type: 'DISPOSE', id, payload: { success: true } });
}

function respond(message: unknown) {
  workerScope.postMessage(message);
}

export {};
