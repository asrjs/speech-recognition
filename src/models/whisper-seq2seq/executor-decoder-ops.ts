import type { AbortSignalLike } from '../../types/index.js';
import { argmax, tokenQualityFromLogits } from '../../inference/index.js';
import { nowMs } from '../../runtime/timing.js';
import {
  whisperDecode,
  type WhisperCoreSession,
} from './core.js';
import {
  cloneDecoderKvTensorDataForInput,
  copyLogitsTail,
  copyTensorData,
  countTensorLocations,
  createWhisperCachePosition,
  createWhisperInt64TokenIds,
  disposeGpuKv,
  disposeGpuTensor,
  disposeOrtTensor,
  disposeOwnedDecoderFeeds,
  disposeReplacedGpuKv,
  disposeUnusedDecoderOutputs,
  getWhisperDecoderCacheLength,
  isGpuBufferTensor,
  mapPresentKvToPastKv,
  readOrtTensorData,
  throwIfDecodeAborted,
} from './executor-kv-cache.js';
import type {
  DecoderSessionTiming,
  DecoderStepResult,
  LoadedExecutorState,
  SplitGraphDecodeCallbacks,
  SplitGraphDecodeResult,
} from './executor-types.js';
import type { WhisperModelConfig } from './generation-config.js';
import { selectWhisperLanguageFromLogits } from './language-detection.js';
import type { OrtTensorLike } from './ort.js';
import type { WhisperSeq2SeqTranscriptionOptions } from './types.js';

export async function splitGraphDecodeLoop(params: {
  promptTokens: readonly number[];
  encoderHiddenStates: Float32Array;
  eosTokenId: number;
  maxNewTokens: number;
  modelConfig: WhisperModelConfig;
  runInit: SplitGraphDecodeCallbacks['runInit'];
  runStep: SplitGraphDecodeCallbacks['runStep'];
  runStepBatch?: SplitGraphDecodeCallbacks['runStepBatch'];
  processLogits?: (logits: Float32Array, generatedTokens: readonly number[], beginIndex: number) => void;
  onTokenLogits?: (chosenTokenId: number, processedLogits: Float32Array, ctx: { readonly tokens: readonly number[]; readonly beginIndex: number }) => void;
  onDecoderInitLogits?: (
    rawLogits: Float32Array,
    ctx: {
      readonly tokens: readonly number[];
      readonly beginIndex: number;
      readonly vocabSize: number;
      readonly noSpeechTokenId?: number;
    },
  ) => void;
  noSpeechTokenId?: number;
  /** Beam search: number of beams (default: 1 = greedy) */
  numBeams?: number;
  /** Final ranking penalty. Undefined uses length normalization; 0 uses raw score. */
  lengthPenalty?: number;
  /** Beam search patience for early stopping. */
  patience?: number;
  /** Greedy decoding temperature. 0 = argmax. */
  temperature?: number;
  /** Number of independent decodings, pick best by score (WhisperX: best_of) */
  bestOf?: number;
  /** Experimental: batch active beam decoder steps into one ORT call when supported. */
  experimentalBatchedBeam?: boolean;
  /** Collect selected-sequence scalar quality traces. */
  trackQuality?: boolean;
  /** Abort between decoder-init/step calls. */
  signal?: AbortSignalLike | null;
}): Promise<SplitGraphDecodeResult> {
  const {
    promptTokens,
    encoderHiddenStates,
    eosTokenId,
    maxNewTokens,
    modelConfig,
    runInit,
    runStep,
    runStepBatch,
    processLogits,
    onTokenLogits,
    onDecoderInitLogits,
    noSpeechTokenId,
    numBeams,
    lengthPenalty,
    patience,
    temperature,
    bestOf,
    experimentalBatchedBeam,
    trackQuality,
    signal,
  } = params;

  const encoderDims: readonly number[] = [1, encoderHiddenStates.length / modelConfig.dModel, modelConfig.dModel];
  const session: WhisperCoreSession = {
    runInit: async (pt, enc, dims) => runInit(pt, enc, dims),
    runStep: async (tid, kv) => runStep(tid, kv),
    ...(runStepBatch ? { runStepBatch: async (tids, kvs) => runStepBatch(tids, kvs) } : {}),
  };
  const result = await whisperDecode(session, {
    promptTokens,
    encoderOutput: encoderHiddenStates,
    encoderDims,
    eosTokenId,
    maxNewTokens,
    processLogits,
    onTokenLogits,
    onDecoderInitLogits,
    noSpeechTokenId,
    strategy: (numBeams ?? 1) > 1 ? 'beam' : 'greedy',
    beamSize: numBeams ?? 1,
    lengthPenalty,
    patience: patience ?? 1,
    temperature: temperature ?? 0,
    bestOf: bestOf ?? 1,
    experimentalBatchedBeam: experimentalBatchedBeam ?? false,
    trackQuality: trackQuality === true,
    signal,
  });
  return {
    tokens: result.tokens,
    ...(result.tokenTraces ? { tokenTraces: result.tokenTraces } : {}),
  };
}

export function percentile(values: readonly number[], p: number): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

export async function runWhisperMergedDecoderStep(
  loaded: LoadedExecutorState,
  encoderHiddenStates: OrtTensorLike<Float32Array>,
  generatedTokens: readonly number[],
  pastKeyValues: Record<string, OrtTensorLike<Float32Array>>,
  isFirstStep: boolean,
): Promise<DecoderStepResult> {
  const decoderInputNames = loaded.decoderSession!.inputNames ?? [];
  const shouldFeed = (name: string): boolean =>
    decoderInputNames.length === 0 || decoderInputNames.includes(name);
  const hasDecoderPast = !isFirstStep && Object.entries(pastKeyValues).some(
    ([name, tensor]) =>
      shouldFeed(name) &&
      /^past_key_values\.\d+\.decoder\.(?:key|value)$/.test(name) &&
      (tensor.dims[2] ?? 0) > 0,
  );
  // The cached Whisper branch advances the decoder KV by one position. Feed
  // only the newest token when a decoder cache is present; cacheless graphs
  // still need the full prefix on every call.
  const decoderTokens = hasDecoderPast ? generatedTokens.slice(-1) : generatedTokens;
  const cachePosition = decoderInputNames.includes('cache_position')
    ? createWhisperCachePosition(
        getWhisperDecoderCacheLength(pastKeyValues, isFirstStep || !hasDecoderPast),
        decoderTokens.length,
      )
    : undefined;
  const inputIds = createWhisperInt64TokenIds(decoderTokens);
  const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, decoderTokens.length]);
  const feeds: Record<string, unknown> = {
    input_ids: inputIdsTensor,
    encoder_hidden_states: encoderHiddenStates,
  };
  if (cachePosition) {
    feeds.cache_position = new loaded.ort.Tensor(
      'int64',
      cachePosition,
      [cachePosition.length],
    );
  }

  if (decoderInputNames.includes('use_cache_branch')) {
    feeds.use_cache_branch = new loaded.ort.Tensor('bool', new Uint8Array([isFirstStep ? 0 : 1]), [1]);
  }

  if (!isFirstStep) {
    for (const [name, tensor] of Object.entries(pastKeyValues)) {
      if (shouldFeed(name)) feeds[name] = tensor;
    }
  } else {
    const numLayers = loaded.modelConfig.decoderLayers;
    const numHeads = loaded.modelConfig.decoderAttentionHeads;
    const headDim = loaded.modelConfig.headDim;
    const encoderSeqLen = encoderHiddenStates.dims[1] as number;
    for (let i = 0; i < numLayers; i++) {
      const decoderKey = `past_key_values.${i}.decoder.key`;
      const decoderValue = `past_key_values.${i}.decoder.value`;
      if (shouldFeed(decoderKey)) {
        feeds[decoderKey] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(0),
          [1, numHeads, 0, headDim],
        );
      }
      if (shouldFeed(decoderValue)) {
        feeds[decoderValue] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(0),
          [1, numHeads, 0, headDim],
        );
      }
      const encoderCacheSize = 1 * numHeads * encoderSeqLen * headDim;
      const encoderKey = `past_key_values.${i}.encoder.key`;
      const encoderValue = `past_key_values.${i}.encoder.value`;
      if (shouldFeed(encoderKey)) {
        feeds[encoderKey] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(encoderCacheSize),
          [1, numHeads, encoderSeqLen, headDim],
        );
      }
      if (shouldFeed(encoderValue)) {
        feeds[encoderValue] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(encoderCacheSize),
          [1, numHeads, encoderSeqLen, headDim],
        );
      }
    }
  }

  const retain = new Set<unknown>([encoderHiddenStates, ...Object.values(pastKeyValues)]);
  let outputs: Record<string, OrtTensorLike>;
  try {
    outputs = await loaded.decoderSession!.run(feeds);
  } finally {
    disposeOwnedDecoderFeeds(feeds, retain);
  }
  try {
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
    const logitsDims = logitsTensor.dims;
    const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;
    const lastLogits = copyLogitsTail(logitsData.data as ArrayBufferView, vocabSize);

    const nextPastKeyValues: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (key.startsWith('present')) {
        const pastName = key.replace(/^present/, 'past_key_values');
        const previousTensor = pastKeyValues[pastName];
        if (
          /^present\.\d+\.encoder\.(?:key|value)$/.test(key) &&
          value.dims.some((dimension) => dimension === 0) &&
          previousTensor &&
          previousTensor.dims.every((dimension) => dimension > 0)
        ) {
          // Some merged Whisper graphs return empty encoder-cache placeholders
          // on the cached branch; encoder KV is immutable and must be retained.
          nextPastKeyValues[pastName] = previousTensor;
          continue;
        }
        nextPastKeyValues[pastName] = value as OrtTensorLike<Float32Array>;
      }
    }
    disposeUnusedDecoderOutputs(outputs, new Set(Object.values(nextPastKeyValues)));

    return {
      lastLogits,
      vocabSize,
      pastKeyValues: nextPastKeyValues,
      crossAttentions: [],
    };
  } catch (error) {
    disposeUnusedDecoderOutputs(outputs, new Set());
    throw error;
  }
}

export async function runWhisperDecoderInit(
  loaded: Required<Pick<LoadedExecutorState, 'decoderInitSession' | 'ort'>> & LoadedExecutorState,
  encoderHiddenStates: OrtTensorLike<Float32Array>,
  promptTokens: readonly number[],
): Promise<{
  logits: Float32Array;
  vocabSize: number;
  presentKv: Record<string, OrtTensorLike<Float32Array>>;
  timings: DecoderSessionTiming;
}> {
  const inputStart = nowMs();
  const inputIds = createWhisperInt64TokenIds(promptTokens);
  const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, promptTokens.length]);
  const feeds: Record<string, unknown> = {
    input_ids: inputIdsTensor,
    encoder_hidden_states: encoderHiddenStates,
  };
  const inputLocations = countTensorLocations(Object.values(feeds));

  const runStart = nowMs();
  const retain = new Set<unknown>([encoderHiddenStates]);
  let outputs: Record<string, OrtTensorLike>;
  try {
    outputs = await loaded.decoderInitSession.run(feeds);
  } finally {
    disposeOwnedDecoderFeeds(feeds, retain);
  }
  const outputStart = nowMs();
  const outputLocations = countTensorLocations(Object.values(outputs));
  const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
  const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
  const logitsReadStart = nowMs();
  const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
  const logitReadMs = nowMs() - logitsReadStart;
  const logitsDims = logitsTensor.dims;
  const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;
  const logitsCopy = copyTensorData(logitsData.data as ArrayBufferView);

  const kvStart = nowMs();
  const presentKv: Record<string, OrtTensorLike<Float32Array>> = {};
  for (const [key, value] of Object.entries(outputs)) {
    if (key.startsWith('present')) {
      presentKv[key] = value as OrtTensorLike<Float32Array>;
    }
  }
  disposeUnusedDecoderOutputs(outputs, new Set(Object.values(presentKv)));
  const kvExtractMs = nowMs() - kvStart;

  const outputEnd = nowMs();
  return {
    logits: logitsCopy,
    vocabSize,
    presentKv,
    timings: {
      inputMs: runStart - inputStart,
      runMs: outputStart - runStart,
      outputMs: outputEnd - outputStart,
      tensorCreateMs: runStart - inputStart,
      logitReadMs,
      kvExtractMs,
      gpuInputCount: inputLocations.gpu,
      cpuInputCount: inputLocations.cpu,
      gpuOutputCount: outputLocations.gpu,
      cpuOutputCount: outputLocations.cpu,
      gpuDownloadCount: logitsData.downloaded ? 1 : 0,
    },
  };
}

/** Multi-token decoder_step: feeds K tokens at once, returns K logits vectors. */
export async function runWhisperDecoderStepMultiToken(
  loaded: Required<Pick<LoadedExecutorState, 'decoderStepSession' | 'ort'>> & LoadedExecutorState,
  tokenIds: readonly number[],
  pastKv: Record<string, OrtTensorLike<Float32Array>>,
  options: { readonly preparedPastKv?: boolean } = {},
): Promise<{
  logits: Float32Array;
  vocabSize: number;
  nextTokenId?: number;
  presentKv: Record<string, OrtTensorLike<Float32Array>>;
  timings: DecoderSessionTiming;
}> {
  const K = tokenIds.length;
  const inputStart = nowMs();
  const cachePosition = loaded.decoderStepSession.inputNames?.includes('cache_position')
    ? createWhisperCachePosition(getWhisperDecoderCacheLength(pastKv), K)
    : undefined;
  const inputIdsTensor = new loaded.ort.Tensor('int64', createWhisperInt64TokenIds(tokenIds), [1, K]);
  const feeds: Record<string, unknown> = { input_ids: inputIdsTensor };
  if (cachePosition) {
    feeds.cache_position = new loaded.ort.Tensor(
      'int64',
      cachePosition,
      [cachePosition.length],
    );
  }

  for (const [name, tensor] of Object.entries(pastKv)) {
    if (options.preparedPastKv) {
      feeds[name] = tensor;
      continue;
    }
    if (isGpuBufferTensor(tensor)) {
      feeds[name] = tensor;
    } else {
      const cloned = cloneDecoderKvTensorDataForInput(tensor);
      feeds[name] = new loaded.ort.Tensor(cloned.type, cloned.data, tensor.dims);
    }
  }
  const inputLocations = countTensorLocations(Object.values(feeds));
  const retain = new Set<unknown>(Object.values(pastKv));

  const runStart = nowMs();
  let outputs: Record<string, OrtTensorLike> | undefined;
  try {
    outputs = await loaded.decoderStepSession.run(feeds);
    const outputStart = nowMs();
    const outputLocations = countTensorLocations(Object.values(outputs));
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const logitReadStart = nowMs();
    const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
    const logitReadMs = nowMs() - logitReadStart;
    const logitsDims = logitsTensor.dims;
    const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;
    const logitsCopy = copyTensorData(logitsData.data as ArrayBufferView);

    const kvStart = nowMs();
    const presentKv: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (key.startsWith('present')) {
        const pastName = key.replace(/^present/, 'past_key_values');
        presentKv[pastName] = value as OrtTensorLike<Float32Array>;
      }
    }
    for (const [key, value] of Object.entries(pastKv)) {
      if (key.includes('encoder') && !presentKv[key]) {
        presentKv[key] = value;
      }
    }

    let nextTokenId: number | undefined;
    const nextTokenTensor = outputs['next_token_id'] as OrtTensorLike<Int32Array> | undefined;
    if (nextTokenTensor && nextTokenTensor.data) {
      nextTokenId = nextTokenTensor.data[0];
    }

    disposeUnusedDecoderOutputs(outputs, new Set(Object.values(presentKv)));
    const kvEnd = nowMs();
    const outputEnd = nowMs();
    return {
      logits: logitsCopy,
      vocabSize,
      nextTokenId,
      presentKv,
      timings: {
        inputMs: runStart - inputStart,
        runMs: outputStart - runStart,
        outputMs: outputEnd - outputStart,
        tensorCreateMs: runStart - inputStart,
        logitReadMs,
        kvExtractMs: kvEnd - kvStart,
        gpuInputCount: inputLocations.gpu,
        cpuInputCount: inputLocations.cpu,
        gpuOutputCount: outputLocations.gpu,
        cpuOutputCount: outputLocations.cpu,
        gpuDownloadCount: logitsData.downloaded ? 1 : 0,
      },
    };
  } catch (error) {
    if (outputs) disposeUnusedDecoderOutputs(outputs, new Set());
    throw error;
  } finally {
    disposeOwnedDecoderFeeds(feeds, retain);
  }
}

export async function runWhisperDecoderStepSplit(
  loaded: Required<Pick<LoadedExecutorState, 'decoderStepSession' | 'ort'>> & LoadedExecutorState,
  tokenId: number,
  pastKv: Record<string, OrtTensorLike<Float32Array>>,
  options: { readonly preparedPastKv?: boolean } = {},
): Promise<{
  logits: Float32Array;
  vocabSize: number;
  nextTokenId?: number;
  presentKv: Record<string, OrtTensorLike<Float32Array>>;
  timings: DecoderSessionTiming;
}> {
  return runWhisperDecoderStepMultiToken(loaded, [tokenId], pastKv, options);
}

export async function detectLanguageFromEncoder(
  loaded: Required<LoadedExecutorState>,
  encoderHiddenStates: OrtTensorLike<Float32Array>,
): Promise<string> {
  try {
    const sotId = loaded.tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
    const inputIds = new BigInt64Array([BigInt(sotId)]);
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, 1]);
    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };
    const outputs = await loaded.decoderInitSession.run(feeds);
    try {
      const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
      const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
      const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
      const vocabSize = logitsTensor.dims[logitsTensor.dims.length - 1] ?? 0;

      return selectWhisperLanguageFromLogits(loaded.tokenizer, logitsData.data, vocabSize) ?? 'auto';
    } finally {
      disposeOwnedDecoderFeeds(feeds, new Set([encoderHiddenStates]));
      for (const output of Object.values(outputs)) {
        disposeOrtTensor(output);
      }
    }
  } catch {
    return 'auto';
  }
}

export async function detectLanguageFromMergedDecoder(
  loaded: LoadedExecutorState,
  encoderHiddenStates: OrtTensorLike<Float32Array>,
): Promise<string> {
  try {
    const sotId = loaded.tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
    const result = await runWhisperMergedDecoderStep(loaded, encoderHiddenStates, [sotId], {}, true);
    try {
      return selectWhisperLanguageFromLogits(loaded.tokenizer, result.lastLogits, result.vocabSize) ?? 'auto';
    } finally {
      for (const tensor of Object.values(result.pastKeyValues)) {
        disposeGpuTensor(tensor);
      }
    }
  } catch {
    return 'auto';
  }
}

export async function runGreedyGpuKvDecode(params: {
  readonly loaded: Required<Pick<LoadedExecutorState, 'decoderInitSession' | 'decoderStepSession' | 'ort'>> & LoadedExecutorState;
  readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
  readonly promptTokens: readonly number[];
  readonly eosTokenId: number;
  readonly maxNewTokens: number;
  readonly processLogits?: (logits: Float32Array, generatedTokens: readonly number[], beginIndex: number) => void;
  readonly onTokenLogits?: WhisperSeq2SeqTranscriptionOptions['onTokenLogits'];
  readonly onDecoderInitLogits?: WhisperSeq2SeqTranscriptionOptions['onDecoderInitLogits'];
  readonly noSpeechTokenId?: number;
  readonly trackQuality?: boolean;
  readonly onInitTiming?: (timings: DecoderSessionTiming, elapsedMs: number) => void;
  readonly onStepTiming?: (timings: DecoderSessionTiming, elapsedMs: number) => void;
  readonly signal?: AbortSignalLike | null;
}): Promise<SplitGraphDecodeResult> {
  const {
    loaded,
    encoderHiddenStates,
    promptTokens,
    eosTokenId,
    maxNewTokens,
    processLogits,
    onTokenLogits,
    onDecoderInitLogits,
    noSpeechTokenId,
    trackQuality = false,
    onInitTiming,
    onStepTiming,
    signal,
  } = params;

  throwIfDecodeAborted(signal);
  const initStart = nowMs();
  const init = await runWhisperDecoderInit(loaded, encoderHiddenStates, promptTokens);
  onInitTiming?.(init.timings, nowMs() - initStart);

  const vocabSize = init.vocabSize;
  const firstLogits = init.logits.subarray(init.logits.length - vocabSize);
  onDecoderInitLogits?.(new Float32Array(firstLogits), {
    tokens: promptTokens,
    beginIndex: promptTokens.length,
    vocabSize,
    noSpeechTokenId,
  });
  processLogits?.(firstLogits, promptTokens, promptTokens.length);

  const firstTokenId = argmax(firstLogits);
  const tokens: number[] = [firstTokenId];
  const tokenTraces: Array<{ tokenId: number; logProb: number; entropy: number }> = [];
  if (trackQuality) {
    const firstQuality = tokenQualityFromLogits(firstLogits, firstTokenId);
    tokenTraces.push({
      tokenId: firstTokenId,
      logProb: firstQuality.logProb,
      entropy: firstQuality.entropy,
    });
  }
  onTokenLogits?.(firstTokenId, firstLogits, { tokens, beginIndex: promptTokens.length });

  const finish = (): SplitGraphDecodeResult => (
    trackQuality ? { tokens, tokenTraces } : { tokens }
  );

  let pastKv = mapPresentKvToPastKv(init.presentKv);
  try {
    throwIfDecodeAborted(signal);
    if (firstTokenId === eosTokenId) {
      return finish();
    }

    for (let stepIndex = 1; stepIndex < maxNewTokens; stepIndex++) {
      throwIfDecodeAborted(signal);
      const stepStart = nowMs();
      const step = await runWhisperDecoderStepSplit(loaded, tokens[tokens.length - 1]!, pastKv);
      onStepTiming?.(step.timings, nowMs() - stepStart);

      const previousKv = pastKv;
      pastKv = step.presentKv;
      disposeReplacedGpuKv(previousKv, pastKv);

      processLogits?.(step.logits, [...promptTokens, ...tokens], promptTokens.length);
      const nextTokenId = step.nextTokenId ?? argmax(step.logits);
      tokens.push(nextTokenId);
      if (trackQuality) {
        const stepQuality = tokenQualityFromLogits(step.logits, nextTokenId);
        tokenTraces.push({
          tokenId: nextTokenId,
          logProb: stepQuality.logProb,
          entropy: stepQuality.entropy,
        });
      }
      onTokenLogits?.(nextTokenId, step.logits, { tokens, beginIndex: promptTokens.length });

      if (nextTokenId === eosTokenId) {
        break;
      }
    }

    return finish();
  } finally {
    disposeGpuKv(pastKv);
  }
}
