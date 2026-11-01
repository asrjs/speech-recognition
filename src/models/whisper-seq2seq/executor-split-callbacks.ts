import { PipelineAbortedError } from '../../pipeline/composition.js';
import { nowMs } from '../../runtime/timing.js';
import type { WhisperKvCache } from './core.js';
import {
  percentile,
  runGreedyGpuKvDecode,
  runWhisperDecoderInit,
  runWhisperDecoderStepSplit,
  splitGraphDecodeLoop,
} from './executor-decoder-ops.js';
import {
  canReuseWhisperEncoderKvTensor,
  canShareWhisperEncoderKvAcrossBatch,
  cloneDecoderKvDataForInput,
  concatDecoderKvDataForBatch,
  copyAndReleaseWhisperPresentKv,
  countTensorLocations,
  createWhisperBatchedStepFeedTensors,
  decoderKvDataLayout,
  disposeOrtTensor,
  disposeOwnedDecoderFeeds,
  disposeWhisperEncoderKvCaches,
  mapPresentKvToPastKv,
  normalizeWhisperKvCacheValue,
  readOrtTensorData,
  sliceDecoderKvDataForBatch,
  supportsWhisperEncoderKvBroadcast,
  withOwnedDecoderFeeds,
  type TensorDataView,
} from './executor-kv-cache.js';
import type {
  DecoderSessionTiming,
  LoadedExecutorState,
  SplitGraphDecodeResult,
  WhisperEncoderKvTensorCacheEntry,
  WhisperPresentKvSnapshot,
} from './executor-types.js';
import type { OrtTensorLike } from './ort.js';
import type { WhisperSeq2SeqTranscriptionOptions } from './types.js';

export interface SplitGraphDecodeTimingMetrics {
  readonly decodeMs: number;
  readonly decoderInitMs: number;
  readonly decoderInitInputMs: number;
  readonly decoderInitRunMs: number;
  readonly decoderInitOutputMs: number;
  readonly decoderStepMs: number;
  readonly decoderStepFeedBuildMs: number;
  readonly decoderStepTensorCloneMs: number;
  readonly decoderStepRunMs: number;
  readonly decoderStepOutputMs: number;
  readonly decoderStepAvgMs: number | undefined;
  readonly decoderStepP50Ms: number | undefined;
  readonly decoderStepP95Ms: number | undefined;
  readonly decoderStepMaxMs: number | undefined;
  readonly decoderLogitProcessMs: number;
  readonly decoderStepCount: number;
  readonly decoderGpuTensorInputs: number;
  readonly decoderCpuTensorInputs: number;
  readonly decoderGpuTensorOutputs: number;
  readonly decoderCpuTensorOutputs: number;
  readonly decoderGpuTensorDownloads: number;
  readonly decoderKvCacheLocation: 'gpu-buffer' | 'cpu';
  readonly decoderInitTensorCreateMs: number;
  readonly decoderInitLogitReadMs: number;
  readonly decoderInitKvExtractMs: number;
  readonly decoderStepTensorCreateMs: number;
  readonly decoderStepLogitReadMs: number;
  readonly decoderStepKvMergeMs: number;
  readonly decoderEncoderKvTensorReuses: number;
  readonly decoderEncoderKvTensorCreates: number;
}

export async function runSplitGraphDecodeWithTiming(params: {
  readonly splitLoaded: Required<LoadedExecutorState>;
  readonly loaded: LoadedExecutorState;
  readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
  readonly promptTokens: readonly number[];
  readonly eosId: number;
  readonly noSpeechTokenId: number;
  readonly maxNewTokens: number;
  readonly requestedNumBeams: number;
  readonly requestedBestOf: number;
  readonly requestedTemperature: number;
  readonly useExperimentalGpuKvCache: boolean;
  readonly gpuKvBeamActive: boolean;
  readonly options: WhisperSeq2SeqTranscriptionOptions;
  readonly applyTimestampProcessor: (
    logits: Float32Array,
    genTokens: readonly number[],
    beginIdx: number,
  ) => void;
}): Promise<{
  readonly result: SplitGraphDecodeResult;
  readonly timing: SplitGraphDecodeTimingMetrics;
}> {
  const {
    splitLoaded,
    loaded,
    encoderHiddenStates,
    promptTokens,
    eosId,
    noSpeechTokenId,
    maxNewTokens,
    requestedNumBeams,
    requestedBestOf,
    requestedTemperature,
    useExperimentalGpuKvCache,
    gpuKvBeamActive,
    options,
    applyTimestampProcessor,
  } = params;

  let kvDims: Record<string, readonly number[]> = {};
  let kvDtype: 'float32' | 'float16' = 'float32';
  let decoderInitMs = 0;
  let decoderInitInputMs = 0;
  let decoderInitRunMs = 0;
  let decoderInitOutputMs = 0;
  let decoderStepMs = 0;
  let decoderStepFeedBuildMs = 0;
  let decoderStepTensorCloneMs = 0;
  let decoderStepRunMs = 0;
  let decoderStepOutputMs = 0;
  let decoderLogitProcessMs = 0;
  let decoderStepCount = 0;
  let decoderGpuTensorInputs = 0;
  let decoderCpuTensorInputs = 0;
  let decoderGpuTensorOutputs = 0;
  let decoderCpuTensorOutputs = 0;
  let decoderGpuTensorDownloads = 0;
  let decoderInitTensorCreateMs = 0;
  let decoderInitLogitReadMs = 0;
  let decoderInitKvExtractMs = 0;
  let decoderStepTensorCreateMs = 0;
  let decoderStepLogitReadMs = 0;
  let decoderStepKvMergeMs = 0;
  let decoderEncoderKvTensorReuses = 0;
  let decoderEncoderKvTensorCreates = 0;
  const decoderStepTimings: number[] = [];
  const requestedDecoderKvCacheLocation = useExperimentalGpuKvCache || gpuKvBeamActive ? 'gpu-buffer' : 'cpu';

  const recordDecoderTiming = (timings: DecoderSessionTiming): void => {
    decoderGpuTensorInputs += timings.gpuInputCount;
    decoderCpuTensorInputs += timings.cpuInputCount;
    decoderGpuTensorOutputs += timings.gpuOutputCount;
    decoderCpuTensorOutputs += timings.cpuOutputCount;
    decoderGpuTensorDownloads += timings.gpuDownloadCount;
  };

  const batchedEncoderKvTensors = new Map<string, {
    readonly sourceData: ArrayBufferView;
    readonly sourceDims?: readonly number[];
    readonly tensor: OrtTensorLike<Float32Array>;
  }>();
  const scalarEncoderKvTensors = new Map<string, WhisperEncoderKvTensorCacheEntry>();
  const gpuKvBeamTracker = {
    touched: new Map<OrtTensorLike, number>(),
    generation: 0,
    lastPruned: [] as string[],
    register(tensors: readonly OrtTensorLike[], generation: number): void {
      for (const tensor of tensors) this.touched.set(tensor, generation);
    },
    pruneIdle(minGeneration: number): void {
      for (const [tensor, generation] of [...this.touched]) {
        if (generation <= minGeneration) {
          this.touched.delete(tensor);
          tensor.dispose?.();
          if (this.lastPruned.length < 40) {
            this.lastPruned.push(`gen${generation}`);
          }
        }
      }
    },
    disposeAll(): void {
      for (const [tensor] of this.touched) tensor.dispose?.();
      this.touched.clear();
    },
  };

  const decoderStart = nowMs();
  const processSplitGraphLogits = (logits: Float32Array, genTokens: readonly number[], beginIdx: number): void => {
    const logitsStart = nowMs();
    applyTimestampProcessor(logits, genTokens, beginIdx);
    decoderLogitProcessMs += nowMs() - logitsStart;
  };

  let result: SplitGraphDecodeResult;
  if (useExperimentalGpuKvCache && requestedNumBeams === 1) {
    result = await runGreedyGpuKvDecode({
      loaded: splitLoaded,
      encoderHiddenStates,
      promptTokens: [...promptTokens],
      eosTokenId: eosId,
      maxNewTokens,
      processLogits: processSplitGraphLogits,
      onTokenLogits: options.onTokenLogits,
      onDecoderInitLogits: options.onDecoderInitLogits,
      noSpeechTokenId,
      trackQuality: options.trackQuality === true,
      onInitTiming: (timings, elapsedMs) => {
        decoderInitMs += elapsedMs;
        decoderInitInputMs += timings.inputMs;
        decoderInitRunMs += timings.runMs;
        decoderInitOutputMs += timings.outputMs;
        decoderInitTensorCreateMs += timings.tensorCreateMs ?? 0;
        decoderInitLogitReadMs += timings.logitReadMs ?? 0;
        decoderInitKvExtractMs += timings.kvExtractMs ?? 0;
        recordDecoderTiming(timings);
      },
      onStepTiming: (timings, elapsedMs) => {
        decoderStepMs += elapsedMs;
        decoderStepTensorCloneMs += timings.inputMs;
        decoderStepRunMs += timings.runMs;
        decoderStepOutputMs += timings.outputMs;
        decoderStepTensorCreateMs += timings.tensorCreateMs ?? 0;
        decoderStepLogitReadMs += timings.logitReadMs ?? 0;
        decoderStepKvMergeMs += timings.kvExtractMs ?? 0;
        decoderStepCount += 1;
        decoderStepTimings.push(elapsedMs);
        recordDecoderTiming(timings);
      },
      signal: options.signal,
    });
  } else {
    try {
      const encDims = encoderHiddenStates.dims;
      const encoderHiddenStatesData = new Float32Array(
        encDims.slice(1).reduce((product, dim) => product * (dim || 1), 1),
      );
      result = await splitGraphDecodeLoop({
        promptTokens: [...promptTokens],
        encoderHiddenStates: encoderHiddenStatesData,
        eosTokenId: eosId,
        maxNewTokens,
        modelConfig: loaded.modelConfig,
        processLogits: processSplitGraphLogits,
        onTokenLogits: options.onTokenLogits,
        onDecoderInitLogits: options.onDecoderInitLogits,
        noSpeechTokenId,
        numBeams: requestedNumBeams,
        lengthPenalty: options.lengthPenalty,
        patience: options.patience ?? 1,
        temperature: requestedTemperature,
        bestOf: requestedBestOf,
        experimentalBatchedBeam: options.experimentalBatchedBeam === true,
        trackQuality: options.trackQuality === true,
        signal: options.signal,
        runInit: async (prompt, _encHs, _dims) => {
          const decoderInitStart = nowMs();
          const init = await runWhisperDecoderInit(splitLoaded, encoderHiddenStates, prompt);
          const decoderInitTotal = nowMs() - decoderInitStart;
          decoderInitMs += decoderInitTotal;
          decoderInitInputMs += init.timings.inputMs;
          decoderInitRunMs += init.timings.runMs;
          decoderInitOutputMs += init.timings.outputMs;
          recordDecoderTiming(init.timings);
          kvDims = {};
          for (const [k, v] of Object.entries(init.presentKv)) {
            kvDims[k] = v.dims;
            kvDims[k.replace(/^present\./, 'past_key_values.')] = v.dims;
          }
          const firstKv = Object.values(init.presentKv)[0];
          if (firstKv) {
            kvDtype = firstKv.type === 'float16' || firstKv.data.constructor.name === 'Float16Array' ? 'float16' : 'float32';
          }
          if (gpuKvBeamActive) {
            const pastKv = mapPresentKvToPastKv(init.presentKv);
            gpuKvBeamTracker.register(Object.values(pastKv), 0);
            return {
              logits: init.logits,
              vocabSize: init.vocabSize,
              presentKv: pastKv,
            };
          }
          return {
            logits: init.logits,
            vocabSize: init.vocabSize,
            presentKv: copyAndReleaseWhisperPresentKv(init.presentKv),
          };
        },
        runStep: async (tokenId, pastKv) => {
          const decoderStepStart = nowMs();
          if (gpuKvBeamActive) {
            gpuKvBeamTracker.generation += 1;
            const generation = gpuKvBeamTracker.generation;
            const feeds: Record<string, OrtTensorLike<Float32Array>> = {};
            for (const [name, tensor] of Object.entries(pastKv)) {
              const gpuTensor = tensor as OrtTensorLike<Float32Array>;
              feeds[name] = gpuTensor;
              gpuKvBeamTracker.register([gpuTensor], generation);
            }
            let step;
            try {
              step = await runWhisperDecoderStepSplit(splitLoaded, tokenId, feeds, { preparedPastKv: true });
            } catch (error) {
              if (error instanceof PipelineAbortedError) throw error;
              const locations = Object.entries(feeds)
                .map(([name, tensor]) => `${name}=${(tensor as OrtTensorLike).location ?? 'unknown'}`)
                .join(' ');
              const message = error instanceof Error ? error.message : String(error);
              const pruned = gpuKvBeamTracker.lastPruned.join(',');
              throw new Error(`gpu-kv beam step failed (gen ${generation}): ${message}; feeds: ${locations}; pruned-so-far: ${pruned}`);
            }
            gpuKvBeamTracker.register(Object.values(step.presentKv), generation);
            gpuKvBeamTracker.pruneIdle(generation - requestedNumBeams * 3);
            return {
              logits: step.logits,
              vocabSize: step.vocabSize,
              presentKv: step.presentKv,
            };
          }
          const feedBuildStart = nowMs();
          const feeds: Record<string, OrtTensorLike<Float32Array>> = {};
          for (const [name, value] of Object.entries(pastKv)) {
            const kvValue = normalizeWhisperKvCacheValue(value);
            const stepName = name.replace(/^present\./, 'past_key_values.');
            const dims = kvValue.dims
              ?? kvDims[name]
              ?? kvDims[stepName]
              ?? kvDims[name.replace(/^past_key_values\./, 'present.')];
            const isEncoderKv = stepName.includes('.encoder.');
            const requestedType = decoderKvDataLayout(
              kvValue.data,
              kvValue.type ?? kvDtype,
            ).type;
            const cachedEncoderKv = isEncoderKv ? scalarEncoderKvTensors.get(stepName) : undefined;
            if (canReuseWhisperEncoderKvTensor(cachedEncoderKv, kvValue.data, dims, requestedType)) {
              feeds[stepName] = cachedEncoderKv!.tensor;
              decoderEncoderKvTensorReuses += 1;
              continue;
            }

            const cloned = cloneDecoderKvDataForInput(kvValue.data, kvValue.type ?? kvDtype);
            let resolvedDims = dims;
            if (!resolvedDims) {
              const numHeads = splitLoaded.modelConfig.decoderAttentionHeads;
              const headDim = splitLoaded.modelConfig.headDim;
              const seqLen = Math.round(cloned.data.length / (numHeads * headDim));
              resolvedDims = [1, numHeads, seqLen, headDim];
            }
            const tensor = new splitLoaded.ort.Tensor(
              cloned.type,
              cloned.data,
              resolvedDims,
            ) as unknown as OrtTensorLike<Float32Array>;
            feeds[stepName] = tensor;
            if (isEncoderKv) {
              scalarEncoderKvTensors.set(stepName, {
                tensor,
                sourceData: kvValue.data,
                stableData: cloned.data,
                dims: resolvedDims,
                type: cloned.type,
              });
              decoderEncoderKvTensorCreates += 1;
            }
          }
          decoderStepFeedBuildMs += nowMs() - feedBuildStart;
          const retainedFeeds = new Set(
            [...scalarEncoderKvTensors.values()].map((entry) => entry.tensor),
          );
          const step = await withOwnedDecoderFeeds(
            feeds,
            retainedFeeds,
            () => runWhisperDecoderStepSplit(splitLoaded, tokenId, feeds, { preparedPastKv: true }),
          );
          const decoderStepTotal = nowMs() - decoderStepStart;
          decoderStepMs += decoderStepTotal;
          decoderStepTensorCloneMs += step.timings.inputMs;
          decoderStepRunMs += step.timings.runMs;
          decoderStepOutputMs += step.timings.outputMs;
          decoderStepCount += 1;
          decoderStepTimings.push(decoderStepTotal);
          recordDecoderTiming(step.timings);
          for (const [k, v] of Object.entries(step.presentKv)) {
            kvDims[k] = v.dims;
          }
          const retained = new Map<OrtTensorLike, WhisperPresentKvSnapshot>();
          for (const [k, v] of Object.entries(step.presentKv)) {
            const cachedEncoderKv = k.includes('.encoder.')
              ? scalarEncoderKvTensors.get(k)
              : undefined;
            if (
              cachedEncoderKv?.tensor === v
              && canReuseWhisperEncoderKvTensor(
                cachedEncoderKv,
                v.data,
                v.dims,
                cachedEncoderKv.type,
              )
            ) {
              retained.set(v, {
                data: cachedEncoderKv.stableData,
                dims: cachedEncoderKv.dims,
                type: cachedEncoderKv.type,
              });
            }
          }
          return {
            logits: step.logits,
            vocabSize: step.vocabSize,
            presentKv: copyAndReleaseWhisperPresentKv(step.presentKv, retained),
          };
        },
        runStepBatch: options.experimentalBatchedBeam && !gpuKvBeamActive
          ? async (tokenIds, pastKvs) => {
              const batchSize = tokenIds.length;
              if (batchSize === 0) return [];
              const decoderStepStart = nowMs();
              const feedBuildStart = nowMs();
              const feeds = createWhisperBatchedStepFeedTensors(
                splitLoaded.ort,
                tokenIds,
                pastKvs,
                splitLoaded.decoderStepSession.inputNames?.includes('cache_position') === true,
              );

              const firstKv = pastKvs[0] ?? {};
              for (const name of Object.keys(firstKv)) {
                const stepName = name.replace(/^present\./, 'past_key_values.');
                const values = pastKvs.map((kv) => {
                  const value = kv[name]
                    ?? kv[stepName]
                    ?? kv[name.replace(/^past_key_values\./, 'present.')];
                  if (!value) {
                    throw new Error(`Missing Whisper batched beam KV "${name}" for one active beam.`);
                  }
                  return normalizeWhisperKvCacheValue(value);
                });
                const firstValue = values[0]!;
                const isEncoderKv = stepName.includes('.encoder.');
                const shareEncoderKv = isEncoderKv
                  && supportsWhisperEncoderKvBroadcast(splitLoaded.decoderStepSession)
                  && canShareWhisperEncoderKvAcrossBatch(values);
                const valuesToBatch = shareEncoderKv ? [firstValue] : values;
                const dims = firstValue.dims
                  ?? kvDims[name]
                  ?? kvDims[stepName]
                  ?? kvDims[name.replace(/^past_key_values\./, 'present.')];
                let perBeamDims = dims;
                if (!perBeamDims) {
                  const numHeads = splitLoaded.modelConfig.decoderAttentionHeads;
                  const headDim = splitLoaded.modelConfig.headDim;
                  const clonedFirst = cloneDecoderKvDataForInput(firstValue.data, firstValue.type ?? kvDtype);
                  const seqLen = Math.round(clonedFirst.data.length / (numHeads * headDim));
                  perBeamDims = [1, numHeads, seqLen, headDim];
                }
                for (const value of values) {
                  if (!value.dims) continue;
                  const sameRank = value.dims.length === perBeamDims.length;
                  const sameNonBatchDims = value.dims.slice(1).every((dim, i) => dim === perBeamDims![i + 1]);
                  if (!sameRank || !sameNonBatchDims) {
                    throw new Error(`Cannot batch Whisper KV "${name}" with mismatched per-beam dims.`);
                  }
                }
                const sourceDataIsShared = values.every((value) => value.data === firstValue.data);
                const cacheKey = `${name}:${shareEncoderKv ? 'broadcast' : batchSize}`;
                const cachedEncoderKv = isEncoderKv && sourceDataIsShared
                  ? batchedEncoderKvTensors.get(cacheKey)
                  : undefined;
                if (
                  cachedEncoderKv
                  && cachedEncoderKv.sourceData === firstValue.data
                  && cachedEncoderKv.sourceDims?.length === perBeamDims.length
                  && cachedEncoderKv.sourceDims.every((dim, index) => dim === perBeamDims![index])
                ) {
                  feeds[stepName] = cachedEncoderKv.tensor;
                  continue;
                }
                const batched = concatDecoderKvDataForBatch(valuesToBatch, kvDtype);
                const batchedTensor = new splitLoaded.ort.Tensor(
                  batched.type,
                  batched.data,
                  [shareEncoderKv ? 1 : batchSize, ...perBeamDims.slice(1)],
                ) as unknown as OrtTensorLike<Float32Array>;
                feeds[stepName] = batchedTensor;
                if (isEncoderKv && sourceDataIsShared) {
                  batchedEncoderKvTensors.set(cacheKey, {
                    sourceData: firstValue.data,
                    sourceDims: perBeamDims,
                    tensor: batchedTensor,
                  });
                }
              }

              decoderStepFeedBuildMs += nowMs() - feedBuildStart;
              const inputLocations = countTensorLocations(Object.values(feeds));
              const runStart = nowMs();
              let outputs: Record<string, OrtTensorLike> | undefined;
              try {
                outputs = await splitLoaded.decoderStepSession.run(feeds);
                const outputStart = nowMs();
                const outputLocations = countTensorLocations(Object.values(outputs));
                const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
                const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
                const logitReadStart = nowMs();
                const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
                const logitReadMs = nowMs() - logitReadStart;
                const logitsDims = logitsTensor.dims;
                const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;
                const logitsBatch = logitsDims[0] ?? batchSize;
                if (logitsBatch !== batchSize) {
                  throw new Error(`Batched Whisper decoder logits batch ${logitsBatch} does not match active beams ${batchSize}.`);
                }
                const perBeamLogitSpan = logitsData.data.length / batchSize;
                if (!Number.isInteger(perBeamLogitSpan) || perBeamLogitSpan < vocabSize) {
                  throw new Error('Batched Whisper decoder logits shape is not divisible by active beam count.');
                }
                const logitTimeSteps = Math.max(1, Math.floor(perBeamLogitSpan / vocabSize));
                const lastLogitOffset = (logitTimeSteps - 1) * vocabSize;
                const results = Array.from({ length: batchSize }, (_, beamIndex) => ({
                  logits: new Float32Array(
                    logitsData.data.subarray(
                      beamIndex * perBeamLogitSpan + lastLogitOffset,
                      beamIndex * perBeamLogitSpan + lastLogitOffset + vocabSize,
                    ),
                  ),
                  vocabSize,
                  presentKv: {} as WhisperKvCache,
                }));

                const kvStart = nowMs();
                let outputDownloadCount = logitsData.downloaded ? 1 : 0;
                for (const [key, value] of Object.entries(outputs)) {
                  if (!key.startsWith('present')) continue;
                  const pastName = key.replace(/^present/, 'past_key_values');
                  const tensor = value as OrtTensorLike<Float32Array>;
                  const tensorData = await readOrtTensorData(tensor, { releaseGpu: true });
                  if (tensorData.downloaded) outputDownloadCount += 1;
                  const data = tensorData.data as unknown as TensorDataView;
                  const tensorBatch = tensor.dims[0] ?? batchSize;
                  if (tensorBatch !== batchSize) {
                    throw new Error(`Batched Whisper KV "${key}" batch ${tensorBatch} does not match active beams ${batchSize}.`);
                  }
                  const perBeamLength = data.length / batchSize;
                  if (!Number.isInteger(perBeamLength)) {
                    throw new Error(`Batched Whisper KV "${key}" data length is not divisible by active beam count.`);
                  }
                  const perBeamDims = [1, ...tensor.dims.slice(1)];
                  for (let beamIndex = 0; beamIndex < batchSize; beamIndex++) {
                    results[beamIndex]!.presentKv[pastName] = {
                      data: cloneDecoderKvDataForInput(
                        sliceDecoderKvDataForBatch(data, beamIndex * perBeamLength, perBeamLength),
                        tensor.type,
                      ).data,
                      dims: perBeamDims,
                      type: tensor.type,
                    };
                  }
                }

                for (let beamIndex = 0; beamIndex < batchSize; beamIndex++) {
                  for (const [name, value] of Object.entries(pastKvs[beamIndex] ?? {})) {
                    const stepName = name.replace(/^present\./, 'past_key_values.');
                    if (!stepName.includes('encoder') || results[beamIndex]!.presentKv[stepName]) continue;
                    const kvValue = normalizeWhisperKvCacheValue(value);
                    results[beamIndex]!.presentKv[stepName] = {
                      data: kvValue.data,
                      ...(kvValue.dims ? { dims: kvValue.dims } : {}),
                      type: kvValue.type ?? kvDtype,
                    };
                  }
                }

                for (const res of results) {
                  for (const [k, v] of Object.entries(res.presentKv)) {
                    const kvValue = normalizeWhisperKvCacheValue(v);
                    if (kvValue.dims) kvDims[k] = kvValue.dims;
                  }
                }

                const kvEnd = nowMs();
                const outputEnd = nowMs();
                const decoderStepTotal = nowMs() - decoderStepStart;
                decoderStepMs += decoderStepTotal;
                decoderStepTensorCloneMs += runStart - feedBuildStart;
                decoderStepRunMs += outputStart - runStart;
                decoderStepOutputMs += outputEnd - outputStart;
                decoderStepTensorCreateMs += runStart - feedBuildStart;
                decoderStepLogitReadMs += logitReadMs;
                decoderStepKvMergeMs += kvEnd - kvStart;
                decoderStepCount += 1;
                decoderStepTimings.push(decoderStepTotal);
                recordDecoderTiming({
                  inputMs: runStart - feedBuildStart,
                  runMs: outputStart - runStart,
                  outputMs: outputEnd - outputStart,
                  tensorCreateMs: runStart - feedBuildStart,
                  logitReadMs,
                  kvExtractMs: kvEnd - kvStart,
                  gpuInputCount: inputLocations.gpu,
                  cpuInputCount: inputLocations.cpu,
                  gpuOutputCount: outputLocations.gpu,
                  cpuOutputCount: outputLocations.cpu,
                  gpuDownloadCount: outputDownloadCount,
                });

                return results;
              } finally {
                if (outputs) {
                  for (const tensor of Object.values(outputs)) disposeOrtTensor(tensor);
                }
                const retainedFeeds = new Set(
                  [...batchedEncoderKvTensors.values()].map((entry) => entry.tensor),
                );
                disposeOwnedDecoderFeeds(feeds, retainedFeeds);
              }
            }
          : undefined,
      });
    } finally {
      gpuKvBeamTracker.disposeAll();
      disposeWhisperEncoderKvCaches(scalarEncoderKvTensors, batchedEncoderKvTensors);
    }
  }

  const decodeMs = nowMs() - decoderStart;
  const decoderStepAvgMs = decoderStepCount > 0 ? decoderStepMs / decoderStepCount : undefined;
  const decoderStepP50Ms = percentile(decoderStepTimings, 50);
  const decoderStepP95Ms = percentile(decoderStepTimings, 95);
  const decoderStepMaxMs = decoderStepTimings.length > 0 ? Math.max(...decoderStepTimings) : undefined;
  const decoderKvCacheLocation =
    useExperimentalGpuKvCache && decoderGpuTensorOutputs > 0
      ? 'gpu-buffer'
      : requestedDecoderKvCacheLocation;

  return {
    result,
    timing: {
      decodeMs,
      decoderInitMs,
      decoderInitInputMs,
      decoderInitRunMs,
      decoderInitOutputMs,
      decoderStepMs,
      decoderStepFeedBuildMs,
      decoderStepTensorCloneMs,
      decoderStepRunMs,
      decoderStepOutputMs,
      decoderStepAvgMs,
      decoderStepP50Ms,
      decoderStepP95Ms,
      decoderStepMaxMs,
      decoderLogitProcessMs,
      decoderStepCount,
      decoderGpuTensorInputs,
      decoderCpuTensorInputs,
      decoderGpuTensorOutputs,
      decoderCpuTensorOutputs,
      decoderGpuTensorDownloads,
      decoderKvCacheLocation,
      decoderInitTensorCreateMs,
      decoderInitLogitReadMs,
      decoderInitKvExtractMs,
      decoderStepTensorCreateMs,
      decoderStepLogitReadMs,
      decoderStepKvMergeMs,
      decoderEncoderKvTensorReuses,
      decoderEncoderKvTensorCreates,
    },
  };
}
