import type {
  AbortSignalLike,
  TranscriptWarning,
  TranscriptionProgressEvent,
} from '../../types/index.js';
import { argmax, confidenceFromLogits } from '../../inference/index.js';
import { PipelineAbortedError } from '../../pipeline/composition.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import { disposeOrtOutputs, type OrtModuleLike, type OrtSessionLike, type OrtTensorLike } from './ort.js';
import type {
  NemoTdtDecoderStateSnapshot,
  NemoTdtModelConfig,
  NemoTdtTranscriptionOptions,
} from './types.js';

export const TDT_BATCH_MIN_FRAMES = 2;
export const TDT_BATCH_MAX_FRAMES = 32;
export const TDT_BATCH_MIN_SAMPLE_COLUMNS = 24;
export const TDT_BATCH_MIN_COLUMN_UTILIZATION = 0.7;
export const TDT_BLANK_REPROBE_VISITS = 6;

export interface NemoTdtBatchState {
  tdtBatchAllowed: boolean;
  tdtBatchWidth: number;
  tdtBatchColumnsScored: number;
  tdtBatchColumnsUseful: number;
  tdtSequentialBlankVisits: number;
}

export function clampProgress(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function estimateRemainingMs(elapsedMs: number, progress: number): number | undefined {
  if (progress <= 0 || progress >= 1) {
    return undefined;
  }

  return roundMetric((elapsedMs / progress) * (1 - progress), 2);
}

export function emitTranscriptionProgress(
  options: NemoTdtTranscriptionOptions,
  event: TranscriptionProgressEvent,
): void {
  options.onProgress?.(event);
}

export function throwIfDecodeAborted(signal: AbortSignalLike | null | undefined): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError('decode');
  }
}

export function snapshotDecoderState(
  state: {
    readonly state1: OrtTensorLike<Float32Array>;
    readonly state2: OrtTensorLike<Float32Array>;
  } | null,
): NemoTdtDecoderStateSnapshot | undefined {
  if (!state) {
    return undefined;
  }

  return {
    s1: new Float32Array(state.state1.data),
    s2: new Float32Array(state.state2.data),
    dims1: [...state.state1.dims],
    dims2: [...state.state2.dims],
  };
}

export interface ExtractedNemoTdtEncoderOutput {
  readonly encoderData: Float32Array;
  readonly encoderIsBdt: boolean;
  readonly featureSize: number;
  readonly frameCount: number;
  readonly encoderTimeStride: number;
}

export function extractAndResolveNemoTdtEncoderOutput(
  encoderOutputs: Record<string, OrtTensorLike> | undefined,
  config: NemoTdtModelConfig,
): ExtractedNemoTdtEncoderOutput {
  let encoderData: Float32Array;
  let dims: number[];
  let encodedFrameCount: number | undefined;
  try {
    const encoderTensor = (encoderOutputs?.outputs ??
      Object.values(encoderOutputs ?? {})[0]) as OrtTensorLike<Float32Array>;
    dims = [...encoderTensor.dims];
    if (dims.length !== 3 || dims[0] !== 1) {
      throw new Error(`Unexpected NeMo TDT encoder output shape: [${dims.join(', ')}].`);
    }
    const encodedLengths = Object.entries(encoderOutputs ?? {}).find(
      ([name]) => name !== 'outputs' && /encoded_?lengths?$/i.test(name),
    )?.[1];
    const encodedLengthValue = (
      encodedLengths?.data as ArrayLike<number | bigint> | undefined
    )?.[0];
    if (encodedLengthValue !== undefined && Number.isFinite(Number(encodedLengthValue))) {
      encodedFrameCount = Number(encodedLengthValue);
    }
    if (
      config.encoderOutputLayout !== undefined &&
      encodedLengths !== undefined &&
      (encodedFrameCount === undefined || !Number.isSafeInteger(encodedFrameCount))
    ) {
      throw new Error(`Invalid NeMo TDT encoded_lengths=${String(encodedLengthValue)}.`);
    }
    encoderData =
      encoderTensor.data instanceof Float32Array
        ? new Float32Array(encoderTensor.data)
        : Float32Array.from(encoderTensor.data as ArrayLike<number>);
  } finally {
    disposeOrtOutputs(encoderOutputs);
  }

  // Resolve the encoder-output layout from the encoder graph's own length
  // output, not from comparing axis magnitudes: a clip longer than the
  // feature dimension (1024 frames, about 82 s at the 10 ms/8x NeMo stride)
  // otherwise flips the inference and feeds a transposed tensor to the
  // decoder, which rejects it as "Got: <frames> Expected: <features>".
  // Equality is tested before the padded-axis cases so that a time-major
  // [1, frames, features] output whose frames stay under the feature count
  // is not mistaken for a padded feature-major tensor.
  let encoderIsBdt: boolean;
  let featureSize: number;
  let frameCount: number;
  if (config.encoderOutputLayout !== undefined) {
    const layout = config.encoderOutputLayout;
    if (layout !== 'bdt' && layout !== 'btd') {
      throw new Error(`Invalid NeMo TDT encoderOutputLayout: ${layout}.`);
    }
    encoderIsBdt = layout === 'bdt';
    featureSize = dims[encoderIsBdt ? 1 : 2]!;
    const physicalFrames = dims[encoderIsBdt ? 2 : 1]!;
    frameCount = encodedFrameCount ?? physicalFrames;
    if (
      !Number.isSafeInteger(featureSize) ||
      featureSize <= 0 ||
      !Number.isSafeInteger(physicalFrames) ||
      physicalFrames <= 0 ||
      !Number.isSafeInteger(frameCount) ||
      frameCount <= 0 ||
      frameCount > physicalFrames
    ) {
      throw new Error(
        `NeMo TDT encoded_lengths=${frameCount} exceeds or is invalid for ${layout} time axis ${physicalFrames}.`,
      );
    }
  } else if (encodedFrameCount === undefined) {
    encoderIsBdt = (dims[1] ?? 0) > (dims[2] ?? 0);
    featureSize = encoderIsBdt ? dims[1]! : dims[2]!;
    frameCount = encoderIsBdt ? dims[2]! : dims[1]!;
  } else if (dims[2] === encodedFrameCount) {
    encoderIsBdt = true;
    featureSize = dims[1]!;
    frameCount = encodedFrameCount;
  } else if (dims[1] === encodedFrameCount) {
    encoderIsBdt = false;
    featureSize = dims[2]!;
    frameCount = encodedFrameCount;
  } else if ((dims[1] ?? 0) <= encodedFrameCount && (dims[2] ?? 0) > encodedFrameCount) {
    encoderIsBdt = true;
    featureSize = dims[1]!;
    frameCount = encodedFrameCount;
  } else if ((dims[2] ?? 0) < encodedFrameCount && (dims[1] ?? 0) > encodedFrameCount) {
    encoderIsBdt = false;
    featureSize = dims[2]!;
    frameCount = encodedFrameCount;
  } else {
    throw new Error(
      `NeMo TDT encoder output shape [${dims.join(', ')}] matches neither axis of encoded_lengths=${encodedFrameCount}.`,
    );
  }

  // Physical length of the encoder time axis. When the encoder pads its time
  // dimension, feature rows are this far apart in the flat buffer, so the
  // valid frameCount above must not be used as a stride.
  const encoderTimeStride = encoderIsBdt ? (dims[2] ?? frameCount) : (dims[1] ?? frameCount);

  return {
    encoderData,
    encoderIsBdt,
    featureSize,
    frameCount,
    encoderTimeStride,
  };
}

export interface RunNemoTdtDecodeLoopParams {
  readonly modelId: string;
  readonly backendId: string;
  readonly config: NemoTdtModelConfig;
  readonly options: NemoTdtTranscriptionOptions;
  readonly ort: OrtModuleLike;
  readonly decoderSession: OrtSessionLike;
  readonly vocabSize: number;
  readonly blankId: number;
  readonly computeConfidence: boolean;
  readonly encoderOutput: ExtractedNemoTdtEncoderOutput;
  readonly batchState: NemoTdtBatchState;
  readonly warnings: TranscriptWarning[];
  readonly transcriptionStart: number;
  readonly preprocessMs: number;
  readonly encodeMs: number;
}

export interface NemoTdtDecodeLoopResult {
  readonly tokenIds: number[];
  readonly tokenTimestamps: Array<[number, number]>;
  readonly tokenConfidences: number[];
  readonly tokenFrameIndices: number[];
  readonly tokenLogProbs: number[];
  readonly tokenTdtSteps: number[];
  readonly frameConfidenceStats: Map<number, { sum: number; count: number }>;
  readonly decodeIterations: number;
  readonly decoderGridBatchRuns: number;
  readonly decodeMs: number;
  readonly decoderStateSnapshot?: NemoTdtDecoderStateSnapshot;
}

export async function runNemoTdtDecodeLoop(
  params: RunNemoTdtDecodeLoopParams,
): Promise<NemoTdtDecodeLoopResult> {
  const {
    modelId,
    backendId,
    config,
    options,
    ort,
    decoderSession,
    vocabSize,
    blankId,
    computeConfidence,
    encoderOutput,
    batchState,
    warnings,
    transcriptionStart,
    preprocessMs,
    encodeMs,
  } = params;
  const { encoderData, encoderIsBdt, featureSize, frameCount, encoderTimeStride } = encoderOutput;

  const frameTime = config.frameShiftSeconds * config.subsamplingFactor;
  if (config.vocabularySize && config.vocabularySize !== vocabSize) {
    warnings.push({
      code: 'nemo-tdt.vocabulary-size-mismatch',
      message: `Configured vocabulary size ${config.vocabularySize} does not match tokenizer vocabulary size ${vocabSize}. Using tokenizer vocabulary size for decoder output slicing.`,
      recoverable: true,
    });
  }
  const encoderFrameBuffer = new Float32Array(featureSize);
  const encoderFrameTensor = new ort.Tensor('float32', encoderFrameBuffer, [
    1,
    featureSize,
    1,
  ]);
  const targetIdBuffer = new Int32Array(1);
  const targetTensor = new ort.Tensor('int32', targetIdBuffer, [1, 1]);
  const targetLengthTensor = new ort.Tensor('int32', new Int32Array([1]), [1]);

  const predictionLayers = config.predictionLayers ?? 2;
  const predictionHiddenSize = config.predictionHiddenSize ?? 640;
  let decoderState: {
    state1: OrtTensorLike<Float32Array>;
    state2: OrtTensorLike<Float32Array>;
  } | null = {
    state1: new ort.Tensor(
      'float32',
      new Float32Array(predictionLayers * predictionHiddenSize),
      [predictionLayers, 1, predictionHiddenSize],
    ),
    state2: new ort.Tensor(
      'float32',
      new Float32Array(predictionLayers * predictionHiddenSize),
      [predictionLayers, 1, predictionHiddenSize],
    ),
  };

  const tokenIds: number[] = [];
  const tokenTimestamps: Array<[number, number]> = [];
  const tokenConfidences: number[] = [];
  const tokenFrameIndices: number[] = [];
  const tokenLogProbs: number[] = [];
  const tokenTdtSteps: number[] = [];
  const frameConfidenceStats = new Map<number, { sum: number; count: number }>();
  let emittedOnFrame = 0;
  let decodeIterations = 0;

  const disposeTensor = (tensor: OrtTensorLike | undefined): void => {
    tensor?.dispose?.();
  };

  const disposeDecoderState = (
    state: { state1: OrtTensorLike; state2: OrtTensorLike } | null,
    keep?: { state1: OrtTensorLike; state2: OrtTensorLike } | null,
  ): void => {
    if (!state) {
      return;
    }

    if (state.state1 !== keep?.state1) {
      disposeTensor(state.state1);
    }
    if (state.state2 !== keep?.state2) {
      disposeTensor(state.state2);
    }
  };

  const decoderStart = nowMs();
  let lastReportedDecodeUnits = -1;
  let decoderGridBatchRuns = 0;
  batchState.tdtBatchColumnsScored = 0;
  batchState.tdtBatchColumnsUseful = 0;
  batchState.tdtSequentialBlankVisits = 0;
  try {
    for (let frameIndex = 0; frameIndex < frameCount; ) {
      throwIfDecodeAborted(options.signal);
      if (
        batchState.tdtBatchAllowed &&
        options.gridBatching !== false &&
        emittedOnFrame === 0 &&
        frameCount - frameIndex > 1 &&
        (batchState.tdtBatchColumnsScored < TDT_BATCH_MIN_SAMPLE_COLUMNS ||
          batchState.tdtBatchColumnsUseful >=
            TDT_BATCH_MIN_COLUMN_UTILIZATION * batchState.tdtBatchColumnsScored)
      ) {
        const width = Math.min(
          frameCount - frameIndex,
          batchState.tdtBatchWidth,
          TDT_BATCH_MAX_FRAMES,
        );
        const batchData = new Float32Array(width * featureSize);
        if (encoderIsBdt) {
          // The batch tensor is declared [1, features, width] and the graph
          // transposes it with perm (0,2,1), so the flat layout must be
          // feature-major: element (f, w) lives at f * width + w. A
          // row-major fill silently transposes the grid for width > 1
          // (width 1 is layout-invariant, which is why the sequential path
          // never hits this and layout-blind unit mocks pass).
          for (let fIdx = 0; fIdx < featureSize; fIdx += 1) {
            const sourceRow = fIdx * encoderTimeStride + frameIndex;
            batchData.set(encoderData.subarray(sourceRow, sourceRow + width), fIdx * width);
          }
        } else {
          // Source is frame-major [1, frames, features]; the target
          // tensor is feature-major [1, features, width], so scatter the
          // copy instead of a flat memcpy (a memcpy transposes width > 1).
          for (let row = 0; row < width; row += 1) {
            const sourceOffset = (frameIndex + row) * featureSize;
            for (let fIdx = 0; fIdx < featureSize; fIdx += 1) {
              batchData[fIdx * width + row] = encoderData[sourceOffset + fIdx] ?? 0;
            }
          }
        }
        const batchEncoderTensor = new ort.Tensor('float32', batchData, [
          1,
          featureSize,
          width,
        ]);
        const batchTargetBuffer = new Int32Array(1);
        batchTargetBuffer[0] = tokenIds.length > 0 ? tokenIds[tokenIds.length - 1]! : blankId;
        const batchTargetTensor = new ort.Tensor('int32', batchTargetBuffer, [1, 1]);
        let batchOutputs: Record<string, OrtTensorLike> | undefined;
        try {
          batchOutputs = await decoderSession.run({
            encoder_outputs: batchEncoderTensor,
            targets: batchTargetTensor,
            target_length: targetLengthTensor,
            input_states_1: decoderState?.state1,
            input_states_2: decoderState?.state2,
          });
          const gridLogits = batchOutputs.outputs as OrtTensorLike<Float32Array>;
          const gridRawData = gridLogits.data;
          const gridTotal =
            gridRawData instanceof Float32Array
              ? gridRawData
              : Float32Array.from(gridRawData as ArrayLike<number>);
          const rowLength = Math.floor(gridTotal.length / width);
          if (
            rowLength <= 0 ||
            gridTotal.length !== rowLength * width ||
            rowLength <= vocabSize
          ) {
            // The graph did not return the per-frame row layout the
            // speculative scan needs; permanently fall back to the
            // sequential single-frame loop below.
            batchState.tdtBatchAllowed = false;
          } else {
            decoderGridBatchRuns += 1;
            let cursor = frameIndex;
            let examined = 0;
            let emissionRow = -1;
            let emissionToken = 0;
            let emissionStep = 0;
            while (cursor - frameIndex < width) {
              const rowIndex = cursor - frameIndex;
              const rowView = gridTotal.subarray(
                rowIndex * rowLength,
                (rowIndex + 1) * rowLength,
              );
              examined += 1;
              const tokenId = argmax(rowView, 0, vocabSize);
              const step = argmax(rowView, vocabSize, rowLength - vocabSize) - vocabSize;
              if (computeConfidence) {
                const scanConfidence = confidenceFromLogits(rowView, tokenId, vocabSize);
                const existingScanStats = frameConfidenceStats.get(cursor);
                if (existingScanStats) {
                  existingScanStats.sum += scanConfidence.confidence;
                  existingScanStats.count += 1;
                } else {
                  frameConfidenceStats.set(cursor, {
                    sum: scanConfidence.confidence,
                    count: 1,
                  });
                }
              }
              if (tokenId !== blankId) {
                emissionRow = rowIndex;
                emissionToken = tokenId;
                emissionStep = step;
                break;
              }
              // Blank rows keep the predictor state, so a duration
              // skip lands on another row that is already scored by
              // this same run.
              cursor += step > 0 ? step : 1;
            }
            decodeIterations += examined;
            batchState.tdtBatchColumnsScored += width;
            batchState.tdtBatchColumnsUseful += examined;
            if (emissionRow === -1) {
              frameIndex = cursor;
              emittedOnFrame = 0;
              batchState.tdtBatchWidth = Math.min(width * 2, TDT_BATCH_MAX_FRAMES);
            } else {
              batchState.tdtBatchWidth = TDT_BATCH_MIN_FRAMES;
              const emissionFrame = frameIndex + emissionRow;
              const rowView = gridTotal.subarray(
                emissionRow * rowLength,
                (emissionRow + 1) * rowLength,
              );
              if (computeConfidence) {
                const stepConfidence = confidenceFromLogits(rowView, emissionToken, vocabSize);
                tokenConfidences.push(stepConfidence.confidence);
                tokenLogProbs.push(stepConfidence.logProb);
              }
              const durationFrames = Math.max(1, emissionStep);
              tokenIds.push(emissionToken);
              tokenTimestamps.push([
                roundTimestampSeconds(emissionFrame * frameTime),
                roundTimestampSeconds(
                  Math.min(frameCount, emissionFrame + durationFrames) * frameTime,
                ),
              ]);
              tokenFrameIndices.push(emissionFrame);
              tokenTdtSteps.push(emissionStep);
              const nextState = {
                state1: batchOutputs.output_states_1 as OrtTensorLike<Float32Array>,
                state2: batchOutputs.output_states_2 as OrtTensorLike<Float32Array>,
              };
              disposeDecoderState(decoderState, nextState);
              decoderState = nextState;
              delete batchOutputs.output_states_1;
              delete batchOutputs.output_states_2;
              if (emissionStep > 0) {
                frameIndex = emissionFrame + emissionStep;
                emittedOnFrame = 0;
              } else if (1 >= (config.maxSymbolsPerStep ?? 10)) {
                frameIndex = emissionFrame + 1;
                emittedOnFrame = 0;
              } else {
                frameIndex = emissionFrame;
                emittedOnFrame = 1;
              }
            }
          }
        } catch (error) {
          if (error instanceof PipelineAbortedError) {
            throw error;
          }
          batchState.tdtBatchAllowed = false;
        } finally {
          if (batchOutputs) {
            for (const tensor of Object.values(batchOutputs)) {
              disposeTensor(tensor);
            }
          }
          batchEncoderTensor.dispose?.();
          batchTargetTensor.dispose?.();
        }
        const batchCompletedUnits = Math.min(frameCount, frameIndex);
        if (batchCompletedUnits > lastReportedDecodeUnits) {
          lastReportedDecodeUnits = batchCompletedUnits;
          const decodeProgress = clampProgress(0.4 + (batchCompletedUnits / frameCount) * 0.5);
          const decodeElapsedMs = nowMs() - transcriptionStart;
          emitTranscriptionProgress(options, {
            stage: 'decode',
            progress: decodeProgress,
            elapsedMs: roundMetric(decodeElapsedMs),
            remainingMs: estimateRemainingMs(decodeElapsedMs, decodeProgress),
            completedUnits: batchCompletedUnits,
            totalUnits: frameCount,
            modelId,
            backendId,
            message:
              'Decoded ' +
              batchCompletedUnits +
              '/' +
              frameCount +
              ' encoder frames for ' +
              modelId +
              '.',
            metrics: {
              preprocessMs: roundMetric(preprocessMs),
              encodeMs: roundMetric(encodeMs),
              decodeMs: roundMetric(nowMs() - decoderStart),
            },
          });
        }
        continue;
      }
      decodeIterations += 1;
      if (encoderIsBdt) {
        for (let featureIndex = 0; featureIndex < featureSize; featureIndex += 1) {
          encoderFrameBuffer[featureIndex] =
            encoderData[featureIndex * encoderTimeStride + frameIndex] ?? 0;
        }
      } else {
        const offset = frameIndex * featureSize;
        encoderFrameBuffer.set(encoderData.subarray(offset, offset + featureSize));
      }

      targetIdBuffer[0] = tokenIds.length > 0 ? tokenIds[tokenIds.length - 1]! : blankId;
      const decoderOutputs = await decoderSession.run({
        encoder_outputs: encoderFrameTensor,
        targets: targetTensor,
        target_length: targetLengthTensor,
        input_states_1: decoderState?.state1,
        input_states_2: decoderState?.state2,
      });

      const logits = decoderOutputs.outputs as OrtTensorLike<Float32Array>;
      const nextState = {
        state1: decoderOutputs.output_states_1 as OrtTensorLike<Float32Array>,
        state2: decoderOutputs.output_states_2 as OrtTensorLike<Float32Array>,
      };
      try {
        // ORT already exposes float32 logits as a typed-array view. Consume
        // that view synchronously before disposing the output tensor instead
        // of allocating a full per-step copy. Non-float32-compatible output
        // views still get normalized into an owned Float32Array.
        const logitsData =
          logits.data instanceof Float32Array
            ? logits.data
            : Float32Array.from(logits.data as ArrayLike<number>);
        if (logitsData.length < vocabSize) {
          throw new Error(
            `NeMo TDT decoder output is too small (${logitsData.length}) for tokenizer vocabulary size ${vocabSize}.`,
          );
        }
        if (logitsData.length === vocabSize) {
          throw new Error('NeMo TDT decoder output is missing required TDT duration logits.');
        }
        const tokenId = argmax(logitsData, 0, vocabSize);
        const durationOffset = vocabSize;
        const step =
          argmax(logitsData, durationOffset, logitsData.length - durationOffset) - durationOffset;
        const stepConfidence = computeConfidence
          ? confidenceFromLogits(logitsData, tokenId, vocabSize)
          : undefined;
        if (stepConfidence) {
          const existingFrameConfidence = frameConfidenceStats.get(frameIndex);
          if (existingFrameConfidence) {
            existingFrameConfidence.sum += stepConfidence.confidence;
            existingFrameConfidence.count += 1;
          } else {
            frameConfidenceStats.set(frameIndex, { sum: stepConfidence.confidence, count: 1 });
          }
        }

        if (tokenId !== blankId) {
          const durationFrames = Math.max(1, step);
          tokenIds.push(tokenId);
          tokenTimestamps.push([
            roundTimestampSeconds(frameIndex * frameTime),
            roundTimestampSeconds(Math.min(frameCount, frameIndex + durationFrames) * frameTime),
          ]);
          if (stepConfidence) {
            tokenConfidences.push(stepConfidence.confidence);
            tokenLogProbs.push(stepConfidence.logProb);
          }
          tokenFrameIndices.push(frameIndex);
          tokenTdtSteps.push(step);
          emittedOnFrame += 1;
          batchState.tdtSequentialBlankVisits = 0;

          disposeDecoderState(decoderState, nextState);
          decoderState = nextState;
        } else {
          disposeDecoderState(nextState, decoderState);
          // A sustained sequential blank run means the audio entered a
          // blank-dominant region after the gate closed (typical for
          // speech/silence alternation and streaming chunks). Re-open the
          // sampling window so the grid can cross the silence; a fluke
          // re-probe costs at most tdtBatchMinSampleColumns wasted
          // columns before the gate closes again.
          batchState.tdtSequentialBlankVisits += 1;
          if (batchState.tdtSequentialBlankVisits >= TDT_BLANK_REPROBE_VISITS) {
            batchState.tdtSequentialBlankVisits = 0;
            batchState.tdtBatchColumnsScored = 0;
            batchState.tdtBatchColumnsUseful = 0;
            batchState.tdtBatchWidth = TDT_BATCH_MIN_FRAMES;
          }
        }

        if (step > 0) {
          frameIndex += step;
          emittedOnFrame = 0;
        } else if (
          tokenId === blankId ||
          emittedOnFrame >= (config.maxSymbolsPerStep ?? 10)
        ) {
          frameIndex += 1;
          emittedOnFrame = 0;
        }

        const completedUnits = Math.min(frameCount, frameIndex);
        if (completedUnits > lastReportedDecodeUnits) {
          lastReportedDecodeUnits = completedUnits;
          const decodeProgress = clampProgress(0.4 + (completedUnits / frameCount) * 0.5);
          const decodeElapsedMs = nowMs() - transcriptionStart;
          emitTranscriptionProgress(options, {
            stage: 'decode',
            progress: decodeProgress,
            elapsedMs: roundMetric(decodeElapsedMs),
            remainingMs: estimateRemainingMs(decodeElapsedMs, decodeProgress),
            completedUnits,
            totalUnits: frameCount,
            modelId,
            backendId,
            message: `Decoded ${completedUnits}/${frameCount} encoder frames for ${modelId}.`,
            metrics: {
              preprocessMs: roundMetric(preprocessMs),
              encodeMs: roundMetric(encodeMs),
              decodeMs: roundMetric(nowMs() - decoderStart),
            },
          });
        }
      } finally {
        disposeTensor(logits);
        for (const tensor of Object.values(decoderOutputs)) {
          if (tensor !== logits && tensor !== nextState.state1 && tensor !== nextState.state2) {
            disposeTensor(tensor);
          }
        }
      }
    }
  } catch (error) {
    disposeTensor(decoderState?.state1);
    disposeTensor(decoderState?.state2);
    decoderState = null;
    throw error;
  } finally {
    targetTensor.dispose?.();
    targetLengthTensor.dispose?.();
    encoderFrameTensor.dispose?.();
  }

  const decodeMs = nowMs() - decoderStart;
  const decoderStateSnapshot = options.returnDecoderState
    ? snapshotDecoderState(decoderState)
    : undefined;
  disposeTensor(decoderState?.state1);
  disposeTensor(decoderState?.state2);

  return {
    tokenIds,
    tokenTimestamps,
    tokenConfidences,
    tokenFrameIndices,
    tokenLogProbs,
    tokenTdtSteps,
    frameConfidenceStats,
    decodeIterations,
    decoderGridBatchRuns,
    decodeMs,
    decoderStateSnapshot,
  };
}
