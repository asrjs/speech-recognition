import type { TranscriptWarning } from '../../types/index.js';
import { computeWhisperDtwTokenTimestamps } from './attention-alignment.js';
import {
  buildWordsFromDtwTimestamps,
  collectSplitGraphTextTokenRows,
  extractCrossAttentions,
  hasVerifiedWhisperDecoderAlignment,
  processSplitGraphAlignment,
} from './executor-alignment.js';
import {
  createWhisperInt64TokenIds,
  disposeOwnedDecoderFeeds,
  disposeUnusedDecoderOutputs,
  isGpuBufferTensor,
  readOrtTensorData,
} from './executor-kv-cache.js';
import type {
  LoadedExecutorState,
  WhisperAlignmentRuntime,
} from './executor-types.js';
import {
  buildWhisperForcedAlignmentTokenIds,
  getWhisperForcedAlignmentTextRowStart,
  maybeCastEncoderHiddenStates,
  resolveWhisperLanguageCode,
  tensorDataAsFloat32,
} from './executor-utils.js';
import type { OrtSessionLike, OrtTensorLike } from './ort.js';
import type { WhisperTokenizer } from './tokenizer.js';
import type {
  WhisperNativeSegment,
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';
import {
  buildWhisperWordTimestampsFromDtwTokens,
  buildWhisperWordTimestampsFromTokenDetails,
} from './word-timestamps.js';

export async function materializeCpuEncoderHiddenStates(
  loaded: LoadedExecutorState,
  encoderHiddenStates: OrtTensorLike<Float32Array>,
  alignSession: OrtSessionLike,
  ownedEncoderTensors?: Set<OrtTensorLike>,
): Promise<OrtTensorLike<Float32Array>> {
  let tensor = encoderHiddenStates;
  if (isGpuBufferTensor(tensor)) {
    const { data } = await readOrtTensorData(tensor);
    const type = tensor.type === 'float16' ? 'float16' : 'float32';
    tensor = new loaded.ort.Tensor(
      type,
      data,
      [...tensor.dims],
    ) as OrtTensorLike<Float32Array>;
    ownedEncoderTensors?.add(tensor);
  }
  const cast = await maybeCastEncoderHiddenStates(tensor, alignSession, loaded.ort);
  if (cast !== encoderHiddenStates) ownedEncoderTensors?.add(cast);
  return cast;
}

export async function runForcedAlignment(
  loaded: LoadedExecutorState,
  configLanguages: readonly string[],
  encoderHiddenStates: OrtTensorLike<Float32Array>,
  language: string,
  textTokenIds: number[],
  task: 'transcribe' | 'translate' = 'transcribe',
  ownedEncoderTensors?: Set<OrtTensorLike>,
): Promise<{
  readonly crossAttentions: readonly OrtTensorLike<Float32Array>[];
  readonly logitsForText: Float32Array;
}> {
  const tokenizer = loaded.tokenizer;
  const resolvedLanguage = resolveWhisperLanguageCode(
    language,
    configLanguages,
  );
  const forcedIds = buildWhisperForcedAlignmentTokenIds(
    tokenizer,
    resolvedLanguage,
    textTokenIds,
    task,
    loaded.generationConfig.noTimestampsTokenId,
  );
  const alignmentPromptLen = forcedIds.length - textTokenIds.length - 1;
  const inputIds = createWhisperInt64TokenIds(forcedIds);
  const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, forcedIds.length]);
  const decoderSession = loaded.decoderSession!;
  const encoderForAlignment = await maybeCastEncoderHiddenStates(
    encoderHiddenStates,
    decoderSession,
    loaded.ort,
  );
  ownedEncoderTensors?.add(encoderForAlignment);

  const feeds: Record<string, unknown> = {
    input_ids: inputIdsTensor,
    encoder_hidden_states: encoderForAlignment,
  };

  const decoderInputNames = decoderSession.inputNames ?? [];
  const shouldFeed = (name: string): boolean =>
    decoderInputNames.length === 0 || decoderInputNames.includes(name);
  if (decoderInputNames.includes('use_cache_branch')) {
    feeds.use_cache_branch = new loaded.ort.Tensor('bool', new Uint8Array([0]), [1]);
  }

  // First step: provide empty past_key_values
  const numLayers = loaded.modelConfig.decoderLayers;
  const numHeads = loaded.modelConfig.decoderAttentionHeads;
  const headDim = loaded.modelConfig.headDim;
  const encoderSeqLen = encoderForAlignment.dims[1] as number;
  for (let i = 0; i < numLayers; i++) {
    const decoderKey = `past_key_values.${i}.decoder.key`;
    const decoderValue = `past_key_values.${i}.decoder.value`;
    const encoderKey = `past_key_values.${i}.encoder.key`;
    const encoderValue = `past_key_values.${i}.encoder.value`;
    if (shouldFeed(decoderKey)) {
      feeds[decoderKey] = new loaded.ort.Tensor(
        'float32', new Float32Array(0), [1, numHeads, 0, headDim]);
    }
    if (shouldFeed(decoderValue)) {
      feeds[decoderValue] = new loaded.ort.Tensor(
        'float32', new Float32Array(0), [1, numHeads, 0, headDim]);
    }
    const encoderCacheSize = 1 * numHeads * encoderSeqLen * headDim;
    if (shouldFeed(encoderKey)) {
      feeds[encoderKey] = new loaded.ort.Tensor(
        'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]);
    }
    if (shouldFeed(encoderValue)) {
      feeds[encoderValue] = new loaded.ort.Tensor(
        'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]);
    }
  }

  const retain = new Set<unknown>([encoderHiddenStates, encoderForAlignment]);
  const outputs = await decoderSession.run(feeds);
  try {
    const crossAttentions = await Promise.all(
      extractCrossAttentions(outputs).map(async (tensor) => {
        const { data } = await readOrtTensorData(tensor, { releaseGpu: true });
        return {
          data: tensorDataAsFloat32(data as ArrayBufferView),
          dims: tensor.dims,
          type: 'float32',
        } satisfies OrtTensorLike<Float32Array>;
      }),
    );

    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const totalVocab = (logitsTensor.dims[logitsTensor.dims.length - 1] as number) ?? 51865;
    const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
    const logits = tensorDataAsFloat32(logitsData.data as ArrayBufferView);
    const logitTimeSteps = Number(logitsTensor.dims[logitsTensor.dims.length - 2] ?? forcedIds.length);

    // forcedIds: [SOT, lang, task, no_timestamps, ...text, EOS]
    const textLogitStart = Math.max(0, alignmentPromptLen - 1);
    const textCount = textTokenIds.length;
    const logitsForText = new Float32Array(textCount * totalVocab);
    const availableRows = Math.max(0, Math.min(textCount, logitTimeSteps - textLogitStart));
    const srcOffset = textLogitStart * totalVocab;
    logitsForText.set(logits.subarray(srcOffset, srcOffset + availableRows * totalVocab));

    return { crossAttentions, logitsForText };
  } finally {
    disposeOwnedDecoderFeeds(feeds, retain);
    disposeUnusedDecoderOutputs(outputs, new Set());
  }
}

export async function runForcedAlignmentSplitGraph(
  loaded: Required<Pick<LoadedExecutorState, 'decoderAlignSession' | 'ort'>> & LoadedExecutorState,
  encoderHiddenStates: OrtTensorLike<Float32Array>,
  allTokenIds: readonly number[],
): Promise<{ readonly data: Float32Array; readonly dims: readonly number[] }> {
  const inputIds = createWhisperInt64TokenIds(allTokenIds);
  const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, allTokenIds.length]);
  const feeds: Record<string, unknown> = {
    input_ids: inputIdsTensor,
    encoder_hidden_states: encoderHiddenStates,
  };

  const retain = new Set<unknown>([encoderHiddenStates]);
  let outputs: Record<string, OrtTensorLike> | undefined;
  try {
    outputs = await loaded.decoderAlignSession.run(feeds);
    const alignKey = Object.keys(outputs)[0]!;
    const alignTensor = outputs[alignKey] as OrtTensorLike<Float32Array>;
    const rawData = isGpuBufferTensor(alignTensor)
      ? (await readOrtTensorData(alignTensor, { releaseGpu: true })).data
      : alignTensor.data;
    return {
      data: tensorDataAsFloat32(rawData as ArrayBufferView),
      dims: [...alignTensor.dims],
    };
  } finally {
    disposeOwnedDecoderFeeds(feeds, retain);
    if (outputs) disposeUnusedDecoderOutputs(outputs, new Set());
  }
}

export async function computeAttentionWordTimestampsSplitGraph(params: {
  readonly loaded: LoadedExecutorState;
  readonly ensureDecoderAlignSession: (loaded: LoadedExecutorState) => Promise<OrtSessionLike | undefined>;
  readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
  readonly tokenizer: WhisperTokenizer;
  readonly tokenDetails: WhisperNativeToken[];
  readonly allTokens: readonly number[];
  readonly promptLen: number;
  readonly language: string;
  readonly options: WhisperSeq2SeqTranscriptionOptions;
  readonly audioDurationSeconds?: number;
  readonly warnings?: TranscriptWarning[];
  readonly alignmentRuntime?: WhisperAlignmentRuntime;
  readonly alignmentReferenceRequiredButUnavailable?: boolean;
  readonly ownedEncoderTensors?: Set<OrtTensorLike>;
}): Promise<WhisperNativeTranscript['words']> {
  const {
    loaded,
    ensureDecoderAlignSession,
    encoderHiddenStates,
    tokenizer,
    tokenDetails,
    allTokens,
    promptLen,
    language,
    options,
    audioDurationSeconds,
    warnings,
    alignmentRuntime,
    alignmentReferenceRequiredButUnavailable = false,
    ownedEncoderTensors,
  } = params;

  const { tokenIds: textTokenIds, rowIndices } = collectSplitGraphTextTokenRows(
    allTokens,
    promptLen,
    (id) => !tokenizer.isSpecialTokenId(id) && !tokenizer.isTimestampTokenId(id),
  );
  if (textTokenIds.length === 0) return [];

  if (alignmentReferenceRequiredButUnavailable) {
    warnings?.push({
      code: 'whisper.alignment-reference-fallback',
      message:
        'The requested Whisper alignment-reference artifacts are unavailable; using generated timestamp interpolation.',
      recoverable: true,
    });
    return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
      timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
      timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
      language,
    });
  }

  if (
    !alignmentRuntime &&
    !hasVerifiedWhisperDecoderAlignment(loaded.decoderAlignCausalSelfAttention)
  ) {
    warnings?.push({
      code: 'whisper.decoder-align-legacy-fallback',
      message:
        'Legacy decoder_align cannot provide verified WebGPU word alignment; using generated timestamp interpolation until the artifact is re-exported.',
      recoverable: true,
    });
    return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
      timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
      timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
      language,
    });
  }

  try {
    const selectedAlignmentRuntime = alignmentRuntime;
    const decoderAlignSession = selectedAlignmentRuntime?.decoderAlignSession
      ?? await ensureDecoderAlignSession(loaded);
    if (!decoderAlignSession) {
      warnings?.push({
        code: 'whisper.decoder-align-unavailable',
        message: 'decoder_align is unavailable; using generated timestamp interpolation.',
        recoverable: true,
      });
      return [];
    }
    const encoderForAlign = await materializeCpuEncoderHiddenStates(
      loaded,
      encoderHiddenStates,
      decoderAlignSession,
      ownedEncoderTensors,
    );
    ownedEncoderTensors?.add(encoderForAlign);
    const alignmentTokenIds = buildWhisperForcedAlignmentTokenIds(
      tokenizer,
      language,
      textTokenIds,
      options.task ?? 'transcribe',
      loaded.generationConfig.noTimestampsTokenId,
    );
    const alignmentPromptLen = alignmentTokenIds.length - textTokenIds.length - 1;
    const alignmentTextRowStart = getWhisperForcedAlignmentTextRowStart(alignmentPromptLen);
    const alignmentRowIndices = Array.from(
      { length: textTokenIds.length + 1 },
      (_unused, index) => alignmentTextRowStart + index,
    );
    const { data: alignmentData, dims } = await runForcedAlignmentSplitGraph(
      { ...loaded, decoderAlignSession, ort: loaded.ort },
      encoderForAlign,
      alignmentTokenIds,
    );
    const frameCount = dims.length > 0 ? Number(dims[dims.length - 1]) : 0;
    const alignmentHeadCount = dims.length === 4 ? Number(dims[dims.length - 3]) : 1;
    const alignmentValuesAreLogits =
      (selectedAlignmentRuntime?.decoderAlignAttentionValues
        ?? loaded.decoderAlignAttentionValues) === 'logits';
    if (alignmentValuesAreLogits && (
      (selectedAlignmentRuntime?.decoderAlignAttentionLayout
        ?? loaded.decoderAlignAttentionLayout) !== 'selected_heads' ||
      dims.length !== 4 ||
      alignmentHeadCount < 1
    )) {
      throw new Error(
        `decoder_align manifest declares selected-head logits, but output shape [${dims.join(', ')}] ` +
        `does not expose the required [batch, head, token, frame] layout.`,
      );
    }
    const cropFrameCount = audioDurationSeconds && audioDurationSeconds > 0
      ? Math.max(1, Math.round(audioDurationSeconds / 0.02))
      : undefined;
    const dtwTimestamps = processSplitGraphAlignment({
      alignmentData,
      totalTokens: alignmentTokenIds.length,
      promptLen: alignmentPromptLen,
      textTokenCount: alignmentRowIndices.length,
      frameCount,
      medianFilterWidth: loaded.modelConfig.medianFilterWidth,
      timePrecisionSeconds: 0.02,
      textTokenRowIndices: alignmentRowIndices,
      cropFrameCount,
      alignmentHeadCount,
      alignmentValuesAreLogits,
    });

    return buildWhisperWordTimestampsFromDtwTokens(
      textTokenIds.map((id, i) => ({
        id,
        text: tokenizer.decode([id]),
        sourceIndex: Math.max(0, (rowIndices[i] ?? promptLen) - promptLen),
      })),
      dtwTimestamps.slice(0, textTokenIds.length + 1),
      { language, preserveLongDurations: true },
    );
  } catch (error) {
    if (error instanceof Error && error.message) {
      warnings?.push({
        code: 'whisper.decoder-align-fallback',
        message: `Whisper decoder alignment failed; using generated timestamp interpolation (${error.message}).`,
        recoverable: true,
      });
    }
    return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
      timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
      timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
      language,
    });
  }
}

export async function computeAttentionWordTimestamps(params: {
  readonly loaded: LoadedExecutorState;
  readonly configLanguages: readonly string[];
  readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
  readonly tokenizer: WhisperTokenizer;
  readonly tokenDetails: WhisperNativeToken[];
  readonly segments: WhisperNativeSegment[];
  readonly language: string;
  readonly options: WhisperSeq2SeqTranscriptionOptions;
  readonly audioDurationSeconds?: number;
  readonly warnings?: TranscriptWarning[];
  readonly ownedEncoderTensors?: Set<OrtTensorLike>;
  readonly runForcedAlignmentFn?: (
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    language: string,
    textTokenIds: number[],
    task?: 'transcribe' | 'translate',
    ownedEncoderTensors?: Set<OrtTensorLike>,
  ) => Promise<{
    readonly crossAttentions: readonly OrtTensorLike<Float32Array>[];
    readonly logitsForText: Float32Array;
  }>;
}): Promise<WhisperNativeTranscript['words']> {
  const {
    loaded,
    configLanguages,
    encoderHiddenStates,
    tokenizer,
    tokenDetails,
    segments,
    language,
    options,
    audioDurationSeconds,
    warnings,
    ownedEncoderTensors,
    runForcedAlignmentFn,
  } = params;
  const alignmentHeads = loaded.generationConfig.alignmentHeads;
  if (alignmentHeads.length === 0) {
    warnings?.push({
      code: 'whisper.decoder-attention-alignment-unavailable',
      message:
        'The merged Whisper artifact does not declare alignment heads; using generated timestamp interpolation for word timestamps.',
      recoverable: true,
    });
    return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
      timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
      timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
      language,
    });
  }

  const textTokenIds: number[] = [];
  for (const seg of segments) {
    const ids = tokenizer.encode(seg.text);
    for (const id of ids) {
      if (!tokenizer.isSpecialTokenId(id) && !tokenizer.isTimestampTokenId(id)) {
        textTokenIds.push(id);
      }
    }
  }
  if (textTokenIds.length === 0) return [];
  const alignmentTokenIds = buildWhisperForcedAlignmentTokenIds(
    tokenizer,
    resolveWhisperLanguageCode(language, configLanguages),
    textTokenIds,
    options.task ?? 'transcribe',
    loaded.generationConfig.noTimestampsTokenId,
  );
  const alignmentPromptLen = alignmentTokenIds.length - textTokenIds.length - 1;
  const alignmentTextRowStart = getWhisperForcedAlignmentTextRowStart(alignmentPromptLen);
  const alignmentRowCount = textTokenIds.length + 1;

  try {
    const alignment = runForcedAlignmentFn
      ? await runForcedAlignmentFn(
          loaded,
          encoderHiddenStates,
          language,
          textTokenIds,
          options.task ?? 'transcribe',
          ownedEncoderTensors,
        )
      : await runForcedAlignment(
          loaded,
          configLanguages,
          encoderHiddenStates,
          language,
          textTokenIds,
          options.task ?? 'transcribe',
          ownedEncoderTensors,
        );
    const crossAttentions = alignment.crossAttentions;
    if (crossAttentions.length === 0) {
      warnings?.push({
        code: 'whisper.decoder-cross-attention-unavailable',
        message:
          'The merged Whisper decoder does not export cross_attentions.*; using generated timestamp interpolation for word timestamps.',
        recoverable: true,
      });
      return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
        timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
        timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
        language,
      });
    }

    const attentionHeads = alignmentHeads.map(({ layer, head }) => {
      const layerTensor = crossAttentions[layer];
      if (!layerTensor) {
        throw new Error(`Cross-attention layer ${layer} missing from decoder outputs.`);
      }
      const numLayerHeads = (layerTensor.dims[1] as number) ?? 6;
      if (head >= numLayerHeads) {
        throw new Error(`Alignment head ${head} exceeds layer ${layer} head count ${numLayerHeads}.`);
      }
      const totalTokens = (layerTensor.dims[2] as number) ?? 0;
      const totalFramesPerHead = (layerTensor.dims[3] as number) ?? 0;
      const requiredTokenRows = Math.max(
        alignmentTokenIds.length,
        alignmentTextRowStart + alignmentRowCount,
      );
      if (totalTokens < requiredTokenRows) {
        throw new Error(
          `Cross-attention layer ${layer} has ${totalTokens} token rows; expected at least ${requiredTokenRows}.`,
        );
      }
      const headSize = totalTokens * totalFramesPerHead;
      const headOffset = head * headSize;
      const headValues = new Float32Array(totalTokens * totalFramesPerHead);
      for (let tokenIndex = 0; tokenIndex < totalTokens; tokenIndex++) {
        const sourceOffset = headOffset + tokenIndex * totalFramesPerHead;
        headValues.set(
          layerTensor.data.subarray(sourceOffset, sourceOffset + totalFramesPerHead),
          tokenIndex * totalFramesPerHead,
        );
      }
      return {
        values: headValues,
        tokenCount: totalTokens,
        frameCount: totalFramesPerHead,
      };
    });

    const fullFrameCount = attentionHeads[0]?.frameCount ?? 0;
    const croppedFrames = audioDurationSeconds && audioDurationSeconds > 0
      ? Math.max(1, Math.min(fullFrameCount, Math.round(audioDurationSeconds / 0.02)))
      : fullFrameCount;
    const dtwTimestamps = computeWhisperDtwTokenTimestamps({
      attentionHeads,
      tokenCount: alignmentRowCount,
      normalizationTokenCount: alignmentTokenIds.length,
      tokenRowIndices: Array.from(
        { length: alignmentRowCount },
        (_unused, index) => alignmentTextRowStart + index,
      ),
      frameCount: croppedFrames,
      timePrecisionSeconds: 0.02,
    });

    const logits = alignment.logitsForText;
    const totalVocab = (logits?.length ?? 0) / textTokenIds.length;
    let tokenLogprobs: Float32Array | undefined;
    if (logits && totalVocab > 0) {
      tokenLogprobs = new Float32Array(textTokenIds.length);
      for (let t = 0; t < textTokenIds.length; t++) {
        const tokenId = textTokenIds[t] ?? 0;
        let maxVal = -Infinity;
        const tStart = t * totalVocab;
        for (let v = 0; v < totalVocab; v++) {
          const val = logits[tStart + v] ?? -Infinity;
          if (val > maxVal) maxVal = val;
        }
        let sumExp = 0;
        for (let v = 0; v < totalVocab; v++) {
          sumExp += Math.exp((logits[tStart + v] ?? -Infinity) - maxVal);
        }
        const logProb = (logits[tStart + tokenId] ?? -Infinity) - maxVal - Math.log(sumExp);
        tokenLogprobs[t] = logProb;
      }
    }

    return buildWordsFromDtwTimestamps(
      tokenizer,
      textTokenIds,
      dtwTimestamps.slice(0, textTokenIds.length + 1),
      tokenLogprobs,
    );
  } catch {
    warnings?.push({
      code: 'whisper.decoder-attention-alignment-fallback',
      message:
        'Merged Whisper attention alignment failed; using generated timestamp interpolation for word timestamps.',
      recoverable: true,
    });
    return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
      timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
      timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
      language,
    });
  }
}
