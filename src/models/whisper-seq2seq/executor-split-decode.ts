import type {
  AudioBufferLike,
  TranscriptMetrics,
} from '../../types/index.js';
import { WhisperMelProcessor } from '../../audio/whisper-mel.js';
import { nowMs, roundMetric } from '../../runtime/timing.js';
import {
  buildWhisperSegments,
  computeAttentionWordTimestampsSplitGraph,
  finalizeWordTimestamps,
  hasVerifiedWhisperDecoderAlignment,
  resolveWhisperWordTimestampSource,
  shouldReturnWordTimestamps,
} from './executor-alignment.js';
import { detectLanguageFromEncoder } from './executor-decoder-ops.js';
import { formatWhisperSessionError } from './executor-init.js';
import {
  assertExperimentalGpuKvCacheIsGreedyOnly,
  clampProgress,
  disposeOwnedWhisperTensors,
  emitTranscriptionProgress,
  estimateRemainingMs,
  isGpuBufferTensor,
} from './executor-kv-cache.js';
import { runSplitGraphDecodeWithTiming } from './executor-split-callbacks.js';
import type {
  LoadedExecutorState,
  WhisperAlignmentRuntime,
} from './executor-types.js';
import {
  maybeCastEncoderHiddenStates,
  maybeCastWhisperFeatureTensor,
  resolveWhisperLanguageCode,
  resolveWhisperLanguageTokenId,
  runWhisperEncoderOnFeatures,
} from './executor-utils.js';
import type { OrtSessionLike, OrtTensorLike } from './ort.js';
import { WhisperTimestampLogitProcessor } from './processors.js';
import type { WhisperTokenizer } from './tokenizer.js';
import type {
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperSeq2SeqModelConfig,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';

export function formatWhisperTokenText(tokenizer: WhisperTokenizer, tokenId: number): string {
  if (tokenizer.isTimestampTokenId(tokenId) || tokenizer.isSpecialTokenId(tokenId)) {
    return tokenizer.idsToTokens([tokenId])[0] ?? '';
  }
  return tokenizer.decode([tokenId]);
}

export interface TranscribeWithSplitGraphContext {
  readonly modelId: string;
  readonly backendId: string;
  readonly config: WhisperSeq2SeqModelConfig;
  readonly loaded: LoadedExecutorState;
  readonly ensureDecoderAlignSession: (
    loaded: LoadedExecutorState,
  ) => Promise<OrtSessionLike | undefined>;
  readonly runAlignmentReferenceEncoder: (
    loaded: LoadedExecutorState,
    paddedFeatures: Float32Array,
    melBins: number,
    melInputFrames: number,
    ownedEncoderTensors: Set<OrtTensorLike>,
  ) => Promise<{
    readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
    readonly alignmentRuntime: WhisperAlignmentRuntime;
  }>;
}

export async function transcribeWithWhisperSplitGraph(
  ctx: TranscribeWithSplitGraphContext,
  audio: AudioBufferLike,
  options: WhisperSeq2SeqTranscriptionOptions,
): Promise<WhisperNativeTranscript> {
  const { modelId, backendId, config, loaded } = ctx;
  const transcriptionStart = nowMs();
  const warnings = [...loaded.warnings];
  const splitLoaded = loaded as Required<LoadedExecutorState>;

  emitTranscriptionProgress(options, {
    stage: 'start',
    progress: 0,
    elapsedMs: 0,
    modelId,
    backendId,
    message: `Starting transcription for ${modelId}.`,
  });

  const requestedNumBeams = Math.max(1, Math.floor(options.numBeams ?? 1));
  const requestedBestOf = Math.max(1, Math.floor(options.bestOf ?? 1));
  const requestedTemperature = options.temperature ?? 0;
  const useExperimentalGpuKvCache = Boolean(
    splitLoaded.experimentalGpuKvCache &&
    splitLoaded.decoderBackendForOrt === 'webgpu',
  );
  const gpuKvBeamActive = splitLoaded.experimentalGpuKvBeam === true
    && splitLoaded.decoderBackendForOrt === 'webgpu'
    && requestedNumBeams > 1
    && requestedBestOf <= 1
    && requestedTemperature === 0;

  assertExperimentalGpuKvCacheIsGreedyOnly({
    enabled: useExperimentalGpuKvCache && !gpuKvBeamActive,
    numBeams: requestedNumBeams,
    bestOf: requestedBestOf,
    temperature: requestedTemperature,
  });

  // 1. Preprocess audio to mel spectrogram
  const preprocessStart = nowMs();
  const melBins = loaded.modelConfig.numMelBins ?? config.melBins;
  const melProcessor = new WhisperMelProcessor({ nMels: melBins });
  const pcmData = audio.channels?.[0] ?? new Float32Array(0);
  const melResult = melProcessor.process(pcmData);
  const encoderOutputPositions = config.maxSourcePositions;
  const melInputFrames = encoderOutputPositions <= 1500 ? encoderOutputPositions * 2 : encoderOutputPositions;
  const paddedFeatures = WhisperMelProcessor.padToFrames(melResult, melInputFrames);
  const preprocessMs = nowMs() - preprocessStart;
  const preprocessElapsedMs = nowMs() - transcriptionStart;
  emitTranscriptionProgress(options, {
    stage: 'preprocess',
    progress: 0.2,
    elapsedMs: roundMetric(preprocessElapsedMs),
    remainingMs: estimateRemainingMs(preprocessElapsedMs, 0.2),
    modelId,
    backendId,
    message: `Prepared Whisper mel features for ${modelId}.`,
    metrics: {
      preprocessMs: roundMetric(preprocessMs),
      audioDurationSec: roundMetric(audio.durationSeconds, 4),
    },
  });

  const featureTensor = await maybeCastWhisperFeatureTensor(new loaded.ort.Tensor(
    'float32', paddedFeatures,
    [1, melBins, melInputFrames],
  ), loaded.encoderSession, loaded.ort);

  const ownedEncoderTensors = new Set<OrtTensorLike>();
  try {
    // 2. Run encoder
    const encodeStart = nowMs();
    const encoderRunStart = nowMs();
    const encoderOutputs = await runWhisperEncoderOnFeatures(loaded.encoderSession, featureTensor);
    for (const tensor of Object.values(encoderOutputs)) ownedEncoderTensors.add(tensor);
    const encoderRunEnd = nowMs();
    const encoderOutputTensorStart = nowMs();
    let encoderHiddenStates = await maybeCastEncoderHiddenStates(
      encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>,
      loaded.decoderInitSession ?? loaded.decoderSession!,
      loaded.ort,
    );
    ownedEncoderTensors.add(encoderHiddenStates);
    const encoderOutputCastEnd = nowMs();
    const encoderOutputEnd = nowMs();
    const encodeMs = nowMs() - encodeStart;
    const encoderRunMs = encoderRunEnd - encoderRunStart;
    const encoderOutputMs = encoderOutputEnd - encoderRunEnd;
    const encoderOutputCastMs = encoderOutputCastEnd - encoderOutputTensorStart;
    const encoderOutputLocation = (encoderHiddenStates as OrtTensorLike<Float32Array>).location ?? 'cpu';
    const encoderOutputDtype = (encoderHiddenStates as OrtTensorLike<Float32Array>).type ?? 'float32';

    let encoderBufferRewrapMs = 0;
    let encoderGpuFlushMs = 0;
    if (loaded.encoderBufferRewrap && isGpuBufferTensor(encoderHiddenStates as OrtTensorLike<Float32Array>)) {
      const rewrapStart = nowMs();
      const origTensor = encoderHiddenStates as OrtTensorLike<Float32Array>;
      const gpuBuffer = (origTensor as unknown as { gpuBuffer: unknown }).gpuBuffer;
      if (gpuBuffer && loaded.ort.Tensor.fromGpuBuffer) {
        const rewrapped = loaded.ort.Tensor.fromGpuBuffer(gpuBuffer, {
          dataType: origTensor.type as string,
          dims: [...origTensor.dims] as readonly number[],
          download: origTensor.getData
            ? async () => origTensor.getData!() as Promise<ArrayBufferView>
            : undefined,
          dispose: undefined,
        });
        encoderHiddenStates = rewrapped as unknown as OrtTensorLike<Float32Array>;
        ownedEncoderTensors.add(encoderHiddenStates);
      }
      encoderBufferRewrapMs = nowMs() - rewrapStart;
    }

    if (loaded.encoderGpuFlush && isGpuBufferTensor(encoderHiddenStates as OrtTensorLike<Float32Array>)) {
      const flushStart = nowMs();
      const origTensor = encoderHiddenStates as OrtTensorLike<Float32Array>;
      const gpuBuffer = (origTensor as unknown as { gpuBuffer: unknown }).gpuBuffer;
      const dims = [...origTensor.dims] as readonly number[];
      const dtype = origTensor.type as string;
      const flushedData = origTensor.getData
        ? await origTensor.getData(false)
        : undefined;
      if (gpuBuffer && loaded.ort.Tensor.fromGpuBuffer) {
        encoderHiddenStates = loaded.ort.Tensor.fromGpuBuffer(gpuBuffer, {
          dataType: dtype,
          dims,
          download: flushedData ? async () => flushedData : undefined,
          dispose: undefined,
        }) as unknown as OrtTensorLike<Float32Array>;
        ownedEncoderTensors.add(encoderHiddenStates);
      }
      encoderGpuFlushMs = nowMs() - flushStart;
    }

    let encoderGpuDrainMs = 0;
    if (encoderGpuFlushMs === 0 &&
        loaded.encoderGpuDrain &&
        isGpuBufferTensor(encoderHiddenStates as OrtTensorLike<Float32Array>)) {
      const drainStart = nowMs();
      const origTensor = encoderHiddenStates as OrtTensorLike<Float32Array>;
      const gpuBuffer = (origTensor as unknown as { gpuBuffer: unknown }).gpuBuffer;
      const dims = [...origTensor.dims] as readonly number[];
      const dtype = origTensor.type as string;
      const drainedData = origTensor.getData && gpuBuffer
        ? await origTensor.getData(false)
        : undefined;
      if (gpuBuffer && loaded.ort.Tensor.fromGpuBuffer) {
        encoderHiddenStates = loaded.ort.Tensor.fromGpuBuffer(gpuBuffer, {
          dataType: dtype,
          dims,
          download: drainedData ? async () => drainedData : undefined,
          dispose: undefined,
        }) as unknown as OrtTensorLike<Float32Array>;
        ownedEncoderTensors.add(encoderHiddenStates);
      }
      encoderGpuDrainMs = nowMs() - drainStart;
    }
    const encoderFrameCount = encoderHiddenStates.dims[1] ?? encoderOutputPositions;
    const encodeElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'encode',
      progress: 0.4,
      elapsedMs: roundMetric(encodeElapsedMs),
      remainingMs: estimateRemainingMs(encodeElapsedMs, 0.4),
      modelId,
      backendId,
      message: `Encoded Whisper frames for ${modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        encoderFrameCount,
      },
    });

    // 3. Detect language if auto
    const tokenizer = loaded.tokenizer;
    let language = options.language ?? 'auto';
    let languageDetectionMs = 0;
    if (language === 'auto' && loaded.isSplitGraph && loaded.decoderInitSession) {
      const languageDetectionStart = nowMs();
      language = await detectLanguageFromEncoder(loaded as Required<LoadedExecutorState>, encoderHiddenStates);
      languageDetectionMs = nowMs() - languageDetectionStart;
    }
    if (language === 'auto') {
      language = resolveWhisperLanguageCode(language, config.languages);
    }
    const taskToken = options.task === 'translate' ? '<|translate|>' : '<|transcribe|>';
    const noTimestampsToken = options.noTimestamps ? '<|notimestamps|>' : undefined;

    const promptTokens: number[] = [
      tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
      resolveWhisperLanguageTokenId(tokenizer, language),
      tokenizer.getTokenId(taskToken) ?? 50360,
    ];
    if (noTimestampsToken) {
      const ntId = tokenizer.getTokenId(noTimestampsToken);
      if (ntId !== undefined) promptTokens.push(ntId);
    }
    if (options.extraPromptTokens && options.extraPromptTokens.length > 0) {
      promptTokens.push(...options.extraPromptTokens);
    }

    // 4. Run 4-graph decode loop
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
    const noSpeechTokenId = loaded.generationConfig.noSpeechTokenId
      ?? tokenizer.getTokenId('<|nospeech|>')
      ?? 50362;
    const maxNewTokens = options.maxNewTokens ?? config.maxTargetPositions ?? 448;

    const timestampBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;
    const splitTimestampProcessor = new WhisperTimestampLogitProcessor({
      eosTokenId: eosId,
      noTimestampsTokenId: loaded.generationConfig.noTimestampsTokenId ?? tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
      timestampBegin,
      suppressTokens: loaded.generationConfig.suppressTokens ?? [],
      beginSuppressTokens: loaded.generationConfig.beginSuppressTokens ?? [],
    });

    const { result, timing } = await runSplitGraphDecodeWithTiming({
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
      applyTimestampProcessor: (logits, genTokens, beginIdx) => {
        splitTimestampProcessor.process(logits, genTokens, beginIdx);
      },
    });

    const {
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
    } = timing;

    const decodeElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'decode',
      progress: clampProgress(0.85),
      elapsedMs: roundMetric(decodeElapsedMs),
      remainingMs: estimateRemainingMs(decodeElapsedMs, 0.85),
      completedUnits: result.tokens.length,
      totalUnits: maxNewTokens,
      modelId,
      backendId,
      message: `Decoded ${result.tokens.length} Whisper tokens for ${modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
        decoderInitMs: roundMetric(decoderInitMs),
        decoderInitInputMs: roundMetric(decoderInitInputMs),
        decoderInitRunMs: roundMetric(decoderInitRunMs),
        decoderInitOutputMs: roundMetric(decoderInitOutputMs),
        decoderStepMs: roundMetric(decoderStepMs),
        decoderStepFeedBuildMs: roundMetric(decoderStepFeedBuildMs),
        decoderStepTensorCloneMs: roundMetric(decoderStepTensorCloneMs),
        decoderStepRunMs: roundMetric(decoderStepRunMs),
        decoderStepOutputMs: roundMetric(decoderStepOutputMs),
        decoderStepAvgMs: decoderStepAvgMs !== undefined ? roundMetric(decoderStepAvgMs) : undefined,
        decoderStepP50Ms: decoderStepP50Ms !== undefined ? roundMetric(decoderStepP50Ms) : undefined,
        decoderStepP95Ms: decoderStepP95Ms !== undefined ? roundMetric(decoderStepP95Ms) : undefined,
        decoderStepMaxMs: decoderStepMaxMs !== undefined ? roundMetric(decoderStepMaxMs) : undefined,
        decoderLogitProcessMs: roundMetric(decoderLogitProcessMs),
        decoderStepCount,
        decoderGpuTensorInputs,
        decoderCpuTensorInputs,
        decoderGpuTensorOutputs,
        decoderCpuTensorOutputs,
        decoderGpuTensorDownloads,
        decoderKvCacheLocation,
        encoderFrameCount,
        decodeIterations: result.tokens.length,
      },
    });

    // 5. Build token details
    const tokenizeStart = nowMs();
    const generatedTokens = [...promptTokens, ...result.tokens];
    const tokenDetails: WhisperNativeToken[] = [];
    for (let i = promptTokens.length; i < generatedTokens.length; i++) {
      const tokenId = generatedTokens[i]!;
      tokenDetails.push({
        index: i - promptTokens.length,
        id: tokenId,
        text: formatWhisperTokenText(tokenizer, tokenId),
        special: tokenizer.isSpecialTokenId(tokenId),
      });
    }

    // 6. Build segments
    const segments = buildWhisperSegments(
      tokenDetails,
      tokenizer,
      options.noTimestamps,
      audio.durationSeconds,
    );

    // 7. Word timestamps via splitgraph alignment
    const wantsWordTimestamps = shouldReturnWordTimestamps(options);
    const requestedWordTimestampSource = resolveWhisperWordTimestampSource(
      options.wordTimestampSource,
      Boolean(loaded.alignmentReference),
    );
    let alignmentEncoderHiddenStates = encoderHiddenStates;
    let alignmentRuntime: WhisperAlignmentRuntime | undefined;
    let alignmentReferenceRequiredButUnavailable = false;
    let wordAlignmentReferenceMs = 0;
    let alignmentReferenceStartMs: number | undefined;
    let wordAlignmentSource: 'fast' | 'reference' | 'interpolation' = 'fast';
    if (wantsWordTimestamps && requestedWordTimestampSource === 'reference') {
      alignmentReferenceStartMs = nowMs();
      try {
        const reference = await ctx.runAlignmentReferenceEncoder(
          loaded,
          paddedFeatures,
          melBins,
          melInputFrames,
          ownedEncoderTensors,
        );
        alignmentEncoderHiddenStates = reference.encoderHiddenStates;
        alignmentRuntime = reference.alignmentRuntime;
        wordAlignmentSource = 'reference';
      } catch (error) {
        wordAlignmentReferenceMs = alignmentReferenceStartMs === undefined
          ? 0
          : nowMs() - alignmentReferenceStartMs;
        alignmentReferenceRequiredButUnavailable = options.wordTimestampSource === 'reference';
        warnings.push({
          code: 'whisper.alignment-reference-unavailable',
          message:
            `Whisper alignment-reference inference was unavailable; ${alignmentReferenceRequiredButUnavailable ? 'using timestamp-token interpolation' : 'falling back to the primary alignment path'} (${formatWhisperSessionError(error)}).`,
          recoverable: true,
        });
      }
    }
    const alignedWords = wantsWordTimestamps
      ? await computeAttentionWordTimestampsSplitGraph({
          loaded,
          ensureDecoderAlignSession: ctx.ensureDecoderAlignSession,
          encoderHiddenStates: alignmentEncoderHiddenStates,
          tokenizer,
          tokenDetails,
          allTokens: generatedTokens,
          promptLen: promptTokens.length,
          language,
          options,
          audioDurationSeconds: audio.durationSeconds,
          warnings,
          alignmentRuntime,
          alignmentReferenceRequiredButUnavailable,
          ownedEncoderTensors,
        })
      : [];
    if (
      wantsWordTimestamps &&
      wordAlignmentSource === 'reference' &&
      alignmentReferenceStartMs !== undefined
    ) {
      wordAlignmentReferenceMs = nowMs() - alignmentReferenceStartMs;
    }
    if (wantsWordTimestamps && wordAlignmentSource === 'fast' &&
        (!loaded.decoderAlignUrl ||
          !hasVerifiedWhisperDecoderAlignment(loaded.decoderAlignCausalSelfAttention))) {
      wordAlignmentSource = 'interpolation';
    }
    const words = wantsWordTimestamps
      ? await finalizeWordTimestamps(
          alignedWords,
          tokenDetails,
          tokenizer,
          language,
          options,
          audio,
          warnings,
        )
      : [];

    const utteranceText = segments.map((s) => s.text).join(' ').trim();
    const tokenizeMs = nowMs() - tokenizeStart;
    const totalMs = roundMetric(nowMs() - transcriptionStart);
    const rtf = audio.durationSeconds > 0 ? totalMs / (audio.durationSeconds * 1000) : 0;
    const rtfx = audio.durationSeconds > 0 ? audio.durationSeconds / (totalMs / 1000) : undefined;
    const metrics: TranscriptMetrics = {
      preprocessMs: roundMetric(preprocessMs),
      encodeMs: roundMetric(encodeMs),
      decodeMs: roundMetric(decodeMs),
      tokenizeMs: roundMetric(tokenizeMs),
      languageDetectionMs: roundMetric(languageDetectionMs),
      decoderInitMs: roundMetric(decoderInitMs),
      decoderInitInputMs: roundMetric(decoderInitInputMs),
      decoderInitRunMs: roundMetric(decoderInitRunMs),
      decoderInitOutputMs: roundMetric(decoderInitOutputMs),
      decoderStepMs: roundMetric(decoderStepMs),
      decoderStepFeedBuildMs: roundMetric(decoderStepFeedBuildMs),
      decoderStepTensorCloneMs: roundMetric(decoderStepTensorCloneMs),
      decoderStepRunMs: roundMetric(decoderStepRunMs),
      decoderStepOutputMs: roundMetric(decoderStepOutputMs),
      decoderStepAvgMs: decoderStepAvgMs !== undefined ? roundMetric(decoderStepAvgMs) : undefined,
      decoderStepP50Ms: decoderStepP50Ms !== undefined ? roundMetric(decoderStepP50Ms) : undefined,
      decoderStepP95Ms: decoderStepP95Ms !== undefined ? roundMetric(decoderStepP95Ms) : undefined,
      decoderStepMaxMs: decoderStepMaxMs !== undefined ? roundMetric(decoderStepMaxMs) : undefined,
      decoderLogitProcessMs: roundMetric(decoderLogitProcessMs),
      decoderStepCount,
      decoderGpuTensorInputs,
      decoderCpuTensorInputs,
      decoderGpuTensorOutputs,
      decoderCpuTensorOutputs,
      decoderGpuTensorDownloads,
      decoderKvCacheLocation,
      decoderInitTensorCreateMs: roundMetric(decoderInitTensorCreateMs),
      decoderInitLogitReadMs: roundMetric(decoderInitLogitReadMs),
      decoderInitKvExtractMs: roundMetric(decoderInitKvExtractMs),
      decoderStepTensorCreateMs: roundMetric(decoderStepTensorCreateMs),
      decoderStepLogitReadMs: roundMetric(decoderStepLogitReadMs),
      decoderStepKvMergeMs: roundMetric(decoderStepKvMergeMs),
      decoderEncoderKvTensorReuses,
      decoderEncoderKvTensorCreates,
      sessionCreateMs: roundMetric(loaded.sessionCreateMs ?? 0),
      encoderRunMs: roundMetric(encoderRunMs),
      encoderOutputMs: roundMetric(encoderOutputMs),
      encoderOutputCastMs: roundMetric(encoderOutputCastMs),
      encoderOutputLocation,
      encoderOutputDtype,
      encoderBufferRewrapMs: roundMetric(encoderBufferRewrapMs),
      encoderGpuFlushMs: roundMetric(encoderGpuFlushMs),
      encoderGpuDrainMs: roundMetric(encoderGpuDrainMs),
      encoderTotalMs: roundMetric(encoderRunMs + encoderGpuDrainMs),
      wordAlignmentReferenceMs: wantsWordTimestamps && wordAlignmentReferenceMs > 0
        ? roundMetric(wordAlignmentReferenceMs)
        : undefined,
      wordAlignmentSource: wantsWordTimestamps ? wordAlignmentSource : undefined,
      totalMs,
      wallMs: totalMs,
      audioDurationSec: roundMetric(audio.durationSeconds, 4),
      rtf: roundMetric(rtf, 4),
      rtfx: rtfx !== undefined ? roundMetric(rtfx, 4) : undefined,
      preprocessorBackend: 'js-whisper-mel',
      requestedPreprocessorBackend: 'js-whisper-mel',
      encoderFrameCount,
      decodeIterations: result.tokens.length,
      emittedTokenCount: tokenDetails.filter((token) => !token.special).length,
      emittedWordCount: words?.length,
    };
    const postprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'postprocess',
      progress: 0.95,
      elapsedMs: roundMetric(postprocessElapsedMs),
      remainingMs: estimateRemainingMs(postprocessElapsedMs, 0.95),
      modelId,
      backendId,
      message: `Built Whisper transcript details for ${modelId}.`,
      metrics,
    });
    emitTranscriptionProgress(options, {
      stage: 'complete',
      progress: 1,
      elapsedMs: metrics.totalMs,
      modelId,
      backendId,
      message: `Finished transcription for ${modelId}.`,
      metrics,
    });

    return {
      utteranceText, isFinal: true, language, segments,
      ...(words && words.length > 0 ? { words } : {}),
      tokens: options.returnSpecialTokens
        ? tokenDetails
        : tokenDetails.filter((t) => !t.special),
      ...(result.tokenTraces ? { tokenTraces: result.tokenTraces } : {}),
      metrics,
      warnings,
    };
  } finally {
    disposeOwnedWhisperTensors(ownedEncoderTensors);
  }
}
