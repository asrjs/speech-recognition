import type {
  AudioBufferLike,
  TranscriptWarning,
} from '../../types/index.js';
import type { TokenQualityTrace } from '../../quality/types.js';
import { computeWhisperDtwTokenTimestamps } from './attention-alignment.js';
import type {
  LoadedExecutorState,
  SplitGraphAlignmentOptions,
  SplitGraphTimestampSpanAlignmentOptions,
  WhisperAlignmentRuntime,
} from './executor-types.js';
import {
  maybeCastEncoderHiddenStates,
  maybeCastWhisperFeatureTensor,
  runWhisperEncoderOnFeatures,
} from './executor-utils.js';
import type { OrtSessionLike, OrtTensorLike } from './ort.js';
import type { WhisperTokenizer } from './tokenizer.js';
import type {
  WhisperNativeSegment,
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperNativeWord,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';
import {
  alignmentWindowForWhisperWords,
  clipShortWhisperWordDurations,
  coalesceWhisperWordTimestamps,
  constrainWhisperWordTimestampsToDuration,
  forcedAlignmentLooksAnchored,
  refineWhisperWordsWithForcedAlignment,
  splitWhisperWordsByPause,
} from './word-timestamps.js';

import {
  computeAttentionWordTimestamps,
  computeAttentionWordTimestampsSplitGraph,
  materializeCpuEncoderHiddenStates,
  runForcedAlignment,
  runForcedAlignmentSplitGraph,
} from './executor-alignment-run.js';

export {
  computeAttentionWordTimestamps,
  computeAttentionWordTimestampsSplitGraph,
  materializeCpuEncoderHiddenStates,
  runForcedAlignment,
  runForcedAlignmentSplitGraph,
};

/**
 * Alignment is verified only when the manifest explicitly proves that the
 * decoder_align graph uses causal self-attention. Missing metadata is a
 * legacy/unknown artifact, not evidence that the graph is safe to run.
 */
export function hasVerifiedWhisperDecoderAlignment(
  causalSelfAttention: boolean | undefined,
): boolean {
  return causalSelfAttention === true;
}

export function resolveWhisperWordTimestampSource(
  requested: WhisperSeq2SeqTranscriptionOptions['wordTimestampSource'] | undefined,
  hasReference: boolean,
): 'fast' | 'reference' {
  if (requested === 'fast') return 'fast';
  if (requested === 'reference') return 'reference';
  return hasReference ? 'reference' : 'fast';
}

export function extractCrossAttentions(
  outputs: Record<string, unknown>,
): OrtTensorLike<Float32Array>[] {
  const entries: { layer: number; tensor: OrtTensorLike<Float32Array> }[] = [];
  for (const [key, value] of Object.entries(outputs)) {
    const match = key.match(/^cross_attentions\.(\d+)$/);
    if (match) {
      entries.push({
        layer: parseInt(match[1]!, 10),
        tensor: value as OrtTensorLike<Float32Array>,
      });
    }
  }
  entries.sort((a, b) => a.layer - b.layer);
  return entries.map((e) => e.tensor);
}

export function collectSplitGraphTextTokenRows(
  tokenIds: readonly number[],
  promptLen: number,
  isTextToken: (tokenId: number) => boolean,
): { readonly tokenIds: number[]; readonly rowIndices: number[] } {
  const textTokenIds: number[] = [];
  const rowIndices: number[] = [];
  for (let index = promptLen; index < tokenIds.length; index++) {
    const tokenId = tokenIds[index]!;
    if (!isTextToken(tokenId)) continue;
    textTokenIds.push(tokenId);
    rowIndices.push(index);
  }
  return { tokenIds: textTokenIds, rowIndices };
}

export function extractSplitGraphAlignmentRows(
  alignmentData: Float32Array,
  rowIndices: readonly number[],
  frameCount: number,
): Float32Array {
  const textValues = new Float32Array(rowIndices.length * frameCount);
  for (let i = 0; i < rowIndices.length; i++) {
    const srcOffset = rowIndices[i]! * frameCount;
    textValues.set(alignmentData.subarray(srcOffset, srcOffset + frameCount), i * frameCount);
  }
  return textValues;
}

/** Extract token rows from one head of a flattened [B, N, T, S] output. */
export function extractSplitGraphAlignmentHeadRows(
  alignmentData: Float32Array,
  rowIndices: readonly number[],
  frameCount: number,
  totalTokens: number,
  headIndex: number,
): Float32Array {
  const headStride = totalTokens * frameCount;
  const headOffset = headIndex * headStride;
  const textValues = new Float32Array(rowIndices.length * frameCount);
  for (let i = 0; i < rowIndices.length; i++) {
    const srcOffset = headOffset + rowIndices[i]! * frameCount;
    const source = alignmentData.subarray(srcOffset, srcOffset + frameCount);
    if (source.length !== frameCount) {
      throw new Error(
        `Whisper alignment head ${headIndex} row ${rowIndices[i]} is truncated; ` +
        `expected ${frameCount} frames.`,
      );
    }
    textValues.set(source, i * frameCount);
  }
  return textValues;
}

export function processSplitGraphAlignment(
  options: SplitGraphAlignmentOptions,
): readonly number[] {
  const {
    alignmentData, promptLen, textTokenCount, frameCount,
    medianFilterWidth, timePrecisionSeconds, textTokenRowIndices, cropFrameCount,
    alignmentHeadCount: requestedHeadCount, alignmentValuesAreLogits,
  } = options;

  if (textTokenCount === 0 && (textTokenRowIndices?.length ?? 0) === 0) return [0];
  if (frameCount === 0) {
    const count = textTokenRowIndices?.length ?? textTokenCount;
    return Array.from({ length: count + 1 }, () => 0);
  }

  const resolvedTextTokenCount = textTokenRowIndices?.length ?? textTokenCount;
  const cropFrames = Math.max(1, Math.min(frameCount, cropFrameCount ?? frameCount));

  const headCount = Math.max(1, Math.floor(requestedHeadCount ?? 1));
  const resolvedRowIndices = textTokenRowIndices && textTokenRowIndices.length > 0
    ? textTokenRowIndices
    : Array.from({ length: textTokenCount }, (_unused, index) => promptLen + index);
  const normalizationTokenCount = options.totalTokens;
  const requiredValues = headCount * options.totalTokens * frameCount;
  if (alignmentData.length < requiredValues) {
    throw new Error(
      `Whisper alignment output has ${alignmentData.length} values; expected at least ` +
      `${requiredValues} for ${headCount} heads.`,
    );
  }

  const attentionHeads = Array.from({ length: headCount }, (_unused, headIndex) => ({
    values: headCount > 1
      ? alignmentData.subarray(
          headIndex * options.totalTokens * frameCount,
          (headIndex + 1) * options.totalTokens * frameCount,
        )
      : alignmentData,
    tokenCount: options.totalTokens,
    frameCount,
    ...(alignmentValuesAreLogits ? { valuesAreLogits: true } : {}),
  }));

  return computeWhisperDtwTokenTimestamps({
    attentionHeads,
    tokenCount: resolvedTextTokenCount,
    frameCount: cropFrames,
    tokenRowIndices: resolvedRowIndices,
    normalizationTokenCount,
    medianFilterWidth,
    timePrecisionSeconds,
  });
}

export function extractSplitGraphAlignmentWindow(
  alignmentData: Float32Array,
  rowIndices: readonly number[],
  fullFrameCount: number,
  startFrame: number,
  windowFrames: number,
): Float32Array {
  const output = new Float32Array(rowIndices.length * windowFrames);
  for (let i = 0; i < rowIndices.length; i++) {
    const srcOffset = rowIndices[i]! * fullFrameCount + startFrame;
    output.set(alignmentData.subarray(srcOffset, srcOffset + windowFrames), i * windowFrames);
  }
  return output;
}

/** Extract a frame window from one head of a flattened [B, N, T, S] output. */
export function extractSplitGraphAlignmentHeadWindow(
  alignmentData: Float32Array,
  rowIndices: readonly number[],
  totalTokens: number,
  fullFrameCount: number,
  headIndex: number,
  startFrame: number,
  windowFrames: number,
): Float32Array {
  const headOffset = headIndex * totalTokens * fullFrameCount;
  const output = new Float32Array(rowIndices.length * windowFrames);
  for (let i = 0; i < rowIndices.length; i++) {
    const srcOffset = headOffset + rowIndices[i]! * fullFrameCount + startFrame;
    const source = alignmentData.subarray(srcOffset, srcOffset + windowFrames);
    if (source.length !== windowFrames) {
      throw new Error(
        `Whisper alignment head ${headIndex} row ${rowIndices[i]} window is truncated; ` +
        `expected ${windowFrames} frames.`,
      );
    }
    output.set(source, i * windowFrames);
  }
  return output;
}

/**
 * Whisper-style word alignment: DTW each timestamp-token segment against only
 * the encoder frames in that segment, then concatenate jump times.
 * Returns undefined when the sequence has no timestamp tokens.
 */
export function processSplitGraphAlignmentByTimestampSpans(
  options: SplitGraphTimestampSpanAlignmentOptions,
): readonly number[] | undefined {
  const hop = options.timePrecisionSeconds ?? 0.02;
  const cropFrames = Math.max(1, Math.min(options.frameCount, options.cropFrameCount ?? options.frameCount));
  const audioEnd = cropFrames * hop;

  const spans: { start: number; end: number; rowIndices: number[] }[] = [];
  let spanStart: number | null = null;
  let spanRows: number[] = [];
  for (let index = options.promptLen; index < options.tokenIds.length; index++) {
    const tokenId = options.tokenIds[index]!;
    if (options.isTimestampToken(tokenId)) {
      const time = options.timestampTokenToSeconds(tokenId);
      if (spanStart !== null && spanRows.length > 0) {
        spans.push({ start: spanStart, end: Math.max(time, spanStart + hop), rowIndices: spanRows });
      }
      spanStart = time;
      spanRows = [];
      continue;
    }
    if (spanStart !== null && options.isTextToken(tokenId)) {
      spanRows.push(index);
    }
  }
  if (spanStart !== null && spanRows.length > 0) {
    spans.push({ start: spanStart, end: Math.max(audioEnd, spanStart + hop), rowIndices: spanRows });
  }
  if (spans.length === 0) return undefined;

  const jumpTimes: number[] = [];
  for (const span of spans) {
    const startFrame = Math.max(0, Math.min(cropFrames - 1, Math.round(span.start / hop)));
    const endFrame = Math.max(startFrame + 1, Math.min(cropFrames, Math.round(span.end / hop)));
    const windowFrames = endFrame - startFrame;
    const headCount = Math.max(1, Math.floor(options.alignmentHeadCount ?? 1));
    const attentionHeads = Array.from({ length: headCount }, (_unused, headIndex) => ({
      values: headCount > 1
        ? extractSplitGraphAlignmentHeadWindow(
            options.alignmentData,
            span.rowIndices,
            options.tokenIds.length,
            options.frameCount,
            headIndex,
            startFrame,
            windowFrames,
          )
        : extractSplitGraphAlignmentWindow(
            options.alignmentData,
            span.rowIndices,
            options.frameCount,
            startFrame,
            windowFrames,
          ),
      tokenCount: span.rowIndices.length,
      frameCount: windowFrames,
      ...(options.alignmentValuesAreLogits ? { valuesAreLogits: true } : {}),
    }));
    const relative = computeWhisperDtwTokenTimestamps({
      attentionHeads,
      tokenCount: span.rowIndices.length,
      frameCount: windowFrames,
      medianFilterWidth: options.medianFilterWidth,
      timePrecisionSeconds: hop,
    });
    const spanEnd = Math.min(span.end, audioEnd);
    for (let i = 0; i < span.rowIndices.length; i++) {
      const time = Math.min(spanEnd, Math.max(span.start, span.start + (relative[i] ?? 0)));
      jumpTimes.push(jumpTimes.length === 0 ? time : Math.max(jumpTimes[jumpTimes.length - 1]!, time));
    }
  }

  const lastSpan = spans[spans.length - 1]!;
  jumpTimes.push(Math.max(jumpTimes[jumpTimes.length - 1] ?? 0, Math.min(lastSpan.end, audioEnd)));
  return jumpTimes;
}

export function sliceAudioBufferLike(
  audio: AudioBufferLike,
  startSeconds: number,
  endSeconds: number,
): AudioBufferLike {
  const startFrame = Math.max(0, Math.floor(startSeconds * audio.sampleRate));
  const endFrame = Math.min(
    audio.numberOfFrames,
    Math.max(startFrame + 1, Math.ceil(endSeconds * audio.sampleRate)),
  );
  const frameCount = Math.max(1, endFrame - startFrame);
  const channels = audio.channels?.map((channel) => channel.subarray(startFrame, startFrame + frameCount));
  const data = channels?.[0] ?? (audio.data instanceof Float32Array
    ? audio.data.subarray(startFrame, startFrame + frameCount)
    : undefined);
  return {
    sampleRate: audio.sampleRate,
    numberOfChannels: audio.numberOfChannels,
    numberOfFrames: frameCount,
    durationSeconds: frameCount / audio.sampleRate,
    ...(channels ? { channels } : {}),
    ...(data ? { data, format: audio.format ?? 'f32-planar' } : {}),
  };
}

export function buildWordsFromDtwTimestamps(
  tokenizer: WhisperTokenizer,
  textTokenIds: number[],
  dtwTimestamps: readonly number[],
  tokenLogprobs?: Float32Array,
): WhisperNativeTranscript['words'] {
  // DTW gives tokenCount+1 timestamps — start of each text token + end of last
  const words: WhisperNativeWord[] = [];
  const text = tokenizer.decode(textTokenIds, { skipSpecialTokens: true });
  const wordTexts = text.split(/\s+/).filter((w) => w.length > 0);
  if (wordTexts.length === 0) return undefined;

  // Approximate word boundaries from token-to-word mapping using tokenizer encode
  const allWordTokenIds: number[][] = [];
  for (const w of wordTexts) {
    const ids = tokenizer.encode(w);
    if (ids.length > 0) allWordTokenIds.push(ids);
  }

  let tokenOffset = 0;
  for (let wi = 0; wi < allWordTokenIds.length; wi++) {
    const wIds = allWordTokenIds[wi]!;
    if (wIds.length === 0) continue;
    const wordStartIdx = tokenOffset;
    const wordEndIdx = tokenOffset + wIds.length - 1;
    const startTime = dtwTimestamps[wordStartIdx] ?? 0;
    const endTime = dtwTimestamps[wordEndIdx + 1] ?? dtwTimestamps[wordStartIdx] ?? 0;

    let confidence = -1;
    if (tokenLogprobs && wIds.length > 0) {
      let probSum = 0;
      for (let t = wordStartIdx; t <= wordEndIdx; t++) {
        probSum += Math.exp(tokenLogprobs[t] ?? -Infinity);
      }
      confidence = probSum / wIds.length;
    }

    words.push({
      index: wi,
      text: wordTexts[wi]!,
      startTime,
      endTime,
      confidence,
      tokenIds: wIds,
    });
    tokenOffset += wIds.length;
  }
  return words.length > 0 ? words : undefined;
}

export function buildWhisperSegments(
  tokens: WhisperNativeToken[],
  tokenizer: WhisperTokenizer,
  noTimestamps?: boolean,
  durationSeconds: number = 30,
): WhisperNativeSegment[] {
  const clipEnd = Number.isFinite(durationSeconds) && durationSeconds >= 0
    ? durationSeconds
    : 30;
  if (noTimestamps) {
    const text = tokenizer.decode(
      tokens.map((t) => t.id ?? 0),
      { skipSpecialTokens: true },
    );
    if (!text.trim()) return [];
    return [
      {
        index: 0,
        text,
        startTime: 0,
        endTime: clipEnd,
        confidence: tokens.length > 0 ? (tokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / tokens.length) : 0,
      },
    ];
  }

  // With timestamps: split on timestamp tokens
  const segments: WhisperNativeSegment[] = [];
  let currentTokens: WhisperNativeToken[] = [];
  let segmentStart = 0;

  for (const token of tokens) {
    if (tokenizer.isTimestampTokenId(token.id ?? 0)) {
      const ts = tokenizer.timestampTokenIdToSeconds(token.id ?? 0);
      if (ts !== undefined) {
        if (currentTokens.length > 0) {
          const text = tokenizer.decode(
            currentTokens.map((t) => t.id ?? 0),
            { skipSpecialTokens: true },
          );
          if (text.trim()) {
            const avgConf =
              currentTokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / currentTokens.length;
            segments.push({
              index: segments.length,
              text,
              startTime: segmentStart,
              endTime: ts,
              confidence: avgConf,
            });
          }
          currentTokens = [];
        }
        segmentStart = ts;
      }
    } else {
      currentTokens.push(token);
    }
  }

  // Remaining tokens
  if (currentTokens.length > 0) {
    const text = tokenizer.decode(
      currentTokens.map((t) => t.id ?? 0),
      { skipSpecialTokens: true },
    );
    if (text.trim()) {
      const avgConf =
        currentTokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / currentTokens.length;
      segments.push({
        index: segments.length,
        text,
        startTime: segmentStart,
        endTime: clipEnd,
        confidence: avgConf,
      });
    }
  }

  return segments.flatMap((segment, index) => {
    const startTime = Math.min(clipEnd, Math.max(0, segment.startTime));
    const endTime = Math.min(clipEnd, Math.max(startTime, segment.endTime));
    if (endTime <= startTime) return [];
    return [{ ...segment, index, startTime, endTime }];
  });
}

export function shouldReturnWordTimestamps(options: WhisperSeq2SeqTranscriptionOptions): boolean {
  return options.returnWords === true || options.returnTimestamps === 'word' || options.detail === 'words' || options.detail === 'detailed';
}

export function resolveWordTimestamps(
  alignedWords: readonly WhisperNativeWord[] | undefined,
  tokens: readonly WhisperNativeToken[],
  tokenizer: WhisperTokenizer,
  language: string,
): WhisperNativeWord[] {
  return coalesceWhisperWordTimestamps(alignedWords, tokens, {
    timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
    timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
    language,
    preserveLongDurations: alignedWords !== undefined && alignedWords.length > 0,
  });
}

export async function finalizeWordTimestamps(
  alignedWords: readonly WhisperNativeWord[] | undefined,
  tokens: readonly WhisperNativeToken[],
  tokenizer: WhisperTokenizer,
  language: string,
  options: WhisperSeq2SeqTranscriptionOptions,
  audio: AudioBufferLike,
  warnings: TranscriptWarning[],
): Promise<WhisperNativeWord[]> {
  const words = resolveWordTimestamps(alignedWords, tokens, tokenizer, language);
  if (!options.wordAligner || words.length === 0) {
    return constrainWhisperWordTimestampsToDuration(
      clipShortWhisperWordDurations(words),
      audio.durationSeconds,
    );
  }

  const groups = splitWhisperWordsByPause(words);
  const refined: WhisperNativeWord[] = [];
  let failedGroups = 0;
  for (const group of groups) {
    try {
      const window = alignmentWindowForWhisperWords(group, audio.durationSeconds);
      const sliced = sliceAudioBufferLike(audio, window.startSeconds, window.endSeconds);
      const aligned = await options.wordAligner.align({
        transcript: group.map((word) => word.text).join(' '),
        audio: sliced,
        durationSeconds: sliced.durationSeconds,
        language,
      });
      if (aligned.length === 0) {
        failedGroups += 1;
        refined.push(...clipShortWhisperWordDurations(group));
        continue;
      }
      const shifted = aligned.map((word) => ({
        ...word,
        startTime: word.startTime + window.startSeconds,
        endTime: word.endTime + window.startSeconds,
      }));
      if (!forcedAlignmentLooksAnchored(group, shifted)) {
        failedGroups += 1;
        refined.push(...clipShortWhisperWordDurations(group));
        continue;
      }
      refined.push(...refineWhisperWordsWithForcedAlignment(group, shifted));
    } catch {
      failedGroups += 1;
      refined.push(...clipShortWhisperWordDurations(group));
    }
  }
  if (failedGroups > 0) {
    warnings.push({
      code: 'whisper.word-alignment-partial',
      message: `Wav2Vec2 word alignment failed for ${failedGroups} of ${groups.length} pause group(s); DTW timestamps were kept for those groups.`,
      recoverable: true,
    });
  }
  return constrainWhisperWordTimestampsToDuration(
    clipShortWhisperWordDurations(refined),
    audio.durationSeconds,
  );
}

export async function executeAlignmentReferenceEncoder(
  loaded: LoadedExecutorState,
  encoderSession: OrtSessionLike,
  alignmentRuntime: WhisperAlignmentRuntime,
  paddedFeatures: Float32Array,
  melBins: number,
  melInputFrames: number,
  ownedEncoderTensors: Set<OrtTensorLike>,
): Promise<{
  readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
  readonly alignmentRuntime: WhisperAlignmentRuntime;
}> {
  const featureTensor = await maybeCastWhisperFeatureTensor(
    new loaded.ort.Tensor('float32', paddedFeatures, [1, melBins, melInputFrames]),
    encoderSession,
    loaded.ort,
  );
  const encoderOutputs = await runWhisperEncoderOnFeatures(encoderSession, featureTensor);
  for (const tensor of Object.values(encoderOutputs)) ownedEncoderTensors.add(tensor);
  const outputKey = Object.keys(encoderOutputs)[0];
  if (!outputKey) {
    throw new Error('Whisper alignment-reference encoder returned no output tensor.');
  }
  const encoderHiddenStates = await maybeCastEncoderHiddenStates(
    encoderOutputs[outputKey] as OrtTensorLike<Float32Array>,
    alignmentRuntime.decoderAlignSession,
    loaded.ort,
  );
  ownedEncoderTensors.add(encoderHiddenStates);
  return { encoderHiddenStates, alignmentRuntime };
}

export async function finalizeMergedWhisperTranscript({
  loaded,
  configLanguages,
  encoderHiddenStates,
  tokenizer,
  tokenDetails,
  options,
  audio,
  language,
  warnings,
  ownedEncoderTensors,
  selectedTokenTraces,
}: {
  readonly loaded: LoadedExecutorState;
  readonly configLanguages?: readonly string[];
  readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
  readonly tokenizer: WhisperTokenizer;
  readonly tokenDetails: WhisperNativeToken[];
  readonly options: WhisperSeq2SeqTranscriptionOptions;
  readonly audio: AudioBufferLike;
  readonly language?: string;
  readonly warnings: TranscriptWarning[];
  readonly ownedEncoderTensors?: Set<OrtTensorLike>;
  readonly selectedTokenTraces?: readonly TokenQualityTrace[];
}): Promise<WhisperNativeTranscript> {
  const resolvedLanguage = language ?? 'auto';
  const segments = buildWhisperSegments(
    tokenDetails,
    tokenizer,
    options.noTimestamps,
    audio.durationSeconds,
  );
  const alignedWords = shouldReturnWordTimestamps(options)
    ? await computeAttentionWordTimestamps({
        loaded,
        configLanguages: configLanguages ?? [],
        encoderHiddenStates,
        tokenizer,
        tokenDetails,
        segments,
        language: resolvedLanguage,
        options,
        audioDurationSeconds: audio.durationSeconds,
        warnings,
        ownedEncoderTensors,
      })
    : [];
  const words = shouldReturnWordTimestamps(options)
    ? await finalizeWordTimestamps(
        alignedWords,
        tokenDetails,
        tokenizer,
        resolvedLanguage,
        options,
        audio,
        warnings,
      )
    : [];
  const utteranceText = segments.map((s) => s.text).join(' ').trim();

  return {
    utteranceText,
    isFinal: true,
    language: resolvedLanguage,
    segments,
    ...(words && words.length > 0 ? { words } : {}),
    tokens: options.returnSpecialTokens
      ? tokenDetails
      : tokenDetails.filter((t) => !t.special),
    ...(selectedTokenTraces && selectedTokenTraces.length > 0 ? { tokenTraces: selectedTokenTraces } : {}),
    warnings,
  };
}


