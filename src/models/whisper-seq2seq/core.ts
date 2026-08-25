/**
 * Whisper Core — pure decode logic (ONNX-agnostic).
 *
 * This module contains the vanilla Whisper inference loop, independent of
 * ONNX Runtime, the asrjs model-family system, or any audio processing.
 *
 * Decode strategies:
 *   - Greedy: argmax per step (fastest, lowest quality)
 *   - Beam: top-k beam search (WhisperX/faster-whisper quality)
 *
 * A "session" is any object that can run decoder_init and decoder_step.
 */

import { argmax, tokenQualityFromLogits } from '../../inference/index.js';
import type { TokenQualityTrace } from '../../quality/types.js';
import type { WhisperBeamState } from './beam-search.js';

// ---------------------------------------------------------------------------
// Session interface
// ---------------------------------------------------------------------------

export interface WhisperCoreSession {
  runInit(
    promptTokens: readonly number[],
    encoderOutput: Float32Array,
    encoderDims: readonly number[],
  ): Promise<WhisperInitResult>;

  /**
   * Implementations must treat pastKv as read-only. Backend adapters may
   * clone/repack it while constructing their next decoder input.
   */
  runStep(
    tokenId: number,
    pastKv: WhisperKvCache,
  ): Promise<WhisperStepResult>;

  /** The same read-only cache contract applies to every batched input. */
  runStepBatch?(
    tokenIds: readonly number[],
    pastKvs: readonly WhisperKvCache[],
  ): Promise<readonly WhisperStepResult[] | undefined>;
}

export interface WhisperKvCacheEntry {
  readonly data: ArrayBufferView;
  readonly dims?: readonly number[];
  readonly type?: string;
}

export type WhisperKvCacheValue = ArrayBufferView | WhisperKvCacheEntry;
export type WhisperKvCache = Record<string, WhisperKvCacheValue>;

export interface WhisperInitResult {
  readonly logits: Float32Array;
  readonly vocabSize: number;
  readonly presentKv: WhisperKvCache;
}

export interface WhisperStepResult {
  readonly logits: Float32Array;
  readonly vocabSize: number;
  readonly presentKv: WhisperKvCache;
}

// ---------------------------------------------------------------------------
// Logit processor
// ---------------------------------------------------------------------------

export type WhisperLogitProcessor = (
  logits: Float32Array,
  generatedTokens: readonly number[],
  beginIndex: number,
) => void;

// ---------------------------------------------------------------------------
// Decode options
// ---------------------------------------------------------------------------

export interface WhisperDecodeOptions {
  readonly promptTokens: readonly number[];
  readonly encoderOutput: Float32Array;
  readonly encoderDims: readonly number[];
  readonly eosTokenId: number;
  readonly maxNewTokens: number;
  readonly processLogits?: WhisperLogitProcessor;
  readonly onTokenLogits?: (
    chosenTokenId: number,
    processedLogits: Float32Array,
    ctx: { readonly tokens: readonly number[]; readonly beginIndex: number },
  ) => void;
  /** Raw decoder-init logits callback, before model-specific logit processing. */
  readonly onDecoderInitLogits?: (
    rawLogits: Float32Array,
    ctx: {
      readonly tokens: readonly number[];
      readonly beginIndex: number;
      readonly vocabSize: number;
      readonly noSpeechTokenId?: number;
    },
  ) => void;
  /** Model-specific no-speech token used by the raw init-logit callback. */
  readonly noSpeechTokenId?: number;
  /** Decoding strategy: greedy (argmax) or beam search */
  readonly strategy?: 'greedy' | 'beam';
  /** Beam size for beam search (default: 5) */
  readonly beamSize?: number;
  /** Final ranking penalty. Undefined uses length normalization; 0 uses raw score. */
  readonly lengthPenalty?: number;
  /** Patience for beam search early stopping (default: 1.0). */
  readonly patience?: number;
  /** Temperature (0 = greedy/beam argmax, >0 = sampling and disables beam search). */
  readonly temperature?: number;
  /** Number of independent sampling decodings to run when temperature > 0. Whisper: best_of. */
  readonly bestOf?: number;
  /** Track cumulative log-probability for best-of scoring. */
  readonly trackScore?: boolean;
  /**
   * Collect scalar logprob/entropy traces for the selected sequence.
   * Greedy/sampling compute traces only when this is true. Beam search always
   * records traces for the winning hypothesis.
   */
  readonly trackQuality?: boolean;
  /** Experimental: run active beam decoder steps in one batch when the session supports it. */
  readonly experimentalBatchedBeam?: boolean;
}

export interface WhisperDecodeResult {
  readonly tokens: readonly number[];
  /** Cumulative log-probability score (sum of log probs per token). */
  readonly score?: number;
  /** Selected-sequence scalar quality traces. Beam always includes these. */
  readonly tokenTraces?: readonly TokenQualityTrace[];
}

// ---------------------------------------------------------------------------
// Unified dispatch
// ---------------------------------------------------------------------------

export async function whisperDecode(
  session: WhisperCoreSession,
  options: WhisperDecodeOptions,
): Promise<WhisperDecodeResult> {
  const temperature = options.temperature ?? 0;
  const isSampling = Number.isFinite(temperature) && temperature > 0;
  const bestOf = isSampling ? options.bestOf ?? 1 : 1;
  if (isSampling && bestOf > 1) {
    return whisperBestOfDecode(session, options, bestOf);
  }
  if (isSampling) {
    return whisperGreedyDecode(session, options);
  }

  const strategy = options.strategy ?? 'greedy';
  if (strategy === 'beam' && (options.beamSize ?? 5) > 1) {
    return whisperBeamDecode(session, options);
  }
  return whisperGreedyDecode(session, options);
}

// ---------------------------------------------------------------------------
// Greedy decode
// ---------------------------------------------------------------------------

/**
 * Pure greedy decode loop for Whisper splitgraph inference.
 */
export async function whisperGreedyDecode(
  session: WhisperCoreSession,
  options: WhisperDecodeOptions,
): Promise<WhisperDecodeResult> {
  const {
    promptTokens,
    encoderOutput,
    encoderDims,
    eosTokenId,
    maxNewTokens,
    processLogits,
    onTokenLogits,
    onDecoderInitLogits,
    noSpeechTokenId,
    temperature = 0,
  } = options;

  const initResult = await session.runInit(promptTokens, encoderOutput, encoderDims);
  const vocabSize = initResult.vocabSize;
  let pastKv = initResult.presentKv;

  const lastLogitOffset = initResult.logits.length - vocabSize;
  const firstLogits = initResult.logits.subarray(lastLogitOffset);
  onDecoderInitLogits?.(new Float32Array(firstLogits), {
    tokens: promptTokens,
    beginIndex: promptTokens.length,
    vocabSize,
    noSpeechTokenId,
  });
  if (processLogits) processLogits(firstLogits, promptTokens, promptTokens.length);
  const firstTokenId = selectToken(firstLogits, temperature);
  const tokens: number[] = [firstTokenId];

  const trackScore = options.trackScore === true;
  const trackQuality = options.trackQuality === true;
  const tokenTraces: TokenQualityTrace[] = [];
  let cumulativeLogProb = 0;
  if (trackScore || trackQuality) {
    const firstQuality = tokenQualityFromLogits(firstLogits, firstTokenId);
    if (trackScore) cumulativeLogProb = firstQuality.logProb;
    if (trackQuality) {
      tokenTraces.push({
        tokenId: firstTokenId,
        logProb: firstQuality.logProb,
        entropy: firstQuality.entropy,
      });
    }
  }

  if (onTokenLogits) onTokenLogits(firstTokenId, firstLogits, { tokens, beginIndex: promptTokens.length });

  for (let step = 1; step < maxNewTokens; step++) {
    const stepResult = await session.runStep(tokens[tokens.length - 1]!, pastKv);
    if (processLogits) processLogits(stepResult.logits, [...promptTokens, ...tokens], promptTokens.length);
    const nextTokenId = selectToken(stepResult.logits, temperature);
    tokens.push(nextTokenId);
    pastKv = stepResult.presentKv;
    if (trackScore || trackQuality) {
      const stepQuality = tokenQualityFromLogits(stepResult.logits, nextTokenId);
      if (trackScore) cumulativeLogProb += stepQuality.logProb;
      if (trackQuality) {
        tokenTraces.push({
          tokenId: nextTokenId,
          logProb: stepQuality.logProb,
          entropy: stepQuality.entropy,
        });
      }
    }
    if (onTokenLogits) onTokenLogits(nextTokenId, stepResult.logits, { tokens, beginIndex: promptTokens.length });
    if (nextTokenId === eosTokenId) break;
  }

  return {
    tokens,
    ...(trackScore ? { score: cumulativeLogProb } : {}),
    ...(trackQuality ? { tokenTraces } : {}),
  };
}

// ---------------------------------------------------------------------------
// Beam search decode
// ---------------------------------------------------------------------------

/**
 * Beam search decode for Whisper splitgraph inference.
 *
 * Matches faster-whisper / WhisperX beam search behavior.
 */
export async function whisperBeamDecode(
  session: WhisperCoreSession,
  options: WhisperDecodeOptions,
): Promise<WhisperDecodeResult> {
  const {
    promptTokens, encoderOutput, encoderDims,
    eosTokenId, maxNewTokens, processLogits,
    onDecoderInitLogits,
    noSpeechTokenId,
    beamSize = 5,
    lengthPenalty,
    patience = 1,
  } = options;

  if (beamSize <= 1) {
    return whisperGreedyDecode(session, { ...options, strategy: 'greedy' });
  }
  if (maxNewTokens <= 0) return { tokens: [], score: 0 };

  const initResult = await session.runInit(promptTokens, encoderOutput, encoderDims);
  const vocabSize = initResult.vocabSize;

  const lastLogitOffset = initResult.logits.length - vocabSize;
  const firstLogits = initResult.logits.subarray(lastLogitOffset);
  onDecoderInitLogits?.(new Float32Array(firstLogits), {
    tokens: promptTokens,
    beginIndex: promptTokens.length,
    vocabSize,
    noSpeechTokenId,
  });
  if (processLogits) processLogits(firstLogits, promptTokens, promptTokens.length);

  const normalizedPatience = Number.isFinite(patience) && patience > 0 ? patience : 1;
  const maxFinishedCandidates = Math.max(1, roundToNearestEven(beamSize * normalizedPatience));
  const finished: WhisperBeamState<TokenQualityTrace[]>[] = [];
  const finishedKeys = new Set<string>();

  const firstExpansion = expandWhisperBeamStep(
    [{ tokens: [...promptTokens], score: 0, completed: false }],
    [firstLogits],
    beamSize,
    eosTokenId,
  );
  appendFinishedWhisperBeams(
    finished,
    finishedKeys,
    firstExpansion.finished,
  );

  let beams: WhisperBeamState<TokenQualityTrace[]>[] = firstExpansion.active;
  // Decoder adapters promise to treat cache inputs as read-only. Keep the
  // initial cache shared across sibling beams so immutable encoder KV does not
  // get cloned once per beam before the first batched step. The adapter still
  // clones/re-packs data when constructing backend inputs.
  let beamKvs = firstExpansion.parentIndexes.map(() => initResult.presentKv);

  let useBatchedBeam = options.experimentalBatchedBeam === true && Boolean(session.runStepBatch);

  for (let s = 1; s < maxNewTokens; s++) {
    if (beams.length === 0) break;

    const logitsByBeam: Float32Array[] = [];
    const activeTokenIds: number[] = [];
    for (const beam of beams) {
      activeTokenIds.push(beam.tokens[beam.tokens.length - 1]!);
    }

    let batchedStepResults: readonly WhisperStepResult[] | undefined;
    if (useBatchedBeam && beams.length > 1) {
      try {
        // Batching is an optimization boundary. A backend may reject dynamic
        // batch-shaped KV inputs even though scalar decoder_step works; keep
        // the active hypotheses untouched and retry this step scalarly.
        batchedStepResults = await session.runStepBatch!(activeTokenIds, beamKvs);
      } catch {
        batchedStepResults = undefined;
        useBatchedBeam = false;
      }
    }

    if (batchedStepResults !== undefined) {
      const stepResults = batchedStepResults;
      if (stepResults.length !== beams.length) {
        throw new Error(
          `Batched Whisper beam step returned ${stepResults.length} results for ${beams.length} active beams.`,
        );
      }
      for (let beamIndex = 0; beamIndex < beams.length; beamIndex++) {
        const beam = beams[beamIndex]!;
        const stepResult = stepResults[beamIndex]!;
        if (processLogits) processLogits(stepResult.logits, beam.tokens, promptTokens.length);
        logitsByBeam.push(stepResult.logits);
        beamKvs[beamIndex] = stepResult.presentKv;
      }
    } else {
      for (let beamIndex = 0; beamIndex < beams.length; beamIndex++) {
        const beam = beams[beamIndex]!;
        const prevToken = beam.tokens[beam.tokens.length - 1]!;
        const stepResult = await session.runStep(prevToken, beamKvs[beamIndex]!);
        if (processLogits) processLogits(stepResult.logits, beam.tokens, promptTokens.length);
        logitsByBeam.push(stepResult.logits);
        beamKvs[beamIndex] = stepResult.presentKv;
      }
    }

    const expansion = expandWhisperBeamStep(
      beams,
      logitsByBeam,
      beamSize,
      eosTokenId,
    );
    appendFinishedWhisperBeams(
      finished,
      finishedKeys,
      expansion.finished,
    );
    // Decoder sessions treat the cache as read-only and clone/repack it while
    // building their next input tensors. Sibling hypotheses can therefore
    // share the parent's immutable cache; cloning every layer here duplicated
    // the full KV payload after every beam expansion.
    beamKvs = expansion.parentIndexes.map((parentIndex) => beamKvs[parentIndex]!);
    beams = expansion.active;

    // Do not stop merely because `beamSize` EOS hypotheses exist. That is
    // equivalent to Transformers' `early_stopping=true` and can discard a
    // higher-scoring continuation (notably a punctuation token on beam 5).
    // Use the same score-based heuristic as the default HF beam scorer: once
    // the requested patience target is filled, stop only when the best active
    // hypothesis cannot overtake the worst retained finished hypothesis.
    if (whisperBeamSearchIsDone(
      finished,
      beams,
      maxFinishedCandidates,
      promptTokens.length,
      lengthPenalty,
    )) break;
  }

  if (finished.length < maxFinishedCandidates) {
    const unfinished = [...beams].sort((a, b) => b.score - a.score);
    for (const beam of unfinished) {
      appendFinishedWhisperBeams(
        finished,
        finishedKeys,
        [{ ...beam, tokens: [...beam.tokens, eosTokenId], completed: true }],
      );
    }
  }

  const best = selectBestFinishedWhisperBeam(
    finished,
    promptTokens.length,
    eosTokenId,
    lengthPenalty,
  );
  if (!best) {
    return whisperGreedyDecode(session, { ...options, strategy: 'greedy', trackQuality: true });
  }

  return {
    tokens: best.tokens.slice(promptTokens.length),
    score: best.score,
    tokenTraces: best.payload ?? [],
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface WhisperTopKToken {
  readonly tokenId: number;
  readonly logProb: number;
}

interface WhisperTopKSelection {
  readonly topTokens: readonly WhisperTopKToken[];
  readonly entropy: number;
}

/**
 * Compute the scalar statistics needed by beam expansion without materializing
 * a full-vocabulary log-softmax array. The old path allocated one Float32Array
 * per active beam and then scanned it again for top-k. Beam quality only needs
 * entropy plus the small candidate set, so keep the selector bounded by k.
 */
function selectTopKWithEntropy(logits: Float32Array, k: number): WhisperTopKSelection {
  const limit = Math.max(0, k);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;

  // A fully suppressed row is not expected for a valid Whisper step, but keep
  // the decode loop finite and deterministic if a custom processor produces it.
  if (!Number.isFinite(max)) {
    const topTokens: WhisperTopKToken[] = [];
    for (let tokenId = 0; tokenId < logits.length && tokenId < limit; tokenId++) {
      topTokens.push({ tokenId, logProb: Number.NEGATIVE_INFINITY });
    }
    return { topTokens, entropy: 0 };
  }

  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i]! - max);
  const logSum = Math.log(sum);
  // Keep the hot scan allocation-free. The previous implementation used
  // Array.splice() for every vocabulary item, which repeatedly shifted a
  // small object array even though only `limit` candidates can survive. A
  // fixed-capacity typed-array keeps the same descending-score / stable-tie
  // ordering while moving object creation to the final bounded result.
  const topTokenIds = new Int32Array(limit);
  const topLogProbs = new Float32Array(limit);
  let topCount = 0;
  let entropy = 0;
  for (let i = 0; i < logits.length; i++) {
    const rawLogProb = logits[i]! - max - logSum;
    const probability = Math.exp(rawLogProb);
    if (probability > 0) entropy -= probability * Math.log(probability);
    if (limit === 0) continue;

    // Preserve the previous Float32Array rounding before ranking candidates.
    // This keeps tie behavior stable against the old materialized path.
    const logProb = Math.fround(rawLogProb);
    if (topCount >= limit && !(logProb > topLogProbs[limit - 1]!)) continue;

    let insertAt = topCount;
    while (insertAt > 0 && logProb > topLogProbs[insertAt - 1]!) {
      insertAt--;
    }
    if (topCount < limit) topCount++;
    for (let index = topCount - 1; index > insertAt; index--) {
      topTokenIds[index] = topTokenIds[index - 1]!;
      topLogProbs[index] = topLogProbs[index - 1]!;
    }
    topTokenIds[insertAt] = i;
    topLogProbs[insertAt] = logProb;
  }
  const topTokens: WhisperTopKToken[] = [];
  for (let index = 0; index < topCount; index++) {
    topTokens.push({ tokenId: topTokenIds[index]!, logProb: topLogProbs[index]! });
  }
  return { topTokens, entropy };
}

interface WhisperBeamExpansion {
  readonly active: WhisperBeamState<TokenQualityTrace[]>[];
  readonly parentIndexes: number[];
  readonly finished: WhisperBeamState<TokenQualityTrace[]>[];
}

function expandWhisperBeamStep(
  beams: readonly WhisperBeamState<TokenQualityTrace[]>[],
  logitsByBeam: readonly Float32Array[],
  beamSize: number,
  eosTokenId: number,
): WhisperBeamExpansion {
  const candidates: Array<{
    readonly beam: WhisperBeamState<TokenQualityTrace[]>;
    readonly parentIndex: number;
    readonly tokenId: number;
    readonly score: number;
    readonly logProb: number;
    readonly entropy: number;
  }> = [];

  for (let parentIndex = 0; parentIndex < beams.length; parentIndex++) {
    const beam = beams[parentIndex];
    const logits = logitsByBeam[parentIndex];
    if (!beam || !logits || logits.length === 0) continue;
    const { topTokens, entropy } = selectTopKWithEntropy(logits, beamSize + 1);
    for (const { tokenId, logProb } of topTokens) {
      candidates.push({
        beam,
        parentIndex,
        tokenId,
        score: beam.score + logProb,
        logProb,
        entropy,
      });
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const active: WhisperBeamState<TokenQualityTrace[]>[] = [];
  const parentIndexes: number[] = [];
  const finished: WhisperBeamState<TokenQualityTrace[]>[] = [];

  for (const candidate of candidates) {
    const traces: TokenQualityTrace[] = [
      ...(candidate.beam.payload ?? []),
      {
        tokenId: candidate.tokenId,
        logProb: candidate.logProb,
        entropy: candidate.entropy,
      },
    ];
    const next: WhisperBeamState<TokenQualityTrace[]> = {
      tokens: [...candidate.beam.tokens, candidate.tokenId],
      score: candidate.score,
      completed: candidate.tokenId === eosTokenId,
      payload: traces,
    };
    if (next.completed) {
      finished.push(next);
      continue;
    }
    active.push(next);
    parentIndexes.push(candidate.parentIndex);
    if (active.length >= beamSize) break;
  }

  return { active, parentIndexes, finished };
}

function appendFinishedWhisperBeams(
  destination: WhisperBeamState<TokenQualityTrace[]>[],
  keys: Set<string>,
  candidates: readonly WhisperBeamState<TokenQualityTrace[]>[],
): void {
  for (const candidate of candidates) {
    const key = candidate.tokens.join(',');
    if (keys.has(key)) continue;
    keys.add(key);
    destination.push(candidate);
  }
}

function whisperBeamSearchIsDone(
  finished: readonly WhisperBeamState<TokenQualityTrace[]>[],
  active: readonly WhisperBeamState<TokenQualityTrace[]>[],
  targetFinished: number,
  promptLength: number,
  lengthPenalty: number | undefined,
): boolean {
  if (finished.length < targetFinished || active.length === 0) return active.length === 0;

  const rankedFinished = [...finished].sort((a, b) => (
    whisperSequenceRankScoreForLength(b, promptLength, lengthPenalty)
    - whisperSequenceRankScoreForLength(a, promptLength, lengthPenalty)
  ));
  const worstFinished = rankedFinished[targetFinished - 1];
  if (!worstFinished) return false;

  const bestActive = active.reduce((best, candidate) => (
    candidate.score > best.score ? candidate : best
  ), active[0]!);
  const bestPossibleActiveScore = whisperSequenceRankScoreForLength(
    bestActive,
    promptLength,
    lengthPenalty,
  );
  const worstFinishedScore = whisperSequenceRankScoreForLength(
    worstFinished,
    promptLength,
    lengthPenalty,
  );
  return worstFinishedScore >= bestPossibleActiveScore;
}

function whisperSequenceRankScoreForLength(
  beam: WhisperBeamState,
  promptLength: number,
  lengthPenalty: number | undefined,
): number {
  const generatedLength = Math.max(1, beam.tokens.length - promptLength - (
    beam.completed && beam.tokens.length > promptLength ? 1 : 0
  ));
  if (lengthPenalty === 0) return beam.score;
  if (lengthPenalty === undefined) return beam.score / generatedLength;
  return beam.score / Math.pow((5 + generatedLength) / 6, lengthPenalty);
}

function selectBestFinishedWhisperBeam(
  beams: readonly WhisperBeamState<TokenQualityTrace[]>[],
  promptLength: number,
  eosTokenId: number,
  lengthPenalty: number | undefined,
): WhisperBeamState<TokenQualityTrace[]> | undefined {
  let best: WhisperBeamState<TokenQualityTrace[]> | undefined;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const beam of beams) {
    const hasEos = beam.tokens[beam.tokens.length - 1] === eosTokenId;
    const generatedLength = Math.max(1, beam.tokens.length - promptLength - (hasEos ? 1 : 0));
    const normalized = whisperSequenceRankScore(beam.score, generatedLength, lengthPenalty);
    if (normalized > bestScore) {
      best = beam;
      bestScore = normalized;
    }
  }
  return best;
}

function whisperSequenceRankScore(
  cumulativeLogProb: number,
  tokenCount: number,
  lengthPenalty: number | undefined,
): number {
  const length = Math.max(1, tokenCount);
  const penalty = lengthPenalty === undefined
    ? length
    : Math.pow((5 + length) / 6, lengthPenalty);
  return cumulativeLogProb / penalty;
}

function roundToNearestEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction === 0.5) return floor % 2 === 0 ? floor : floor + 1;
  return Math.round(value);
}

function selectToken(logits: Float32Array, temperature: number): number {
  return Number.isFinite(temperature) && temperature > 0 ? sampleFromLogits(logits, temperature) : argmax(logits);
}

function sampleFromLogits(logits: Float32Array, temperature: number): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const value = (logits[i] as number) / temperature;
    if (value > max) max = value;
  }
  if (!Number.isFinite(max)) return argmax(logits);

  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const probability = Math.exp((logits[i] as number) / temperature - max);
    sum += probability;
  }
  if (!Number.isFinite(sum) || sum <= 0) return argmax(logits);

  let sample = Math.random() * sum;
  for (let i = 0; i < logits.length; i++) {
    sample -= Math.exp((logits[i] as number) / temperature - max);
    if (sample <= 0) return i;
  }
  return logits.length - 1;
}

// ---------------------------------------------------------------------------
// BestOf independent decodings
// ---------------------------------------------------------------------------

/**
 * Run N independent decodings and return the one with the best score.
 *
 * Matches Whisper/faster-whisper best_of behavior:
 * multiple independent sampling decodes, pick the best by normalized
 * cumulative log-probability score. Beam search is a temperature=0 path.
 */
async function whisperBestOfDecode(
  session: WhisperCoreSession,
  options: WhisperDecodeOptions,
  bestOf: number,
): Promise<WhisperDecodeResult> {
  const lengthPenalty = options.lengthPenalty;
  let bestResult: WhisperDecodeResult | null = null;
  let bestScore = -Infinity;

  for (let i = 0; i < bestOf; i++) {
    const result = await whisperGreedyDecode(session, {
      ...options,
      strategy: 'greedy',
      beamSize: 1,
      trackScore: true,
      trackQuality: options.trackQuality === true,
    });

    const tokenCount = result.tokens.length;
    const normScore = result.score !== undefined
      ? whisperSequenceRankScore(result.score, tokenCount, lengthPenalty)
      : -Infinity;

    if (normScore > bestScore) {
      bestScore = normScore;
      bestResult = result;
    }
  }

  return bestResult!;
}
