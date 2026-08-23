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

import { argmax } from '../../inference/index.js';
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

  runStep(
    tokenId: number,
    pastKv: WhisperKvCache,
  ): Promise<WhisperStepResult>;

  runStepBatch?(
    tokenIds: readonly number[],
    pastKvs: readonly WhisperKvCache[],
  ): Promise<readonly WhisperStepResult[]>;
}

export interface WhisperKvCacheEntry {
  readonly data: ArrayBufferView;
  readonly dims?: readonly number[];
  readonly type?: string;
}

export type WhisperKvCacheValue = ArrayBufferView | WhisperKvCacheEntry;
export type WhisperKvCache = Record<string, WhisperKvCacheValue>;

type WhisperKvDataView = ArrayBufferView & { readonly length?: number };
type WhisperKvDataConstructor = {
  new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): ArrayBufferView;
};

function isWhisperKvCacheEntry(value: WhisperKvCacheValue): value is WhisperKvCacheEntry {
  return !ArrayBuffer.isView(value);
}

function cloneWhisperKvData(data: ArrayBufferView): ArrayBufferView {
  const view = data as WhisperKvDataView;
  const buffer = (view.buffer as ArrayBuffer).slice(view.byteOffset, view.byteOffset + view.byteLength);
  if (typeof view.length === 'number') {
    const ctor = view.constructor as WhisperKvDataConstructor;
    return new ctor(buffer, 0, view.length);
  }
  return new DataView(buffer);
}

function cloneWhisperKvCacheValue(value: WhisperKvCacheValue): WhisperKvCacheValue {
  if (isWhisperKvCacheEntry(value)) {
    return {
      data: cloneWhisperKvData(value.data),
      ...(value.dims ? { dims: value.dims } : {}),
      ...(value.type ? { type: value.type } : {}),
    };
  }
  return cloneWhisperKvData(value);
}

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
  /** Experimental: run active beam decoder steps in one batch when the session supports it. */
  readonly experimentalBatchedBeam?: boolean;
}

export interface WhisperDecodeResult {
  readonly tokens: readonly number[];
  /** Cumulative log-probability score (sum of log probs per token). */
  readonly score?: number;
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
    temperature = 0,
  } = options;

  const initResult = await session.runInit(promptTokens, encoderOutput, encoderDims);
  const vocabSize = initResult.vocabSize;
  let pastKv = initResult.presentKv;

  const lastLogitOffset = initResult.logits.length - vocabSize;
  const firstLogits = initResult.logits.subarray(lastLogitOffset);
  if (processLogits) processLogits(firstLogits, promptTokens, promptTokens.length);
  const firstTokenId = selectToken(firstLogits, temperature);
  const tokens: number[] = [firstTokenId];

  const trackScore = options.trackScore === true;
  let cumulativeLogProb = trackScore ? logProbOfToken(firstLogits, firstTokenId) : 0;

  if (onTokenLogits) onTokenLogits(firstTokenId, firstLogits, { tokens, beginIndex: promptTokens.length });

  for (let step = 1; step < maxNewTokens; step++) {
    const stepResult = await session.runStep(tokens[tokens.length - 1]!, pastKv);
    if (processLogits) processLogits(stepResult.logits, [...promptTokens, ...tokens], promptTokens.length);
    const nextTokenId = selectToken(stepResult.logits, temperature);
    tokens.push(nextTokenId);
    pastKv = stepResult.presentKv;
    if (trackScore) {
      cumulativeLogProb += logProbOfToken(stepResult.logits, nextTokenId);
    }
    if (onTokenLogits) onTokenLogits(nextTokenId, stepResult.logits, { tokens, beginIndex: promptTokens.length });
    if (nextTokenId === eosTokenId) break;
  }

  return trackScore ? { tokens, score: cumulativeLogProb } : { tokens };
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
  if (processLogits) processLogits(firstLogits, promptTokens, promptTokens.length);

  const normalizedPatience = Number.isFinite(patience) && patience > 0 ? patience : 1;
  const maxFinishedCandidates = Math.max(1, roundToNearestEven(beamSize * normalizedPatience));
  const finished: WhisperBeamState[] = [];
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
    maxFinishedCandidates,
  );

  let beams = firstExpansion.active;
  let beamKvs = firstExpansion.parentIndexes.map(() => cloneWhisperKvCache(initResult.presentKv));

  const useBatchedBeam = options.experimentalBatchedBeam === true && Boolean(session.runStepBatch);

  for (let s = 1; s < maxNewTokens; s++) {
    if (finished.length >= maxFinishedCandidates || beams.length === 0) break;

    const logitsByBeam: Float32Array[] = [];
    const activeTokenIds: number[] = [];
    for (const beam of beams) {
      activeTokenIds.push(beam.tokens[beam.tokens.length - 1]!);
    }

    if (useBatchedBeam && beams.length > 1) {
      const stepResults = await session.runStepBatch!(activeTokenIds, beamKvs);
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
      maxFinishedCandidates,
    );
    beamKvs = expansion.parentIndexes.map((parentIndex) => cloneWhisperKvCache(beamKvs[parentIndex]!));
    beams = expansion.active;
  }

  if (finished.length < beamSize) {
    const unfinished = [...beams].sort((a, b) => b.score - a.score);
    for (const beam of unfinished) {
      appendFinishedWhisperBeams(
        finished,
        finishedKeys,
        [{ ...beam, tokens: [...beam.tokens, eosTokenId], completed: true }],
        beamSize,
      );
      if (finished.length >= beamSize) break;
    }
  }

  const best = selectBestFinishedWhisperBeam(
    finished,
    promptTokens.length,
    eosTokenId,
    lengthPenalty,
  );
  if (!best) return whisperGreedyDecode(session, { ...options, strategy: 'greedy' });

  return { tokens: best.tokens.slice(promptTokens.length), score: best.score };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function logSoftmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i]! - max);
  const logSum = Math.log(sum);
  const result = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) result[i] = logits[i]! - max - logSum;
  return result;
}

function selectTopK(logProbs: Float32Array, k: number): { tokenId: number; logProb: number }[] {
  const limit = Math.max(0, k);
  const top: { tokenId: number; logProb: number }[] = [];
  for (let tokenId = 0; tokenId < logProbs.length; tokenId++) {
    const logProb = logProbs[tokenId] ?? Number.NEGATIVE_INFINITY;
    let insertAt = top.length;
    while (insertAt > 0 && logProb > (top[insertAt - 1]?.logProb ?? Number.NEGATIVE_INFINITY)) {
      insertAt--;
    }
    if (insertAt >= limit) continue;
    top.splice(insertAt, 0, { tokenId, logProb });
    if (top.length > limit) top.pop();
  }
  return top;
}

interface WhisperBeamExpansion {
  readonly active: WhisperBeamState[];
  readonly parentIndexes: number[];
  readonly finished: WhisperBeamState[];
}

function expandWhisperBeamStep(
  beams: readonly WhisperBeamState[],
  logitsByBeam: readonly Float32Array[],
  beamSize: number,
  eosTokenId: number,
): WhisperBeamExpansion {
  const candidates: Array<{
    readonly beam: WhisperBeamState;
    readonly parentIndex: number;
    readonly tokenId: number;
    readonly score: number;
  }> = [];

  for (let parentIndex = 0; parentIndex < beams.length; parentIndex++) {
    const beam = beams[parentIndex];
    const logits = logitsByBeam[parentIndex];
    if (!beam || !logits || logits.length === 0) continue;
    const topTokens = selectTopK(logSoftmax(logits), beamSize + 1);
    for (const { tokenId, logProb } of topTokens) {
      candidates.push({ beam, parentIndex, tokenId, score: beam.score + logProb });
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  const active: WhisperBeamState[] = [];
  const parentIndexes: number[] = [];
  const finished: WhisperBeamState[] = [];

  for (const candidate of candidates) {
    const next: WhisperBeamState = {
      tokens: [...candidate.beam.tokens, candidate.tokenId],
      score: candidate.score,
      completed: candidate.tokenId === eosTokenId,
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

function cloneWhisperKvCache(cache: WhisperKvCache): WhisperKvCache {
  return Object.fromEntries(
    Object.entries(cache).map(([key, value]) => [key, cloneWhisperKvCacheValue(value)]),
  );
}

function appendFinishedWhisperBeams(
  destination: WhisperBeamState[],
  keys: Set<string>,
  candidates: readonly WhisperBeamState[],
  limit: number,
): void {
  for (const candidate of candidates) {
    if (destination.length >= limit) return;
    const key = candidate.tokens.join(',');
    if (keys.has(key)) continue;
    keys.add(key);
    destination.push(candidate);
  }
}

function selectBestFinishedWhisperBeam(
  beams: readonly WhisperBeamState[],
  promptLength: number,
  eosTokenId: number,
  lengthPenalty: number | undefined,
): WhisperBeamState | undefined {
  let best: WhisperBeamState | undefined;
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

/**
 * Compute log-probability of a specific token from raw logits.
 * Returns log_softmax(logits)[tokenId].
 */
function logProbOfToken(logits: Float32Array, tokenId: number): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i]! - max);
  return (logits[tokenId] ?? -Infinity) - max - Math.log(sum);
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
