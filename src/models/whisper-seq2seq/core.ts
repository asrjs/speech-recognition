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
import {
  createInitialWhisperBeam,
  rankWhisperBeamCandidates,
  selectBestWhisperBeam,
} from './beam-search.js';

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
  /** Length penalty for beam search (default: 0.0) */
  readonly lengthPenalty?: number;
  /** Patience for beam search early stopping (default: 1.0). */
  readonly patience?: number;
  /** Temperature (0 = greedy/beam argmax, >0 = sampling and disables beam search). */
  readonly temperature?: number;
  /** Number of independent sampling decodings to run when temperature > 0. Whisper: best_of. */
  readonly bestOf?: number;
  /** Track cumulative log-probability for best-of scoring. */
  readonly trackScore?: boolean;
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
    beamSize = 5, lengthPenalty = 0,
    patience = 1,
  } = options;

  const initResult = await session.runInit(promptTokens, encoderOutput, encoderDims);
  const vocabSize = initResult.vocabSize;

  const lastLogitOffset = initResult.logits.length - vocabSize;
  const firstLogits = initResult.logits.subarray(lastLogitOffset);
  if (processLogits) processLogits(firstLogits, promptTokens, promptTokens.length);

  const firstLogProbs = logSoftmax(firstLogits);
  const topK = selectTopK(firstLogProbs, beamSize);

  let beams = topK.map(({ tokenId, logProb }) =>
    createInitialWhisperBeam([...promptTokens, tokenId], logProb) as any,
  );

  // Clone KV cache per beam
  let beamKvs: WhisperKvCache[] = beams.map(() => {
    const c: WhisperKvCache = {};
    for (const [k, v] of Object.entries(initResult.presentKv)) c[k] = cloneWhisperKvCacheValue(v);
    return c;
  });

  let completedSteps = 0;

  for (let s = 1; s < maxNewTokens; s++) {
    if (beams.every(b => (b as any).completed)) break;

    const logitsByBeam: Float32Array[] = [];
    for (let bi = 0; bi < beams.length; bi++) {
      const beam = beams[bi] as any;
      if (beam.completed) { logitsByBeam.push(new Float32Array(0)); continue; }
      const prevToken = beam.tokens[beam.tokens.length - 1];
      const stepResult = await session.runStep(prevToken, beamKvs[bi]!);
      const sl = stepResult.logits;
      if (processLogits) processLogits(sl, beam.tokens, promptTokens.length);
      logitsByBeam.push(sl);
      beamKvs[bi] = stepResult.presentKv;
    }

    const candidates = rankWhisperBeamCandidates({
      beams: beams as any,
      logitsByBeam,
      beamWidth: beamSize,
      eosTokenId,
      lengthPenalty,
    });

    // Rebuild KV cache for survivors
    const newKvs: WhisperKvCache[] = [];
    for (const cand of candidates) {
      const cTokens = (cand as any).tokens as number[];
      let matchedKv: WhisperKvCache | undefined;
      for (let bi = 0; bi < beams.length; bi++) {
        const parent = beams[bi] as any;
        const pTokens = parent.tokens as number[];
        const isExpandedFromParent = pTokens.length + 1 === cTokens.length &&
          cTokens.slice(0, pTokens.length).every((t, i) => t === pTokens[i]);
        const isRetainedParent = pTokens.length === cTokens.length &&
          cTokens.every((t, i) => t === pTokens[i]);
        if (isExpandedFromParent || isRetainedParent) {
          const clone: WhisperKvCache = {};
          for (const [k, v] of Object.entries(beamKvs[bi]!)) clone[k] = cloneWhisperKvCacheValue(v);
          matchedKv = clone;
          break;
        }
      }
      newKvs.push(matchedKv ?? {});
    }

    beams = candidates as any[];
    beamKvs = newKvs;

    // Patience: stop early if best beam has been completed for N consecutive steps
    const bestBeam = selectBestWhisperBeam(beams as any, lengthPenalty);
    if (bestBeam && (bestBeam as any).completed) {
      completedSteps++;
      if (completedSteps >= patience) break;
    } else {
      completedSteps = 0;
    }

    if (beams.every(b => (b as any).completed)) break;
  }

  const best = selectBestWhisperBeam(beams as any, lengthPenalty) as any;
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
  const lengthPenalty = options.lengthPenalty ?? 0;
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
      ? (lengthPenalty === 0 ? result.score : result.score / Math.pow(Math.max(1, tokenCount), lengthPenalty))
      : -Infinity;

    if (normScore > bestScore) {
      bestScore = normScore;
      bestResult = result;
    }
  }

  return bestResult!;
}
