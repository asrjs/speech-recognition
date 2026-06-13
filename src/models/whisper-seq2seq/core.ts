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
    pastKv: Record<string, Float32Array>,
  ): Promise<WhisperStepResult>;
}

export interface WhisperInitResult {
  readonly logits: Float32Array;
  readonly vocabSize: number;
  readonly presentKv: Record<string, Float32Array>;
}

export interface WhisperStepResult {
  readonly logits: Float32Array;
  readonly vocabSize: number;
  readonly presentKv: Record<string, Float32Array>;
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
  /** Temperature (0 = greedy argmax, >0 = sample). Greedy mode only. */
  readonly temperature?: number;
  /** Number of independent decodings to run, pick best by score. WhisperX: best_of */
  readonly bestOf?: number;
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
  const bestOf = options.bestOf ?? 1;
  if (bestOf > 1) {
    return whisperBestOfDecode(session, options, bestOf);
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
  const { promptTokens, encoderOutput, encoderDims, eosTokenId, maxNewTokens, processLogits, onTokenLogits } = options;

  const initResult = await session.runInit(promptTokens, encoderOutput, encoderDims);
  const vocabSize = initResult.vocabSize;
  let pastKv = initResult.presentKv;

  const lastLogitOffset = initResult.logits.length - vocabSize;
  const firstLogits = initResult.logits.subarray(lastLogitOffset);
  if (processLogits) processLogits(firstLogits, promptTokens, promptTokens.length);
  const firstTokenId = argmax(firstLogits);
  const tokens: number[] = [firstTokenId];

  // Track cumulative log-probability for bestOf scoring
  let cumulativeLogProb = logProbOfToken(firstLogits, firstTokenId);

  if (onTokenLogits) onTokenLogits(firstTokenId, firstLogits, { tokens, beginIndex: promptTokens.length });

  for (let step = 1; step < maxNewTokens; step++) {
    const stepResult = await session.runStep(tokens[tokens.length - 1]!, pastKv);
    if (processLogits) processLogits(stepResult.logits, [...promptTokens, ...tokens], promptTokens.length);
    const nextTokenId = argmax(stepResult.logits);
    tokens.push(nextTokenId);
    pastKv = stepResult.presentKv;
    cumulativeLogProb += logProbOfToken(stepResult.logits, nextTokenId);
    if (onTokenLogits) onTokenLogits(nextTokenId, stepResult.logits, { tokens, beginIndex: promptTokens.length });
    if (nextTokenId === eosTokenId) break;
  }

  return { tokens, score: cumulativeLogProb };
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
  let beamKvs: Record<string, Float32Array>[] = beams.map(() => {
    const c: Record<string, Float32Array> = {};
    for (const [k, v] of Object.entries(initResult.presentKv)) c[k] = new Float32Array(v);
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
    const newKvs: Record<string, Float32Array>[] = [];
    for (const cand of candidates) {
      const cTokens = (cand as any).tokens as number[];
      for (let bi = 0; bi < beams.length; bi++) {
        const parent = beams[bi] as any;
        const pTokens = parent.tokens as number[];
        if (pTokens.length + 1 === cTokens.length &&
            cTokens.slice(0, pTokens.length).every((t, i) => t === pTokens[i])) {
          const clone: Record<string, Float32Array> = {};
          for (const [k, v] of Object.entries(beamKvs[bi]!)) clone[k] = new Float32Array(v);
          newKvs.push(clone);
          break;
        }
      }
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
  const indexed = Array.from(logProbs, (lp, i) => ({ tokenId: i, logProb: lp }));
  indexed.sort((a, b) => b.logProb - a.logProb);
  return indexed.slice(0, k);
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

// ---------------------------------------------------------------------------
// BestOf independent decodings
// ---------------------------------------------------------------------------

/**
 * Run N independent decodings and return the one with the best score.
 *
 * Matches WhisperX/faster-whisper best_of behavior:
 * multiple independent beam/greedy decodes, pick the best by
 * normalized cumulative log-probability score.
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
    const result = await (options.strategy === 'beam' && (options.beamSize ?? 5) > 1
      ? whisperBeamDecode(session, options)
      : whisperGreedyDecode(session, options));

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
