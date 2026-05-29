/**
 * Whisper Core — pure decode logic (ONNX-agnostic).
 *
 * This module contains the vanilla Whisper inference loop, independent of
 * ONNX Runtime, the asrjs model-family system, or any audio processing.
 *
 * A "session" is any object that can run decoder_init and decoder_step.
 * The session is responsible for:
 *   - Converting prompt tokens + encoder output into init logits + KV cache
 *   - Converting a single token + KV cache into step logits + updated KV cache
 *   - Handling dtype conversion (float16 → float32) internally
 *   - Mapping ONNX tensor names (present. ↔ past_key_values.) internally
 *
 * This design follows the same pattern as Nemo TDT's executor where
 * the decode loop is separated from the ONNX bridge.
 */

import { argmax } from '../../inference/index.js';

// ---------------------------------------------------------------------------
// Session interface
// ---------------------------------------------------------------------------

/**
 * A Whisper decoder session — provides init and step methods.
 * Implementations wrap ONNX Runtime, a mock, or any backend.
 */
export interface WhisperCoreSession {
  /**
   * Run decoder_init: prompt tokens + encoder output → logits + KV cache.
   * The returned presentKv must use keys compatible with runStep input
   * (implementation handles present. ↔ past_key_values. mapping internally).
   */
  runInit(
    promptTokens: readonly number[],
    encoderOutput: Float32Array,
    encoderDims: readonly number[],
  ): Promise<WhisperInitResult>;

  /**
   * Run decoder_step: single token + KV cache → logits + updated KV cache.
   * The pastKv keys match what runInit returned (or previous runStep returned).
   */
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
// Logit processor (callback-based — same as transformers.js)
// ---------------------------------------------------------------------------

/**
 * A logit processor mutates logits before argmax.
 * Typical implementations: timestamp suppression, EOS handling, begin/prevent tokens.
 */
export type WhisperLogitProcessor = (
  logits: Float32Array,
  generatedTokens: readonly number[],
  beginIndex: number,
) => void;

// ---------------------------------------------------------------------------
// Decode options
// ---------------------------------------------------------------------------

export interface WhisperDecodeOptions {
  /** Prompt token IDs: [<|startoftranscript|>, <|lang|>, <|transcribe|>, ...] */
  readonly promptTokens: readonly number[];
  /** Encoder output tensor data (flat Float32Array) */
  readonly encoderOutput: Float32Array;
  /** Encoder output dims (for computing time steps) */
  readonly encoderDims: readonly number[];
  /** EOS token ID (typically 50257) */
  readonly eosTokenId: number;
  /** Maximum new tokens to generate */
  readonly maxNewTokens: number;
  /** Optional logit processor (e.g. WhisperTimestampLogitProcessor) */
  readonly processLogits?: WhisperLogitProcessor;
}

export interface WhisperDecodeResult {
  readonly tokens: readonly number[];
}

// ---------------------------------------------------------------------------
// Core decode loop
// ---------------------------------------------------------------------------

/**
 * Pure greedy decode loop for Whisper splitgraph inference.
 *
 * Algorithm (matches OpenAI Whisper, HF Transformers, faster-whisper):
 *   1. runInit(prompt, encoder) → first logits + KV cache
 *   2. argmax first logits → first token
 *   3. Loop: runStep(prevToken, KV) → next logits
 *   4. argmax → next token; stop on EOS or max tokens
 *
 * The session handles all ONNX/backend details.
 * The processLogits callback handles timestamp/suppress rules.
 */
export async function whisperGreedyDecode(
  session: WhisperCoreSession,
  options: WhisperDecodeOptions,
): Promise<WhisperDecodeResult> {
  const { promptTokens, encoderOutput, encoderDims, eosTokenId, maxNewTokens, processLogits } = options;

  // Init: prefill with prompt tokens
  const initResult = await session.runInit(promptTokens, encoderOutput, encoderDims);
  const vocabSize = initResult.vocabSize;
  let pastKv = initResult.presentKv;

  // First token from init logits (last position, after prompt)
  const lastLogitOffset = initResult.logits.length - vocabSize;
  const firstLogits = initResult.logits.subarray(lastLogitOffset);
  if (processLogits) {
    processLogits(firstLogits, promptTokens, promptTokens.length);
  }
  const firstTokenId = argmax(firstLogits);
  const tokens: number[] = [firstTokenId];

  // Autoregressive step loop
  for (let step = 1; step < maxNewTokens; step++) {
    const stepResult = await session.runStep(tokens[tokens.length - 1]!, pastKv);
    if (processLogits) {
      processLogits(stepResult.logits, [...promptTokens, ...tokens], promptTokens.length);
    }
    const nextTokenId = argmax(stepResult.logits);
    tokens.push(nextTokenId);
    pastKv = stepResult.presentKv;

    if (nextTokenId === eosTokenId) break;
  }

  return { tokens };
}
