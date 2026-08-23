export interface WhisperTimestampLogitProcessorOptions {
  readonly eosTokenId: number;
  readonly noTimestampsTokenId: number;
  readonly timestampBegin: number;
  readonly suppressTokens?: readonly number[];
  readonly beginSuppressTokens?: readonly number[];
  readonly maxInitialTimestampIndex?: number;
}

export class WhisperTimestampLogitProcessor {
  private readonly eosTokenId: number;
  private readonly noTimestampsTokenId: number;
  private readonly timestampBegin: number;
  private readonly suppressTokens: readonly number[];
  private readonly beginSuppressTokens: readonly number[];
  private readonly maxInitialTimestampIndex?: number;

  constructor(options: WhisperTimestampLogitProcessorOptions) {
    this.eosTokenId = options.eosTokenId;
    this.noTimestampsTokenId = options.noTimestampsTokenId;
    this.timestampBegin = options.timestampBegin;
    this.suppressTokens = options.suppressTokens ?? [];
    this.beginSuppressTokens = options.beginSuppressTokens ?? [];
    this.maxInitialTimestampIndex = options.maxInitialTimestampIndex;
  }

  process(logits: Float32Array, generatedTokens: readonly number[], beginIndex: number): void {
    // 1. Always suppress suppress_tokens
    for (const tokenId of this.suppressTokens) {
      if (tokenId < logits.length) logits[tokenId] = -Infinity;
    }

    // 2. Suppress begin_suppress_tokens only on the first generated token
    if (generatedTokens.length === beginIndex) {
      for (const tokenId of this.beginSuppressTokens) {
        if (tokenId < logits.length) logits[tokenId] = -Infinity;
      }
    }

    // 3. Detect no_timestamps mode: check if prompt contains <|notimestamps|>
    const hasNoTimestamps = generatedTokens.includes(this.noTimestampsTokenId);

    if (hasNoTimestamps) {
      // Suppress all timestamp tokens
      for (let ts = this.timestampBegin; ts < logits.length; ts++) {
        logits[ts] = -Infinity;
      }
      return;
    }

    // The control token belongs in the prompt, never in timestamped output.
    if (this.noTimestampsTokenId < logits.length) {
      logits[this.noTimestampsTokenId] = -Infinity;
    }

    // 4. Timestamp state processing (only when timestamps are allowed)
    const sampledTokens = generatedTokens.slice(beginIndex);
    const seq = sampledTokens;

    if (seq.length === 0) {
      // First generated token: suppress all text tokens, only timestamps/EOS allowed
      for (let t = 0; t < this.timestampBegin; t++) {
        logits[t] = -Infinity;
      }
      // Apply max_initial_timestamp_index if set
      if (this.maxInitialTimestampIndex !== undefined) {
        const lastAllowed = this.timestampBegin + this.maxInitialTimestampIndex;
        for (let ts = lastAllowed + 1; ts < logits.length; ts++) {
          logits[ts] = -Infinity;
        }
      }
      return;
    }

    const lastIsTimestamp = seq[seq.length - 1]! >= this.timestampBegin;
    const penultimateIsTimestamp = seq.length < 2 || seq[seq.length - 2]! >= this.timestampBegin;

    if (lastIsTimestamp) {
      if (penultimateIsTimestamp) {
        // Two timestamps in a row — suppress ALL timestamps (force non-timestamp)
        for (let ts = this.timestampBegin; ts < logits.length; ts++) {
          logits[ts] = -Infinity;
        }
      } else {
        // Last is timestamp, penultimate is text — suppress text tokens (force EOS)
        for (let t = 0; t < this.eosTokenId; t++) {
          logits[t] = -Infinity;
        }
      }
    }

    // 5. Enforce monotonically increasing timestamps
    const timestamps = sampledTokens.filter((t) => t >= this.timestampBegin);
    if (timestamps.length > 0) {
      let lastTimestampValue: number;
      if (lastIsTimestamp && !penultimateIsTimestamp) {
        // Last token is a timestamp and it's not a pair → use the exact last value
        lastTimestampValue = timestamps[timestamps.length - 1]!;
      } else {
        // Avoid emitting the same timestamp again
        lastTimestampValue = timestamps[timestamps.length - 1]! + 1;
      }
      // Suppress all timestamps smaller than this value
      for (let ts = this.timestampBegin; ts < lastTimestampValue; ts++) {
        logits[ts] = -Infinity;
      }
    }

    // Whisper forces a timestamp when their aggregate probability exceeds the
    // most likely text token. The common softmax denominator cancels, so the
    // comparison can be performed directly in logit space.
    const timestampLogSumExp = logSumExp(logits, this.timestampBegin, logits.length);
    let maxTextLogit = Number.NEGATIVE_INFINITY;
    for (let tokenId = 0; tokenId < this.timestampBegin; tokenId++) {
      maxTextLogit = Math.max(maxTextLogit, logits[tokenId] ?? Number.NEGATIVE_INFINITY);
    }
    if (timestampLogSumExp > maxTextLogit) {
      for (let tokenId = 0; tokenId < this.timestampBegin; tokenId++) {
        logits[tokenId] = -Infinity;
      }
    }
  }
}

function logSumExp(values: Float32Array, start: number, end: number): number {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = start; i < end; i++) {
    max = Math.max(max, values[i] ?? Number.NEGATIVE_INFINITY);
  }
  if (!Number.isFinite(max)) return max;

  let sum = 0;
  for (let i = start; i < end; i++) {
    sum += Math.exp((values[i] ?? Number.NEGATIVE_INFINITY) - max);
  }
  return max + Math.log(sum);
}
