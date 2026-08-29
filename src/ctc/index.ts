/**
 * Generic CTC module — shared by all CTC-based models.
 *
 * @module ctc
 */

export type {
  CtcTokenizerLike,
  CtcRawTokenSpan,
  CtcTokenSpan,
  CtcUtteranceTiming,
  CtcSentenceTiming,
  CtcNativeWord,
  CtcArgmaxResult,
  CtcCollapseResult,
  CtcDecoderConfig,
  CtcFrameTimingOptions,
  CtcDecodeResult,
} from './types.js';

export {
  CtcDecoder,
  argmaxAndSelectedLogProbs,
  argmaxAndSelectedLogProbsFp16,
  buildSentenceTimings,
  buildUtteranceTiming,
  buildWordsFromCharSpans,
  ctcCollapseWithSpans,
  addTimesToTokenSpans,
  estimateSecondsPerOutputFrame,
} from './decoder.js';
