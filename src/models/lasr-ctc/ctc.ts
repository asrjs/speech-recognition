/**
 * Legacy CTC decode functions — re-exported from the shared src/ctc/ module.
 *
 * This file preserves backward compatibility for existing consumers:
 * - src/models/lasr-ctc/executor.ts (imports from './ctc.js')
 * - src/models/wav2vec2/executor.ts (imports from '../lasr-ctc/ctc.js')
 * - tests/lasr-ctc-medasr-port-helpers.test.ts
 *
 * New code should import directly from '../../ctc/index.js' instead.
 *
 * @deprecated Import from 'src/ctc/' instead.
 * @module models/lasr-ctc/ctc
 */

// Re-export all public functions from the shared CTC module.
// Internal types (RawTokenSpan, TokenDecoderLike) are kept local for now
// since they are not used outside this module.

export {
  argmaxAndSelectedLogProbs,
  ctcCollapseWithSpans,
  estimateSecondsPerOutputFrame,
  addTimesToTokenSpans,
  buildUtteranceTiming,
  buildSentenceTimings,
} from '../../ctc/decoder.js';

// Re-export the shared CTC types that were previously imported from ./types.js
// and are structurally identical to the new shared types.
export type {
  CtcTokenSpan as LasrCtcTokenSpan,
  CtcUtteranceTiming as LasrCtcUtteranceTiming,
  CtcSentenceTiming as LasrCtcSentenceTiming,
} from '../../ctc/types.js';
