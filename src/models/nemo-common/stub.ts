import { estimateFrameBasedProcessorDescriptor } from '../../audio/index.js';
import type { AudioBufferLike } from '../../types/index.js';
import { StubSentencePieceTokenizer } from '../../tokenizers/index.js';
import type {
  NemoFeatureDescriptor,
  NemoFeatureExtractor,
  NemoModelConfig,
  NemoNativeWord,
  NemoTokenizer,
} from './types.js';

export class StubNemoFeatureExtractor implements NemoFeatureExtractor {
  readonly kind = 'nemo-mel' as const;
  readonly sharedModule = 'audio' as const;

  compute(audio: AudioBufferLike, config: NemoModelConfig): NemoFeatureDescriptor {
    return {
      ...estimateFrameBasedProcessorDescriptor(audio, {
        processor: 'nemo-mel',
        featureSize: config.melBins,
        frameShiftSeconds: config.frameShiftSeconds,
        subsamplingFactor: config.subsamplingFactor,
      }),
      featureSize: config.melBins,
    };
  }
}

export class StubNemoTokenizer extends StubSentencePieceTokenizer implements NemoTokenizer {}

export function buildStubTimedWords(
  lexemes: readonly string[],
  durationSeconds: number,
): NemoNativeWord[] {
  const wordSpan = Math.max(durationSeconds, 0.3) / Math.max(1, lexemes.length);

  return lexemes.map((text, index) => ({
    index,
    text,
    startTime: Number((index * wordSpan).toFixed(3)),
    endTime: Number(((index + 1) * wordSpan).toFixed(3)),
    confidence: 0.92,
  }));
}
