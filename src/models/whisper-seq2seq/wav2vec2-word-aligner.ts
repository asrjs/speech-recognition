import { createWav2Vec2AlignerFromLogits } from '../../alignment/index.js';
import type { AudioBufferLike } from '../../types/index.js';
import type { Wav2Vec2Executor } from '../wav2vec2/index.js';
import type { WhisperWordAligner } from './types.js';

/**
 * WhisperX-style word aligner: run Wav2Vec2 CTC logits, then Viterbi-align
 * the Whisper transcript. Callers pass this as `wordAligner` so GPU-KV greedy
 * stays unchanged unless alignment is requested.
 */
export function createWhisperWav2Vec2WordAligner(
  wav2vec2: Pick<Wav2Vec2Executor, 'extractLogits'>,
): WhisperWordAligner {
  return {
    async align({ transcript, audio }) {
      const logits = await wav2vec2.extractLogits(audio as AudioBufferLike);
      const aligner = createWav2Vec2AlignerFromLogits({
        logits: logits.logits,
        frameCount: logits.frameCount,
        vocabSize: logits.vocabSize,
        blankId: logits.blankId,
        tokenizer: {
          encode: (text) => logits.tokenizer.encode(text),
          decode: (ids) => logits.tokenizer.decode(ids),
          decodeTokenPiece: logits.tokenizer.decodeTokenPiece?.bind(logits.tokenizer),
        },
        sampleRate: logits.sampleRate,
        audioDurationSeconds: logits.audioDurationSeconds,
      });
      const aligned = aligner.align({
        transcript,
        audioDurationSeconds: logits.audioDurationSeconds,
      });
      return aligned.words.map((word) => ({
        text: word.text,
        startTime: word.start,
        endTime: word.end,
        confidence: word.confidence,
      }));
    },
  };
}
