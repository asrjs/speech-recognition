import { chunkPcmAudio, normalizePcmInput } from '../../audio/index.js';
import type {
  AudioInputLike,
  BaseTranscriptionOptions,
  SpeechSession,
  TranscriptResult,
} from '../../types/index.js';
import { mergeTranscriptResults } from './merge.js';

export interface LongAudioTranscriptionOptions extends BaseTranscriptionOptions {
  readonly chunkLengthSeconds?: number;
  readonly overlapSeconds?: number;
}

export class LongAudioCoordinator<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> {
  constructor(private readonly session: SpeechSession<TOptions, TNative>) {}

  async transcribe(
    input: AudioInputLike,
    options: LongAudioTranscriptionOptions = {},
  ): Promise<TranscriptResult> {
    const audio = normalizePcmInput(input);
    const chunkLengthSeconds = options.chunkLengthSeconds ?? 0;
    const overlapSeconds = options.overlapSeconds ?? 0;

    if (chunkLengthSeconds <= 0 || audio.durationSeconds <= chunkLengthSeconds) {
      return this.session.transcribe(audio, {
        ...(options as TOptions),
        responseFlavor: 'canonical',
      });
    }

    const chunks = chunkPcmAudio(
      audio,
      Math.max(1, Math.floor(chunkLengthSeconds * audio.sampleRate)),
      Math.max(0, Math.floor(overlapSeconds * audio.sampleRate)),
    );

    const results: TranscriptResult[] = [];
    for (const chunk of chunks) {
      const result = await this.session.transcribe(chunk, {
        ...(options as TOptions),
        timeOffsetSeconds: chunk.startTimeSeconds ?? 0,
        responseFlavor: 'canonical',
      });
      results.push(result);
    }

    return mergeTranscriptResults(results);
  }
}
