import type { AudioInputLike, BaseTranscriptionOptions, ModelInferenceLimits, TranscriptResult } from '../types/index.js';
import type { PcmAudioBuffer } from '../audio/index.js';
import type { PipelineStage } from './composition.js';
import { transcribeWithWindowing } from './long-audio-windowing.js';

export interface WindowingStageOptions<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
> {
  readonly id?: string;
  readonly inference?: ModelInferenceLimits;
  readonly transcribeWindow: (input: PcmAudioBuffer, options: TOptions) => Promise<TranscriptResult>;
  readonly transcribeDirect?: (input: AudioInputLike, options: TOptions) => Promise<TranscriptResult>;
}

export function createWindowingStage<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
>(options: WindowingStageOptions<TOptions>): PipelineStage<TOptions> {
  return {
    id: options.id ?? 'windowing',
    async run(context) {
      if (context.input === undefined || context.input === null) {
        throw new Error('Windowing stage requires context.input.');
      }

      const input = context.input as AudioInputLike;
      const transcriptionOptions = withContextSignal(context.options, context.signal) as TOptions;
      if (transcriptionOptions.windowing === 'disabled' && options.transcribeDirect) {
        return {
          transcript: await options.transcribeDirect(input, transcriptionOptions),
        };
      }

      return {
        transcript: await transcribeWithWindowing({
          input,
          inference: options.inference,
          options: transcriptionOptions,
          transcribeWindow: options.transcribeWindow,
        }),
      };
    },
  };
}

function withContextSignal<TOptions extends BaseTranscriptionOptions>(
  options: TOptions | undefined,
  signal: TOptions['signal'] | undefined,
): TOptions {
  return {
    ...(options ?? {}),
    signal: signal ?? options?.signal,
  } as TOptions;
}
