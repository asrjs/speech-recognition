export * from './session.js';
export * from './types.js';

import { createQwen3AsrModelFamily } from '../qwen-asr/model.js';
import type { SpeechModelFactory } from '../../types/index.js';
import type {
  Qwen3AsrModelOptions,
  Qwen3AsrNativeTranscript,
  Qwen3AsrTranscriptionOptions,
} from '../qwen-asr/types.js';

export function createQwen3AsrVerifyModelFamily(): SpeechModelFactory<
  Qwen3AsrModelOptions,
  Qwen3AsrTranscriptionOptions,
  Qwen3AsrNativeTranscript
> {
  return createQwen3AsrModelFamily({ family: 'qwen3-asr' });
}
