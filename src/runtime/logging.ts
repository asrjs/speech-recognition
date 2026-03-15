import type { RuntimeLogger, SpeechRuntimeHooks } from '../types/index.js';

export const defaultRuntimeLogger: RuntimeLogger = {
  debug: () => undefined,
  info: () => undefined,
  warn: () => undefined,
  error: () => undefined,
};

export function createRuntimeHooks(hooks?: SpeechRuntimeHooks): Required<SpeechRuntimeHooks> {
  return {
    logger: hooks?.logger ?? defaultRuntimeLogger,
    onProgress: hooks?.onProgress ?? (() => undefined),
  };
}
