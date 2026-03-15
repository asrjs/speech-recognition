import {
  loadBuiltInSpeechModel,
  type BuiltInSpeechModelHandle,
  type LoadBuiltInSpeechModelOptions,
} from './builtins.js';
import type { BaseTranscriptionOptions } from '../types/index.js';

/**
 * App-facing convenience handle returned by `loadSpeechModel()`.
 *
 * This is a thin alias over the built-in model handle so consumers can start
 * from the root package without learning the built-ins namespace first.
 */
export type LoadedSpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> = BuiltInSpeechModelHandle<TLoadOptions, TTranscriptionOptions, TNative>;

/** Root-level convenience options for loading a built-in speech model. */
export type LoadSpeechModelOptions<TLoadOptions = unknown> =
  LoadBuiltInSpeechModelOptions<TLoadOptions>;

/**
 * Loads a built-in speech model, creates a ready session, and returns a small
 * handle with `transcribe()` and `dispose()`.
 *
 * Advanced callers can still pass an explicit runtime, or bypass this helper
 * entirely and use `createSpeechRuntime().loadModel(...)` directly.
 */
export async function loadSpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
>(
  options: LoadSpeechModelOptions<TLoadOptions>,
): Promise<LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>> {
  return loadBuiltInSpeechModel<TLoadOptions, TTranscriptionOptions, TNative>(options);
}
