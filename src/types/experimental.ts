/**
 * Public discovery contracts for experimental (non-preset) speech families.
 *
 * These values are JSON / structured-clone-safe so they can be posted to
 * workers and browser threads. They are not a load API: callers still pass a
 * local ONNX directory via `options.source`.
 */

/** Discovery status. Experimental families are never listed by `listSpeechModels()`. */
export type ExperimentalSpeechFamilyStatus = 'experimental';

/**
 * How an experimental family consumes audio. This is a discovery hint, not a
 * streaming or windowing switch.
 *
 * - `offline-ctc`: full-utterance CTC
 * - `offline-rnnt`: full-utterance RNN-T
 * - `short-clip-speech-llm`: clipped encoder + speech-LLM (not long-audio windowing)
 * - `encoder-cache-streaming`: true encoder-cache streaming (not silent window looping)
 */
export type ExperimentalSpeechAudioContract =
  | 'offline-ctc'
  | 'offline-rnnt'
  | 'short-clip-speech-llm'
  | 'encoder-cache-streaming';

/** Experimental families are loaded from a caller-supplied local ONNX directory. */
export type ExperimentalSpeechFamilyLocator = 'local-onnx-dir';

/**
 * Clone-safe descriptor returned by `listExperimentalSpeechFamilies()` and
 * `getExperimentalSpeechFamily()`.
 *
 * `audioContract` and `limitations` are the fields UIs and workers should
 * surface before attempting a load. `verifiedPreset` and `publicHostedWeights`
 * are always false.
 */
export interface ExperimentalSpeechFamilyDescriptor {
  readonly family: string;
  readonly modelIdHint: string;
  readonly status: ExperimentalSpeechFamilyStatus;
  /** Always false: these are not listed by `listSpeechModels()`. */
  readonly verifiedPreset: false;
  /** Always false: callers must supply a local ONNX directory. */
  readonly publicHostedWeights: false;
  readonly locator: ExperimentalSpeechFamilyLocator;
  readonly envSmokeFlag: string;
  readonly factoryExport: string;
  /** Prefix used in missing-source errors (`No ${artifactLabel} artifact source...`). */
  readonly artifactLabel: string;
  readonly languages: readonly string[];
  readonly audioContract: ExperimentalSpeechAudioContract;
  /** Human-readable constraints (languages, clip length, not-a-preset, local ONNX). */
  readonly limitations: readonly string[];
  readonly notes: string;
}
