import {
  buildBuiltInHubLoadOptions,
  buildBuiltInTranscriptionOptions,
  detectBuiltInModelRepoQuantizations,
  getBuiltInModelDescriptor,
  getBuiltInModelLanguageName,
  listBuiltInModelDescriptors,
  listBuiltInModelOptions,
  listApprovedSpeechModelManifests as listBuiltInApprovedSpeechModelManifests,
  resolveApprovedSpeechExecutionPlan as resolveBuiltInApprovedSpeechExecutionPlan,
  resolveBuiltInModelComponentBackends,
  type BuildBuiltInHubLoadOptionsInput,
  type BuildBuiltInTranscriptionOptionsInput,
  type BuiltInDetectedQuantizations,
  type BuiltInExecutionBackend,
  type BuiltInModelDescriptor,
  type BuiltInModelOption,
  type ApprovedSpeechExecutionPlan,
  type ResolveSpeechExecutionPlanInput,
} from '../presets/descriptors.js';

export {
  createExperimentalArtifactMissingError,
  getExperimentalSpeechFamily,
  hasExperimentalArtifactSource,
  listExperimentalSpeechFamilies,
  type ExperimentalSpeechAudioContract,
  type ExperimentalSpeechFamilyDescriptor,
  type ExperimentalSpeechFamilyLocator,
  type ExperimentalSpeechFamilyStatus,
} from './experimental-families.js';

/** Root-friendly alias for built-in speech model metadata. */
export type SpeechModelDescriptor = BuiltInModelDescriptor;

/** Root-friendly alias for lightweight built-in speech model options. */
export type SpeechModelOption = BuiltInModelOption;

/** Root-friendly alias for built-in execution backends used by load helpers. */
export type SpeechExecutionBackend = BuiltInExecutionBackend;

/** Root-friendly alias for detected quantization metadata. */
export type SpeechModelQuantizations = BuiltInDetectedQuantizations;

export type SpeechModelExecutionPlan = ApprovedSpeechExecutionPlan;

/** Root-friendly alias for convenience load-option inputs. */
export type BuildSpeechModelLoadOptionsInput = BuildBuiltInHubLoadOptionsInput;

/** Root-friendly alias for convenience transcription-option inputs. */
export type BuildSpeechTranscriptionOptionsInput = BuildBuiltInTranscriptionOptionsInput;

/** Lists the library's built-in speech models with consolidated metadata. */
export function listSpeechModels(): readonly SpeechModelDescriptor[] {
  return listBuiltInModelDescriptors();
}

/** Lists lightweight built-in speech model options for selectors and menus. */
export function listSpeechModelOptions(): readonly SpeechModelOption[] {
  return listBuiltInModelOptions();
}

/** Lists approved immutable model artifacts and their execution profiles. */
export function listApprovedSpeechModelManifests(): readonly (SpeechModelExecutionPlan & { readonly modelId: string })[] {
  return listBuiltInApprovedSpeechModelManifests();
}

/** Selects a supported model execution plan and identifies exact approval matches. */
export function resolveSpeechModelExecutionPlan(
  modelId: string,
  input: ResolveSpeechExecutionPlanInput,
): SpeechModelExecutionPlan {
  return resolveBuiltInApprovedSpeechExecutionPlan(modelId, input);
}

/** Probes the built-in browser backends before selecting a manifest profile. */
export async function probeSpeechModelExecutionPlan(
  modelId: string,
  input: Omit<ResolveSpeechExecutionPlanInput, 'runtime'> = {},
): Promise<{
  readonly runtime: ResolveSpeechExecutionPlanInput['runtime'];
  readonly plan: SpeechModelExecutionPlan;
}> {
  const { probeWebGpuCapabilities, probeWasmCapabilities } = await import('../inference/backends/builtin.js');
  const [webgpu, wasm] = await Promise.all([probeWebGpuCapabilities(), probeWasmCapabilities()]);
  const runtime = {
    webgpu: webgpu.available,
    wasm: wasm.available,
    webgpuPrecisions: webgpu.supportedPrecisions as ('int8' | 'fp32' | 'fp16')[],
    // WASM's generic precision capabilities don't describe ONNX weight storage compatibility.
    wasmPrecisions: wasm.supportedPrecisions as ('int8' | 'fp32' | 'fp16')[],
  };
  return {
    runtime,
    plan: resolveBuiltInApprovedSpeechExecutionPlan(modelId, { ...input, runtime }),
  };
}

/** Resolves a built-in speech model descriptor by model id or known alias. */
export function getSpeechModelDescriptor(modelId: string): SpeechModelDescriptor | null {
  return getBuiltInModelDescriptor(modelId);
}

/** Resolves human-readable labels for common built-in speech model languages. */
export function getSpeechModelLanguageName(languageCode: string): string {
  return getBuiltInModelLanguageName(languageCode);
}

/**
 * Resolves model-specific component backends from a high-level backend request.
 *
 * This is useful when a UI or loader accepts broad choices such as
 * `webgpu-hybrid` but still needs the concrete encoder/decoder backend split.
 */
export function resolveSpeechModelComponentBackends(
  modelId: string,
  input: {
    readonly backend?: string;
    readonly encoderBackend?: SpeechExecutionBackend;
    readonly decoderBackend?: SpeechExecutionBackend;
  } = {},
): {
  readonly encoderBackend: SpeechExecutionBackend;
  readonly decoderBackend: SpeechExecutionBackend;
} {
  return resolveBuiltInModelComponentBackends(modelId, input);
}

/** Detects the quantization variants currently published for a built-in model repo revision. */
export async function detectSpeechModelRepoQuantizations(
  modelId: string,
  options: {
    readonly revision?: string;
  } = {},
): Promise<SpeechModelQuantizations> {
  return detectBuiltInModelRepoQuantizations(modelId, options);
}

/** Builds a preset-aware load request for the library's high-level runtime helpers. */
export function buildSpeechModelLoadOptions(
  input: BuildSpeechModelLoadOptionsInput,
): Record<string, unknown> {
  return buildBuiltInHubLoadOptions(input);
}

/** Builds preset-aware transcription options from common app-facing inputs. */
export function buildSpeechTranscriptionOptions(
  modelId: string,
  input: BuildSpeechTranscriptionOptionsInput = {},
): Record<string, unknown> {
  return buildBuiltInTranscriptionOptions(modelId, input);
}
