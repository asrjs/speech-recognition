/**
 * Artifact-gated experimental families that are registered on the runtime but
 * are not verified public presets. `listSpeechModels()` stays preset-only.
 *
 * Public `audioContract` / `limitations` types live in `src/types/experimental.ts`.
 * List/get return structured-clone copies so workers can postMessage them.
 */

import type { ExperimentalSpeechFamilyDescriptor } from '../types/experimental.js';
import { ExperimentalArtifactMissingError } from './errors.js';

export type {
  ExperimentalSpeechAudioContract,
  ExperimentalSpeechFamilyDescriptor,
  ExperimentalSpeechFamilyLocator,
  ExperimentalSpeechFamilyStatus,
} from '../types/experimental.js';

const EXPERIMENTAL_SPEECH_FAMILIES: readonly ExperimentalSpeechFamilyDescriptor[] = [
  {
    family: 'gigaam-ctc',
    modelIdHint: 'gigaam-multilingual-ctc',
    status: 'experimental',
    verifiedPreset: false,
    publicHostedWeights: false,
    locator: 'local-onnx-dir',
    envSmokeFlag: 'GIGAAM_CTC_ONNX_SMOKE',
    factoryExport: 'createGigaAmCtcModelFamily',
    artifactLabel: 'GigaAM',
    languages: ['ru', 'en', 'kk', 'ky', 'uz'],
    audioContract: 'offline-ctc',
    limitations: [
      'Not a public preset; listSpeechModels() will not list this family.',
      'Requires a local ONNX directory; no hosted weights.',
    ],
    notes: 'Official GigaAM multilingual CTC. Chrome WebGPU and WASM match the JFK oracle. Load with a local ONNX directory.',
  },
  {
    family: 'gigaam-rnnt',
    modelIdHint: 'gigaam-v3-e2e-rnnt',
    status: 'experimental',
    verifiedPreset: false,
    publicHostedWeights: false,
    locator: 'local-onnx-dir',
    envSmokeFlag: 'GIGAAM_RNNT_ONNX_SMOKE',
    factoryExport: 'createGigaAmRnntModelFamily',
    artifactLabel: 'GigaAM RNN-T',
    languages: ['ru'],
    audioContract: 'offline-rnnt',
    limitations: [
      'Russian-only; do not mix with English JFK CTC claims.',
      'Not a public preset; listSpeechModels() will not list this family.',
      'Requires a local ONNX directory; no hosted weights.',
    ],
    notes: 'Official GigaAM v3 E2E RNN-T. Russian-only; do not mix with JFK CTC claims. Encoder/decoder/joint from model.to_onnx. Chrome WebGPU and Node WASM match the official example.wav oracle. Local ONNX only.',
  },
  {
    family: 'sensevoice',
    modelIdHint: 'SenseVoiceSmall',
    status: 'experimental',
    verifiedPreset: false,
    publicHostedWeights: false,
    locator: 'local-onnx-dir',
    envSmokeFlag: 'SENSEVOICE_ONNX_SMOKE',
    factoryExport: 'createSenseVoiceModelFamily',
    artifactLabel: 'SenseVoice',
    languages: ['auto', 'zh', 'en', 'yue', 'ja', 'ko'],
    audioContract: 'offline-ctc',
    limitations: [
      'Not a public preset; listSpeechModels() will not list this family.',
      'Requires a local ONNX directory; no hosted weights.',
    ],
    notes: 'FunAudioLLM SenseVoiceSmall export. Chrome WebGPU and WASM match the JFK oracle. Local ONNX only.',
  },
  {
    family: 'x-asr',
    modelIdHint: 'x-asr-zh-en-160ms',
    status: 'experimental',
    verifiedPreset: false,
    publicHostedWeights: false,
    locator: 'local-onnx-dir',
    envSmokeFlag: 'XASR_ONNX_SMOKE',
    factoryExport: 'createXAsrModelFamily',
    artifactLabel: 'X-ASR',
    languages: ['zh', 'en'],
    audioContract: 'encoder-cache-streaming',
    limitations: [
      'True encoder-cache streaming, not silent window looping.',
      'Not a public preset; listSpeechModels() will not list this family.',
      'Requires a local ONNX directory; no hosted weights.',
    ],
    notes: 'Sherpa Zipformer2 streaming zh-en 160ms. Chrome WebGPU and WASM match the JFK oracle. True encoder-cache streaming; local ONNX only.',
  },
  {
    family: 'qwen-asr',
    modelIdHint: 'Qwen/Qwen3-ASR-0.6B',
    status: 'experimental',
    verifiedPreset: false,
    publicHostedWeights: false,
    locator: 'local-onnx-dir',
    envSmokeFlag: 'QWEN_OFFICIAL_ONNX_SMOKE',
    factoryExport: 'createQwen3AsrModelFamily',
    artifactLabel: 'Qwen3-ASR',
    languages: ['multilingual'],
    audioContract: 'short-clip-speech-llm',
    limitations: [
      'Short-clip offline speech-LLM, not encoder-cache streaming or long-audio windowing.',
      'Not a public preset; listSpeechModels() will not list this family.',
      'Requires a local ONNX directory; no hosted weights.',
    ],
    notes: 'Official stacked Qwen3-ASR 0.6B. Default encoder is audio-encoder-dynamic.onnx with pad-to-100 and official token crop. Short-clip offline speech-LLM; local ONNX only.',
  },
];

function freezeExperimentalSpeechFamilyDescriptor(
  descriptor: ExperimentalSpeechFamilyDescriptor,
): ExperimentalSpeechFamilyDescriptor {
  Object.freeze(descriptor.languages);
  Object.freeze(descriptor.limitations);
  return Object.freeze(descriptor);
}

const FROZEN_EXPERIMENTAL_SPEECH_FAMILIES: readonly ExperimentalSpeechFamilyDescriptor[] =
  Object.freeze(EXPERIMENTAL_SPEECH_FAMILIES.map(freezeExperimentalSpeechFamilyDescriptor));

function cloneExperimentalSpeechFamilyDescriptor(
  descriptor: ExperimentalSpeechFamilyDescriptor,
): ExperimentalSpeechFamilyDescriptor {
  return structuredClone(descriptor);
}

export function listExperimentalSpeechFamilies(): readonly ExperimentalSpeechFamilyDescriptor[] {
  return FROZEN_EXPERIMENTAL_SPEECH_FAMILIES.map(cloneExperimentalSpeechFamilyDescriptor);
}

export function getExperimentalSpeechFamily(
  familyOrHint: string,
): ExperimentalSpeechFamilyDescriptor | null {
  const needle = familyOrHint.trim().toLowerCase();
  if (!needle) return null;
  const found = FROZEN_EXPERIMENTAL_SPEECH_FAMILIES.find((entry) => (
    entry.family.toLowerCase() === needle
    || entry.modelIdHint.toLowerCase() === needle
    || entry.modelIdHint.toLowerCase().includes(needle)
  ));
  return found ? cloneExperimentalSpeechFamilyDescriptor(found) : null;
}

export function hasExperimentalArtifactSource(loadOptions: unknown): boolean {
  if (!loadOptions || typeof loadOptions !== 'object') return false;
  const source = (loadOptions as { readonly source?: unknown }).source;
  return source != null && typeof source === 'object';
}

export function createExperimentalArtifactMissingError(
  familyOrHint: string,
  modelId?: string,
): ExperimentalArtifactMissingError {
  const descriptor = getExperimentalSpeechFamily(familyOrHint);
  const family = descriptor?.family ?? familyOrHint;
  const id = modelId?.trim() || descriptor?.modelIdHint || family;
  const label = descriptor?.artifactLabel ?? family;
  const limitations = descriptor ? [...descriptor.limitations] : [];
  const languages = descriptor ? [...descriptor.languages] : [];
  const limitationText = limitations.length > 0 ? ` Limitations: ${limitations.join(' ')}` : '';
  const audio = descriptor ? ` Audio contract: ${descriptor.audioContract}.` : '';
  const languageText = languages.length > 0 ? ` Languages: ${languages.join(', ')}.` : '';
  const message =
    `No ${label} artifact source is configured for "${id}". ` +
    `This family is experimental and is not listed by listSpeechModels(). ` +
    `Pass options.source with a local ONNX directory (locator: local-onnx-dir). ` +
    `Discover via listExperimentalSpeechFamilies() / getExperimentalSpeechFamily("${family}").` +
    languageText +
    audio +
    limitationText;
  return new ExperimentalArtifactMissingError(message, {
    family,
    modelId: id,
    locator: descriptor?.locator ?? 'local-onnx-dir',
    verifiedPreset: false,
    publicHostedWeights: false,
    audioContract: descriptor?.audioContract,
    languages,
    limitations,
  });
}
