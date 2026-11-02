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
    publicHostedWeights: true,
    locator: 'huggingface',
    envSmokeFlag: 'GIGAAM_CTC_ONNX_SMOKE',
    factoryExport: 'createGigaAmCtcModelFamily',
    artifactLabel: 'GigaAM',
    languages: ['ru', 'en', 'kk', 'ky', 'uz'],
    audioContract: 'offline-ctc',
    limitations: [
      'Catalog entry is experimental and is not a verified preset.',
    ],
    notes: 'Experimental catalog entry and local adapter are available; verifiedPreset remains false. Official GigaAM multilingual CTC has a pinned Hub bundle and browser transcription verification.',
  },
  {
    family: 'gigaam-rnnt',
    modelIdHint: 'gigaam-v3-e2e-rnnt',
    status: 'experimental',
    verifiedPreset: false,
    publicHostedWeights: true,
    locator: 'huggingface',
    envSmokeFlag: 'GIGAAM_RNNT_ONNX_SMOKE',
    factoryExport: 'createGigaAmRnntModelFamily',
    artifactLabel: 'GigaAM RNN-T',
    languages: ['ru'],
    audioContract: 'offline-rnnt',
    limitations: [
      'Russian-only; do not mix with English JFK CTC claims.',
      'Catalog entry is experimental and is not a verified preset.',
    ],
    notes: 'Experimental catalog entry and local adapter are available; verifiedPreset remains false. Official Russian GigaAM v3 E2E RNN-T has a pinned Hub bundle. Encoder, decoder and joint come from model.to_onnx. Do not mix Russian-only output with English CTC claims.',
  },
  {
    family: 'sensevoice',
    modelIdHint: 'SenseVoiceSmall',
    status: 'experimental',
    verifiedPreset: false,
    publicHostedWeights: true,
    locator: 'huggingface',
    envSmokeFlag: 'SENSEVOICE_ONNX_SMOKE',
    factoryExport: 'createSenseVoiceModelFamily',
    artifactLabel: 'SenseVoice',
    languages: ['auto', 'zh', 'en', 'yue', 'ja', 'ko'],
    audioContract: 'offline-ctc',
    limitations: [
      'Catalog entry is experimental and is not a verified preset.',
    ],
    notes: 'Experimental catalog entry and local adapter are available; verifiedPreset remains false. Official FunAudioLLM SenseVoiceSmall export has a pinned Hub bundle and browser transcription verification.',
  },
  {
    family: 'x-asr',
    modelIdHint: 'x-asr-zh-en-160ms',
    status: 'experimental',
    verifiedPreset: false,
    publicHostedWeights: true,
    locator: 'huggingface',
    envSmokeFlag: 'XASR_ONNX_SMOKE',
    factoryExport: 'createXAsrModelFamily',
    artifactLabel: 'X-ASR',
    languages: ['zh', 'en'],
    audioContract: 'encoder-cache-streaming',
    limitations: [
      'True encoder-cache streaming, not silent window looping.',
      'Catalog entry is experimental and is not a verified preset.',
    ],
    notes: 'Experimental catalog entry and local adapter are available; verifiedPreset remains false. Official X-ASR Zipformer2 zh-en 160 ms deployment graphs have a pinned Hub bundle. The family provides true encoder-cache streaming.',
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
      'Short-clip/within-model-limit offline speech-LLM; no encoder-cache streaming or verified long-audio quality contract.',
      'Not a public preset; listSpeechModels() will not list this family.',
      'Requires a local ONNX directory; no hosted weights.',
    ],
    notes:
      'Official stacked Qwen3-ASR 0.6B. Default encoder is audio-encoder-dynamic.onnx with pad-to-100 and official token crop. Direct requests are validated through the 30-second model limit; generic window composition beyond that remains compatibility-only. Local ONNX only.',
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
  const sourceInstruction = descriptor?.publicHostedWeights
    ? 'Pass options.source with compatible local artifacts or a Hugging Face source. '
    : 'Pass options.source with a local ONNX directory (locator: local-onnx-dir). ';
  const message =
    `No ${label} artifact source is configured for "${id}". ` +
    `This family is experimental and is not a verified preset. ` +
    sourceInstruction +
    `Discover via listExperimentalSpeechFamilies() / getExperimentalSpeechFamily("${family}").` +
    languageText +
    audio +
    limitationText;
  return new ExperimentalArtifactMissingError(message, {
    family,
    modelId: id,
    locator: descriptor?.locator ?? 'local-onnx-dir',
    verifiedPreset: false,
    publicHostedWeights: descriptor?.publicHostedWeights ?? false,
    audioContract: descriptor?.audioContract,
    languages,
    limitations,
  });
}
