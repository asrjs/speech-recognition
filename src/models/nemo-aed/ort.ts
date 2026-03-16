import type {
  NemoAedArtifactSource,
  NemoAedDirectArtifacts,
  NemoAedExecutionBackend,
  NemoAedHuggingFaceSource,
  NemoAedPreprocessorBackend,
  NemoAedQuantization,
} from './types.js';
import {
  createOrtSession,
  initOrt,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from '../nemo-tdt/ort.js';
import { getDefaultNemoAedWeightSetup, normalizeNemoAedWeightBackend } from './weights.js';

export { createOrtSession, initOrt, type OrtModuleLike, type OrtSessionLike, type OrtTensorLike };

export interface ResolvedNemoAedArtifacts {
  readonly artifacts: NemoAedDirectArtifacts;
  readonly preprocessorBackend: NemoAedPreprocessorBackend;
  readonly warnings: readonly { readonly code: string; readonly message: string }[];
  readonly ortBackend: NemoAedExecutionBackend;
  readonly encoderBackendForOrt: NemoAedExecutionBackend;
  readonly decoderBackendForOrt: NemoAedExecutionBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

const QUANTIZATION_SUFFIX: Record<NemoAedQuantization, string> = {
  int8: '.int8.onnx',
  fp16: '.fp16.onnx',
  fp32: '.onnx',
};

function buildResolveUrl(repoId: string, revision: string, filename: string): string {
  const encodedRepo = repoId
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  const encodedRevision = encodeURIComponent(revision);
  const encodedFilename = filename
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  return `https://huggingface.co/${encodedRepo}/resolve/${encodedRevision}/${encodedFilename}`;
}

function getQuantizedFilename(baseName: string, quantization: NemoAedQuantization): string {
  return `${baseName}${QUANTIZATION_SUFFIX[quantization]}`;
}

function resolveQuantization(
  requested: NemoAedQuantization | undefined,
  backendForOrt: NemoAedExecutionBackend,
  role: 'encoder' | 'decoder',
): NemoAedQuantization {
  if (requested) {
    return requested;
  }
  const setup = getDefaultNemoAedWeightSetup(backendForOrt);
  return role === 'encoder' ? setup.encoderDefault : setup.decoderDefault;
}

function resolveComponentBackend(
  requested: NemoAedExecutionBackend | undefined,
  fallback: NemoAedExecutionBackend,
): NemoAedExecutionBackend {
  return requested ?? fallback;
}

function resolveHuggingFaceArtifacts(
  source: NemoAedHuggingFaceSource,
  backendId: string,
): ResolvedNemoAedArtifacts {
  const revision = source.revision ?? 'main';
  const fallbackBackend = normalizeNemoAedWeightBackend(backendId);
  const encoderBackendForOrt = resolveComponentBackend(source.encoderBackend, fallbackBackend);
  const decoderBackendForOrt = resolveComponentBackend(source.decoderBackend, fallbackBackend);
  const ortBackend =
    encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm';
  const encoderQuant = resolveQuantization(source.encoderQuant, encoderBackendForOrt, 'encoder');
  const decoderQuant = resolveQuantization(source.decoderQuant, decoderBackendForOrt, 'decoder');
  const encoderFilename = getQuantizedFilename('encoder-model', encoderQuant);
  const decoderFilename = getQuantizedFilename('decoder-model', decoderQuant);
  const preprocessorBackend = source.preprocessorBackend ?? 'onnx';
  const preprocessorFilename =
    preprocessorBackend === 'onnx' ? `${source.preprocessorName ?? 'nemo128'}.onnx` : undefined;
  const tokenizerFilename = source.tokenizerName ?? 'tokenizer.json';
  const configFilename = source.configName ?? 'config.json';

  return {
    artifacts: {
      encoderUrl: buildResolveUrl(source.repoId, revision, encoderFilename),
      decoderUrl: buildResolveUrl(source.repoId, revision, decoderFilename),
      tokenizerUrl: buildResolveUrl(source.repoId, revision, tokenizerFilename),
      configUrl: buildResolveUrl(source.repoId, revision, configFilename),
      preprocessorUrl: preprocessorFilename
        ? buildResolveUrl(source.repoId, revision, preprocessorFilename)
        : undefined,
      encoderFilename,
      decoderFilename,
      tokenizerFilename,
      configFilename,
      preprocessorFilename,
    },
    preprocessorBackend,
    warnings: [],
    ortBackend,
    encoderBackendForOrt,
    decoderBackendForOrt,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

function resolveDirectArtifacts(
  source: Extract<NemoAedArtifactSource, { kind: 'direct' }>,
  backendId: string,
): ResolvedNemoAedArtifacts {
  const fallbackBackend = normalizeNemoAedWeightBackend(backendId);
  const encoderBackendForOrt = resolveComponentBackend(source.encoderBackend, fallbackBackend);
  const decoderBackendForOrt = resolveComponentBackend(source.decoderBackend, fallbackBackend);
  return {
    artifacts: source.artifacts,
    preprocessorBackend: source.preprocessorBackend ?? 'onnx',
    warnings: [],
    ortBackend:
      encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm',
    encoderBackendForOrt,
    decoderBackendForOrt,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

export function resolveNemoAedArtifacts(
  source: NemoAedArtifactSource,
  backendId: string,
): ResolvedNemoAedArtifacts {
  return source.kind === 'huggingface'
    ? resolveHuggingFaceArtifacts(source, backendId)
    : resolveDirectArtifacts(source, backendId);
}
