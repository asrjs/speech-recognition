import type {
  NemotronRnntArtifactSource,
  NemotronRnntDirectArtifactSource,
  NemotronRnntExecutionBackend,
  NemotronRnntHuggingFaceSource,
} from './types.js';

/**
 * Resolved Nemotron RNNT artifact layout consumed by the executor. The
 * streaming pipeline needs three model files (encoder, predictor,
 * joint) plus the BPE vocab; external `.data` siblings are optional and
 * are passed through to ORT Web so single-file ONNX packages do not
 * need them.
 */
export interface ResolvedNemotronRnntArtifacts {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly jointUrl: string;
  readonly tokenizerUrl: string;
  readonly encoderDataUrl?: string;
  readonly decoderDataUrl?: string;
  readonly jointDataUrl?: string;
  readonly encoderFilename?: string;
  readonly decoderFilename?: string;
  readonly jointFilename?: string;
}

export interface ResolvedNemotronRnntOptions {
  readonly ortBackend: 'webgpu' | 'wasm';
  readonly encoderBackend: NemotronRnntExecutionBackend;
  readonly preprocessorBackend: 'js' | 'onnx';
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling: boolean;
}

function normalizeEncoderBackend(
  backendId: string,
  request?: NemotronRnntExecutionBackend,
): NemotronRnntExecutionBackend {
  const fallback: NemotronRnntExecutionBackend = backendId.startsWith('webgpu')
    ? 'webgpu'
    : 'wasm';
  return request ?? fallback;
}

function buildResolveUrl(repoId: string, revision: string, filename: string): string {
  return `https://huggingface.co/${repoId}/resolve/${revision}/${filename}`;
}

function resolveHuggingFaceArtifacts(
  source: NemotronRnntHuggingFaceSource,
): ResolvedNemotronRnntArtifacts {
  const revision = source.revision ?? 'main';
  return {
    encoderUrl: buildResolveUrl(source.repoId, revision, 'encoder-model.onnx'),
    decoderUrl: buildResolveUrl(source.repoId, revision, 'decoder_joint-model.onnx'),
    jointUrl: buildResolveUrl(source.repoId, revision, 'decoder_joint-model.onnx'),
    tokenizerUrl: buildResolveUrl(source.repoId, revision, 'vocab.txt'),
  };
}

function resolveDirectArtifacts(
  source: NemotronRnntDirectArtifactSource,
): ResolvedNemotronRnntArtifacts {
  return {
    encoderUrl: source.artifacts.encoderUrl,
    decoderUrl: source.artifacts.decoderUrl,
    jointUrl: source.artifacts.jointUrl,
    tokenizerUrl: source.artifacts.tokenizerUrl,
    encoderDataUrl: source.artifacts.encoderDataUrl,
    decoderDataUrl: source.artifacts.decoderDataUrl,
    jointDataUrl: source.artifacts.jointDataUrl,
    encoderFilename: source.artifacts.encoderFilename,
    decoderFilename: source.artifacts.decoderFilename,
    jointFilename: source.artifacts.jointFilename,
  };
}

export function resolveNemotronRnntArtifacts(
  source: NemotronRnntArtifactSource,
  backendId: string,
): { artifacts: ResolvedNemotronRnntArtifacts; options: ResolvedNemotronRnntOptions } {
  const artifacts =
    source.kind === 'huggingface'
      ? resolveHuggingFaceArtifacts(source)
      : resolveDirectArtifacts(source);

  const encoderBackend = normalizeEncoderBackend(backendId, source.encoderBackend);
  const preprocessorBackend = source.preprocessorBackend ?? 'js';
  const ortBackend: 'webgpu' | 'wasm' = encoderBackend;

  return {
    artifacts,
    options: {
      ortBackend,
      encoderBackend,
      preprocessorBackend,
      wasmPaths: source.wasmPaths,
      cpuThreads: source.cpuThreads,
      enableProfiling: source.enableProfiling ?? false,
    },
  };
}
