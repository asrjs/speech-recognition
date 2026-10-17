import type {
  NemotronRnntArtifactSource,
  NemotronRnntDirectArtifactSource,
  NemotronRnntEncoderVariant,
  NemotronRnntExecutionBackend,
  NemotronRnntHuggingFaceSource,
  NemotronRnntStreamingWindowMs,
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
  readonly encoderVariants?: Readonly<
    Partial<Record<NemotronRnntStreamingWindowMs, NemotronRnntEncoderVariant>>
  >;
}

export interface ResolvedNemotronRnntOptions {
  readonly ortBackend: 'webgpu' | 'wasm';
  readonly encoderBackend: NemotronRnntExecutionBackend;
  readonly decoderBackend: NemotronRnntExecutionBackend;
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
  // Our repository keeps the default fp16 export at the root: encoder.onnx,
  // decoder.onnx, joint.onnx, vocab.txt, and external-data siblings.
  return {
    encoderUrl: buildResolveUrl(source.repoId, revision, 'encoder.onnx'),
    decoderUrl: buildResolveUrl(source.repoId, revision, 'decoder.onnx'),
    jointUrl: buildResolveUrl(source.repoId, revision, 'joint.onnx'),
    tokenizerUrl: buildResolveUrl(source.repoId, revision, 'vocab.txt'),
    encoderDataUrl: buildResolveUrl(source.repoId, revision, 'encoder.onnx.data'),
    decoderDataUrl: buildResolveUrl(source.repoId, revision, 'decoder.onnx.data'),
    jointDataUrl: buildResolveUrl(source.repoId, revision, 'joint.onnx.data'),
    encoderFilename: 'encoder.onnx',
    decoderFilename: 'decoder.onnx',
    jointFilename: 'joint.onnx',
    encoderVariants: {
      80: {
        url: buildResolveUrl(source.repoId, revision, 'windows/80/encoder.onnx'),
        dataUrl: buildResolveUrl(source.repoId, revision, 'windows/80/encoder.onnx.data'),
        filename: 'encoder.onnx',
        sourceFilename: 'windows/80/encoder.onnx',
      },
      160: {
        url: buildResolveUrl(source.repoId, revision, 'windows/160/encoder.onnx'),
        dataUrl: buildResolveUrl(source.repoId, revision, 'windows/160/encoder.onnx.data'),
        filename: 'encoder.onnx',
        sourceFilename: 'windows/160/encoder.onnx',
      },
      320: {
        url: buildResolveUrl(source.repoId, revision, 'windows/320/encoder.onnx'),
        dataUrl: buildResolveUrl(source.repoId, revision, 'windows/320/encoder.onnx.data'),
        filename: 'encoder.onnx',
        sourceFilename: 'windows/320/encoder.onnx',
      },
      1120: {
        url: buildResolveUrl(source.repoId, revision, 'windows/1120/encoder.onnx'),
        dataUrl: buildResolveUrl(source.repoId, revision, 'windows/1120/encoder.onnx.data'),
        filename: 'encoder.onnx',
        sourceFilename: 'windows/1120/encoder.onnx',
      },
    },
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
    encoderVariants: source.artifacts.encoderVariants,
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
  // Hybrid default (mirrors the parakeet-tdt family): the WebGPU encoder
  // pairs with WASM predictor/joint unless a decoder backend is requested.
  const decoderBackend = source.decoderBackend ?? 'wasm';
  const ortBackend: 'webgpu' | 'wasm' =
    encoderBackend === 'webgpu' || decoderBackend === 'webgpu' ? 'webgpu' : 'wasm';

  return {
    artifacts,
    options: {
      ortBackend,
      encoderBackend,
      decoderBackend,
      preprocessorBackend,
      wasmPaths: source.wasmPaths,
      cpuThreads: source.cpuThreads,
      enableProfiling: source.enableProfiling ?? false,
    },
  };
}
