import type {
  AbortSignalLike,
  AssetProvider,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptWarning,
} from '../../types/index.js';
import { rethrowIfAssetAborted } from '../../io/abort.js';
import { fetchModelFiles } from '../../runtime/huggingface.js';
import { nowMs } from '../../runtime/timing.js';
import { hasVerifiedWhisperDecoderAlignment } from './executor-alignment.js';
import {
  createWhisperGpuKvOutputLocation,
  roundMiB,
} from './executor-kv-cache.js';
import type { LoadedExecutorState } from './executor-types.js';
import {
  parseWhisperGenerationConfig,
  parseWhisperModelConfig,
  type WhisperGenerationConfig,
  type WhisperModelConfig,
} from './generation-config.js';
import {
  parseWhisperManifest,
  type WhisperAlignmentExportMetadata,
} from './manifest.js';
import {
  createWhisperOrtSession,
  initWhisperOrt,
  releaseOrtSession,
  resolveWhisperArtifacts,
  type OrtModuleLike,
  type OrtSessionLike,
  type ResolvedWhisperArtifacts,
} from './ort.js';
import { fetchText, WhisperTokenizer } from './tokenizer.js';
import type { WhisperArtifactSource } from './types.js';

export type WhisperOrtSessionOptions = Parameters<typeof createWhisperOrtSession>[2];

/**
 * Create an ORT session with an opt-in graph-capture request.
 *
 * Graph capture is deliberately diagnostic: ORT requires every graph node to
 * be partitioned to the requested execution provider, which is not true for
 * the current Whisper exports on all WebGPU bundles. Retrying without capture
 * keeps that experiment from turning into a model-load failure while allowing
 * the caller to surface the fallback as a recoverable warning.
 */
export async function createWhisperOrtSessionWithGraphCaptureFallback(
  ort: OrtModuleLike,
  url: string,
  options: WhisperOrtSessionOptions,
  onFallback?: (error: unknown) => void,
): Promise<OrtSessionLike> {
  try {
    return await createWhisperOrtSession(ort, url, options);
  } catch (error) {
    if (
      !options.enableGraphCapture ||
      !options.backendId.startsWith('webgpu') ||
      !isWhisperGraphCaptureUnavailableError(error)
    ) {
      throw error;
    }
    onFallback?.(error);
    return createWhisperOrtSession(ort, url, {
      ...options,
      enableGraphCapture: false,
    });
  }
}

function isWhisperGraphCaptureUnavailableError(error: unknown): boolean {
  const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
  return message.includes('graph capture') || message.includes('enablegraphcapture');
}

export function formatWhisperSessionError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.length > 240 ? `${message.slice(0, 237)}...` : message;
}

export function createAssetProgressEvent(
  modelId: string,
  file: string,
  event: { readonly loaded: number; readonly total?: number; readonly done?: boolean; readonly aborted?: boolean },
): RuntimeProgressEvent {
  const percent =
    event.total && event.total > 0
      ? Math.min(100, Math.round((event.loaded / event.total) * 100))
      : event.done && !event.aborted
        ? 100
        : undefined;
  return {
    phase: 'asset:download',
    modelId,
    file,
    loaded: event.loaded,
    total: event.total,
    percent,
    loadedMiB: roundMiB(event.loaded),
    totalMiB: roundMiB(event.total),
    isComplete: Boolean(event.done) && !event.aborted,
    aborted: event.aborted,
    message: event.aborted ? `Cancelled ${file}.` : event.done ? `Prepared ${file}.` : `Downloading ${file}.`,
  };
}

function isAssetMissingError(error: unknown): boolean {
  if (error instanceof Error) {
    return /\b404\b/.test(error.message) || /\bnot found\b/i.test(error.message);
  }
  return false;
}

function normalizeRepoPath(path: string): string {
  return String(path || '').replace(/^\.\/+/, '').replace(/\\/g, '/');
}

function hasListedRepoFile(files: readonly string[], filename: string): boolean {
  const target = normalizeRepoPath(filename);
  return files.some(
    (p) => normalizeRepoPath(p) === target || normalizeRepoPath(p).endsWith(`/${target}`),
  );
}

export interface MaterializeWhisperArtifactsContext {
  readonly modelId: string;
  readonly sourceOptions: WhisperArtifactSource | undefined;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
  readonly assetHandles: ResolvedAssetHandle[];
}

export async function materializeWhisperResolvedArtifacts(
  ctx: MaterializeWhisperArtifactsContext,
  resolved: ResolvedWhisperArtifacts,
): Promise<ResolvedWhisperArtifacts> {
  const source = ctx.sourceOptions;
  if (!ctx.assetProvider || !source) {
    return resolved;
  }

  const resolveRemoteUrl = async (
    url: string | undefined,
    fileLabel: string,
  ): Promise<string | undefined> => {
    if (!url || !/^https?:\/\//i.test(url)) {
      return url;
    }

    // Local browser demos serve very large ONNX external-data files from the
    // same origin. Hand those URLs directly to ORT instead of routing them
    // through IndexedDB/blob URLs, which is useful for remote assets but a
    // fragile extra hop for multi-GB local files.
    if (
      typeof window !== 'undefined' &&
      url.startsWith(`${window.location.origin}/models/`)
    ) {
      return url;
    }

    const handle = await ctx.assetProvider!.resolve({
      id: `url:${url}`,
      provider: 'url',
      url,
      preferBlobUrl: true,
      cacheKey: `url:${url}`,
      onProgress: (event) => {
        ctx.runtimeHooks?.onProgress?.(createAssetProgressEvent(ctx.modelId, fileLabel, event));
      },
    });
    ctx.assetHandles.push(handle);
    const locator = await handle.getLocator('url');
    if (!locator) {
      throw new Error(`Could not create a URL locator for "${fileLabel}".`);
    }
    return locator;
  };

  if (source.kind === 'splitgraph') {
    const materializeExternalData = async (
      externalData: ResolvedWhisperArtifacts['externalData'],
      labelPrefix = '',
    ): Promise<Record<string, { dataUrl: string; path: string }[]>> => {
      const materialized: Record<string, { dataUrl: string; path: string }[]> = {};
      for (const [graphName, entries] of Object.entries(externalData ?? {})) {
        const nextEntries: { dataUrl: string; path: string }[] = [];
        for (const entry of entries ?? []) {
          nextEntries.push({
            path: entry.path,
            dataUrl:
              (await resolveRemoteUrl(
                entry.dataUrl,
                `${labelPrefix}${graphName}/${entry.path.replace(/^\.\//, '')}`,
              )) ?? entry.dataUrl,
          });
        }
        if (nextEntries.length > 0) {
          materialized[graphName] = nextEntries;
        }
      }
      return materialized;
    };

    const materializedExternalData = await materializeExternalData(resolved.externalData);

    const encoderUrl =
      (await resolveRemoteUrl(resolved.artifacts.encoderUrl, 'encoder_model.onnx')) ??
      resolved.artifacts.encoderUrl;
    const decoderInitUrl =
      (await resolveRemoteUrl(resolved.decoderInitUrl, 'decoder_init.onnx')) ??
      resolved.decoderInitUrl;
    const decoderStepUrl =
      (await resolveRemoteUrl(resolved.decoderStepUrl, 'decoder_step.onnx')) ??
      resolved.decoderStepUrl;
    const decoderAlignUrl =
      (await resolveRemoteUrl(resolved.decoderAlignUrl, 'decoder_align.onnx')) ??
      resolved.decoderAlignUrl;
    const manifestUrl =
      (await resolveRemoteUrl(resolved.manifestUrl, 'manifest.json')) ??
      resolved.manifestUrl;

    let alignmentReference: ResolvedWhisperArtifacts['alignmentReference'];
    if (resolved.alignmentReference) {
      const referenceExternalData = await materializeExternalData(
        resolved.alignmentReference.externalData,
        'alignment-reference/',
      );
      alignmentReference = {
        ...resolved.alignmentReference,
        encoderUrl:
          (await resolveRemoteUrl(
            resolved.alignmentReference.encoderUrl,
            'alignment-reference/encoder_model.onnx',
          )) ?? resolved.alignmentReference.encoderUrl,
        decoderAlignUrl:
          (await resolveRemoteUrl(
            resolved.alignmentReference.decoderAlignUrl,
            'alignment-reference/decoder_align.onnx',
          )) ?? resolved.alignmentReference.decoderAlignUrl,
        manifestUrl:
          (await resolveRemoteUrl(
            resolved.alignmentReference.manifestUrl,
            'alignment-reference/manifest.json',
          )) ?? resolved.alignmentReference.manifestUrl,
        externalData:
          Object.keys(referenceExternalData).length > 0
            ? {
                ...(referenceExternalData.encoder
                  ? { encoder: referenceExternalData.encoder }
                  : {}),
                ...(referenceExternalData.decoder_align
                  ? { decoder_align: referenceExternalData.decoder_align }
                  : {}),
              }
            : undefined,
      };
    }

    return {
      ...resolved,
      artifacts: {
        ...resolved.artifacts,
        encoderUrl,
        decoderUrl: decoderInitUrl ?? resolved.artifacts.decoderUrl,
        tokenizerUrl: resolved.artifacts.tokenizerUrl,
      },
      decoderInitUrl,
      decoderStepUrl,
      decoderAlignUrl,
      manifestUrl,
      alignmentReference,
      externalData:
        Object.keys(materializedExternalData).length > 0
          ? materializedExternalData
          : undefined,
    };
  }

  if (source.kind !== 'huggingface') {
    return resolved;
  }

  const revision = source.revision ?? 'main';
  let repoFilesPromise: Promise<readonly string[]> | null = null;
  const getRepoFiles = (): Promise<readonly string[]> => {
    repoFilesPromise ??= fetchModelFiles(source.repoId, revision);
    return repoFilesPromise;
  };

  const resolveFile = async (filename: string | undefined): Promise<string | undefined> => {
    if (!filename) return undefined;
    const cacheKey = `huggingface:${source.repoId}:${revision}:${filename}`;
    const cacheKeyFallbacks = (source.cacheKeyFallbackRevisions ?? [])
      .filter((fb) => fb !== revision)
      .map((fb) => `huggingface:${source.repoId}:${fb}:${filename}`);

    const handle = await ctx.assetProvider!.resolve({
      id: `huggingface:${source.repoId}:${revision}:${filename}`,
      provider: 'huggingface',
      repoId: source.repoId,
      revision,
      filename,
      preferBlobUrl: true,
      cacheKey,
      cacheKeyFallbacks,
      onProgress: (event) => {
        ctx.runtimeHooks?.onProgress?.(createAssetProgressEvent(ctx.modelId, filename, event));
      },
    });
    ctx.assetHandles.push(handle);
    const locator = await handle.getLocator('url');
    if (!locator) {
      throw new Error(`Could not create a URL locator for "${filename}".`);
    }
    return locator;
  };

  const resolveOptionalFile = async (filename: string | undefined): Promise<string | undefined> => {
    if (!filename) return undefined;
    const repoFiles = await getRepoFiles();
    if (repoFiles.length > 0 && !hasListedRepoFile(repoFiles, filename)) {
      return undefined;
    }
    try {
      return await resolveFile(filename);
    } catch (error) {
      if (isAssetMissingError(error)) return undefined;
      throw error;
    }
  };

  const resolveArtifactFile = async (url: string): Promise<string> => {
    const repoPath = source.repoId.split('/').map(encodeURIComponent).join('/');
    const resolvePrefix = `/${repoPath}/resolve/${encodeURIComponent(revision)}/`;
    const pathname = new URL(url).pathname;
    const filename = pathname.startsWith(resolvePrefix)
      ? pathname.slice(resolvePrefix.length).split('/').map(decodeURIComponent).join('/')
      : pathname.split('/').pop();
    return (await resolveFile(filename)) ?? url;
  };

  void resolveOptionalFile;

  return {
    ...resolved,
    artifacts: {
      ...resolved.artifacts,
      encoderUrl: await resolveArtifactFile(resolved.artifacts.encoderUrl),
      decoderUrl: await resolveArtifactFile(resolved.artifacts.decoderUrl),
      tokenizerUrl: resolved.artifacts.tokenizerUrl,
    },
  };
}

export async function loadWhisperGenerationConfig(
  artifacts: { readonly tokenizerUrl: string; readonly generationConfigUrl?: string },
  signal?: AbortSignalLike | null,
): Promise<WhisperGenerationConfig> {
  try {
    const genConfigUrl = artifacts.generationConfigUrl ??
      artifacts.tokenizerUrl.replace(/tokenizer\.json$/, 'generation_config.json');
    const text = await fetchText(genConfigUrl, signal);
    const json = JSON.parse(text) as Record<string, unknown>;
    return parseWhisperGenerationConfig(json);
  } catch (error) {
    rethrowIfAssetAborted(error);
    return parseWhisperGenerationConfig({});
  }
}

export async function loadWhisperDecoderAlignMetadata(
  manifestUrl: string | undefined,
  signal?: AbortSignalLike | null,
): Promise<WhisperAlignmentExportMetadata | undefined> {
  if (!manifestUrl) return undefined;
  try {
    const raw = JSON.parse(await fetchText(manifestUrl, signal)) as Record<string, unknown>;
    return parseWhisperManifest(raw).alignmentExport;
  } catch (error) {
    rethrowIfAssetAborted(error);
    return undefined;
  }
}

export async function loadWhisperModelConfig(
  artifacts: {
    readonly tokenizerUrl: string;
    readonly configUrl?: string;
    readonly generationConfigUrl?: string;
  },
  signal?: AbortSignalLike | null,
): Promise<WhisperModelConfig> {
  try {
    const configUrl = artifacts.configUrl ?? artifacts.tokenizerUrl.replace(/tokenizer\.json$/, 'config.json');
    const text = await fetchText(configUrl, signal);
    const json = JSON.parse(text) as Record<string, unknown>;
    const config = parseWhisperModelConfig(json);

    if (!config.numMelBins) {
      try {
        const genUrl = artifacts.generationConfigUrl ??
          artifacts.tokenizerUrl.replace(/tokenizer\.json$/, 'generation_config.json');
        const genText = await fetchText(genUrl, signal);
        const genJson = JSON.parse(genText) as Record<string, unknown>;
        if (typeof genJson.num_mel_bins === 'number') {
          return { ...config, numMelBins: genJson.num_mel_bins };
        }
      } catch (error) {
        rethrowIfAssetAborted(error);
      }
    }
    return config;
  } catch (error) {
    rethrowIfAssetAborted(error);
    return parseWhisperModelConfig({});
  }
}

export interface InitializeWhisperExecutorContext {
  readonly modelId: string;
  readonly backendId: string;
  readonly sourceOptions: WhisperArtifactSource | undefined;
  readonly signal?: AbortSignalLike | null;
  readonly isDisposed: () => boolean;
  readonly materializeResolvedArtifacts: (
    resolved: ResolvedWhisperArtifacts,
  ) => Promise<ResolvedWhisperArtifacts>;
}

export async function initializeWhisperExecutorState(
  ctx: InitializeWhisperExecutorContext,
): Promise<LoadedExecutorState> {
  if (!ctx.sourceOptions) {
    throw new Error(`No artifact source is configured for "${ctx.modelId}".`);
  }

  const resolved = await ctx.materializeResolvedArtifacts(
    resolveWhisperArtifacts(ctx.sourceOptions, ctx.backendId),
  );
  const artifacts = resolved.artifacts;

  const ort = await initWhisperOrt(resolved.ortBackend, {
    wasmPaths: resolved.wasmPaths,
    cpuThreads: resolved.cpuThreads,
    enableProfiling: resolved.enableProfiling,
    signal: ctx.signal,
  });

  const warnings: TranscriptWarning[] = [...resolved.warnings];

  const tokenizerPromise = WhisperTokenizer.fromUrl(artifacts.tokenizerUrl, ctx.signal);
  const genConfigPromise = loadWhisperGenerationConfig(artifacts, ctx.signal);
  const modelConfigPromise = loadWhisperModelConfig(artifacts, ctx.signal);
  const decoderAlignMetadataPromise = loadWhisperDecoderAlignMetadata(resolved.manifestUrl, ctx.signal);
  const alignmentReferenceMetadataPromise = resolved.alignmentReference
    ? loadWhisperDecoderAlignMetadata(resolved.alignmentReference.manifestUrl, ctx.signal)
    : Promise.resolve(undefined);

  const sessionStart = nowMs();
  const encoderSessionOptions: WhisperOrtSessionOptions = {
    backendId: resolved.encoderBackendForOrt,
    enableProfiling: resolved.enableProfiling,
    enableGraphCapture: resolved.experimentalWebGpuEncoderGraphCapture,
    ...((resolved.experimentalGpuKvCache && resolved.encoderBackendForOrt === 'webgpu' && !resolved.encoderOutputCpu)
      ? { preferredOutputLocation: 'gpu-buffer' as const }
      : {}),
    ...(resolved.externalData?.encoder
      ? { externalData: resolved.externalData.encoder }
      : {}),
  };
  const encoderSessionPromise = createWhisperOrtSessionWithGraphCaptureFallback(
    ort,
    artifacts.encoderUrl,
    encoderSessionOptions,
    resolved.experimentalWebGpuEncoderGraphCapture && resolved.encoderBackendForOrt === 'webgpu'
      ? (error) => warnings.push({
          code: 'whisper.encoder-graph-capture-fallback',
          message: `WebGPU encoder graph capture was unavailable; using the regular encoder session (${formatWhisperSessionError(error)}).`,
          recoverable: true,
        })
      : undefined,
  );

  const isSplitGraph = resolved.isSplitGraph;

  const [tokenizer, genConfig, modelConfig, decoderAlignMetadata, alignmentReferenceMetadata, encoderSession] = await Promise.all([
    tokenizerPromise,
    genConfigPromise,
    modelConfigPromise,
    decoderAlignMetadataPromise,
    alignmentReferenceMetadataPromise,
    encoderSessionPromise,
  ]);
  if (ctx.isDisposed()) {
    releaseOrtSession(encoderSession);
    throw new Error(`Whisper executor was disposed during load for "${ctx.modelId}".`);
  }
  const decoderAlignCausalSelfAttention = decoderAlignMetadata?.causalSelfAttention;
  if (
    resolved.decoderAlignUrl &&
    !hasVerifiedWhisperDecoderAlignment(decoderAlignCausalSelfAttention) &&
    !resolved.alignmentReference
  ) {
    warnings.push({
      code: 'whisper.decoder-align-legacy',
      message:
        'decoder_align is missing the causal-self-attention export marker; using generated timestamp interpolation until the artifact is re-exported.',
      recoverable: true,
    });
  }

  if (resolved.alignmentReference && !hasVerifiedWhisperDecoderAlignment(alignmentReferenceMetadata?.causalSelfAttention)) {
    warnings.push({
      code: 'whisper.alignment-reference-invalid',
      message:
        'The alignment-reference decoder_align artifact is missing the causal-self-attention marker; reference word timestamps remain unavailable.',
      recoverable: true,
    });
  }

  let decoderSession: OrtSessionLike | undefined;
  let decoderInitSession: OrtSessionLike | undefined;
  let decoderStepSession: OrtSessionLike | undefined;
  let decoderAlignSession: OrtSessionLike | undefined;
  const decoderInitPreferredOutputLocation =
    (resolved.experimentalGpuKvCache || resolved.experimentalGpuKvBeam) && resolved.decoderBackendForOrt === 'webgpu'
      ? createWhisperGpuKvOutputLocation(modelConfig, 'init')
      : undefined;
  const decoderStepPreferredOutputLocation =
    (resolved.experimentalGpuKvCache || resolved.experimentalGpuKvBeam) && resolved.decoderBackendForOrt === 'webgpu'
      ? createWhisperGpuKvOutputLocation(modelConfig, 'step')
      : undefined;

  if (isSplitGraph && resolved.decoderInitUrl && resolved.decoderStepUrl) {
    decoderInitSession = await createWhisperOrtSession(ort, resolved.decoderInitUrl, {
      backendId: resolved.decoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      preferredOutputLocation: decoderInitPreferredOutputLocation,
      ...(resolved.externalData?.decoder_init
        ? { externalData: resolved.externalData.decoder_init }
        : {}),
    });
    if (ctx.isDisposed()) {
      releaseOrtSession(encoderSession);
      releaseOrtSession(decoderInitSession);
      throw new Error(`Whisper executor was disposed during load for "${ctx.modelId}".`);
    }
    const decoderStepSessionOptions: WhisperOrtSessionOptions = {
      backendId: resolved.decoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      preferredOutputLocation: decoderStepPreferredOutputLocation,
      ...(resolved.decoderGraphCapture ? { enableGraphCapture: true } : {}),
      ...(resolved.decoderFreeDimensionOverrides ? { freeDimensionOverrides: resolved.decoderFreeDimensionOverrides } : {}),
      ...(resolved.externalData?.decoder_step
        ? { externalData: resolved.externalData.decoder_step }
        : {}),
    };
    decoderStepSession = await createWhisperOrtSessionWithGraphCaptureFallback(
      ort,
      resolved.decoderStepUrl,
      decoderStepSessionOptions,
      resolved.decoderGraphCapture && resolved.decoderBackendForOrt === 'webgpu'
        ? (error) => warnings.push({
            code: 'whisper.decoder-step-graph-capture-fallback',
            message: `WebGPU decoder_step graph capture was unavailable; using the regular decoder_step session (${formatWhisperSessionError(error)}).`,
            recoverable: true,
          })
        : undefined,
    );
    if (ctx.isDisposed()) {
      releaseOrtSession(encoderSession);
      releaseOrtSession(decoderInitSession);
      releaseOrtSession(decoderStepSession);
      throw new Error(`Whisper executor was disposed during load for "${ctx.modelId}".`);
    }
  } else {
    decoderSession = await createWhisperOrtSession(ort, artifacts.decoderUrl, {
      backendId: resolved.decoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      ...(resolved.externalData?.decoder_init
        ? { externalData: resolved.externalData.decoder_init }
        : {}),
    });
    if (ctx.isDisposed()) {
      releaseOrtSession(encoderSession);
      releaseOrtSession(decoderSession);
      throw new Error(`Whisper executor was disposed during load for "${ctx.modelId}".`);
    }
  }

  return {
    ort, tokenizer, encoderSession, decoderSession,
    generationConfig: genConfig, modelConfig, warnings,
    isSplitGraph, decoderInitSession, decoderStepSession, decoderAlignSession,
    decoderAlignUrl: resolved.decoderAlignUrl,
    decoderAlignCausalSelfAttention,
    decoderAlignAttentionValues: decoderAlignMetadata?.attentionValues,
    decoderAlignAttentionLayout: decoderAlignMetadata?.attentionLayout,
    enableProfiling: resolved.enableProfiling,
    decoderAlignExternalData: resolved.externalData?.decoder_align,
    alignmentReference: resolved.alignmentReference
      ? {
          encoderUrl: resolved.alignmentReference.encoderUrl,
          decoderAlignUrl: resolved.alignmentReference.decoderAlignUrl,
          manifestUrl: resolved.alignmentReference.manifestUrl,
          backendForOrt: resolved.alignmentReference.backendForOrt,
          encoderExternalData: resolved.alignmentReference.externalData?.encoder,
          decoderAlignExternalData: resolved.alignmentReference.externalData?.decoder_align,
          decoderAlignCausalSelfAttention: alignmentReferenceMetadata?.causalSelfAttention,
          decoderAlignAttentionValues: alignmentReferenceMetadata?.attentionValues,
          decoderAlignAttentionLayout: alignmentReferenceMetadata?.attentionLayout,
        }
      : undefined,
    decoderBackendForOrt: resolved.decoderBackendForOrt,
    experimentalGpuKvCache: resolved.experimentalGpuKvCache,
    experimentalGpuKvBeam: resolved.experimentalGpuKvBeam,
    encoderBufferRewrap: resolved.encoderBufferRewrap,
    encoderGpuFlush: resolved.encoderGpuFlush,
    encoderGpuDrain: resolved.encoderGpuDrain,
    sessionCreateMs: nowMs() - sessionStart,
  };
}
