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
import { roundMetric } from '../../runtime/timing.js';
import {
  createOrtSession,
  initOrt,
  releaseOrtSession,
  resolveNemoTdtArtifacts,
  type OrtModuleLike,
  type OrtOutputLocation,
  type OrtSessionLike,
} from './ort.js';
import { JsNemoPreprocessor, type NemoPreprocessor, OnnxNemoPreprocessor } from './preprocessor.js';
import { ParakeetTokenizer } from './tokenizer.js';
import type { NemoTdtArtifactSource, NemoTdtModelConfig } from './types.js';
import { getDefaultNemoTdtWeightSetup } from './weights.js';

export interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: ParakeetTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly decoderSession: OrtSessionLike;
  readonly preprocessor: NemoPreprocessor;
  readonly preprocessorBackend: string;
  readonly encoderBackend: string;
  readonly decoderBackend: string;
  readonly warnings: readonly TranscriptWarning[];
}

function roundMiB(bytes: number | undefined): number | undefined {
  if (!Number.isFinite(bytes)) {
    return undefined;
  }
  return roundMetric((bytes as number) / (1024 * 1024), 2);
}

export function createAssetProgressEvent(
  modelId: string,
  file: string,
  event: {
    readonly loaded: number;
    readonly total?: number;
    readonly done?: boolean;
    readonly aborted?: boolean;
  },
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
    message: event.aborted
      ? `Cancelled ${file}.`
      : event.done
        ? `Prepared ${file}.`
        : `Downloading ${file}.`,
  };
}

export function normalizeRepoPath(path: string): string {
  return String(path || '')
    .replace(/^\.\/+/, '')
    .replace(/\\/g, '/');
}

export function hasListedRepoFile(files: readonly string[], filename: string): boolean {
  const target = normalizeRepoPath(filename);
  return files.some(
    (path) => normalizeRepoPath(path) === target || normalizeRepoPath(path).endsWith(`/${target}`),
  );
}

export function isExternalDataLoadError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  return /failed to load external data file/i.test(error.message);
}

export interface MaterializeNemoTdtArtifactsContext {
  readonly modelId: string;
  readonly sourceOptions: NemoTdtArtifactSource | undefined;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
  readonly assetHandles: ResolvedAssetHandle[];
}

export async function materializeNemoTdtHuggingFaceArtifacts(
  ctx: MaterializeNemoTdtArtifactsContext,
  artifacts: ReturnType<typeof resolveNemoTdtArtifacts>['artifacts'],
): Promise<typeof artifacts> {
  const source = ctx.sourceOptions;
  if (!ctx.assetProvider || !source || source.kind !== 'huggingface') {
    return artifacts;
  }

  const revision = source.revision ?? 'main';
  let repoFilesPromise: Promise<readonly string[]> | null = null;
  const getRepoFiles = (): Promise<readonly string[]> => {
    if (source.repoFiles && source.repoFiles.length > 0) {
      return Promise.resolve(source.repoFiles);
    }
    repoFilesPromise ??= fetchModelFiles(source.repoId, revision);
    return repoFilesPromise;
  };

  const resolveFile = async (filename: string | undefined): Promise<string | undefined> => {
    if (!filename) {
      return undefined;
    }
    const cacheKey = `huggingface:${source.repoId}:${revision}:${filename}`;
    const cacheKeyFallbacks = (source.cacheKeyFallbackRevisions ?? [])
      .filter((fallbackRevision) => fallbackRevision !== revision)
      .map((fallbackRevision) => `huggingface:${source.repoId}:${fallbackRevision}:${filename}`);

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

  const resolveOptionalFile = async (
    filename: string | undefined,
  ): Promise<string | undefined> => {
    if (!filename) {
      return undefined;
    }
    const repoFiles = await getRepoFiles();
    if (repoFiles.length > 0 && !hasListedRepoFile(repoFiles, filename)) {
      return undefined;
    }
    try {
      return await resolveFile(filename);
    } catch (error) {
      // The sidecar is optional. Only cancellation propagates; a 404, an
      // offline listing or a transport failure all mean "not available"
      // so cached graphs keep loading without network access.
      rethrowIfAssetAborted(error);
      return undefined;
    }
  };

  const preprocessorFilename = artifacts.preprocessorUrl
    ? artifacts.preprocessorUrl.split('/').pop()
    : undefined;

  return {
    ...artifacts,
    encoderUrl: (await resolveFile(artifacts.encoderFilename)) ?? artifacts.encoderUrl,
    decoderUrl: (await resolveFile(artifacts.decoderFilename)) ?? artifacts.decoderUrl,
    tokenizerUrl: (await resolveFile('vocab.txt')) ?? artifacts.tokenizerUrl,
    preprocessorUrl: (await resolveFile(preprocessorFilename)) ?? artifacts.preprocessorUrl,
    encoderDataUrl:
      artifacts.encoderDataUrl ??
      (artifacts.encoderFilename
        ? await resolveOptionalFile(`${artifacts.encoderFilename}.data`)
        : undefined),
    decoderDataUrl:
      artifacts.decoderDataUrl ??
      (artifacts.decoderFilename
        ? await resolveOptionalFile(`${artifacts.decoderFilename}.data`)
        : undefined),
  };
}

export interface InitializeNemoTdtExecutorContext {
  readonly modelId: string;
  readonly backendId: string;
  readonly config: NemoTdtModelConfig;
  readonly sourceOptions: NemoTdtArtifactSource | undefined;
  readonly runtimeHooks?: SpeechRuntimeHooks;
  readonly signal?: AbortSignalLike | null;
  readonly isDisposed: () => boolean;
  readonly materializeArtifacts: (
    artifacts: ReturnType<typeof resolveNemoTdtArtifacts>['artifacts'],
  ) => Promise<ReturnType<typeof resolveNemoTdtArtifacts>['artifacts']>;
}

export async function initializeNemoTdtExecutorState(
  ctx: InitializeNemoTdtExecutorContext,
): Promise<LoadedExecutorState> {
  if (!ctx.sourceOptions) {
    throw new Error(`No artifact source is configured for "${ctx.modelId}".`);
  }

  const resolved = resolveNemoTdtArtifacts(ctx.sourceOptions, ctx.backendId);
  let artifacts = await ctx.materializeArtifacts(resolved.artifacts);
  if (resolved.preprocessorBackend === 'onnx' && !artifacts.preprocessorUrl) {
    throw new Error(
      `The NeMo TDT source for "${ctx.modelId}" does not provide a preprocessor model.`,
    );
  }

  const ort = await initOrt(resolved.ortBackend, {
    wasmPaths: resolved.wasmPaths,
    cpuThreads: resolved.cpuThreads,
    signal: ctx.signal,
  });
  const tokenizer = await ParakeetTokenizer.fromUrl(artifacts.tokenizerUrl, {
    signal: ctx.signal,
  });
  const warnings = resolved.warnings.map((warning) => ({
    ...warning,
    recoverable: true,
  }));
  let encoderSession: OrtSessionLike | undefined;
  try {
    encoderSession = await createOrtSession(ort, artifacts.encoderUrl, {
      backendId: resolved.encoderBackendForOrt,
      cpuThreads: resolved.cpuThreads,
      signal: ctx.signal,
      enableProfiling: resolved.enableProfiling,
      webgpuOptions: resolved.webgpuOptions,
      externalDataUrl: artifacts.encoderDataUrl,
      externalDataPath: artifacts.encoderFilename
        ? `${artifacts.encoderFilename}.data`
        : undefined,
    });
  } catch (error) {
    rethrowIfAssetAborted(error);
    const implicitFp16Encoder =
      ctx.sourceOptions.kind === 'huggingface' &&
      !ctx.sourceOptions.encoderQuant &&
      resolved.encoderBackendForOrt === 'webgpu' &&
      getDefaultNemoTdtWeightSetup(resolved.encoderBackendForOrt).encoderDefault === 'fp16';

    if (!implicitFp16Encoder) {
      throw error;
    }

    const fallbackCandidates: ReadonlyArray<{
      quant: 'fp32' | 'int8';
      warningCode: string;
      warningMessage: string;
    }> = [
      {
        quant: 'fp32',
        warningCode: 'nemo-tdt.encoder-fp16-fallback-fp32',
        warningMessage:
          'Default FP16 encoder weights could not be initialized on this WebGPU setup. Falling back to FP32 encoder weights.',
      },
      {
        quant: 'int8',
        warningCode: 'nemo-tdt.encoder-fp16-fallback-int8',
        warningMessage:
          'Default FP16 encoder weights could not be initialized on this WebGPU setup. Falling back to INT8 encoder weights.',
      },
    ];

    let fallbackError: unknown = error;
    let fallbackSucceeded = false;

    for (const candidate of fallbackCandidates) {
      const fallbackResolved = resolveNemoTdtArtifacts(
        {
          ...ctx.sourceOptions,
          encoderQuant: candidate.quant,
        },
        ctx.backendId,
      );
      const cause =
        fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
      const warning = {
        code: candidate.warningCode,
        message: `${candidate.warningMessage} Cause: ${cause}`,
        recoverable: true as const,
      } satisfies TranscriptWarning;
      ctx.runtimeHooks?.onProgress?.({
        phase: 'model-load:fallback',
        modelId: ctx.modelId,
        backendId: resolved.encoderBackendForOrt,
        file: fallbackResolved.artifacts.encoderFilename,
        message: warning.message,
        warning,
      });
      const fallbackArtifacts = await ctx.materializeArtifacts(fallbackResolved.artifacts);
      artifacts = {
        ...artifacts,
        encoderUrl: fallbackArtifacts.encoderUrl,
        encoderDataUrl: fallbackArtifacts.encoderDataUrl,
        encoderFilename: fallbackArtifacts.encoderFilename,
      };

      try {
        encoderSession = await createOrtSession(ort, artifacts.encoderUrl, {
          backendId: resolved.encoderBackendForOrt,
          cpuThreads: resolved.cpuThreads,
          signal: ctx.signal,
          enableProfiling: resolved.enableProfiling,
          webgpuOptions: resolved.webgpuOptions,
          externalDataUrl: artifacts.encoderDataUrl,
          externalDataPath: artifacts.encoderFilename
            ? `${artifacts.encoderFilename}.data`
            : undefined,
        });
        warnings.push(warning);
        fallbackSucceeded = true;
        break;
      } catch (candidateError) {
        rethrowIfAssetAborted(candidateError);
        fallbackError = candidateError;
      }
    }

    if (!fallbackSucceeded || !encoderSession) {
      throw fallbackError;
    }
  }
  let decoderSession: OrtSessionLike | undefined;
  try {
    if (ctx.isDisposed()) {
      throw new Error(`NeMo TDT executor was disposed during load for "${ctx.modelId}".`);
    }
    const decoderPreferredOutputLocation:
      | Readonly<Record<string, OrtOutputLocation>>
      | undefined =
      resolved.decoderBackendForOrt === 'webgpu' && ctx.sourceOptions.decoderStateOutputLocation
        ? {
            outputs: 'cpu',
            output_states_1: ctx.sourceOptions.decoderStateOutputLocation,
            output_states_2: ctx.sourceOptions.decoderStateOutputLocation,
          }
        : undefined;
    try {
      decoderSession = await createOrtSession(ort, artifacts.decoderUrl, {
        backendId: resolved.decoderBackendForOrt,
        cpuThreads: resolved.cpuThreads,
        signal: ctx.signal,
        enableProfiling: resolved.enableProfiling,
        webgpuOptions: resolved.webgpuOptions,
        preferredOutputLocation: decoderPreferredOutputLocation,
        externalDataUrl: artifacts.decoderDataUrl,
        externalDataPath: artifacts.decoderFilename
          ? `${artifacts.decoderFilename}.data`
          : undefined,
      });
    } catch (error) {
      rethrowIfAssetAborted(error);
      const canFallbackDecoderToInt8 =
        resolved.decoderBackendForOrt !== 'native-cpu' &&
        ctx.sourceOptions.kind === 'huggingface' &&
        ctx.sourceOptions.decoderQuant !== 'int8';

      if (!canFallbackDecoderToInt8 || !isExternalDataLoadError(error)) {
        throw error;
      }

      const fallbackResolved = resolveNemoTdtArtifacts(
        {
          ...ctx.sourceOptions,
          decoderQuant: 'int8',
        },
        ctx.backendId,
      );
      const cause = error instanceof Error ? error.message : String(error);
      const warning = {
        code: 'nemo-tdt.decoder-fp32-fallback',
        message: `Decoder weights required missing external data at the selected quantization. Falling back to INT8 decoder weights. Cause: ${cause}`,
        recoverable: true as const,
      } satisfies TranscriptWarning;
      ctx.runtimeHooks?.onProgress?.({
        phase: 'model-load:fallback',
        modelId: ctx.modelId,
        backendId: resolved.decoderBackendForOrt,
        file: fallbackResolved.artifacts.decoderFilename,
        message: warning.message,
        warning,
      });
      const fallbackArtifacts = await ctx.materializeArtifacts(fallbackResolved.artifacts);
      artifacts = {
        ...artifacts,
        decoderUrl: fallbackArtifacts.decoderUrl,
        decoderDataUrl: fallbackArtifacts.decoderDataUrl,
        decoderFilename: fallbackArtifacts.decoderFilename,
      };

      decoderSession = await createOrtSession(ort, artifacts.decoderUrl, {
        backendId: resolved.decoderBackendForOrt,
        cpuThreads: resolved.cpuThreads,
        signal: ctx.signal,
        enableProfiling: resolved.enableProfiling,
        webgpuOptions: resolved.webgpuOptions,
        preferredOutputLocation: decoderPreferredOutputLocation,
        externalDataUrl: artifacts.decoderDataUrl,
        externalDataPath: artifacts.decoderFilename
          ? `${artifacts.decoderFilename}.data`
          : undefined,
      });
      warnings.push(warning);
    }
    if (ctx.isDisposed()) {
      throw new Error(`NeMo TDT executor was disposed during load for "${ctx.modelId}".`);
    }
    const preprocessor: NemoPreprocessor =
      resolved.preprocessorBackend === 'js'
        ? new JsNemoPreprocessor({
            melBins: ctx.config.melBins,
            validLengthMode: ctx.config.preprocessorValidLengthMode,
            normalization: ctx.config.preprocessorNormalization,
          })
        : new OnnxNemoPreprocessor(
            ort,
            artifacts.preprocessorUrl!,
            resolved.enableProfiling,
            ctx.signal,
            resolved.ortBackend === 'native-cpu' ? 'cpu' : 'wasm',
            resolved.cpuThreads,
          );

    return {
      ort,
      tokenizer,
      encoderSession,
      decoderSession,
      preprocessor,
      preprocessorBackend: resolved.preprocessorBackend,
      encoderBackend:
        resolved.encoderBackendForOrt === 'native-cpu' ? 'cpu' : resolved.encoderBackendForOrt,
      decoderBackend:
        resolved.decoderBackendForOrt === 'native-cpu' ? 'cpu' : resolved.decoderBackendForOrt,
      warnings,
    };
  } catch (error) {
    // Initialization owns these sessions until the loaded state is returned.
    // Preserve the load error and expose failed cleanup of acquired resources.
    const results = await Promise.allSettled([
      Promise.resolve().then(() => releaseOrtSession(encoderSession)),
      Promise.resolve().then(() => releaseOrtSession(decoderSession)),
    ]);
    const failures = results.flatMap((result) =>
      result.status === 'rejected' ? [result.reason] : [],
    );
    if (failures.length) {
      throw new AggregateError(
        [error, ...failures],
        `NeMo TDT initialization and cleanup failed for "${ctx.modelId}".`,
        { cause: error },
      );
    }
    throw error;
  }
}
