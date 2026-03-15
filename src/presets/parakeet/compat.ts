import { PcmAudioBuffer } from '../../audio/index.js';
import { createBlobAssetProvider } from '../../io/index.js';
import { createBuiltInSpeechRuntime } from '../../runtime/index.js';
import {
  fetchModelFiles,
  getAvailableQuantModes,
  getModelFile,
  pickPreferredQuant,
  type ModelFileProgress,
  type QuantizationMode,
} from '../../runtime/huggingface.js';
import type { DefaultSpeechRuntime } from '../../runtime/session.js';
import type {
  ResolvedAssetHandle,
  RuntimeLogger,
  SpeechModel,
  SpeechSession,
} from '../../types/index.js';
import type {
  NemoTdtModelOptions,
  NemoTdtNativeTranscript,
  NemoTdtTranscriptionOptions,
} from '../../models/nemo-tdt/index.js';
import {
  DEFAULT_MODEL,
  getModelConfig,
  getModelKeyFromRepoId,
  getParakeetDefaultWeightSetup,
  MODELS,
} from './catalog.js';

export { MODELS, DEFAULT_MODEL };

/** Browser/backend execution modes exposed by the Parakeet convenience helpers. */
export type ParakeetBackend = 'wasm' | 'webgpu' | 'webgpu-hybrid' | 'webgpu-strict';

/** Minimal browser file-handle shape accepted by Parakeet local-folder helpers. */
export interface ParakeetLocalFileHandleLike {
  readonly kind?: 'file';
  getFile(): Promise<File | Blob>;
}

/** Minimal browser directory-handle shape accepted by Parakeet local-folder helpers. */
export interface ParakeetLocalDirectoryHandleLike {
  readonly kind?: 'directory';
  readonly name?: string;
  entries(): AsyncIterable<
    [string, ParakeetLocalFileHandleLike | ParakeetLocalDirectoryHandleLike]
  >;
}

/** Flat local-entry record used by Parakeet local-folder resolution utilities. */
export interface ParakeetLocalEntry {
  readonly path: string;
  readonly basename: string;
  readonly file?: File | Blob;
  readonly handle?: ParakeetLocalFileHandleLike;
}

/** Summary of what a local Parakeet folder contains before concrete files are selected. */
export interface ParakeetLocalInspection {
  readonly encoderQuantizations: readonly QuantizationMode[];
  readonly decoderQuantizations: readonly QuantizationMode[];
  readonly tokenizerNames: readonly string[];
  readonly preprocessorNames: readonly ('nemo80' | 'nemo128')[];
}

/** Options for resolving or loading a Parakeet model from a local folder. */
export interface ResolveParakeetLocalEntriesOptions {
  readonly modelId?: string;
  readonly encoderQuant?: QuantizationMode;
  readonly decoderQuant?: QuantizationMode;
  readonly tokenizerName?: string;
  readonly preprocessorName?: 'nemo80' | 'nemo128';
  readonly preprocessorBackend?: 'js' | 'onnx';
  readonly backend?: ParakeetBackend;
  readonly verbose?: boolean;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly runtime?: DefaultSpeechRuntime;
}

/** Fully resolved local Parakeet artifact selection plus owned asset handles. */
export interface ResolvedParakeetLocalArtifacts {
  readonly config: ParakeetFromUrlsConfig;
  readonly assetHandles: readonly ResolvedAssetHandle[];
  readonly selection: {
    readonly encoderName: string;
    readonly decoderName: string;
    readonly tokenizerName: string;
    readonly preprocessorName?: string;
    readonly encoderQuant: QuantizationMode;
    readonly decoderQuant: QuantizationMode;
  };
}

/** Direct artifact URLs and metadata describing a Parakeet model bundle. */
export interface ParakeetModelUrls {
  readonly urls: {
    readonly encoderUrl: string;
    readonly decoderUrl: string;
    readonly tokenizerUrl: string;
    readonly preprocessorUrl?: string;
    readonly encoderDataUrl?: string | null;
    readonly decoderDataUrl?: string | null;
  };
  readonly filenames: {
    readonly encoder: string;
    readonly decoder: string;
  };
  readonly quantisation: {
    readonly encoder: QuantizationMode;
    readonly decoder: QuantizationMode;
  };
  readonly modelConfig: ReturnType<typeof getModelConfig>;
  readonly preprocessorBackend: 'js' | 'onnx';
}

/** Options for resolving a Parakeet model bundle from the Hugging Face hub. */
export interface GetParakeetModelOptions {
  readonly revision?: string;
  readonly encoderQuant?: QuantizationMode;
  readonly decoderQuant?: QuantizationMode;
  readonly preprocessor?: 'nemo80' | 'nemo128';
  readonly preprocessorBackend?: 'js' | 'onnx';
  readonly backend?: ParakeetBackend;
  readonly progress?: (progress: ModelFileProgress) => void;
  readonly verbose?: boolean;
}

/** Direct-artifact configuration accepted by `ParakeetModel.fromUrls()`. */
export interface ParakeetFromUrlsConfig {
  readonly modelId?: string;
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly tokenizerUrl: string;
  readonly preprocessorUrl?: string;
  readonly encoderDataUrl?: string | null;
  readonly decoderDataUrl?: string | null;
  readonly filenames?: {
    readonly encoder?: string;
    readonly decoder?: string;
  };
  readonly preprocessorBackend?: 'js' | 'onnx';
  readonly backend?: ParakeetBackend;
  readonly verbose?: boolean;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly runtime?: DefaultSpeechRuntime;
}

/** Legacy metrics shape preserved for compatibility with earlier Parakeet.js consumers. */
export interface LegacyParakeetMetrics {
  readonly preprocess_ms?: number;
  readonly encode_ms?: number;
  readonly decode_ms?: number;
  readonly tokenize_ms?: number;
  readonly total_ms?: number;
  readonly wall_ms?: number;
  readonly audio_duration_sec?: number;
  readonly rtf?: number;
  readonly rtfx?: number;
  readonly preprocessor_backend_requested?: string;
  readonly preprocessor_backend?: string;
  readonly audio_decode_ms?: number;
  readonly downmix_ms?: number;
  readonly resample_ms?: number;
  readonly audio_preparation_ms?: number;
  readonly input_sample_rate?: number;
  readonly output_sample_rate?: number;
  readonly resampler?: string;
  readonly resampler_quality?: string | null;
  readonly encoder_frame_count?: number;
  readonly decode_iterations?: number;
  readonly emitted_token_count?: number;
  readonly emitted_word_count?: number;
}

/** Legacy JSON transcript shape returned by the Parakeet compatibility wrapper. */
export interface LegacyParakeetTranscript {
  readonly utterance_text: string;
  readonly words: ReadonlyArray<{
    readonly text: string;
    readonly start_time: number;
    readonly end_time: number;
    readonly confidence?: number;
  }>;
  readonly tokens?: ReadonlyArray<{
    readonly id?: number;
    readonly token: string;
    readonly raw_text?: string;
    readonly start_time: number;
    readonly end_time: number;
    readonly confidence?: number;
    readonly frame_index?: number;
    readonly log_prob?: number;
    readonly tdt_step?: number;
  }>;
  readonly confidence_scores?: {
    readonly utterance?: number | null;
    readonly word_avg?: number | null;
    readonly token_avg?: number | null;
    readonly frame_avg?: number | null;
    readonly overall_log_prob?: number | null;
    readonly frame?: readonly number[] | null;
  };
  readonly metrics?: LegacyParakeetMetrics;
  readonly is_final: boolean;
}

/** Legacy transcription options accepted by the Parakeet compatibility wrapper. */
export interface LegacyParakeetTranscribeOptions {
  readonly returnTimestamps?: boolean;
  readonly returnConfidences?: boolean;
  readonly returnTokenIds?: boolean;
  readonly returnFrameIndices?: boolean;
  readonly returnLogProbs?: boolean;
  readonly returnTdtSteps?: boolean;
  readonly returnDecoderState?: boolean;
  readonly frameStride?: number;
  readonly enableProfiling?: boolean;
}

function getRequiredPreprocessorFilename(
  preprocessor: 'nemo80' | 'nemo128',
): `${'nemo80' | 'nemo128'}.onnx` {
  return `${preprocessor}.onnx`;
}

const QUANT_SUFFIX: Record<QuantizationMode, string> = {
  int8: '.int8.onnx',
  fp16: '.fp16.onnx',
  fp32: '.onnx',
};

function getQuantizedModelName(baseName: string, quant: QuantizationMode): string {
  return `${baseName}${QUANT_SUFFIX[quant]}`;
}

function getBasename(path: string): string {
  return (
    String(path || '')
      .split('/')
      .pop() || ''
  );
}

function normalizeRelativePath(path: string): string {
  return String(path || '')
    .replace(/\\/g, '/')
    .replace(/^\.\//, '');
}

function detectLocalQuantModes(
  entries: readonly ParakeetLocalEntry[],
  baseName: string,
): QuantizationMode[] {
  const names = new Set(entries.map((entry) => entry.basename.toLowerCase()));
  const out: QuantizationMode[] = [];
  if (names.has(`${baseName}.onnx`)) out.push('fp32');
  if (names.has(`${baseName}.fp16.onnx`)) out.push('fp16');
  if (names.has(`${baseName}.int8.onnx`)) out.push('int8');
  return out;
}

function findLocalEntry(
  entries: readonly ParakeetLocalEntry[],
  expectedName: string,
): ParakeetLocalEntry | null {
  const lower = expectedName.toLowerCase();
  return (
    entries.find(
      (entry) =>
        entry.path.toLowerCase() === lower ||
        entry.basename.toLowerCase() === lower ||
        entry.path.toLowerCase().endsWith(`/${lower}`),
    ) ?? null
  );
}

/** Converts a flat list of browser `File` objects into Parakeet local-entry records. */
export function createParakeetLocalEntries(files: readonly File[]): ParakeetLocalEntry[] {
  return files.map((file) => {
    const path = normalizeRelativePath(file.webkitRelativePath || file.name);
    return {
      path,
      basename: getBasename(path),
      file,
    };
  });
}

/** Recursively collects file entries from a local directory handle. */
export async function collectParakeetLocalEntries(
  dirHandle: ParakeetLocalDirectoryHandleLike,
  prefix = '',
): Promise<ParakeetLocalEntry[]> {
  const entries: ParakeetLocalEntry[] = [];
  for await (const [name, handle] of dirHandle.entries()) {
    if (handle.kind === 'directory' && name === '.git') {
      continue;
    }

    const relativePath = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === 'file') {
      const path = normalizeRelativePath(relativePath);
      entries.push({
        path,
        basename: getBasename(path),
        handle,
      });
      continue;
    }

    if (handle.kind === 'directory') {
      const nested = await collectParakeetLocalEntries(handle, relativePath);
      entries.push(...nested);
    }
  }

  return entries;
}

/** Resolves the actual `File` or `Blob` behind a local Parakeet entry. */
export async function getParakeetLocalEntryFile(entry: ParakeetLocalEntry): Promise<File | Blob> {
  if (entry.file) {
    return entry.file;
  }
  if (entry.handle?.kind === 'file') {
    return entry.handle.getFile();
  }
  throw new Error(
    `Could not access local file entry: ${entry.path || entry.basename || 'unknown'}`,
  );
}

/**
 * Inspects a local folder against the canonical Parakeet model layout.
 *
 * The expected tokenizer file is `vocab.txt`, matching the original
 * `parakeet.js` model repositories. Other `.txt` files are treated only as a
 * fallback for local debugging folders that have not been normalized yet.
 */
export function inspectParakeetLocalEntries(
  entries: readonly ParakeetLocalEntry[],
): ParakeetLocalInspection {
  const encoderQuantizations = detectLocalQuantModes(entries, 'encoder-model');
  const decoderQuantizations = detectLocalQuantModes(entries, 'decoder_joint-model');

  const tokenizerCandidates: string[] = [];
  if (findLocalEntry(entries, 'vocab.txt')) tokenizerCandidates.push('vocab.txt');
  if (!tokenizerCandidates.length) {
    for (const entry of entries) {
      if (entry.basename.toLowerCase().endsWith('.txt')) {
        tokenizerCandidates.push(entry.basename);
      }
    }
  }

  const preprocessorCandidates: Array<'nemo80' | 'nemo128'> = [];
  if (findLocalEntry(entries, 'nemo128.onnx')) preprocessorCandidates.push('nemo128');
  if (findLocalEntry(entries, 'nemo80.onnx')) preprocessorCandidates.push('nemo80');

  return {
    encoderQuantizations,
    decoderQuantizations,
    tokenizerNames: [...new Set(tokenizerCandidates)],
    preprocessorNames: [...new Set(preprocessorCandidates)],
  };
}

function normalizeBackendId(backend: ParakeetBackend | undefined): 'webgpu' | 'wasm' {
  return String(backend || 'webgpu-hybrid').startsWith('webgpu') ? 'webgpu' : 'wasm';
}

function createConsoleLogger(enabled: boolean | undefined): RuntimeLogger | undefined {
  if (!enabled) {
    return undefined;
  }

  return {
    debug(message, meta) {
      console.debug(message, meta);
    },
    info(message, meta) {
      console.info(message, meta);
    },
    warn(message, meta) {
      console.warn(message, meta);
    },
    error(message, meta) {
      console.error(message, meta);
    },
  };
}

function revokeBlobUrls(urls: Record<string, unknown>): void {
  for (const value of Object.values(urls)) {
    if (typeof value === 'string' && value.startsWith('blob:')) {
      URL.revokeObjectURL(value);
    }
  }
}

function toFromUrlsConfig(
  modelUrls: ParakeetModelUrls,
  options: GetParakeetModelOptions = {},
): ParakeetFromUrlsConfig {
  return {
    modelId: getModelKeyFromRepoId(modelUrls.modelConfig?.repoId ?? '') ?? DEFAULT_MODEL,
    ...modelUrls.urls,
    filenames: modelUrls.filenames,
    preprocessorBackend: modelUrls.preprocessorBackend,
    backend: options.backend,
    verbose: options.verbose,
  };
}

function shouldRetryWithFp32(quantisation: ParakeetModelUrls['quantisation'] | undefined): boolean {
  return quantisation?.encoder === 'fp16' || quantisation?.decoder === 'fp16';
}

function buildRetryOptions(
  options: GetParakeetModelOptions,
  quantisation: ParakeetModelUrls['quantisation'] | undefined,
): GetParakeetModelOptions {
  const retryOptions = { ...options };
  if (quantisation?.encoder === 'fp16') {
    retryOptions.encoderQuant = 'fp32';
  }
  if (quantisation?.decoder === 'fp16') {
    retryOptions.decoderQuant = 'fp32';
  }
  return retryOptions;
}

function toLegacyTranscript(native: NemoTdtNativeTranscript): LegacyParakeetTranscript {
  return {
    utterance_text: native.utteranceText,
    words: (native.words ?? []).map((word) => ({
      text: word.text,
      start_time: word.startTime,
      end_time: word.endTime,
      confidence: word.confidence,
    })),
    tokens: native.tokens?.map((token) => ({
      id: token.id,
      token: token.text,
      raw_text: token.rawText,
      start_time: token.startTime ?? 0,
      end_time: token.endTime ?? 0,
      confidence: token.confidence,
      frame_index: token.frameIndex,
      log_prob: token.logProb,
      tdt_step: token.tdtStep,
    })),
    confidence_scores: native.confidence
      ? {
          utterance: native.confidence.utterance ?? null,
          word_avg: native.confidence.wordAverage ?? null,
          token_avg: native.confidence.tokenAverage ?? null,
          frame_avg: native.confidence.frameAverage ?? null,
          overall_log_prob: native.confidence.averageLogProb ?? null,
          frame: native.confidence.frames ?? null,
        }
      : undefined,
    metrics: native.metrics
      ? {
          preprocess_ms: native.metrics.preprocessMs,
          encode_ms: native.metrics.encodeMs,
          decode_ms: native.metrics.decodeMs,
          tokenize_ms: native.metrics.tokenizeMs,
          total_ms: native.metrics.totalMs,
          wall_ms: native.metrics.wallMs,
          audio_duration_sec: native.metrics.audioDurationSec,
          rtf: native.metrics.rtf,
          rtfx: native.metrics.rtfx,
          preprocessor_backend_requested: native.metrics.requestedPreprocessorBackend,
          preprocessor_backend: native.metrics.preprocessorBackend,
          audio_decode_ms: native.metrics.decodeAudioMs,
          downmix_ms: native.metrics.downmixMs,
          resample_ms: native.metrics.resampleMs,
          audio_preparation_ms: native.metrics.audioPreparationMs,
          input_sample_rate: native.metrics.inputSampleRate,
          output_sample_rate: native.metrics.outputSampleRate,
          resampler: native.metrics.resampler,
          resampler_quality: native.metrics.resamplerQuality,
          encoder_frame_count: native.metrics.encoderFrameCount,
          decode_iterations: native.metrics.decodeIterations,
          emitted_token_count: native.metrics.emittedTokenCount,
          emitted_word_count: native.metrics.emittedWordCount,
        }
      : undefined,
    is_final: native.isFinal,
  };
}

function mapTranscribeOptions(
  options: LegacyParakeetTranscribeOptions = {},
): NemoTdtTranscriptionOptions & { readonly responseFlavor: 'native' } {
  return {
    detail: options.returnTimestamps ? 'words' : 'text',
    responseFlavor: 'native',
    returnFrameIndices: options.returnFrameIndices,
    returnLogProbs: options.returnLogProbs,
    returnTdtSteps: options.returnTdtSteps,
    returnDecoderState: options.returnDecoderState,
    returnTokenIds: options.returnTokenIds,
  };
}

/** Formats the resolved encoder/decoder quantization for UI logging and diagnostics. */
export function formatResolvedQuantization(
  quantisation: ParakeetModelUrls['quantisation'],
): string {
  return `Resolved quantization: encoder=${quantisation.encoder}, decoder=${quantisation.decoder}`;
}

/** Resolves a Parakeet model bundle from the Hugging Face hub into concrete artifact URLs. */
export async function getParakeetModel(
  repoIdOrModelKey: string,
  options: GetParakeetModelOptions = {},
): Promise<ParakeetModelUrls> {
  const modelConfig = getModelConfig(repoIdOrModelKey);
  const repoId = modelConfig?.repoId || repoIdOrModelKey;
  const revision = options.revision ?? 'main';
  const preprocessor = options.preprocessor ?? modelConfig?.preprocessor ?? 'nemo128';
  const preprocessorBackend = options.preprocessorBackend ?? 'js';
  const backend = options.backend ?? 'webgpu-hybrid';
  const repoFiles = await fetchModelFiles(repoId, revision, {
    logger: createConsoleLogger(options.verbose),
  });

  const preferredSetup = getParakeetDefaultWeightSetup(repoIdOrModelKey, backend);
  const encoderAvailable = getAvailableQuantModes(repoFiles, 'encoder-model');
  const decoderAvailable = getAvailableQuantModes(repoFiles, 'decoder_joint-model');
  const encoderQuant =
    options.encoderQuant ??
    preferredSetup.encoderPreferred.find((quantization) =>
      encoderAvailable.includes(quantization),
    ) ??
    pickPreferredQuant(encoderAvailable, backend, 'encoder');
  const decoderQuant =
    options.decoderQuant ??
    preferredSetup.decoderPreferred.find((quantization) =>
      decoderAvailable.includes(quantization),
    ) ??
    pickPreferredQuant(decoderAvailable, backend, 'decoder');
  const encoderFilename = getQuantizedModelName('encoder-model', encoderQuant);
  const decoderFilename = getQuantizedModelName('decoder_joint-model', decoderQuant);

  if (
    encoderQuant === 'fp16' &&
    !repoFiles.includes(encoderFilename) &&
    !repoFiles.some((path) => path.endsWith(`/${encoderFilename}`))
  ) {
    throw new Error(
      `[Hub] Encoder FP16 file is missing in ${repoId}. Choose encoderQuant='fp32' explicitly.`,
    );
  }
  if (
    decoderQuant === 'fp16' &&
    !repoFiles.includes(decoderFilename) &&
    !repoFiles.some((path) => path.endsWith(`/${decoderFilename}`))
  ) {
    throw new Error(
      `[Hub] Decoder FP16 file is missing in ${repoId}. Choose decoderQuant='fp32' explicitly.`,
    );
  }

  const urls: {
    encoderUrl: string;
    decoderUrl: string;
    tokenizerUrl: string;
    preprocessorUrl?: string;
    encoderDataUrl?: string | null;
    decoderDataUrl?: string | null;
  } = {
    encoderUrl: await getModelFile(repoId, encoderFilename, {
      revision,
      progress: options.progress,
    }),
    decoderUrl: await getModelFile(repoId, decoderFilename, {
      revision,
      progress: options.progress,
    }),
    tokenizerUrl: await getModelFile(repoId, 'vocab.txt', { revision, progress: options.progress }),
  };

  // The restored Parakeet/TDT path still executes the ONNX mel frontend.
  // Keep downloading the preprocessor artifact even when callers request the
  // future JS mode so the executor can fall back cleanly.
  urls.preprocessorUrl = await getModelFile(repoId, getRequiredPreprocessorFilename(preprocessor), {
    revision,
    progress: options.progress,
  });

  const encoderDataName = `${encoderFilename}.data`;
  const decoderDataName = `${decoderFilename}.data`;
  const hasEncoderData = repoFiles.some(
    (path) => path === encoderDataName || path.endsWith(`/${encoderDataName}`),
  );
  const hasDecoderData = repoFiles.some(
    (path) => path === decoderDataName || path.endsWith(`/${decoderDataName}`),
  );

  if (hasEncoderData) {
    urls.encoderDataUrl = await getModelFile(repoId, encoderDataName, {
      revision,
      progress: options.progress,
    });
  }
  if (hasDecoderData) {
    urls.decoderDataUrl = await getModelFile(repoId, decoderDataName, {
      revision,
      progress: options.progress,
    });
  }

  return {
    urls,
    filenames: {
      encoder: encoderFilename,
      decoder: decoderFilename,
    },
    quantisation: {
      encoder: encoderQuant,
      decoder: decoderQuant,
    },
    modelConfig,
    preprocessorBackend,
  };
}

/**
 * Loads a Parakeet model with an automatic FP16-to-FP32 retry when compilation
 * fails on platforms that cannot run the preferred precision.
 */
export async function loadModelWithFallback({
  repoIdOrModelKey,
  options,
  getParakeetModelFn,
  fromUrlsFn,
  onBeforeCompile,
}: {
  readonly repoIdOrModelKey: string;
  readonly options: GetParakeetModelOptions;
  readonly getParakeetModelFn: (
    repoIdOrModelKey: string,
    options: GetParakeetModelOptions,
  ) => Promise<ParakeetModelUrls>;
  readonly fromUrlsFn: (config: ParakeetFromUrlsConfig) => Promise<ParakeetModel>;
  readonly onBeforeCompile?: (ctx: {
    attempt: number;
    modelUrls: ParakeetModelUrls;
    options: GetParakeetModelOptions;
  }) => void;
}): Promise<{ model: ParakeetModel; modelUrls: ParakeetModelUrls; retryUsed: boolean }> {
  const firstModelUrls = await getParakeetModelFn(repoIdOrModelKey, options);
  onBeforeCompile?.({ attempt: 1, modelUrls: firstModelUrls, options });

  try {
    const model = await fromUrlsFn(toFromUrlsConfig(firstModelUrls, options));
    return { model, modelUrls: firstModelUrls, retryUsed: false };
  } catch (firstError) {
    if (!shouldRetryWithFp32(firstModelUrls.quantisation)) {
      throw firstError;
    }

    revokeBlobUrls(firstModelUrls.urls);

    const retryOptions = buildRetryOptions(options, firstModelUrls.quantisation);
    let retryModelUrls: ParakeetModelUrls;
    try {
      retryModelUrls = await getParakeetModelFn(repoIdOrModelKey, retryOptions);
    } catch (retryDownloadError) {
      const firstMessage = firstError instanceof Error ? firstError.message : String(firstError);
      const retryDownloadMessage =
        retryDownloadError instanceof Error
          ? retryDownloadError.message
          : String(retryDownloadError);
      throw new Error(
        `[ModelLoader] Initial compile failed (${firstMessage}). FP32 retry download failed (${retryDownloadMessage}).`,
      );
    }

    onBeforeCompile?.({ attempt: 2, modelUrls: retryModelUrls, options: retryOptions });

    try {
      const model = await fromUrlsFn(toFromUrlsConfig(retryModelUrls, retryOptions));
      return { model, modelUrls: retryModelUrls, retryUsed: true };
    } catch (retryError) {
      const firstMessage = firstError instanceof Error ? firstError.message : String(firstError);
      const retryMessage = retryError instanceof Error ? retryError.message : String(retryError);
      throw new Error(
        `[ModelLoader] Initial compile failed (${firstMessage}). FP32 retry also failed (${retryMessage}).`,
      );
    }
  }
}

/** Convenience wrapper around `loadModelWithFallback()` for the built-in Parakeet hub loader. */
export async function loadParakeetModelWithFallback(
  repoIdOrModelKey: string,
  options: GetParakeetModelOptions,
): Promise<{ model: ParakeetModel; modelUrls: ParakeetModelUrls; retryUsed: boolean }> {
  return loadModelWithFallback({
    repoIdOrModelKey,
    options,
    getParakeetModelFn: getParakeetModel,
    fromUrlsFn: ParakeetModel.fromUrls,
  });
}

/**
 * Selects concrete local artifacts for a Parakeet model and returns owned asset
 * handles that must be disposed once the model is no longer needed.
 */
export async function resolveParakeetLocalEntries(
  entries: readonly ParakeetLocalEntry[],
  options: ResolveParakeetLocalEntriesOptions = {},
): Promise<ResolvedParakeetLocalArtifacts> {
  if (entries.length === 0) {
    throw new Error('Pick a local model folder first.');
  }

  const inspection = inspectParakeetLocalEntries(entries);
  const backend = options.backend ?? 'webgpu-hybrid';
  const preferredSetup = getParakeetDefaultWeightSetup(options.modelId ?? DEFAULT_MODEL, backend);
  const availableEncoderQuantizations: readonly QuantizationMode[] =
    inspection.encoderQuantizations.length > 0 ? inspection.encoderQuantizations : ['fp32'];
  const availableDecoderQuantizations: readonly QuantizationMode[] =
    inspection.decoderQuantizations.length > 0 ? inspection.decoderQuantizations : ['fp32'];
  const encoderQuant =
    options.encoderQuant ??
    preferredSetup.encoderPreferred.find((quantization) =>
      availableEncoderQuantizations.includes(quantization),
    ) ??
    pickPreferredQuant(availableEncoderQuantizations, backend, 'encoder');
  const decoderQuant =
    options.decoderQuant ??
    preferredSetup.decoderPreferred.find((quantization) =>
      availableDecoderQuantizations.includes(quantization),
    ) ??
    pickPreferredQuant(availableDecoderQuantizations, backend, 'decoder');
  const encoderName = getQuantizedModelName('encoder-model', encoderQuant);
  const decoderName = getQuantizedModelName('decoder_joint-model', decoderQuant);
  const tokenizerName = options.tokenizerName ?? inspection.tokenizerNames[0];
  const preprocessorBackend = options.preprocessorBackend ?? 'js';
  const preprocessorName = getRequiredPreprocessorFilename(
    options.preprocessorName ?? inspection.preprocessorNames[0] ?? 'nemo128',
  );

  const encoderEntry = findLocalEntry(entries, encoderName);
  const decoderEntry = findLocalEntry(entries, decoderName);
  const tokenizerEntry = tokenizerName ? findLocalEntry(entries, tokenizerName) : null;

  if (!encoderEntry) {
    throw new Error(`Missing encoder file: ${encoderName}`);
  }
  if (!decoderEntry) {
    throw new Error(`Missing decoder file: ${decoderName}`);
  }
  if (!tokenizerEntry) {
    throw new Error(`Missing tokenizer file: ${tokenizerName ?? 'vocab.txt'}`);
  }

  const assetProvider = createBlobAssetProvider();
  const assetHandles: ResolvedAssetHandle[] = [];
  const toLocator = async (entry: ParakeetLocalEntry): Promise<string> => {
    const file = await getParakeetLocalEntryFile(entry);
    const handle = await assetProvider.resolve({
      id: entry.path,
      provider: 'blob',
      blob: file instanceof Blob ? file : new Blob([file]),
    });
    assetHandles.push(handle);
    const locator = await handle.getLocator('url');
    if (!locator) {
      throw new Error(`Could not create a URL locator for local asset "${entry.path}".`);
    }
    return locator;
  };

  try {
    const preprocessorEntry = preprocessorName ? findLocalEntry(entries, preprocessorName) : null;
    if (preprocessorName && !preprocessorEntry) {
      throw new Error(
        `Missing preprocessor file: ${preprocessorName}. ` +
          'The restored @asrjs/speech-recognition Parakeet/TDT path still needs the ONNX preprocessor artifact, even when JS preprocessing is requested.',
      );
    }

    const encoderDataEntry = findLocalEntry(entries, `${encoderEntry.basename}.data`);
    const decoderDataEntry = findLocalEntry(entries, `${decoderEntry.basename}.data`);
    const resolvedTokenizerName = tokenizerEntry.basename;

    const config: ParakeetFromUrlsConfig = {
      modelId: options.modelId,
      encoderUrl: await toLocator(encoderEntry),
      decoderUrl: await toLocator(decoderEntry),
      tokenizerUrl: await toLocator(tokenizerEntry),
      preprocessorUrl: preprocessorEntry ? await toLocator(preprocessorEntry) : undefined,
      encoderDataUrl: encoderDataEntry ? await toLocator(encoderDataEntry) : undefined,
      decoderDataUrl: decoderDataEntry ? await toLocator(decoderDataEntry) : undefined,
      filenames: {
        encoder: encoderEntry.basename,
        decoder: decoderEntry.basename,
      },
      preprocessorBackend,
      backend: options.backend,
      verbose: options.verbose,
      cpuThreads: options.cpuThreads,
      enableProfiling: options.enableProfiling,
      runtime: options.runtime,
    };

    return {
      config,
      assetHandles,
      selection: {
        encoderName,
        decoderName,
        tokenizerName: resolvedTokenizerName,
        preprocessorName,
        encoderQuant,
        decoderQuant,
      },
    };
  } catch (error) {
    await Promise.all(
      assetHandles.map(async (handle) => {
        await handle.dispose();
      }),
    );
    throw error;
  }
}

/** Loads a Parakeet model directly from a previously collected local folder. */
export async function loadParakeetModelFromLocalEntries(
  entries: readonly ParakeetLocalEntry[],
  options: ResolveParakeetLocalEntriesOptions = {},
): Promise<{ model: ParakeetModel; selection: ResolvedParakeetLocalArtifacts['selection'] }> {
  const resolved = await resolveParakeetLocalEntries(entries, options);
  const model = await ParakeetModel.fromResolvedLocalArtifacts(resolved);
  return {
    model,
    selection: resolved.selection,
  };
}

/**
 * Thin compatibility wrapper that exposes Parakeet-style loading and legacy JSON
 * transcript output on top of the `@asrjs/speech-recognition` runtime.
 */
export class ParakeetModel {
  constructor(
    private readonly runtime: DefaultSpeechRuntime,
    private readonly model: SpeechModel<
      NemoTdtModelOptions,
      NemoTdtTranscriptionOptions,
      NemoTdtNativeTranscript
    >,
    private readonly session: SpeechSession<NemoTdtTranscriptionOptions, NemoTdtNativeTranscript>,
    private readonly onDispose?: () => void | Promise<void>,
  ) {}

  /** Creates a Parakeet model from explicitly provided artifact URLs. */
  static async fromUrls(config: ParakeetFromUrlsConfig): Promise<ParakeetModel> {
    const runtime =
      config.runtime ??
      createBuiltInSpeechRuntime({
        hooks: {
          logger: createConsoleLogger(config.verbose),
        },
      });

    const modelId = config.modelId ?? DEFAULT_MODEL;
    const model = await runtime.loadModel<NemoTdtModelOptions, NemoTdtNativeTranscript>({
      preset: 'parakeet',
      modelId,
      backend: normalizeBackendId(config.backend),
      options: {
        source: {
          kind: 'direct',
          artifacts: {
            encoderUrl: config.encoderUrl,
            decoderUrl: config.decoderUrl,
            tokenizerUrl: config.tokenizerUrl,
            preprocessorUrl: config.preprocessorUrl,
            encoderDataUrl: config.encoderDataUrl ?? undefined,
            decoderDataUrl: config.decoderDataUrl ?? undefined,
            encoderFilename: config.filenames?.encoder,
            decoderFilename: config.filenames?.decoder,
          },
          preprocessorBackend: config.preprocessorBackend,
          cpuThreads: config.cpuThreads,
          enableProfiling: config.enableProfiling,
        },
      },
    });
    const session = await model.createSession();
    return new ParakeetModel(runtime, model, session);
  }

  /** Creates a Parakeet model from a local folder represented by collected entries. */
  static async fromLocalEntries(
    entries: readonly ParakeetLocalEntry[],
    options: ResolveParakeetLocalEntriesOptions = {},
  ): Promise<ParakeetModel> {
    const resolved = await resolveParakeetLocalEntries(entries, options);
    return ParakeetModel.fromResolvedLocalArtifacts(resolved);
  }

  /** Creates a Parakeet model from already resolved local artifacts and owned asset handles. */
  static async fromResolvedLocalArtifacts(
    resolved: ResolvedParakeetLocalArtifacts,
  ): Promise<ParakeetModel> {
    const model = await ParakeetModel.fromUrls(resolved.config);
    return new ParakeetModel(model.runtime, model.model, model.session, async () => {
      await Promise.all(
        resolved.assetHandles.map(async (handle) => {
          await handle.dispose();
        }),
      );
    });
  }

  /** Resolves and loads a Parakeet model directly from the Hugging Face hub. */
  static async fromHub(
    repoIdOrModelKey: string,
    options: GetParakeetModelOptions = {},
  ): Promise<ParakeetModel> {
    const urls = await getParakeetModel(repoIdOrModelKey, options);
    return ParakeetModel.fromUrls(toFromUrlsConfig(urls, options));
  }

  /** Runs transcription and converts the native NeMo-TDT output into legacy Parakeet JSON. */
  async transcribe(
    pcm: Float32Array,
    sampleRate: number,
    options: LegacyParakeetTranscribeOptions = {},
  ): Promise<LegacyParakeetTranscript> {
    const native = await this.session.transcribe(
      PcmAudioBuffer.fromMono(pcm, sampleRate),
      mapTranscribeOptions(options),
    );
    return toLegacyTranscript(native);
  }

  /** Releases the session, model, and any temporary local asset handles owned by this wrapper. */
  async dispose(): Promise<void> {
    await this.session.dispose();
    await this.model.dispose();
    await this.onDispose?.();
    void this.runtime;
  }
}
