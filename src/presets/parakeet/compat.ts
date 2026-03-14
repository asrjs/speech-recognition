import { PcmAudioBuffer } from '../../processors/index.js';
import { createBuiltInSpeechRuntime } from '../../runtime/index.js';
import {
  fetchModelFiles,
  getAvailableQuantModes,
  getModelFile,
  pickPreferredQuant,
  type ModelFileProgress,
  type QuantizationMode
} from '../../runtime/huggingface.js';
import type { DefaultSpeechRuntime } from '../../runtime/session.js';
import type { RuntimeLogger, SpeechModel, SpeechSession } from '../../types/index.js';
import type {
  NemoTdtModelOptions,
  NemoTdtNativeTranscript,
  NemoTdtTranscriptionOptions
} from '../../models/nemo-tdt/index.js';
import { DEFAULT_MODEL, getModelConfig, getModelKeyFromRepoId, MODELS } from './catalog.js';

export { MODELS, DEFAULT_MODEL };

export type ParakeetBackend = 'wasm' | 'webgpu' | 'webgpu-hybrid' | 'webgpu-strict';

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

export interface LegacyParakeetMetrics {
  readonly preprocess_ms?: number;
  readonly encode_ms?: number;
  readonly decode_ms?: number;
  readonly tokenize_ms?: number;
  readonly total_ms?: number;
  readonly rtf?: number;
}

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

const QUANT_SUFFIX: Record<QuantizationMode, string> = {
  int8: '.int8.onnx',
  fp16: '.fp16.onnx',
  fp32: '.onnx'
};

function getQuantizedModelName(baseName: string, quant: QuantizationMode): string {
  return `${baseName}${QUANT_SUFFIX[quant]}`;
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
    }
  };
}

function revokeBlobUrls(urls: Record<string, unknown>): void {
  for (const value of Object.values(urls)) {
    if (typeof value === 'string' && value.startsWith('blob:')) {
      URL.revokeObjectURL(value);
    }
  }
}

function toFromUrlsConfig(modelUrls: ParakeetModelUrls, options: GetParakeetModelOptions = {}): ParakeetFromUrlsConfig {
  return {
    modelId: getModelKeyFromRepoId(modelUrls.modelConfig?.repoId ?? '') ?? DEFAULT_MODEL,
    ...modelUrls.urls,
    filenames: modelUrls.filenames,
    preprocessorBackend: modelUrls.preprocessorBackend,
    backend: options.backend,
    verbose: options.verbose
  };
}

function shouldRetryWithFp32(quantisation: ParakeetModelUrls['quantisation'] | undefined): boolean {
  return quantisation?.encoder === 'fp16' || quantisation?.decoder === 'fp16';
}

function buildRetryOptions(
  options: GetParakeetModelOptions,
  quantisation: ParakeetModelUrls['quantisation'] | undefined
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
      confidence: word.confidence
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
      tdt_step: token.tdtStep
    })),
    confidence_scores: native.confidence ? {
      utterance: native.confidence.utterance ?? null,
      word_avg: native.confidence.wordAverage ?? null,
      token_avg: native.confidence.tokenAverage ?? null,
      frame_avg: native.confidence.frameAverage ?? null,
      overall_log_prob: native.confidence.averageLogProb ?? null,
      frame: native.confidence.frames ?? null
    } : undefined,
    metrics: native.metrics ? {
      preprocess_ms: native.metrics.preprocessMs,
      encode_ms: native.metrics.encodeMs,
      decode_ms: native.metrics.decodeMs,
      tokenize_ms: native.metrics.tokenizeMs,
      total_ms: native.metrics.totalMs,
      rtf: native.metrics.rtf
    } : undefined,
    is_final: native.isFinal
  };
}

function mapTranscribeOptions(
  options: LegacyParakeetTranscribeOptions = {}
): NemoTdtTranscriptionOptions & { readonly responseFlavor: 'native' } {
  return {
    detail: options.returnTimestamps ? 'words' : 'text',
    responseFlavor: 'native',
    returnFrameIndices: options.returnFrameIndices,
    returnLogProbs: options.returnLogProbs,
    returnTdtSteps: options.returnTdtSteps,
    returnDecoderState: options.returnDecoderState,
    returnTokenIds: options.returnTokenIds
  };
}

export function formatResolvedQuantization(quantisation: ParakeetModelUrls['quantisation']): string {
  return `Resolved quantization: encoder=${quantisation.encoder}, decoder=${quantisation.decoder}`;
}

export async function getParakeetModel(
  repoIdOrModelKey: string,
  options: GetParakeetModelOptions = {}
): Promise<ParakeetModelUrls> {
  const modelConfig = getModelConfig(repoIdOrModelKey);
  const repoId = modelConfig?.repoId || repoIdOrModelKey;
  const revision = options.revision ?? 'main';
  const preprocessor = options.preprocessor ?? modelConfig?.preprocessor ?? 'nemo128';
  const preprocessorBackend = options.preprocessorBackend ?? 'js';
  const backend = options.backend ?? 'webgpu-hybrid';
  const repoFiles = await fetchModelFiles(repoId, revision);

  const encoderAvailable = getAvailableQuantModes(repoFiles, 'encoder-model');
  const decoderAvailable = getAvailableQuantModes(repoFiles, 'decoder_joint-model');
  const encoderQuant = options.encoderQuant ?? pickPreferredQuant(encoderAvailable, backend, 'encoder');
  const decoderQuant = options.decoderQuant ?? pickPreferredQuant(decoderAvailable, backend, 'decoder');
  const encoderFilename = getQuantizedModelName('encoder-model', encoderQuant);
  const decoderFilename = getQuantizedModelName('decoder_joint-model', decoderQuant);

  if (encoderQuant === 'fp16' && !repoFiles.includes(encoderFilename) && !repoFiles.some((path) => path.endsWith(`/${encoderFilename}`))) {
    throw new Error(`[Hub] Encoder FP16 file is missing in ${repoId}. Choose encoderQuant='fp32' explicitly.`);
  }
  if (decoderQuant === 'fp16' && !repoFiles.includes(decoderFilename) && !repoFiles.some((path) => path.endsWith(`/${decoderFilename}`))) {
    throw new Error(`[Hub] Decoder FP16 file is missing in ${repoId}. Choose decoderQuant='fp32' explicitly.`);
  }

  const urls: {
    encoderUrl: string;
    decoderUrl: string;
    tokenizerUrl: string;
    preprocessorUrl?: string;
    encoderDataUrl?: string | null;
    decoderDataUrl?: string | null;
  } = {
    encoderUrl: await getModelFile(repoId, encoderFilename, { revision, progress: options.progress }),
    decoderUrl: await getModelFile(repoId, decoderFilename, { revision, progress: options.progress }),
    tokenizerUrl: await getModelFile(repoId, 'vocab.txt', { revision, progress: options.progress })
  };

  if (preprocessorBackend !== 'js') {
    urls.preprocessorUrl = await getModelFile(repoId, `${preprocessor}.onnx`, {
      revision,
      progress: options.progress
    });
  }

  const encoderDataName = `${encoderFilename}.data`;
  const decoderDataName = `${decoderFilename}.data`;
  const hasEncoderData = repoFiles.some((path) => path === encoderDataName || path.endsWith(`/${encoderDataName}`));
  const hasDecoderData = repoFiles.some((path) => path === decoderDataName || path.endsWith(`/${decoderDataName}`));

  if (hasEncoderData) {
    urls.encoderDataUrl = await getModelFile(repoId, encoderDataName, { revision, progress: options.progress });
  }
  if (hasDecoderData) {
    urls.decoderDataUrl = await getModelFile(repoId, decoderDataName, { revision, progress: options.progress });
  }

  return {
    urls,
    filenames: {
      encoder: encoderFilename,
      decoder: decoderFilename
    },
    quantisation: {
      encoder: encoderQuant,
      decoder: decoderQuant
    },
    modelConfig,
    preprocessorBackend
  };
}

export async function loadModelWithFallback({
  repoIdOrModelKey,
  options,
  getParakeetModelFn,
  fromUrlsFn,
  onBeforeCompile
}: {
  readonly repoIdOrModelKey: string;
  readonly options: GetParakeetModelOptions;
  readonly getParakeetModelFn: (repoIdOrModelKey: string, options: GetParakeetModelOptions) => Promise<ParakeetModelUrls>;
  readonly fromUrlsFn: (config: ParakeetFromUrlsConfig) => Promise<ParakeetModel>;
  readonly onBeforeCompile?: (ctx: { attempt: number; modelUrls: ParakeetModelUrls; options: GetParakeetModelOptions }) => void;
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
      const retryDownloadMessage = retryDownloadError instanceof Error ? retryDownloadError.message : String(retryDownloadError);
      throw new Error(
        `[ModelLoader] Initial compile failed (${firstMessage}). FP32 retry download failed (${retryDownloadMessage}).`
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
        `[ModelLoader] Initial compile failed (${firstMessage}). FP32 retry also failed (${retryMessage}).`
      );
    }
  }
}

export async function loadParakeetModelWithFallback(
  repoIdOrModelKey: string,
  options: GetParakeetModelOptions
): Promise<{ model: ParakeetModel; modelUrls: ParakeetModelUrls; retryUsed: boolean }> {
  return loadModelWithFallback({
    repoIdOrModelKey,
    options,
    getParakeetModelFn: getParakeetModel,
    fromUrlsFn: ParakeetModel.fromUrls
  });
}

export class ParakeetModel {
  constructor(
    private readonly runtime: DefaultSpeechRuntime,
    private readonly model: SpeechModel<NemoTdtModelOptions, NemoTdtTranscriptionOptions, NemoTdtNativeTranscript>,
    private readonly session: SpeechSession<NemoTdtTranscriptionOptions, NemoTdtNativeTranscript>
  ) {}

  static async fromUrls(config: ParakeetFromUrlsConfig): Promise<ParakeetModel> {
    const runtime = config.runtime ?? createBuiltInSpeechRuntime({
      hooks: {
        logger: createConsoleLogger(config.verbose)
      }
    });

    const modelId = config.modelId ?? DEFAULT_MODEL;
    const model = await runtime.loadModel<NemoTdtModelOptions, NemoTdtTranscriptionOptions, NemoTdtNativeTranscript>({
      family: 'parakeet',
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
            decoderFilename: config.filenames?.decoder
          },
          preprocessorBackend: config.preprocessorBackend,
          cpuThreads: config.cpuThreads,
          enableProfiling: config.enableProfiling
        }
      }
    });
    const session = await model.createSession();
    return new ParakeetModel(runtime, model, session);
  }

  static async fromHub(repoIdOrModelKey: string, options: GetParakeetModelOptions = {}): Promise<ParakeetModel> {
    const urls = await getParakeetModel(repoIdOrModelKey, options);
    return ParakeetModel.fromUrls(toFromUrlsConfig(urls, options));
  }

  async transcribe(
    pcm: Float32Array,
    sampleRate: number,
    options: LegacyParakeetTranscribeOptions = {}
  ): Promise<LegacyParakeetTranscript> {
    const native = await this.session.transcribe(
      PcmAudioBuffer.fromMono(pcm, sampleRate),
      mapTranscribeOptions(options)
    );
    return toLegacyTranscript(native);
  }

  async dispose(): Promise<void> {
    await this.session.dispose();
    await this.model.dispose();
    void this.runtime;
  }
}
