import { PcmAudioBuffer } from '../../audio/index.js';
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
  RuntimeLogger,
  SpeechModel,
  SpeechSession,
  TranscriptResult,
  TranscriptSegment,
  TranscriptWarning,
  TranscriptWord,
  TranscriptionEnvelope,
} from '../../types/index.js';
import type {
  NemoAedModelOptions,
  NemoAedNativeToken,
  NemoAedNativeTranscript,
  NemoAedPreprocessorBackend,
  NemoAedQuantization,
  NemoAedTranscriptionOptions,
} from '../../models/nemo-aed/index.js';
import { getDefaultNemoAedWeightSetup } from '../../models/nemo-aed/weights.js';
import { DEFAULT_MODEL, resolveCanaryArtifactSource } from './manifest.js';
import { getModelConfig, getModelKeyFromRepoId, MODELS } from './catalog.js';

export { MODELS, DEFAULT_MODEL };

export type CanaryBackend = 'wasm' | 'webgpu' | 'webgpu-hybrid' | 'webgpu-strict';
export type CanaryExecutionBackend = 'wasm' | 'webgpu';

export interface CanaryModelUrls {
  readonly urls: {
    readonly encoderUrl: string;
    readonly decoderUrl: string;
    readonly tokenizerUrl: string;
    readonly configUrl: string;
    readonly preprocessorUrl?: string;
    readonly encoderDataUrl?: string | null;
    readonly decoderDataUrl?: string | null;
  };
  readonly filenames: {
    readonly encoder: string;
    readonly decoder: string;
    readonly tokenizer: string;
    readonly config: string;
  };
  readonly quantisation: {
    readonly encoder: QuantizationMode;
    readonly decoder: QuantizationMode;
  };
  readonly modelConfig: ReturnType<typeof getModelConfig>;
  readonly preprocessorBackend: NemoAedPreprocessorBackend;
}

export interface CanaryFromHubOptions {
  readonly revision?: string;
  readonly encoderBackend?: CanaryExecutionBackend;
  readonly decoderBackend?: CanaryExecutionBackend;
  readonly encoderQuant?: QuantizationMode;
  readonly decoderQuant?: QuantizationMode;
  readonly preprocessorBackend?: NemoAedPreprocessorBackend;
  readonly preprocessorName?: 'nemo128';
  readonly backend?: CanaryBackend;
  readonly progress?: (progress: ModelFileProgress) => void;
  readonly verbose?: boolean;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly runtime?: DefaultSpeechRuntime;
}

export interface CanaryFromUrlsConfig {
  readonly modelId?: string;
  readonly encoderBackend?: CanaryExecutionBackend;
  readonly decoderBackend?: CanaryExecutionBackend;
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly tokenizerUrl: string;
  readonly configUrl?: string;
  readonly preprocessorUrl?: string;
  readonly encoderDataUrl?: string | null;
  readonly decoderDataUrl?: string | null;
  readonly filenames?: {
    readonly encoder?: string;
    readonly decoder?: string;
    readonly tokenizer?: string;
    readonly config?: string;
  };
  readonly preprocessorBackend?: NemoAedPreprocessorBackend;
  readonly backend?: CanaryBackend;
  readonly verbose?: boolean;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly runtime?: DefaultSpeechRuntime;
}

export type CanaryTranscribeOptions = NemoAedTranscriptionOptions;

export interface CanaryTranscriptionResult {
  readonly text: string;
  readonly language?: string;
  readonly words?: readonly TranscriptWord[];
  readonly segments?: readonly TranscriptSegment[];
  readonly timestamp: {
    readonly word?: readonly TranscriptWord[];
    readonly segment?: readonly TranscriptSegment[];
  };
  readonly tokens?: readonly NemoAedNativeToken[];
  readonly prompt?: NemoAedNativeTranscript['prompt'];
  readonly warnings: readonly TranscriptWarning[];
  readonly metrics?: NemoAedNativeTranscript['metrics'];
  readonly canonical: TranscriptResult;
  readonly native: NemoAedNativeTranscript;
}

export interface TranscribeCanaryOptions extends CanaryFromHubOptions {
  readonly modelId?: string;
  readonly transcribeOptions?: CanaryTranscribeOptions;
}

const QUANT_SUFFIX: Record<QuantizationMode, string> = {
  fp16: '.fp16.onnx',
  int8: '.int8.onnx',
  fp32: '.onnx',
};

function getQuantizedModelName(baseName: string, quant: QuantizationMode): string {
  return `${baseName}${QUANT_SUFFIX[quant]}`;
}

function normalizeBackendId(backend: CanaryBackend | undefined): 'webgpu' | 'wasm' {
  return String(backend || 'webgpu-hybrid').startsWith('webgpu') ? 'webgpu' : 'wasm';
}

function resolveEncoderBackend(options: {
  readonly backend?: CanaryBackend;
  readonly encoderBackend?: CanaryExecutionBackend;
}): CanaryExecutionBackend {
  return options.encoderBackend ?? normalizeBackendId(options.backend);
}

function resolveDecoderBackend(options: {
  readonly backend?: CanaryBackend;
  readonly decoderBackend?: CanaryExecutionBackend;
}): CanaryExecutionBackend {
  if (options.decoderBackend) {
    return options.decoderBackend;
  }
  const backend = String(options.backend || 'webgpu-hybrid');
  if (backend === 'wasm') {
    return 'wasm';
  }
  if (backend === 'webgpu' || backend === 'webgpu-strict') {
    return 'webgpu';
  }
  return 'wasm';
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

function resolveRepoId(modelIdOrRepoId: string): string {
  const modelConfig = getModelConfig(modelIdOrRepoId);
  if (modelConfig) {
    return modelConfig.repoId;
  }

  const manifestSource = resolveCanaryArtifactSource(modelIdOrRepoId);
  if (manifestSource?.kind === 'huggingface') {
    return manifestSource.repoId;
  }

  return modelIdOrRepoId;
}

function getDefaultModelId(modelIdOrRepoId: string): string {
  return getModelKeyFromRepoId(modelIdOrRepoId) ?? DEFAULT_MODEL;
}

function toNemoQuantization(quant: QuantizationMode): NemoAedQuantization {
  return quant;
}

function mapCanaryTranscribeOptions(
  options: CanaryTranscribeOptions = {},
): NemoAedTranscriptionOptions & {
  readonly responseFlavor: 'canonical+native';
  readonly detail: 'detailed';
} {
  const { source_lang, target_lang, pnc, timestamp, responseFlavor, detail, ...rest } = options;
  void responseFlavor;
  void detail;

  return {
    ...rest,
    sourceLanguage: rest.sourceLanguage ?? source_lang,
    targetLanguage: rest.targetLanguage ?? target_lang,
    punctuate: rest.punctuate,
    pnc,
    timestamps: rest.timestamps,
    timestamp,
    responseFlavor: 'canonical+native',
    detail: 'detailed',
  };
}

function toCanaryTranscriptionResult(
  envelope: TranscriptionEnvelope<NemoAedNativeTranscript>,
): CanaryTranscriptionResult {
  const canonical = envelope.canonical;
  const native = envelope.native ?? {
    utteranceText: canonical.text,
    isFinal: canonical.meta.isFinal,
    warnings: canonical.warnings,
  };

  return {
    text: canonical.text,
    language: canonical.meta.language ?? native.language,
    words: canonical.words,
    segments: canonical.segments,
    timestamp: {
      word: canonical.words,
      segment: canonical.segments,
    },
    tokens: native.tokens,
    prompt: native.prompt,
    warnings: native.warnings ?? canonical.warnings,
    metrics: native.metrics,
    canonical,
    native,
  };
}

export async function getCanaryModel(
  modelIdOrRepoId: string = DEFAULT_MODEL,
  options: CanaryFromHubOptions = {},
): Promise<CanaryModelUrls> {
  const repoId = resolveRepoId(modelIdOrRepoId);
  const modelConfig = getModelConfig(modelIdOrRepoId) ?? getModelConfig(repoId);
  const revision = options.revision ?? 'main';
  const encoderBackend = resolveEncoderBackend(options);
  const decoderBackend = resolveDecoderBackend(options);
  const repoFiles = await fetchModelFiles(repoId, revision);
  const encoderAvailable = getAvailableQuantModes(repoFiles, 'encoder-model');
  const decoderAvailable = getAvailableQuantModes(repoFiles, 'decoder-model');
  const encoderSetup = getDefaultNemoAedWeightSetup(encoderBackend);
  const decoderSetup = getDefaultNemoAedWeightSetup(decoderBackend);
  const preprocessorBackend = options.preprocessorBackend ?? 'js';
  const preprocessorName = options.preprocessorName ?? 'nemo128';

  const encoderQuant =
    options.encoderQuant ??
    encoderSetup.encoderPreferred.find((quantization) =>
      encoderAvailable.includes(quantization as QuantizationMode),
    ) ??
    pickPreferredQuant(encoderAvailable, encoderBackend, 'encoder');
  const decoderQuant =
    options.decoderQuant ??
    decoderSetup.decoderPreferred.find((quantization) =>
      decoderAvailable.includes(quantization as QuantizationMode),
    ) ??
    pickPreferredQuant(decoderAvailable, decoderBackend, 'decoder');

  const encoderFilename = getQuantizedModelName('encoder-model', encoderQuant);
  const decoderFilename = getQuantizedModelName('decoder-model', decoderQuant);
  const tokenizerFilename = 'tokenizer.json';
  const configFilename = 'config.json';

  if (
    preprocessorBackend === 'onnx' &&
    !repoFiles.some(
      (path) => path === `${preprocessorName}.onnx` || path.endsWith(`/${preprocessorName}.onnx`),
    )
  ) {
    throw new Error(
      `[Hub] Canary ONNX preprocessor ${preprocessorName}.onnx is missing in ${repoId}. Use preprocessorBackend='js' or upload the frontend artifact.`,
    );
  }

  let encoderDataUrl: string | null = null;
  let decoderDataUrl: string | null = null;
  const baseUrls = {
    encoderUrl: await getModelFile(repoId, encoderFilename, {
      revision,
      progress: options.progress,
    }),
    decoderUrl: await getModelFile(repoId, decoderFilename, {
      revision,
      progress: options.progress,
    }),
    tokenizerUrl: await getModelFile(repoId, tokenizerFilename, {
      revision,
      progress: options.progress,
    }),
    configUrl: await getModelFile(repoId, configFilename, {
      revision,
      progress: options.progress,
    }),
    preprocessorUrl:
      preprocessorBackend === 'onnx'
        ? await getModelFile(repoId, `${preprocessorName}.onnx`, {
            revision,
            progress: options.progress,
          })
        : undefined,
  };

  const encoderDataName = `${encoderFilename}.data`;
  const decoderDataName = `${decoderFilename}.data`;
  if (repoFiles.some((path) => path === encoderDataName || path.endsWith(`/${encoderDataName}`))) {
    encoderDataUrl = await getModelFile(repoId, encoderDataName, {
      revision,
      progress: options.progress,
    });
  }
  if (repoFiles.some((path) => path === decoderDataName || path.endsWith(`/${decoderDataName}`))) {
    decoderDataUrl = await getModelFile(repoId, decoderDataName, {
      revision,
      progress: options.progress,
    });
  }

  const urls: CanaryModelUrls['urls'] = {
    ...baseUrls,
    encoderDataUrl,
    decoderDataUrl,
  };

  return {
    urls,
    filenames: {
      encoder: encoderFilename,
      decoder: decoderFilename,
      tokenizer: tokenizerFilename,
      config: configFilename,
    },
    quantisation: {
      encoder: encoderQuant,
      decoder: decoderQuant,
    },
    modelConfig,
    preprocessorBackend,
  };
}

export class CanaryModel {
  constructor(
    private readonly runtime: DefaultSpeechRuntime,
    private readonly model: SpeechModel<NemoAedModelOptions, any, NemoAedNativeTranscript>,
    private readonly session: SpeechSession<NemoAedTranscriptionOptions, NemoAedNativeTranscript>,
    private readonly onDispose?: () => void | Promise<void>,
  ) {}

  static async fromUrls(config: CanaryFromUrlsConfig): Promise<CanaryModel> {
    const runtime =
      config.runtime ??
      createBuiltInSpeechRuntime({
        hooks: {
          logger: createConsoleLogger(config.verbose),
        },
      });

    const modelId = config.modelId ?? DEFAULT_MODEL;
    const model = await runtime.loadModel<NemoAedModelOptions, NemoAedNativeTranscript>({
      preset: 'canary',
      modelId,
      backend: normalizeBackendId(config.backend),
      options: {
        source: {
          kind: 'direct',
          encoderBackend: config.encoderBackend,
          decoderBackend: config.decoderBackend,
          artifacts: {
            encoderUrl: config.encoderUrl,
            decoderUrl: config.decoderUrl,
            tokenizerUrl: config.tokenizerUrl,
            configUrl: config.configUrl,
            preprocessorUrl: config.preprocessorUrl,
            encoderDataUrl: config.encoderDataUrl ?? undefined,
            decoderDataUrl: config.decoderDataUrl ?? undefined,
            encoderFilename: config.filenames?.encoder,
            decoderFilename: config.filenames?.decoder,
            tokenizerFilename: config.filenames?.tokenizer,
            configFilename: config.filenames?.config,
          },
          preprocessorBackend: config.preprocessorBackend,
          cpuThreads: config.cpuThreads,
          enableProfiling: config.enableProfiling,
        },
      },
    });
    const session = await model.createSession();
    return new CanaryModel(runtime, model, session);
  }

  static async fromPretrained(
    modelId: string = DEFAULT_MODEL,
    options: CanaryFromHubOptions = {},
  ): Promise<CanaryModel> {
    const runtime =
      options.runtime ??
      createBuiltInSpeechRuntime({
        hooks: {
          logger: createConsoleLogger(options.verbose),
        },
      });

    const repoId = resolveRepoId(modelId);
    const model = await runtime.loadModel<NemoAedModelOptions, NemoAedNativeTranscript>({
      preset: 'canary',
      modelId: getDefaultModelId(modelId),
      backend: normalizeBackendId(options.backend),
      options: {
        source: {
          kind: 'huggingface',
          repoId,
          revision: options.revision,
          encoderBackend: options.encoderBackend,
          decoderBackend: options.decoderBackend,
          encoderQuant: options.encoderQuant ? toNemoQuantization(options.encoderQuant) : undefined,
          decoderQuant: options.decoderQuant ? toNemoQuantization(options.decoderQuant) : undefined,
          preprocessorName: options.preprocessorName,
          preprocessorBackend: options.preprocessorBackend,
          cpuThreads: options.cpuThreads,
          enableProfiling: options.enableProfiling,
        },
      },
    });
    const session = await model.createSession();
    return new CanaryModel(runtime, model, session);
  }

  static async fromHub(
    modelIdOrRepoId: string = DEFAULT_MODEL,
    options: CanaryFromHubOptions = {},
  ): Promise<CanaryModel> {
    return CanaryModel.fromPretrained(modelIdOrRepoId, options);
  }

  async transcribe(
    pcm: Float32Array,
    sampleRate: number,
    options: CanaryTranscribeOptions = {},
  ): Promise<CanaryTranscriptionResult> {
    const envelope = await this.session.transcribe(
      PcmAudioBuffer.fromMono(pcm, sampleRate),
      mapCanaryTranscribeOptions(options),
    );
    return toCanaryTranscriptionResult(envelope as TranscriptionEnvelope<NemoAedNativeTranscript>);
  }

  async dispose(): Promise<void> {
    await this.session.dispose();
    await this.model.dispose();
    await this.onDispose?.();
    void this.runtime;
  }
}

export async function transcribeCanary(
  pcm: Float32Array,
  sampleRate: number,
  options: TranscribeCanaryOptions = {},
): Promise<CanaryTranscriptionResult> {
  const { modelId = DEFAULT_MODEL, transcribeOptions, ...loadOptions } = options;
  const model = await CanaryModel.fromPretrained(modelId, loadOptions);
  try {
    return await model.transcribe(pcm, sampleRate, transcribeOptions);
  } finally {
    await model.dispose();
  }
}
