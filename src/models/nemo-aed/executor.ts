import type {
  AssetProvider,
  AudioBufferLike,
  ModelClassification,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptWarning,
} from '../../types/index.js';
import type { NemoDecodeContext } from '../nemo-common/index.js';
import { argmax, confidenceFromLogits } from '../../inference/index.js';
import { nowMs, roundMetric } from '../../runtime/timing.js';
import type { TranscriptMetrics, TranscriptionProgressEvent } from '../../types/index.js';
import {
  JsNemoPreprocessor,
  type NemoPreprocessor,
  OnnxNemoPreprocessor,
} from '../nemo-tdt/preprocessor.js';
import { CanaryTokenizer } from './tokenizer.js';
import {
  createOrtSession,
  initOrt,
  resolveNemoAedArtifacts,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from './ort.js';
import { getDefaultNemoAedWeightSetup } from './weights.js';
import type {
  NemoAedArtifactSource,
  NemoAedExecutor,
  NemoAedModelConfig,
  NemoAedModelOptions,
  NemoAedNativeToken,
  NemoAedNativeTranscript,
  NemoAedPromptSettings,
  NemoAedTranscriptionOptions,
} from './types.js';

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: CanaryTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly decoderSession: OrtSessionLike;
  readonly preprocessor: NemoPreprocessor;
  readonly preprocessorBackend: string;
  readonly warnings: readonly TranscriptWarning[];
}

function clampProgress(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function estimateRemainingMs(elapsedMs: number, progress: number): number | undefined {
  if (progress <= 0 || progress >= 1) {
    return undefined;
  }
  return roundMetric((elapsedMs / progress) * (1 - progress), 2);
}

function emitTranscriptionProgress(
  options: NemoAedTranscriptionOptions,
  event: TranscriptionProgressEvent,
): void {
  options.onProgress?.(event);
}

function roundMiB(bytes: number | undefined): number | undefined {
  if (!Number.isFinite(bytes)) {
    return undefined;
  }
  return roundMetric((bytes as number) / (1024 * 1024), 2);
}

function createAssetProgressEvent(
  modelId: string,
  file: string,
  event: {
    readonly loaded: number;
    readonly total?: number;
    readonly done?: boolean;
  },
): RuntimeProgressEvent {
  const percent =
    event.total && event.total > 0
      ? Math.min(100, Math.round((event.loaded / event.total) * 100))
      : event.done
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
    isComplete: event.done,
    message: event.done ? `Prepared ${file}.` : `Downloading ${file}.`,
  };
}

function buildEmptyTranscript(
  warnings: readonly TranscriptWarning[] = [],
  language?: string,
  prompt?: {
    readonly settings: NemoAedPromptSettings;
    readonly ids: readonly number[];
    readonly pieces: readonly string[];
  },
): NemoAedNativeTranscript {
  return {
    utteranceText: '',
    isFinal: true,
    language,
    warnings,
    prompt,
  };
}

function buildNativeTokens(
  tokenizer: CanaryTokenizer,
  tokenIds: readonly number[],
  confidences: readonly number[],
  logProbs: readonly number[],
): NemoAedNativeToken[] {
  const tokens: NemoAedNativeToken[] = [];
  for (let index = 0; index < tokenIds.length; index += 1) {
    const id = tokenIds[index]!;
    if (tokenizer.isSpecialId(id)) {
      continue;
    }
    const rawPiece = tokenizer.idsToTokens([id])[0] ?? '';
    tokens.push({
      index: tokens.length,
      id,
      text: tokenizer.toNativeTokenText(rawPiece),
      rawText: rawPiece,
      isWordStart: rawPiece.startsWith('\u2581'),
      confidence: confidences[index],
      logProb: logProbs[index],
      special: false,
    });
  }
  return tokens;
}

function resolveOutputTensor<TTensor extends OrtTensorLike>(
  outputs: Record<string, OrtTensorLike>,
  preferredKey: string,
  fallbackIndex: number,
): TTensor {
  return (outputs[preferredKey] ?? Object.values(outputs)[fallbackIndex]) as TTensor;
}

function resolveMaxNewTokens(
  requestedMaxNewTokens: number | undefined,
  maxTargetPositions: number,
  promptLength: number,
): number {
  const remainingBudget = Math.max(1, maxTargetPositions - promptLength);
  if (requestedMaxNewTokens === undefined) {
    return remainingBudget;
  }

  const normalized = Number.isFinite(requestedMaxNewTokens)
    ? Math.trunc(requestedMaxNewTokens)
    : remainingBudget;

  return Math.max(1, Math.min(normalized, remainingBudget));
}

export class OrtNemoAedExecutor implements NemoAedExecutor {
  private readonly sourceOptions: NemoAedModelOptions['source'];
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly assetHandles: ResolvedAssetHandle[] = [];

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: NemoAedModelConfig,
    private readonly backendId: string,
    loadOptions: NemoAedModelOptions | undefined,
    dependencies: {
      readonly assetProvider?: AssetProvider;
      readonly runtimeHooks?: SpeechRuntimeHooks;
    } = {},
  ) {
    this.sourceOptions = loadOptions?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;
    if (this.sourceOptions) {
      this.loadStatePromise = this.initialize();
    }
  }

  private async materializeHuggingFaceArtifacts(
    artifacts: ReturnType<typeof resolveNemoAedArtifacts>['artifacts'],
  ): Promise<typeof artifacts> {
    const source = this.sourceOptions;
    if (!this.assetProvider || !source || source.kind !== 'huggingface') {
      return artifacts;
    }

    const revision = source.revision ?? 'main';
    const resolveFile = async (filename: string | undefined): Promise<string | undefined> => {
      if (!filename) {
        return undefined;
      }

      const handle = await this.assetProvider!.resolve({
        id: `huggingface:${source.repoId}:${revision}:${filename}`,
        provider: 'huggingface',
        repoId: source.repoId,
        revision,
        filename,
        cacheKey: `huggingface:${source.repoId}:${revision}:${filename}`,
        onProgress: (event) => {
          this.runtimeHooks?.onProgress?.(createAssetProgressEvent(this.modelId, filename, event));
        },
      });
      this.assetHandles.push(handle);
      const locator = await handle.getLocator('url');
      if (!locator) {
        throw new Error(`Could not create a URL locator for "${filename}".`);
      }
      return locator;
    };

    return {
      ...artifacts,
      encoderUrl: (await resolveFile(artifacts.encoderFilename)) ?? artifacts.encoderUrl,
      decoderUrl: (await resolveFile(artifacts.decoderFilename)) ?? artifacts.decoderUrl,
      tokenizerUrl: (await resolveFile(artifacts.tokenizerFilename)) ?? artifacts.tokenizerUrl,
      configUrl: (await resolveFile(artifacts.configFilename)) ?? artifacts.configUrl,
      preprocessorUrl:
        (await resolveFile(artifacts.preprocessorFilename)) ?? artifacts.preprocessorUrl,
    };
  }

  private async initialize(): Promise<LoadedExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const resolved = resolveNemoAedArtifacts(this.sourceOptions, this.backendId);
    let artifacts = await this.materializeHuggingFaceArtifacts(resolved.artifacts);
    if (resolved.preprocessorBackend === 'onnx' && !artifacts.preprocessorUrl) {
      throw new Error(
        `The NeMo AED source for "${this.modelId}" does not provide a preprocessor model.`,
      );
    }

    const ort = await initOrt(resolved.ortBackend, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
    });
    const tokenizer = await CanaryTokenizer.fromUrl(artifacts.tokenizerUrl);
    const warnings = resolved.warnings.map((warning) => ({
      ...warning,
      recoverable: true,
    }));

    let encoderSession: OrtSessionLike;
    try {
      encoderSession = await createOrtSession(ort, artifacts.encoderUrl, {
        backendId: resolved.encoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
      });
    } catch (error) {
      const source = this.sourceOptions;
      const implicitFp16Encoder =
        source.kind === 'huggingface' &&
        !source.encoderQuant &&
        resolved.encoderBackendForOrt === 'webgpu' &&
        getDefaultNemoAedWeightSetup(resolved.encoderBackendForOrt).encoderDefault === 'fp16';

      if (!implicitFp16Encoder) {
        throw error;
      }

      const fallbackResolved = resolveNemoAedArtifacts(
        {
          ...source,
          encoderQuant: 'fp32',
        } satisfies NemoAedArtifactSource,
        this.backendId,
      );
      artifacts = await this.materializeHuggingFaceArtifacts(fallbackResolved.artifacts);
      encoderSession = await createOrtSession(ort, artifacts.encoderUrl, {
        backendId: resolved.encoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
      });
      warnings.push({
        code: 'nemo-aed.encoder-fp16-fallback',
        message:
          'Default FP16 encoder weights could not be initialized on this WebGPU setup. Falling back to FP32 encoder weights.',
        recoverable: true,
      });
    }

    let decoderSession: OrtSessionLike;
    try {
      decoderSession = await createOrtSession(ort, artifacts.decoderUrl, {
        backendId: resolved.decoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
      });
    } catch (error) {
      const source = this.sourceOptions;
      const implicitFp16Decoder =
        source.kind === 'huggingface' &&
        !source.decoderQuant &&
        resolved.decoderBackendForOrt === 'webgpu' &&
        getDefaultNemoAedWeightSetup(resolved.decoderBackendForOrt).decoderDefault === 'fp16';

      if (!implicitFp16Decoder) {
        throw error;
      }

      const fallbackResolved = resolveNemoAedArtifacts(
        {
          ...source,
          decoderQuant: 'fp32',
        } satisfies NemoAedArtifactSource,
        this.backendId,
      );
      artifacts = await this.materializeHuggingFaceArtifacts(fallbackResolved.artifacts);
      decoderSession = await createOrtSession(ort, artifacts.decoderUrl, {
        backendId: resolved.decoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
      });
      warnings.push({
        code: 'nemo-aed.decoder-fp16-fallback',
        message:
          'Default FP16 decoder weights could not be initialized on this WebGPU setup. Falling back to FP32 decoder weights.',
        recoverable: true,
      });
    }

    const preprocessor: NemoPreprocessor =
      resolved.preprocessorBackend === 'js'
        ? new JsNemoPreprocessor({
            melBins: this.config.melBins,
            validLengthMode: this.config.preprocessorValidLengthMode,
            normalization: this.config.preprocessorNormalization,
          })
        : new OnnxNemoPreprocessor(ort, artifacts.preprocessorUrl!, resolved.enableProfiling);

    return {
      ort,
      tokenizer,
      encoderSession,
      decoderSession,
      preprocessor,
      preprocessorBackend: resolved.preprocessorBackend,
      warnings,
    };
  }

  private async getLoadedState(): Promise<LoadedExecutorState> {
    if (!this.loadStatePromise) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }
    return this.loadStatePromise;
  }

  async ready(): Promise<void> {
    await this.getLoadedState();
  }

  async transcribe(
    audio: AudioBufferLike,
    options: NemoAedTranscriptionOptions,
    _context: NemoDecodeContext<NemoAedModelConfig>,
  ): Promise<NemoAedNativeTranscript> {
    const transcriptionStart = nowMs();
    const loaded = await this.getLoadedState();
    const warnings = [...loaded.warnings];

    emitTranscriptionProgress(options, {
      stage: 'start',
      progress: 0,
      elapsedMs: 0,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Starting transcription for ${this.modelId}.`,
    });

    if (audio.sampleRate !== this.config.sampleRate) {
      warnings.push({
        code: 'nemo-aed.sample-rate-mismatch',
        message: `Expected ${this.config.sampleRate} Hz audio but received ${audio.sampleRate} Hz. No resampler is wired into the restored AED path yet.`,
        recoverable: true,
      });
    }

    const promptSettings = loaded.tokenizer.resolvePromptSettings(this.config, options);
    const promptIds = loaded.tokenizer.buildPromptIds(promptSettings);
    const promptPieces = loaded.tokenizer.idsToTokens(promptIds);

    const preprocessStart = nowMs();
    const processed = await loaded.preprocessor.process(audio);
    const preprocessMs = nowMs() - preprocessStart;
    const preprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'preprocess',
      progress: 0.2,
      elapsedMs: roundMetric(preprocessElapsedMs),
      remainingMs: estimateRemainingMs(preprocessElapsedMs, 0.2),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Prepared audio features for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
      },
    });

    if (processed.features.length === 0 || processed.frameCount === 0) {
      emitTranscriptionProgress(options, {
        stage: 'complete',
        progress: 1,
        elapsedMs: roundMetric(nowMs() - transcriptionStart),
        modelId: this.modelId,
        backendId: this.backendId,
        message: `Finished transcription for ${this.modelId}.`,
        metrics: {
          preprocessMs: roundMetric(preprocessMs),
          totalMs: roundMetric(nowMs() - transcriptionStart),
        },
      });
      return buildEmptyTranscript(warnings, promptSettings.targetLanguage, {
        settings: promptSettings,
        ids: promptIds,
        pieces: promptPieces,
      });
    }

    const inputTensor = new loaded.ort.Tensor('float32', processed.features, [
      1,
      this.config.melBins,
      processed.frameCount,
    ]);
    const encoderLengthTensor = new loaded.ort.Tensor(
      'int64',
      BigInt64Array.from([BigInt(processed.validLength)]),
      [1],
    );

    const encodeStart = nowMs();
    let encoderOutputs: Record<string, OrtTensorLike> | undefined;
    try {
      encoderOutputs = await loaded.encoderSession.run({
        processed_signal: inputTensor,
        processed_signal_length: encoderLengthTensor,
      });
    } finally {
      inputTensor.dispose?.();
      encoderLengthTensor.dispose?.();
    }
    const encodeMs = nowMs() - encodeStart;
    const encodeElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'encode',
      progress: 0.4,
      elapsedMs: roundMetric(encodeElapsedMs),
      remainingMs: estimateRemainingMs(encodeElapsedMs, 0.4),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Encoded acoustic frames for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
      },
    });

    const encoderTensor = resolveOutputTensor<OrtTensorLike<Float32Array>>(
      encoderOutputs,
      'encoder_states',
      0,
    );
    const encodedLengthTensorOut = resolveOutputTensor<OrtTensorLike<BigInt64Array | Int32Array>>(
      encoderOutputs,
      'encoded_length',
      1,
    );
    const encoderMaskTensor = resolveOutputTensor<OrtTensorLike<Float32Array>>(
      encoderOutputs,
      'encoder_mask',
      2,
    );
    const encoderDims = [...encoderTensor.dims];
    if (encoderDims.length !== 3 || encoderDims[0] !== 1) {
      throw new Error(`Unexpected NeMo AED encoder output shape: [${encoderDims.join(', ')}].`);
    }

    const encoderFrameCount = encoderDims[1] ?? 0;
    const vocabSize = loaded.tokenizer.vocabSize;
    if (this.config.vocabularySize && this.config.vocabularySize !== vocabSize) {
      warnings.push({
        code: 'nemo-aed.vocabulary-size-mismatch',
        message: `Configured vocabulary size ${this.config.vocabularySize} does not match tokenizer vocabulary size ${vocabSize}. Using tokenizer vocabulary size for decoder output slicing.`,
        recoverable: true,
      });
    }

    const maxNewTokens = resolveMaxNewTokens(
      options.maxNewTokens,
      this.config.maxTargetPositions,
      promptIds.length,
    );

    const generatedTokenIds: number[] = [];
    const generatedLogProbs: number[] = [];
    const generatedConfidences: number[] = [];
    const decoderStart = nowMs();
    let eosReached = false;
    let lastReportedDecodeUnits = -1;

    try {
      for (let stepIndex = 0; stepIndex < maxNewTokens; stepIndex += 1) {
        const inputIds = [...promptIds, ...generatedTokenIds];
        const inputIdTensor = new loaded.ort.Tensor(
          'int64',
          BigInt64Array.from(inputIds.map((id) => BigInt(id))),
          [1, inputIds.length],
        );

        let decoderOutputs: Record<string, OrtTensorLike>;
        try {
          decoderOutputs = await loaded.decoderSession.run({
            input_ids: inputIdTensor,
            encoder_states: encoderTensor,
            encoder_mask: encoderMaskTensor,
          });
        } finally {
          inputIdTensor.dispose?.();
        }

        const logits = resolveOutputTensor<OrtTensorLike<Float32Array>>(
          decoderOutputs,
          'next_logits',
          0,
        );
        const logitsData = logits.data;
        if (logitsData.length < vocabSize) {
          throw new Error(
            `NeMo AED decoder output is too small (${logitsData.length}) for tokenizer vocabulary size ${vocabSize}.`,
          );
        }

        const tokenId = argmax(logitsData, 0, vocabSize);
        const confidence = confidenceFromLogits(logitsData, tokenId, vocabSize);
        generatedTokenIds.push(tokenId);
        generatedLogProbs.push(confidence.logProb);
        generatedConfidences.push(confidence.confidence);
        logits.dispose?.();

        const completedUnits = stepIndex + 1;
        if (completedUnits > lastReportedDecodeUnits) {
          lastReportedDecodeUnits = completedUnits;
          const decodeProgress = clampProgress(0.4 + (completedUnits / maxNewTokens) * 0.5);
          const decodeElapsedMs = nowMs() - transcriptionStart;
          emitTranscriptionProgress(options, {
            stage: 'decode',
            progress: decodeProgress,
            elapsedMs: roundMetric(decodeElapsedMs),
            remainingMs: estimateRemainingMs(decodeElapsedMs, decodeProgress),
            completedUnits,
            totalUnits: maxNewTokens,
            modelId: this.modelId,
            backendId: this.backendId,
            message: `Decoded ${completedUnits}/${maxNewTokens} autoregressive steps for ${this.modelId}.`,
            metrics: {
              preprocessMs: roundMetric(preprocessMs),
              encodeMs: roundMetric(encodeMs),
              decodeMs: roundMetric(nowMs() - decoderStart),
            },
          });
        }

        if (tokenId === loaded.tokenizer.eosId) {
          eosReached = true;
          break;
        }
      }
    } finally {
      encoderTensor.dispose?.();
      encoderMaskTensor.dispose?.();
      encodedLengthTensorOut.dispose?.();
    }
    const decodeMs = nowMs() - decoderStart;

    if (!eosReached) {
      warnings.push({
        code: 'nemo-aed.max-new-tokens-exhausted',
        message: `Stopped decoding ${this.modelId} after ${maxNewTokens} autoregressive steps without emitting EOS.`,
        recoverable: true,
      });
    }

    const tokenizeStart = nowMs();
    const text = loaded.tokenizer.decode(generatedTokenIds);
    const nativeTokens = buildNativeTokens(
      loaded.tokenizer,
      generatedTokenIds,
      generatedConfidences,
      generatedLogProbs,
    );
    const tokenizeMs = nowMs() - tokenizeStart;
    const postprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'postprocess',
      progress: 0.95,
      elapsedMs: roundMetric(postprocessElapsedMs),
      remainingMs: estimateRemainingMs(postprocessElapsedMs, 0.95),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Built transcript details for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
        postprocessMs: roundMetric(tokenizeMs),
      },
    });

    const nonSpecialConfidences = nativeTokens
      .map((token) => token.confidence)
      .filter((value): value is number => typeof value === 'number');
    const nonSpecialLogProbs = nativeTokens
      .map((token) => token.logProb)
      .filter((value): value is number => typeof value === 'number');
    const totalMs = roundMetric(nowMs() - transcriptionStart);
    const rtf = audio.durationSeconds > 0 ? totalMs / (audio.durationSeconds * 1000) : 0;
    const rtfx = audio.durationSeconds > 0 ? audio.durationSeconds / (totalMs / 1000) : undefined;
    const totalMetrics: TranscriptMetrics = {
      preprocessMs: roundMetric(preprocessMs),
      encodeMs: roundMetric(encodeMs),
      decodeMs: roundMetric(decodeMs),
      tokenizeMs: roundMetric(tokenizeMs),
      postprocessMs: roundMetric(tokenizeMs),
      totalMs,
      wallMs: totalMs,
      audioDurationSec: roundMetric(audio.durationSeconds, 4),
      rtf: roundMetric(rtf, 4),
      rtfx: rtfx !== undefined ? roundMetric(rtfx, 4) : undefined,
      requestedPreprocessorBackend: this.sourceOptions?.preprocessorBackend ?? 'onnx',
      preprocessorBackend: loaded.preprocessorBackend,
      encoderFrameCount,
      decodeIterations: generatedTokenIds.length,
      emittedTokenCount: nativeTokens.length,
      emittedWordCount: undefined,
    };

    const transcript: NemoAedNativeTranscript = {
      utteranceText: text,
      isFinal: true,
      language: promptSettings.targetLanguage,
      tokens: nativeTokens,
      confidence: {
        utterance:
          nonSpecialConfidences.length > 0
            ? nonSpecialConfidences.reduce((sum, value) => sum + value, 0) /
              nonSpecialConfidences.length
            : undefined,
        tokenAverage:
          nonSpecialConfidences.length > 0
            ? nonSpecialConfidences.reduce((sum, value) => sum + value, 0) /
              nonSpecialConfidences.length
            : undefined,
        averageLogProb:
          nonSpecialLogProbs.length > 0
            ? nonSpecialLogProbs.reduce((sum, value) => sum + value, 0) / nonSpecialLogProbs.length
            : undefined,
      },
      metrics: {
        preprocessMs: totalMetrics.preprocessMs,
        encodeMs: totalMetrics.encodeMs,
        decodeMs: totalMetrics.decodeMs,
        tokenizeMs: totalMetrics.tokenizeMs,
        totalMs: totalMetrics.totalMs,
        wallMs: totalMetrics.wallMs,
        audioDurationSec: totalMetrics.audioDurationSec,
        rtf: totalMetrics.rtf,
        rtfx: totalMetrics.rtfx,
        requestedPreprocessorBackend: totalMetrics.requestedPreprocessorBackend,
        preprocessorBackend: totalMetrics.preprocessorBackend,
        encoderFrameCount: totalMetrics.encoderFrameCount,
        decodeIterations: totalMetrics.decodeIterations,
        emittedTokenCount: totalMetrics.emittedTokenCount,
      },
      warnings,
      prompt: {
        settings: promptSettings,
        ids: promptIds,
        pieces: promptPieces,
      },
      debug: {
        tokenIds: options.returnTokenIds ? generatedTokenIds : undefined,
        promptIds: options.returnPromptIds ? promptIds : undefined,
        logProbs: options.returnLogProbs ? generatedLogProbs : undefined,
      },
    };

    emitTranscriptionProgress(options, {
      stage: 'complete',
      progress: 1,
      elapsedMs: totalMetrics.totalMs,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Finished transcription for ${this.modelId}.`,
      metrics: totalMetrics,
    });

    return transcript;
  }

  dispose(): void {
    for (const handle of this.assetHandles) {
      handle.dispose();
    }
  }
}
