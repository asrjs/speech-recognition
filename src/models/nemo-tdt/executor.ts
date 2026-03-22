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
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import type { TranscriptMetrics, TranscriptionProgressEvent } from '../../types/index.js';
import {
  createOrtSession,
  initOrt,
  resolveNemoTdtArtifacts,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from './ort.js';
import { JsNemoPreprocessor, type NemoPreprocessor, OnnxNemoPreprocessor } from './preprocessor.js';
import { ParakeetTokenizer } from './tokenizer.js';
import { buildEmptyTranscript, buildWordAndTokenDetails } from './transcript-details.js';
import { getDefaultNemoTdtWeightSetup } from './weights.js';
import type {
  NemoTdtDecoderStateSnapshot,
  NemoTdtExecutor,
  NemoTdtModelConfig,
  NemoTdtModelOptions,
  NemoTdtNativeTranscript,
  NemoTdtTranscriptionOptions,
} from './types.js';

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: ParakeetTokenizer;
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
  options: NemoTdtTranscriptionOptions,
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

export class OrtNemoTdtExecutor implements NemoTdtExecutor {
  private readonly sourceOptions: NemoTdtModelOptions['source'];
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly assetHandles: ResolvedAssetHandle[] = [];

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: NemoTdtModelConfig,
    private readonly backendId: string,
    loadOptions: NemoTdtModelOptions | undefined,
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
    artifacts: ReturnType<typeof resolveNemoTdtArtifacts>['artifacts'],
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

    const preprocessorFilename = artifacts.preprocessorUrl
      ? artifacts.preprocessorUrl.split('/').pop()
      : undefined;

    return {
      ...artifacts,
      encoderUrl: (await resolveFile(artifacts.encoderFilename)) ?? artifacts.encoderUrl,
      decoderUrl: (await resolveFile(artifacts.decoderFilename)) ?? artifacts.decoderUrl,
      tokenizerUrl: (await resolveFile('vocab.txt')) ?? artifacts.tokenizerUrl,
      preprocessorUrl: (await resolveFile(preprocessorFilename)) ?? artifacts.preprocessorUrl,
      encoderDataUrl: artifacts.encoderDataUrl
        ? await resolveFile(`${artifacts.encoderFilename}.data`)
        : artifacts.encoderDataUrl,
      decoderDataUrl: artifacts.decoderDataUrl
        ? await resolveFile(`${artifacts.decoderFilename}.data`)
        : artifacts.decoderDataUrl,
    };
  }

  private async initialize(): Promise<LoadedExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const resolved = resolveNemoTdtArtifacts(this.sourceOptions, this.backendId);
    let artifacts = await this.materializeHuggingFaceArtifacts(resolved.artifacts);
    if (resolved.preprocessorBackend === 'onnx' && !artifacts.preprocessorUrl) {
      throw new Error(
        `The NeMo TDT source for "${this.modelId}" does not provide a preprocessor model.`,
      );
    }

    const ort = await initOrt(resolved.ortBackend, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
    });
    const tokenizer = await ParakeetTokenizer.fromUrl(artifacts.tokenizerUrl);
    const warnings = resolved.warnings.map((warning) => ({
      ...warning,
      recoverable: true,
    }));
    let encoderSession: OrtSessionLike;
    try {
      encoderSession = await createOrtSession(ort, artifacts.encoderUrl, {
        backendId: resolved.encoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
        externalDataUrl: artifacts.encoderDataUrl,
        externalDataPath: artifacts.encoderFilename
          ? `${artifacts.encoderFilename}.data`
          : undefined,
      });
    } catch (error) {
      const implicitFp16Encoder =
        this.sourceOptions.kind === 'huggingface' &&
        !this.sourceOptions.encoderQuant &&
        resolved.encoderBackendForOrt === 'webgpu' &&
        getDefaultNemoTdtWeightSetup(resolved.encoderBackendForOrt).encoderDefault === 'fp16';

      if (!implicitFp16Encoder) {
        throw error;
      }

      const fallbackResolved = resolveNemoTdtArtifacts(
        {
          ...this.sourceOptions,
          encoderQuant: 'fp32',
        },
        this.backendId,
      );
      const fallbackArtifacts = await this.materializeHuggingFaceArtifacts(
        fallbackResolved.artifacts,
      );
      artifacts = {
        ...artifacts,
        encoderUrl: fallbackArtifacts.encoderUrl,
        encoderDataUrl: fallbackArtifacts.encoderDataUrl,
        encoderFilename: fallbackArtifacts.encoderFilename,
      };
      encoderSession = await createOrtSession(ort, artifacts.encoderUrl, {
        backendId: resolved.encoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
        externalDataUrl: artifacts.encoderDataUrl,
        externalDataPath: artifacts.encoderFilename
          ? `${artifacts.encoderFilename}.data`
          : undefined,
      });
      warnings.push({
        code: 'nemo-tdt.encoder-fp16-fallback',
        message:
          'Default FP16 encoder weights could not be initialized on this WebGPU setup. Falling back to FP32 encoder weights.',
        recoverable: true,
      });
    }
    const decoderSession = await createOrtSession(ort, artifacts.decoderUrl, {
      backendId: resolved.decoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: artifacts.decoderDataUrl,
      externalDataPath: artifacts.decoderFilename ? `${artifacts.decoderFilename}.data` : undefined,
    });
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

  private snapshotDecoderState(
    state: {
      readonly state1: OrtTensorLike<Float32Array>;
      readonly state2: OrtTensorLike<Float32Array>;
    } | null,
  ): NemoTdtDecoderStateSnapshot | undefined {
    if (!state) {
      return undefined;
    }

    return {
      s1: new Float32Array(state.state1.data),
      s2: new Float32Array(state.state2.data),
      dims1: [...state.state1.dims],
      dims2: [...state.state2.dims],
    };
  }

  async transcribe(
    audio: AudioBufferLike,
    options: NemoTdtTranscriptionOptions,
    _context: NemoDecodeContext<NemoTdtModelConfig>,
  ): Promise<NemoTdtNativeTranscript> {
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
        code: 'nemo-tdt.sample-rate-mismatch',
        message: `Expected ${this.config.sampleRate} Hz audio but received ${audio.sampleRate} Hz. No resampler is wired into the restored TDT path yet.`,
        recoverable: true,
      });
    }

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
      return buildEmptyTranscript(warnings);
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
        audio_signal: inputTensor,
        length: encoderLengthTensor,
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

    const encoderTensor = (encoderOutputs.outputs ??
      Object.values(encoderOutputs)[0]) as OrtTensorLike<Float32Array>;
    const dims = [...encoderTensor.dims];
    if (dims.length !== 3 || dims[0] !== 1) {
      throw new Error(`Unexpected NeMo TDT encoder output shape: [${dims.join(', ')}].`);
    }

    const encoderIsBdt = (dims[1] ?? 0) > (dims[2] ?? 0);
    const featureSize = encoderIsBdt ? dims[1]! : dims[2]!;
    const frameCount = encoderIsBdt ? dims[2]! : dims[1]!;
    const frameTime = this.config.frameShiftSeconds * this.config.subsamplingFactor;
    const vocabSize = loaded.tokenizer.vocabSize;
    const blankId = loaded.tokenizer.blankId ?? this.config.tokenizer.blankTokenId ?? 0;
    if (this.config.vocabularySize && this.config.vocabularySize !== vocabSize) {
      warnings.push({
        code: 'nemo-tdt.vocabulary-size-mismatch',
        message: `Configured vocabulary size ${this.config.vocabularySize} does not match tokenizer vocabulary size ${vocabSize}. Using tokenizer vocabulary size for decoder output slicing.`,
        recoverable: true,
      });
    }
    const encoderFrameBuffer = new Float32Array(featureSize);
    const encoderFrameTensor = new loaded.ort.Tensor('float32', encoderFrameBuffer, [
      1,
      featureSize,
      1,
    ]);
    const targetIdBuffer = new Int32Array(1);
    const targetTensor = new loaded.ort.Tensor('int32', targetIdBuffer, [1, 1]);
    const targetLengthTensor = new loaded.ort.Tensor('int32', new Int32Array([1]), [1]);

    const predictionLayers = this.config.predictionLayers ?? 2;
    const predictionHiddenSize = this.config.predictionHiddenSize ?? 640;
    let decoderState: {
      state1: OrtTensorLike<Float32Array>;
      state2: OrtTensorLike<Float32Array>;
    } | null = {
      state1: new loaded.ort.Tensor(
        'float32',
        new Float32Array(predictionLayers * predictionHiddenSize),
        [predictionLayers, 1, predictionHiddenSize],
      ),
      state2: new loaded.ort.Tensor(
        'float32',
        new Float32Array(predictionLayers * predictionHiddenSize),
        [predictionLayers, 1, predictionHiddenSize],
      ),
    };

    const tokenIds: number[] = [];
    const tokenTimestamps: Array<[number, number]> = [];
    const tokenConfidences: number[] = [];
    const tokenFrameIndices: number[] = [];
    const tokenLogProbs: number[] = [];
    const tokenTdtSteps: number[] = [];
    const frameConfidenceStats = new Map<number, { sum: number; count: number }>();
    let emittedOnFrame = 0;
    let decodeIterations = 0;

    const disposeTensor = (tensor: OrtTensorLike | undefined): void => {
      tensor?.dispose?.();
    };

    const disposeDecoderState = (
      state: { state1: OrtTensorLike; state2: OrtTensorLike } | null,
      keep?: { state1: OrtTensorLike; state2: OrtTensorLike } | null,
    ): void => {
      if (!state) {
        return;
      }

      if (state.state1 !== keep?.state1) {
        disposeTensor(state.state1);
      }
      if (state.state2 !== keep?.state2) {
        disposeTensor(state.state2);
      }
    };

    const decoderStart = nowMs();
    let lastReportedDecodeUnits = -1;
    try {
      for (let frameIndex = 0; frameIndex < frameCount; ) {
        decodeIterations += 1;
        if (encoderIsBdt) {
          for (let featureIndex = 0; featureIndex < featureSize; featureIndex += 1) {
            encoderFrameBuffer[featureIndex] =
              encoderTensor.data[featureIndex * frameCount + frameIndex] ?? 0;
          }
        } else {
          const offset = frameIndex * featureSize;
          encoderFrameBuffer.set(encoderTensor.data.subarray(offset, offset + featureSize));
        }

        targetIdBuffer[0] = tokenIds.length > 0 ? tokenIds[tokenIds.length - 1]! : blankId;
        const decoderOutputs = await loaded.decoderSession.run({
          encoder_outputs: encoderFrameTensor,
          targets: targetTensor,
          target_length: targetLengthTensor,
          input_states_1: decoderState?.state1,
          input_states_2: decoderState?.state2,
        });

        const logits = decoderOutputs.outputs as OrtTensorLike<Float32Array>;
        const nextState = {
          state1: decoderOutputs.output_states_1 as OrtTensorLike<Float32Array>,
          state2: decoderOutputs.output_states_2 as OrtTensorLike<Float32Array>,
        };
        const logitsData = logits.data;
        if (logitsData.length < vocabSize) {
          throw new Error(
            `NeMo TDT decoder output is too small (${logitsData.length}) for tokenizer vocabulary size ${vocabSize}.`,
          );
        }
        if (logitsData.length === vocabSize) {
          throw new Error('NeMo TDT decoder output is missing required TDT duration logits.');
        }
        const tokenId = argmax(logitsData, 0, vocabSize);
        const durationOffset = vocabSize;
        const step =
          argmax(logitsData, durationOffset, logitsData.length - durationOffset) - durationOffset;
        const confidence = confidenceFromLogits(logitsData, tokenId, vocabSize);

        const existingFrameConfidence = frameConfidenceStats.get(frameIndex);
        if (existingFrameConfidence) {
          existingFrameConfidence.sum += confidence.confidence;
          existingFrameConfidence.count += 1;
        } else {
          frameConfidenceStats.set(frameIndex, { sum: confidence.confidence, count: 1 });
        }

        if (tokenId !== blankId) {
          const durationFrames = Math.max(1, step);
          tokenIds.push(tokenId);
          tokenTimestamps.push([
            roundTimestampSeconds(frameIndex * frameTime),
            roundTimestampSeconds(Math.min(frameCount, frameIndex + durationFrames) * frameTime),
          ]);
          tokenConfidences.push(confidence.confidence);
          tokenFrameIndices.push(frameIndex);
          tokenLogProbs.push(confidence.logProb);
          tokenTdtSteps.push(step);
          emittedOnFrame += 1;

          disposeDecoderState(decoderState, nextState);
          decoderState = nextState;
        } else {
          disposeDecoderState(nextState, decoderState);
        }

        disposeTensor(logits);

        if (step > 0) {
          frameIndex += step;
          emittedOnFrame = 0;
        } else if (tokenId === blankId || emittedOnFrame >= (this.config.maxSymbolsPerStep ?? 10)) {
          frameIndex += 1;
          emittedOnFrame = 0;
        }

        const completedUnits = Math.min(frameCount, frameIndex);
        if (completedUnits > lastReportedDecodeUnits) {
          lastReportedDecodeUnits = completedUnits;
          const decodeProgress = clampProgress(0.4 + (completedUnits / frameCount) * 0.5);
          const decodeElapsedMs = nowMs() - transcriptionStart;
          emitTranscriptionProgress(options, {
            stage: 'decode',
            progress: decodeProgress,
            elapsedMs: roundMetric(decodeElapsedMs),
            remainingMs: estimateRemainingMs(decodeElapsedMs, decodeProgress),
            completedUnits,
            totalUnits: frameCount,
            modelId: this.modelId,
            backendId: this.backendId,
            message: `Decoded ${completedUnits}/${frameCount} encoder frames for ${this.modelId}.`,
            metrics: {
              preprocessMs: roundMetric(preprocessMs),
              encodeMs: roundMetric(encodeMs),
              decodeMs: roundMetric(nowMs() - decoderStart),
            },
          });
        }
      }
    } finally {
      disposeTensor(encoderTensor);
      targetTensor.dispose?.();
      targetLengthTensor.dispose?.();
      encoderFrameTensor.dispose?.();
    }
    const decodeMs = nowMs() - decoderStart;

    const tokenizeStart = nowMs();
    const text = loaded.tokenizer.decode(tokenIds);
    const details = buildWordAndTokenDetails(
      loaded.tokenizer,
      tokenIds,
      tokenTimestamps,
      tokenConfidences,
      tokenFrameIndices,
      tokenLogProbs,
      tokenTdtSteps,
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

    const frameConfidences = [...frameConfidenceStats.values()].map(
      (entry) => entry.sum / entry.count,
    );
    const utteranceConfidence =
      tokenConfidences.length > 0
        ? tokenConfidences.reduce((sum, value) => sum + value, 0) / tokenConfidences.length
        : undefined;
    const wordAverage =
      details.words && details.words.length > 0
        ? details.words.reduce((sum, word) => sum + (word.confidence ?? 0), 0) /
          details.words.length
        : undefined;
    const tokenAverage =
      tokenConfidences.length > 0
        ? tokenConfidences.reduce((sum, value) => sum + value, 0) / tokenConfidences.length
        : undefined;
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
      encoderFrameCount: frameCount,
      decodeIterations,
      emittedTokenCount: tokenIds.length,
      emittedWordCount: details.words?.length,
    };

    const transcript: NemoTdtNativeTranscript = {
      utteranceText: text,
      isFinal: true,
      words: details.words,
      tokens: details.tokens,
      confidence: {
        utterance: utteranceConfidence,
        wordAverage,
        tokenAverage,
        frameAverage:
          frameConfidences.length > 0
            ? frameConfidences.reduce((sum, value) => sum + value, 0) / frameConfidences.length
            : undefined,
        averageLogProb:
          tokenLogProbs.length > 0
            ? tokenLogProbs.reduce((sum, value) => sum + value, 0) / tokenLogProbs.length
            : undefined,
        frames: frameConfidences,
      },
      metrics: {
        preprocessMs: totalMetrics.preprocessMs,
        encodeMs: totalMetrics.encodeMs,
        decodeMs: totalMetrics.decodeMs,
        tokenizeMs: roundMetric(tokenizeMs),
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
        emittedWordCount: totalMetrics.emittedWordCount,
      },
      warnings,
      debug: {
        tokenIds: options.returnTokenIds ? tokenIds : undefined,
        frameIndices: options.returnFrameIndices ? tokenFrameIndices : undefined,
        logProbs: options.returnLogProbs ? tokenLogProbs : undefined,
        tdtSteps: options.returnTdtSteps ? tokenTdtSteps : undefined,
      },
      decoderState: options.returnDecoderState
        ? this.snapshotDecoderState(decoderState)
        : undefined,
    };

    disposeTensor(decoderState?.state1);
    disposeTensor(decoderState?.state2);

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
    return undefined;
  }
}
