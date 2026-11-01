import type {
  AssetProvider,
  AudioBufferLike,
  ModelClassification,
  ResolvedAssetHandle,
  SpeechRuntimeHooks,
  TranscriptMetrics,
} from '../../types/index.js';
import type { NemoDecodeContext } from '../nemo-common/index.js';
import { nowMs, roundMetric } from '../../runtime/timing.js';
import {
  initializeNemoTdtExecutorState,
  materializeNemoTdtHuggingFaceArtifacts,
  type LoadedExecutorState,
} from './executor-artifacts.js';
import {
  emitTranscriptionProgress,
  estimateRemainingMs,
  extractAndResolveNemoTdtEncoderOutput,
  runNemoTdtDecodeLoop,
  TDT_BATCH_MIN_FRAMES,
  type NemoTdtBatchState,
} from './executor-decode.js';
import {
  releaseOrtSession,
  type resolveNemoTdtArtifacts,
  type OrtTensorLike,
} from './ort.js';
import { buildEmptyTranscript, buildWordAndTokenDetails } from './transcript-details.js';
import type {
  NemoTdtExecutor,
  NemoTdtModelConfig,
  NemoTdtModelOptions,
  NemoTdtNativeTranscript,
  NemoTdtTranscriptionOptions,
} from './types.js';

export class OrtNemoTdtExecutor implements NemoTdtExecutor, NemoTdtBatchState {
  private readonly sourceOptions: NemoTdtModelOptions['source'];
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private disposed = false;
  private disposePromise?: Promise<void>;
  tdtBatchAllowed = true;
  tdtBatchWidth = TDT_BATCH_MIN_FRAMES;
  tdtBatchColumnsScored = 0;
  tdtBatchColumnsUseful = 0;
  tdtSequentialBlankVisits = 0;

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: NemoTdtModelConfig,
    private readonly backendId: string,
    loadOptions: NemoTdtModelOptions | undefined,
    dependencies: {
      readonly assetProvider?: AssetProvider;
      readonly runtimeHooks?: SpeechRuntimeHooks;
      readonly signal?: import('../../types/index.js').AbortSignalLike | null;
    } = {},
  ) {
    this.sourceOptions = loadOptions?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;
    this.signal = dependencies.signal;
    if (this.sourceOptions) {
      this.loadStatePromise = this.initialize();
    }
  }

  private async materializeHuggingFaceArtifacts(
    artifacts: ReturnType<typeof resolveNemoTdtArtifacts>['artifacts'],
  ): Promise<typeof artifacts> {
    return materializeNemoTdtHuggingFaceArtifacts(
      {
        modelId: this.modelId,
        sourceOptions: this.sourceOptions,
        assetProvider: this.assetProvider,
        runtimeHooks: this.runtimeHooks,
        assetHandles: this.assetHandles,
      },
      artifacts,
    );
  }

  private async initialize(): Promise<LoadedExecutorState> {
    return initializeNemoTdtExecutorState({
      modelId: this.modelId,
      backendId: this.backendId,
      config: this.config,
      sourceOptions: this.sourceOptions,
      runtimeHooks: this.runtimeHooks,
      signal: this.signal,
      isDisposed: () => this.disposed,
      materializeArtifacts: (artifacts) => this.materializeHuggingFaceArtifacts(artifacts),
    });
  }

  private async getLoadedState(): Promise<LoadedExecutorState> {
    if (this.disposed) {
      throw new Error(`NeMo TDT executor is disposed for "${this.modelId}".`);
    }
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
    options: NemoTdtTranscriptionOptions,
    _context: NemoDecodeContext<NemoTdtModelConfig>,
  ): Promise<NemoTdtNativeTranscript> {
    const transcriptionStart = nowMs();
    const computeConfidence = options.returnConfidence ?? true;
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

    const encoderOutput = extractAndResolveNemoTdtEncoderOutput(encoderOutputs, this.config);
    const vocabSize = loaded.tokenizer.vocabSize;
    const blankId = loaded.tokenizer.blankId ?? this.config.tokenizer.blankTokenId ?? 0;

    const decodeResult = await runNemoTdtDecodeLoop({
      modelId: this.modelId,
      backendId: this.backendId,
      config: this.config,
      options,
      ort: loaded.ort,
      decoderSession: loaded.decoderSession,
      vocabSize,
      blankId,
      computeConfidence,
      encoderOutput,
      batchState: this,
      warnings,
      transcriptionStart,
      preprocessMs,
      encodeMs,
    });

    const {
      tokenIds,
      tokenTimestamps,
      tokenConfidences,
      tokenFrameIndices,
      tokenLogProbs,
      tokenTdtSteps,
      frameConfidenceStats,
      decodeIterations,
      decoderGridBatchRuns,
      decodeMs,
      decoderStateSnapshot,
    } = decodeResult;
    const { frameCount } = encoderOutput;

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
      totalMs,
      wallMs: totalMs,
      audioDurationSec: roundMetric(audio.durationSeconds, 4),
      rtf: roundMetric(rtf, 4),
      rtfx: rtfx !== undefined ? roundMetric(rtfx, 4) : undefined,
      requestedPreprocessorBackend: this.sourceOptions?.preprocessorBackend ?? 'onnx',
      preprocessorBackend: loaded.preprocessorBackend,
      encoderBackend: loaded.encoderBackend,
      decoderBackend: loaded.decoderBackend,
      jointBackend: loaded.decoderBackend,
      encoderFrameCount: frameCount,
      decodeIterations,
      decoderGridBatchRuns,
      emittedTokenCount: tokenIds.length,
      emittedWordCount: details.words?.length,
    };

    const transcript: NemoTdtNativeTranscript = {
      utteranceText: text,
      isFinal: true,
      words: details.words,
      tokens: details.tokens,
      confidence: computeConfidence
        ? {
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
          }
        : undefined,
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
        encoderBackend: totalMetrics.encoderBackend,
        decoderBackend: totalMetrics.decoderBackend,
        jointBackend: totalMetrics.jointBackend,
        encoderFrameCount: totalMetrics.encoderFrameCount,
        decodeIterations: totalMetrics.decodeIterations,
        decoderGridBatchRuns: totalMetrics.decoderGridBatchRuns,
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
      decoderState: decoderStateSnapshot,
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

  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.disposePromise = this.flushDispose();
    return this.disposePromise;
  }

  private async flushDispose(): Promise<void> {
    let loaded: LoadedExecutorState | undefined;
    if (this.loadStatePromise) {
      try {
        loaded = await this.loadStatePromise;
      } catch {
        // initialize() already settled partial sessions; retain its load error.
      }
    }
    const results: PromiseSettledResult<void>[] = loaded
      ? await Promise.allSettled([
          Promise.resolve().then(() => releaseOrtSession(loaded.encoderSession)),
          Promise.resolve().then(() => releaseOrtSession(loaded.decoderSession)),
          Promise.resolve().then(() => loaded.preprocessor.dispose?.()),
        ])
      : [];
    const handles = this.assetHandles.splice(0);
    results.push(
      ...(await Promise.allSettled(
        handles.map((handle) => Promise.resolve().then(() => handle.dispose())),
      )),
    );
    const failures = results.flatMap((result) =>
      result.status === 'rejected' ? [result.reason] : [],
    );
    if (failures.length)
      throw new AggregateError(failures, `NeMo TDT disposal failed for "${this.modelId}".`);
  }
}
