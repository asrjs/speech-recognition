import type {
  AbortSignalLike,
  AssetProvider,
  AudioBufferLike,
  ModelClassification,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptMetrics,
  TranscriptWarning,
  TranscriptionProgressEvent,
} from '../../types/index.js';
import { argmax, confidenceFromLogits } from '../../inference/index.js';
import { fetchModelFiles } from '../../runtime/huggingface.js';
import { nowMs, roundMetric } from '../../runtime/timing.js';
import { PipelineAbortedError } from '../../pipeline/composition.js';
import type { NemoDecodeContext } from '../nemo-common/index.js';
import {
  createOrtSession,
  disposeOrtOutputs,
  initOrt,
  releaseOrtSession,
  resolveNemoRnntArtifacts,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from './ort.js';
import {
  JsNemoPreprocessor,
  type NemoPreprocessor,
  OnnxNemoPreprocessor,
} from '../nemo-tdt/preprocessor.js';
import { ParakeetTokenizer } from '../nemo-tdt/tokenizer.js';
import { rethrowIfAssetAborted } from '../../io/abort.js';
import { buildEmptyTranscript, buildRnntTranscriptDetails } from './transcript-details.js';
import { getDefaultNemoRnntWeightSetup } from './weights.js';
import type {
  NemoRnntExecutor,
  NemoRnntModelConfig,
  NemoRnntModelOptions,
  NemoRnntNativeTranscript,
  NemoRnntTranscriptionOptions,
} from './types.js';

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: ParakeetTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly decoderSession: OrtSessionLike;
  readonly preprocessor: NemoPreprocessor;
  readonly preprocessorBackend: string;
  readonly warnings: readonly TranscriptWarning[];
  readonly decoderQuantization?: 'fp32' | 'fp16' | 'int8';
}

function decoderQuantizationFromFilename(
  filename: string | undefined,
): 'fp32' | 'fp16' | 'int8' | undefined {
  if (!filename) {
    return undefined;
  }
  if (filename.includes('.int8.')) {
    return 'int8';
  }
  if (filename.includes('.fp16.')) {
    return 'fp16';
  }
  if (filename.endsWith('.onnx')) {
    return 'fp32';
  }
  return undefined;
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
  options: NemoRnntTranscriptionOptions,
  event: TranscriptionProgressEvent,
): void {
  options.onProgress?.(event);
}

function throwIfDecodeAborted(signal: AbortSignalLike | null | undefined): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError('decode');
  }
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
  return String(path || '')
    .replace(/^\.\/+/, '')
    .replace(/\\/g, '/');
}

function hasListedRepoFile(files: readonly string[], filename: string): boolean {
  const target = normalizeRepoPath(filename);
  return files.some((path) => {
    const normalized = normalizeRepoPath(path);
    return normalized === target || normalized.endsWith(`/${target}`);
  });
}

function extractLatestLogitsSlice(
  tensor: OrtTensorLike<Float32Array>,
  distributionSize: number,
): Float32Array {
  const dims = [...tensor.dims];
  const lastAxis = dims.at(-1) ?? tensor.data.length;
  const sliceSize = lastAxis > 0 ? lastAxis : tensor.data.length;
  const sliceOffset = Math.max(0, tensor.data.length - sliceSize);
  const slice = tensor.data.subarray(sliceOffset, sliceOffset + sliceSize);
  if (slice.length < distributionSize) {
    throw new Error(
      `NeMo RNNT decoder output is too small (${slice.length}) for required distribution size ${distributionSize}.`,
    );
  }

  return slice.subarray(0, distributionSize);
}

export class OrtNemoRnntExecutor implements NemoRnntExecutor {
  private readonly sourceOptions: NemoRnntModelOptions['source'];
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private disposed = false;
  private disposePromise?: Promise<void>;
  private rnntBatchAllowed = true;
  private static readonly rnntBatchMinFrames = 2;
  private static readonly rnntBatchMaxFrames = 32;
  private rnntBatchWidth = OrtNemoRnntExecutor.rnntBatchMinFrames;
  private static readonly rnntBatchMinSampleColumns = 24;
  private static readonly rnntBatchMinColumnUtilization = 0.7;
  private rnntBatchColumnsScored = 0;
  private rnntBatchColumnsUseful = 0;

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: NemoRnntModelConfig,
    private readonly backendId: string,
    loadOptions: NemoRnntModelOptions | undefined,
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
    artifacts: ReturnType<typeof resolveNemoRnntArtifacts>['artifacts'],
  ): Promise<typeof artifacts> {
    const source = this.sourceOptions;
    if (!this.assetProvider || !source || source.kind !== 'huggingface') {
      return artifacts;
    }

    const revision = source.revision ?? 'main';
    let repoFilesPromise: Promise<readonly string[]> | null = null;
    const getRepoFiles = (): Promise<readonly string[]> => {
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

      const handle = await this.assetProvider!.resolve({
        id: `huggingface:${source.repoId}:${revision}:${filename}`,
        provider: 'huggingface',
        repoId: source.repoId,
        revision,
        filename,
        preferBlobUrl: true,
        cacheKey,
        cacheKeyFallbacks,
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
        if (isAssetMissingError(error)) {
          return undefined;
        }
        throw error;
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

  private async initialize(): Promise<LoadedExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const resolved = resolveNemoRnntArtifacts(this.sourceOptions, this.backendId);
    let artifacts = await this.materializeHuggingFaceArtifacts(resolved.artifacts);
    if (resolved.preprocessorBackend === 'onnx' && !artifacts.preprocessorUrl) {
      throw new Error(
        `The NeMo RNNT source for "${this.modelId}" does not provide a preprocessor model.`,
      );
    }

    const ort = await initOrt(resolved.ortBackend, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
      signal: this.signal,
    });
    const tokenizer = await ParakeetTokenizer.fromUrl(artifacts.tokenizerUrl, {
      blankId: this.config.tokenizer.blankTokenId,
      signal: this.signal,
    });
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
      rethrowIfAssetAborted(error);
      const implicitFp16Encoder =
        this.sourceOptions.kind === 'huggingface' &&
        !this.sourceOptions.encoderQuant &&
        resolved.encoderBackendForOrt === 'webgpu' &&
        getDefaultNemoRnntWeightSetup(resolved.encoderBackendForOrt).encoderDefault === 'fp16';

      if (!implicitFp16Encoder) {
        throw error;
      }

      const fallbackResolved = resolveNemoRnntArtifacts(
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
        code: 'nemo-rnnt.encoder-fp16-fallback',
        message:
          'Default FP16 encoder weights could not be initialized on this WebGPU setup. Falling back to FP32 encoder weights.',
        recoverable: true,
      });
    }

    if (this.disposed) {
      releaseOrtSession(encoderSession);
      throw new Error(`NeMo RNNT executor was disposed during load for "${this.modelId}".`);
    }

    const decoderSession = await createOrtSession(ort, artifacts.decoderUrl, {
      backendId: resolved.decoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: artifacts.decoderDataUrl,
      externalDataPath: artifacts.decoderFilename ? `${artifacts.decoderFilename}.data` : undefined,
    });
    if (this.disposed) {
      releaseOrtSession(encoderSession);
      releaseOrtSession(decoderSession);
      throw new Error(`NeMo RNNT executor was disposed during load for "${this.modelId}".`);
    }
    const preprocessor: NemoPreprocessor =
      resolved.preprocessorBackend === 'js'
        ? new JsNemoPreprocessor({
            melBins: this.config.melBins,
            validLengthMode: this.config.preprocessorValidLengthMode,
            normalization: this.config.preprocessorNormalization,
          })
        : new OnnxNemoPreprocessor(ort, artifacts.preprocessorUrl!, resolved.enableProfiling, this.signal);

    return {
      ort,
      tokenizer,
      encoderSession,
      decoderSession,
      preprocessor,
      preprocessorBackend: resolved.preprocessorBackend,
      warnings,
      decoderQuantization: decoderQuantizationFromFilename(artifacts.decoderFilename),
    };
  }

  private async getLoadedState(): Promise<LoadedExecutorState> {
    if (this.disposed) {
      throw new Error(`NeMo RNNT executor is disposed for "${this.modelId}".`);
    }
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
  ) {
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
    options: NemoRnntTranscriptionOptions,
    _context: NemoDecodeContext<NemoRnntModelConfig>,
  ): Promise<NemoRnntNativeTranscript> {
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
        code: 'nemo-rnnt.sample-rate-mismatch',
        message: `Expected ${this.config.sampleRate} Hz audio but received ${audio.sampleRate} Hz. No resampler is wired into the restored RNNT path yet.`,
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

    let encoderData: Float32Array;
    let dims: number[];
    try {
      const encoderTensor = (encoderOutputs.outputs ??
        Object.values(encoderOutputs)[0]) as OrtTensorLike<Float32Array>;
      dims = [...encoderTensor.dims];
      if (dims.length !== 3 || dims[0] !== 1) {
        throw new Error(`Unexpected NeMo RNNT encoder output shape: [${dims.join(', ')}].`);
      }
      encoderData = encoderTensor.data instanceof Float32Array
        ? new Float32Array(encoderTensor.data)
        : Float32Array.from(encoderTensor.data as ArrayLike<number>);
    } finally {
      disposeOrtOutputs(encoderOutputs);
    }

    // The exported NeMo RNNT encoder uses [batch, features, frames] (BDT).
    // Inferring layout from dimension sizes breaks once long utterances make T > D.
    const featureSize = dims[1]!;
    const frameCount = dims[2]!;
    if (featureSize <= 0 || frameCount <= 0) {
      throw new Error(`Unexpected NeMo RNNT encoder output shape: [${dims.join(', ')}].`);
    }
    const frameTime = this.config.frameShiftSeconds * this.config.subsamplingFactor;
    const tokenizerVocabSize = loaded.tokenizer.vocabSize;
    const blankId =
      loaded.tokenizer.blankId ?? this.config.tokenizer.blankTokenId ?? tokenizerVocabSize;
    const distributionSize = Math.max(tokenizerVocabSize, blankId + 1);
    if (this.config.vocabularySize && this.config.vocabularySize !== tokenizerVocabSize) {
      warnings.push({
        code: 'nemo-rnnt.vocabulary-size-mismatch',
        message: `Configured vocabulary size ${this.config.vocabularySize} does not match tokenizer vocabulary size ${tokenizerVocabSize}. Using tokenizer vocabulary size for emitted token decoding.`,
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

    const predictionLayers = this.config.predictionLayers ?? 1;
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
    const tokenConfidences: number[] = [];
    const tokenFrameIndices: number[] = [];
    const tokenLogProbs: number[] = [];
    const frameConfidenceStats = new Map<number, { sum: number; count: number }>();
    let decodeIterations = 0;
    let decoderGridBatchRuns = 0;
    this.rnntBatchColumnsScored = 0;
    this.rnntBatchColumnsUseful = 0;

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

    try {
      const decoderStart = nowMs();
      let lastReportedDecodeUnits = -1;
      try {
        let frameIndex = 0;
        let emittedOnFrame = 0;
        while (frameIndex < frameCount) {
          throwIfDecodeAborted(options.signal);
          // Real-artifact A/B (eou-120m v1, Node WASM): fp32 decoder
          // wins ~16-18% (jfk-short 216 vs 265 ms; tr-tdk-18s 342 vs 408
          // ms) with identical transcripts. int8 regressed on jfk-short
          // (~40%) and produced a DIFFERENT transcript on tr-tdk-18s
          // (dynamic-range requantization spans the wider
          // [1, features, width] input, so rows are not independent).
          // fp16 was parity at best. Default: grid batching ON for fp32
          // decoders only; other quants stay sequential unless explicitly
          // opted in via options.gridBatching.
          const gridBatchingEnabled =
            options.gridBatching ?? loaded.decoderQuantization === 'fp32';
          if (
            this.rnntBatchAllowed &&
            gridBatchingEnabled &&
            frameCount - frameIndex > 1 &&
            (this.rnntBatchColumnsScored <
              OrtNemoRnntExecutor.rnntBatchMinSampleColumns ||
              this.rnntBatchColumnsUseful >=
                OrtNemoRnntExecutor.rnntBatchMinColumnUtilization *
                  this.rnntBatchColumnsScored)
          ) {
            const width = Math.min(
              frameCount - frameIndex,
              this.rnntBatchWidth,
              OrtNemoRnntExecutor.rnntBatchMaxFrames,
            );
            const batchData = new Float32Array(featureSize * width);
            // The encoder output is [1, features, frames] (BDT) and the
            // batch tensor is [1, features, width], so the flat layout is
            // feature-major: element (f, w) lives at f * width + w.
            for (let fIdx = 0; fIdx < featureSize; fIdx += 1) {
              const sourceRow = fIdx * frameCount + frameIndex;
              batchData.set(encoderData.subarray(sourceRow, sourceRow + width), fIdx * width);
            }
            const batchEncoderTensor = new loaded.ort.Tensor('float32', batchData, [
              1,
              featureSize,
              width,
            ]);
            const batchTargetBuffer = new Int32Array(1);
            batchTargetBuffer[0] =
              tokenIds.length > 0 ? tokenIds[tokenIds.length - 1]! : blankId;
            const batchTargetTensor = new loaded.ort.Tensor('int32', batchTargetBuffer, [1, 1]);
            const batchLengthTensor = new loaded.ort.Tensor('int32', new Int32Array([1]), [1]);
            let batchOutputs: Record<string, OrtTensorLike> | undefined;
            try {
              const batchFeeds: Record<string, unknown> = {
                encoder_outputs: batchEncoderTensor,
                targets: batchTargetTensor,
                input_states_1: decoderState?.state1,
                input_states_2: decoderState?.state2,
              };
              if (
                !loaded.decoderSession.inputNames ||
                loaded.decoderSession.inputNames.includes('target_length')
              ) {
                batchFeeds.target_length = batchLengthTensor;
              }
              batchOutputs = await loaded.decoderSession.run(batchFeeds);
              const gridLogits = batchOutputs.outputs as OrtTensorLike<Float32Array>;
              const gridRawData = gridLogits.data;
              const gridTotal = gridRawData instanceof Float32Array
                ? gridRawData
                : Float32Array.from(gridRawData as ArrayLike<number>);
              // Row layout is [width, extras, distribution] row-major: row w
              // occupies w * rowLength elements, and the sequential path
              // reads the last distribution slice of its [1,1,2,dist]
              // output, so read the same trailing slice per row. Verified
              // slot-exact against single-frame runs for the eou-120m v1
              // fp32/fp16 graphs (int8 rows differ in logit scale only,
              // argmax parity held across targets and states).
              const rowLength = Math.floor(gridTotal.length / width);
              if (rowLength <= distributionSize || gridTotal.length !== rowLength * width) {
                this.rnntBatchAllowed = false;
              } else {
                decoderGridBatchRuns += 1;
                let examined = 0;
                let emissionRow = -1;
                let emissionToken = 0;
                let emissionConfidence: ReturnType<typeof confidenceFromLogits> | undefined;
                while (examined < width) {
                  // Count every row the scan consumes (including the
                  // eventual emission row) so decodeIterations and the
                  // utilization gate match the sequential joint-call count.
                  const rowIndex = examined;
                  examined += 1;
                  const rowStart = rowIndex * rowLength;
                  const rowView = gridTotal.subarray(
                    rowStart + rowLength - distributionSize,
                    rowStart + rowLength,
                  );
                  const tokenId = argmax(rowView, 0, distributionSize);
                  const confidence = confidenceFromLogits(rowView, tokenId, distributionSize);
                  const frame = frameIndex + rowIndex;
                  const existingFrameConfidence = frameConfidenceStats.get(frame);
                  if (existingFrameConfidence) {
                    existingFrameConfidence.sum += confidence.confidence;
                    existingFrameConfidence.count += 1;
                  } else {
                    frameConfidenceStats.set(frame, {
                      sum: confidence.confidence,
                      count: 1,
                    });
                  }
                  if (tokenId === blankId) {
                    continue;
                  }
                  emissionRow = rowIndex;
                  emissionToken = tokenId;
                  emissionConfidence = confidence;
                  break;
                }
                decodeIterations += examined;
                this.rnntBatchColumnsScored += width;
                this.rnntBatchColumnsUseful += examined;
                if (emissionRow >= 0) {
                  this.rnntBatchWidth = OrtNemoRnntExecutor.rnntBatchMinFrames;
                  const emissionFrame = frameIndex + emissionRow;
                  const nextState = {
                    state1: batchOutputs.output_states_1 as OrtTensorLike<Float32Array>,
                    state2: batchOutputs.output_states_2 as OrtTensorLike<Float32Array>,
                  };
                  disposeDecoderState(decoderState, nextState);
                  decoderState = nextState;
                  delete batchOutputs.output_states_1;
                  delete batchOutputs.output_states_2;
                  tokenIds.push(emissionToken);
                  tokenConfidences.push(emissionConfidence!.confidence);
                  tokenLogProbs.push(emissionConfidence!.logProb);
                  tokenFrameIndices.push(emissionFrame);
                  emittedOnFrame += 1;
                  // Blank rows before the emission row are consumed frames;
                  // resume at the emitting frame (or advance past it when
                  // the per-frame symbol cap is reached), mirroring the
                  // sequential loop exactly.
                  if (emittedOnFrame >= (this.config.maxSymbolsPerStep ?? 10)) {
                    frameIndex = emissionFrame + 1;
                    emittedOnFrame = 0;
                  } else {
                    frameIndex = emissionFrame;
                  }
                } else {
                  frameIndex += width;
                  emittedOnFrame = 0;
                  this.rnntBatchWidth = Math.min(
                    width * 2,
                    OrtNemoRnntExecutor.rnntBatchMaxFrames,
                  );
                }
              }
            } catch (error) {
              if (error instanceof PipelineAbortedError) {
                throw error;
              }
              this.rnntBatchAllowed = false;
            } finally {
              batchTargetTensor.dispose?.();
              batchLengthTensor.dispose?.();
              batchEncoderTensor.dispose?.();
              if (batchOutputs) {
                for (const tensor of Object.values(batchOutputs)) {
                  disposeTensor(tensor);
                }
              }
            }
            if (this.rnntBatchAllowed) {
              continue;
            }
          }
          for (let featureIndex = 0; featureIndex < featureSize; featureIndex += 1) {
            encoderFrameBuffer[featureIndex] =
            encoderData[featureIndex * frameCount + frameIndex] ?? 0;
          }
          while (emittedOnFrame < (this.config.maxSymbolsPerStep ?? 10)) {
            throwIfDecodeAborted(options.signal);
            decodeIterations += 1;
            targetIdBuffer[0] = tokenIds.length > 0 ? tokenIds[tokenIds.length - 1]! : blankId;
            const decoderFeeds: Record<string, unknown> = {
              encoder_outputs: encoderFrameTensor,
              targets: targetTensor,
              input_states_1: decoderState?.state1,
              input_states_2: decoderState?.state2,
            };
            if (
              !loaded.decoderSession.inputNames ||
              loaded.decoderSession.inputNames.includes('target_length')
            ) {
              decoderFeeds.target_length = targetLengthTensor;
            }
            const decoderOutputs = await loaded.decoderSession.run(decoderFeeds);

            const logits = decoderOutputs.outputs as OrtTensorLike<Float32Array>;
            const nextState = {
              state1: decoderOutputs.output_states_1 as OrtTensorLike<Float32Array>,
              state2: decoderOutputs.output_states_2 as OrtTensorLike<Float32Array>,
            };
            let transferredNextState = false;
            try {
              const logitsSlice = extractLatestLogitsSlice(logits, distributionSize);
              const tokenId = argmax(logitsSlice, 0, distributionSize);
              const confidence = confidenceFromLogits(logitsSlice, tokenId, distributionSize);

              const existingFrameConfidence = frameConfidenceStats.get(frameIndex);
              if (existingFrameConfidence) {
                existingFrameConfidence.sum += confidence.confidence;
                existingFrameConfidence.count += 1;
              } else {
                frameConfidenceStats.set(frameIndex, { sum: confidence.confidence, count: 1 });
              }

              if (tokenId === blankId) {
                break;
              }

              tokenIds.push(tokenId);
              tokenConfidences.push(confidence.confidence);
              tokenFrameIndices.push(frameIndex);
              tokenLogProbs.push(confidence.logProb);
              emittedOnFrame += 1;

              disposeDecoderState(decoderState, nextState);
              decoderState = nextState;
              transferredNextState = true;
            } finally {
              if (!transferredNextState) {
                disposeDecoderState(nextState);
              }
              disposeTensor(logits);
              for (const tensor of Object.values(decoderOutputs)) {
                if (tensor !== logits && tensor !== nextState.state1 && tensor !== nextState.state2) {
                  disposeTensor(tensor);
                }
              }
            }

            if (!transferredNextState) {
              break;
            }
          }

          const completedUnits = Math.min(frameCount, frameIndex + 1);
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
          emittedOnFrame = 0;
          frameIndex += 1;
        }
      } finally {
        targetTensor.dispose?.();
        targetLengthTensor.dispose?.();
        encoderFrameTensor.dispose?.();
      }
      const decodeMs = nowMs() - decoderStart;

      const tokenizeStart = nowMs();
      const details = buildRnntTranscriptDetails(
        loaded.tokenizer,
        tokenIds,
        tokenFrameIndices,
        tokenConfidences,
        tokenLogProbs,
        {
          frameTimeSeconds: frameTime,
          eouTokenId: loaded.tokenizer.getTokenId('<EOU>'),
          eobTokenId: loaded.tokenizer.getTokenId('<EOB>'),
        },
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
      const visibleTokenConfidences =
        details.tokens?.flatMap((token) =>
          typeof token.confidence === 'number' ? [token.confidence] : [],
        ) ?? [];
      const utteranceConfidence =
        visibleTokenConfidences.length > 0
          ? visibleTokenConfidences.reduce((sum, value) => sum + value, 0) /
            visibleTokenConfidences.length
          : undefined;
      const wordAverage =
        details.words && details.words.length > 0
          ? details.words.reduce((sum, word) => sum + (word.confidence ?? 0), 0) /
            details.words.length
          : undefined;
      const tokenAverage =
        visibleTokenConfidences.length > 0
          ? visibleTokenConfidences.reduce((sum, value) => sum + value, 0) /
            visibleTokenConfidences.length
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
        encoderFrameCount: frameCount,
        decodeIterations,
        decoderGridBatchRuns,
        emittedTokenCount: details.tokens?.length,
        emittedWordCount: details.words?.length,
      };

      const transcript: NemoRnntNativeTranscript = {
        utteranceText: details.utteranceText,
        rawUtteranceText: details.rawUtteranceText,
        isFinal: true,
        words: details.words,
        tokens: details.tokens,
        specialTokens: details.specialTokens,
        control: details.control,
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
          decoderGridBatchRuns: totalMetrics.decoderGridBatchRuns,
          emittedTokenCount: totalMetrics.emittedTokenCount,
          emittedWordCount: totalMetrics.emittedWordCount,
        },
        warnings,
        debug: {
          tokenIds: options.returnTokenIds ? tokenIds : undefined,
          frameIndices: options.returnFrameIndices ? tokenFrameIndices : undefined,
          logProbs: options.returnLogProbs ? tokenLogProbs : undefined,
        },
        decoderState: options.returnDecoderState
          ? this.snapshotDecoderState(decoderState)
          : undefined,
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
    } finally {
      disposeTensor(decoderState?.state1);
      disposeTensor(decoderState?.state2);
    }
  }

  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.disposePromise = this.flushDispose();
    return this.disposePromise;
  }

  private async flushDispose(): Promise<void> {
    if (this.loadStatePromise) {
      try {
        const loaded = await this.loadStatePromise;
        releaseOrtSession(loaded.encoderSession);
        releaseOrtSession(loaded.decoderSession);
        await loaded.preprocessor.dispose?.();
      } catch {
        // Keep the original load error; still drop asset handles.
      }
    }
    const handles = this.assetHandles.splice(0);
    await Promise.all(handles.map((handle) => Promise.resolve(handle.dispose())));
  }
}
