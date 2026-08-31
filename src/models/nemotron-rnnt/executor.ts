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
import { nowMs, roundMetric } from '../../runtime/timing.js';
import { JsNemoPreprocessor } from '../nemo-tdt/preprocessor.js';
import { ParakeetTokenizer } from '../nemo-tdt/tokenizer.js';
import {
  createOrtSession,
  disposeOrtOutputs,
  initOrt,
  releaseOrtSession,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from '../nemo-tdt/ort.js';
import {
  resolveNemotronRnntArtifacts,
  type ResolvedNemotronRnntArtifacts,
  type ResolvedNemotronRnntOptions,
} from './ort.js';
import type {
  NemotronRnntArtifactSource,
  NemotronRnntExecutor,
  NemotronRnntModelConfig,
  NemotronRnntNativeTranscript,
  NemotronRnntNativeWord,
  NemotronRnntTranscriptionOptions,
} from './types.js';
import type { NemoDecodeContext } from '../nemo-common/index.js';
import { throwIfAssetAborted } from '../../io/abort.js';
import {
  aggregateNemotronRnntFrameConfidences,
  buildEmptyNemotronRnntTranscript,
  buildNemotronRnntTranscriptDetails,
  withNemotronRnntControl,
} from './transcript-details.js';

interface LoadedNemotronExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: ParakeetTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly predictorSession: OrtSessionLike;
  readonly jointSession: OrtSessionLike;
  readonly preprocessor: JsNemoPreprocessor;
  readonly warnings: readonly TranscriptWarning[];
  readonly promptIdDefault: number;
  readonly options: ResolvedNemotronRnntOptions;
  readonly assetHandles: readonly ResolvedAssetHandle[];
}

type Float32Buffer = Float32Array<ArrayBuffer>;
type BigInt64Buffer = BigInt64Array<ArrayBuffer>;

function emitTranscriptionProgress(
  options: NemotronRnntTranscriptionOptions,
  event: TranscriptionProgressEvent,
): void {
  options.onProgress?.(event);
}

function throwIfDecodeAborted(signal: AbortSignalLike | null | undefined): void {
  throwIfAssetAborted(signal);
}

function buildEmptyTranscript(
  warnings: readonly TranscriptWarning[],
): NemotronRnntNativeTranscript {
  return buildEmptyNemotronRnntTranscript(warnings);
}

function copyTensorData(
  tensor: OrtTensorLike<Float32Array> | OrtTensorLike,
): Float32Buffer {
  const data = tensor.data;
  if (data instanceof Float32Array) {
    const view = data as Float32Array<ArrayBuffer>;
    const out = new Float32Array(new ArrayBuffer(view.byteLength));
    out.set(view);
    return out;
  }
  if (data instanceof ArrayBuffer) {
    return new Float32Array(data.slice(0));
  }
  throw new Error('copyTensorData: unsupported tensor data shape.');
}

function copyBigInt64Data(tensor: OrtTensorLike): BigInt64Buffer {
  const data = tensor.data;
  if (data instanceof BigInt64Array) {
    const view = data as BigInt64Array<ArrayBuffer>;
    const out = new BigInt64Array(new ArrayBuffer(view.byteLength));
    out.set(view);
    return out;
  }
  if (data instanceof ArrayBuffer) {
    return new BigInt64Array(data.slice(0));
  }
  throw new Error('copyBigInt64Data: unsupported tensor data shape.');
}
interface RowConfidence {
  confidence: number;
  logProb: number;
}

function rowConfidence(row: Float32Array, emittedIdx: number): RowConfidence {
  let max = -Infinity;
  for (let i = 0; i < row.length; i += 1) {
    const x = row[i];
    if (x !== undefined && x > max) max = x;
  }
  if (!Number.isFinite(max)) {
    return { confidence: 0, logProb: -Infinity };
  }
  let sum = 0;
  for (let i = 0; i < row.length; i += 1) {
    const x = row[i];
    if (x !== undefined) sum += Math.exp(x - max);
  }
  const logZ = Math.log(sum) + max;
  const emittedLogit = row[emittedIdx] ?? 0;
  const logProb = logZ - emittedLogit;
  return { confidence: Math.exp(-logProb), logProb };
}

export class OrtNemotronRnntExecutor implements NemotronRnntExecutor {
  private loadStatePromise?: Promise<LoadedNemotronExecutorState>;
  private disposed = false;
  private disposePromise?: Promise<void>;
  private readonly assetHandles: ResolvedAssetHandle[] = [];

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: NemotronRnntModelConfig,
    private readonly backendId: string,
    private readonly sourceOptions: NemotronRnntArtifactSource | undefined,
    private readonly dependencies: {
      readonly assetProvider?: AssetProvider;
      readonly runtimeHooks?: SpeechRuntimeHooks;
      readonly signal?: AbortSignalLike | null;
    } = {},
  ) {
    if (this.sourceOptions) {
      this.loadStatePromise = this.initialize(this.dependencies.signal ?? null);
    }
  }

  /**
   * Materializes HuggingFace artifacts through the runtime asset
   * provider (cache + progress + abort) and rewrites the resolved URLs
   * to the cached locations. Direct sources pass through untouched.
   */
  private async materializeHuggingFaceArtifacts(
    artifacts: ResolvedNemotronRnntArtifacts,
  ): Promise<ResolvedNemotronRnntArtifacts> {
    const source = this.sourceOptions;
    if (!this.dependencies.assetProvider || !source || source.kind !== 'huggingface') {
      return artifacts;
    }

    const revision = source.revision ?? 'main';
    const resolveFile = async (filename: string): Promise<string> => {
      const cacheKey = `huggingface:${source.repoId}:${revision}:${filename}`;
      const cacheKeyFallbacks = (source.cacheKeyFallbackRevisions ?? [])
        .filter((fallbackRevision) => fallbackRevision !== revision)
        .map(
          (fallbackRevision) =>
            `huggingface:${source.repoId}:${fallbackRevision}:${filename}`,
        );
      const handle = await this.dependencies.assetProvider!.resolve({
        id: `huggingface:${source.repoId}:${revision}:${filename}`,
        provider: 'huggingface',
        repoId: source.repoId,
        revision,
        filename,
        preferBlobUrl: true,
        cacheKey,
        cacheKeyFallbacks,
        onProgress: (event) => {
          this.dependencies.runtimeHooks?.onProgress?.({
            phase: 'asset:progress',
            modelId: this.modelId,
            file: filename,
            ...event,
          } as RuntimeProgressEvent);
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
      encoderUrl: await resolveFile('encoder.onnx'),
      decoderUrl: await resolveFile('decoder.onnx'),
      jointUrl: await resolveFile('joint.onnx'),
      tokenizerUrl: await resolveFile('tokenizer.json'),
      encoderDataUrl: await resolveFile('encoder.onnx.data'),
      decoderDataUrl: await resolveFile('decoder.onnx.data'),
      jointDataUrl: await resolveFile('joint.onnx.data'),
      encoderFilename: 'encoder.onnx',
      decoderFilename: 'decoder.onnx',
      jointFilename: 'joint.onnx',
    };
  }

  private async initialize(
    signal: AbortSignalLike | null,
  ): Promise<LoadedNemotronExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const { artifacts: resolvedArtifacts, options } = resolveNemotronRnntArtifacts(
      this.sourceOptions,
      this.backendId,
    );
    const artifacts = await this.materializeHuggingFaceArtifacts(resolvedArtifacts);
    const warnings: TranscriptWarning[] = [];

    const ort = await initOrt(options.ortBackend, {
      wasmPaths: options.wasmPaths,
      cpuThreads: options.cpuThreads,
      signal,
    });

    const tokenizer = /tokenizer\.json(?:$|[?#])/.test(artifacts.tokenizerUrl)
      ? await ParakeetTokenizer.fromTokenizerJson(artifacts.tokenizerUrl, {
          blankId: this.config.blankTokenId,
          signal,
        })
      : await ParakeetTokenizer.fromUrl(artifacts.tokenizerUrl, {
          blankId: this.config.blankTokenId,
          signal,
        });

    if (tokenizer.vocabSize !== this.config.vocabularySize) {
      warnings.push({
        code: 'nemotron-rnnt.vocabulary-size-mismatch',
        message: `Tokenizer vocabulary size ${tokenizer.vocabSize} does not match config vocabulary size ${this.config.vocabularySize}.`,
        recoverable: true,
      });
    }

    const encoderSession = await createOrtSession(ort, artifacts.encoderUrl, {
      backendId: options.encoderBackend,
      enableProfiling: options.enableProfiling,
      externalDataUrl: artifacts.encoderDataUrl,
      externalDataPath: artifacts.encoderFilename
        ? `${artifacts.encoderFilename}.data`
        : undefined,
    });
    if (this.disposed) {
      releaseOrtSession(encoderSession);
      throw new Error(`Nemotron RNNT executor disposed during load for "${this.modelId}".`);
    }

    const predictorSession = await createOrtSession(ort, artifacts.decoderUrl, {
      backendId: options.encoderBackend,
      enableProfiling: options.enableProfiling,
      externalDataUrl: artifacts.decoderDataUrl,
      externalDataPath: artifacts.decoderFilename
        ? `${artifacts.decoderFilename}.data`
        : undefined,
    });
    if (this.disposed) {
      releaseOrtSession(encoderSession);
      releaseOrtSession(predictorSession);
      throw new Error(`Nemotron RNNT executor disposed during load for "${this.modelId}".`);
    }

    const jointSession = await createOrtSession(ort, artifacts.jointUrl, {
      backendId: options.encoderBackend,
      enableProfiling: options.enableProfiling,
      externalDataUrl: artifacts.jointDataUrl,
      externalDataPath: artifacts.jointFilename
        ? `${artifacts.jointFilename}.data`
        : undefined,
    });
    if (this.disposed) {
      releaseOrtSession(encoderSession);
      releaseOrtSession(predictorSession);
      releaseOrtSession(jointSession);
      throw new Error(`Nemotron RNNT executor disposed during load for "${this.modelId}".`);
    }

    const preprocessor = new JsNemoPreprocessor({
      melBins: this.config.melBins,
      validLengthMode: this.config.preprocessorValidLengthMode,
      normalization: this.config.preprocessorNormalization,
    });

    return {
      ort,
      tokenizer,
      encoderSession,
      predictorSession,
      jointSession,
      preprocessor,
      warnings,
      promptIdDefault: this.config.defaultPromptId,
      options,
      assetHandles: this.assetHandles,
    };
  }

  private async getLoadedState(): Promise<LoadedNemotronExecutorState> {
    if (this.disposed) {
      throw new Error(`Nemotron RNNT executor is disposed for "${this.modelId}".`);
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
    options: NemotronRnntTranscriptionOptions,
    _context: NemoDecodeContext<NemotronRnntModelConfig>,
  ): Promise<NemotronRnntNativeTranscript> {
    const transcriptionStart = nowMs();
    const loaded = await this.getLoadedState();
    const warnings: TranscriptWarning[] = [...loaded.warnings];

    if (audio.sampleRate !== this.config.sampleRate) {
      warnings.push({
        code: 'nemotron-rnnt.sample-rate-mismatch',
        message: `Expected ${this.config.sampleRate} Hz audio but received ${audio.sampleRate} Hz.`,
        recoverable: true,
      });
    }

    emitTranscriptionProgress(options, {
      stage: 'start',
      progress: 0,
      elapsedMs: 0,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Starting transcription for ${this.modelId}.`,
    });

    const preprocessStart = nowMs();
    const processed = loaded.preprocessor.process(audio);
    const preprocessMs = nowMs() - preprocessStart;
    const frameCount = processed.frameCount;
    const validLength = processed.validLength;

    emitTranscriptionProgress(options, {
      stage: 'preprocess',
      progress: 0.2,
      elapsedMs: roundMetric(nowMs() - transcriptionStart),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Prepared audio features for ${this.modelId}.`,
      metrics: { preprocessMs: roundMetric(preprocessMs) },
    });

    if (frameCount === 0 || processed.features.length === 0) {
      const elapsed = roundMetric(nowMs() - transcriptionStart);
      emitTranscriptionProgress(options, {
        stage: 'complete',
        progress: 1,
        elapsedMs: elapsed,
        modelId: this.modelId,
        backendId: this.backendId,
        message: `Finished transcription for ${this.modelId} (silent input).`,
        metrics: { preprocessMs: roundMetric(preprocessMs), totalMs: elapsed },
      });
      return buildEmptyTranscript(warnings);
    }

    const promptId = options.promptId ?? loaded.promptIdDefault;
    const blankId = loaded.tokenizer.blankId ?? this.config.blankTokenId;
    const distributionSize = Math.max(
      loaded.tokenizer.vocabSize,
      this.config.vocabularySize ?? 0,
      blankId + 1,
    );

    const encodeStart = nowMs();
    const encFlat = await this.runStreamingEncoder(
      loaded,
      processed.features,
      frameCount,
      validLength,
      promptId,
      options.signal ?? null,
    );
    const encodeMs = nowMs() - encodeStart;
    const T_enc = encFlat.length / 1024;

    emitTranscriptionProgress(options, {
      stage: 'encode',
      progress: 0.4,
      elapsedMs: roundMetric(nowMs() - transcriptionStart),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Encoded ${T_enc} streaming chunks for ${this.modelId}.`,
      metrics: { preprocessMs: roundMetric(preprocessMs), encodeMs: roundMetric(encodeMs) },
    });

    if (T_enc === 0) {
      const elapsed = roundMetric(nowMs() - transcriptionStart);
      emitTranscriptionProgress(options, {
        stage: 'complete',
        progress: 1,
        elapsedMs: elapsed,
        modelId: this.modelId,
        backendId: this.backendId,
        message: `Finished transcription for ${this.modelId} (no encoder output).`,
        metrics: { preprocessMs: roundMetric(preprocessMs), encodeMs: roundMetric(encodeMs) },
      });
      return buildEmptyTranscript(warnings);
    }

    const decodeStart = nowMs();
    const decodeResult = await this.runGreedyDecoder(
      loaded,
      encFlat,
      blankId,
      distributionSize,
      options.signal ?? null,
    );
    const decodeMs = nowMs() - decodeStart;

    emitTranscriptionProgress(options, {
      stage: 'decode',
      progress: 0.9,
      elapsedMs: roundMetric(nowMs() - transcriptionStart),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Decoded ${decodeResult.tokenIds.length} tokens for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
      },
    });

    const frameTimeSeconds = this.config.frameShiftSeconds * this.config.subsamplingFactor;
    const details = buildNemotronRnntTranscriptDetails(
      loaded.tokenizer,
      decodeResult.tokenIds,
      decodeResult.frameIndices,
      decodeResult.confidences,
      decodeResult.logProbs,
      { frameTimeSeconds },
    );
    const frameConfidences = aggregateNemotronRnntFrameConfidences(
      decodeResult.frameConfidenceStats,
    );

    const tokenAverage = decodeResult.confidences.length > 0
      ? decodeResult.confidences.reduce((a: number, b: number) => a + b, 0) /
        decodeResult.confidences.length
      : undefined;
    const wordsWithConfidence: ReadonlyArray<NemotronRnntNativeWord & { confidence: number }> =
      details.words.filter(
        (w): w is NemotronRnntNativeWord & { confidence: number } =>
          typeof w.confidence === 'number',
      );
    const wordAverage = wordsWithConfidence.length > 0
      ? wordsWithConfidence.reduce(
          (a: number, w) => a + w.confidence,
          0,
        ) / wordsWithConfidence.length
      : undefined;
    const averageLogProb = decodeResult.logProbs.length > 0
      ? decodeResult.logProbs.reduce((a: number, b: number) => a + b, 0) /
        decodeResult.logProbs.length
      : undefined;
    const utteranceConfidence = tokenAverage;
    const totalMs = roundMetric(nowMs() - transcriptionStart);
    const rtf = audio.durationSeconds > 0 ? totalMs / (audio.durationSeconds * 1000) : 0;
    const rtfx = audio.durationSeconds > 0 ? audio.durationSeconds / (totalMs / 1000) : undefined;

    const transcriptBase: NemotronRnntNativeTranscript = {
      utteranceText: details.utteranceText,
      rawUtteranceText: details.rawUtteranceText,
      isFinal: true,
      words: details.words,
      tokens: details.tokens,
      specialTokens: details.specialTokens,
      confidence: {
        utterance: utteranceConfidence,
        tokenAverage,
        wordAverage,
        frameAverage:
          frameConfidences.length > 0
            ? frameConfidences.reduce((a: number, b: number) => a + b, 0) /
              frameConfidences.length
            : undefined,
        averageLogProb,
        frames: frameConfidences,
      },
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
        totalMs,
        wallMs: totalMs,
        audioDurationSec: roundMetric(audio.durationSeconds, 4),
        rtf: roundMetric(rtf, 4),
        rtfx: rtfx !== undefined ? roundMetric(rtfx, 4) : undefined,
        preprocessorBackend: 'js',
        encoderFrameCount: T_enc,
        decodeIterations: decodeResult.iterations,
        emittedTokenCount: decodeResult.tokenIds.length,
        emittedWordCount: details.words.length,
      } as TranscriptMetrics,
      warnings,
      debug: {
        tokenIds: options.returnTokenIds ? decodeResult.tokenIds : undefined,
        frameIndices: options.returnFrameIndices ? decodeResult.frameIndices : undefined,
        logProbs: options.returnLogProbs ? decodeResult.logProbs : undefined,
      },
    };
    const transcript = withNemotronRnntControl(transcriptBase);

    emitTranscriptionProgress(options, {
      stage: 'complete',
      progress: 1,
      elapsedMs: totalMs,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Finished transcription for ${this.modelId}.`,
      metrics: transcript.metrics,
    });

    return transcript;
  }

  private async runStreamingEncoder(
    loaded: LoadedNemotronExecutorState,
    features: Float32Array,
    frameCount: number,
    validLength: number,
    promptId: number,
    signal: AbortSignalLike | null,
  ): Promise<Float32Buffer> {
    const chunkFrames = this.config.chunkFrames;
    const framesPerChunk = this.config.encoderOutputFramesPerChunk;
    const featureDim = this.config.melBins;
    const hiddenDim = 1024;
    const { channelLayers, channelFrames, channelDim, timeFrames, timeDim } =
      this.config.encoderCache;

    const cacheCh = new Float32Array(
      new ArrayBuffer(1 * channelLayers * channelFrames * channelDim * 4),
    );
    const cacheT = new Float32Array(
      new ArrayBuffer(1 * channelLayers * timeDim * timeFrames * 4),
    );
    let liveCh: Float32Buffer = cacheCh;
    let liveT: Float32Buffer = cacheT;
    let liveChLen: BigInt64Buffer = new BigInt64Array(new ArrayBuffer(8));

    const flatChunk = new Float32Array(
      new ArrayBuffer(chunkFrames * featureDim * 4),
    );
    const totalFrames =
      Math.ceil(frameCount / chunkFrames) * framesPerChunk;
    const encFlat = new Float32Array(new ArrayBuffer(totalFrames * hiddenDim * 4));

    const melLengthTensor = new loaded.ort.Tensor(
      'int64',
      BigInt64Array.from([BigInt(validLength)]),
      [1],
    );
    const langIdTensor = new loaded.ort.Tensor(
      'int64',
      BigInt64Array.from([BigInt(promptId)]),
      [1],
    );

    try {
      let chunkIndex = 0;
      for (let melIdx = 0; melIdx < frameCount; melIdx += chunkFrames) {
        throwIfDecodeAborted(signal);

        const take = Math.min(chunkFrames, frameCount - melIdx);
        flatChunk.fill(0);
        // JSMelProcessor emits bin-major layout: element (bin, frame) at
        // bin * frameCount + frame. The encoder wants [1, chunkFrames, bins]
        // (frame-major rows), so transpose each chunk's slice.
        for (let i = 0; i < take; i += 1) {
          const dst = i * featureDim;
          for (let d = 0; d < featureDim; d += 1) {
            flatChunk[dst + d] = features[d * frameCount + (melIdx + i)] ?? 0;
          }
        }

        const audioTensor = new loaded.ort.Tensor('float32', flatChunk, [
          1,
          chunkFrames,
          featureDim,
        ]);
        const cacheChTensor = new loaded.ort.Tensor('float32', liveCh, [
          1,
          channelLayers,
          channelFrames,
          channelDim,
        ]);
        const cacheTTensor = new loaded.ort.Tensor('float32', liveT, [
          1,
          channelLayers,
          timeDim,
          timeFrames,
        ]);
        const cacheChLenTensor = new loaded.ort.Tensor('int64', liveChLen, [1]);

        let encoderOut: Record<string, OrtTensorLike> | undefined;
        try {
          encoderOut = await loaded.encoderSession.run({
            audio_signal: audioTensor,
            length: melLengthTensor,
            cache_last_channel: cacheChTensor,
            cache_last_time: cacheTTensor,
            cache_last_channel_len: cacheChLenTensor,
            lang_id: langIdTensor,
          });

          const encoderTensor = (encoderOut.outputs ??
            Object.values(encoderOut)[0]) as OrtTensorLike<Float32Array>;
          const data = copyTensorData(encoderTensor);
          // Cache-aware encoder emits framesPerChunk rows per chunk;
          // write them contiguously into the flat encoder buffer.
          const chunkOutOffset = chunkIndex * framesPerChunk * hiddenDim;
          for (let f = 0; f < framesPerChunk; f += 1) {
            for (let d = 0; d < hiddenDim; d += 1) {
              encFlat[chunkOutOffset + f * hiddenDim + d] =
                data[f * hiddenDim + d] ?? 0;
            }
          }
          chunkIndex += 1;

          const nextCh = encoderOut.cache_last_channel_next as OrtTensorLike | undefined;
          const nextT = encoderOut.cache_last_time_next as OrtTensorLike | undefined;
          const nextChLen = encoderOut.cache_last_channel_len_next as
            | OrtTensorLike
            | undefined;
          if (nextCh) {
            liveCh = copyTensorData(nextCh);
          }
          if (nextT) {
            liveT = copyTensorData(nextT);
          }
          if (nextChLen) {
            liveChLen = copyBigInt64Data(nextChLen);
          }
        } finally {
          audioTensor.dispose?.();
          cacheChTensor.dispose?.();
          cacheTTensor.dispose?.();
          cacheChLenTensor.dispose?.();
          disposeOrtOutputs(encoderOut);
        }
      }
    } finally {
      melLengthTensor.dispose?.();
      langIdTensor.dispose?.();
    }

    return encFlat;
  }

  private async runGreedyDecoder(
    loaded: LoadedNemotronExecutorState,
    encFlatSrc: Float32Buffer | Float32Array,
    blankId: number,
    distributionSize: number,
    signal: AbortSignalLike | null,
  ): Promise<{
    tokenIds: number[];
    frameIndices: number[];
    confidences: number[];
    logProbs: number[];
    iterations: number;
    frameConfidenceStats: Map<number, { sum: number; count: number }>;
  }> {
    const encFlat = encFlatSrc instanceof Float32Array ? encFlatSrc : encFlatSrc;
    const hiddenDim = 1024;
    const predHidden = this.config.predictionHiddenSize;
    const T_enc = encFlat.length / hiddenDim;
    const targets: number[] = [blankId];
    const tokenIds: number[] = [];
    const tokenFrameIndices: number[] = [];
    const tokenConfidences: number[] = [];
    const tokenLogProbs: number[] = [];
    const frameConfidenceStats = new Map<number, { sum: number; count: number }>();
    let iterations = 0;
    let lastT = 0;

    const h = new Float32Array(new ArrayBuffer(2 * 1 * predHidden * 4));
    const c = new Float32Array(new ArrayBuffer(2 * 1 * predHidden * 4));
    let liveH: Float32Buffer = h;
    let liveC: Float32Buffer = c;

    for (let step = 0; step < this.config.maxDecodeSteps; step += 1) {
      throwIfDecodeAborted(signal);
      iterations += 1;

      const targetsTensor = new loaded.ort.Tensor(
        'int64',
        BigInt64Array.from(targets.map((t) => BigInt(t))),
        [1, targets.length],
      );
      const hTensor = new loaded.ort.Tensor('float32', liveH, [2, 1, predHidden]);
      const cTensor = new loaded.ort.Tensor('float32', liveC, [2, 1, predHidden]);

      let predictorOut: Record<string, OrtTensorLike> | undefined;
      let decoderT: Float32Array;
      let newH: Float32Buffer;
      let newC: Float32Buffer;
      try {
        predictorOut = await loaded.predictorSession.run({
          targets: targetsTensor,
          h_in: hTensor,
          c_in: cTensor,
        });
        const decoderTensor = predictorOut.decoder_output as OrtTensorLike<Float32Array>;
        const U = decoderTensor.dims[2] ?? 1;
        const decoderRaw = copyTensorData(decoderTensor);
        // Layout [1, 640, U] row-major: element (d, u) at d*U + u.
        // Transpose to [U, 640] for the joint input.
        decoderT = new Float32Array(new ArrayBuffer(U * 640 * 4));
        for (let u = 0; u < U; u += 1) {
          for (let d = 0; d < 640; d += 1) {
            decoderT[u * 640 + d] = decoderRaw[d * U + u] ?? 0;
          }
        }
        newH = copyTensorData(predictorOut.h_out as OrtTensorLike<Float32Array>);
        newC = copyTensorData(predictorOut.c_out as OrtTensorLike<Float32Array>);
      } finally {
        targetsTensor.dispose?.();
        hTensor.dispose?.();
        cTensor.dispose?.();
        disposeOrtOutputs(predictorOut);
      }
      liveH = newH;
      liveC = newC;

      const T_remain = T_enc - lastT;
      if (T_remain <= 0) {
        break;
      }

      // Joint logits are position-wise independent given the decoder
      // state, so scanning in fixed-size windows yields the exact same
      // token stream as one joint call over all remaining frames while
      // keeping decode cost linear in utterance length. Each window
      // scans the last decoder column (u = targets.length - 1) for the
      // first non-blank argmax, mirroring the proven Python pipeline.
      const lastU = targets.length - 1;
      const V = distributionSize;
      const windowFrames = Math.max(1, this.config.jointWindowFrames);
      let emittedToken = -1;
      let emittedFrame = -1;
      let emittedConf: RowConfidence | undefined;

      while (lastT < T_enc && emittedToken < 0) {
        throwIfDecodeAborted(signal);
        iterations += 1;

        const windowEnd = Math.min(lastT + windowFrames, T_enc);
        const T_window = windowEnd - lastT;
        const encRemOffset = lastT * hiddenDim;
        const encRemLength = T_window * hiddenDim;
        const encRem = new Float32Array(new ArrayBuffer(encRemLength * 4));
        encRem.set(encFlat.subarray(encRemOffset, encRemOffset + encRemLength));

        const encTensor = new loaded.ort.Tensor('float32', encRem, [
          1,
          T_window,
          hiddenDim,
        ]);
        const decTensor = new loaded.ort.Tensor('float32', decoderT, [
          1,
          targets.length,
          640,
        ]);

        let jointOut: Record<string, OrtTensorLike> | undefined;
        let logits: Float32Array;
        try {
          jointOut = await loaded.jointSession.run({
            encoder_output: encTensor,
            decoder_output: decTensor,
          });
          const logitsTensor = jointOut.joint_output as OrtTensorLike<Float32Array>;
          logits = copyTensorData(logitsTensor);
        } finally {
          encTensor.dispose?.();
          decTensor.dispose?.();
          disposeOrtOutputs(jointOut);
        }

        for (let t = 0; t < T_window; t += 1) {
          const rowBase = t * targets.length * V + lastU * V;
          let maxV = -Infinity;
          let maxIdx = -1;
          for (let v = 0; v < V; v += 1) {
            const x = logits[rowBase + v] ?? -Infinity;
            if (x > maxV) {
              maxV = x;
              maxIdx = v;
            }
          }
          if (maxIdx === blankId) {
            continue;
          }
          emittedToken = maxIdx;
          emittedFrame = lastT + t;
          const row = new Float32Array(new ArrayBuffer(V * 4));
          for (let v = 0; v < V; v += 1) {
            row[v] = logits[rowBase + v] ?? 0;
          }
          emittedConf = rowConfidence(row, maxIdx);
          break;
        }

        if (emittedToken < 0) {
          // Window was all-blank: those frames are consumed. A later
          // window may still produce an emission.
          lastT = windowEnd;
        }
      }

      if (emittedToken < 0) {
        // Frames exhausted without an emission; decoding is complete.
        break;
      }

      tokenIds.push(emittedToken);
      tokenFrameIndices.push(emittedFrame);
      tokenConfidences.push(emittedConf?.confidence ?? 0);
      tokenLogProbs.push(emittedConf?.logProb ?? 0);

      const stats = frameConfidenceStats.get(emittedFrame) ?? {
        sum: 0,
        count: 0,
      };
      stats.sum += emittedConf?.confidence ?? 0;
      stats.count += 1;
      frameConfidenceStats.set(emittedFrame, stats);

      targets.push(emittedToken);

      // Resume AT the emission frame (the next predictor call may
      // extend targets and the same audio frame may produce another
      // non-blank token on the next joint invocation).
      lastT = emittedFrame;

      if (tokenIds.length >= this.config.maxOutputTokens) {
        break;
      }
    }

    return {
      tokenIds,
      frameIndices: tokenFrameIndices,
      confidences: tokenConfidences,
      logProbs: tokenLogProbs,
      iterations,
      frameConfidenceStats,
    };
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
        releaseOrtSession(loaded.predictorSession);
        releaseOrtSession(loaded.jointSession);
      } catch {
        // Keep the original load error; still drop asset handles below.
      }
    }
    const handles = this.assetHandles.splice(0);
    await Promise.all(handles.map((handle) => Promise.resolve(handle.dispose())));
  }
}
