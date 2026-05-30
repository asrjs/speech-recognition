import type {
  AssetProvider,
  AudioBufferLike,
  ModelClassification,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptWarning,
  TranscriptionProgressEvent,
} from '../../types/index.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import {
  addTimesToTokenSpans,
  argmaxAndSelectedLogProbs,
  buildSentenceTimings,
  buildUtteranceTiming,
  buildWordsFromCharSpans,
  ctcCollapseWithSpans,
  estimateSecondsPerOutputFrame,
} from '../../ctc/index.js';
import {
  createOrtSession,
  initOrt,
  resolveWav2Vec2Artifacts,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
  type ResolvedWav2Vec2Artifacts,
} from './ort.js';
import { Wav2Vec2CharTokenizer } from './tokenizer.js';
import type {
  Wav2Vec2ArtifactSource,
  Wav2Vec2Executor,
  Wav2Vec2LogitsResult,
  Wav2Vec2ModelConfig,
  Wav2Vec2ModelDependencies,
  Wav2Vec2ModelOptions,
  Wav2Vec2NativeSegment,
  Wav2Vec2NativeToken,
  Wav2Vec2NativeTranscript,
  Wav2Vec2SentenceTiming,
  Wav2Vec2TokenSpan,
  Wav2Vec2TranscriptionOptions,
  Wav2Vec2UtteranceTiming,
} from './types.js';

// ---------------------------------------------------------------------------
// Loaded state
// ---------------------------------------------------------------------------

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly session: OrtSessionLike;
  readonly tokenizer: Wav2Vec2CharTokenizer;
  readonly config: Wav2Vec2ModelConfig;
  readonly warnings: readonly TranscriptWarning[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clampProgress(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function estimateRemainingMs(elapsedMs: number, progress: number): number | undefined {
  if (progress <= 0 || progress >= 1) {
    return undefined;
  }

  return roundMetric((elapsedMs / progress) * (1 - progress), 2);
}

function emitTranscriptionProgress(
  options: Wav2Vec2TranscriptionOptions,
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

function ensureFloat32Buffer(length: number, buffer?: Float32Array): Float32Array {
  return !buffer || buffer.length < length ? new Float32Array(length) : buffer;
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

// ---------------------------------------------------------------------------
// Audio → mono float32 PCM
// ---------------------------------------------------------------------------

function prepareMonoBuffer(frames: number, destination?: Float32Array): Float32Array {
  const mono = destination
    ? (destination.length === frames ? destination : destination.subarray(0, frames))
    : new Float32Array(frames);
  mono.fill(0);
  return mono;
}

function toMonoPcm(audio: AudioBufferLike, destination?: Float32Array): Float32Array {
  if (audio.channels && audio.channels.length > 0) {
    if (audio.channels.length === 1) {
      return audio.channels[0] ?? new Float32Array(0);
    }

    const mono = prepareMonoBuffer(audio.numberOfFrames, destination);
    const channelCount = audio.channels.length;

    if (channelCount === 2) {
      const left = audio.channels[0];
      const right = audio.channels[1];
      if (left && right) {
        for (let frameIndex = 0; frameIndex < audio.numberOfFrames; frameIndex += 1) {
          mono[frameIndex] = ((left[frameIndex] ?? 0) + (right[frameIndex] ?? 0)) * 0.5;
        }
        return mono;
      }
    }

    const invChannels = 1 / channelCount;
    for (let frameIndex = 0; frameIndex < audio.numberOfFrames; frameIndex += 1) {
      let sampleSum = 0;
      for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
        sampleSum += audio.channels[channelIndex]?.[frameIndex] ?? 0;
      }
      mono[frameIndex] = sampleSum * invChannels;
    }

    return mono;
  }

  const numberOfChannels = Math.max(1, audio.numberOfChannels || 1);
  if (audio.data instanceof Float32Array || audio.data instanceof Float64Array) {
    const frames = Math.floor(audio.data.length / numberOfChannels);
    if (numberOfChannels === 1) {
      return Float32Array.from(audio.data.subarray(0, frames));
    }

    const mono = prepareMonoBuffer(frames, destination);
    const data = audio.data;

    if (numberOfChannels === 2) {
      for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
        const baseIndex = frameIndex * 2;
        mono[frameIndex] = ((data[baseIndex] ?? 0) + (data[baseIndex + 1] ?? 0)) * 0.5;
      }
      return mono;
    }

    const invChannels = 1 / numberOfChannels;
    for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
      let sampleSum = 0;
      const baseIndex = frameIndex * numberOfChannels;
      for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
        sampleSum += data[baseIndex + channelIndex] ?? 0;
      }
      mono[frameIndex] = sampleSum * invChannels;
    }
    return mono;
  }

  if (audio.data instanceof Int16Array) {
    const frames = Math.floor(audio.data.length / numberOfChannels);
    const mono = prepareMonoBuffer(frames, destination);
    const data = audio.data;
    const int16Scale = 1 / 32768;

    if (numberOfChannels === 2) {
      for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
        const baseIndex = frameIndex * 2;
        mono[frameIndex] = ((data[baseIndex] ?? 0) + (data[baseIndex + 1] ?? 0)) * 0.5 * int16Scale;
      }
      return mono;
    }

    const sampleScale = int16Scale / numberOfChannels;
    for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
      let sampleSum = 0;
      const baseIndex = frameIndex * numberOfChannels;
      for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
        sampleSum += data[baseIndex + channelIndex] ?? 0;
      }
      mono[frameIndex] = sampleSum * sampleScale;
    }
    return mono;
  }

  throw new Error('Unsupported audio buffer shape for Wav2Vec2 executor.');
}

// ---------------------------------------------------------------------------
// Logits helpers
// ---------------------------------------------------------------------------

const FLOAT32_BITS_VIEW = new Float32Array(1);
const UINT32_BITS_VIEW = new Uint32Array(FLOAT32_BITS_VIEW.buffer);

function float16BitsToFloat32(value: number): number {
  const sign = (value & 0x8000) << 16;
  const exponent = (value >>> 10) & 0x1f;
  const mantissa = value & 0x03ff;

  if (exponent === 0) {
    if (mantissa === 0) {
      UINT32_BITS_VIEW[0] = sign;
      return FLOAT32_BITS_VIEW[0] ?? 0;
    }

    let normalizedMantissa = mantissa;
    let shift = -1;
    do {
      shift += 1;
      normalizedMantissa <<= 1;
    } while ((normalizedMantissa & 0x0400) === 0);
    normalizedMantissa &= 0x03ff;
    UINT32_BITS_VIEW[0] = sign | ((127 - 15 - shift) << 23) | (normalizedMantissa << 13);
    return FLOAT32_BITS_VIEW[0] ?? 0;
  }

  if (exponent === 0x1f) {
    UINT32_BITS_VIEW[0] = sign | 0x7f800000 | (mantissa << 13);
    return FLOAT32_BITS_VIEW[0] ?? 0;
  }

  UINT32_BITS_VIEW[0] = sign | ((exponent + 112) << 23) | (mantissa << 13);
  return FLOAT32_BITS_VIEW[0] ?? 0;
}

function normalizeLogitsData(logitsTensor: OrtTensorLike): Float32Array {
  const tensorType = logitsTensor.type ?? 'float32';
  if (tensorType !== 'float16') {
    const source = logitsTensor.data as Float32Array;
    return source instanceof Float32Array ? source : Float32Array.from(source);
  }

  const source = logitsTensor.data as Uint16Array;
  const normalized = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    normalized[index] = float16BitsToFloat32(source[index] ?? 0);
  }

  return normalized;
}

function findLogitsTensor(outputs: Record<string, OrtTensorLike>): OrtTensorLike {
  if (outputs.logits) {
    return outputs.logits;
  }

  const first = Object.values(outputs)[0];
  if (!first) {
    throw new Error('Wav2Vec2 encoder run returned no output tensors.');
  }

  return first;
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

export class OrtWav2Vec2Executor implements Wav2Vec2Executor {
  private readonly sourceOptions: Wav2Vec2ModelOptions['source'];
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private sharedMonoBuffer?: Float32Array;

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: Wav2Vec2ModelConfig,
    private readonly backendId: string,
    loadOptions: Wav2Vec2ModelOptions | undefined,
    dependencies: Wav2Vec2ModelDependencies = {},
  ) {
    this.sourceOptions = loadOptions?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;

    if (this.sourceOptions) {
      this.loadStatePromise = this.initialize();
    }
  }

  // -------------------------------------------------------------------------
  // Artifact materialization (HuggingFace)
  // -------------------------------------------------------------------------

  private async materializeHuggingFaceArtifacts(
    source: Extract<Wav2Vec2ArtifactSource, { kind: 'huggingface' }>,
    artifacts: ResolvedWav2Vec2Artifacts['artifacts'],
  ): Promise<{
    readonly artifacts: ResolvedWav2Vec2Artifacts['artifacts'];
    readonly warnings: readonly TranscriptWarning[];
  }> {
    if (!this.assetProvider) {
      return {
        artifacts,
        warnings: [],
      };
    }

    const warnings: TranscriptWarning[] = [];
    const revision = source.revision ?? 'main';
    const resolveFile = async (filename: string, optional = false): Promise<string | undefined> => {
      try {
        const handle = await this.assetProvider!.resolve({
          id: `huggingface:${source.repoId}:${revision}::${filename}`,
          provider: 'huggingface',
          repoId: source.repoId,
          revision,
          filename,
          cacheKey: `huggingface:${source.repoId}:${revision}::${filename}`,
          onProgress: (event) => {
            this.runtimeHooks?.onProgress?.(
              createAssetProgressEvent(this.modelId, filename, {
                loaded: event.loaded,
                total: event.total,
                done: event.done,
              }),
            );
          },
        });
        this.assetHandles.push(handle);
        const locator = await handle.getLocator('url');
        if (!locator) {
          throw new Error(`Could not create a URL locator for "${filename}".`);
        }
        return locator;
      } catch (error) {
        if (optional) {
          warnings.push({
            code: 'wav2vec2.optional-asset-missing',
            message: `Optional asset "${filename}" was not found for ${this.modelId}.`,
            recoverable: true,
          });
          return undefined;
        }
        throw error;
      }
    };

    const modelFilename = source.modelFilename ?? 'model.onnx';
    const modelDataFilename = source.modelDataFilename ?? 'model.onnx.data';
    const tokenizerFilename = source.tokenizerFilename ?? 'vocab.json';

    const modelUrl = await resolveFile(modelFilename);
    const modelDataUrl = await resolveFile(modelDataFilename, true);
    const tokenizerUrl = await resolveFile(tokenizerFilename);

    return {
      artifacts: {
        modelUrl: modelUrl ?? artifacts.modelUrl,
        tokenizerUrl: tokenizerUrl ?? artifacts.tokenizerUrl,
        modelDataUrl: modelDataUrl ?? artifacts.modelDataUrl,
        modelDataFilename: modelDataUrl ? modelDataFilename : artifacts.modelDataFilename,
      },
      warnings,
    };
  }

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------

  private async initialize(): Promise<LoadedExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const resolved = resolveWav2Vec2Artifacts(this.sourceOptions, this.backendId);
    let artifacts = resolved.artifacts;
    const warnings: TranscriptWarning[] = [];

    if (this.sourceOptions.kind === 'huggingface') {
      const materialized = await this.materializeHuggingFaceArtifacts(
        this.sourceOptions,
        artifacts,
      );
      artifacts = materialized.artifacts;
      warnings.push(...materialized.warnings);
    }

    const ort = await initOrt(this.backendId, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
    });

    const session = await createOrtSession(ort, artifacts.modelUrl, {
      backendId: resolved.backendForOrt,
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: artifacts.modelDataUrl,
      externalDataPath: artifacts.modelDataFilename,
    });

    const tokenizer = await Wav2Vec2CharTokenizer.fromUrl(artifacts.tokenizerUrl);

    return {
      ort,
      session,
      tokenizer,
      config: this.config,
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

  // -------------------------------------------------------------------------
  // Transcription
  // -------------------------------------------------------------------------

  async extractLogits(
    audio: AudioBufferLike,
    options: Wav2Vec2TranscriptionOptions = {},
  ): Promise<Wav2Vec2LogitsResult> {
    const state = await this.getLoadedState();
    const warnings: TranscriptWarning[] = [...state.warnings];
    const extractionStart = nowMs();

    emitTranscriptionProgress(options, {
      stage: 'start',
      progress: 0,
      elapsedMs: 0,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Starting Wav2Vec2 logits extraction for ${this.modelId}.`,
    });

    if (audio.sampleRate !== state.config.sampleRate) {
      warnings.push({
        code: 'wav2vec2.sample-rate-mismatch',
        message: `Expected ${state.config.sampleRate} Hz audio but received ${audio.sampleRate} Hz. No resampler is active on this path.`,
        recoverable: true,
      });
    }

    this.sharedMonoBuffer = ensureFloat32Buffer(audio.numberOfFrames, this.sharedMonoBuffer);
    const mono = toMonoPcm(audio, this.sharedMonoBuffer);

    const preprocessElapsedMs = nowMs() - extractionStart;
    emitTranscriptionProgress(options, {
      stage: 'preprocess',
      progress: 0.2,
      elapsedMs: roundMetric(preprocessElapsedMs),
      remainingMs: estimateRemainingMs(preprocessElapsedMs, 0.2),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Prepared raw waveform for ${this.modelId}.`,
    });

    const inputTensor = new state.ort.Tensor('float32', mono, [1, mono.length]);
    const encodeStart = nowMs();
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({
        input_values: inputTensor,
      });
    } finally {
      inputTensor.dispose?.();
    }
    const encodeMs = nowMs() - encodeStart;
    const encodeElapsedMs = nowMs() - extractionStart;

    emitTranscriptionProgress(options, {
      stage: 'encode',
      progress: 0.6,
      elapsedMs: roundMetric(encodeElapsedMs),
      remainingMs: estimateRemainingMs(encodeElapsedMs, 0.6),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Encoded logits for ${this.modelId}.`,
      metrics: {
        encodeMs: roundMetric(encodeMs),
      },
    });

    const logitsTensor = findLogitsTensor(outputs);
    const logits = normalizeLogitsData(logitsTensor);
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || (dims[0] ?? 0) !== 1) {
      throw new Error(`Unexpected Wav2Vec2 logits shape: [${dims.join(', ')}].`);
    }

    const frameCount = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
    if (frameCount <= 0 || vocabSize <= 0) {
      throw new Error(`Wav2Vec2 logits shape is invalid: [${dims.join(', ')}].`);
    }

    return {
      logits,
      frameCount,
      vocabSize,
      sampleRate: state.config.sampleRate,
      audioDurationSeconds: mono.length / state.config.sampleRate,
      blankId: state.config.ctcBlankId,
      tokenizer: state.tokenizer,
      warnings: warnings.length > 0 ? warnings : undefined,
      encodeMs: roundMetric(encodeMs),
    };
  }

  async transcribe(
    audio: AudioBufferLike,
    options: Wav2Vec2TranscriptionOptions,
  ): Promise<Wav2Vec2NativeTranscript> {
    const state = await this.getLoadedState();
    const warnings: TranscriptWarning[] = [...state.warnings];
    const transcriptionStart = nowMs();

    emitTranscriptionProgress(options, {
      stage: 'start',
      progress: 0,
      elapsedMs: 0,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Starting transcription for ${this.modelId}.`,
    });

    if (audio.sampleRate !== state.config.sampleRate) {
      warnings.push({
        code: 'wav2vec2.sample-rate-mismatch',
        message: `Expected ${state.config.sampleRate} Hz audio but received ${audio.sampleRate} Hz. No resampler is active on this path.`,
        recoverable: true,
      });
    }

    // 1. Normalize audio to mono float32 PCM
    this.sharedMonoBuffer = ensureFloat32Buffer(audio.numberOfFrames, this.sharedMonoBuffer);
    const mono = toMonoPcm(audio, this.sharedMonoBuffer);

    const preprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'preprocess',
      progress: 0.2,
      elapsedMs: roundMetric(preprocessElapsedMs),
      remainingMs: estimateRemainingMs(preprocessElapsedMs, 0.2),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Prepared raw waveform for ${this.modelId}.`,
    });

    // 2. Create ONNX tensor from waveform
    //    Wav2Vec2 takes raw waveform — no mel features!
    const inputTensor = new state.ort.Tensor('float32', mono, [1, mono.length]);

    // 3. Run ONNX session
    const encodeStart = nowMs();
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({
        input_values: inputTensor,
      });
    } finally {
      inputTensor.dispose?.();
    }
    const encodeMs = nowMs() - encodeStart;
    const encodeElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'encode',
      progress: 0.6,
      elapsedMs: roundMetric(encodeElapsedMs),
      remainingMs: estimateRemainingMs(encodeElapsedMs, 0.6),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Encoded logits for ${this.modelId}.`,
      metrics: {
        encodeMs: roundMetric(encodeMs),
      },
    });

    // 4. Extract logits
    const logitsTensor = findLogitsTensor(outputs);
    const logits = normalizeLogitsData(logitsTensor);
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || (dims[0] ?? 0) !== 1) {
      throw new Error(`Unexpected Wav2Vec2 logits shape: [${dims.join(', ')}].`);
    }

    const outFrames = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
    if (outFrames <= 0 || vocabSize <= 0) {
      throw new Error(`Wav2Vec2 logits shape is invalid: [${dims.join(', ')}].`);
    }

    // 5. CTC decode
    const decodeStart = nowMs();
    const { frameIds, selectedLogProbs } = argmaxAndSelectedLogProbs(logits, outFrames, vocabSize);
    const blankId = state.config.ctcBlankId;
    const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(frameIds, selectedLogProbs, blankId);

    // 6. Compute timing
    const audioDuration = mono.length / state.config.sampleRate;
    const secondsPerFrame = estimateSecondsPerOutputFrame({
      audioDurationSec: audioDuration,
      outFrames,
    });

    // 7. Add timestamps to token spans
    const timedSpans = addTimesToTokenSpans(state.tokenizer, tokenSpans, secondsPerFrame);

    // 8. Decode text
    const text = state.tokenizer.decode(collapsedIds);

    // 9. Build sentence timings
    const sentenceTimings = buildSentenceTimings(
      text,
      state.tokenizer,
      collapsedIds,
      timedSpans,
    );

    // 10. Build utterance timing
    const utterance = buildUtteranceTiming(frameIds, selectedLogProbs, blankId, secondsPerFrame);
    const decodeMs = nowMs() - decodeStart;

    // 11. Map timed spans to Wav2Vec2 token spans
    const wav2vec2TokenSpans: Wav2Vec2TokenSpan[] = timedSpans.map((span) => ({
      tokenId: span.tokenId,
      text: span.text,
      startFrame: span.startFrame,
      endFrame: span.endFrame,
      frameCount: span.frameCount,
      startTime: roundTimestampSeconds(span.startTime),
      endTime: roundTimestampSeconds(span.endTime),
      duration: roundMetric(span.duration, 4),
      confidence: roundMetric(span.confidence, 4),
      averageLogProb: roundMetric(span.averageLogProb, 6),
    }));

    // 12. Build native tokens
    const tokens: Wav2Vec2NativeToken[] = wav2vec2TokenSpans.map((span, index) => ({
      index,
      id: options.returnTokenIds ? span.tokenId : undefined,
      text: span.text,
      startTime: span.startTime,
      endTime: span.endTime,
      confidence: options.returnConfidence ? span.confidence : undefined,
    }));

    // 13. Build segments from sentence timings
    const segments: Wav2Vec2NativeSegment[] = sentenceTimings.map(
      (sentence: Wav2Vec2SentenceTiming, index: number) => ({
        index,
        text: sentence.text,
        startTime: roundTimestampSeconds(sentence.startTime),
        endTime: roundTimestampSeconds(sentence.endTime),
        confidence: roundMetric(sentence.confidence, 4),
      }),
    );

    // 14. Build words from spans (char-level, space separator)
    const words = buildWordsFromCharSpans(wav2vec2TokenSpans, ' ');

    // 15. Build utterance timing result
    const utteranceTiming: Wav2Vec2UtteranceTiming = {
      hasSpeech: utterance.hasSpeech,
      startFrame: utterance.startFrame,
      endFrame: utterance.endFrame,
      startTime: roundTimestampSeconds(utterance.startTime),
      endTime: roundTimestampSeconds(utterance.endTime),
      duration: roundMetric(utterance.duration, 4),
      confidence: roundMetric(utterance.confidence, 4),
    };
    void utteranceTiming; // kept for future use in extended transcript types

    // Progress: postprocess
    const postprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'postprocess',
      progress: 0.9,
      elapsedMs: roundMetric(postprocessElapsedMs),
      remainingMs: estimateRemainingMs(postprocessElapsedMs, 0.9),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Built transcript details for ${this.modelId}.`,
      metrics: {
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
      },
    });

    // 16. Build native transcript
    const totalMs = roundMetric(nowMs() - transcriptionStart);

    const nativeTranscript: Wav2Vec2NativeTranscript = {
      utteranceText: text,
      isFinal: true,
      language: state.config.languages[0],
      segments,
      words,
      tokens,
      warnings: warnings.length > 0 ? warnings : undefined,
    };

    emitTranscriptionProgress(options, {
      stage: 'complete',
      progress: clampProgress(1),
      elapsedMs: totalMs,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Finished transcription for ${this.modelId}.`,
    });

    return nativeTranscript;
  }

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------

  dispose(): void {
    for (const handle of this.assetHandles) {
      handle.dispose();
    }
  }
}
