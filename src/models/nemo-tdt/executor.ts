import type {
  AudioBufferLike,
  ModelClassification,
  TranscriptWarning
} from '../../types/index.js';
import type { NemoDecodeContext } from '../nemo-common/index.js';
import { createOrtSession, initOrt, resolveNemoTdtArtifacts, type OrtModuleLike, type OrtSessionLike, type OrtTensorLike } from './ort.js';
import { OnnxNemoPreprocessor } from './preprocessor.js';
import { ParakeetTokenizer } from './tokenizer.js';
import type {
  NemoTdtDecoderStateSnapshot,
  NemoTdtExecutor,
  NemoTdtModelConfig,
  NemoTdtModelOptions,
  NemoTdtNativeToken,
  NemoTdtNativeTranscript,
  NemoTdtTranscriptionOptions
} from './types.js';

function nowMs(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function roundMetric(value: number, digits = 3): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function roundTs(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function isFiniteNumber(value: number | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function argmax(values: ArrayLike<number>, offset = 0, length = values.length - offset): number {
  let maxIndex = offset;
  let maxValue = Number.NEGATIVE_INFINITY;
  const end = offset + length;
  for (let index = offset; index < end; index += 1) {
    const value = values[index] ?? Number.NEGATIVE_INFINITY;
    if (value > maxValue) {
      maxValue = value;
      maxIndex = index;
    }
  }

  return maxIndex;
}

function confidenceFromLogits(logits: Float32Array, tokenId: number, vocabSize: number): { confidence: number; logProb: number } {
  let maxLogit = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < vocabSize; index += 1) {
    const value = logits[index] ?? Number.NEGATIVE_INFINITY;
    if (value > maxLogit) {
      maxLogit = value;
    }
  }

  let expSum = 0;
  for (let index = 0; index < vocabSize; index += 1) {
    expSum += Math.exp((logits[index] ?? 0) - maxLogit);
  }

  const logSumExp = maxLogit + Math.log(expSum);
  const logProb = (logits[tokenId] ?? Number.NEGATIVE_INFINITY) - logSumExp;

  return {
    confidence: Math.exp(logProb),
    logProb
  };
}

function buildWordAndTokenDetails(
  tokenizer: ParakeetTokenizer,
  tokenIds: readonly number[],
  tokenTimestamps: readonly (readonly [number, number])[],
  tokenConfidences: readonly number[],
  tokenFrameIndices: readonly number[],
  tokenLogProbs: readonly number[],
  tokenTdtSteps: readonly number[]
): Pick<NemoTdtNativeTranscript, 'words' | 'tokens'> {
  const rawTokens = tokenizer.idsToTokens(tokenIds);
  const tokens: NemoTdtNativeToken[] = [];
  const words: Array<{
    index: number;
    text: string;
    startTime: number;
    endTime: number;
    confidence?: number;
  }> = [];

  let activeWord:
    | {
        index: number;
        parts: string[];
        startTime: number;
        endTime: number;
        confidences: number[];
      }
    | undefined;

  for (let index = 0; index < tokenIds.length; index += 1) {
    const rawToken = rawTokens[index] ?? '';
    const tokenText = rawToken.replace(/\u2581/g, ' ');
    const isWordStart = rawToken.startsWith('\u2581') || !activeWord;
    const [startTime, endTime] = tokenTimestamps[index] ?? [0, 0];
    const confidence = tokenConfidences[index];

    tokens.push({
      index,
      id: tokenIds[index],
      text: tokenText.trim().length > 0 ? tokenText.trim() : tokenText,
      rawText: rawToken,
      isWordStart,
      startTime,
      endTime,
      confidence,
      frameIndex: tokenFrameIndices[index],
      logProb: tokenLogProbs[index],
      tdtStep: tokenTdtSteps[index]
    });

    if (isWordStart) {
      if (activeWord) {
        words.push({
          index: activeWord.index,
          text: activeWord.parts.join('').trim(),
          startTime: activeWord.startTime,
          endTime: activeWord.endTime,
          confidence: activeWord.confidences.length > 0
            ? activeWord.confidences.reduce((sum, value) => sum + value, 0) / activeWord.confidences.length
            : undefined
        });
      }

      activeWord = {
        index: words.length,
        parts: [tokenText],
        startTime,
        endTime,
        confidences: isFiniteNumber(confidence) ? [confidence] : []
      };
      continue;
    }

    if (!activeWord) {
      activeWord = {
        index: words.length,
        parts: [tokenText],
        startTime,
        endTime,
        confidences: isFiniteNumber(confidence) ? [confidence] : []
      };
      continue;
    }

    activeWord.parts.push(tokenText);
    activeWord.endTime = endTime;
    if (isFiniteNumber(confidence)) {
      activeWord.confidences.push(confidence);
    }
  }

  if (activeWord) {
    words.push({
      index: activeWord.index,
      text: activeWord.parts.join('').trim(),
      startTime: activeWord.startTime,
      endTime: activeWord.endTime,
      confidence: activeWord.confidences.length > 0
        ? activeWord.confidences.reduce((sum, value) => sum + value, 0) / activeWord.confidences.length
        : undefined
    });
  }

  return {
    words,
    tokens
  };
}

function buildEmptyTranscript(warnings: readonly TranscriptWarning[]): NemoTdtNativeTranscript {
  return {
    utteranceText: '',
    isFinal: true,
    words: [],
    tokens: [],
    warnings
  };
}

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: ParakeetTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly decoderSession: OrtSessionLike;
  readonly preprocessor: OnnxNemoPreprocessor;
  readonly warnings: readonly TranscriptWarning[];
}

export class OrtNemoTdtExecutor implements NemoTdtExecutor {
  private readonly sourceOptions: NemoTdtModelOptions['source'];
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: NemoTdtModelConfig,
    private readonly backendId: string,
    private readonly loadOptions: NemoTdtModelOptions | undefined
  ) {
    this.sourceOptions = loadOptions?.source;
    if (this.sourceOptions) {
      this.loadStatePromise = this.initialize();
    }
  }

  private async initialize(): Promise<LoadedExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const resolved = resolveNemoTdtArtifacts(this.sourceOptions, this.backendId);
    const { artifacts } = resolved;
    if (!artifacts.preprocessorUrl) {
      throw new Error(`The NeMo TDT source for "${this.modelId}" does not provide a preprocessor model.`);
    }

    const ort = await initOrt(this.backendId, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads
    });
    const tokenizer = await ParakeetTokenizer.fromUrl(artifacts.tokenizerUrl);
    const encoderSession = await createOrtSession(ort, artifacts.encoderUrl, {
      backendId: resolved.backendForOrt,
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: artifacts.encoderDataUrl,
      externalDataPath: artifacts.encoderFilename ? `${artifacts.encoderFilename}.data` : undefined
    });
    const decoderSession = await createOrtSession(ort, artifacts.decoderUrl, {
      backendId: 'wasm',
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: artifacts.decoderDataUrl,
      externalDataPath: artifacts.decoderFilename ? `${artifacts.decoderFilename}.data` : undefined
    });
    const preprocessor = new OnnxNemoPreprocessor(ort, artifacts.preprocessorUrl, resolved.enableProfiling);

    return {
      ort,
      tokenizer,
      encoderSession,
      decoderSession,
      preprocessor,
      warnings: resolved.warnings.map((warning) => ({
        ...warning,
        recoverable: true
      }))
    };
  }

  private async getLoadedState(): Promise<LoadedExecutorState> {
    if (!this.loadStatePromise) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    return this.loadStatePromise;
  }

  private snapshotDecoderState(
    state: {
      readonly state1: OrtTensorLike<Float32Array>;
      readonly state2: OrtTensorLike<Float32Array>;
    } | null
  ): NemoTdtDecoderStateSnapshot | undefined {
    if (!state) {
      return undefined;
    }

    return {
      s1: new Float32Array(state.state1.data),
      s2: new Float32Array(state.state2.data),
      dims1: [...state.state1.dims],
      dims2: [...state.state2.dims]
    };
  }

  async transcribe(
    audio: AudioBufferLike,
    options: NemoTdtTranscriptionOptions,
    _context: NemoDecodeContext<NemoTdtModelConfig>
  ): Promise<NemoTdtNativeTranscript> {
    const loaded = await this.getLoadedState();
    const warnings = [...loaded.warnings];

    if (audio.sampleRate !== this.config.sampleRate) {
      warnings.push({
        code: 'nemo-tdt.sample-rate-mismatch',
        message: `Expected ${this.config.sampleRate} Hz audio but received ${audio.sampleRate} Hz. No resampler is wired into the restored TDT path yet.`,
        recoverable: true
      });
    }

    const preprocessStart = nowMs();
    const processed = await loaded.preprocessor.process(audio);
    const preprocessMs = nowMs() - preprocessStart;

    if (processed.features.length === 0 || processed.frameCount === 0) {
      return buildEmptyTranscript(warnings);
    }

    const totalStart = nowMs();
    const inputTensor = new loaded.ort.Tensor('float32', processed.features, [1, this.config.melBins, processed.frameCount]);
    const encoderLengthTensor = new loaded.ort.Tensor('int64', BigInt64Array.from([BigInt(processed.validLength)]), [1]);

    const encodeStart = nowMs();
    let encoderOutputs: Record<string, OrtTensorLike> | undefined;
    try {
      encoderOutputs = await loaded.encoderSession.run({
        audio_signal: inputTensor,
        length: encoderLengthTensor
      });
    } finally {
      inputTensor.dispose?.();
      encoderLengthTensor.dispose?.();
    }
    const encodeMs = nowMs() - encodeStart;

    const encoderTensor = (encoderOutputs.outputs ?? Object.values(encoderOutputs)[0]) as OrtTensorLike<Float32Array>;
    const dims = [...encoderTensor.dims];
    if (dims.length !== 3 || dims[0] !== 1) {
      throw new Error(`Unexpected NeMo TDT encoder output shape: [${dims.join(', ')}].`);
    }

    const encoderIsBdt = (dims[1] ?? 0) > (dims[2] ?? 0);
    const featureSize = encoderIsBdt ? dims[1]! : dims[2]!;
    const frameCount = encoderIsBdt ? dims[2]! : dims[1]!;
    const frameTime = this.config.frameShiftSeconds * this.config.subsamplingFactor;
    const vocabSize = this.config.vocabularySize ?? loaded.tokenizer.vocabSize;
    const blankId = loaded.tokenizer.blankId ?? this.config.tokenizer.blankTokenId ?? 0;
    const encoderFrameBuffer = new Float32Array(featureSize);
    const encoderFrameTensor = new loaded.ort.Tensor('float32', encoderFrameBuffer, [1, featureSize, 1]);
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
        [predictionLayers, 1, predictionHiddenSize]
      ),
      state2: new loaded.ort.Tensor(
        'float32',
        new Float32Array(predictionLayers * predictionHiddenSize),
        [predictionLayers, 1, predictionHiddenSize]
      )
    };

    const tokenIds: number[] = [];
    const tokenTimestamps: Array<[number, number]> = [];
    const tokenConfidences: number[] = [];
    const tokenFrameIndices: number[] = [];
    const tokenLogProbs: number[] = [];
    const tokenTdtSteps: number[] = [];
    const frameConfidenceStats = new Map<number, { sum: number; count: number }>();
    let emittedOnFrame = 0;

    const disposeTensor = (tensor: OrtTensorLike | undefined): void => {
      tensor?.dispose?.();
    };

    const disposeDecoderState = (
      state: { state1: OrtTensorLike; state2: OrtTensorLike } | null,
      keep?: { state1: OrtTensorLike; state2: OrtTensorLike } | null
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
    try {
      for (let frameIndex = 0; frameIndex < frameCount;) {
        if (encoderIsBdt) {
          for (let featureIndex = 0; featureIndex < featureSize; featureIndex += 1) {
            encoderFrameBuffer[featureIndex] = encoderTensor.data[(featureIndex * frameCount) + frameIndex] ?? 0;
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
          input_states_2: decoderState?.state2
        });

        const logits = decoderOutputs.outputs as OrtTensorLike<Float32Array>;
        const nextState = {
          state1: decoderOutputs.output_states_1 as OrtTensorLike<Float32Array>,
          state2: decoderOutputs.output_states_2 as OrtTensorLike<Float32Array>
        };
        const logitsData = logits.data;
        const tokenId = argmax(logitsData, 0, vocabSize);
        const durationOffset = vocabSize;
        const step = logitsData.length > durationOffset
          ? argmax(logitsData, durationOffset, logitsData.length - durationOffset) - durationOffset
          : 0;
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
            roundTs(frameIndex * frameTime),
            roundTs(Math.min(frameCount, frameIndex + durationFrames) * frameTime)
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
      tokenTdtSteps
    );
    const tokenizeMs = nowMs() - tokenizeStart;

    const frameConfidences = [...frameConfidenceStats.values()].map((entry) => entry.sum / entry.count);
    const utteranceConfidence = tokenConfidences.length > 0
      ? tokenConfidences.reduce((sum, value) => sum + value, 0) / tokenConfidences.length
      : undefined;
    const wordAverage = details.words && details.words.length > 0
      ? details.words.reduce((sum, word) => sum + (word.confidence ?? 0), 0) / details.words.length
      : undefined;
    const tokenAverage = tokenConfidences.length > 0
      ? tokenConfidences.reduce((sum, value) => sum + value, 0) / tokenConfidences.length
      : undefined;
    const totalMs = nowMs() - totalStart;
    const rtf = audio.durationSeconds > 0 ? totalMs / (audio.durationSeconds * 1000) : 0;

    const transcript: NemoTdtNativeTranscript = {
      utteranceText: text,
      isFinal: true,
      words: details.words,
      tokens: details.tokens,
      confidence: {
        utterance: utteranceConfidence,
        wordAverage,
        tokenAverage,
        frameAverage: frameConfidences.length > 0
          ? frameConfidences.reduce((sum, value) => sum + value, 0) / frameConfidences.length
          : undefined,
        averageLogProb: tokenLogProbs.length > 0
          ? tokenLogProbs.reduce((sum, value) => sum + value, 0) / tokenLogProbs.length
          : undefined,
        frames: frameConfidences
      },
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
        tokenizeMs: roundMetric(tokenizeMs),
        totalMs: roundMetric(totalMs),
        rtf: roundMetric(rtf, 4)
      },
      warnings,
      debug: {
        tokenIds: options.returnTokenIds ? tokenIds : undefined,
        frameIndices: options.returnFrameIndices ? tokenFrameIndices : undefined,
        logProbs: options.returnLogProbs ? tokenLogProbs : undefined,
        tdtSteps: options.returnTdtSteps ? tokenTdtSteps : undefined
      },
      decoderState: options.returnDecoderState ? this.snapshotDecoderState(decoderState) : undefined
    };

    disposeTensor(decoderState?.state1);
    disposeTensor(decoderState?.state2);

    return transcript;
  }

  dispose(): void {
    return undefined;
  }
}
