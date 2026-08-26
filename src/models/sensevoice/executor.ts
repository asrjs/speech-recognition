import { normalizePcmInput } from '../../audio/index.js';
import {
  addTimesToTokenSpans,
  argmaxAndSelectedLogProbs,
  buildUtteranceTiming,
  ctcCollapseWithSpans,
  estimateSecondsPerOutputFrame,
} from '../../ctc/index.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import type { AudioBufferLike } from '../../types/index.js';
import { createOrtSession, initOrt, type OrtModuleLike, type OrtSessionLike, type OrtTensorLike } from '../lasr-ctc/ort.js';
import { SenseVoiceJsPreprocessor } from './frontend.js';
import { createSenseVoicePrompt } from './prompt.js';
import { SenseVoiceTokenizer } from './tokenizer.js';
import type {
  SenseVoiceArtifactSource,
  SenseVoiceExecutor,
  SenseVoiceModelConfig,
  SenseVoiceModelOptions,
  SenseVoiceNativeMetadata,
  SenseVoiceNativeToken,
  SenseVoiceNativeTranscript,
  SenseVoiceTranscriptionOptions,
} from './types.js';

interface LoadedState {
  readonly ort: OrtModuleLike;
  readonly session: OrtSessionLike;
  readonly tokenizer: SenseVoiceTokenizer;
  readonly warnings: readonly { readonly code: string; readonly message: string }[];
}

function hfUrl(repoId: string, revision: string, filename: string): string {
  const repo = repoId.split('/').map(encodeURIComponent).join('/');
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${filename
    .split('/')
    .map(encodeURIComponent)
    .join('/')}`;
}

function resolveSource(source: SenseVoiceArtifactSource): {
  readonly modelUrl: string;
  readonly tokenizerUrl: string;
  readonly modelDataUrl?: string;
  readonly modelDataFilename?: string;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
} {
  if (source.kind === 'direct') {
    return {
      modelUrl: source.artifacts.modelUrl,
      tokenizerUrl: source.artifacts.tokenizerUrl,
      modelDataUrl: source.artifacts.modelDataUrl,
      modelDataFilename: source.artifacts.modelDataFilename ?? source.artifacts.modelDataUrl?.split('/').pop(),
      wasmPaths: source.wasmPaths,
      cpuThreads: source.cpuThreads,
      enableProfiling: source.enableProfiling,
    };
  }

  const revision = source.revision ?? 'main';
  const modelFilename = source.modelFilename ?? 'model.onnx';
  const modelDataFilename = source.modelDataFilename ?? 'model.onnx_data';
  return {
    modelUrl: hfUrl(source.repoId, revision, modelFilename),
    tokenizerUrl: hfUrl(source.repoId, revision, source.tokenizerFilename ?? 'vocab.txt'),
    modelDataUrl: hfUrl(source.repoId, revision, modelDataFilename),
    modelDataFilename,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

function toInt64(ort: OrtModuleLike, value: number): OrtTensorLike {
  return new ort.Tensor('int64', BigInt64Array.from([BigInt(value)]), [1]);
}

function tensorFloat32(ort: OrtModuleLike, data: Float32Array, dims: readonly number[]): OrtTensorLike {
  return new ort.Tensor('float32', data, dims);
}

function int64Batch(ort: OrtModuleLike, values: readonly number[]): OrtTensorLike {
  return new ort.Tensor('int64', BigInt64Array.from(values, (value) => BigInt(value)), [values.length]);
}

function tensorLength(ortTensor: OrtTensorLike | undefined, fallback: number): number {
  const value = (ortTensor?.data as unknown as ArrayLike<number | bigint> | undefined)?.[0];
  if (typeof value === 'bigint') return Number(value);
  if (typeof value === 'number' && Number.isFinite(value)) return Math.max(0, Math.floor(value));
  return fallback;
}

function tensorLengths(ortTensor: OrtTensorLike | undefined, count: number, fallback: number): number[] {
  const data = ortTensor?.data as unknown as ArrayLike<number | bigint> | undefined;
  return Array.from({ length: count }, (_, index) => {
    const value = data?.[index];
    return typeof value === 'bigint' ? Number(value) : typeof value === 'number' ? Math.floor(value) : fallback;
  });
}

function readLogits(tensor: OrtTensorLike): Float32Array {
  if (tensor.type !== 'float16') {
    return tensor.data instanceof Float32Array
      ? tensor.data
      : Float32Array.from(tensor.data as unknown as ArrayLike<number>);
  }
  const source = tensor.data as Uint16Array;
  const result = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    const bits = source[index] ?? 0;
    const sign = (bits & 0x8000) << 16;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) {
      if (mantissa === 0) {
        result[index] = sign ? -0 : 0;
      } else {
        let normalized = mantissa;
        let exponentValue = -14;
        while ((normalized & 0x400) === 0) {
          normalized <<= 1;
          exponentValue -= 1;
        }
        normalized &= 0x3ff;
        result[index] = (sign ? -1 : 1) * (1 + normalized / 1024) * 2 ** exponentValue;
      }
    }
    else if (exponent === 0x1f) result[index] = mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
    else result[index] = (sign ? -1 : 1) * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return result;
}

function findOutput(outputs: Record<string, OrtTensorLike>, names: readonly string[]): OrtTensorLike | undefined {
  for (const name of names) if (outputs[name]) return outputs[name];
  return Object.values(outputs)[0];
}

function readTokenText(tokenizer: SenseVoiceTokenizer, id: number): string {
  return tokenizer.idToToken[id] ?? '';
}

function extractMetadata(tokenizer: SenseVoiceTokenizer, ids: readonly number[]): SenseVoiceNativeMetadata {
  let language: string | undefined;
  let emotion: string | undefined;
  let event: string | undefined;
  for (const id of ids) {
    const token = readTokenText(tokenizer, id);
    const languageMatch = /^<\|(auto|zh|en|yue|ja|ko)\|>$/.exec(token);
    if (languageMatch) language = languageMatch[1];
    else if (/^<\|[^|]+\|>$/.test(token) && token !== '<|woitn|>' && token !== '<|withitn|>') {
      const value = token.slice(2, -2);
      if (value === 'Speech' || value === 'BGM' || value === 'Applause') event = value;
      else if (value.length > 0) emotion = value;
    }
  }
  return { language, emotion, event };
}

export class OrtSenseVoiceExecutor implements SenseVoiceExecutor {
  private readonly source?: SenseVoiceArtifactSource;
  private readonly loadStatePromise?: Promise<LoadedState>;
  private readonly preprocessor = new SenseVoiceJsPreprocessor();
  private readonly config: SenseVoiceModelConfig;

  constructor(
    private readonly modelId: string,
    private readonly backendId: string,
    options: SenseVoiceModelOptions | undefined,
  ) {
    this.source = options?.source;
    this.config = {
      ecosystem: 'funasr',
      architecture: 'sensevoice',
      processorArchitecture: 'kaldi-fbank',
      encoderArchitecture: 'sensevoice-conformer',
      decoderArchitecture: 'ctc',
      sampleRate: 16000,
      featureHopSeconds: 0.01,
      nMels: 80,
      vocabularySize: 25055,
      ctcBlankId: 0,
      languages: ['auto', 'zh', 'en', 'yue', 'ja', 'ko'],
      ...(options?.config ?? {}),
    };
    if (this.source) this.loadStatePromise = this.initialize();
  }

  private async initialize(): Promise<LoadedState> {
    if (!this.source) throw new Error(`No SenseVoice artifact source is configured for "${this.modelId}".`);
    const resolved = resolveSource(this.source);
    const ort = await initOrt(this.backendId, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
    });
    const session = await createOrtSession(ort, resolved.modelUrl, {
      backendId: this.backendId.startsWith('webgpu') ? 'webgpu' : 'wasm',
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: resolved.modelDataUrl,
      externalDataPath: resolved.modelDataFilename,
    });
    const tokenizer = await SenseVoiceTokenizer.fromUrl(resolved.tokenizerUrl);
    return { ort, session, tokenizer, warnings: [] };
  }

  async ready(): Promise<void> {
    if (!this.loadStatePromise) throw new Error(`No SenseVoice artifact source is configured for "${this.modelId}".`);
    await this.loadStatePromise;
  }

  async transcribe(audioInput: AudioBufferLike, options: SenseVoiceTranscriptionOptions = {}): Promise<SenseVoiceNativeTranscript> {
    const state = await this.loadStatePromise;
    if (!state) throw new Error(`No SenseVoice artifact source is configured for "${this.modelId}".`);
    const audio = normalizePcmInput(audioInput).toMono();
    const started = nowMs();
    const prompt = createSenseVoicePrompt({ language: options.language, useItn: options.useItn });
    const featuresStart = nowMs();
    const featureBatch = this.preprocessor.process(audio.channels[0] ?? new Float32Array(0));
    const preprocessMs = nowMs() - featuresStart;
    if (featureBatch.frameCount <= 0) {
      return { utteranceText: '', isFinal: true, language: prompt.language, warnings: [...state.warnings] };
    }
    const features = tensorFloat32(state.ort, featureBatch.features, [1, featureBatch.frameCount, 80]);
    const lengths = toInt64(state.ort, featureBatch.validFrameCount);
    const language = toInt64(state.ort, prompt.languageId);
    const textnorm = toInt64(state.ort, prompt.textnormId);
    const encodeStart = nowMs();
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({ features, features_lens: lengths, language, textnorm });
    } finally {
      features.dispose?.(); lengths.dispose?.(); language.dispose?.(); textnorm.dispose?.();
    }
    const encodeMs = nowMs() - encodeStart;
    const logitsTensor = findOutput(outputs, ['logprobs', 'logits']);
    if (!logitsTensor) throw new Error('SenseVoice graph returned no logprobs output.');
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || dims[0] !== 1) throw new Error(`Unexpected SenseVoice logits shape: [${dims.join(', ')}].`);
    const outFrames = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
    const frameLength = Math.min(outFrames, tensorLength(findOutput(outputs, ['logprobs_lens', 'output_lens']), outFrames));
    const logits = readLogits(logitsTensor).subarray(0, frameLength * vocabSize);
    const decodeStart = nowMs();
    const { frameIds, selectedLogProbs } = argmaxAndSelectedLogProbs(logits, frameLength, vocabSize);
    const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(frameIds, selectedLogProbs, 0);
    const text = state.tokenizer.decode(collapsedIds);
    const secondsPerFrame = estimateSecondsPerOutputFrame({
      audioDurationSec: audio.durationSeconds,
      inputFrames: featureBatch.frameCount,
      inputFrameHopSeconds: this.config.featureHopSeconds,
      outFrames: frameLength,
    });
    const timedSpans = addTimesToTokenSpans(state.tokenizer, tokenSpans, secondsPerFrame);
    const timing = buildUtteranceTiming(frameIds, selectedLogProbs, 0, secondsPerFrame);
    const tokens: SenseVoiceNativeToken[] = timedSpans
      .filter((span) => span.text.length > 0)
      .map((span, index) => ({
        index,
        id: options.returnTokenIds ? span.tokenId : undefined,
        text: span.text,
        startTime: roundTimestampSeconds(span.startTime),
        endTime: roundTimestampSeconds(span.endTime),
        confidence: roundMetric(span.confidence, 4),
      }));
    const totalMs = nowMs() - started;
    const metadata = extractMetadata(state.tokenizer, collapsedIds);
    return {
      utteranceText: text,
      isFinal: true,
      language: metadata.language ?? prompt.language,
      metadata,
      tokens,
      confidence: { utterance: timing.confidence, tokenAverage: timing.confidence },
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(nowMs() - decodeStart),
        totalMs: roundMetric(totalMs),
        wallMs: roundMetric(totalMs),
        audioDurationSec: roundMetric(audio.durationSeconds, 4),
        rtf: audio.durationSeconds > 0 ? roundMetric(totalMs / (audio.durationSeconds * 1000), 4) : 0,
        rtfx: audio.durationSeconds > 0 ? roundMetric(audio.durationSeconds / (totalMs / 1000), 4) : undefined,
        preprocessorBackend: 'js',
        encoderFrameCount: frameLength,
        decodeIterations: frameLength,
        emittedTokenCount: tokens.length,
      },
      warnings: [...state.warnings],
    };
  }

  /**
   * Runs a true padded batch through one ONNX graph invocation. Each item is
   * padded with its final valid fbank frame, matching the exported FunASR
   * graph's LFR padding rule; `logprobs_lens` trims the result per item.
   */
  async transcribeBatch(
    audioInputs: readonly AudioBufferLike[],
    options: SenseVoiceTranscriptionOptions = {},
  ): Promise<readonly SenseVoiceNativeTranscript[]> {
    const state = await this.loadStatePromise;
    if (!state) throw new Error(`No SenseVoice artifact source is configured for "${this.modelId}".`);
    if (audioInputs.length === 0) return [];
    const audios = audioInputs.map((input) => normalizePcmInput(input).toMono());
    const prepared = audios.map((audio) => this.preprocessor.process(audio.channels[0] ?? new Float32Array(0)));
    const maxFrames = Math.max(...prepared.map((item) => item.frameCount));
    const batchFeatures = new Float32Array(audioInputs.length * maxFrames * 80);
    const lengths = prepared.map((item) => item.frameCount);
    prepared.forEach((item, batchIndex) => {
      const targetOffset = batchIndex * maxFrames * 80;
      for (let frame = 0; frame < maxFrames; frame += 1) {
        const sourceFrame = Math.min(frame, Math.max(0, item.frameCount - 1));
        const sourceOffset = sourceFrame * 80;
        const destinationOffset = targetOffset + frame * 80;
        for (let mel = 0; mel < 80; mel += 1) {
          batchFeatures[destinationOffset + mel] = item.features[sourceOffset + mel] ?? 0;
        }
      }
    });
    const prompt = createSenseVoicePrompt({ language: options.language, useItn: options.useItn });
    const features = tensorFloat32(state.ort, batchFeatures, [audioInputs.length, maxFrames, 80]);
    const featureLengths = int64Batch(state.ort, lengths);
    const languages = int64Batch(state.ort, audioInputs.map(() => prompt.languageId));
    const textnorms = int64Batch(state.ort, audioInputs.map(() => prompt.textnormId));
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({ features, features_lens: featureLengths, language: languages, textnorm: textnorms });
    } finally {
      features.dispose?.(); featureLengths.dispose?.(); languages.dispose?.(); textnorms.dispose?.();
    }
    const logitsTensor = findOutput(outputs, ['logprobs', 'logits']);
    if (!logitsTensor) throw new Error('SenseVoice graph returned no logprobs output.');
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || dims[0] !== audioInputs.length) throw new Error(`Unexpected SenseVoice batch logits shape: [${dims.join(', ')}].`);
    const batchSize = dims[0] ?? 0;
    const outFrames = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
    const lengthsTensor = findOutput(outputs, ['logprobs_lens', 'output_lens']);
    const outputLengths = tensorLengths(lengthsTensor, batchSize, outFrames);
    const allLogits = readLogits(logitsTensor);
    return Array.from({ length: batchSize }, (_, batchIndex) => {
      const frameCount = Math.min(outFrames, outputLengths[batchIndex] ?? outFrames);
      const logits = new Float32Array(frameCount * vocabSize);
      const sourceOffset = batchIndex * outFrames * vocabSize;
      logits.set(allLogits.subarray(sourceOffset, sourceOffset + logits.length));
      const { frameIds, selectedLogProbs } = argmaxAndSelectedLogProbs(logits, frameCount, vocabSize);
      const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(frameIds, selectedLogProbs, 0);
      const tokenizer = state.tokenizer;
      const text = tokenizer.decode(collapsedIds);
      const audio = audios[batchIndex]!;
      const secondsPerFrame = estimateSecondsPerOutputFrame({ audioDurationSec: audio.durationSeconds, inputFrames: lengths[batchIndex]!, inputFrameHopSeconds: this.config.featureHopSeconds, outFrames: frameCount });
      const timedSpans = addTimesToTokenSpans(tokenizer, tokenSpans, secondsPerFrame);
      const timing = buildUtteranceTiming(frameIds, selectedLogProbs, 0, secondsPerFrame);
      const metadata = extractMetadata(tokenizer, collapsedIds);
      return {
        utteranceText: text,
        isFinal: true,
        language: metadata.language ?? prompt.language,
        metadata,
        tokens: timedSpans.filter((span) => span.text.length > 0).map((span, index) => ({ index, id: options.returnTokenIds ? span.tokenId : undefined, text: span.text, startTime: roundTimestampSeconds(span.startTime), endTime: roundTimestampSeconds(span.endTime), confidence: roundMetric(span.confidence, 4) })),
        confidence: { utterance: timing.confidence, tokenAverage: timing.confidence },
        warnings: [...state.warnings],
      };
    });
  }

  dispose(): void {
    // ORT sessions are released when their owning model/session is disposed.
    // Tensor feeds and outputs are short-lived and disposed at the graph boundary.
  }
}
