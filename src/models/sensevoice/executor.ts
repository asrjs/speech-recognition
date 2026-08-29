import { normalizePcmInput } from '../../audio/index.js';
import {
  addTimesToTokenSpans,
  argmaxAndSelectedLogProbs,
  argmaxAndSelectedLogProbsFp16,
  buildUtteranceTiming,
  ctcCollapseWithSpans,
  estimateSecondsPerOutputFrame,
} from '../../ctc/index.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import { importNodeModule } from '../../io/node-compat.js';
import { createExperimentalArtifactMissingError } from '../../runtime/experimental-families.js';
import type {
  AssetProvider,
  AudioBufferLike,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptWarning,
} from '../../types/index.js';
import { createOrtSession, disposeOrtOutputs, initOrt, releaseOrtSession, type OrtModuleLike, type OrtSessionLike, type OrtTensorLike } from '../lasr-ctc/ort.js';
import { SenseVoiceJsPreprocessor, parseSenseVoiceCmvn, type SenseVoiceCmvn } from './frontend.js';
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
  readonly warnings: readonly TranscriptWarning[];
  readonly cmvn?: SenseVoiceCmvn;
  readonly graph: SenseVoiceGraphContract;
}

type SenseVoiceGraphContract = 'official' | 'folded';

function roundMiB(bytes: number | undefined): number | undefined {
  return Number.isFinite(bytes) ? roundMetric((bytes as number) / (1024 * 1024), 2) : undefined;
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
  const percent = event.total && event.total > 0
    ? Math.min(100, Math.round((event.loaded / event.total) * 100))
    : event.done && !event.aborted ? 100 : undefined;
  return {
    phase: 'asset:download', modelId, file, loaded: event.loaded, total: event.total,
    percent, loadedMiB: roundMiB(event.loaded), totalMiB: roundMiB(event.total),
    isComplete: Boolean(event.done) && !event.aborted,
    aborted: event.aborted,
    message: event.aborted ? `Cancelled ${file}.` : event.done ? `Prepared ${file}.` : `Downloading ${file}.`,
  };
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
  readonly cmvnUrl?: string;
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
      cmvnUrl: source.artifacts.cmvnUrl,
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
    cmvnUrl: source.cmvnFilename ? hfUrl(source.repoId, revision, source.cmvnFilename) : undefined,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

function toInt64(ort: OrtModuleLike, value: number): OrtTensorLike {
  return new ort.Tensor('int64', BigInt64Array.from([BigInt(value)]), [1]);
}

function toInt32(ort: OrtModuleLike, value: number): OrtTensorLike {
  return new ort.Tensor('int32', Int32Array.from([value]), [1]);
}

function int32Batch(ort: OrtModuleLike, values: readonly number[]): OrtTensorLike {
  return new ort.Tensor('int32', Int32Array.from(values), [values.length]);
}

function parseOrtTensorElementType(ortType: string | undefined, fallback: string): string {
  if (!ortType) return fallback;
  const match = /^tensor\((.+)\)$/.exec(ortType.trim());
  const elementType = (match?.[1] ?? ortType).trim();
  return elementType === 'float' ? 'float32' : elementType;
}

function sessionInputNames(session: OrtSessionLike): readonly string[] {
  const metadata = session.inputMetadata as
    | Record<string, { readonly type?: string; readonly name?: string }>
    | Array<{ readonly name?: string; readonly type?: string }>
    | undefined;
  if (!metadata) return [];
  if (Array.isArray(metadata)) return metadata.map((entry) => entry.name).filter((name): name is string => Boolean(name));
  return Object.keys(metadata);
}

function getInputElementType(session: OrtSessionLike, inputName: string, fallback: string): string {
  const metadata = session.inputMetadata as
    | Record<string, { readonly type?: string; readonly name?: string }>
    | Array<{ readonly name?: string; readonly type?: string }>
    | undefined;
  if (!metadata) return fallback;
  if (Array.isArray(metadata)) {
    const found = metadata.find((entry) => entry.name === inputName);
    return parseOrtTensorElementType(found?.type, fallback);
  }
  return parseOrtTensorElementType(metadata[inputName]?.type, fallback);
}

function detectGraphContract(session: OrtSessionLike): SenseVoiceGraphContract {
  const names = sessionInputNames(session);
  return names.includes('speech') ? 'official' : 'folded';
}

function intScalar(ort: OrtModuleLike, session: OrtSessionLike, name: string, value: number, fallback: 'int32' | 'int64'): OrtTensorLike {
  const type = getInputElementType(session, name, fallback);
  return type === 'int32' ? toInt32(ort, value) : toInt64(ort, value);
}

function intVector(ort: OrtModuleLike, session: OrtSessionLike, name: string, values: readonly number[], fallback: 'int32' | 'int64'): OrtTensorLike {
  const type = getInputElementType(session, name, fallback);
  return type === 'int32' ? int32Batch(ort, values) : int64Batch(ort, values);
}

async function readTextUrl(url: string): Promise<string> {
  if (/^file:/i.test(url)) {
    const { readFile } = await importNodeModule<typeof import('node:fs/promises')>('node:fs/promises');
    const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
    return readFile(fileURLToPath(url), 'utf8');
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch SenseVoice CMVN at "${url}".`);
  return response.text();
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
      ? new Float32Array(tensor.data)
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

/**
 * Greedy frame decode for one logits row-block. float16 tensors keep their
 * raw half-bit representation and go through the lookup-table fast path in
 * the shared CTC decoder; float32 tensors keep the generic pipeline.
 */
function decodeLogitsBlock(
  tensor: OrtTensorLike,
  start: number,
  frameCount: number,
  vocabSize: number,
): { frameIds: number[]; selectedLogProbs: Float32Array } {
  if (tensor.type === 'float16') {
    const source = tensor.data as Uint16Array;
    return argmaxAndSelectedLogProbsFp16(source.subarray(start), frameCount, vocabSize);
  }
  const logits = readLogits(tensor).subarray(start, start + frameCount * vocabSize);
  return argmaxAndSelectedLogProbs(logits, frameCount, vocabSize);
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
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private disposed = false;
  private disposePromise?: Promise<void>;

  constructor(
    private readonly modelId: string,
    private readonly backendId: string,
    options: SenseVoiceModelOptions | undefined,
    dependencies: { readonly assetProvider?: AssetProvider; readonly runtimeHooks?: SpeechRuntimeHooks; readonly signal?: import('../../types/index.js').AbortSignalLike | null } = {},
  ) {
    this.source = options?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;
    this.signal = dependencies.signal;
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

  private async materializeHuggingFaceArtifacts(
    source: Extract<SenseVoiceArtifactSource, { kind: 'huggingface' }>,
    artifacts: ReturnType<typeof resolveSource>,
  ): Promise<{ readonly artifacts: ReturnType<typeof resolveSource>; readonly warnings: readonly TranscriptWarning[] }> {
    if (!this.assetProvider) return { artifacts, warnings: [] };
    const warnings: TranscriptWarning[] = [];
    const revision = source.revision ?? 'main';
    const handles = this.assetHandles;
    const resolveFile = async (filename: string, optional = false): Promise<string | undefined> => {
      try {
        const handle = await this.assetProvider!.resolve({
          id: `huggingface:${source.repoId}:${revision}:${filename}`,
          provider: 'huggingface', repoId: source.repoId, revision, filename,
          cacheKey: `huggingface:${source.repoId}:${revision}:${filename}`,
          onProgress: (event) => this.runtimeHooks?.onProgress?.(
            createAssetProgressEvent(this.modelId, filename, event),
          ),
        });
        handles.push(handle);
        const locator = await handle.getLocator('url');
        if (!locator) throw new Error(`Could not create a URL locator for "${filename}".`);
        return locator;
      } catch (error) {
        if (!optional) throw error;
        warnings.push({
          code: 'sensevoice.optional-asset-missing',
          message: `Optional asset "${filename}" was not found for ${this.modelId}.`,
          recoverable: true,
        });
        return undefined;
      }
    };
    const modelFilename = source.modelFilename ?? 'model.onnx';
    const tokenizerFilename = source.tokenizerFilename ?? 'vocab.txt';
    const modelDataFilename = source.modelDataFilename ?? artifacts.modelDataFilename;
    const tokenizerUrl = await resolveFile(tokenizerFilename);
    const modelUrl = await resolveFile(modelFilename);
    const modelDataUrl = modelDataFilename ? await resolveFile(modelDataFilename, true) : undefined;
    const cmvnUrl = source.cmvnFilename ? await resolveFile(source.cmvnFilename, true) : artifacts.cmvnUrl;
    return {
      artifacts: { ...artifacts, modelUrl: modelUrl ?? artifacts.modelUrl, tokenizerUrl: tokenizerUrl ?? artifacts.tokenizerUrl, modelDataUrl, modelDataFilename, cmvnUrl },
      warnings,
    };
  }

  private async initialize(): Promise<LoadedState> {
    if (!this.source) throw createExperimentalArtifactMissingError('sensevoice', this.modelId);
    const resolved = resolveSource(this.source);
    let artifacts = resolved;
    const warnings: TranscriptWarning[] = [];
    if (this.source.kind === 'huggingface') {
      const materialized = await this.materializeHuggingFaceArtifacts(this.source, resolved);
      artifacts = materialized.artifacts;
      warnings.push(...materialized.warnings);
    }
    const ort = await initOrt(this.backendId, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
      signal: this.signal,
    });
    const session = await createOrtSession(ort, artifacts.modelUrl, {
      backendId: this.backendId.startsWith('webgpu') ? 'webgpu' : 'wasm',
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: artifacts.modelDataUrl,
      externalDataPath: artifacts.modelDataFilename,
      signal: this.signal,
    });
    if (this.disposed) {
      releaseOrtSession(session);
      throw new Error(`SenseVoice executor was disposed during load for "${this.modelId}".`);
    }
    const tokenizer = await SenseVoiceTokenizer.fromUrl(artifacts.tokenizerUrl, this.signal);
    const graph = detectGraphContract(session);
    let cmvn: SenseVoiceCmvn | undefined;
    if (artifacts.cmvnUrl) {
      cmvn = parseSenseVoiceCmvn(await readTextUrl(artifacts.cmvnUrl));
    } else if (graph === 'official') {
      throw new Error(
        'Official SenseVoice ONNX expects LFR+CMVN features. Provide artifacts.cmvnUrl (am.mvn).',
      );
    }
    return { ort, session, tokenizer, warnings, cmvn, graph };
  }

  async ready(): Promise<void> {
    if (!this.loadStatePromise) throw createExperimentalArtifactMissingError('sensevoice', this.modelId);
    await this.loadStatePromise;
  }

  async transcribe(audioInput: AudioBufferLike, options: SenseVoiceTranscriptionOptions = {}): Promise<SenseVoiceNativeTranscript> {
    if (this.disposed) throw new Error(`SenseVoice executor is disposed for "${this.modelId}".`);
    const state = await this.loadStatePromise;
    if (!state) throw createExperimentalArtifactMissingError('sensevoice', this.modelId);
    const audio = normalizePcmInput(audioInput).toMono();
    const started = nowMs();
    const prompt = createSenseVoicePrompt({ language: options.language, useItn: options.useItn });
    const featuresStart = nowMs();
    const featureBatch = state.graph === 'official'
      ? this.preprocessor.processOfficial(audio.channels[0] ?? new Float32Array(0), state.cmvn!)
      : this.preprocessor.process(audio.channels[0] ?? new Float32Array(0));
    const preprocessMs = nowMs() - featuresStart;
    if (featureBatch.frameCount <= 0) {
      return { utteranceText: '', isFinal: true, language: prompt.language, warnings: [...state.warnings] };
    }
    const featureWidth = state.graph === 'official' ? featureBatch.featureSize : 80;
    const speechName = state.graph === 'official' ? 'speech' : 'features';
    const lengthName = state.graph === 'official' ? 'speech_lengths' : 'features_lens';
    const features = tensorFloat32(state.ort, featureBatch.features, [1, featureBatch.frameCount, featureWidth]);
    const lengths = intScalar(state.ort, state.session, lengthName, featureBatch.validFrameCount, state.graph === 'official' ? 'int32' : 'int64');
    const language = intScalar(state.ort, state.session, 'language', prompt.languageId, state.graph === 'official' ? 'int32' : 'int64');
    const textnorm = intScalar(state.ort, state.session, 'textnorm', prompt.textnormId, state.graph === 'official' ? 'int32' : 'int64');
    const encodeStart = nowMs();
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({ [speechName]: features, [lengthName]: lengths, language, textnorm });
    } finally {
      features.dispose?.(); lengths.dispose?.(); language.dispose?.(); textnorm.dispose?.();
    }
    const encodeMs = nowMs() - encodeStart;
    try {
    const logitsTensor = findOutput(outputs, ['ctc_logits', 'logprobs', 'logits']);
    if (!logitsTensor) throw new Error('SenseVoice graph returned no logprobs output.');
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || dims[0] !== 1) throw new Error(`Unexpected SenseVoice logits shape: [${dims.join(', ')}].`);
    const outFrames = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
    const frameLength = Math.min(outFrames, tensorLength(findOutput(outputs, ['encoder_out_lens', 'logprobs_lens', 'output_lens']), outFrames));
    const decodeStart = nowMs();
    const { frameIds, selectedLogProbs } = decodeLogitsBlock(logitsTensor, 0, frameLength, vocabSize);
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
    } finally {
      disposeOrtOutputs(outputs);
    }
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
    if (this.disposed) throw new Error(`SenseVoice executor is disposed for "${this.modelId}".`);
    if (audioInputs.length === 0) return [];
    const state = await this.loadStatePromise;
    if (!state) throw createExperimentalArtifactMissingError('sensevoice', this.modelId);
    const started = nowMs();
    const preprocessStarted = nowMs();
    const audios = audioInputs.map((input) => normalizePcmInput(input).toMono());
    const prepared = audios.map((audio) => (
      state.graph === 'official'
        ? this.preprocessor.processOfficial(audio.channels[0] ?? new Float32Array(0), state.cmvn!)
        : this.preprocessor.process(audio.channels[0] ?? new Float32Array(0))
    ));
    const featureWidth = state.graph === 'official' ? (prepared[0]?.featureSize ?? 560) : 80;
    const maxFrames = Math.max(...prepared.map((item) => item.frameCount));
    const batchFeatures = new Float32Array(audioInputs.length * maxFrames * featureWidth);
    const lengths = prepared.map((item) => item.frameCount);
    prepared.forEach((item, batchIndex) => {
      const targetOffset = batchIndex * maxFrames * featureWidth;
      for (let frame = 0; frame < maxFrames; frame += 1) {
        const padWithLast = state.graph === 'folded';
        const sourceFrame = padWithLast
          ? Math.min(frame, Math.max(0, item.frameCount - 1))
          : frame;
        if (sourceFrame >= item.frameCount) continue;
        const sourceOffset = sourceFrame * featureWidth;
        const destinationOffset = targetOffset + frame * featureWidth;
        for (let dim = 0; dim < featureWidth; dim += 1) {
          batchFeatures[destinationOffset + dim] = item.features[sourceOffset + dim] ?? 0;
        }
      }
    });
    const preprocessMs = nowMs() - preprocessStarted;
    const prompt = createSenseVoicePrompt({ language: options.language, useItn: options.useItn });
    const speechName = state.graph === 'official' ? 'speech' : 'features';
    const lengthName = state.graph === 'official' ? 'speech_lengths' : 'features_lens';
    const features = tensorFloat32(state.ort, batchFeatures, [audioInputs.length, maxFrames, featureWidth]);
    const featureLengths = intVector(state.ort, state.session, lengthName, lengths, state.graph === 'official' ? 'int32' : 'int64');
    const languages = intVector(state.ort, state.session, 'language', audioInputs.map(() => prompt.languageId), state.graph === 'official' ? 'int32' : 'int64');
    const textnorms = intVector(state.ort, state.session, 'textnorm', audioInputs.map(() => prompt.textnormId), state.graph === 'official' ? 'int32' : 'int64');
    let outputs: Record<string, OrtTensorLike>;
    const encodeStarted = nowMs();
    try {
      outputs = await state.session.run({ [speechName]: features, [lengthName]: featureLengths, language: languages, textnorm: textnorms });
    } finally {
      features.dispose?.(); featureLengths.dispose?.(); languages.dispose?.(); textnorms.dispose?.();
    }
    const encodeMs = nowMs() - encodeStarted;
    try {
    const logitsTensor = findOutput(outputs, ['ctc_logits', 'logprobs', 'logits']);
    if (!logitsTensor) throw new Error('SenseVoice graph returned no logprobs output.');
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || dims[0] !== audioInputs.length) throw new Error(`Unexpected SenseVoice batch logits shape: [${dims.join(', ')}].`);
    const batchSize = dims[0] ?? 0;
    const outFrames = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
    const lengthsTensor = findOutput(outputs, ['encoder_out_lens', 'logprobs_lens', 'output_lens']);
    const outputLengths = tensorLengths(lengthsTensor, batchSize, outFrames);
    const fp16Bits = logitsTensor.type === 'float16' ? (logitsTensor.data as Uint16Array) : null;
    const fp32Logits = fp16Bits === null ? readLogits(logitsTensor) : null;
    return Array.from({ length: batchSize }, (_, batchIndex) => {
      const decodeStarted = nowMs();
      const frameCount = Math.min(outFrames, outputLengths[batchIndex] ?? outFrames);
      const sourceOffset = batchIndex * outFrames * vocabSize;
      const { frameIds, selectedLogProbs } = fp16Bits !== null
        ? argmaxAndSelectedLogProbsFp16(fp16Bits.subarray(sourceOffset), frameCount, vocabSize)
        : argmaxAndSelectedLogProbs(
            fp32Logits!.subarray(sourceOffset, sourceOffset + frameCount * vocabSize),
            frameCount,
            vocabSize,
          );
      const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(frameIds, selectedLogProbs, 0);
      const tokenizer = state.tokenizer;
      const text = tokenizer.decode(collapsedIds);
      const audio = audios[batchIndex]!;
      const secondsPerFrame = estimateSecondsPerOutputFrame({ audioDurationSec: audio.durationSeconds, inputFrames: lengths[batchIndex]!, inputFrameHopSeconds: this.config.featureHopSeconds, outFrames: frameCount });
      const timedSpans = addTimesToTokenSpans(tokenizer, tokenSpans, secondsPerFrame);
      const timing = buildUtteranceTiming(frameIds, selectedLogProbs, 0, secondsPerFrame);
      const metadata = extractMetadata(tokenizer, collapsedIds);
      const tokens = timedSpans
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
          decodeMs: roundMetric(nowMs() - decodeStarted),
          totalMs: roundMetric(totalMs),
          wallMs: roundMetric(totalMs),
          audioDurationSec: roundMetric(audio.durationSeconds, 4),
          rtf: audio.durationSeconds > 0 ? roundMetric(totalMs / (audio.durationSeconds * 1000), 4) : 0,
          rtfx: audio.durationSeconds > 0 ? roundMetric(audio.durationSeconds / (totalMs / 1000), 4) : undefined,
          preprocessorBackend: 'js',
          encoderFrameCount: frameCount,
          decodeIterations: frameCount,
          emittedTokenCount: tokens.length,
        },
        warnings: [...state.warnings],
      };
    });
    } finally {
      disposeOrtOutputs(outputs);
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
        releaseOrtSession(loaded.session);
      } catch {
        // Keep the original load error; still drop asset handles.
      }
    }
    const handles = this.assetHandles.splice(0);
    await Promise.all(handles.map((handle) => Promise.resolve(handle.dispose())));
  }
}
