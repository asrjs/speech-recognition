import type {
  AudioBufferLike,
  AssetProvider,
  ModelClassification,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptWarning,
} from '../../types/index.js';
import { argmax, confidenceFromLogits } from '../../inference/index.js';
import { fetchModelFiles } from '../../runtime/huggingface.js';
import { roundMetric } from '../../runtime/timing.js';
import { WhisperMelProcessor } from '../../audio/whisper-mel.js';
import { planWhisperChunks } from '../../pipeline/whisper-chunking.js';
import {
  createInitialWhisperBeam,
  rankWhisperBeamCandidates,
  selectBestWhisperBeam,
  type WhisperBeamState,
} from './beam-search.js';
import { mergeWhisperChunkTranscripts } from './chunking.js';
import {
  createWhisperOrtSession,
  initWhisperOrt,
  resolveWhisperArtifacts,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from './ort.js';
import { WhisperTokenizer } from './tokenizer.js';
import { WhisperTimestampLogitProcessor } from './processors.js';
import { buildWhisperWordTimestampsFromTokenDetails } from './word-timestamps.js';
import { computeWhisperDtwTokenTimestamps } from './attention-alignment.js';
import {
  parseWhisperGenerationConfig,
  parseWhisperModelConfig,
  type WhisperGenerationConfig,
  type WhisperModelConfig,
} from './generation-config.js';
import type {
  WhisperArtifactSource,
  WhisperNativeSegment,
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperNativeWord,
  WhisperSeq2SeqModelConfig,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: WhisperTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly decoderSession: OrtSessionLike;
  readonly generationConfig: WhisperGenerationConfig;
  readonly modelConfig: WhisperModelConfig;
  readonly warnings: readonly TranscriptWarning[];
  readonly isSplitGraph: boolean;
  readonly decoderInitSession?: OrtSessionLike;
  readonly decoderStepSession?: OrtSessionLike;
  readonly decoderAlignSession?: OrtSessionLike;
}

interface DecoderStepResult {
  readonly lastLogits: Float32Array;
  readonly vocabSize: number;
  readonly pastKeyValues: Record<string, OrtTensorLike<Float32Array>>;
  readonly crossAttentions: readonly OrtTensorLike<Float32Array>[];
}

function extractCrossAttentions(
  outputs: Record<string, unknown>,
): OrtTensorLike<Float32Array>[] {
  const entries: { layer: number; tensor: OrtTensorLike<Float32Array> }[] = [];
  for (const [key, value] of Object.entries(outputs)) {
    const match = key.match(/^cross_attentions\.(\d+)$/);
    if (match) {
      entries.push({
        layer: parseInt(match[1]!, 10),
        tensor: value as OrtTensorLike<Float32Array>,
      });
    }
  }
  entries.sort((a, b) => a.layer - b.layer);
  return entries.map((e) => e.tensor);
}

interface BeamPayload {
  readonly tokenDetails: readonly WhisperNativeToken[];
  readonly pastKeyValues: Record<string, OrtTensorLike<Float32Array>>;
}

function roundMiB(bytes: number | undefined): number | undefined {
  if (!Number.isFinite(bytes)) return undefined;
  return roundMetric((bytes as number) / (1024 * 1024), 2);
}

function createAssetProgressEvent(
  modelId: string,
  file: string,
  event: { readonly loaded: number; readonly total?: number; readonly done?: boolean },
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

function isAssetMissingError(error: unknown): boolean {
  if (error instanceof Error) {
    return /\b404\b/.test(error.message) || /\bnot found\b/i.test(error.message);
  }
  return false;
}

function normalizeRepoPath(path: string): string {
  return String(path || '').replace(/^\.\/+/, '').replace(/\\/g, '/');
}

function hasListedRepoFile(files: readonly string[], filename: string): boolean {
  const target = normalizeRepoPath(filename);
  return files.some(
    (p) => normalizeRepoPath(p) === target || normalizeRepoPath(p).endsWith(`/${target}`),
  );
}

export function computeEmptyPastKeyValueShapes(
  config: WhisperModelConfig,
  encoderSeqLen: number,
): Record<string, readonly number[]> {
  const shapes: Record<string, readonly number[]> = {};
  const { decoderLayers, decoderAttentionHeads, headDim } = config;
  for (let i = 0; i < decoderLayers; i++) {
    shapes[`past_key_values.${i}.decoder.key`] = [1, decoderAttentionHeads, 0, headDim];
    shapes[`past_key_values.${i}.decoder.value`] = [1, decoderAttentionHeads, 0, headDim];
    shapes[`past_key_values.${i}.encoder.key`] = [1, decoderAttentionHeads, encoderSeqLen, headDim];
    shapes[`past_key_values.${i}.encoder.value`] = [1, decoderAttentionHeads, encoderSeqLen, headDim];
  }
  return shapes;
}

export interface SplitGraphDecodeCallbacks {
  runInit(
    promptTokens: readonly number[],
    encoderHiddenStates: Float32Array,
    encoderDims: readonly number[],
  ): Promise<{ logits: Float32Array; vocabSize: number; presentKv: Record<string, Float32Array> }>;
  runStep(
    tokenId: number,
    pastKv: Record<string, Float32Array>,
  ): Promise<{ logits: Float32Array; vocabSize: number; presentKv: Record<string, Float32Array> }>;
}

export interface SplitGraphDecodeResult {
  readonly tokens: readonly number[];
}

export async function splitGraphDecodeLoop(params: {
  promptTokens: readonly number[];
  encoderHiddenStates: Float32Array;
  eosTokenId: number;
  maxNewTokens: number;
  modelConfig: WhisperModelConfig;
  runInit: SplitGraphDecodeCallbacks['runInit'];
  runStep: SplitGraphDecodeCallbacks['runStep'];
  processLogits?: (logits: Float32Array, generatedTokens: readonly number[], beginIndex: number) => void;
}): Promise<SplitGraphDecodeResult> {
  const { promptTokens, encoderHiddenStates, eosTokenId, maxNewTokens, runInit, runStep, processLogits } = params;

  // Init: prefill with prompt tokens
  const initResult = await runInit(
    promptTokens,
    encoderHiddenStates,
    [1, encoderHiddenStates.length / params.modelConfig.dModel, params.modelConfig.dModel],
  );
  const initLogits = initResult.logits;
  const vocabSize = initResult.vocabSize;
  let pastKv = initResult.presentKv;

  // First token from init logits (last position)
  const lastLogitOffset = initLogits.length - vocabSize;
  const firstLogits = initLogits.subarray(lastLogitOffset);
  if (processLogits) {
    processLogits(firstLogits, promptTokens, promptTokens.length);
  }
  const firstTokenId = argmax(firstLogits);
  const tokens: number[] = [firstTokenId];

  // Autoregressive step loop
  for (let step = 1; step < maxNewTokens; step++) {
    const stepResult = await runStep(tokens[tokens.length - 1]!, pastKv);
    if (processLogits) {
      processLogits(stepResult.logits, [...promptTokens, ...tokens], promptTokens.length);
    }
    const nextTokenId = argmax(stepResult.logits);
    tokens.push(nextTokenId);
    pastKv = stepResult.presentKv;

    if (nextTokenId === eosTokenId) break;
  }

  return { tokens };
}

export interface SplitGraphAlignmentOptions {
  readonly alignmentData: Float32Array;
  readonly totalTokens: number;
  readonly promptLen: number;
  readonly textTokenCount: number;
  readonly frameCount: number;
  readonly medianFilterWidth?: number;
  readonly timePrecisionSeconds?: number;
}

export function processSplitGraphAlignment(
  options: SplitGraphAlignmentOptions,
): readonly number[] {
  const {
    alignmentData, totalTokens: _totalTokens, promptLen, textTokenCount, frameCount,
    medianFilterWidth, timePrecisionSeconds,
  } = options;

  if (textTokenCount === 0) return [0];
  if (frameCount === 0) return Array.from({ length: textTokenCount + 1 }, () => 0);

  // Slice off prompt rows: alignment[T_all, S] → extract text-only rows [promptLen:totalTokens, :]
  const textValues = new Float32Array(textTokenCount * frameCount);
  const srcOffset = promptLen * frameCount;
  textValues.set(alignmentData.subarray(srcOffset, srcOffset + textTokenCount * frameCount));

  const headMatrix = {
    values: textValues,
    tokenCount: textTokenCount,
    frameCount,
  };

  return computeWhisperDtwTokenTimestamps({
    attentionHeads: [headMatrix],
    tokenCount: textTokenCount,
    frameCount,
    medianFilterWidth,
    timePrecisionSeconds,
  });
}

export class WhisperOnnxExecutor {
  private readonly sourceOptions: WhisperArtifactSource | undefined;
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly assetHandles: ResolvedAssetHandle[] = [];

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: WhisperSeq2SeqModelConfig,
    private readonly backendId: string,
    loadOptions: { readonly source?: WhisperArtifactSource } | undefined,
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
    artifacts: ReturnType<typeof resolveWhisperArtifacts>['artifacts'],
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
      if (!filename) return undefined;
      const cacheKey = `huggingface:${source.repoId}:${revision}:${filename}`;
      const cacheKeyFallbacks = (source.cacheKeyFallbackRevisions ?? [])
        .filter((fb) => fb !== revision)
        .map((fb) => `huggingface:${source.repoId}:${fb}:${filename}`);

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

    const resolveOptionalFile = async (filename: string | undefined): Promise<string | undefined> => {
      if (!filename) return undefined;
      const repoFiles = await getRepoFiles();
      if (repoFiles.length > 0 && !hasListedRepoFile(repoFiles, filename)) {
        return undefined;
      }
      try {
        return await resolveFile(filename);
      } catch (error) {
        if (isAssetMissingError(error)) return undefined;
        throw error;
      }
    };

    void resolveOptionalFile;

    return {
      encoderUrl: (await resolveFile(artifacts.encoderUrl.split('/').pop())) ?? artifacts.encoderUrl,
      decoderUrl: (await resolveFile(artifacts.decoderUrl.split('/').pop())) ?? artifacts.decoderUrl,
      tokenizerUrl: (await resolveFile('tokenizer.json')) ?? artifacts.tokenizerUrl,
    };
  }

  private async initialize(): Promise<LoadedExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const resolved = resolveWhisperArtifacts(this.sourceOptions, this.backendId);
    const artifacts = await this.materializeHuggingFaceArtifacts(resolved.artifacts);

    const ort = await initWhisperOrt(resolved.ortBackend, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
    });

    const tokenizer = await WhisperTokenizer.fromUrl(artifacts.tokenizerUrl);
    const warnings = [...resolved.warnings];

    const encoderSession = await createWhisperOrtSession(ort, artifacts.encoderUrl, {
      backendId: resolved.encoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      ...(resolved.externalData?.encoder?.[0]
        ? { externalDataUrl: resolved.externalData.encoder[0].dataUrl, externalDataPath: resolved.externalData.encoder[0].path }
        : {}),
    });

    const decoderSession = await createWhisperOrtSession(ort, artifacts.decoderUrl, {
      backendId: resolved.decoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      ...(resolved.externalData?.decoder_init?.[0]
        ? { externalDataUrl: resolved.externalData.decoder_init[0].dataUrl, externalDataPath: resolved.externalData.decoder_init[0].path }
        : {}),
    });

    const genConfig = await this.loadGenerationConfig(artifacts);
    const modelConfig = await this.loadModelConfig(artifacts);
    const isSplitGraph = resolved.isSplitGraph;

    let decoderInitSession: OrtSessionLike | undefined;
    let decoderStepSession: OrtSessionLike | undefined;
    let decoderAlignSession: OrtSessionLike | undefined;

    if (isSplitGraph && resolved.decoderInitUrl && resolved.decoderStepUrl) {
      decoderInitSession = await createWhisperOrtSession(ort, resolved.decoderInitUrl, {
        backendId: resolved.decoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
        ...(resolved.externalData?.decoder_init?.[0]
          ? { externalDataUrl: resolved.externalData.decoder_init[0].dataUrl, externalDataPath: resolved.externalData.decoder_init[0].path }
          : {}),
      });
      decoderStepSession = await createWhisperOrtSession(ort, resolved.decoderStepUrl, {
        backendId: resolved.decoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
        ...(resolved.externalData?.decoder_step?.[0]
          ? { externalDataUrl: resolved.externalData.decoder_step[0].dataUrl, externalDataPath: resolved.externalData.decoder_step[0].path }
          : {}),
      });
      if (resolved.decoderAlignUrl) {
        decoderAlignSession = await createWhisperOrtSession(ort, resolved.decoderAlignUrl, {
          backendId: resolved.decoderBackendForOrt,
          enableProfiling: resolved.enableProfiling,
          ...(resolved.externalData?.decoder_align?.[0]
            ? { externalDataUrl: resolved.externalData.decoder_align[0].dataUrl, externalDataPath: resolved.externalData.decoder_align[0].path }
            : {}),
        });
      }
    }

    return {
      ort, tokenizer, encoderSession, decoderSession,
      generationConfig: genConfig, modelConfig, warnings,
      isSplitGraph, decoderInitSession, decoderStepSession, decoderAlignSession,
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

  private async runDecoderStep(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    generatedTokens: readonly number[],
    pastKeyValues: Record<string, OrtTensorLike<Float32Array>>,
    isFirstStep: boolean,
  ): Promise<DecoderStepResult> {
    const inputIds = new BigInt64Array(generatedTokens.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, generatedTokens.length]);
    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };

    const decoderInputNames = loaded.decoderSession.inputNames ?? [];
    if (decoderInputNames.includes('use_cache_branch')) {
      feeds.use_cache_branch = new loaded.ort.Tensor('bool', new Uint8Array([isFirstStep ? 1 : 0]), [1]);
    }

    if (!isFirstStep) {
      for (const [name, tensor] of Object.entries(pastKeyValues)) {
        feeds[name] = tensor;
      }
    } else {
      const numLayers = loaded.modelConfig.decoderLayers;
      const numHeads = loaded.modelConfig.decoderAttentionHeads;
      const headDim = loaded.modelConfig.headDim;
      const encoderSeqLen = encoderHiddenStates.dims[1] as number;
      for (let i = 0; i < numLayers; i++) {
        feeds[`past_key_values.${i}.decoder.key`] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(0),
          [1, numHeads, 0, headDim],
        );
        feeds[`past_key_values.${i}.decoder.value`] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(0),
          [1, numHeads, 0, headDim],
        );
        const encoderCacheSize = 1 * numHeads * encoderSeqLen * headDim;
        feeds[`past_key_values.${i}.encoder.key`] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(encoderCacheSize),
          [1, numHeads, encoderSeqLen, headDim],
        );
        feeds[`past_key_values.${i}.encoder.value`] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(encoderCacheSize),
          [1, numHeads, encoderSeqLen, headDim],
        );
      }
    }

    const outputs = await loaded.decoderSession.run(feeds);
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const logits = logitsTensor.data;
    const logitsDims = logitsTensor.dims;
    const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;
    const lastLogitsOffset = logits.length - vocabSize;

    const nextPastKeyValues: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (key.startsWith('present')) {
        const pastName = key.replace(/^present/, 'past_key_values');
        nextPastKeyValues[pastName] = value as OrtTensorLike<Float32Array>;
      }
    }

    return {
      lastLogits: logits.subarray(lastLogitsOffset),
      vocabSize,
      pastKeyValues: nextPastKeyValues,
      crossAttentions: extractCrossAttentions(outputs),
    };
  }

  private async runForcedAlignment(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    language: string,
    textTokenIds: number[],
  ): Promise<{
    readonly crossAttentions: readonly OrtTensorLike<Float32Array>[];
    readonly logitsForText: Float32Array;
  }> {
    const tokenizer = loaded.tokenizer;
    const sotId = tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
    const langToken = language === 'auto' ? '<|en|>' : `<|${language}|>`;
    const langId = tokenizer.getTokenId(langToken) ?? 50268;
    const taskId = tokenizer.getTokenId('<|transcribe|>') ?? 50359;
    const noTsId = tokenizer.getTokenId('<|notimestamps|>') ?? 50363;
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

    const forcedIds = [sotId, langId, taskId, noTsId, ...textTokenIds, eosId];
    const inputIds = new BigInt64Array(forcedIds.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, forcedIds.length]);

    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };

    const decoderInputNames = loaded.decoderSession.inputNames ?? [];
    if (decoderInputNames.includes('use_cache_branch')) {
      feeds.use_cache_branch = new loaded.ort.Tensor('bool', new Uint8Array([1]), [1]);
    }

    // First step: provide empty past_key_values
    // Use config-driven layer/head counts
    const numLayers = loaded.modelConfig.decoderLayers;
    const numHeads = loaded.modelConfig.decoderAttentionHeads;
    const headDim = loaded.modelConfig.headDim;
    const encoderSeqLen = encoderHiddenStates.dims[1] as number;
    for (let i = 0; i < numLayers; i++) {
      feeds[`past_key_values.${i}.decoder.key`] = new loaded.ort.Tensor(
        'float32', new Float32Array(0), [1, numHeads, 0, headDim]);
      feeds[`past_key_values.${i}.decoder.value`] = new loaded.ort.Tensor(
        'float32', new Float32Array(0), [1, numHeads, 0, headDim]);
      const encoderCacheSize = 1 * numHeads * encoderSeqLen * headDim;
      feeds[`past_key_values.${i}.encoder.key`] = new loaded.ort.Tensor(
        'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]);
      feeds[`past_key_values.${i}.encoder.value`] = new loaded.ort.Tensor(
        'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]);
    }

    const outputs = await loaded.decoderSession.run(feeds);
    const crossAttentions = extractCrossAttentions(outputs);

    // Extract logits for the text tokens (skip prompt + EOS)
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const totalVocab = (logitsTensor.dims[logitsTensor.dims.length - 1] as number) ?? 51865;

    // forcedIds: [SOT, lang, task, notimestamps, ...text, EOS]
    const promptLen = 4; // SOT + lang + task + notimestamps
    const textStart = promptLen;
    const textCount = textTokenIds.length;
    const logitsForText = new Float32Array(textCount * totalVocab);
    const srcOffset = textStart * totalVocab;
    logitsForText.set(logitsTensor.data.subarray(srcOffset, srcOffset + textCount * totalVocab));

    return { crossAttentions, logitsForText };
  }

  private async runForcedAlignmentSplitGraph(
    loaded: Required<Pick<LoadedExecutorState, 'decoderAlignSession' | 'ort'>> & LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    allTokenIds: readonly number[],
    _promptLen: number,
  ): Promise<Float32Array> {
    const inputIds = new BigInt64Array(allTokenIds.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, allTokenIds.length]);
    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };

    const outputs = await loaded.decoderAlignSession.run(feeds);
    const alignKey = Object.keys(outputs)[0]!;
    const alignTensor = outputs[alignKey] as OrtTensorLike<Float32Array>;

    return alignTensor.data;
  }

  private async computeAttentionWordTimestampsSplitGraph(
    loaded: Required<Pick<LoadedExecutorState, 'decoderAlignSession' | 'ort'>> & LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    tokenizer: WhisperTokenizer,
    segments: WhisperNativeSegment[],
    allTokens: readonly number[],
    promptLen: number,
    _options: WhisperSeq2SeqTranscriptionOptions,
  ): Promise<WhisperNativeTranscript['words']> {
    // Collect text token IDs from segments
    const textTokenIds: number[] = [];
    for (const seg of segments) {
      const ids = tokenizer.encode(seg.text);
      for (const id of ids) {
        if (!tokenizer.isSpecialTokenId(id) && !tokenizer.isTimestampTokenId(id)) {
          textTokenIds.push(id);
        }
      }
    }
    if (textTokenIds.length === 0) return [];

    try {
      const alignmentData = await this.runForcedAlignmentSplitGraph(
        loaded, encoderHiddenStates, allTokens, promptLen,
      );

      const encoderFrameCount = (encoderHiddenStates.dims[1] as number) ?? 0;
      const frameCount = encoderFrameCount; // decoder_align outputs full encoder seq
      const totalTokens = allTokens.length;

      const dtwTimestamps = processSplitGraphAlignment({
        alignmentData,
        totalTokens,
        promptLen,
        textTokenCount: textTokenIds.length,
        frameCount,
        medianFilterWidth: loaded.modelConfig.medianFilterWidth,
        timePrecisionSeconds: 0.02,
      });

      return this.buildWordsFromDtwTimestamps(
        tokenizer, textTokenIds, dtwTimestamps,
      );
    } catch {
      // Alignment failed — fall back to timestamp-token interpolation
      return buildWhisperWordTimestampsFromTokenDetails(
        [], // No token details with timestamps to build from
        {
          timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
          timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
          language: '',
        },
      );
    }
  }

  private async computeAttentionWordTimestamps(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    tokenizer: WhisperTokenizer,
    tokenDetails: WhisperNativeToken[],
    segments: WhisperNativeSegment[],
    language: string,
    _options: WhisperSeq2SeqTranscriptionOptions,
  ): Promise<WhisperNativeTranscript['words']> {
    const alignmentHeads = loaded.generationConfig.alignmentHeads;
    if (alignmentHeads.length === 0) {
      // No alignment heads configured — fall back to timestamp-token interpolation
      return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
        timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
        timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
        language,
      });
    }

    // Collect text token IDs from segments (exclude special/timestamp tokens)
    const textTokenIds: number[] = [];
    for (const seg of segments) {
      const ids = tokenizer.encode(seg.text);
      for (const id of ids) {
        if (!tokenizer.isSpecialTokenId(id) && !tokenizer.isTimestampTokenId(id)) {
          textTokenIds.push(id);
        }
      }
    }
    if (textTokenIds.length === 0) return [];

    try {
      const alignment = await this.runForcedAlignment(
        loaded, encoderHiddenStates, language, textTokenIds,
      );
      const crossAttentions = alignment.crossAttentions;
      if (crossAttentions.length === 0) {
        // Decoder had no cross-attention outputs — fall back
        return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
          timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
          timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
          language,
        });
      }

      // Build attention head matrices for DTW — select only alignment_heads
      const encoderFrameCount = (encoderHiddenStates.dims[1] as number) ?? 0;
      const croppedFrames = Math.floor(encoderFrameCount / 2); // Whisper encoder downsamples by 2

      const attentionHeads = alignmentHeads.map(({ layer, head }) => {
        const layerTensor = crossAttentions[layer];
        if (!layerTensor) {
          throw new Error(`Cross-attention layer ${layer} missing from decoder outputs.`);
        }
        const numLayerHeads = (layerTensor.dims[1] as number) ?? 6;
        if (head >= numLayerHeads) {
          throw new Error(`Alignment head ${head} exceeds layer ${layer} head count ${numLayerHeads}.`);
        }
        const totalTokens = (layerTensor.dims[2] as number) ?? 0;
        const totalFramesPerHead = (layerTensor.dims[3] as number) ?? 0;
        // Extract single head: tensor has shape [batch=1, heads, tokens, frames]
        const headSize = totalTokens * totalFramesPerHead;
        const headOffset = head * headSize;
        const headValues = layerTensor.data.subarray(headOffset, headOffset + headSize);
        return {
          values: new Float32Array(headValues),
          tokenCount: totalTokens,
          frameCount: croppedFrames,
        };
      });

      const dtwTimestamps = computeWhisperDtwTokenTimestamps({
        attentionHeads,
        tokenCount: textTokenIds.length,
        frameCount: croppedFrames,
        timePrecisionSeconds: 0.02,
      });

      // Compute token logprobs from forced alignment logits
      const logits = alignment.logitsForText;
      const totalVocab = (logits?.length ?? 0) / textTokenIds.length;
      let tokenLogprobs: Float32Array | undefined;
      if (logits && totalVocab > 0) {
        tokenLogprobs = new Float32Array(textTokenIds.length);
        for (let t = 0; t < textTokenIds.length; t++) {
          const tokenId = textTokenIds[t] ?? 0;
          // logprob = log(softmax(logits[t]))[tokenId]
          // = logits[t][tokenId] - logsumexp(logits[t])
          let maxVal = -Infinity;
          const tStart = t * totalVocab;
          for (let v = 0; v < totalVocab; v++) {
            const val = logits[tStart + v] ?? -Infinity;
            if (val > maxVal) maxVal = val;
          }
          let sumExp = 0;
          for (let v = 0; v < totalVocab; v++) {
            sumExp += Math.exp((logits[tStart + v] ?? -Infinity) - maxVal);
          }
          const logProb = (logits[tStart + tokenId] ?? -Infinity) - maxVal - Math.log(sumExp);
          tokenLogprobs[t] = logProb;
        }
      }

      // Build word timestamps from token timestamps using tokenizer
      return this.buildWordsFromDtwTimestamps(
        tokenizer, textTokenIds, dtwTimestamps, tokenLogprobs,
      );
    } catch {
      // Forced alignment failed — fall back to timestamp-token interpolation
      return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
        timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
        timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
        language,
      });
    }
  }

  private buildWordsFromDtwTimestamps(
    tokenizer: WhisperTokenizer,
    textTokenIds: number[],
    dtwTimestamps: readonly number[],
    tokenLogprobs?: Float32Array,
  ): WhisperNativeTranscript['words'] {
    // DTW gives tokenCount+1 timestamps — start of each text token + end of last
    const words: WhisperNativeWord[] = [];
    const text = tokenizer.decode(textTokenIds, { skipSpecialTokens: true });
    const wordTexts = text.split(/\s+/).filter((w) => w.length > 0);
    if (wordTexts.length === 0) return undefined;

    // Approximate word boundaries from token-to-word mapping using tokenizer encode
    const allWordTokenIds: number[][] = [];
    for (const w of wordTexts) {
      const ids = tokenizer.encode(w);
      if (ids.length > 0) allWordTokenIds.push(ids);
    }

    let tokenOffset = 0;
    for (let wi = 0; wi < allWordTokenIds.length; wi++) {
      const wIds = allWordTokenIds[wi]!;
      if (wIds.length === 0) continue;
      const wordStartIdx = tokenOffset;
      const wordEndIdx = tokenOffset + wIds.length - 1;
      const startTime = dtwTimestamps[wordStartIdx] ?? 0;
      const endTime = dtwTimestamps[wordEndIdx + 1] ?? dtwTimestamps[wordStartIdx] ?? 0;

      // Compute word probability as mean of token probabilities
      let confidence = -1;
      if (tokenLogprobs && wIds.length > 0) {
        let probSum = 0;
        for (let t = wordStartIdx; t <= wordEndIdx; t++) {
          probSum += Math.exp(tokenLogprobs[t] ?? -Infinity);
        }
        confidence = probSum / wIds.length;
      }

      words.push({
        index: wi,
        text: wordTexts[wi]!,
        startTime,
        endTime,
        confidence,
        tokenIds: wIds,
      });
      tokenOffset += wIds.length;
    }
    return words.length > 0 ? words : undefined;
  }

  private async loadGenerationConfig(
    artifacts: { readonly tokenizerUrl: string },
  ): Promise<WhisperGenerationConfig> {
    try {
      const genConfigUrl = artifacts.tokenizerUrl.replace(/tokenizer\.json$/, 'generation_config.json');
      const response = await fetch(genConfigUrl);
      if (!response.ok) return parseWhisperGenerationConfig({});
      const json = (await response.json()) as Record<string, unknown>;
      return parseWhisperGenerationConfig(json);
    } catch {
      return parseWhisperGenerationConfig({});
    }
  }

  private async loadModelConfig(
    artifacts: { readonly tokenizerUrl: string },
  ): Promise<WhisperModelConfig> {
    try {
      const configUrl = artifacts.tokenizerUrl.replace(/tokenizer\.json$/, 'config.json');
      const response = await fetch(configUrl);
      if (!response.ok) return parseWhisperModelConfig({});
      const json = (await response.json()) as Record<string, unknown>;
      return parseWhisperModelConfig(json);
    } catch {
      return parseWhisperModelConfig({});
    }
  }

  private async runDecoderInit(
    loaded: Required<Pick<LoadedExecutorState, 'decoderInitSession' | 'ort'>> & LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    promptTokens: readonly number[],
  ): Promise<{
    logits: Float32Array;
    vocabSize: number;
    presentKv: Record<string, OrtTensorLike<Float32Array>>;
  }> {
    const inputIds = new BigInt64Array(promptTokens.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, promptTokens.length]);
    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };

    const outputs = await loaded.decoderInitSession.run(feeds);
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const logitsDims = logitsTensor.dims;
    const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;

    const presentKv: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (key.startsWith('present')) {
        presentKv[key] = value as OrtTensorLike<Float32Array>;
      }
    }

    return { logits: logitsTensor.data, vocabSize, presentKv };
  }

  private async runDecoderStepSplit(
    loaded: Required<Pick<LoadedExecutorState, 'decoderStepSession' | 'ort'>> & LoadedExecutorState,
    tokenId: number,
    pastKv: Record<string, OrtTensorLike<Float32Array>>,
  ): Promise<{
    logits: Float32Array;
    vocabSize: number;
    presentKv: Record<string, OrtTensorLike<Float32Array>>;
  }> {
    const inputIdsTensor = new loaded.ort.Tensor('int64', new BigInt64Array([BigInt(tokenId)]), [1, 1]);
    const feeds: Record<string, unknown> = { input_ids: inputIdsTensor };

    // Add all past_key_values (decoder + encoder KV). Step model expects both.
    for (const [name, tensor] of Object.entries(pastKv)) {
      feeds[name] = tensor;
    }

    const outputs = await loaded.decoderStepSession.run(feeds);
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const logitsDims = logitsTensor.dims;
    const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;

    // decoder_step outputs only self-attention present KV. Merge with encoder KV from input.
    const presentKv: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (key.startsWith('present')) {
        const pastName = key.replace(/^present/, 'past_key_values');
        presentKv[pastName] = value as OrtTensorLike<Float32Array>;
      }
    }
    // Preserve encoder KV from input (they don't change and step model doesn't output them)
    for (const [key, value] of Object.entries(pastKv)) {
      if (key.includes('encoder') && !presentKv[key]) {
        presentKv[key] = value;
      }
    }

    return { logits: logitsTensor.data, vocabSize, presentKv };
  }

  async transcribe(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    _context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    const loaded = await this.getLoadedState();
    const warnings = [...loaded.warnings];

    if (loaded.isSplitGraph && loaded.decoderInitSession && loaded.decoderStepSession) {
      return this.transcribeWithSplitGraph(audio, options, _context);
    }

    if (this.shouldChunkAudio(audio, options)) {
      return this.transcribeLongAudio(audio, options, _context);
    }

    // 1. Preprocess audio to mel spectrogram
    const melProcessor = new WhisperMelProcessor({ nMels: this.config.melBins });
    // Audio is already normalized to mono by the session before calling executor
    const pcmData = audio.channels?.[0] ?? new Float32Array(0);
    const melResult = melProcessor.process(pcmData);
    // Whisper conv layers downsample by 2x: input 3000 frames → output 1500 time positions.
    const encoderOutputPositions = this.config.maxSourcePositions;
    const melInputFrames = encoderOutputPositions <= 1500 ? encoderOutputPositions * 2 : encoderOutputPositions;
    const paddedFeatures = WhisperMelProcessor.padToFrames(melResult, melInputFrames);

    // Reshape to [1, n_mels, melInputFrames] channels-first
    const featureTensor = new loaded.ort.Tensor(
      'float32',
      paddedFeatures,
      [1, this.config.melBins, melInputFrames],
    );

    // 2. Run encoder
    const encoderOutputs = await loaded.encoderSession.run({
      input_features: featureTensor,
    });
    const encoderHiddenStates = encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>;

    // 3. Build initial decoder input IDs
    const tokenizer = loaded.tokenizer;
    const language = options.language ?? this.config.languages[0] ?? 'auto';
    const langToken = language === 'auto' ? '<|en|>' : `<|${language}|>`;
    const taskToken = options.task === 'translate' ? '<|translate|>' : '<|transcribe|>';
    const noTimestampsToken = options.noTimestamps ? '<|notimestamps|>' : undefined;

    const promptTokens: number[] = [
      tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
      tokenizer.getTokenId(langToken) ?? 50268,
      tokenizer.getTokenId(taskToken) ?? 50359,
    ];
    if (noTimestampsToken) {
      const ntId = tokenizer.getTokenId(noTimestampsToken);
      if (ntId !== undefined) {
        promptTokens.push(ntId);
      }
    }

    // 4. Decode loop (greedy by default, beam search when numBeams > 1)
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
    const maxNewTokens = options.maxNewTokens ?? this.config.maxTargetPositions ?? 448;
    const numBeams = Math.max(1, Math.floor(options.numBeams ?? 1));
    const lengthPenalty = options.lengthPenalty ?? 0;
    const beamCandidateWidth = Math.max(
      numBeams,
      Math.ceil(numBeams * Math.max(1, options.patience ?? 1)),
    );
    let tokenDetails: WhisperNativeToken[] = [];

    // Build timestamp logit processor
    const timestampBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;
    const timestampProcessor = new WhisperTimestampLogitProcessor({
      eosTokenId: eosId,
      noTimestampsTokenId: loaded.generationConfig.noTimestampsTokenId ?? tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
      timestampBegin,
      suppressTokens: loaded.generationConfig.suppressTokens ?? [],
      beginSuppressTokens: loaded.generationConfig.beginSuppressTokens ?? [],
    });

    if (numBeams === 1) {
      const generatedTokens: number[] = [...promptTokens];
      let pastKeyValues: Record<string, OrtTensorLike<Float32Array>> = {};

      for (let step = 0; step < maxNewTokens; step++) {
        const result = await this.runDecoderStep(
          loaded,
          encoderHiddenStates,
          generatedTokens,
          pastKeyValues,
          step === 0,
        );
        pastKeyValues = result.pastKeyValues;
        timestampProcessor.process(result.lastLogits, generatedTokens, promptTokens.length);
        const nextTokenId = argmax(result.lastLogits);
        generatedTokens.push(nextTokenId);

        const { confidence } = confidenceFromLogits(
          new Float32Array(result.lastLogits),
          nextTokenId,
          result.vocabSize,
        );

        const tokenText = this.formatTokenText(tokenizer, nextTokenId);
        tokenDetails.push({
          index: step,
          id: nextTokenId,
          text: tokenText,
          confidence,
          special: tokenizer.isSpecialTokenId(nextTokenId),
        });

        if (nextTokenId === eosId) break;
      }
    } else {
      let beams: WhisperBeamState<BeamPayload>[] = [
        createInitialWhisperBeam(promptTokens, 0, { tokenDetails: [], pastKeyValues: {} }),
      ];

      for (let step = 0; step < maxNewTokens; step++) {
        const activeBeams = beams.filter((beam) => !beam.completed);
        if (activeBeams.length === 0) break;

        const logitsByBeam: Float32Array[] = [];
        const nextPastByBeam = new Map<WhisperBeamState<BeamPayload>, Record<string, OrtTensorLike<Float32Array>>>();
        let vocabSize = 0;

        for (const beam of beams) {
          if (beam.completed) {
            logitsByBeam.push(new Float32Array(0));
            continue;
          }
          const result = await this.runDecoderStep(
            loaded,
            encoderHiddenStates,
            beam.tokens,
            beam.payload?.pastKeyValues ?? {},
            step === 0,
          );
          timestampProcessor.process(result.lastLogits, beam.tokens, promptTokens.length);
          logitsByBeam.push(result.lastLogits);
          nextPastByBeam.set(beam, result.pastKeyValues);
          vocabSize = result.vocabSize;
        }

        beams = rankWhisperBeamCandidates({
          beams,
          logitsByBeam,
          beamWidth: beamCandidateWidth,
          eosTokenId: eosId,
          lengthPenalty,
          expandPayload: (beam, tokenId) => {
            const { confidence } = confidenceFromLogits(
              logitsByBeam[beams.indexOf(beam)] ?? new Float32Array(0),
              tokenId,
              vocabSize,
            );
            const tokenText = this.formatTokenText(tokenizer, tokenId);
            return {
              tokenDetails: [
                ...(beam.payload?.tokenDetails ?? []),
                {
                  index: step,
                  id: tokenId,
                  text: tokenText,
                  confidence,
                  special: tokenizer.isSpecialTokenId(tokenId),
                },
              ],
              pastKeyValues: nextPastByBeam.get(beam) ?? {},
            };
          },
        });
      }

      tokenDetails = [...(selectBestWhisperBeam(beams, lengthPenalty)?.payload?.tokenDetails ?? [])];
    }

    // 5. Build segments from decoded tokens
    const segments = this.buildSegments(tokenDetails, tokenizer, options.noTimestamps);
    const words = this.shouldReturnWordTimestamps(options)
      ? await this.computeAttentionWordTimestamps(
          loaded,
          encoderHiddenStates,
          tokenizer,
          tokenDetails,
          segments,
          language,
          options,
        )
      : [];
    const utteranceText = segments.map((s) => s.text).join(' ').trim();

    return {
      utteranceText,
      isFinal: true,
      language,
      segments,
      ...(words && words.length > 0 ? { words } : {}),
      tokens: options.returnSpecialTokens
        ? tokenDetails
        : tokenDetails.filter((t) => !t.special),
      warnings,
    };
  }

  private buildSegments(
    tokens: WhisperNativeToken[],
    tokenizer: WhisperTokenizer,
    noTimestamps?: boolean,
  ): WhisperNativeSegment[] {
    if (noTimestamps) {
      // No timestamps: single segment with all text
      const text = tokenizer.decode(
        tokens.map((t) => t.id ?? 0),
        { skipSpecialTokens: true },
      );
      if (!text.trim()) return [];
      return [
        {
          index: 0,
          text,
          startTime: 0,
          endTime: 30,
          confidence: tokens.length > 0 ? (tokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / tokens.length) : 0,
        },
      ];
    }

    // With timestamps: split on timestamp tokens
    const segments: WhisperNativeSegment[] = [];
    let currentTokens: WhisperNativeToken[] = [];
    let segmentStart = 0;

    for (const token of tokens) {
      if (tokenizer.isTimestampTokenId(token.id ?? 0)) {
        const ts = tokenizer.timestampTokenIdToSeconds(token.id ?? 0);
        if (ts !== undefined) {
          if (currentTokens.length > 0) {
            const text = tokenizer.decode(
              currentTokens.map((t) => t.id ?? 0),
              { skipSpecialTokens: true },
            );
            if (text.trim()) {
              const avgConf =
                currentTokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / currentTokens.length;
              segments.push({
                index: segments.length,
                text,
                startTime: segmentStart,
                endTime: ts,
                confidence: avgConf,
              });
            }
            currentTokens = [];
          }
          segmentStart = ts;
        }
      } else {
        currentTokens.push(token);
      }
    }

    // Remaining tokens
    if (currentTokens.length > 0) {
      const text = tokenizer.decode(
        currentTokens.map((t) => t.id ?? 0),
        { skipSpecialTokens: true },
      );
      if (text.trim()) {
        const avgConf =
          currentTokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / currentTokens.length;
        segments.push({
          index: segments.length,
          text,
          startTime: segmentStart,
          endTime: 30,
          confidence: avgConf,
        });
      }
    }

    return segments;
  }

  private shouldReturnWordTimestamps(options: WhisperSeq2SeqTranscriptionOptions): boolean {
    return options.returnWords === true || options.returnTimestamps === 'word' || options.detail === 'words' || options.detail === 'detailed';
  }

  private shouldChunkAudio(audio: AudioBufferLike, options: WhisperSeq2SeqTranscriptionOptions): boolean {
    if (options.windowing === 'disabled' || options.unsafeAllowOverMaxWindow) return false;
    const maxDuration = options.chunkLengthSeconds ?? options.maxInputDurationSeconds ?? 30;
    return audio.durationSeconds > maxDuration;
  }

  private async transcribeWithSplitGraph(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    _context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    const loaded = await this.getLoadedState();
    const warnings = [...loaded.warnings];
    const splitLoaded = loaded as Required<LoadedExecutorState>;

    // 1. Preprocess audio to mel spectrogram
    const melProcessor = new WhisperMelProcessor({ nMels: this.config.melBins });
    const pcmData = audio.channels?.[0] ?? new Float32Array(0);
    const melResult = melProcessor.process(pcmData);
    // Whisper conv layers downsample by 2x: input 3000 frames → output 1500 time positions.
    // config.maxSourcePositions is encoder output positions (1500); mel input needs 2x.
    const encoderOutputPositions = this.config.maxSourcePositions;
    const melInputFrames = encoderOutputPositions <= 1500 ? encoderOutputPositions * 2 : encoderOutputPositions;
    const paddedFeatures = WhisperMelProcessor.padToFrames(melResult, melInputFrames);

    const featureTensor = new loaded.ort.Tensor(
      'float32', paddedFeatures,
      [1, this.config.melBins, melInputFrames],
    );

    // 2. Run encoder
    const encoderOutputs = await loaded.encoderSession.run({ input_features: featureTensor });
    const encoderHiddenStates = encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>;

    // 3. Build initial prompt tokens
    const tokenizer = loaded.tokenizer;
    const language = options.language ?? this.config.languages[0] ?? 'auto';
    const langToken = language === 'auto' ? '<|en|>' : `<|${language}|>`;
    const taskToken = options.task === 'translate' ? '<|translate|>' : '<|transcribe|>';
    const noTimestampsToken = options.noTimestamps ? '<|notimestamps|>' : undefined;

    const promptTokens: number[] = [
      tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
      tokenizer.getTokenId(langToken) ?? 50268,
      tokenizer.getTokenId(taskToken) ?? 50359,
    ];
    if (noTimestampsToken) {
      const ntId = tokenizer.getTokenId(noTimestampsToken);
      if (ntId !== undefined) promptTokens.push(ntId);
    }

    // 4. Run 4-graph decode loop
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
    const maxNewTokens = options.maxNewTokens ?? this.config.maxTargetPositions ?? 448;

    // Only greedy decoding supported for splitgraph (no beam search yet)
    const timestampBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;
    const splitTimestampProcessor = new WhisperTimestampLogitProcessor({
      eosTokenId: eosId,
      noTimestampsTokenId: loaded.generationConfig.noTimestampsTokenId ?? tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
      timestampBegin,
      suppressTokens: loaded.generationConfig.suppressTokens ?? [],
      beginSuppressTokens: loaded.generationConfig.beginSuppressTokens ?? [],
    });

    const result = await splitGraphDecodeLoop({
      promptTokens,
      encoderHiddenStates: encoderHiddenStates.data,
      eosTokenId: eosId,
      maxNewTokens,
      modelConfig: loaded.modelConfig,
      processLogits: (logits, genTokens, beginIdx) => {
        splitTimestampProcessor.process(logits, genTokens, beginIdx);
      },
      runInit: async (prompt, _encHs, _dims) => {
        const init = await this.runDecoderInit(splitLoaded, encoderHiddenStates, prompt);
        return {
          logits: init.logits,
          vocabSize: init.vocabSize,
          presentKv: Object.fromEntries(
            Object.entries(init.presentKv).map(([k, v]) => [k, v.data]),
          ),
        };
      },
      runStep: async (tokenId, pastKv) => {
        // Reconstruct OrtTensorLike wrapping from data arrays
        const pastKvTensors: Record<string, OrtTensorLike<Float32Array>> = {};
        for (const [name, data] of Object.entries(pastKv)) {
          pastKvTensors[name] = { data, dims: [] } as OrtTensorLike<Float32Array>;
        }
        // Step model expects `past_key_values.` prefix but init outputs `present.` prefix
        // Convert present→past_key_values for step input
        const stepKv: Record<string, OrtTensorLike<Float32Array>> = {};
        for (const [key, tensor] of Object.entries(pastKvTensors)) {
          const stepKey = key.replace(/^present\./, 'past_key_values.');
          stepKv[stepKey] = tensor;
        }
        const step = await this.runDecoderStepSplit(splitLoaded, tokenId, stepKv);
        return {
          logits: step.logits,
          vocabSize: step.vocabSize,
          presentKv: Object.fromEntries(
            Object.entries(step.presentKv).map(([k, v]) => [k, v.data]),
          ),
        };
      },
    });

    // 5. Build token details
    const generatedTokens = [...promptTokens, ...result.tokens];
    const tokenDetails: WhisperNativeToken[] = [];
    for (let i = promptTokens.length; i < generatedTokens.length; i++) {
      const tokenId = generatedTokens[i]!;
      tokenDetails.push({
        index: i - promptTokens.length,
        id: tokenId,
        text: this.formatTokenText(tokenizer, tokenId),
        special: tokenizer.isSpecialTokenId(tokenId),
      });
    }

    // 6. Build segments
    const segments = this.buildSegments(tokenDetails, tokenizer, options.noTimestamps);

    // 7. Word timestamps via splitgraph alignment
    const words = this.shouldReturnWordTimestamps(options)
      ? loaded.decoderAlignSession
        ? await this.computeAttentionWordTimestampsSplitGraph(
            loaded as Required<LoadedExecutorState>,
            encoderHiddenStates, tokenizer, segments,
            generatedTokens, promptTokens.length, options,
          )
        : []
      : [];

    const utteranceText = segments.map((s) => s.text).join(' ').trim();

    return {
      utteranceText, isFinal: true, language, segments,
      ...(words && words.length > 0 ? { words } : {}),
      tokens: options.returnSpecialTokens
        ? tokenDetails
        : tokenDetails.filter((t) => !t.special),
      warnings,
    };
  }

  private async transcribeLongAudio(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    const pcmData = audio.channels?.[0] ?? new Float32Array(0);
    const chunkLengthSeconds = options.chunkLengthSeconds ?? 30;
    const chunks = planWhisperChunks(
      pcmData.length,
      audio.sampleRate,
      chunkLengthSeconds,
      options.strideLengthSeconds,
    );

    const chunkTranscripts = [];
    for (const chunk of chunks) {
      const samples = pcmData.slice(chunk.startSample, chunk.endSample);
      const chunkAudio: AudioBufferLike = {
        sampleRate: audio.sampleRate,
        durationSeconds: samples.length / audio.sampleRate,
        channels: [samples],
        numberOfChannels: 1,
        numberOfFrames: samples.length,
      };
      const transcript = await this.transcribe(
        chunkAudio,
        { ...options, unsafeAllowOverMaxWindow: true },
        context,
      );
      chunkTranscripts.push({ chunkStartTime: chunk.startTime, transcript });
    }

    return mergeWhisperChunkTranscripts(chunkTranscripts);
  }

  private formatTokenText(tokenizer: WhisperTokenizer, tokenId: number): string {
    if (tokenizer.isTimestampTokenId(tokenId) || tokenizer.isSpecialTokenId(tokenId)) {
      return tokenizer.idsToTokens([tokenId])[0] ?? '';
    }
    return tokenizer.decode([tokenId]);
  }

  async dispose(): Promise<void> {
    await Promise.all(
      this.assetHandles.map(async (handle) => {
        await handle.dispose();
      }),
    );
    this.assetHandles.length = 0;
  }
}
