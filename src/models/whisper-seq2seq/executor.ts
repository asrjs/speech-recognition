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
import {
  createWhisperOrtSession,
  initWhisperOrt,
  resolveWhisperArtifacts,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from './ort.js';
import { WhisperTokenizer } from './tokenizer.js';
import type {
  WhisperArtifactSource,
  WhisperNativeSegment,
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperSeq2SeqModelConfig,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: WhisperTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly decoderSession: OrtSessionLike;
  readonly warnings: readonly TranscriptWarning[];
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
    });

    const decoderSession = await createWhisperOrtSession(ort, artifacts.decoderUrl, {
      backendId: resolved.decoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
    });

    return { ort, tokenizer, encoderSession, decoderSession, warnings };
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
    options: WhisperSeq2SeqTranscriptionOptions,
    _context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    const loaded = await this.getLoadedState();
    const warnings = [...loaded.warnings];

    // 1. Preprocess audio to mel spectrogram
    const melProcessor = new WhisperMelProcessor({ nMels: this.config.melBins });
    // Audio is already normalized to mono by the session before calling executor
    const pcmData = audio.channels?.[0] ?? new Float32Array(0);
    const melResult = melProcessor.process(pcmData);
    const maxFrames = this.config.maxSourcePositions; // 1500 for Whisper
    const paddedFeatures = WhisperMelProcessor.padToFrames(melResult, maxFrames);

    // Reshape to [1, n_mels, maxFrames] channels-first
    const featureTensor = new loaded.ort.Tensor(
      'float32',
      paddedFeatures,
      [1, this.config.melBins, maxFrames],
    );

    // 2. Run encoder
    const encoderOutputs = await loaded.encoderSession.run({
      input_features: featureTensor,
    });
    const encoderHiddenStates = encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>;

    // 3. Build initial decoder input IDs
    const tokenizer = loaded.tokenizer;
    const language = options.language ?? this.config.languages[0] ?? 'auto';
    const langToken = language === 'auto' ? '<|tr|>' : `<|${language}|>`;
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

    // 4. Greedy decode loop
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
    const maxNewTokens = options.maxNewTokens ?? this.config.maxTargetPositions ?? 448;
    const generatedTokens: number[] = [...promptTokens];
    const tokenDetails: WhisperNativeToken[] = [];

    // Inspect decoder session for cache tensor names
    const decoderInputNames = loaded.decoderSession.inputNames ?? [];
    const hasCacheBranch = decoderInputNames.includes('use_cache_branch');

    let pastKeyValues: Record<string, OrtTensorLike<Float32Array>> = {};

    for (let step = 0; step < maxNewTokens; step++) {
      const isFirstStep = step === 0;
      const inputIds = new BigInt64Array(generatedTokens.map((id) => BigInt(id)));
      const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, generatedTokens.length]);

      const feeds: Record<string, unknown> = {
        input_ids: inputIdsTensor,
        encoder_hidden_states: encoderHiddenStates,
      };

      if (hasCacheBranch) {
        // ORT merged decoder uses a bool scalar
        feeds.use_cache_branch = new loaded.ort.Tensor('bool', new Uint8Array([isFirstStep ? 1 : 0]), [1]);
      }

      // Add past key values from previous step
      if (!isFirstStep) {
        for (const [name, tensor] of Object.entries(pastKeyValues)) {
          feeds[name] = tensor;
        }
      } else {
        // First step: provide empty past_key_values tensors for the merged decoder
        const numLayers = 4;
        const numHeads = 6;
        const headDim = 64;
        const encoderSeqLen = encoderHiddenStates.dims[1] as number;
        for (let i = 0; i < numLayers; i++) {
          feeds[`past_key_values.${i}.decoder.key`] = new loaded.ort.Tensor(
            'float32', new Float32Array(0), [1, numHeads, 0, headDim]
          );
          feeds[`past_key_values.${i}.decoder.value`] = new loaded.ort.Tensor(
            'float32', new Float32Array(0), [1, numHeads, 0, headDim]
          );
          const encoderCacheSize = 1 * numHeads * encoderSeqLen * headDim;
          feeds[`past_key_values.${i}.encoder.key`] = new loaded.ort.Tensor(
            'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]
          );
          feeds[`past_key_values.${i}.encoder.value`] = new loaded.ort.Tensor(
            'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]
          );
        }
      }

      const outputs = await loaded.decoderSession.run(feeds);

      // Extract logits
      const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
      const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
      const logits = logitsTensor.data;
      const logitsDims = logitsTensor.dims;
      const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;

      // Get last token logits
      const lastLogitsOffset = logits.length - vocabSize;
      const lastLogits = logits.subarray(lastLogitsOffset);
      const nextTokenId = argmax(lastLogits);

      // Extract present key values for next step
      pastKeyValues = {};
      for (const [key, value] of Object.entries(outputs)) {
        if (key.startsWith('present')) {
          const pastName = key.replace(/^present/, 'past_key_values');
          pastKeyValues[pastName] = value as OrtTensorLike<Float32Array>;
        }
      }

      generatedTokens.push(nextTokenId);

      const { confidence } = confidenceFromLogits(
        new Float32Array(lastLogits),
        nextTokenId,
        vocabSize,
      );

      const tokenText = tokenizer.idsToTokens([nextTokenId])[0] ?? '';
      tokenDetails.push({
        index: step,
        id: nextTokenId,
        text: tokenText,
        confidence,
        special: tokenizer.isSpecialTokenId(nextTokenId),
      });

      if (nextTokenId === eosId) {
        break;
      }
    }

    // 5. Build segments from decoded tokens
    const segments = this.buildSegments(tokenDetails, tokenizer, options.noTimestamps);
    const utteranceText = segments.map((s) => s.text).join(' ').trim();

    return {
      utteranceText,
      isFinal: true,
      language,
      segments,
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

  async dispose(): Promise<void> {
    await Promise.all(
      this.assetHandles.map(async (handle) => {
        await handle.dispose();
      }),
    );
    this.assetHandles.length = 0;
  }
}
