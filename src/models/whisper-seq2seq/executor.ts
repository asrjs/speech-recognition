import type {
  AssetProvider,
  AudioBufferLike,
  ModelClassification,
  ResolvedAssetHandle,
  SpeechRuntimeHooks,
} from '../../types/index.js';
import { WhisperMelProcessor } from '../../audio/whisper-mel.js';
import { argmax, tokenQualityFromLogits } from '../../inference/index.js';
import {
  createInitialWhisperBeam,
  rankWhisperBeamCandidates,
  selectBestWhisperBeam,
  type WhisperBeamState,
} from './beam-search.js';
import {
  shouldChunkWhisperAudio,
  transcribeWhisperLongAudio,
} from './chunking.js';
import {
  buildWhisperSegments,
  computeAttentionWordTimestamps,
  executeAlignmentReferenceEncoder,
  finalizeMergedWhisperTranscript,
  finalizeWordTimestamps,
  hasVerifiedWhisperDecoderAlignment,
  runForcedAlignment,
} from './executor-alignment.js';
import {
  detectLanguageFromEncoder,
  detectLanguageFromMergedDecoder,
  runGreedyGpuKvDecode,
  runWhisperDecoderStepMultiToken,
  runWhisperMergedDecoderStep,
} from './executor-decoder-ops.js';
import {
  initializeWhisperExecutorState,
  loadWhisperGenerationConfig,
  loadWhisperModelConfig,
  materializeWhisperResolvedArtifacts,
} from './executor-init.js';
import {
  disposeOrtKv,
  disposeOrtTensor,
  disposeOwnedWhisperTensors,
  disposeReplacedOrtKv,
  throwIfDecodeAborted,
} from './executor-kv-cache.js';
import {
  formatWhisperTokenText,
  transcribeWithWhisperSplitGraph,
} from './executor-split-decode.js';
import type {
  BeamPayload,
  LoadedExecutorState,
  WhisperAlignmentRuntime,
} from './executor-types.js';
import {
  maybeCastEncoderHiddenStates,
  maybeCastWhisperFeatureTensor,
  resolveWhisperLanguageCode,
  resolveWhisperLanguageTokenId,
  runWhisperEncoderOnFeatures,
} from './executor-utils.js';
import {
  createWhisperOrtSession,
  releaseOrtSession,
  type OrtSessionLike,
  type OrtTensorLike,
  type ResolvedWhisperArtifacts,
} from './ort.js';
import { WhisperTimestampLogitProcessor } from './processors.js';
import type {
  WhisperArtifactSource,
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperSeq2SeqModelConfig,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';

export {
  assertExperimentalGpuKvCacheIsGreedyOnly,
  canReuseWhisperEncoderKvTensor,
  canShareWhisperEncoderKvAcrossBatch,
  cloneDecoderKvDataForInput,
  computeEmptyPastKeyValueShapes,
  concatDecoderKvDataForBatch,
  copyAndReleaseWhisperPresentKv,
  createWhisperBatchedCachePosition,
  createWhisperBatchedStepFeedTensors,
  disposeOrtKv,
  disposeOwnedDecoderFeeds,
  disposeOwnedWhisperTensors,
  disposeReplacedOrtKv,
  disposeWhisperEncoderKvCache,
  disposeWhisperEncoderKvCaches,
  sliceDecoderKvDataForBatch,
  supportsWhisperEncoderKvBroadcast,
  withOwnedDecoderFeeds,
} from './executor-kv-cache.js';

export {
  collectSplitGraphTextTokenRows,
  executeAlignmentReferenceEncoder,
  extractSplitGraphAlignmentHeadRows,
  extractSplitGraphAlignmentHeadWindow,
  extractSplitGraphAlignmentRows,
  extractSplitGraphAlignmentWindow,
  finalizeMergedWhisperTranscript,
  hasVerifiedWhisperDecoderAlignment,
  processSplitGraphAlignment,
  processSplitGraphAlignmentByTimestampSpans,
  resolveWhisperWordTimestampSource,
} from './executor-alignment.js';

export {
  createWhisperOrtSessionWithGraphCaptureFallback,
} from './executor-init.js';

export {
  splitGraphDecodeLoop,
} from './executor-decoder-ops.js';

export type {
  SplitGraphAlignmentOptions,
  SplitGraphDecodeCallbacks,
  SplitGraphDecodeResult,
  SplitGraphTimestampSpanAlignmentOptions,
  WhisperPresentKvSnapshot,
} from './executor-types.js';

export {
  buildWhisperForcedAlignmentTokenIds,
  float16BitsToFloat32,
  float32ToFloat16Bits,
  getWhisperForcedAlignmentTextRowStart,
  maybeCastEncoderHiddenStates,
  maybeCastWhisperFeatureTensor,
  resolveWhisperLanguageCode,
  resolveWhisperLanguageTokenId,
  runWhisperEncoderOnFeatures,
  tensorDataAsFloat32,
} from './executor-utils.js';

export class WhisperOnnxExecutor {
  private readonly sourceOptions: WhisperArtifactSource | undefined;
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private decoderAlignSession?: OrtSessionLike;
  private decoderAlignLoadPromise?: Promise<OrtSessionLike | undefined>;
  private alignmentReferenceEncoderSession?: OrtSessionLike;
  private alignmentReferenceEncoderLoadPromise?: Promise<OrtSessionLike | undefined>;
  private alignmentReferenceDecoderAlignSession?: OrtSessionLike;
  private alignmentReferenceDecoderAlignLoadPromise?: Promise<OrtSessionLike | undefined>;
  private disposed = false;
  private disposePromise?: Promise<void>;

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: WhisperSeq2SeqModelConfig,
    private readonly backendId: string,
    loadOptions: { readonly source?: WhisperArtifactSource } | undefined,
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

  private async materializeResolvedArtifacts(
    resolved: ResolvedWhisperArtifacts,
  ): Promise<ResolvedWhisperArtifacts> {
    return materializeWhisperResolvedArtifacts(
      {
        modelId: this.modelId,
        sourceOptions: this.sourceOptions,
        assetProvider: this.assetProvider,
        runtimeHooks: this.runtimeHooks,
        assetHandles: this.assetHandles,
      },
      resolved,
    );
  }

  private async initialize(): Promise<LoadedExecutorState> {
    return initializeWhisperExecutorState({
      modelId: this.modelId,
      backendId: this.backendId,
      sourceOptions: this.sourceOptions,
      signal: this.signal,
      isDisposed: () => this.disposed,
      materializeResolvedArtifacts: (resolved) => this.materializeResolvedArtifacts(resolved),
    });
  }

  private async getLoadedState(): Promise<LoadedExecutorState> {
    if (this.disposed) {
      throw new Error(`Whisper executor is disposed for "${this.modelId}".`);
    }
    if (!this.loadStatePromise) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }
    return this.loadStatePromise;
  }

  async ready(): Promise<void> {
    await this.getLoadedState();
  }

  private async lazyLoadSession(
    getCached: () => OrtSessionLike | undefined,
    setCached: (session: OrtSessionLike) => void,
    getPromise: () => Promise<OrtSessionLike | undefined> | undefined,
    setPromise: (promise: Promise<OrtSessionLike | undefined> | undefined) => void,
    createSession: () => Promise<OrtSessionLike>,
  ): Promise<OrtSessionLike | undefined> {
    const cached = getCached();
    if (cached) return cached;
    if (this.disposed) return undefined;
    const existing = getPromise();
    if (existing) return existing;
    const promise = (async () => {
      try {
        if (this.disposed) return undefined;
        const session = await createSession();
        if (this.disposed) {
          releaseOrtSession(session);
          return undefined;
        }
        setCached(session);
        return session;
      } catch (error) {
        setPromise(undefined);
        throw error;
      }
    })();
    setPromise(promise);
    return promise;
  }

  private async ensureDecoderAlignSession(
    loaded: LoadedExecutorState,
  ): Promise<OrtSessionLike | undefined> {
    if (this.decoderAlignSession) return this.decoderAlignSession;
    if (loaded.decoderAlignSession) {
      this.decoderAlignSession = loaded.decoderAlignSession;
      return this.decoderAlignSession;
    }
    if (!loaded.decoderAlignUrl) return undefined;
    return this.lazyLoadSession(
      () => this.decoderAlignSession,
      (s) => { this.decoderAlignSession = s; },
      () => this.decoderAlignLoadPromise,
      (p) => { this.decoderAlignLoadPromise = p; },
      () => createWhisperOrtSession(loaded.ort, loaded.decoderAlignUrl!, {
        backendId: loaded.decoderBackendForOrt ?? this.backendId,
        enableProfiling: loaded.enableProfiling,
        ...(loaded.decoderAlignExternalData
          ? { externalData: loaded.decoderAlignExternalData }
          : {}),
      }),
    );
  }

  private async ensureAlignmentReferenceEncoderSession(
    loaded: LoadedExecutorState,
  ): Promise<OrtSessionLike | undefined> {
    const reference = loaded.alignmentReference;
    if (!reference) return undefined;
    return this.lazyLoadSession(
      () => this.alignmentReferenceEncoderSession,
      (s) => { this.alignmentReferenceEncoderSession = s; },
      () => this.alignmentReferenceEncoderLoadPromise,
      (p) => { this.alignmentReferenceEncoderLoadPromise = p; },
      () => createWhisperOrtSession(loaded.ort, reference.encoderUrl, {
        backendId: reference.backendForOrt,
        enableProfiling: loaded.enableProfiling,
        ...(reference.backendForOrt.startsWith('webgpu')
          ? { preferredOutputLocation: 'gpu-buffer' as const }
          : {}),
        ...(reference.encoderExternalData
          ? { externalData: reference.encoderExternalData }
          : {}),
      }),
    );
  }

  private async ensureAlignmentReferenceDecoderAlignSession(
    loaded: LoadedExecutorState,
  ): Promise<OrtSessionLike | undefined> {
    const reference = loaded.alignmentReference;
    if (!reference) return undefined;
    return this.lazyLoadSession(
      () => this.alignmentReferenceDecoderAlignSession,
      (s) => { this.alignmentReferenceDecoderAlignSession = s; },
      () => this.alignmentReferenceDecoderAlignLoadPromise,
      (p) => { this.alignmentReferenceDecoderAlignLoadPromise = p; },
      () => createWhisperOrtSession(loaded.ort, reference.decoderAlignUrl, {
        backendId: reference.backendForOrt,
        enableProfiling: loaded.enableProfiling,
        ...(reference.decoderAlignExternalData
          ? { externalData: reference.decoderAlignExternalData }
          : {}),
      }),
    );
  }

  private async ensureAlignmentReferenceRuntime(
    loaded: LoadedExecutorState,
  ): Promise<WhisperAlignmentRuntime | undefined> {
    const reference = loaded.alignmentReference;
    if (
      !reference ||
      !hasVerifiedWhisperDecoderAlignment(reference.decoderAlignCausalSelfAttention)
    ) {
      return undefined;
    }
    const decoderAlignSession = await this.ensureAlignmentReferenceDecoderAlignSession(loaded);
    if (!decoderAlignSession) return undefined;
    return {
      decoderAlignSession,
      decoderAlignCausalSelfAttention: reference.decoderAlignCausalSelfAttention,
      decoderAlignAttentionValues: reference.decoderAlignAttentionValues,
      decoderAlignAttentionLayout: reference.decoderAlignAttentionLayout,
      decoderAlignExternalData: reference.decoderAlignExternalData,
    };
  }

  private async runAlignmentReferenceEncoder(
    loaded: LoadedExecutorState,
    paddedFeatures: Float32Array,
    melBins: number,
    melInputFrames: number,
    ownedEncoderTensors: Set<OrtTensorLike>,
  ): Promise<{
    readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
    readonly alignmentRuntime: WhisperAlignmentRuntime;
  }> {
    const reference = loaded.alignmentReference;
    if (!reference) {
      throw new Error('No Whisper alignment-reference artifact is configured.');
    }
    const encoderSession = await this.ensureAlignmentReferenceEncoderSession(loaded);
    const alignmentRuntime = await this.ensureAlignmentReferenceRuntime(loaded);
    if (!encoderSession || !alignmentRuntime) {
      throw new Error(
        'Whisper alignment-reference artifacts are unavailable or lack a verified causal alignment graph.',
      );
    }
    return executeAlignmentReferenceEncoder(
      loaded,
      encoderSession,
      alignmentRuntime,
      paddedFeatures,
      melBins,
      melInputFrames,
      ownedEncoderTensors,
    );
  }

  private async transcribeWithSplitGraph(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    _context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
    preloaded?: LoadedExecutorState,
  ): Promise<WhisperNativeTranscript> {
    const loaded = preloaded ?? (await this.getLoadedState());
    return transcribeWithWhisperSplitGraph(
      {
        modelId: this.modelId,
        backendId: this.backendId,
        config: this.config,
        loaded,
        ensureDecoderAlignSession: (state) => this.ensureDecoderAlignSession(state),
        runAlignmentReferenceEncoder: (...args) => this.runAlignmentReferenceEncoder(...args),
      },
      audio,
      options,
    );
  }

  async transcribe(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    _context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    const loaded = await this.getLoadedState();
    const warnings = [...loaded.warnings];

    if (loaded.isSplitGraph && loaded.decoderInitSession && loaded.decoderStepSession) {
      return this.transcribeWithSplitGraph(audio, options, _context, loaded);
    }

    if (this.shouldChunkAudio(audio, options)) {
      return this.transcribeLongAudio(audio, options, _context);
    }

    // 1. Preprocess audio to mel spectrogram
    const melBins = loaded.modelConfig.numMelBins ?? this.config.melBins;
    const melProcessor = new WhisperMelProcessor({ nMels: melBins });
    const pcmData = audio.channels?.[0] ?? new Float32Array(0);
    const melResult = melProcessor.process(pcmData);
    const encoderOutputPositions = this.config.maxSourcePositions;
    const melInputFrames = encoderOutputPositions <= 1500 ? encoderOutputPositions * 2 : encoderOutputPositions;
    const paddedFeatures = WhisperMelProcessor.padToFrames(melResult, melInputFrames);

    const featureTensor = await maybeCastWhisperFeatureTensor(new loaded.ort.Tensor(
      'float32',
      paddedFeatures,
      [1, melBins, melInputFrames],
    ), loaded.encoderSession, loaded.ort);

    const ownedEncoderTensors = new Set<OrtTensorLike>();
    try {
      // 2. Run encoder
      const encoderOutputs = await runWhisperEncoderOnFeatures(loaded.encoderSession, featureTensor);
      for (const tensor of Object.values(encoderOutputs)) ownedEncoderTensors.add(tensor);
      const encoderHiddenStates = await maybeCastEncoderHiddenStates(
        encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>,
        loaded.decoderInitSession ?? loaded.decoderSession!,
        loaded.ort,
      );
      ownedEncoderTensors.add(encoderHiddenStates);

      // 3. Detect language if auto
      const tokenizer = loaded.tokenizer;
      let language = options.language ?? this.config.languages[0] ?? 'auto';
      if (language === 'auto' && loaded.decoderSession) {
        language = await detectLanguageFromMergedDecoder(loaded, encoderHiddenStates);
      }
      language = resolveWhisperLanguageCode(language, this.config.languages);
      const taskToken = options.task === 'translate' ? '<|translate|>' : '<|transcribe|>';
      const noTimestampsToken = options.noTimestamps ? '<|notimestamps|>' : undefined;

      const promptTokens: number[] = [
        tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
        resolveWhisperLanguageTokenId(tokenizer, language),
        tokenizer.getTokenId(taskToken) ?? 50360,
      ];
      if (noTimestampsToken) {
        const ntId = tokenizer.getTokenId(noTimestampsToken);
        if (ntId !== undefined) {
          promptTokens.push(ntId);
        }
      }

      // 4. Decode loop (greedy by default, beam search when numBeams > 1)
      const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
      const noSpeechTokenId = loaded.generationConfig.noSpeechTokenId
        ?? tokenizer.getTokenId('<|nospeech|>')
        ?? 50362;
      const maxNewTokens = options.maxNewTokens ?? this.config.maxTargetPositions ?? 448;
      const numBeams = Math.max(1, Math.floor(options.numBeams ?? 1));
      const lengthPenalty = options.lengthPenalty;
      const beamCandidateWidth = Math.max(
        numBeams,
        Math.ceil(numBeams * Math.max(1, options.patience ?? 1)),
      );
      let tokenDetails: WhisperNativeToken[] = [];

      const timestampBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;
      const timestampProcessor = new WhisperTimestampLogitProcessor({
        eosTokenId: eosId,
        noTimestampsTokenId: loaded.generationConfig.noTimestampsTokenId ?? tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
        timestampBegin,
        suppressTokens: loaded.generationConfig.suppressTokens ?? [],
        beginSuppressTokens: loaded.generationConfig.beginSuppressTokens ?? [],
      });

      let selectedTokenTraces: Array<{ tokenId: number; logProb: number; entropy: number }> = [];

      if (numBeams === 1) {
        const generatedTokens: number[] = [...promptTokens];
        let pastKeyValues: Record<string, OrtTensorLike<Float32Array>> = {};

        try {
          for (let step = 0; step < maxNewTokens; step++) {
            throwIfDecodeAborted(options.signal);
            const result = await runWhisperMergedDecoderStep(
              loaded,
              encoderHiddenStates,
              generatedTokens,
              pastKeyValues,
              step === 0,
            );
            disposeReplacedOrtKv(pastKeyValues, result.pastKeyValues);
            pastKeyValues = result.pastKeyValues;
            if (step === 0) {
              options.onDecoderInitLogits?.(new Float32Array(result.lastLogits), {
                tokens: promptTokens,
                beginIndex: promptTokens.length,
                vocabSize: result.vocabSize,
                noSpeechTokenId,
              });
            }
            timestampProcessor.process(result.lastLogits, generatedTokens, promptTokens.length);
            const nextTokenId = argmax(result.lastLogits);
            generatedTokens.push(nextTokenId);

            const quality = tokenQualityFromLogits(
              result.lastLogits,
              nextTokenId,
              result.vocabSize,
            );
            selectedTokenTraces.push({
              tokenId: nextTokenId,
              logProb: quality.logProb,
              entropy: quality.entropy,
            });

            const tokenText = formatWhisperTokenText(tokenizer, nextTokenId);
            tokenDetails.push({
              index: step,
              id: nextTokenId,
              text: tokenText,
              confidence: quality.confidence,
              special: tokenizer.isSpecialTokenId(nextTokenId),
            });

            if (nextTokenId === eosId) break;
          }
        } finally {
          disposeOrtKv(pastKeyValues);
        }
      } else {
        let beams: WhisperBeamState<BeamPayload>[] = [
          createInitialWhisperBeam(promptTokens, 0, { tokenDetails: [], pastKeyValues: {}, tokenTraces: [] }),
        ];

        try {
          for (let step = 0; step < maxNewTokens; step++) {
            throwIfDecodeAborted(options.signal);
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
              throwIfDecodeAborted(options.signal);
              const result = await runWhisperMergedDecoderStep(
                loaded,
                encoderHiddenStates,
                beam.tokens,
                beam.payload?.pastKeyValues ?? {},
                step === 0,
              );
              if (step === 0) {
                options.onDecoderInitLogits?.(new Float32Array(result.lastLogits), {
                  tokens: promptTokens,
                  beginIndex: promptTokens.length,
                  vocabSize: result.vocabSize,
                  noSpeechTokenId,
                });
              }
              timestampProcessor.process(result.lastLogits, beam.tokens, promptTokens.length);
              logitsByBeam.push(result.lastLogits);
              nextPastByBeam.set(beam, result.pastKeyValues);
              vocabSize = result.vocabSize;
            }

            const staleTensors = new Set<OrtTensorLike>();
            for (const beam of beams) {
              for (const tensor of Object.values(beam.payload?.pastKeyValues ?? {})) {
                staleTensors.add(tensor);
              }
            }
            for (const kv of nextPastByBeam.values()) {
              for (const tensor of Object.values(kv)) {
                staleTensors.add(tensor);
              }
            }

            beams = rankWhisperBeamCandidates({
              beams,
              logitsByBeam,
              beamWidth: beamCandidateWidth,
              eosTokenId: eosId,
              lengthPenalty,
              expandPayload: (beam, tokenId, logProb) => {
                const parentLogits = logitsByBeam[beams.indexOf(beam)] ?? new Float32Array(0);
                const quality = tokenQualityFromLogits(parentLogits, tokenId, vocabSize);
                const tokenText = formatWhisperTokenText(tokenizer, tokenId);
                return {
                  tokenDetails: [
                    ...(beam.payload?.tokenDetails ?? []),
                    {
                      index: step,
                      id: tokenId,
                      text: tokenText,
                      confidence: quality.confidence,
                      special: tokenizer.isSpecialTokenId(tokenId),
                    },
                  ],
                  pastKeyValues: nextPastByBeam.get(beam) ?? {},
                  tokenTraces: [
                    ...(beam.payload?.tokenTraces ?? []),
                    {
                      tokenId,
                      logProb,
                      entropy: quality.entropy,
                    },
                  ],
                };
              },
            });
            const liveTensors = new Set<OrtTensorLike>();
            for (const beam of beams) {
              for (const tensor of Object.values(beam.payload?.pastKeyValues ?? {})) {
                liveTensors.add(tensor);
              }
            }
            for (const tensor of staleTensors) {
              if (!liveTensors.has(tensor)) disposeOrtTensor(tensor);
            }
          }

          const bestBeam = selectBestWhisperBeam(beams, lengthPenalty);
          tokenDetails = [...(bestBeam?.payload?.tokenDetails ?? [])];
          selectedTokenTraces = [...(bestBeam?.payload?.tokenTraces ?? [])];
        } finally {
          const seen = new Set<OrtTensorLike>();
          for (const beam of beams) {
            for (const tensor of Object.values(beam.payload?.pastKeyValues ?? {})) {
              if (seen.has(tensor)) continue;
              seen.add(tensor);
              disposeOrtTensor(tensor);
            }
          }
        }
      }

      // 5. Build segments and finalize transcript
      return finalizeMergedWhisperTranscript({
        loaded,
        configLanguages: this.config.languages,
        encoderHiddenStates,
        tokenizer,
        tokenDetails,
        options,
        audio,
        language,
        warnings,
        ownedEncoderTensors,
        selectedTokenTraces,
      });
    } finally {
      disposeOwnedWhisperTensors(ownedEncoderTensors);
    }
  }

  private shouldChunkAudio(audio: AudioBufferLike, options: WhisperSeq2SeqTranscriptionOptions): boolean {
    return shouldChunkWhisperAudio(audio, options);
  }

  private async transcribeLongAudio(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    return transcribeWhisperLongAudio(audio, options, (chunkAudio, chunkOpts) =>
      this.transcribe(chunkAudio, chunkOpts, context),
    );
  }

  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.disposePromise = this.flushDispose();
    return this.disposePromise;
  }

  private async flushDispose(): Promise<void> {
    const sessions = new Set<OrtSessionLike>();
    const collect = (session: OrtSessionLike | undefined) => {
      if (session) sessions.add(session);
    };

    if (this.loadStatePromise) {
      try {
        const loaded = await this.loadStatePromise;
        collect(loaded.encoderSession);
        collect(loaded.decoderSession);
        collect(loaded.decoderInitSession);
        collect(loaded.decoderStepSession);
        collect(loaded.decoderAlignSession);
      } catch {
        // Keep the original load error; still drop lazy sessions and asset handles.
      }
    }

    for (const promise of [
      this.decoderAlignLoadPromise,
      this.alignmentReferenceEncoderLoadPromise,
      this.alignmentReferenceDecoderAlignLoadPromise,
    ]) {
      if (!promise) continue;
      try {
        collect(await promise);
      } catch {
        // Keep the original lazy-load error.
      }
    }

    collect(this.decoderAlignSession);
    collect(this.alignmentReferenceEncoderSession);
    collect(this.alignmentReferenceDecoderAlignSession);

    for (const session of sessions) {
      releaseOrtSession(session);
    }

    this.decoderAlignSession = undefined;
    this.alignmentReferenceEncoderSession = undefined;
    this.alignmentReferenceDecoderAlignSession = undefined;
    this.decoderAlignLoadPromise = undefined;
    this.alignmentReferenceEncoderLoadPromise = undefined;
    this.alignmentReferenceDecoderAlignLoadPromise = undefined;

    const handles = this.assetHandles.splice(0);
    await Promise.all(handles.map((handle) => Promise.resolve(handle.dispose())));
  }

  protected readonly loadGenerationConfig = loadWhisperGenerationConfig;
  protected readonly loadModelConfig = loadWhisperModelConfig;
  protected readonly runGreedyGpuKvDecode = runGreedyGpuKvDecode;
  protected readonly runDecoderStep = runWhisperMergedDecoderStep;
  protected readonly runDecoderStepMultiToken = runWhisperDecoderStepMultiToken;
  protected readonly detectLanguageFromEncoder = detectLanguageFromEncoder;
  protected readonly detectLanguageFromMergedDecoder = detectLanguageFromMergedDecoder;
  protected readonly finalizeWordTimestamps = finalizeWordTimestamps;
  protected readonly buildSegments = buildWhisperSegments;

  protected async runForcedAlignment(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    language: string,
    textTokenIds: number[],
    task: 'transcribe' | 'translate' = 'transcribe',
    ownedEncoderTensors?: Set<OrtTensorLike>,
  ) {
    return runForcedAlignment(
      loaded, this.config.languages, encoderHiddenStates, language, textTokenIds, task, ownedEncoderTensors,
    );
  }

  protected async computeAttentionWordTimestamps(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    tokenizer: LoadedExecutorState['tokenizer'],
    tokenDetails: WhisperNativeToken[],
    segments: import('./types.js').WhisperNativeSegment[],
    language: string,
    options: WhisperSeq2SeqTranscriptionOptions,
    audioDurationSeconds?: number,
    warnings?: import('../../types/index.js').TranscriptWarning[],
    ownedEncoderTensors?: Set<OrtTensorLike>,
  ) {
    return computeAttentionWordTimestamps({
      loaded, configLanguages: this.config.languages, encoderHiddenStates,
      tokenizer, tokenDetails, segments, language, options, audioDurationSeconds, warnings,
      ownedEncoderTensors,
      runForcedAlignmentFn: (l, e, lang, ids, t, owned) => this.runForcedAlignment(l, e, lang, ids, t, owned),
    });
  }
}
