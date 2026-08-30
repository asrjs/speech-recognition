import { argmax } from '../../inference/index.js';
import { nowMs, roundMetric } from '../../runtime/timing.js';
import { createExperimentalArtifactMissingError } from '../../runtime/experimental-families.js';
import type { AbortSignalLike, AssetProvider, AudioBufferLike, ResolvedAssetHandle, SpeechRuntimeHooks } from '../../types/index.js';
import { PipelineAbortedError } from '../../pipeline/composition.js';
import { createOrtSession, initOrt, releaseOrtSession, type OrtModuleLike, type OrtOutputLocation, type OrtSessionLike, type OrtTensorLike } from '../lasr-ctc/ort.js';
import type { XAsrArtifactSource, XAsrModelConfig, XAsrModelOptions, XAsrNativeTranscript, XAsrStateTensorSpec, XAsrTranscriptionOptions } from './types.js';
import { XAsrJsFrontend } from './frontend.js';
import { XAsrTokenizer } from './tokenizer.js';

export interface XAsrStreamState {
  readonly audio: Float32Array;
  /**
   * Amortized backing storage for cumulative audio. The public `audio` view
   * keeps its exact logical length while appends reuse this capacity buffer.
   */
  readonly audioBuffer?: ArrayBuffer;
  readonly features: Float32Array;
  /** Bounded raw-audio tail used by the incremental fbank frontend. */
  readonly frontendTail?: Float32Array;
  readonly encodedFrames: number;
  readonly inputFrames: number;
  readonly tokenIds: readonly number[];
  readonly encoderStates: readonly OrtTensorLike[];
}

interface AppendedStreamAudio {
  readonly audio: Float32Array;
  readonly audioBuffer?: ArrayBuffer;
}

function appendStreamAudio(state: XAsrStreamState, appended: Float32Array): AppendedStreamAudio {
  const previous = state.audio;
  const required = previous.length + appended.length;
  if (required === 0) return { audio: previous, audioBuffer: state.audioBuffer };

  const reusable = state.audioBuffer && previous.buffer === state.audioBuffer && previous.byteOffset === 0
    ? state.audioBuffer
    : undefined;
  const currentCapacity = reusable ? Math.floor(reusable.byteLength / Float32Array.BYTES_PER_ELEMENT) : 0;
  if (reusable && currentCapacity >= required) {
    const view = new Float32Array(reusable, 0, required);
    view.set(appended, previous.length);
    return { audio: view, audioBuffer: reusable };
  }

  let capacity = Math.max(required, currentCapacity > 0 ? Math.ceil(currentCapacity * 1.5) : required);
  while (capacity < required) capacity = Math.max(required, Math.ceil(capacity * 1.5));
  const buffer = new ArrayBuffer(capacity * Float32Array.BYTES_PER_ELEMENT);
  const view = new Float32Array(buffer, 0, required);
  view.set(previous);
  view.set(appended, previous.length);
  return { audio: view, audioBuffer: buffer };
}

function disposeStreamTensors(state: XAsrStreamState, retained?: XAsrStreamState): void {
  const retainedTensors = new Set(retained?.encoderStates ?? []);
  for (const value of state.encoderStates) {
    if (!retainedTensors.has(value)) value?.dispose?.();
  }
}

interface LoadedGraph {
  readonly featureInputName: string;
  readonly encoderOutputName: string;
  readonly encoderFrameSize: number;
  readonly encoderFrameShift: number;
  readonly encoderStateInputs: readonly XAsrStateTensorSpec[];
  readonly decoderInputName: string;
  readonly decoderOutputName: string;
  readonly decoderContextSize: number;
  readonly decoderIndexType: 'int32' | 'int64';
  readonly joinerEncoderInputName: string;
  readonly joinerDecoderInputName: string;
  readonly joinerOutputName: string;
}

interface LoadedState {
  readonly ort: OrtModuleLike;
  readonly encoder: OrtSessionLike;
  readonly decoder: OrtSessionLike;
  readonly joiner: OrtSessionLike;
  readonly tokenizer: XAsrTokenizer;
  readonly graph: LoadedGraph;
}

function urlFor(repo: string, revision: string, file: string, subfolder?: string): string {
  const parts = [repo, revision, subfolder, file].filter(Boolean).join('/');
  return `https://huggingface.co/${parts.split('/').map(encodeURIComponent).join('/')}`
    .replace(`/${encodeURIComponent(revision)}/`, `/resolve/${encodeURIComponent(revision)}/`);
}

function first(outputs: Record<string, OrtTensorLike>, preferred?: string): OrtTensorLike {
  if (preferred && outputs[preferred]) return outputs[preferred]!;
  const value = Object.values(outputs)[0];
  if (!value) throw new Error('X-ASR graph returned no outputs.');
  return value;
}

function readFloat(value: OrtTensorLike): Float32Array {
  if (value.type !== 'float16' && value.data instanceof Float32Array) return new Float32Array(value.data);
  const source = value.data as unknown as ArrayLike<number>;
  const out = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    const bits = Number(source[i] ?? 0);
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 31;
    const mantissa = bits & 1023;
    out[i] = exponent === 0 ? sign * (mantissa / 1024) * 2 ** -14 : exponent === 31 ? (mantissa ? NaN : sign * Infinity) : sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return out;
}

function cloneState(ort: OrtModuleLike, spec: XAsrStateTensorSpec): OrtTensorLike {
  const dims = spec.dims.map((value) => (value < 0 ? 1 : value));
  const count = Math.max(1, dims.reduce((left, right) => left * right, 1));
  const data = spec.type === 'float16'
    ? new Uint16Array(count)
    : spec.type === 'int64'
      ? new BigInt64Array(count)
      : spec.type === 'int32'
        ? new Int32Array(count)
        : new Float32Array(count);
  return new ort.Tensor(spec.type, data, dims);
}

function sessionEntries(
  session: OrtSessionLike,
  kind: 'input' | 'output',
): Array<{ readonly name: string; readonly type?: string; readonly dims?: readonly number[] }> {
  const record = session as unknown as {
    readonly inputNames?: readonly string[];
    readonly outputNames?: readonly string[];
    readonly inputMetadata?: Record<string, { readonly type?: string; readonly dimensions?: readonly number[] }> | Array<{ readonly name?: string; readonly type?: string; readonly dimensions?: readonly number[]; readonly shape?: readonly number[] }>;
    readonly outputMetadata?: Record<string, { readonly type?: string; readonly dimensions?: readonly number[] }> | Array<{ readonly name?: string; readonly type?: string; readonly dimensions?: readonly number[]; readonly shape?: readonly number[] }>;
  };
  const names = kind === 'input' ? record.inputNames : record.outputNames;
  const metadata = kind === 'input' ? record.inputMetadata : record.outputMetadata;
  if (Array.isArray(metadata)) {
    return metadata
      .filter((entry) => entry.name)
      .map((entry) => ({ name: entry.name!, type: entry.type, dims: entry.dimensions ?? entry.shape }));
  }
  if (metadata && typeof metadata === 'object') {
    return Object.entries(metadata).map(([name, entry]) => ({ name, type: entry?.type, dims: entry?.dimensions }));
  }
  return (names ?? []).map((name) => ({ name }));
}

function parseOrtType(ortType: string | undefined, fallback: XAsrStateTensorSpec['type']): XAsrStateTensorSpec['type'] {
  const raw = (ortType ?? '').replace(/^tensor\(/, '').replace(/\)$/, '');
  if (raw.includes('int64')) return 'int64';
  if (raw.includes('int32')) return 'int32';
  if (raw.includes('float16')) return 'float16';
  if (raw.includes('float')) return 'float32';
  return fallback;
}

function numericDims(dims: readonly number[] | undefined, batchSize = 1): number[] {
  if (!dims || dims.length === 0) return [batchSize];
  return dims.map((value) => (typeof value === 'number' && value > 0 ? value : batchSize));
}

function discoverGraph(encoder: OrtSessionLike, decoder: OrtSessionLike, joiner: OrtSessionLike, fallback: XAsrModelConfig['graph']): LoadedGraph {
  const encoderInputs = sessionEntries(encoder, 'input');
  const decoderInputs = sessionEntries(decoder, 'input');
  const joinerInputs = sessionEntries(joiner, 'input');
  const feature = encoderInputs.find((entry) => entry.name === (fallback.featureInputName ?? 'x')) ?? encoderInputs[0];
  const featureDims = numericDims(feature?.dims, 1);
  const encoderFrameSize = fallback.encoderFrameSize || featureDims[1] || 29;
  const encoderStateInputs: XAsrStateTensorSpec[] = encoderInputs
    .filter((entry) => entry.name && entry.name !== feature?.name)
    .map((entry) => ({
      name: entry.name,
      type: parseOrtType(entry.type, entry.name === 'processed_lens' ? 'int64' : 'float32'),
      dims: numericDims(entry.dims, 1),
    }));
  return {
    featureInputName: feature?.name ?? 'x',
    encoderOutputName: fallback.encoderOutputName ?? 'encoder_out',
    encoderFrameSize,
    encoderFrameShift: fallback.encoderFrameShift || 16,
    encoderStateInputs,
    decoderInputName: decoderInputs[0]?.name ?? fallback.decoderInputName ?? 'y',
    decoderOutputName: fallback.decoderOutputName ?? 'decoder_out',
    decoderContextSize: fallback.decoderContextSize || 2,
    decoderIndexType: parseOrtType(decoderInputs[0]?.type, 'int64') === 'int32' ? 'int32' : 'int64',
    joinerEncoderInputName: joinerInputs[0]?.name ?? fallback.joinerEncoderInputName ?? 'encoder_out',
    joinerDecoderInputName: joinerInputs[1]?.name ?? fallback.joinerDecoderInputName ?? 'decoder_out',
    joinerOutputName: fallback.joinerOutputName ?? 'logit',
  };
}

function decoderContextTensor(ort: OrtModuleLike, graph: LoadedGraph, tokenIds: readonly number[]): OrtTensorLike {
  const context = new Array<number>(graph.decoderContextSize).fill(0);
  const suffix = tokenIds.slice(-graph.decoderContextSize);
  context.splice(graph.decoderContextSize - suffix.length, suffix.length, ...suffix);
  if (graph.decoderIndexType === 'int32') {
    return new ort.Tensor('int32', Int32Array.from(context), [1, graph.decoderContextSize]);
  }
  return new ort.Tensor('int64', BigInt64Array.from(context, (value) => BigInt(value)), [1, graph.decoderContextSize]);
}

function throwIfDecodeAborted(signal: AbortSignalLike | null | undefined): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError('decode');
  }
}

export interface XAsrExecutor {
  ready(): Promise<void>;
  transcribe(audio: AudioBufferLike, options?: XAsrTranscriptionOptions): Promise<XAsrNativeTranscript>;
  createStream(): XAsrStreamState;
  pushStream(state: XAsrStreamState, audio: Float32Array, final?: boolean, options?: XAsrTranscriptionOptions): Promise<{ state: XAsrStreamState; transcript: XAsrNativeTranscript }>;
  disposeStream(state: XAsrStreamState): void;
  dispose(): void | Promise<void>;
}

export class OrtXAsrExecutor implements XAsrExecutor {
  private readonly source?: XAsrArtifactSource;
  private readonly provider?: AssetProvider;
  private readonly hooks?: SpeechRuntimeHooks;
  private readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  private readonly handles: ResolvedAssetHandle[] = [];
  private readonly activeStreams = new Set<XAsrStreamState>();
  private readonly frontend = new XAsrJsFrontend();
  private readonly state?: Promise<LoadedState>;
  private disposed = false;
  private disposePromise?: Promise<void>;
  /**
   * Speculative joiner batching calibration. The row size is learned from the
   * first single-frame joiner output; the flag is latched off permanently for
   * this executor when a batch run fails or returns non-row-parallel shapes.
   */
  private joinerRowSize?: number;
  private joinerBatchAllowed = true;

  constructor(private readonly modelId: string, private readonly backendId: string, private readonly config: XAsrModelConfig, options?: XAsrModelOptions, dependencies: { readonly assetProvider?: AssetProvider; readonly runtimeHooks?: SpeechRuntimeHooks; readonly signal?: import('../../types/index.js').AbortSignalLike | null } = {}) {
    this.source = options?.source; this.provider = dependencies.assetProvider; this.hooks = dependencies.runtimeHooks; this.signal = dependencies.signal;
    if (this.source) this.state = this.initialize();
  }

  private async resolve(source: Extract<XAsrArtifactSource, { kind: 'huggingface' }>, file: string): Promise<string> {
    const revision = source.revision ?? 'main';
    if (!this.provider) return urlFor(source.repoId, revision, file, source.subfolder);
    const handle = await this.provider.resolve({ id: `huggingface:${source.repoId}:${revision}:${file}`, provider: 'huggingface', repoId: source.repoId, revision, filename: source.subfolder ? `${source.subfolder}/${file}` : file, cacheKey: `huggingface:${source.repoId}:${revision}:${file}`, onProgress: (event) => this.hooks?.onProgress?.({ phase: 'asset:download', modelId: this.modelId, file, loaded: event.loaded, total: event.total, percent: event.total ? Math.round(event.loaded / event.total * 100) : event.done && !event.aborted ? 100 : undefined, isComplete: Boolean(event.done) && !event.aborted, aborted: event.aborted, message: event.aborted ? `Cancelled ${file}.` : event.done ? `Prepared ${file}.` : `Downloading ${file}.` }) });
    this.handles.push(handle); const locator = await handle.getLocator('url');
    if (!locator) throw new Error(`Could not create a URL locator for "${file}".`);
    return locator;
  }

  private async initialize(): Promise<LoadedState> {
    if (!this.source) throw createExperimentalArtifactMissingError('x-asr', this.modelId);
    let encoderUrl: string; let decoderUrl: string; let joinerUrl: string; let tokenizerUrl: string;
    let encoderDataUrl: string | undefined; let decoderDataUrl: string | undefined; let joinerDataUrl: string | undefined;
    let encoderDataFilename: string | undefined; let decoderDataFilename: string | undefined; let joinerDataFilename: string | undefined;
    if (this.source.kind === 'direct') {
      ({ encoderUrl, decoderUrl, joinerUrl, tokenizerUrl, encoderDataUrl, decoderDataUrl, joinerDataUrl, encoderDataFilename, decoderDataFilename, joinerDataFilename } = this.source.artifacts);
    } else {
      encoderUrl = await this.resolve(this.source, this.source.encoderFilename ?? 'encoder-160ms.onnx');
      decoderUrl = await this.resolve(this.source, this.source.decoderFilename ?? 'decoder-160ms.onnx');
      joinerUrl = await this.resolve(this.source, this.source.joinerFilename ?? 'joiner-160ms.onnx');
      tokenizerUrl = await this.resolve(this.source, this.source.tokenizerFilename ?? 'tokens.txt');
      encoderDataFilename = this.source.encoderDataFilename; decoderDataFilename = this.source.decoderDataFilename; joinerDataFilename = this.source.joinerDataFilename;
      if (encoderDataFilename) encoderDataUrl = await this.resolve(this.source, encoderDataFilename);
      if (decoderDataFilename) decoderDataUrl = await this.resolve(this.source, decoderDataFilename);
      if (joinerDataFilename) joinerDataUrl = await this.resolve(this.source, joinerDataFilename);
    }
    const ort = await initOrt(this.backendId, { cpuThreads: this.source.cpuThreads, wasmPaths: this.source.wasmPaths, signal: this.signal });
    const backend = this.backendId.startsWith('webgpu') ? 'webgpu' : 'wasm';
    const make = (
      url: string,
      dataUrl?: string,
      dataPath?: string,
      preferredOutputLocation?: Readonly<Record<string, OrtOutputLocation>>,
    ) => createOrtSession(ort, url, { backendId: backend, enableProfiling: this.source?.enableProfiling, externalDataUrl: dataUrl, externalDataPath: dataPath, preferredOutputLocation, signal: this.signal });
    const tokenizerPromise = XAsrTokenizer.fromUrl(tokenizerUrl, this.signal);
    const encoderStateOutputs = this.config.graph.encoderStateOutputs ?? [];
    const encoderOutputLocations = backend === 'webgpu' && encoderStateOutputs.length > 0
      ? {
          [this.config.graph.encoderOutputName ?? 'encoder_out']: 'cpu' as const,
          ...Object.fromEntries(encoderStateOutputs.map((name) => [name, 'gpu-buffer' as const])),
        }
      : undefined;
    const encoder = await make(
      encoderUrl,
      encoderDataUrl,
      encoderDataFilename,
      encoderOutputLocations,
    );
    if (this.disposed) {
      releaseOrtSession(encoder);
      throw new Error(`X-ASR executor was disposed during load for "${this.modelId}".`);
    }
    const decoder = await make(decoderUrl, decoderDataUrl, decoderDataFilename);
    if (this.disposed) {
      releaseOrtSession(encoder);
      releaseOrtSession(decoder);
      throw new Error(`X-ASR executor was disposed during load for "${this.modelId}".`);
    }
    const joiner = await make(joinerUrl, joinerDataUrl, joinerDataFilename);
    if (this.disposed) {
      releaseOrtSession(encoder);
      releaseOrtSession(decoder);
      releaseOrtSession(joiner);
      throw new Error(`X-ASR executor was disposed during load for "${this.modelId}".`);
    }
    const tokenizer = await tokenizerPromise;
    const graph = discoverGraph(encoder, decoder, joiner, this.config.graph);
    return { ort, encoder, decoder, joiner, tokenizer, graph };
  }

  async ready(): Promise<void> { if (!this.state) throw createExperimentalArtifactMissingError('x-asr', this.modelId); await this.state; }

  createStream(): XAsrStreamState {
    if (this.disposed) throw new Error(`X-ASR executor is disposed for "${this.modelId}".`);
    if (!this.source) throw createExperimentalArtifactMissingError('x-asr', this.modelId);
    const state = { audio: new Float32Array(0), features: new Float32Array(0), frontendTail: new Float32Array(0), encodedFrames: 0, inputFrames: 0, tokenIds: [], encoderStates: [] as OrtTensorLike[] };
    this.activeStreams.add(state);
    return state;
  }

  /**
   * Streaming encoder/decoder/joiner loop.
   * Abort does not commit the in-flight chunk: caller encoder-state tensors stay
   * intact so the same stream can be retried. `reset()` is optional, not required.
   */
  private async decodeFeatures(
    state: XAsrStreamState,
    features: Float32Array,
    final: boolean,
    signal?: AbortSignalLike | null,
  ): Promise<XAsrStreamState> {
    const loaded = await this.state!;
    const graph = loaded.graph;
    const featureWidth = this.config.featureDim;
    let pending = features;
    const originalEncoderStates = new Set(state.encoderStates);
    let nextState = [...state.encoderStates];
    if (nextState.length === 0) {
      nextState = graph.encoderStateInputs.map((spec) => cloneState(loaded.ort, spec));
    }
    let encodedFrames = state.encodedFrames;
    const tokens = [...state.tokenIds];
    const needed = graph.encoderFrameSize * featureWidth;
    const advance = graph.encoderFrameShift * featureWidth;
    let liveEncoded: OrtTensorLike | undefined;
    let liveEncoderOutput: Record<string, OrtTensorLike> | undefined;
    let liveDecoderOutput: Record<string, OrtTensorLike> | undefined;
    let liveDecoderValue: OrtTensorLike | undefined;

    const disposeLiveDecoder = (): void => {
      liveDecoderValue?.dispose?.();
      if (liveDecoderOutput) {
        for (const value of Object.values(liveDecoderOutput)) {
          if (value !== liveDecoderValue) value.dispose?.();
        }
      }
      liveDecoderOutput = undefined;
      liveDecoderValue = undefined;
    };

    const disposeLiveEncoder = (): void => {
      liveEncoded?.dispose?.();
      if (liveEncoderOutput) {
        for (const value of Object.values(liveEncoderOutput)) {
          if (value !== liveEncoded && !nextState.includes(value)) value.dispose?.();
        }
      }
      liveEncoded = undefined;
      liveEncoderOutput = undefined;
    };

    try {
      throwIfDecodeAborted(signal);
      while (pending.length >= needed || (final && pending.length >= advance)) {
        throwIfDecodeAborted(signal);
        const chunk = new Float32Array(needed);
        chunk.set(pending.subarray(0, Math.min(needed, pending.length)));
        pending = pending.length >= advance ? pending.subarray(advance) : new Float32Array(0);
        const featureTensor = new loaded.ort.Tensor('float32', chunk, [1, graph.encoderFrameSize, featureWidth]);
        const feeds: Record<string, unknown> = { [graph.featureInputName]: featureTensor };
        graph.encoderStateInputs.forEach((spec, index) => {
          feeds[spec.name] = nextState[index] ?? cloneState(loaded.ort, spec);
        });
        let output: Record<string, OrtTensorLike>;
        try {
          output = await loaded.encoder.run(feeds);
        } finally {
          featureTensor.dispose?.();
        }
        liveEncoderOutput = output;
        liveEncoded = first(output, graph.encoderOutputName);
        const encodedData = readFloat(liveEncoded);
        const dims = [...liveEncoded.dims];
        const frames = dims.length === 3 ? (dims[1] ?? 1) : 1;
        const hidden = dims.length === 3 ? (dims[2] ?? encodedData.length / Math.max(frames, 1)) : (dims[1] ?? encodedData.length);
        const previousState = nextState;
        nextState = graph.encoderStateInputs.map((spec) => output[`new_${spec.name}`] ?? output[spec.name] ?? cloneState(loaded.ort, spec));
        previousState.forEach((value) => {
          if (!nextState.includes(value) && !originalEncoderStates.has(value)) value.dispose?.();
        });

        let decoderTensor = decoderContextTensor(loaded.ort, graph, tokens);
        liveDecoderOutput = await loaded.decoder.run({ [graph.decoderInputName]: decoderTensor });
        decoderTensor.dispose?.();
        liveDecoderValue = first(liveDecoderOutput, graph.decoderOutputName);
        let decoderData = readFloat(liveDecoderValue);

        // Greedy RNNT decode over the frames produced by this chunk. The joiner
        // graph is row-parallel (leading dimension is the frame batch), so all
        // remaining frames are scored speculatively against the current
        // decoder state in one run. Blank rows never change the decoder state,
        // so the first non-blank row is exactly what the per-frame sequential
        // loop would emit; after an emission the decoder advances and the
        // suffix of frames is re-batched. This is token-for-token equivalent
        // to the sequential loop while replacing up to `frames` tiny
        // dispatches per chunk with one run per emitted token. Graphs that
        // reject batch shapes fall back to the sequential path permanently.
        let frame = 0;
        while (frame < frames) {
          throwIfDecodeAborted(signal);
          const remaining = frames - frame;
          let consumedByBatch = 0;
          if (this.joinerBatchAllowed && this.joinerRowSize !== undefined && dims.length === 3 && remaining > 1) {
            const rowSize = this.joinerRowSize;
            const encBatch = encodedData.subarray(frame * hidden, frame * hidden + remaining * hidden);
            const decBatch = new Float32Array(remaining * decoderData.length);
            for (let row = 0; row < remaining; row += 1) decBatch.set(decoderData, row * decoderData.length);
            const encTensor = new loaded.ort.Tensor('float32', encBatch, [remaining, hidden]);
            const decTensor = new loaded.ort.Tensor('float32', decBatch, [remaining, decoderData.length]);
            let batchValues: Float32Array | undefined;
            let batchRowParallel = false;
            try {
              const batchOutput = await loaded.joiner.run({
                [graph.joinerEncoderInputName]: encTensor,
                [graph.joinerDecoderInputName]: decTensor,
              });
              const batchLogits = first(batchOutput, graph.joinerOutputName);
              batchValues = readFloat(batchLogits);
              const batchDims = batchLogits.dims;
              const rowParallel = batchDims.length === 2
                && batchDims[0] === remaining
                && batchDims[1] === rowSize
                && batchValues.length >= remaining * rowSize;
              if (!rowParallel) this.joinerBatchAllowed = false;
              batchRowParallel = rowParallel;
              batchLogits.dispose?.();
              Object.values(batchOutput).forEach((value) => { if (value !== batchLogits) value.dispose?.(); });
            } catch (error) {
              if (error instanceof PipelineAbortedError) throw error;
              this.joinerBatchAllowed = false;
              batchValues = undefined;
            } finally {
              encTensor.dispose?.();
              decTensor.dispose?.();
            }
            if (batchValues && batchRowParallel) {
              // A rejected batch run leaves batchValues undefined, and a
              // non-row-parallel output latches batching off above; in both
              // cases the identical sequential path below takes over.
              let emission: { row: number; id: number } | undefined;
              for (let row = 0; row < remaining; row += 1) {
                // The shared argmax returns an absolute index; the row-local
                // token id is the offset within the flattened logits block.
                const id = argmax(batchValues, row * rowSize, rowSize) - row * rowSize;
                if (id !== 0) {
                  emission = { row, id };
                  break;
                }
              }
              if (emission) {
                disposeLiveDecoder();
                tokens.push(emission.id);
                decoderTensor = decoderContextTensor(loaded.ort, graph, tokens);
                liveDecoderOutput = await loaded.decoder.run({ [graph.decoderInputName]: decoderTensor });
                decoderTensor.dispose?.();
                liveDecoderValue = first(liveDecoderOutput, graph.decoderOutputName);
                decoderData = readFloat(liveDecoderValue);
                consumedByBatch = emission.row + 1;
              } else {
                consumedByBatch = remaining;
              }
            }
          }
          if (consumedByBatch > 0) {
            frame += consumedByBatch;
            continue;
          }
          const enc = encodedData.subarray(frame * hidden, frame * hidden + hidden);
          const encTensor = new loaded.ort.Tensor('float32', enc, dims.length === 3 ? [1, hidden] : [...dims]);
          const decTensor = new loaded.ort.Tensor('float32', decoderData, liveDecoderValue.dims);
          let joinerOutput: Record<string, OrtTensorLike>;
          try {
            joinerOutput = await loaded.joiner.run({
              [graph.joinerEncoderInputName]: encTensor,
              [graph.joinerDecoderInputName]: decTensor,
            });
          } finally {
            encTensor.dispose?.();
            decTensor.dispose?.();
          }
          const logits = first(joinerOutput, graph.joinerOutputName);
          const values = readFloat(logits);
          if (this.joinerRowSize === undefined && logits.dims.length === 2 && logits.dims[0] === 1) {
            this.joinerRowSize = logits.dims[1];
          }
          const id = argmax(values, 0, values.length);
          logits.dispose?.();
          Object.values(joinerOutput).forEach((value) => { if (value !== logits) value.dispose?.(); });
          if (id !== 0) {
            tokens.push(id);
            disposeLiveDecoder();
            decoderTensor = decoderContextTensor(loaded.ort, graph, tokens);
            liveDecoderOutput = await loaded.decoder.run({ [graph.decoderInputName]: decoderTensor });
            decoderTensor.dispose?.();
            liveDecoderValue = first(liveDecoderOutput, graph.decoderOutputName);
            decoderData = readFloat(liveDecoderValue);
          }
          frame += 1;
        }
        disposeLiveDecoder();
        disposeLiveEncoder();
        encodedFrames += frames;
        if (final && pending.length > 0 && pending.length < needed && pending.length < advance) {
          break;
        }
      }
      return { audio: state.audio, features: pending, encodedFrames, inputFrames: state.inputFrames, tokenIds: tokens, encoderStates: nextState };
    } catch (error) {
      disposeLiveDecoder();
      disposeLiveEncoder();
      for (const tensor of nextState) {
        if (!originalEncoderStates.has(tensor)) tensor.dispose?.();
      }
      throw error;
    }
  }

  async pushStream(state: XAsrStreamState, audio: Float32Array, final = false, options: XAsrTranscriptionOptions = {}): Promise<{ state: XAsrStreamState; transcript: XAsrNativeTranscript }> {
    const started = nowMs();
    const appendedAudio = appendStreamAudio(state, audio);
    const incremental = this.frontend.processIncremental(
      state.frontendTail ?? state.audio.subarray(Math.max(0, state.audio.length - 400)),
      state.audio.length,
      audio,
      state.inputFrames,
      final,
    );
    const newFeatures = incremental.features;
    const combined = new Float32Array(state.features.length + newFeatures.length);
    combined.set(state.features);
    combined.set(newFeatures, state.features.length);
    const next = await this.decodeFeatures({ ...state, audio: appendedAudio.audio, audioBuffer: appendedAudio.audioBuffer }, combined, final, options.signal);
    const ids = [...next.tokenIds];
    const tokenizer = (await this.state!).tokenizer;
    const transcript: XAsrNativeTranscript = {
      utteranceText: tokenizer.decode(ids),
      isFinal: final,
      tokens: options.returnTokenIds ? ids.map((id, index) => ({ index, id, text: tokenizer.decodeTokenPiece(id) })) : undefined,
      metrics: {
        preprocessMs: 0,
        encodeMs: roundMetric(nowMs() - started),
        decodeMs: 0,
        totalMs: roundMetric(nowMs() - started),
        wallMs: roundMetric(nowMs() - started),
        audioDurationSec: roundMetric(appendedAudio.audio.length / this.config.sampleRate, 4),
        encoderFrameCount: next.encodedFrames,
        emittedTokenCount: ids.length,
        preprocessorBackend: 'js',
      },
      warnings: [],
    };
    const nextState = { ...next, audio: appendedAudio.audio, audioBuffer: appendedAudio.audioBuffer, features: next.features, frontendTail: incremental.tail, inputFrames: state.inputFrames + incremental.frameCount };
    this.activeStreams.delete(state);
    disposeStreamTensors(state, nextState);
    this.activeStreams.add(nextState);
    return { state: nextState, transcript };
  }

  async transcribe(audio: AudioBufferLike, options: XAsrTranscriptionOptions = {}): Promise<XAsrNativeTranscript> {
    const state = this.createStream();
    const pcm = audio.channels?.[0] ?? (audio.data instanceof Float32Array ? audio.data : Float32Array.from(audio.data ?? []));
    try {
      const result = await this.pushStream(state, pcm, true, options);
      this.disposeStream(result.state);
      return result.transcript;
    } catch (error) {
      this.disposeStream(state);
      throw error;
    }
  }
  disposeStream(state: XAsrStreamState): void { this.activeStreams.delete(state); disposeStreamTensors(state); }
  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.disposePromise = this.flushDispose();
    return this.disposePromise;
  }

  private async flushDispose(): Promise<void> {
    for (const stream of this.activeStreams) disposeStreamTensors(stream);
    this.activeStreams.clear();
    if (this.state) {
      try {
        const loaded = await this.state;
        releaseOrtSession(loaded.encoder);
        releaseOrtSession(loaded.decoder);
        releaseOrtSession(loaded.joiner);
      } catch {
        // Keep the original load error; still drop asset handles.
      }
    }
    const handles = this.handles.splice(0);
    await Promise.all(handles.map((handle) => Promise.resolve(handle.dispose())));
  }
}
