import { normalizePcmInput } from '../../audio/index.js';
import { confidenceFromLogits, argmax } from '../../inference/index.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import type { AssetProvider, AudioBufferLike, ResolvedAssetHandle, SpeechRuntimeHooks, TranscriptWarning } from '../../types/index.js';
import { createOrtSession, initOrt, type OrtModuleLike, type OrtSessionLike, type OrtTensorLike } from '../lasr-ctc/ort.js';
import { GigaAmJsPreprocessor } from './frontend.js';
import { GigaAmRnntTokenizer } from './tokenizer.js';
import type { GigaAmRnntArtifactSource, GigaAmRnntModelConfig, GigaAmRnntModelOptions, GigaAmRnntNativeTranscript, GigaAmRnntTranscriptionOptions } from './types.js';

interface LoadedState {
  readonly ort: OrtModuleLike;
  readonly encoder: OrtSessionLike;
  readonly decoder: OrtSessionLike;
  readonly joint: OrtSessionLike;
  readonly tokenizer: GigaAmRnntTokenizer;
  readonly warnings: readonly TranscriptWarning[];
}

function tensor(ort: OrtModuleLike, type: 'float32' | 'int64' | 'int32', data: ArrayBufferView, dims: readonly number[]): OrtTensorLike {
  return new ort.Tensor(type, data, dims);
}

function firstOutput(outputs: Record<string, OrtTensorLike>, ...names: string[]): OrtTensorLike {
  for (const name of names) if (outputs[name]) return outputs[name]!;
  const output = Object.values(outputs)[0];
  if (!output) throw new Error('GigaAM RNN-T graph returned no output.');
  return output;
}

function tensorData(value: OrtTensorLike): Float32Array {
  if (value.type !== 'float16') return value.data instanceof Float32Array ? value.data : Float32Array.from(value.data as unknown as ArrayLike<number>);
  const source = value.data as unknown as ArrayLike<number>;
  const result = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    const bits = Number(source[index] ?? 0); const sign = bits & 0x8000 ? -1 : 1; const exponent = (bits >>> 10) & 31; const mantissa = bits & 1023;
    result[index] = exponent === 0 ? sign * (mantissa / 1024) * 2 ** -14 : exponent === 31 ? (mantissa ? NaN : sign * Infinity) : sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return result;
}

function hfUrl(repoId: string, revision: string, filename: string): string {
  return `https://huggingface.co/${repoId.split('/').map(encodeURIComponent).join('/')}/resolve/${encodeURIComponent(revision)}/${filename.split('/').map(encodeURIComponent).join('/')}`;
}

export class OrtGigaAmRnntExecutor {
  private readonly preprocessor = new GigaAmJsPreprocessor();
  private readonly handles: ResolvedAssetHandle[] = [];
  private readonly source?: GigaAmRnntArtifactSource;
  private readonly provider?: AssetProvider;
  private readonly hooks?: SpeechRuntimeHooks;
  private readonly state?: Promise<LoadedState>;

  constructor(private readonly modelId: string, private readonly backendId: string, private readonly config: GigaAmRnntModelConfig, options?: GigaAmRnntModelOptions, dependencies: { readonly assetProvider?: AssetProvider; readonly runtimeHooks?: SpeechRuntimeHooks } = {}) {
    this.source = options?.source; this.provider = dependencies.assetProvider; this.hooks = dependencies.runtimeHooks;
    if (this.source) this.state = this.initialize();
  }

  private async resolve(source: Extract<GigaAmRnntArtifactSource, { kind: 'huggingface' }>, filename: string): Promise<string> {
    if (!this.provider) return hfUrl(source.repoId, source.revision ?? 'main', filename);
    const handle = await this.provider.resolve({ id: `huggingface:${source.repoId}:${source.revision ?? 'main'}:${filename}`, provider: 'huggingface', repoId: source.repoId, revision: source.revision ?? 'main', filename, cacheKey: `huggingface:${source.repoId}:${source.revision ?? 'main'}:${filename}`, onProgress: (event) => this.hooks?.onProgress?.({ phase: 'asset:download', modelId: this.modelId, file: filename, loaded: event.loaded, total: event.total, percent: event.total ? Math.round(event.loaded / event.total * 100) : event.done ? 100 : undefined, isComplete: event.done, message: event.done ? `Prepared ${filename}.` : `Downloading ${filename}.` }) });
    this.handles.push(handle); const url = await handle.getLocator('url'); if (!url) throw new Error(`Could not create a URL locator for "${filename}".`); return url;
  }

  private async initialize(): Promise<LoadedState> {
    if (!this.source) throw new Error(`No GigaAM RNN-T artifact source is configured for "${this.modelId}".`);
    let encoderUrl: string; let decoderUrl: string; let jointUrl: string; let tokenizerUrl: string;
    let encoderDataUrl: string | undefined; let decoderDataUrl: string | undefined; let jointDataUrl: string | undefined;
    let encoderDataFilename: string | undefined; let decoderDataFilename: string | undefined; let jointDataFilename: string | undefined;
    if (this.source.kind === 'direct') {
      ({ encoderUrl, decoderUrl, jointUrl, tokenizerUrl, encoderDataUrl, decoderDataUrl, jointDataUrl, encoderDataFilename, decoderDataFilename, jointDataFilename } = this.source.artifacts);
    } else {
      const revision = this.source.revision ?? 'main';
      const ef = this.source.encoderFilename ?? 'v3_e2e_rnnt_encoder.onnx'; const df = this.source.decoderFilename ?? 'v3_e2e_rnnt_decoder.onnx'; const jf = this.source.jointFilename ?? 'v3_e2e_rnnt_joint.onnx'; const tf = this.source.tokenizerFilename ?? 'v3_e2e_rnnt_vocab.txt';
      encoderUrl = await this.resolve(this.source, ef); decoderUrl = await this.resolve(this.source, df); jointUrl = await this.resolve(this.source, jf); tokenizerUrl = await this.resolve(this.source, tf);
      encoderDataFilename = this.source.encoderDataFilename; decoderDataFilename = this.source.decoderDataFilename; jointDataFilename = this.source.jointDataFilename;
      if (encoderDataFilename) encoderDataUrl = await this.resolve(this.source, encoderDataFilename); if (decoderDataFilename) decoderDataUrl = await this.resolve(this.source, decoderDataFilename); if (jointDataFilename) jointDataUrl = await this.resolve(this.source, jointDataFilename);
      void revision;
    }
    const ort = await initOrt(this.backendId, { cpuThreads: this.source.cpuThreads });
    const make = (url: string, dataUrl?: string, dataPath?: string) => createOrtSession(ort, url, { backendId: this.backendId.startsWith('webgpu') ? 'webgpu' : 'wasm', externalDataUrl: dataUrl, externalDataPath: dataPath });
    const [encoder, decoder, joint] = await Promise.all([make(encoderUrl, encoderDataUrl, encoderDataFilename), make(decoderUrl, decoderDataUrl, decoderDataFilename), make(jointUrl, jointDataUrl, jointDataFilename)]);
    return { ort, encoder, decoder, joint, tokenizer: await GigaAmRnntTokenizer.fromUrl(tokenizerUrl), warnings: [] };
  }

  async ready(): Promise<void> { if (!this.state) throw new Error(`No GigaAM RNN-T artifact source is configured for "${this.modelId}".`); await this.state; }

  async transcribe(input: AudioBufferLike, options: GigaAmRnntTranscriptionOptions = {}): Promise<GigaAmRnntNativeTranscript> {
    const state = await this.state; if (!state) throw new Error(`No GigaAM RNN-T artifact source is configured for "${this.modelId}".`);
    const audio = normalizePcmInput(input).toMono(); const started = nowMs(); const prepared = this.preprocessor.process(audio.channels[0] ?? new Float32Array(0));
    if (prepared.frameCount <= 0) return { utteranceText: '', isFinal: true, warnings: [...state.warnings] };
    const featureTensor = tensor(state.ort, 'float32', prepared.features, [1, this.config.nMels, prepared.frameCount]); const lengthTensor = tensor(state.ort, 'int64', BigInt64Array.from([BigInt(prepared.frameCount)]), [1]);
    let encoderOutputs: Record<string, OrtTensorLike>; try { encoderOutputs = await state.encoder.run({ audio_signal: featureTensor, length: lengthTensor }); } finally { featureTensor.dispose?.(); lengthTensor.dispose?.(); }
    const encoded = firstOutput(encoderOutputs, 'encoded'); const encodedLength = firstOutput(encoderOutputs, 'encoded_len'); const encDims = [...encoded.dims];
    if (encDims.length !== 3 || encDims[0] !== 1) throw new Error(`Unexpected GigaAM RNN-T encoder shape: [${encDims.join(', ')}].`);
    const hidden = encDims[1] ?? 0; const frames = Number((encodedLength.data as unknown as ArrayLike<number>)[0] ?? encDims[2] ?? 0); const encData = tensorData(encoded); const blankId = state.tokenizer.blankId; const ids: number[] = []; const tokens: { index: number; id?: number; text: string; startTime: number; endTime: number; confidence: number; logitIndex?: number }[] = []; const h = new Float32Array(this.config.predictionHiddenSize); const c = new Float32Array(this.config.predictionHiddenSize); let decoderOutput: OrtTensorLike | undefined; let decodeIterations = 0;
    for (let frame = 0; frame < frames; frame += 1) {
      let emitted = 0;
      while (emitted < this.config.maxTokensPerFrame) {
        const target = tensor(state.ort, 'int32', new Int32Array([ids.at(-1) ?? blankId]), [1, 1]); const hTensor = tensor(state.ort, 'float32', h, [1, 1, h.length]); const cTensor = tensor(state.ort, 'float32', c, [1, 1, c.length]); const decoderInputs: Record<string, unknown> = { x: target, 'h.1': hTensor, 'c.1': cTensor }; let decoderOutputs: Record<string, OrtTensorLike>;
        try { decoderOutputs = await state.decoder.run(decoderInputs); } finally { target.dispose?.(); hTensor.dispose?.(); cTensor.dispose?.(); }
        decoderOutput = firstOutput(decoderOutputs, 'dec'); const nextHOutput = firstOutput(decoderOutputs, 'h'); const nextCOutput = firstOutput(decoderOutputs, 'c'); const nextH = tensorData(nextHOutput); const nextC = tensorData(nextCOutput); h.set(nextH.subarray(0, h.length)); c.set(nextC.subarray(0, c.length));
        nextHOutput.dispose?.(); nextCOutput.dispose?.();
        const encFrame = new Float32Array(hidden); for (let i = 0; i < hidden; i += 1) encFrame[i] = encData[i * (encDims[2] ?? frames) + frame] ?? 0;
        const encTensor = tensor(state.ort, 'float32', encFrame, [1, hidden, 1]); const decData = tensorData(decoderOutput); const decTensor = tensor(state.ort, 'float32', decData, [1, decData.length, 1]); decoderOutput.dispose?.(); let jointOutputs: Record<string, OrtTensorLike>; try { jointOutputs = await state.joint.run({ enc: encTensor, dec: decTensor }); } finally { encTensor.dispose?.(); decTensor.dispose?.(); }
        const logitsTensor = firstOutput(jointOutputs, 'joint'); const logits = tensorData(logitsTensor); const vocab = logits.length; const tokenId = argmax(logits, 0, vocab); const confidence = confidenceFromLogits(logits, tokenId, vocab); decodeIterations += 1; logitsTensor.dispose?.();
        if (tokenId === blankId) break; ids.push(tokenId); emitted += 1; const startTime = roundTimestampSeconds(frame * this.config.featureHopSeconds * this.config.rawStride); tokens.push({ index: tokens.length, id: options.returnTokenIds ? tokenId : undefined, text: state.tokenizer.decodeTokenPiece(tokenId), startTime, endTime: roundTimestampSeconds((frame + 1) * this.config.featureHopSeconds * this.config.rawStride), confidence: roundMetric(confidence.confidence, 4), logitIndex: options.returnLogitIndices ? frame : undefined });
      }
    }
    encoded.dispose?.(); encodedLength.dispose?.();
    const totalMs = nowMs() - started; return { utteranceText: state.tokenizer.decode(ids), isFinal: true, tokens, confidence: { tokenAverage: tokens.length ? tokens.reduce((sum, item) => sum + item.confidence, 0) / tokens.length : 0 }, metrics: { preprocessMs: 0, encodeMs: roundMetric(totalMs), decodeMs: 0, totalMs: roundMetric(totalMs), wallMs: roundMetric(totalMs), audioDurationSec: roundMetric(audio.durationSeconds, 4), rtf: audio.durationSeconds ? roundMetric(totalMs / (audio.durationSeconds * 1000), 4) : 0, rtfx: audio.durationSeconds ? roundMetric(audio.durationSeconds / (totalMs / 1000), 4) : undefined, preprocessorBackend: 'js', encoderFrameCount: frames, decodeIterations, emittedTokenCount: tokens.length }, warnings: [...state.warnings] };
  }

  dispose(): void { for (const handle of this.handles) void handle.dispose(); }
}
