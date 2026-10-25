/**
 * Streaming Moonshine session over the validated chunked graph set.
 *
 * Graph contracts (export receipts under
 * tools/data/results/model-candidates/moonshine-streaming-*-2026-10-*.json,
 * contract write-up in tools/model-debugging/reference/moonshine-streaming/README.md):
 *
 * - `encoder_chunked_model.onnx` is a pure function: a window of `emit + 60`
 *   raw audio frames plus explicit state returns hidden states valid for the
 *   first `emit / 4` embedded frames plus the same state advanced to the
 *   emission cut. The trailing 60 audio frames are the six attention layers'
 *   compounded right context; the next call re-feeds them, so the window
 *   advances by `emit` and the conv tails roll back to the emission cut.
 * - `encoder_model.onnx` (whole utterance) serves as the final flush on an
 *   audio tail: positions are relative, so a tail run equals the stream once
 *   its own left context (90 embedded frames) is inside the tail.
 * - The decoder pair (prefill, step) is stateless across previews: cross
 *   attention is computed in-graph from the encoder states, so a preview
 *   simply re-decodes the hidden frames accumulated so far.
 *
 * All state lives in explicit tensors or plain counters; nothing hides in
 * closures. Every geometry constant comes from the options so a different
 * checkpoint can be bound without touching this file.
 */

import type { MoonshineTokenizer } from './session.js';

/** ORT tensor factory subset the session needs (same shape as the offline session). */
export interface MoonshineStreamingOrt {
  readonly Tensor: new (type: string, data: ArrayBufferView, dims: readonly number[]) => unknown;
}

interface GraphRunner {
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
}

export interface MoonshineStreamingSessionOptions {
  readonly ort: MoonshineStreamingOrt;
  /** Pure-function chunked encoder (window + state in, hiddens + state out). */
  readonly chunkedEncoder: GraphRunner;
  /** Whole-utterance encoder, used only for the final flush. */
  readonly wholeEncoder: GraphRunner;
  /** Decoder seed: one BOS token over the accumulated hidden states. */
  readonly prefill: GraphRunner;
  /** Decoder step: one token plus per-layer past key/value histories. */
  readonly step: GraphRunner;
  readonly tokenizer: MoonshineTokenizer;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  /** Embedded-frame left context carried by the chunked graph (90). */
  readonly contextFrames?: number;
  /** Audio-frame right context the chunked window re-feeds (60). */
  readonly lookaheadFrames?: number;
  /** Embedded frames committed per chunked call (audio window = emit + lookahead). */
  readonly emitFrames?: number;
  /** Audio frames of already-committed audio retained for the final-flush tail. */
  readonly tailContextFrames?: number;
  readonly decoderLayers?: number;
  readonly maxTokens?: number;
  /** Activation dtype of the graph set: 'float16' for the fp16 graphs. */
  readonly dtype?: 'float32' | 'float16';
}

export interface MoonshineStreamingText {
  readonly text: string;
  readonly tokenIds: readonly number[];
  /** Embedded encoder frames the text was decoded from. */
  readonly encoderFrames: number;
}

interface TensorLike {
  readonly dims: readonly number[];
  readonly data: Float32Array;
}

interface ChunkedState {
  embeddedContext: TensorLike;
  contextLen: number;
  conv1Tail: TensorLike;
  conv2Tail: TensorLike;
  frames1: number;
  frames2: number;
}

/** Embedded frames the two stride-2 convs produce for `audioFrames` inputs. */
export function embeddedFrameCount(audioFrames: number): number {
  if (audioFrames < 1) {
    return 0;
  }
  return Math.ceil(Math.ceil(audioFrames / 2) / 2);
}

/** IEEE-754 binary32 to binary16, round-to-nearest-even (ORT JS wants half bits). */
function float32ToFloat16(input: Float32Array): Uint16Array {
  const out = new Uint16Array(input.length);
  const view = new DataView(new ArrayBuffer(4));
  for (let index = 0; index < input.length; index += 1) {
    view.setFloat32(0, input[index]!, true);
    const bits = view.getUint32(0, true);
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    const mantissa = bits & 0x7fffff;
    if (exponent === 0xff) {
      out[index] = sign | 0x7c00 | (mantissa ? 0x200 : 0);
    } else if (exponent > 112) {
      out[index] = sign | ((exponent - 112) << 10) | (mantissa >> 13);
    } else if (exponent > 102) {
      const shift = 126 - exponent;
      out[index] = sign | ((mantissa | 0x800000) >> (shift + 13));
    } else {
      out[index] = sign;
    }
  }
  return out;
}

const FRAME_SAMPLES = 80;
const CONTEXT_FRAMES = 90;
const LOOKAHEAD_FRAMES = 60;
const EMIT_FRAMES = 256;
const TAIL_CONTEXT_FRAMES = 480;

export class MoonshineStreamingSession {
  private readonly options: MoonshineStreamingSessionOptions;
  private readonly pending: number[] = [];
  private readonly retained: number[] = [];
  private readonly hiddens: number[] = [];
  private hiddenFrames = 0;
  private committedFrames = 0;
  private state: ChunkedState | null = null;
  private disposed = false;

  constructor(options: MoonshineStreamingSessionOptions) {
    this.options = options;
  }

  /** Embedded frames committed so far (excludes the buffered lookahead). */
  get committedEncoderFrames(): number {
    return this.committedFrames;
  }

  /** Embedded frames held for decoding; equals committedEncoderFrames. */
  get encoderFrames(): number {
    return this.hiddenFrames;
  }

  /**
   * Buffers raw 16 kHz samples. Runs zero or more chunked-encoder calls; each
   * completed call commits exactly `emitFrames / 4` embedded frames whose
   * hidden states are final.
   */
  async pushAudio(pcm: Float32Array): Promise<number> {
    this.assertUsable();
    // Spread-based pushes overflow the call stack on long inputs.
    for (let index = 0; index < pcm.length; index += 4096) {
      this.pending.push(...pcm.subarray(index, index + 4096));
    }
    const emit = this.emitFrames();
    const windowSamples = (emit + this.lookaheadFrames()) * FRAME_SAMPLES;
    while (this.pending.length >= windowSamples) {
      const window = this.pending.slice(0, windowSamples);
      const state = this.state ?? this.initialState();
      const out = await this.options.chunkedEncoder.run(this.chunkedFeeds(window, state));
      this.state = this.nextState(state, out);
      const hidden = MoonshineStreamingSession.named(out, 'hidden');
      // The graph emits the whole window; only the first emit/4 frames carry
      // final right context. The remainder belongs to the re-fed lookahead.
      const keep = emit / 4;
      this.hiddens.push(...Array.from(hidden.data.slice(0, keep * 320)));
      this.hiddenFrames += keep;
      this.committedFrames += keep;
      // The lookahead is re-fed as pending audio on the next call. Retain only
      // the committed prefix so the final flush tail remains contiguous.
      this.retained.push(...window.slice(0, emit * FRAME_SAMPLES));
      const tail = this.tailContextFrames() * FRAME_SAMPLES;
      if (this.retained.length > tail) {
        this.retained.splice(0, this.retained.length - tail);
      }
      this.pending.splice(0, emit * FRAME_SAMPLES);
    }
    return this.committedFrames;
  }

  /**
   * Greedy-decodes the hidden states accumulated so far. The decoder pair has
   * no cross-call state, so previews are repeatable while audio streams.
   */
  async preview(): Promise<MoonshineStreamingText> {
    this.assertUsable();
    return this.decodeHidden();
  }

  /**
   * Flushes the remaining audio through the whole-utterance encoder on a tail
   * that includes the retained left context, then decodes everything. The
   * session is single-use after this; call reset() to stream again.
   */
  async finalize(): Promise<MoonshineStreamingText> {
    this.assertUsable();
    if (this.pending.length > 0) {
      const tailAudio = [...this.retained, ...this.pending];
      const committedInTail = this.retained.length;
      // The official feature extractor pads audio to a whole 5 ms (80 sample)
      // frame and keeps the original sample count in its attention mask. The
      // encoder uses mask.sum() // 80, so a trailing partial frame is ignored.
      const paddedLength = Math.ceil(tailAudio.length / FRAME_SAMPLES) * FRAME_SAMPLES;
      const encoderAudio = new Float32Array(paddedLength);
      encoderAudio.set(tailAudio);
      const whole = MoonshineStreamingSession.named(
        await this.options.wholeEncoder.run({
          input_values: this.floatTensor(encoderAudio, [1, encoderAudio.length]),
          attention_mask: this.int32Mask(tailAudio.length, paddedLength),
        }), 'last_hidden_state');
      const tailFrames = Math.floor(tailAudio.length / FRAME_SAMPLES);
      const validEndFrame = embeddedFrameCount(tailFrames);
      const committedEndFrame = embeddedFrameCount(committedInTail / FRAME_SAMPLES);
      const keep = validEndFrame - committedEndFrame;
      // The processor may pad one partial 80-sample frame, producing an
      // extra hidden row. Slice the valid prefix before removing context;
      // taking the final `keep` rows would shift the tail onto padding.
      const fresh = whole.data.slice(committedEndFrame * 320, validEndFrame * 320);
      this.hiddens.push(...Array.from(fresh));
      this.hiddenFrames += keep;
      this.committedFrames += keep;
    }
    this.pending.length = 0;
    this.retained.length = 0;
    const text = await this.decodeHidden();
    this.disposed = true;
    return text;
  }

  reset(): void {
    this.pending.length = 0;
    this.retained.length = 0;
    this.hiddens.length = 0;
    this.hiddenFrames = 0;
    this.committedFrames = 0;
    this.state = null;
    this.disposed = false;
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new Error('Moonshine streaming session was finalized. Call reset() to stream again.');
    }
  }

  private emitFrames(): number {
    return this.options.emitFrames ?? EMIT_FRAMES;
  }

  private lookaheadFrames(): number {
    return this.options.lookaheadFrames ?? LOOKAHEAD_FRAMES;
  }

  private tailContextFrames(): number {
    return this.options.tailContextFrames ?? TAIL_CONTEXT_FRAMES;
  }

  private chunkedFeeds(window: readonly number[], state: ChunkedState): Record<string, unknown> {
    if (window.length % FRAME_SAMPLES !== 0) {
      throw new Error('Chunked encoder windows must be whole 80-sample frames');
    }
    const frames = window.length / FRAME_SAMPLES;
    return {
      input_values: this.floatTensor(window, [1, frames, FRAME_SAMPLES]),
      embedded_context: this.floatTensor(state.embeddedContext.data, state.embeddedContext.dims),
      context_len: this.intScalar(state.contextLen),
      conv1_tail: this.floatTensor(state.conv1Tail.data, state.conv1Tail.dims),
      conv2_tail: this.floatTensor(state.conv2Tail.data, state.conv2Tail.dims),
      frames1: this.intScalar(state.frames1),
      frames2: this.intScalar(state.frames2),
    };
  }

  private initialState(): ChunkedState {
    const context = this.options.contextFrames ?? CONTEXT_FRAMES;
    return {
      embeddedContext: this.zeros([1, context, 320]),
      contextLen: 0,
      conv1Tail: this.zeros([1, 320, 4]),
      conv2Tail: this.zeros([1, 640, 4]),
      frames1: 0,
      frames2: 0,
    };
  }

  private nextState(state: ChunkedState, out: Record<string, unknown>): ChunkedState {
    void state;
    return {
      embeddedContext: MoonshineStreamingSession.named(out, 'new_embedded_context'),
      // The graph emits the absolute new context length (already clamped).
      contextLen: Number(MoonshineStreamingSession.named(out, 'new_context_len').data[0]),
      conv1Tail: MoonshineStreamingSession.named(out, 'new_conv1_tail'),
      conv2Tail: MoonshineStreamingSession.named(out, 'new_conv2_tail'),
      frames1: Number(MoonshineStreamingSession.named(out, 'new_frames1').data[0]),
      frames2: Number(MoonshineStreamingSession.named(out, 'new_frames2').data[0]),
    };
  }

  /**
   * Greedy decode over the accumulated hidden states. Past key/value tensors
   * are read by name from each graph's `present_*` outputs and fed to the next
   * step as `past_*`, matching the exported decoder contract.
   */
  private async decodeHidden(): Promise<MoonshineStreamingText> {
    const frames = this.hiddens.length / 320;
    if (frames === 0 || !Number.isInteger(frames)) {
      return { text: '', tokenIds: [], encoderFrames: frames };
    }
    const layers = this.options.decoderLayers ?? 6;
    const encoderStates = this.floatTensor(this.hiddens, [1, frames, 320]);
    let out = await this.options.prefill.run({
      input_ids: this.intToken(this.options.bosTokenId),
      encoder_hidden_states: encoderStates,
    });
    const tokenIds: number[] = [];
    const limit = this.options.maxTokens ?? 256;
    while (tokenIds.length < limit) {
      const bestId = MoonshineStreamingSession.argMax(
        MoonshineStreamingSession.named(out, 'logits').data);
      if (bestId === this.options.eosTokenId) {
        break;
      }
      tokenIds.push(bestId);
      const feeds: Record<string, unknown> = {
        input_ids: this.intToken(bestId),
        encoder_hidden_states: encoderStates,
      };
      for (let index = 0; index < layers; index += 1) {
        feeds[`past_k_${index}`] = MoonshineStreamingSession.named(out, `present_k_${index}`);
        feeds[`past_v_${index}`] = MoonshineStreamingSession.named(out, `present_v_${index}`);
      }
      out = await this.options.step.run(feeds);
    }
    return {
      text: this.options.tokenizer.decode(tokenIds),
      tokenIds,
      encoderFrames: frames,
    };
  }

  private floatTensor(data: readonly number[] | Float32Array, dims: readonly number[]): unknown {
    const array = data instanceof Float32Array ? data : new Float32Array(data);
    if ((this.options.dtype ?? 'float32') === 'float16') {
      return new this.options.ort.Tensor('float16', float32ToFloat16(array), dims);
    }
    return new this.options.ort.Tensor('float32', array, dims);
  }

  private zeros(dims: readonly number[]): TensorLike {
    const total = dims.reduce((product, dim) => product * dim, 1);
    return this.floatTensor(new Float32Array(total), dims) as TensorLike;
  }

  private intScalar(value: number): unknown {
    return new this.options.ort.Tensor('int64', BigInt64Array.from([BigInt(value)]), []);
  }

  private int32Mask(validSamples: number, paddedSamples: number): unknown {
    const mask = new Int32Array(paddedSamples);
    mask.fill(1, 0, validSamples);
    return new this.options.ort.Tensor('int32', mask, [1, paddedSamples]);
  }

  private intToken(id: number): unknown {
    return new this.options.ort.Tensor('int64', BigInt64Array.from([BigInt(id)]), [1, 1]);
  }

  private static named(result: Record<string, unknown>, name: string): TensorLike {
    const tensor = result?.[name];
    if (!tensor) {
      throw new Error(`Moonshine streaming graph did not return ${name}`);
    }
    return tensor as TensorLike;
  }

  private static argMax(logits: Float32Array): number {
    let bestId = 0;
    let best = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < logits.length; index += 1) {
      const value = logits[index] ?? Number.NEGATIVE_INFINITY;
      if (value > best) {
        best = value;
        bestId = index;
      }
    }
    return bestId;
  }
}
