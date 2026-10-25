import { describe, expect, it, vi } from 'vitest';
import {
  embeddedFrameCount,
  MoonshineStreamingSession,
} from '../src/models/moonshine/streaming-session.js';
import { MoonshineTokenizer } from '../src/models/moonshine/session.js';

const FRAME = 80;
const CONTEXT = 90;
const LOOKAHEAD = 60;
const VOCAB = 32;

class FakeTensor {
  constructor(readonly type: string, readonly data: Float32Array | Int32Array | BigInt64Array, readonly dims: readonly number[]) {}
}

const ort = { Tensor: FakeTensor as unknown as new (type: string, data: ArrayBufferView, dims: readonly number[]) => unknown };

function floatTensor(data: readonly number[], dims: readonly number[]): FakeTensor {
  return new FakeTensor('float32', new Float32Array(data), dims);
}

function intScalar(value: number): FakeTensor {
  return new FakeTensor('int64', BigInt64Array.from([BigInt(value)]), []);
}

function tokenizer(): MoonshineTokenizer {
  // SentencePiece-style vocab where token id maps to a plain piece.
  return new MoonshineTokenizer(
    Array.from({ length: VOCAB }, (_, id) => `▁t${id}`), 1);
}

interface ChunkedCall {
  windowAudioFrames: number;
  contextLen: number;
  frames1: number;
  frames2: number;
  conv1TailSum: number;
}

/** Chunked encoder mock: hidden frames are tagged with their global embedded index. */
function chunkedEncoderMock(calls: ChunkedCall[]) {
  let embeddedSoFar = 0;
  return {
    run: vi.fn(async (feeds: Record<string, unknown>) => {
      const window = feeds['input_values'] as FakeTensor;
      const audioFrames = window.dims[1]!; // rank-3 feed: [1, frames, 80]
      const frames = audioFrames / 4;
      const data = new Float32Array(frames * 320);
      for (let index = 0; index < frames; index += 1) {
        data[index * 320] = embeddedSoFar + index; // tag: global embedded frame id
      }
      embeddedSoFar += frames;
      calls.push({
        windowAudioFrames: audioFrames,
        contextLen: Number((feeds['context_len'] as FakeTensor).data[0]),
        frames1: Number((feeds['frames1'] as FakeTensor).data[0]),
        frames2: Number((feeds['frames2'] as FakeTensor).data[0]),
        conv1TailSum: Array.from((feeds['conv1_tail'] as FakeTensor).data as Float32Array)
          .reduce((sum, value) => sum + value, 0),
      });
      return {
        hidden: floatTensor(Array.from(data), [1, frames, 320]),
        new_embedded_context: floatTensor(
          Array.from({ length: CONTEXT * 320 }, (_, index) => (index % 320) + 1), [1, CONTEXT, 320]),
        new_context_len: intScalar(CONTEXT),
        new_conv1_tail: floatTensor(Array.from({ length: 320 * 4 }, (_, index) => index + 1), [1, 320, 4]),
        new_conv2_tail: floatTensor(Array.from({ length: 640 * 4 }, (_, index) => index + 2), [1, 640, 4]),
        new_frames1: intScalar(audioFrames),
        new_frames2: intScalar(audioFrames / 4),
      };
    }),
  };
}

/** Whole-utterance encoder mock: tags frames from a configurable start id. */
function wholeEncoderMock(startId: number) {
  return {
    run: vi.fn(async (feeds: Record<string, unknown>) => {
      const window = feeds['input_values'] as FakeTensor;
      const frames = embeddedFrameCount(window.dims[1]! / FRAME);
      const data = new Float32Array(frames * 320);
      for (let index = 0; index < frames; index += 1) {
        data[index * 320] = startId + index;
      }
      return { last_hidden_state: floatTensor(Array.from(data), [1, frames, 320]) };
    }),
  };
}

interface DecodeTrace {
  prefills: number;
  steps: number[];
  lastPastKeys: FakeTensor | null;
  encoderFrames: number[];
}

/** Decoder mocks: prefill tags token 7 (continue), step emits 8 then EOS (2). */
function decoderMocks(trace: DecodeTrace, layers = 6) {
  const logits = (token: number) => floatTensor(Array.from({ length: VOCAB },
    (_, id) => (id === token ? 5 : 0)), [1, 1, VOCAB]);
  const present = () => Object.fromEntries(
    Array.from({ length: layers }, (_, index) => [
      `present_k_${index}`, floatTensor([1], [1, 8, 1, 40]),
    ]).concat(Array.from({ length: layers }, (_, index) => [
      `present_v_${index}`, floatTensor([1], [1, 8, 1, 40]),
    ])));
  return {
    prefill: {
      run: vi.fn(async (feeds: Record<string, unknown>) => {
        trace.prefills += 1;
        trace.encoderFrames.push((feeds['encoder_hidden_states'] as FakeTensor).dims[1]!);
        return { logits: logits(7), ...present() };
      }),
    },
    step: {
      run: vi.fn(async (feeds: Record<string, unknown>) => {
        const id = Number((feeds['input_ids'] as FakeTensor).data[0]);
        trace.steps.push(id);
        trace.lastPastKeys = feeds['past_k_0'] as FakeTensor;
        return { logits: logits(id === 7 ? 8 : 2), ...present() };
      }),
    },
  };
}

function streamingSession(emit: number, overrides: Partial<Record<string, unknown>> = {}) {
  const calls: ChunkedCall[] = [];
  const trace: DecodeTrace = { prefills: 0, steps: [], lastPastKeys: null, encoderFrames: [] };
  const chunkedEncoder = chunkedEncoderMock(calls);
  let wholeStart = 1000;
  const wholeEncoder = {
    run: vi.fn(async (feeds: Record<string, unknown>) => {
      const session = wholeEncoderMock(wholeStart);
      wholeStart += 1000;
      return session.run(feeds);
    }),
  };
  const decoders = decoderMocks(trace);
  const session = new MoonshineStreamingSession({
    ort, chunkedEncoder, wholeEncoder, ...decoders,
    tokenizer: tokenizer(), bosTokenId: 1, eosTokenId: 2,
    contextFrames: CONTEXT, lookaheadFrames: LOOKAHEAD, emitFrames: emit,
    tailContextFrames: 480, decoderLayers: 6,
    ...overrides,
  } as ConstructorParameters<typeof MoonshineStreamingSession>[0]);
  return { session, calls, trace, wholeEncoder, chunkedEncoder, ...decoders };
}

describe('embeddedFrameCount', () => {
  it('matches the two stride-2 convs for aligned and ragged inputs', () => {
    expect(embeddedFrameCount(0)).toBe(0);
    expect(embeddedFrameCount(76)).toBe(19);
    expect(embeddedFrameCount(440)).toBe(110);
    expect(embeddedFrameCount(441)).toBe(111);
  });
});

describe('MoonshineStreamingSession window contract', () => {
  it('feeds emit+lookahead windows, advances by emit, and commits emit/4 frames per call', async () => {
    const emit = 64;
    const { session, chunkedEncoder } = streamingSession(emit);
    // 252 frames: calls run while >= emit+lookahead frames are buffered, each
    // consuming emit, so calls 1..3 run and 124 frames of lookahead remain.
    const totalFrames = 252;
    const committed = await session.pushAudio(new Float32Array(totalFrames * FRAME));
    expect(chunkedEncoder.run).toHaveBeenCalledTimes(3);
    expect(committed).toBe(3 * (emit / 4));
    expect(session.encoderFrames).toBe(committed);
  });

  it('carries the emitted state tensors into the next call', async () => {
    const emit = 64;
    const { session, calls } = streamingSession(emit);
    await session.pushAudio(new Float32Array(2 * (emit + LOOKAHEAD) * FRAME));
    expect(calls.length).toBe(2);
    expect(calls[0]!.contextLen).toBe(0);
    expect(calls[0]!.frames1).toBe(0);
    expect(calls[1]!.contextLen).toBe(CONTEXT); // graph returned the clamped absolute value
    expect(calls[1]!.frames1).toBe(emit + LOOKAHEAD);
    expect(calls[1]!.frames2).toBe((emit + LOOKAHEAD) / 4);
  });

  it('retains only tailContextFrames of consumed audio for the final flush', async () => {
    const emit = 64;
    const { session } = streamingSession(emit);
    await session.pushAudio(new Float32Array(3 * (emit + LOOKAHEAD) * FRAME));
    // @ts-expect-error private field inspection for the retention bound
    expect(session.retained.length).toBeLessThanOrEqual(480 * FRAME);
  });

  it('flushes a contiguous original-audio tail across re-fed lookahead and the 480-frame horizon', async () => {
    const emit = 64;
    const { session, wholeEncoder } = streamingSession(emit);
    const totalFrames = 9 * emit + LOOKAHEAD;
    const samples = Float32Array.from(
      { length: totalFrames * FRAME },
      (_, index) => (index % 997) / 997,
    );

    await session.pushAudio(samples);
    expect(session.committedEncoderFrames).toBe(9 * (emit / 4));
    await session.finalize();

    const feeds = wholeEncoder.run.mock.calls[0]![0] as Record<string, FakeTensor>;
    const values = feeds['input_values']!.data as Float32Array;
    const validSamples = (feeds['attention_mask']!.data as Int32Array)
      .reduce((sum, value) => sum + value, 0);
    const expectedTail = samples.subarray(samples.length - (480 + LOOKAHEAD) * FRAME);

    expect(validSamples).toBe(expectedTail.length);
    expect(Array.from(values.subarray(0, validSamples))).toEqual(Array.from(expectedTail));
  });

  it('finalize flushes the tail through the whole encoder to the whole-utterance framing', async () => {
    const emit = 64;
    const { session, wholeEncoder, trace } = streamingSession(emit);
    const windowFrames = emit + LOOKAHEAD;
    const tailFrames = emit + 40;
    const totalFrames = 2 * windowFrames + tailFrames;
    await session.pushAudio(new Float32Array(2 * windowFrames * FRAME));
    await session.pushAudio(new Float32Array(tailFrames * FRAME));
    const result = await session.finalize();
    expect(wholeEncoder.run).toHaveBeenCalledTimes(1);
    // The streamed framing must land on the whole-utterance framing: total
    // embedded frames equals embeddedFrameCount of all audio ever pushed.
    expect(result.encoderFrames).toBe(embeddedFrameCount(totalFrames));
    expect(trace.prefills).toBe(1);
    // Greedy loop: prefill says 7, step(7) says 8, step(8) says EOS.
    expect(result.tokenIds).toEqual([7, 8]);
    expect(result.text).toBe('t7 t8');
  });

  it('pads ragged samples to 80-sample frames and masks the padded suffix', async () => {
    const { session, wholeEncoder } = streamingSession(64);
    const samples = new Float32Array(8_584).fill(0.25);
    await session.pushAudio(samples);

    const result = await session.finalize();
    const feeds = wholeEncoder.run.mock.calls[0]![0] as Record<string, FakeTensor>;
    const values = feeds['input_values']!;
    const mask = feeds['attention_mask']!;

    expect(values.dims).toEqual([1, 8_640]);
    expect(Array.from((values.data as Float32Array).slice(0, 8_584))).toEqual(Array.from(samples));
    expect(Array.from((values.data as Float32Array).slice(8_584))).toEqual(Array(56).fill(0));
    expect(mask.dims).toEqual([1, 8_640]);
    expect((mask.data as Int32Array).reduce((sum, value) => sum + value, 0)).toBe(8_584);
    expect(Array.from((mask.data as Int32Array).slice(8_584))).toEqual(Array(56).fill(0));
    // Processor mask semantics count only complete 80-sample frames.
    expect(result.encoderFrames).toBe(embeddedFrameCount(107));
  });

  it('keeps the valid hidden prefix when an incomplete padded frame adds an output row', async () => {
    const { session, wholeEncoder } = streamingSession(64);
    const samples = new Float32Array(108 * FRAME + 64).fill(0.25);
    await session.pushAudio(samples);

    const result = await session.finalize();
    const feeds = wholeEncoder.run.mock.calls[0]![0] as Record<string, FakeTensor>;
    expect(feeds['input_values']!.dims).toEqual([1, 8_720]);
    // The whole graph returns 28 rows for 109 padded frames, but only the
    // first 27 rows correspond to the 108 complete valid frames.
    expect(result.encoderFrames).toBe(27);
    // @ts-expect-error private hidden data pins prefix-vs-suffix selection
    expect(session.hiddens[0]).toBe(1_000);
    // @ts-expect-error private hidden data pins prefix-vs-suffix selection
    expect(session.hiddens[(27 - 1) * 320]).toBe(1_026);
  });

  it('crops padded hidden rows after removing committed tail context', async () => {
    const { session } = streamingSession(64);
    const samples = new Float32Array(192 * FRAME + 40).fill(0.25);
    await session.pushAudio(samples);

    const result = await session.finalize();
    expect(result.encoderFrames).toBe(embeddedFrameCount(192));
    // 32 committed frames precede 16 fresh valid rows; the 49th whole-graph
    // row came from padding and must not replace either edge of that slice.
    // @ts-expect-error private hidden data pins the committed-context offset
    expect(session.hiddens[32 * 320]).toBe(1_032);
    // @ts-expect-error private hidden data pins the valid-tail endpoint
    expect(session.hiddens[(48 - 1) * 320]).toBe(1_047);
  });

  it('does not append masked encoder frames when fewer than 80 samples remain', async () => {
    const { session, trace, wholeEncoder } = streamingSession(64);
    await session.pushAudio(new Float32Array(79).fill(0.25));

    const result = await session.finalize();
    const feeds = wholeEncoder.run.mock.calls[0]![0] as Record<string, FakeTensor>;
    expect((feeds['input_values']!.data as Float32Array).length).toBe(80);
    expect((feeds['attention_mask']!.data as Int32Array).reduce((sum, value) => sum + value, 0)).toBe(79);
    expect(result).toEqual({ text: '', tokenIds: [], encoderFrames: 0 });
    expect(trace.prefills).toBe(0);
  });

  it('keeps the whole-utterance framing at an exact 80-sample boundary', async () => {
    const { session, wholeEncoder } = streamingSession(64);
    await session.pushAudio(new Float32Array(80).fill(0.25));

    const result = await session.finalize();
    const feeds = wholeEncoder.run.mock.calls[0]![0] as Record<string, FakeTensor>;
    expect(feeds['input_values']!.dims).toEqual([1, 80]);
    expect((feeds['attention_mask']!.data as Int32Array).reduce((sum, value) => sum + value, 0)).toBe(80);
    expect(result.encoderFrames).toBe(embeddedFrameCount(1));
  });

  it('wires per-layer past tensors from present outputs into step feeds', async () => {
    const emit = 64;
    const { session, trace } = streamingSession(emit);
    await session.pushAudio(new Float32Array((emit + LOOKAHEAD + emit) * FRAME));
    await session.finalize();
    expect(trace.lastPastKeys).toBeInstanceOf(FakeTensor);
  });

  it('preview is repeatable and does not consume the session', async () => {
    const emit = 64;
    const { session, trace } = streamingSession(emit);
    await session.pushAudio(new Float32Array((emit + LOOKAHEAD) * FRAME));
    const first = await session.preview();
    const second = await session.preview();
    expect(first).toEqual(second);
    expect(trace.prefills).toBe(2);
    expect(session.encoderFrames).toBe(emit / 4);
  });

  it('finalize disposes the session and reset revives it', async () => {
    const emit = 64;
    const { session } = streamingSession(emit);
    await session.pushAudio(new Float32Array((emit + LOOKAHEAD) * FRAME));
    await session.finalize();
    await expect(session.pushAudio(new Float32Array(FRAME))).rejects.toThrow('finalize');
    session.reset();
    await expect(session.pushAudio(new Float32Array(FRAME))).resolves.toBe(0);
  });

  it('decodeHidden on empty audio returns empty text without graph calls', async () => {
    const { session, trace } = streamingSession(64);
    const result = await session.finalize();
    expect(result).toEqual({ text: '', tokenIds: [], encoderFrames: 0 });
    expect(trace.prefills).toBe(0);
  });
});
