import { expect, it, vi } from 'vitest';
import { runWindowedController } from '../examples/demo/src/windowed-controller.js';

it('feeds every sample once, invokes real controller windows and resets after finalization', async () => {
  const pcm = Float32Array.from({ length: 111 }, (_, index) => index / 111);
  const transcribe = vi.fn(async (request) => {
    expect(request.pcm).toEqual(pcm.slice(request.startFrame, request.endFrame));
    expect(request.signal.aborted).toBe(false);
    return { text: 'hello', meta: { detailLevel: 'text', isFinal: true } };
  });
  const result = await runWindowedController({ pcm, sampleRate: 10, transcribe });
  expect(result.chunkRanges).toEqual([
    { startFrame: 0, endFrame: 20 }, { startFrame: 20, endFrame: 40 },
    { startFrame: 40, endFrame: 60 }, { startFrame: 60, endFrame: 80 },
    { startFrame: 80, endFrame: 100 }, { startFrame: 100, endFrame: 111 },
  ]);
  expect(transcribe.mock.calls.length).toBeGreaterThan(1);
  expect(result.updates.some((update) => update.kind === 'partial')).toBe(true);
  expect(result.updates.at(-1)?.kind).toBe('final');
  expect(result.finalized).toBe(true);
  expect(result.pushedSamples).toBe(pcm.length);
  expect(result.finalText).toBe('hello');
  expect(result.resetState).toEqual({ currentFrame: 0, isFinalized: false, text: '' });
});

it('propagates actual callback failure instead of passing a simulated transcript', async () => {
  const transcribe = vi.fn(async () => { throw new Error('inference failed'); });
  const receipt = {};
  await expect(runWindowedController({ pcm: new Float32Array(30), sampleRate: 10, transcribe, receipt }))
    .rejects.toThrow('inference failed');
  expect(transcribe).toHaveBeenCalledOnce();
  expect(receipt).toMatchObject({ chunkRanges: [{ startFrame: 0, endFrame: 20 }],
    resetState: { currentFrame: 0, isFinalized: false, text: '' } });
});

it('rejects empty or invalid input before inference', async () => {
  const transcribe = vi.fn();
  for (const options of [{ pcm: new Float32Array(0), sampleRate: 10 },
    { pcm: new Float32Array(30), sampleRate: 0 },
    { pcm: new Float32Array(30), sampleRate: 10, chunkSeconds: 0 }]) {
    await expect(runWindowedController({ ...options, transcribe })).rejects.toThrow(TypeError);
  }
  expect(transcribe).not.toHaveBeenCalled();
});

it('records immutable local and global timing payloads across nonzero window origins', async () => {
  const originals: any[] = [];
  const result = await runWindowedController({ pcm: new Float32Array(340), sampleRate: 10,
    transcribe(request) {
      const item = { index: 0, text: `call-${originals.length}.`, startTime: 0.25, endTime: 1 };
      const canonical = { text: item.text, meta: { detailLevel: 'detailed', isFinal: false },
        words: [item], segments: [item], sentences: [item],
        tokens: [{ ...item, id: 1 }, { index: 1, text: 'untimed', id: 2 }] };
      originals.push(canonical);
      return canonical;
    },
  });
  expect(result.updates.some((update) => update.window.startFrame > 0)).toBe(true);
  for (const update of result.updates) {
    const source = result.inferenceResults[update.inferenceIndex];
    expect(source.startFrame).toBe(update.window.startFrame);
    expect(source.endFrame).toBe(update.window.endFrame);
    for (const field of ['words', 'segments', 'sentences', 'tokens']) {
      const expected = source.localTiming[field].map((item) => ({ ...item,
        ...(item.startTime === undefined ? {} : { startTime: item.startTime + source.startFrame / 10 }),
        ...(item.endTime === undefined ? {} : { endTime: item.endTime + source.startFrame / 10 }),
      }));
      expect(update.canonicalTiming[field]).toEqual(expected);
      expect(update.partialTiming[field]).toEqual(expected);
    }
  }
  originals[0].words[0].startTime = 999;
  expect(result.inferenceResults[0].localTiming.words[0].startTime).toBe(0.25);
  expect(result.updates[0].canonicalTiming.words[0].startTime).toBe(0.25);
});

it('does not attribute a finalization-only update to the previous inference', async () => {
  const result = await runWindowedController({ pcm: new Float32Array(40), sampleRate: 10,
    transcribe(request) {
      // A single sentence stays pending; two sentences advance the mature cursor
      // and leave less than the minimum window for finalize (no new inference).
      const duration = request.pcm.length / request.sampleRate;
      const words = ['one', 'two', 'three.', 'tail'].map((text, index) => ({ index, text,
        startTime: 0, endTime: index < 3 ? duration - 0.5 : duration }));
      return { text: 'one two three. tail', meta: { detailLevel: 'detailed', isFinal: false },
        words, segments: [], sentences: [], tokens: [] };
    },
  });
  expect(result.updates.length).toBe(result.inferenceResults.length + 1);
  expect(result.updates.at(-1).inferenceIndex).toBeNull();
  expect(result.updates.at(-1).canonicalTiming).toEqual({ words: null, segments: null,
    sentences: null, tokens: null });
});
