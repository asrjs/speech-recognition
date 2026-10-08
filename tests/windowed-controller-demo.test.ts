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
