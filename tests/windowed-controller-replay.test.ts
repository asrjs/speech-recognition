import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { decodeWavToMonoPcm } from '../src/audio/wav.js';
import { runWindowedController } from '../examples/demo/src/windowed-controller.js';

const receiptPath = process.env.ASRJS_CONTROLLER_REPLAY_RECEIPT;
const requested = [
  receiptPath,
  process.env.ASRJS_CONTROLLER_REPLAY_SHA256,
  process.env.ASRJS_CONTROLLER_REPLAY_AUDIO,
].some((value) => value !== undefined);

function sha256(bytes: Uint8Array) {
  return createHash('sha256').update(bytes).digest('hex');
}

function pcmDigest(pcm: Float32Array) {
  const bytes = Buffer.alloc(pcm.length * 4);
  pcm.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  return sha256(bytes);
}

// Receipt regression only: no model inference, official oracle or quality PASS.
it.skipIf(!requested)(
  'replays recorded Chrome windows through the current public controller',
  async () => {
    expect(receiptPath).toBeTruthy();
    expect(process.env.ASRJS_CONTROLLER_REPLAY_SHA256).toMatch(/^[a-f0-9]{64}$/);
    expect(process.env.ASRJS_CONTROLLER_REPLAY_AUDIO).toBeTruthy();
    const bytes = readFileSync(receiptPath!);
    expect(sha256(bytes)).toBe(process.env.ASRJS_CONTROLLER_REPLAY_SHA256);
    const saved = JSON.parse(bytes.toString('utf8'));
    const audio = readFileSync(process.env.ASRJS_CONTROLLER_REPLAY_AUDIO!);
    expect(sha256(audio)).toBe(saved.audio_sha256);
    const { pcm, sampleRate, numberOfChannels } = decodeWavToMonoPcm(audio);
    expect(numberOfChannels).toBe(1);
    expect(pcmDigest(pcm)).toBe(saved.expected_pcm_sha256);
    const original = saved.result.controller;
    expect(original.scope).toBe('chunk-fed-windowed-offline-asr');
    expect(original.requests.length).toBeGreaterThan(0);
    expect(original.inferenceResults.length).toBe(original.requests.length);
    expect(pcm.length).toBe(original.pushedSamples);
    let calls = 0;
    const replay = await runWindowedController({
      pcm,
      sampleRate,
      chunkSeconds: original.chunkSamples / sampleRate,
      transcribe(request) {
        const recorded = original.requests[calls];
        const source = original.inferenceResults[calls++];
        expect(recorded).toBeDefined();
        expect([request.startFrame, request.endFrame, request.sampleRate, request.reason]).toEqual([
          recorded.startFrame,
          recorded.endFrame,
          recorded.sampleRate,
          recorded.reason,
        ]);
        expect([source.startFrame, source.endFrame, source.sampleRate]).toEqual([
          request.startFrame,
          request.endFrame,
          request.sampleRate,
        ]);
        expect(pcmDigest(request.pcm)).toBe(recorded.pcmSha256);
        const timing = Object.fromEntries(
          Object.entries(source.localTiming).filter(([, value]) => value !== null),
        );
        return structuredClone({
          text: recorded.text,
          ...timing,
          meta: { detailLevel: 'detailed', isFinal: true },
        });
      },
    });
    expect(calls).toBe(original.requests.length);
    expect(replay.chunkRanges).toEqual(original.chunkRanges);
    expect(replay.inferenceResults).toEqual(original.inferenceResults);
    // Runtime timing measurements are not deterministic; transcript/timing payloads are.
    const stableUpdate = ({ metrics: _metrics, ...update }: Record<string, unknown>) => update;
    expect(replay.updates.map(stableUpdate)).toEqual(original.updates.map(stableUpdate));
    expect(replay.finalText).toBe(original.finalText);
    expect(replay.finalized).toBe(original.finalized);
    expect(replay.resetState).toEqual(original.resetState);
  },
);
