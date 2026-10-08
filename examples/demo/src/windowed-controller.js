import { RealtimeTranscriptionController } from '../../../src/realtime.ts';

/** Chunk-fed offline ASR example; no native streaming state or microphone claim. */
export async function runWindowedController({ pcm, sampleRate, transcribe, chunkSeconds = 2, receipt = {} }) {
  if (!(pcm instanceof Float32Array) || !pcm.length || !Number.isInteger(sampleRate) || sampleRate <= 0 ||
      !Number.isFinite(chunkSeconds) || chunkSeconds <= 0) {
    throw new TypeError('Nonempty float32 PCM, positive integer sample rate and chunk duration are required');
  }
  const chunkSamples = Math.max(1, Math.round(chunkSeconds * sampleRate));
  const controller = new RealtimeTranscriptionController({
    sampleRate, transcribe, bufferDurationSeconds: 30, latency: true,
    finalizeSilenceSeconds: 0,
    window: { minDurationSec: 2, minInitialDurationSec: 1.5, maxDurationSec: 30,
      useActivityBoundaries: false },
  });
  Object.assign(receipt, { scope: 'chunk-fed-windowed-offline-asr', chunkSamples, chunkRanges: [], updates: [] });
  function record(update) {
    if (!update) return;
    receipt.updates.push({ kind: update.kind, trigger: update.trigger, text: update.partial.text,
      committedText: update.partial.committedText, previewText: update.partial.previewText,
      window: update.window, metrics: update.canonical.meta.metrics });
  }
  try {
    for (let start = 0; start < pcm.length; start += chunkSamples) {
      const end = Math.min(start + chunkSamples, pcm.length);
      receipt.chunkRanges.push({ startFrame: start, endFrame: end });
      record(await controller.pushAudio(pcm.subarray(start, end)));
    }
    const final = await controller.finalize();
    record(final);
    const state = controller.getState();
    receipt.finalized = state.isFinalized;
    receipt.pushedSamples = state.currentFrame;
    receipt.finalText = final?.partial.text ?? '';
    receipt.latency = state.latency;
    if (!final || final.kind !== 'final' || !receipt.finalText.trim()) {
      throw new Error('Controller did not produce a nonempty final transcript');
    }
    return receipt;
  } finally {
    controller.reset();
    const reset = controller.getState();
    receipt.resetState = { currentFrame: reset.currentFrame, isFinalized: reset.isFinalized,
      text: reset.snapshot.fullText };
  }
}
