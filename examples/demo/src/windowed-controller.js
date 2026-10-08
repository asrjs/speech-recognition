import { RealtimeTranscriptionController } from '../../../src/realtime.ts';

function captureTiming(transcript) {
  return structuredClone(Object.fromEntries(['words', 'segments', 'sentences', 'tokens']
    .map((field) => [field, transcript[field] ?? null])));
}

/** Chunk-fed offline ASR example; no native streaming state or microphone claim. */
export async function runWindowedController({ pcm, sampleRate, transcribe, chunkSeconds = 2, receipt = {} }) {
  if (!(pcm instanceof Float32Array) || !pcm.length || !Number.isInteger(sampleRate) || sampleRate <= 0 ||
      !Number.isFinite(chunkSeconds) || chunkSeconds <= 0) {
    throw new TypeError('Nonempty float32 PCM, positive integer sample rate and chunk duration are required');
  }
  const chunkSamples = Math.max(1, Math.round(chunkSeconds * sampleRate));
  Object.assign(receipt, { scope: 'chunk-fed-windowed-offline-asr', chunkSamples,
    chunkRanges: [], updates: [], inferenceResults: [] });
  const controller = new RealtimeTranscriptionController({
    sampleRate, bufferDurationSeconds: 30, latency: true,
    async transcribe(request) {
      const canonical = await transcribe(request);
      receipt.inferenceResults.push({ startFrame: request.startFrame, endFrame: request.endFrame,
        sampleRate: request.sampleRate, localTiming: captureTiming(canonical) });
      return canonical;
    },
    finalizeSilenceSeconds: 0,
    window: { minDurationSec: 2, minInitialDurationSec: 1.5, maxDurationSec: 30,
      useActivityBoundaries: false },
  });
  function record(update, inferenceIndex) {
    if (!update) return;
    receipt.updates.push({ kind: update.kind, trigger: update.trigger, text: update.partial.text,
      committedText: update.partial.committedText, previewText: update.partial.previewText,
      window: update.window, metrics: update.canonical.meta.metrics,
      inferenceIndex,
      canonicalTiming: captureTiming(update.canonical), partialTiming: captureTiming(update.partial) });
  }
  async function recordOperation(operation) {
    const before = receipt.inferenceResults.length;
    const update = await operation();
    record(update, receipt.inferenceResults.length > before ? receipt.inferenceResults.length - 1 : null);
    return update;
  }
  try {
    for (let start = 0; start < pcm.length; start += chunkSamples) {
      const end = Math.min(start + chunkSamples, pcm.length);
      receipt.chunkRanges.push({ startFrame: start, endFrame: end });
      await recordOperation(() => controller.pushAudio(pcm.subarray(start, end)));
    }
    const final = await recordOperation(() => controller.finalize());
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
