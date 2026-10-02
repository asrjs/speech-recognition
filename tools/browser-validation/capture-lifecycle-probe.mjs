import { startMicrophoneCapture } from '/dist/browser.js';

function check(condition, message) {
  if (!condition) throw new Error(message);
}

export async function runCaptureLifecycle() {
  check(window.isSecureContext, 'Capture requires a secure context');
  check(typeof AudioWorkletNode === 'function', 'AudioWorkletNode unavailable');
  const runs = [];
  for (let iteration = 0; iteration < 2; iteration += 1) {
    let context;
    let handle;
    let workletLoads = 0;
    let chunks = 0;
    let endFrame = 0;
    let peak = 0;
    const errors = [];
    let ready;
    let rejectReady;
    const enoughChunks = new Promise((resolve, reject) => {
      ready = resolve;
      rejectReady = reject;
    });
    const timer = setTimeout(() => rejectReady(new Error('Capture chunk timeout')), 10000);
    // Install a rejection handler before acquisition; startup can fail independently.
    enoughChunks.catch(() => {});
    try {
      handle = await startMicrophoneCapture({
        targetSampleRate: 16000,
        chunkFrames: 256,
        createAudioContext(sampleRate) {
          context = new AudioContext({ sampleRate });
          const addModule = context.audioWorklet.addModule.bind(context.audioWorklet);
          context.audioWorklet.addModule = async (...args) => {
            await addModule(...args);
            workletLoads += 1;
          };
          return context;
        },
        onChunk(chunk) {
          try {
            check(chunk.pcm.length === 256, 'Unexpected chunk length');
            check(chunk.sampleRate === 16000, 'Unexpected processing sample rate');
            check(chunk.startFrame === endFrame, 'Noncontiguous capture frame timeline');
            check(chunk.endFrame === chunk.startFrame + 256, 'Incorrect end frame');
            check(chunk.startTimeSeconds === chunk.startFrame / 16000, 'Incorrect start time');
            check(chunk.endTimeSeconds === chunk.endFrame / 16000, 'Incorrect end time');
            for (const sample of chunk.pcm) {
              check(Number.isFinite(sample), 'Nonfinite captured PCM');
              peak = Math.max(peak, Math.abs(sample));
            }
            endFrame = chunk.endFrame;
            chunks += 1;
            if (chunks >= 4) ready();
          } catch (error) {
            errors.push(String(error));
            rejectReady(error);
          }
        },
        onError(error) {
          errors.push(String(error));
          rejectReady(error);
        },
      });
      await enoughChunks;
      check(workletLoads === 1, 'Actual AudioWorklet module was not loaded');
      check(context.state === 'running', 'Context was not running during capture');
      check(handle.stream.getAudioTracks().length > 0, 'No actual microphone track');
      check(
        handle.stream.getTracks().every((track) => track.readyState === 'live'),
        'Capture track was not live',
      );
      check(peak > 0, 'Fake device produced no nonzero PCM');
      const firstStop = handle.stop();
      const secondStop = handle.stop();
      check(firstStop === secondStop, 'Concurrent stops did not share cleanup');
      await Promise.all([firstStop, secondStop]);
      await handle.stop();
      check(context.state === 'closed', 'AudioContext remains open');
      check(
        handle.stream.getTracks().every((track) => track.readyState === 'ended'),
        'Owned track remains live',
      );
      const stoppedChunks = chunks;
      // Explicit bounded observation for worklet messages already in flight.
      await new Promise((resolve) => setTimeout(resolve, 150));
      check(chunks === stoppedChunks, 'Chunks delivered after awaited stop');
      check(errors.length === 0, `Capture errors: ${errors.join('; ')}`);
      runs.push({
        iteration,
        chunks,
        endFrame,
        peak,
        workletLoads,
        processingSampleRate: handle.sampleRate,
        deviceSampleRate: handle.deviceSampleRate,
        contextSampleRate: handle.contextSampleRate,
        contextState: context.state,
        trackStates: handle.stream.getTracks().map((track) => track.readyState),
        concurrentStopShared: firstStop === secondStop,
        postStopObservationMs: 150,
      });
    } finally {
      clearTimeout(timer);
      if (handle) await handle.stop();
      else if (context && context.state !== 'closed') await context.close();
    }
  }
  return { status: 'PASS', scope: 'REAL_BROWSER_FAKE_DEVICE_CAPTURE_OWNER_ONLY', runs };
}
