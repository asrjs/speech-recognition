import {
  startMicrophoneCapture,
  createBrowserRealtimeMicrophoneController,
} from '/dist/browser.js';

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

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => {
    resolve = done;
    reject = fail;
  });
  promise.catch(() => {});
  return { promise, resolve, reject };
}

export async function runControllerLifecycle() {
  const sessions = [];
  const utterances = [];
  const errors = [];
  const timers = [];
  function createController(handoff = null) {
    const acquired = deferred();
    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      onUtterance: (utterance) => utterances.push(utterance),
      startCapture: async (options) => {
        const session = {
          context: null,
          handle: null,
          chunks: 0,
          accepted: [],
          stopCalls: 0,
          workletLoads: 0,
          endFrame: 0,
          ready: deferred(),
        };
        sessions.push(session);
        const timer = setTimeout(
          () => session.ready.reject(new Error('Controller chunk timeout')),
          10000,
        );
        timers.push(timer);
        session.handle = await startMicrophoneCapture({
          ...options,
          createAudioContext(sampleRate) {
            session.context = new AudioContext({ sampleRate });
            const addModule = session.context.audioWorklet.addModule.bind(
              session.context.audioWorklet,
            );
            session.context.audioWorklet.addModule = async (...args) => {
              await addModule(...args);
              session.workletLoads += 1;
            };
            return session.context;
          },
          onChunk(chunk) {
            try {
              check(chunk.sampleRate === 16000 && chunk.pcm.length > 0, 'Invalid controller PCM');
              check(
                chunk.startFrame === session.endFrame,
                'Controller capture timeline discontinuity',
              );
              check(
                chunk.endFrame === chunk.startFrame + chunk.pcm.length,
                'Controller capture end frame mismatch',
              );
              check(chunk.pcm.every(Number.isFinite), 'Nonfinite controller PCM');
              session.endFrame = chunk.endFrame;
              session.chunks += 1;
              if (controller.getState().isMicActive) session.accepted.push(chunk.pcm.slice());
              options.onChunk(chunk);
              if (session.chunks >= 4) {
                clearTimeout(timer);
                session.ready.resolve();
              }
            } catch (error) {
              errors.push(String(error));
              session.ready.reject(error);
            }
          },
          onError(error) {
            errors.push(String(error));
            options.onError?.(error);
            session.ready.reject(error);
          },
        });
        acquired.resolve(session);
        if (handoff) await handoff.promise;
        return {
          ...session.handle,
          stop: () => {
            session.stopCalls += 1;
            return session.handle.stop();
          },
        };
      },
    });
    return { controller, acquired };
  }
  function verifyClosed(session) {
    check(session.workletLoads === 1, 'Controller did not use actual worklet');
    check(session.stopCalls === 1, 'Controller did not stop capture exactly once');
    check(session.context.state === 'closed', 'Controller left context open');
    check(
      session.handle.stream.getTracks().every((track) => track.readyState === 'ended'),
      'Controller left track live',
    );
  }
  const normal = createController();
  let pending;
  let handoff;
  try {
    await normal.controller.start();
    check(normal.controller.getState().isMicActive, 'Controller never became active');
    const first = sessions[0];
    await first.ready.promise;
    await normal.controller.stop({ flush: true });
    verifyClosed(first);
    check(
      utterances.length === 1 && utterances[0].reason === 'stop',
      'Stop did not flush exactly one utterance',
    );
    const expected = first.accepted.flatMap((chunk) => Array.from(chunk));
    check(
      expected.length > 0 && utterances[0].pcm.length === expected.length,
      'Flushed PCM length mismatch',
    );
    check(
      expected.every((sample, index) => sample === utterances[0].pcm[index]),
      'Flushed PCM content mismatch',
    );
    check(utterances[0].sampleRate === 16000, 'Flushed sample rate mismatch');
    await normal.controller.start();
    const second = sessions[1];
    await second.ready.promise;
    const disposing = normal.controller.dispose();
    check(!normal.controller.getState().isMicActive, 'Dispose did not immediately deactivate');
    await disposing;
    verifyClosed(second);
    check(utterances.length === 1, 'Dispose unexpectedly flushed captured audio');
    await normal.controller.dispose();
    const state = JSON.stringify(normal.controller.getState());
    normal.controller.flush();
    await new Promise((resolve) => setTimeout(resolve, 150));
    check(JSON.stringify(normal.controller.getState()) === state, 'State changed after disposal');
    check(utterances.length === 1, 'Late utterance after disposal');
    let rejected = false;
    try {
      await normal.controller.start();
    } catch (error) {
      rejected = /disposed/.test(String(error));
    }
    check(rejected, 'Start after disposal did not reject');

    handoff = deferred();
    pending = createController(handoff);
    const starting = pending.controller.start();
    const third = await pending.acquired.promise;
    await third.ready.promise;
    check(third.context.state === 'running', 'Pending handoff context was not live');
    const pendingDispose = pending.controller.dispose();
    check(!pending.controller.getState().isMicActive, 'Pending dispose activated mic');
    handoff.resolve();
    await Promise.all([starting, pendingDispose]);
    verifyClosed(third);
    check(
      third.accepted.length === 0 && utterances.length === 1,
      'Pending handoff published audio',
    );
    check(errors.length === 0, `Controller capture errors: ${errors.join('; ')}`);
    return {
      status: 'PASS',
      scope: 'REAL_BROWSER_FAKE_DEVICE_MANUAL_CONTROLLER',
      utterances: utterances.length,
      flushedPcmSamples: expected.length,
      restartPassed: true,
      startAfterDisposeRejected: rejected,
      pendingHandoff: 'CONTROLLED_REAL_HANDLE_DELIVERY_NOT_PERMISSION_DIALOG',
      runs: sessions.map((session, index) => ({
        index,
        chunks: session.chunks,
        acceptedChunks: session.accepted.length,
        stopCalls: session.stopCalls,
        workletLoads: session.workletLoads,
        contextState: session.context.state,
        trackStates: session.handle.stream.getTracks().map((track) => track.readyState),
      })),
    };
  } finally {
    handoff?.resolve();
    timers.forEach(clearTimeout);
    try {
      await normal.controller.dispose();
    } finally {
      try {
        await pending?.controller.dispose();
      } finally {
        // Resource-owner fallback only for a failing diagnostic, never a PASS shortcut.
        await Promise.all(sessions.map((session) => session.handle?.stop()));
      }
    }
  }
}
