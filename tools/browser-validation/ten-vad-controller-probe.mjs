import { createBrowserRealtimeMicrophoneController } from '/dist/browser.js';
import { decodeWavToMonoPcm } from '/dist/browser-media.js';

function check(condition, message) {
  if (!condition) throw new Error(message);
}

async function sha(bytes) {
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, '0')).join('');
}

export async function runTenVadController() {
  const response = await fetch('/fixture.wav');
  check(response.ok, 'Fixture unavailable');
  const { pcm, sampleRate } = decodeWavToMonoPcm(new Uint8Array(await response.arrayBuffer()));
  check(sampleRate === 16000, 'This fixture requires 16 kHz PCM');
  const workers = [],
    acknowledgements = [];
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(...args) {
      super(...args);
      workers.push(this);
      this.addEventListener('message', ({ data }) => {
        if (data.type !== 'RESULT') acknowledgements.push(data);
      });
    }
    terminate() {
      this.wasTerminated = true;
      super.terminate();
    }
  };
  let captureOptions;
  const handles = [],
    utterances = [],
    segments = [],
    errors = [],
    runs = [];
  // Only the PCM input boundary is injected. Default starter, detector, model
  // worker, probability buffer, monitor and utterance publication remain real.
  const controller = createBrowserRealtimeMicrophoneController({
    micMode: 'speech-detect',
    config: { vadBackend: 'ten-vad', tenVadEnabled: true, gateMode: 'rough-only' },
    tenVadConfig: { fallbackToBundledAssets: false },
    onUtterance: (utterance) => utterances.push(utterance),
    startCapture: async (options) => {
      captureOptions = options;
      const handle = {
        stream: new MediaStream(),
        sampleRate,
        deviceSampleRate: null,
        contextSampleRate: null,
        chunkFrames: 256,
        chunkDurationMs: 16,
        stopCalls: 0,
        async stop() {
          this.stopCalls += 1;
        },
      };
      handles.push(handle);
      return handle;
    },
  });
  const progress = { stage: 'init', runs };
  window.tenVadProgress = progress;
  const unsubscribe = controller.starter.subscribe((event) => {
    if (event.type === 'error') errors.push(String(event.payload));
    if (event.type === 'segment-ready') segments.push(event.payload);
  });
  try {
    for (let iteration = 0; iteration < 3; iteration += 1) {
      progress.stage = `run-${iteration}`;
      const input = iteration < 2 ? pcm : new Float32Array(16000);
      const firstSegment = segments.length,
        firstUtterance = utterances.length;
      await controller.start();
      check(controller.getState().isMicActive, 'Controller did not become active');
      const probabilities = [],
        flags = [];
      let offsetFailure = false;
      const unsubscribeModel = controller.starter.tenVad.subscribe((event) => {
        if (event.type !== 'result') return;
        offsetFailure ||=
          event.payload.globalSampleOffset !== probabilities.length * 256 ||
          event.payload.hopCount !== event.payload.probabilities.length ||
          event.payload.hopCount !== event.payload.flags.length;
        probabilities.push(...event.payload.probabilities);
        flags.push(...event.payload.flags);
      });
      try {
        check(controller.starter.getSnapshot().tenVad.state === 'ready', 'Model not ready');
        for (let startFrame = 0; startFrame < input.length; startFrame += 256) {
          const endFrame = Math.min(startFrame + 256, input.length);
          captureOptions.onChunk({
            pcm: input.subarray(startFrame, endFrame),
            sampleRate,
            startFrame,
            endFrame,
            startTimeSeconds: startFrame / sampleRate,
            endTimeSeconds: endFrame / sampleRate,
          });
        }
        const expectedHops = Math.floor(input.length / 256);
        const deadline = performance.now() + 5000;
        while (probabilities.length < expectedHops && performance.now() < deadline) {
          check(controller.starter.getSnapshot().tenVad.state === 'ready', 'Model degraded');
          await new Promise((resolve) => setTimeout(resolve, 5));
        }
        check(
          !offsetFailure && probabilities.length === expectedHops && flags.length === expectedHops,
          'Incomplete model timeline',
        );
        check(
          probabilities.every((p) => Number.isFinite(p) && p >= 0 && p <= 1),
          'Invalid model probability',
        );
        check(
          flags.every((flag) => flag === 0 || flag === 1),
          'Invalid model flag',
        );
        const snapshot = controller.starter.getSnapshot();
        check(snapshot.gate.effectiveMode === 'rough-only', 'Unexpected segmentation contract');
        check(
          snapshot.vadBuffer.timeline.some((point) => point.probability > 0),
          'Model results absent from public buffer',
        );
        const buffer = controller.starter.vadBuffer;
        check(
          buffer.hopFrames === 256 && buffer.getLatestFrame() === expectedHops * 256,
          'Incorrect public buffer timeline',
        );
        const buffered = buffer.getWindowSummary(
          buffer.getLatestFrame(),
          (input.length / sampleRate) * 1000,
        ).recent;
        check(
          buffered.length === expectedHops &&
            buffered.every(
              (point, index) =>
                point.startFrame === index * 256 &&
                point.endFrame === (index + 1) * 256 &&
                point.probability === probabilities[index],
            ),
          'Public buffer changed model results',
        );
        await controller.stop({ flush: true });
        check(handles[iteration].stopCalls === 1, 'Input handle not stopped once');
        check(
          !controller.getState().isMicActive && !controller.getState().micError,
          'Controller failed stop',
        );
        const currentSegments = segments.slice(firstSegment);
        const currentUtterances = utterances.slice(firstUtterance);
        check(
          currentSegments.length === currentUtterances.length,
          'Segment/utterance delivery mismatch',
        );
        check(
          iteration < 2 ? currentSegments.length > 0 : currentSegments.length === 0,
          'Unexpected speech/silence segmentation',
        );
        const windows = [];
        for (let index = 0; index < currentSegments.length; index += 1) {
          const segment = currentSegments[index],
            utterance = currentUtterances[index];
          check(
            segment.startFrame >= 0 &&
              segment.endFrame <= input.length &&
              segment.endFrame > segment.startFrame,
            'Invalid segment bounds',
          );
          check(
            segment.metadata.source === 'rough-fallback' &&
              segment.metadata.tenVad.state === 'ready',
            'Incorrect segment provenance',
          );
          const expected = input.slice(segment.startFrame, segment.endFrame);
          check(
            utterance.sampleRate === sampleRate && utterance.reason === segment.reason,
            'Utterance contract mismatch',
          );
          check(
            utterance.durationSeconds === expected.length / sampleRate,
            'Utterance duration mismatch',
          );
          check(
            utterance.pcm.length === expected.length &&
              expected.every((value, i) => value === utterance.pcm[i]),
            'Utterance PCM identity mismatch',
          );
          check(
            segment.readPcm().length === expected.length &&
              expected.every((value, i) => value === segment.readPcm()[i]),
            'Segment PCM identity mismatch',
          );
          windows.push({
            startFrame: segment.startFrame,
            endFrame: segment.endFrame,
            reason: segment.reason,
            source: segment.metadata.source,
            pcmSha256: await sha(expected.buffer),
          });
        }
        const stoppedUtterances = utterances.length;
        captureOptions.onChunk({
          pcm: input.subarray(0, 256),
          sampleRate,
          startFrame: 0,
          endFrame: 256,
        });
        check(
          utterances.length === stoppedUtterances &&
            controller.starter.vadBuffer.getLatestFrame() === expectedHops * 256,
          'Stale input accepted after stop',
        );
        runs.push({
          iteration,
          samples: input.length,
          hops: expectedHops,
          probabilitySha256: await sha(new Float32Array(probabilities).buffer),
          bufferProbabilitySha256: await sha(
            new Float32Array(buffered.map((point) => point.probability)).buffer,
          ),
          flagsSha256: await sha(new Uint8Array(flags).buffer),
          speechHops: flags.filter((flag) => flag === 1).length,
          windows,
          gate: snapshot.gate,
          warnings: snapshot.warnings,
        });
      } finally {
        unsubscribeModel();
      }
    }
    check(
      JSON.stringify(runs[0].windows) === JSON.stringify(runs[1].windows),
      'Restart changed segment PCM',
    );
    check(
      runs[0].probabilitySha256 === runs[1].probabilitySha256 &&
        runs[0].flagsSha256 === runs[1].flagsSha256,
      'Restart changed actual model outputs',
    );
    check(runs[2].speechHops === 0, 'Reset silence retained speech flags');
    progress.stage = 'dispose';
    await controller.dispose();
    await controller.dispose();
    let rejected = false;
    try {
      await controller.start();
    } catch (error) {
      rejected = /disposed/.test(String(error));
    }
    check(rejected, 'Controller restarted after disposal');
    check(
      errors.length === 0 && !acknowledgements.some((r) => r.type === 'ERROR'),
      'Model/controller error',
    );
    check(
      acknowledgements.filter((r) => r.type === 'DISPOSE' && r.payload.success).length === 1,
      'Missing model disposal acknowledgement',
    );
    check(
      workers.length === 1 && workers.every((worker) => worker.wasTerminated),
      'Owned worker not terminated',
    );
    return {
      status: 'PASS_TEN_VAD_CONTROLLER_PCM',
      scope:
        'PUBLIC_SPEECH_DETECT_CONTROLLER_INJECTED_FILE_PCM_REAL_TEN_DIAGNOSTICS_ROUGH_SEGMENTS',
      runs,
      pcmSha256: await sha(pcm.buffer),
      inputBoundary: 'INJECTED_IDENTIFIED_FILE_PCM_NOT_MICROPHONE_CAPTURE',
      disposed: true,
      terminatedWorkers: workers.length,
      workerResponses: acknowledgements,
      inputStopCalls: handles.map((handle) => handle.stopCalls),
      startAfterDisposeRejected: rejected,
    };
  } finally {
    unsubscribe();
    try {
      await controller.dispose();
    } finally {
      window.Worker = NativeWorker;
    }
  }
}
