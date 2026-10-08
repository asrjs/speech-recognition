import { TenVadAdapter } from '/dist/browser.js';
import { decodeWavToMonoPcm } from '/dist/browser-media.js';

function check(condition, message) {
  if (!condition) throw new Error(message);
}

async function sha(bytes) {
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, '0')).join('');
}

async function collect(adapter, pcm, schedule) {
  const hopSize = 256;
  const expected = Math.floor(pcm.length / hopSize);
  const probabilities = [],
    flags = [];
  let failure = null;
  const unsubscribe = adapter.subscribe((event) => {
    if (event.type !== 'result') return;
    const result = event.payload;
    if (result.globalSampleOffset !== probabilities.length * hopSize) {
      failure = 'Non-contiguous model output offsets';
    }
    for (let index = 0; index < result.hopCount; index += 1) {
      probabilities.push(result.probabilities[index]);
      flags.push(result.flags[index]);
    }
  });
  try {
    for (let offset = 0, chunk = 0; offset < pcm.length; chunk += 1) {
      const end = Math.min(pcm.length, offset + schedule[chunk % schedule.length]);
      check(adapter.process(pcm.subarray(offset, end), offset), 'Real VAD rejected PCM');
      offset = end;
    }
    const deadline = performance.now() + 5000;
    while (probabilities.length < expected && !failure && performance.now() < deadline) {
      check(adapter.getStatus().state === 'ready', JSON.stringify(adapter.getStatus()));
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    check(!failure, failure);
    check(probabilities.length === expected && expected > 0, 'Missing or extra inference hops');
    check(
      probabilities.every((p) => Number.isFinite(p) && p >= 0 && p <= 1),
      'Invalid probabilities',
    );
    check(
      flags.every((flag) => flag === 0 || flag === 1),
      'Invalid flags',
    );
    const values = new Float32Array(probabilities);
    return {
      hops: expected,
      probabilitySha256: await sha(values.buffer),
      flagsSha256: await sha(new Uint8Array(flags).buffer),
      minProbability: Math.min(...probabilities),
      maxProbability: Math.max(...probabilities),
      speechHops: flags.filter((flag) => flag === 1).length,
    };
  } finally {
    unsubscribe();
  }
}

export async function runTenVadLifecycle() {
  const workerResponses = [],
    workers = [];
  const NativeWorker = window.Worker;
  // Observe real default workers, without replacing inference or its messages.
  window.Worker = class extends NativeWorker {
    constructor(...args) {
      super(...args);
      workers.push(this);
      this.addEventListener('message', (event) => {
        if (event.data.type !== 'RESULT') workerResponses.push(event.data);
      });
    }
    terminate() {
      this.wasTerminated = true;
      super.terminate();
    }
  };
  const response = await fetch('/fixture.wav');
  check(response.ok, 'Fixture unavailable');
  const wav = await response.arrayBuffer();
  const { pcm, sampleRate } = decodeWavToMonoPcm(new Uint8Array(wav));
  check(sampleRate === 16000, 'This acceptance fixture requires 16 kHz PCM');
  const adapter = new TenVadAdapter({ hopSize: 256, fallbackToBundledAssets: false });
  const fresh = new TenVadAdapter({ hopSize: 256, fallbackToBundledAssets: false });
  const runs = [];
  const progress = { stage: 'init', runs };
  window.tenVadProgress = progress;
  try {
    await adapter.init();
    runs.push(await collect(adapter, pcm, [256]));
    check(runs[0].speechHops > 0, 'Speech fixture contains no model-detected speech');
    progress.stage = 'reset-after-speech';
    await adapter.reset();
    check(
      adapter.getStatus().probability === 0 && !adapter.getStatus().speaking,
      'Reset retained temporal state',
    );
    runs.push(await collect(adapter, pcm, [17, 511, 79, 4096]));
    check(
      JSON.stringify(runs[0]) === JSON.stringify(runs[1]),
      'Reset/chunking changed model output',
    );
    // Discard the incomplete speech tail before silence; compare with a fresh model.
    progress.stage = 'reset-before-silence';
    await adapter.reset();
    const silence = new Float32Array(16000);
    runs.push(await collect(adapter, silence, [73, 256, 511]));
    await fresh.init();
    runs.push(await collect(fresh, silence, [256]));
    check(
      JSON.stringify(runs[2]) === JSON.stringify(runs[3]),
      'Reset model state differs from fresh silence',
    );
    check(runs[2].speechHops === 0, 'Silence produced model speech flags');
    progress.stage = 'dispose';
    await adapter.dispose();
    await fresh.dispose();
    check(
      adapter.getStatus().state === 'idle' && !adapter.process(pcm, 0),
      'Disposed worker still accepts PCM',
    );
    check(
      workerResponses.filter((r) => r.type === 'DISPOSE' && r.payload.success === true).length ===
        2,
      'Missing actual worker disposal acknowledgements',
    );
    check(
      !workerResponses.some((r) => r.type === 'ERROR'),
      'Worker failure was hidden during disposal',
    );
    check(
      workers.length === 2 && workers.every((worker) => worker.wasTerminated),
      'Workers not terminated',
    );
    return {
      status: 'PASS_TEN_VAD_MODEL_LIFECYCLE',
      scope: 'real packaged TEN-VAD WASM worker reset/chunking',
      runs,
      sampleRate,
      samples: pcm.length,
      pcmSha256: await sha(pcm.buffer),
      disposed: true,
      workerResponses,
      terminatedWorkers: workers.length,
    };
  } finally {
    await adapter.dispose();
    await fresh.dispose();
    window.Worker = NativeWorker;
  }
}
