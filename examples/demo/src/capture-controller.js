/** Real capture -> public microphone controller -> caller-owned ASR. No decoder here. */

function require(condition, message) {
  if (!condition) throw new Error(message);
}

async function pcmSha256(pcm) {
  const bytes = new ArrayBuffer(pcm.length * 4);
  const view = new DataView(bytes);
  pcm.forEach((sample, index) => view.setFloat32(index * 4, sample, true));
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/** Observe native worker construction without replacing the library's factory. */
export function observeCaptureWorkers(expectedUrl) {
  const NativeWorker = globalThis.Worker;
  const target = new URL(expectedUrl, location.href);
  const workers = [], acknowledgements = [];
  class ObservedWorker extends NativeWorker {
    constructor(url, options) {
      super(url, options);
      const actual = new URL(url, location.href);
      if (actual.origin !== target.origin || actual.pathname !== target.pathname) return;
      const owned = { worker: this, url: actual.href, terminated: false };
      workers.push(owned);
      this.addEventListener('message', ({ data }) => {
        if (data.type !== 'RESULT') acknowledgements.push(data);
      });
      const terminate = this.terminate.bind(this);
      this.terminate = () => { owned.terminated = true; terminate(); };
    }
  }
  globalThis.Worker = ObservedWorker;
  return { workers, acknowledgements, restore() {
    require(globalThis.Worker === ObservedWorker || globalThis.Worker === NativeWorker,
      'Worker observer ownership changed');
    globalThis.Worker = NativeWorker;
  } };
}

export async function runCapturedController({ receipt, transcribe, onCapturedPcm, micMode = 'manual',
  workerUrl, library, workerObserver }) {
  require(micMode === 'manual' || micMode === 'speech-detect', 'Unsupported capture mode');
  const { createBrowserRealtimeMicrophoneController, startMicrophoneCapture } =
    library ?? await import('../../../src/browser.ts');
  require(typeof createBrowserRealtimeMicrophoneController === 'function' && typeof startMicrophoneCapture === 'function',
    'Public capture API required');
  const speechDetect = micMode === 'speech-detect';
  require(!speechDetect || workerUrl || workerObserver, 'TEN worker URL or default-worker observer required');
  require(!workerObserver || (!workerUrl && Array.isArray(workerObserver.workers) &&
    Array.isArray(workerObserver.acknowledgements)), 'Invalid default-worker observer');
  receipt.scope = `fake-device-${micMode}-capture-utterance-asr`;
  receipt.runs = [];
  receipt.disposed = false;
  const sessions = [], errors = [], utterances = [], pending = [];
  const workers = workerObserver?.workers ?? [], acknowledgements = workerObserver?.acknowledgements ?? [];
  let asrTail = Promise.resolve();
  const controller = createBrowserRealtimeMicrophoneController({
    micMode,
    ...(speechDetect ? {
      config: { vadBackend: 'ten-vad', tenVadEnabled: true, gateMode: 'rough-only' },
      tenVadConfig: { fallbackToBundledAssets: false },
      ...(workerUrl ? { tenVadOptions: { workerFactory() {
        // Supported injection: a real Vite-transformed TS worker, not a mock.
        const worker = new Worker(workerUrl, { type: 'module' });
        const owned = { worker, terminated: false };
        workers.push(owned);
        worker.addEventListener('message', ({ data }) => {
          if (data.type !== 'RESULT') acknowledgements.push(data);
        });
        const terminate = worker.terminate.bind(worker);
        worker.terminate = () => { owned.terminated = true; terminate(); };
        return worker;
      } } } : {}),
    } : {}),
    onUtterance(utterance) {
      const entry = { pcm: utterance.pcm.slice(), sampleRate: utterance.sampleRate, reason: utterance.reason,
        beforeStop: !sessions.at(-1)?.stopRequested };
      utterances.push(entry);
      // onUtterance is synchronous; this consumer explicitly owns and awaits ASR.
      // Serialize calls to the caller's model; capture keeps running while ASR awaits.
      asrTail = asrTail.then(async () => {
        entry.response = await transcribe(entry.pcm, entry.sampleRate);
      }).catch((error) => errors.push(String(error)));
      pending.push(asrTail);
    },
    startCapture: async (options) => {
      const session = { chunks: [], context: null, handle: null, workletLoads: 0, stopCalls: 0,
        segments: [], probabilities: [], flags: [], offsetFailure: false, stopRequested: false };
      sessions.push(session);
      session.handle = await startMicrophoneCapture({
        ...options,
        createAudioContext(sampleRate) {
          session.context = new AudioContext({ sampleRate });
          const addModule = session.context.audioWorklet.addModule.bind(session.context.audioWorklet);
          session.context.audioWorklet.addModule = async (...args) => {
            await addModule(...args);
            session.workletLoads += 1;
          };
          return session.context;
        },
        onChunk(chunk) {
          session.chunks.push({ ...chunk, pcm: chunk.pcm.slice() });
          options.onChunk(chunk);
        },
        onError(error) {
          errors.push(String(error));
          options.onError?.(error);
        },
      });
      return { ...session.handle, stop: async () => {
        session.stopCalls += 1;
        await session.handle.stop();
      } };
    },
  });
  const unsubscribeSegments = controller.starter.subscribe((event) => {
    const session = sessions.at(-1);
    if (event.type === 'error') errors.push(String(event.payload));
    if (event.type === 'segment-ready' && session) {
      session.segments.push({ ...event.payload, pcm: event.payload.readPcm().slice() });
    }
  });
  const unsubscribeModel = speechDetect ? controller.starter.tenVad.subscribe((event) => {
    const session = sessions.at(-1);
    if (event.type !== 'result' || !session) return;
    const result = event.payload;
    session.offsetFailure ||= result.globalSampleOffset !== session.probabilities.length * 256 ||
      result.hopCount !== result.probabilities.length || result.hopCount !== result.flags.length;
    session.probabilities.push(...result.probabilities);
    session.flags.push(...result.flags);
  }) : () => {};
  try {
    for (let iteration = 0; iteration < 2; iteration += 1) {
      const firstUtterance = utterances.length;
      await controller.start();
      const session = sessions[iteration];
      const targetChunks = speechDetect ? 625 : 125;
      const deadline = performance.now() + (speechDetect ? 20000 : 10000);
      while ((session.chunks.length < targetChunks || (speechDetect &&
        !utterances.slice(firstUtterance).some((entry) => entry.reason === 'silence' && entry.beforeStop))) &&
        !errors.length && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      require(session.chunks.length >= targetChunks, 'Capture did not deliver required duration');
      require(session.context.state === 'running' && session.handle.stream.getTracks().length > 0 &&
        session.handle.stream.getTracks().every((track) => track.readyState === 'live'), 'Capture was not live');
      session.stopRequested = true;
      await controller.stop({ flush: true });
      if (speechDetect) {
        const deadline = performance.now() + 5000;
        while (session.probabilities.length < session.chunks.length && performance.now() < deadline && !errors.length) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      }
      await Promise.all(pending);
      require(!errors.length && !controller.getState().micError, errors.join('; ') || controller.getState().micError);
      const count = session.chunks.length;
      await new Promise((resolve) => setTimeout(resolve, 150));
      require(session.chunks.length === count, 'Capture continued after stop');
      const all = new Float32Array(session.chunks.reduce((total, chunk) => total + chunk.pcm.length, 0));
      let offset = 0;
      for (const chunk of session.chunks) {
        require(chunk.sampleRate === 16000 && chunk.startFrame === offset &&
          chunk.endFrame === offset + chunk.pcm.length && chunk.pcm.every(Number.isFinite), 'Capture timeline/PCM mismatch');
        all.set(chunk.pcm, offset);
        offset += chunk.pcm.length;
      }
      require(all.some((sample) => sample !== 0), 'Capture is silent');
      if (speechDetect) {
        const current = utterances.slice(firstUtterance), windows = [];
        require(current.some((entry) => entry.reason === 'silence' && entry.beforeStop), 'No natural pause before stop');
        require(current.length === session.segments.length, 'Segment/utterance count mismatch');
        for (let index = 0; index < current.length; index += 1) {
          const utterance = current[index], segment = session.segments[index];
          require(Number.isSafeInteger(segment.startFrame) && Number.isSafeInteger(segment.endFrame) &&
            segment.startFrame >= 0 && segment.endFrame <= all.length && segment.endFrame > segment.startFrame,
          'Invalid extracted segment bounds');
          const expected = all.slice(segment.startFrame, segment.endFrame);
          require(utterance.sampleRate === 16000 && utterance.reason === segment.reason &&
            expected.length === utterance.pcm.length && expected.length === segment.pcm.length &&
            expected.every((sample, i) => Object.is(sample, utterance.pcm[i]) && Object.is(sample, segment.pcm[i])),
          'Speech segment changed captured PCM');
          require(segment.metadata.source === 'rough-fallback' && segment.metadata.tenVad.state === 'ready',
            'Unexpected segmentation provenance');
          const response = utterance.response;
          require(response && typeof response.text === 'string' && response.text.trim() &&
            Array.isArray(response.tokenIds) && response.tokenIds.length > 0, 'Speech ASR missing output');
          const replay = await transcribe(expected, 16000);
          require(response.text === replay.text && JSON.stringify(response.tokenIds) === JSON.stringify(replay.tokenIds),
            'Speech ASR differs from same-PCM public replay');
          windows.push({ startFrame: segment.startFrame, endFrame: segment.endFrame, sampleRate: 16000,
            samples: expected.length, reason: segment.reason, beforeStop: utterance.beforeStop,
            source: segment.metadata.source, tenVadState: segment.metadata.tenVad.state,
            capturedSliceSha256: await pcmSha256(expected), segmentPcmSha256: await pcmSha256(segment.pcm),
            utterancePcmSha256: await pcmSha256(utterance.pcm), pcmExact: true,
            text: response.text, tokenIds: response.tokenIds, metrics: response.metrics,
            replayText: replay.text, replayTokenIds: replay.tokenIds, replayMetrics: replay.metrics });
        }
        const snapshot = controller.starter.getSnapshot(), hops = session.probabilities.length;
        require(!session.offsetFailure && hops === session.chunks.length && session.flags.length === hops &&
          session.probabilities.every((p) => Number.isFinite(p) && p >= 0 && p <= 1) &&
          session.flags.every((flag) => flag === 0 || flag === 1), 'Incomplete/invalid TEN diagnostic timeline');
        const buffer = controller.starter.vadBuffer;
        const buffered = buffer.getWindowSummary(buffer.getLatestFrame(), all.length / 16).recent;
        require(buffer.getLatestFrame() === all.length && buffered.length === hops &&
          buffered.every((point, i) => point.startFrame === i * 256 && point.endFrame === (i + 1) * 256 &&
            point.probability === session.probabilities[i]), 'TEN public buffer changed model results');
        require(snapshot.tenVad.state === 'ready' && snapshot.gate.effectiveMode === 'rough-only', 'TEN degraded/gate changed');
        require(session.context.state === 'closed' && session.workletLoads === 1 && session.stopCalls === 1 &&
          session.handle.stream.getTracks().every((track) => track.readyState === 'ended'), 'Capture cleanup failed');
        receipt.runs.push({ iteration, capturedSamples: all.length, sampleRate: 16000,
          chunkCount: session.chunks.length, capturedPcmSha256: await pcmSha256(all), utterances: windows,
          diagnostics: { state: snapshot.tenVad.state, effectiveGate: snapshot.gate.effectiveMode, hops,
            offsetExact: true, probabilitySha256: await pcmSha256(new Float32Array(session.probabilities)),
            bufferProbabilitySha256: await pcmSha256(new Float32Array(buffered.map((point) => point.probability))),
            speechHops: session.flags.filter((flag) => flag === 1).length },
          contextState: session.context.state, trackStates: session.handle.stream.getTracks().map((track) => track.readyState),
          workletLoads: session.workletLoads, stopCalls: session.stopCalls });
        await onCapturedPcm?.(all.slice(), 16000, iteration);
        continue;
      }
      require(utterances.length === firstUtterance + 1, 'Manual stop must publish one utterance');
      const utterance = utterances[firstUtterance];
      require(utterance.sampleRate === 16000 && utterance.reason === 'stop' &&
        utterance.pcm.length === all.length && all.every((sample, index) => sample === utterance.pcm[index]),
      'Utterance lost or changed actual captured PCM');
      const response = utterance.response;
      require(response && typeof response.text === 'string' && response.text.trim() &&
        Array.isArray(response.tokenIds) && response.tokenIds.length > 0, 'Actual capture ASR missing output');
      // Same captured PCM, independent public call. Not an original-engine oracle.
      const replay = await transcribe(all, 16000);
      require(response.text === replay.text && JSON.stringify(response.tokenIds) === JSON.stringify(replay.tokenIds),
        'Capture consumer ASR differs from same-PCM public replay');
      require(session.context.state === 'closed' && session.workletLoads === 1 && session.stopCalls === 1 &&
        session.handle.stream.getTracks().every((track) => track.readyState === 'ended'), 'Capture cleanup failed');
      receipt.runs.push({ iteration, capturedSamples: all.length, sampleRate: 16000,
        chunkCount: session.chunks.length, capturedPcmSha256: await pcmSha256(all),
        utteranceSamples: utterance.pcm.length, utterancePcmSha256: await pcmSha256(utterance.pcm),
        pcmExact: true, text: response.text, tokenIds: response.tokenIds, metrics: response.metrics,
        replayText: replay.text, replayTokenIds: replay.tokenIds, replayMetrics: replay.metrics,
        contextState: session.context.state, trackStates: session.handle.stream.getTracks().map((track) => track.readyState),
        workletLoads: session.workletLoads, stopCalls: session.stopCalls });
      // Optional evidence consumer owns this copy; never quantize or resample it.
      await onCapturedPcm?.(all.slice(), 16000, iteration);
    }
    await controller.dispose();
    receipt.disposed = true;
    const count = utterances.length;
    await controller.dispose();
    require(utterances.length === count && !controller.getState().isMicActive, 'Disposal published extra audio');
    let rejected = false;
    try { await controller.start(); } catch { rejected = true; }
    require(rejected, 'Disposed microphone controller restarted');
    receipt.restartRejected = true;
    if (speechDetect) {
      require(workers.length === 1 && workers.every((owned) => owned.terminated) &&
        acknowledgements.filter((event) => event.type === 'DISPOSE' && event.payload.success).length === 1 &&
        !acknowledgements.some((event) => event.type === 'ERROR'), 'TEN worker cleanup failed');
      receipt.terminatedWorkers = workers.length;
      receipt.workerResponses = acknowledgements;
      receipt.workerUrl = workerUrl;
      if (workerObserver) {
        receipt.workerFactory = 'library-default';
        receipt.workerUrls = workers.map((owned) => owned.url);
      }
    }
  } finally {
    try { await controller.dispose(); } finally {
      await Promise.all(pending);
      await Promise.all(sessions.map((session) => session.handle?.stop()));
      unsubscribeSegments();
      unsubscribeModel();
    }
  }
}
