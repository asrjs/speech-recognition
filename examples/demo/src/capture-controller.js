/** Real capture -> public manual controller -> caller-owned ASR. No decoder here. */
import { createBrowserRealtimeMicrophoneController, startMicrophoneCapture } from '../../../src/browser.ts';

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

export async function runCapturedController({ receipt, transcribe, onCapturedPcm }) {
  receipt.scope = 'fake-device-manual-capture-utterance-asr';
  receipt.runs = [];
  receipt.disposed = false;
  const sessions = [], errors = [], utterances = [], pending = [];
  const controller = createBrowserRealtimeMicrophoneController({
    micMode: 'manual',
    onUtterance(utterance) {
      const entry = { pcm: utterance.pcm.slice(), sampleRate: utterance.sampleRate, reason: utterance.reason };
      utterances.push(entry);
      // onUtterance is synchronous; this consumer explicitly owns and awaits ASR.
      pending.push(Promise.resolve().then(async () => {
        entry.response = await transcribe(entry.pcm, entry.sampleRate);
      }).catch((error) => errors.push(String(error))));
    },
    startCapture: async (options) => {
      const session = { chunks: [], context: null, handle: null, workletLoads: 0, stopCalls: 0 };
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
  try {
    for (let iteration = 0; iteration < 2; iteration += 1) {
      const firstUtterance = utterances.length;
      await controller.start();
      const session = sessions[iteration];
      const deadline = performance.now() + 10000;
      while (session.chunks.length < 125 && !errors.length && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      require(session.chunks.length >= 125, 'Capture did not deliver two seconds');
      require(session.context.state === 'running' && session.handle.stream.getTracks().length > 0 &&
        session.handle.stream.getTracks().every((track) => track.readyState === 'live'), 'Capture was not live');
      await controller.stop({ flush: true });
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
  } finally {
    try { await controller.dispose(); } finally {
      await Promise.all(pending);
      await Promise.all(sessions.map((session) => session.handle?.stop()));
    }
  }
}
