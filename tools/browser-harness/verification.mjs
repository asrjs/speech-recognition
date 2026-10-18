// Shared by the browser UI and CLI. Execution alone is never a parity PASS.
export const JFK_ORACLE = Object.freeze({
  fixture: 'jfk-short.wav',
  sha256: '59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e',
  text: 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.',
});

export function normalizeTranscript(text) {
  return String(text ?? '').normalize('NFC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}

export function validateOracle(oracle) {
  if (!oracle || typeof oracle.fixture !== 'string' || !oracle.fixture ||
      !/^[a-f\d]{64}$/i.test(oracle.sha256 ?? '') || !normalizeTranscript(oracle.text)) {
    throw new Error('Oracle requires fixture, SHA-256 of the audio file, and nonempty expected text.');
  }
  return oracle;
}

export function aggregateStatus(statuses) {
  if (statuses.length === 0 || statuses.includes('FAIL')) return 'FAIL';
  return statuses.every(status => status === 'PASS') ? 'PASS' : 'UNVERIFIED';
}

export function verificationExitCode(status) {
  return status === 'PASS' ? 0 : status === 'UNVERIFIED' ? 2 : 1;
}

export function verifyRun(run, { mode, backend = 'webgpu', oracle = null } = {}) {
  if (oracle) validateOracle(oracle);
  const failures = [];
  if (!run || run.mode !== mode) failures.push(`Expected ${mode} execution.`);
  if (!['EXECUTED', 'PASS'].includes(run?.status)) failures.push('Execution did not complete.');
  if (typeof run?.transcript !== 'string' || !run.transcript.trim()) failures.push('Transcript is empty.');
  if (!Number.isSafeInteger(run?.audio?.samples) || run.audio.samples <= 0 ||
      !Number.isFinite(run?.audio?.sampleRate) || run.audio.sampleRate <= 0) failures.push('Audio evidence is missing.');
  if (!Number.isFinite(run?.metrics?.inferenceMs) || run.metrics.inferenceMs < 0) failures.push('Inference timing is missing.');
  const execution = { status: failures.length ? 'FAIL' : 'PASS', failures };
  const fixtureMatches = oracle ? run?.audio?.fixture === oracle.fixture &&
    run?.audio?.sha256?.toLowerCase() === oracle.sha256.toLowerCase() : null;
  const matchExpected = oracle ? fixtureMatches &&
    normalizeTranscript(run?.transcript) === normalizeTranscript(oracle.text) : null;
  const transcript = {
    status: oracle ? (matchExpected ? 'PASS' : 'FAIL') : 'UNVERIFIED',
    comparison: 'normalized-text-exact', fixtureMatches, matchExpected,
  };
  const gpuRequested = backend === 'webgpu' || backend === 'webgpu-hybrid';
  const dispatches = run?.metrics?.gpuDispatches;
  const observed = run?.gpuObservation?.instrumented === true && Number.isInteger(dispatches) && dispatches > 0;
  const reported = run?.reportedBackendId ?? null;
  const providerMismatch = gpuRequested ? reported === 'wasm' : reported !== null && reported !== backend;
  const providerStatus = providerMismatch ? 'FAIL' : gpuRequested
    ? (run?.gpuObservation?.available === true && observed ? 'PASS' : 'UNVERIFIED')
    : (reported === backend ? 'PASS' : 'UNVERIFIED');
  const provider = {
    status: providerStatus, requested: backend, reported,
    // A dispatch proves some GPU execution, not that every graph uses WebGPU.
    evidence: gpuRequested ? (observed ? 'gpu-compute-observed' : 'gpu-compute-not-proven') : 'reported-backend',
  };
  return {
    status: aggregateStatus([execution.status, transcript.status, provider.status]),
    execution, transcript, provider,
  };
}

function leafRuns(run, mode) {
  return mode === 'both' ? [run?.offline, run?.streaming] : [run];
}

export function verifyRuns(cold, warm, spec) {
  const modes = spec.mode === 'both' ? ['offline', 'streaming'] : [spec.mode];
  if (!modes.every(mode => ['offline', 'streaming'].includes(mode))) throw new Error('Invalid execution mode.');
  const coldLeaves = leafRuns(cold, spec.mode);
  const warmLeaves = warm ? leafRuns(warm, spec.mode) : [];
  const runs = coldLeaves.map((run, index) => ({
    phase: 'cold', mode: modes[index], ...verifyRun(run, { ...spec, mode: modes[index] }),
  }));
  for (let index = 0; index < warmLeaves.length; index++) {
    runs.push({ phase: 'warm', mode: modes[index], ...verifyRun(warmLeaves[index], { ...spec, mode: modes[index] }) });
  }
  const warmSessionReused = warm ? warmLeaves.every((run, index) => run?.cached === true &&
    typeof run.instanceId === 'string' && run.instanceId.length > 0 &&
    run.instanceId === coldLeaves[index]?.instanceId) : null;
  const statuses = runs.map(run => run.status);
  if (warm && !warmSessionReused) statuses.push('FAIL');
  return { status: aggregateStatus(statuses), warmSessionReused, runs };
}
