import { aggregateStatus, verifyRuns } from './verification.mjs';

export async function runMode(harness, mode, spec) {
  if (mode === 'offline') return harness.runOffline(spec);
  if (mode === 'streaming') return harness.runStreaming(spec);
  if (mode !== 'both') throw new Error(`Unsupported execution mode: ${mode}`);
  const offline = await harness.runOffline(spec);
  const streaming = await harness.runStreaming({ ...spec, forceReload: false });
  return {
    schema: 'asrjs.browser-execution.v2', status: 'EXECUTED', mode,
    modelId: offline.modelId, backendId: offline.backendId,
    adapter: offline.adapter, limits: offline.limits,
    transcript: offline.transcript, offline, streaming,
    metrics: { ...offline.metrics, ttftMs: streaming.metrics.ttftMs },
  };
}

export async function runColdAndWarm(harness, spec) {
  const cold = await runMode(harness, spec.mode, { ...spec, forceReload: true });
  const warm = spec.warmRerun === false ? null : await runMode(harness, spec.mode, { ...spec, forceReload: false });
  return { cold, warm };
}

export async function runMatrix(harness, spec) {
  if (!Array.isArray(spec.dirs) || spec.dirs.length === 0) throw new Error('Matrix requires at least one model directory.');
  const results = [];
  for (const folder of spec.dirs) {
    const item = { folder, modelId: null, status: 'FAIL', cold: null, warm: null, error: null };
    try {
      const pair = await runColdAndWarm(harness, { ...spec, localFolder: folder, modelId: undefined });
      Object.assign(item, pair);
      item.verification = verifyRuns(pair.cold, pair.warm, spec);
      item.status = item.verification.status;
      const cold = spec.mode === 'both' ? pair.cold.offline : pair.cold;
      const warmRun = pair.warm ?? pair.cold;
      const warm = spec.mode === 'both' ? warmRun.offline : warmRun;
      item.modelId = cold.modelId;
      item.summaryPhase = pair.warm ? 'warm' : 'cold';
      item.transcript = warm.transcript;
      item.metrics = {
        coldLoadMs: cold.metrics.loadMs, warmLoadMs: pair.warm ? warm.metrics.loadMs : null,
        coldInferenceMs: cold.metrics.inferenceMs, warmInferenceMs: pair.warm ? warm.metrics.inferenceMs : null,
        rtfx: warm.metrics.rtfx, gpuDispatches: warm.metrics.gpuDispatches,
      };
    } catch (error) {
      item.error = String(error.message ?? error);
    } finally {
      try { await harness.disposeAll(); }
      catch (error) { item.status = 'FAIL'; item.disposalError = String(error.message ?? error); }
    }
    results.push(item);
  }
  const probe = await harness.probeWebGpu();
  return {
    schema: 'asrjs.browser-verification-matrix.v2', scope: 'public-api-execution-and-normalized-text',
    status: aggregateStatus(results.map(item => item.status)), mode: spec.mode,
    adapter: probe.adapter, limits: probe.limits, results, timestamp: new Date().toISOString(),
  };
}

export async function runAction(harness, action, spec) {
  if (action === 'probe') return harness.probeWebGpu();
  if (action === 'catalog') return harness.getCatalog();
  if (action === 'dispose') return harness.disposeAll();
  if (action === 'cold_and_warm') return runColdAndWarm(harness, spec);
  if (action === 'matrix') return runMatrix(harness, spec);
  return runMode(harness, action, spec);
}
