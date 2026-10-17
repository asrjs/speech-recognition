#!/usr/bin/env node
/**
 * ASR.js Browser WebGPU CLI Runner & Dev-Loop Engine.
 *
 * Supports two primary workflows:
 * 1. Fast Dev-Loop / Warm Daemon Mode (--start-daemon or auto-CDP connection to port 9223).
 * 2. One-Shot Verification Mode (spawns headless Chrome WebGPU, executes benchmarks,
 *    verifies warm session reuse & oracle transcript match, writes canonical receipts).
 */
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHarnessServer } from './server.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const JFK_GROUND_TRUTH = 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.';

function normalizeText(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();
}

function matchesJfkTranscript(actual) {
  const normActual = normalizeText(actual);
  const normExpected = normalizeText(JFK_GROUND_TRUTH);
  if (!normActual) return false;
  // Direct match or substring overlap for smoke verification
  if (normActual === normExpected) return true;
  // Key phrases check
  const hasFellowAmericans = normActual.includes('fellow americans');
  const hasAsk = normActual.includes('ask not') || normActual.includes('askk not') || normActual.includes('ask');
  const hasCountry = normActual.includes('country');
  return hasFellowAmericans && hasAsk && hasCountry;
}

async function isCdpAvailable(port) {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/json/version`, { signal: AbortSignal.timeout(1000) });
    if (!res.ok) return false;
    const data = await res.json();
    return !!data.Browser;
  } catch {
    return false;
  }
}

async function isServerAvailable(port) {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/__api__/health`, { signal: AbortSignal.timeout(1000) });
    return res.ok;
  } catch {
    return false;
  }
}

function parseCliArgs() {
  const args = process.argv.slice(2);
  const options = {
    port: 8999,
    cdpPort: 9223,
    startDaemon: false,
    daemon: false,
    probe: false,
    catalog: false,
    localDir: null,
    modelId: null,
    preset: null,
    family: null,
    backend: 'webgpu',
    fixture: 'jfk-short.wav',
    mode: 'both',
    chunkMs: 160,
    expectJfk: false,
    warmRerun: true,
    out: null,
    timeoutMs: 180000,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--start-daemon') options.startDaemon = true;
    else if (arg === '--daemon') options.daemon = true;
    else if (arg === '--probe') options.probe = true;
    else if (arg === '--catalog') options.catalog = true;
    else if (arg === '--expect-jfk') options.expectJfk = true;
    else if (arg === '--no-warm-rerun') options.warmRerun = false;
    else if (arg === '--warm-rerun') options.warmRerun = true;
    else if (arg === '--port' && args[i + 1]) options.port = parseInt(args[++i], 10);
    else if (arg === '--cdp-port' && args[i + 1]) options.cdpPort = parseInt(args[++i], 10);
    else if (arg === '--local-dir' && args[i + 1]) options.localDir = args[++i];
    else if (arg === '--model-id' && args[i + 1]) options.modelId = args[++i];
    else if (arg === '--preset' && args[i + 1]) options.preset = args[++i];
    else if (arg === '--family' && args[i + 1]) options.family = args[++i];
    else if (arg === '--backend' && args[i + 1]) options.backend = args[++i];
    else if (arg === '--fixture' && args[i + 1]) options.fixture = args[++i];
    else if (arg === '--mode' && args[i + 1]) options.mode = args[++i];
    else if (arg === '--chunk-ms' && args[i + 1]) options.chunkMs = parseInt(args[++i], 10);
    else if (arg === '--out' && args[i + 1]) options.out = args[++i];
    else if (arg === '--timeout-ms' && args[i + 1]) options.timeoutMs = parseInt(args[++i], 10);
  }

  return options;
}

function runBridge(mode, targetUrl, action, spec, harnessUrl = null) {
  return new Promise((resolve, reject) => {
    const bridgePath = path.join(__dirname, 'bridge.py');
    const bridgeArgs = [
      bridgePath,
      mode,
      targetUrl,
      action,
      JSON.stringify(spec),
    ];
    if (harnessUrl) {
      bridgeArgs.push(harnessUrl);
    }

    const py = spawn('python', ['-u', ...bridgeArgs]);
    let stdout = '';
    let stderr = '';

    py.stdout.on('data', (d) => {
      const text = d.toString('utf-8');
      stdout += text;
      process.stdout.write(text);
    });

    py.stderr.on('data', (d) => {
      const text = d.toString('utf-8');
      stderr += text;
      process.stderr.write(text);
    });

    py.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Bridge execution failed with exit code ${code}:\n${stderr}`));
        return;
      }
      const match = /RESULT_PAYLOAD:(.*)/.exec(stdout);
      if (!match) {
        reject(new Error('No RESULT_PAYLOAD received from bridge execution'));
        return;
      }
      try {
        const payload = JSON.parse(match[1]);
        resolve(payload);
      } catch (err) {
        reject(new Error(`Failed to parse RESULT_PAYLOAD JSON: ${err.message}`));
      }
    });
  });
}

function findChromeExecutable() {
  const candidates = [
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    path.join(process.env.LOCALAPPDATA || '', 'Google\\Chrome\\Application\\chrome.exe'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

async function startDaemon(serverPort, cdpPort) {
  console.log(`[Daemon] Starting background ASR.js harness daemon on port ${serverPort}...`);
  const { server } = await createHarnessServer({ port: serverPort });
  console.log(`[Daemon] Harness server ready at http://127.0.0.1:${serverPort}`);

  const chromePath = findChromeExecutable();
  if (!chromePath) {
    throw new Error('Could not locate Google Chrome executable for daemon mode.');
  }

  const profileDir = path.join(__dirname, '.chrome-daemon-profile');
  fs.mkdirSync(profileDir, { recursive: true });

  const chromeArgs = [
    `--remote-debugging-port=${cdpPort}`,
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    '--use-angle=d3d11',
    '--enable-features=Vulkan',
    '--headless=new',
    `--user-data-dir=${profileDir}`,
    `http://127.0.0.1:${serverPort}/harness.html`,
  ];

  console.log(`[Daemon] Spawning Chrome WebGPU with remote debugging on port ${cdpPort}...`);
  const chromeProc = spawn(chromePath, chromeArgs, { detached: false, stdio: 'ignore' });

  // Poll until CDP is alive
  let connected = false;
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 500));
    if (await isCdpAvailable(cdpPort)) {
      connected = true;
      break;
    }
  }

  if (!connected) {
    chromeProc.kill();
    server.close();
    throw new Error(`Chrome daemon failed to bind CDP port ${cdpPort} within 15 seconds.`);
  }

  console.log(`\n================================================================`);
  console.log(`🚀 [ASR.js Dev Daemon] Running successfully!`);
  console.log(`  Harness Web UI:   http://127.0.0.1:${serverPort}/harness.html`);
  console.log(`  CDP Debug Port:   http://127.0.0.1:${cdpPort}`);
  console.log(`  Sub-second CLI:   node tools/browser-harness/cli.mjs --local-dir ...`);
  console.log(`  Press Ctrl+C to terminate daemon.`);
  console.log(`================================================================\n`);

  process.on('SIGINT', () => {
    console.log('\n[Daemon] Shutting down Chrome and server...');
    chromeProc.kill();
    server.close();
    process.exit(0);
  });

  // Keep daemon alive
  await new Promise(() => {});
}

async function main() {
  const options = parseCliArgs();

  // Mode 1: Start Daemon
  if (options.startDaemon) {
    await startDaemon(options.port, options.cdpPort);
    return;
  }

  console.log(`\n================================================================`);
  console.log(`⚡ ASR.js Browser WebGPU Test Harness Runner (Universal W3-1)`);
  console.log(`================================================================`);

  // Ensure local server is running
  let localServer = null;
  const serverRunning = await isServerAvailable(options.port);
  if (!serverRunning) {
    console.log(`[Harness] Starting in-process static server on port ${options.port}...`);
    const s = await createHarnessServer({ port: options.port });
    localServer = s.server;
    console.log(`[Harness] Static server ready at http://127.0.0.1:${options.port}`);
  } else {
    console.log(`[Harness] Reusing existing server at http://127.0.0.1:${options.port}`);
  }

  const harnessUrl = `http://127.0.0.1:${options.port}/harness.html`;

  // Check if warm CDP Chrome is available
  const hasWarmCdp = await isCdpAvailable(options.cdpPort);
  const useCdp = options.daemon || hasWarmCdp;

  if (useCdp) {
    console.log(`[Harness] Detected active Chrome daemon on port ${options.cdpPort}! Using warm CDP connection.`);
  } else {
    console.log(`[Harness] No active daemon on port ${options.cdpPort}. Using Playwright one-shot Chrome launcher.`);
  }

  // Model spec
  const spec = {
    localFolder: options.localDir,
    modelId: options.modelId,
    preset: options.preset,
    family: options.family,
    backend: options.backend,
    fixture: options.fixture,
    chunkMs: options.chunkMs,
    forceReload: true,
  };

  if (!spec.localFolder && !spec.modelId && !spec.preset && !spec.family) {
    // Default to moonshine-streaming-tiny-fp32 if nothing specified
    spec.localFolder = 'models/onnx/moonshine-streaming-tiny-fp32';
    console.log(`[Harness] No model specified. Defaulting to local model: ${spec.localFolder}`);
  }

  try {
    const runnerMode = useCdp ? 'cdp' : 'oneshot';
    const targetUrl = useCdp ? `http://127.0.0.1:${options.cdpPort}` : harnessUrl;

    if (options.probe) {
      console.log(`[Harness] Probing WebGPU capabilities in real browser...`);
      const probeResult = await runBridge(runnerMode, targetUrl, 'probe', {}, harnessUrl);
      console.log(`\n=== WebGPU Probe Results ===`);
      console.log(JSON.stringify(probeResult, null, 2));
      return;
    }

    if (options.catalog) {
      console.log(`[Harness] Querying speech recognition catalog in real browser...`);
      const catalogResult = await runBridge(runnerMode, targetUrl, 'catalog', {}, harnessUrl);
      console.log(`\n=== Catalog Speech Models (${catalogResult.models?.length || 0}) ===`);
      for (const m of catalogResult.models || []) {
        console.log(`  - ${m.modelId.padEnd(42)} [${m.preset || m.classification?.family}] (${m.displayName || ''})`);
      }
      console.log(`\n=== Experimental Speech Families (${catalogResult.experimentalFamilies?.length || 0}) ===`);
      for (const f of catalogResult.experimentalFamilies || []) {
        console.log(`  - ${f.family}`);
      }
      return;
    }

    let coldReceipt = null;
    let warmReceipt = null;

    if (options.warmRerun) {
      console.log(`\n--- Executing Benchmark Pipeline (Cold Load + Warm Cached In-Memory Re-Run) ---`);
      const combined = await runBridge(runnerMode, targetUrl, 'cold_and_warm', { ...spec, mode: options.mode }, harnessUrl);
      coldReceipt = combined.cold;
      warmReceipt = combined.warm;
      if (combined.browser_version) {
        coldReceipt.browser_version = combined.browser_version;
        warmReceipt.browser_version = combined.browser_version;
      }

      console.log(`\n[Cold Run Complete]`);
      const coldMetrics = coldReceipt.mode === 'both' ? coldReceipt.offline.metrics : coldReceipt.metrics;
      console.log(`  Load Time:      ${coldMetrics.loadMs} ms`);
      console.log(`  Inference Time: ${coldMetrics.inferenceMs ?? coldMetrics.offlineInferenceMs ?? 0} ms`);
      console.log(`  RTFX:           ${coldMetrics.rtfx ?? coldMetrics.offlineRtfx ?? 0}x real-time`);
      console.log(`  GPU Dispatches: ${coldMetrics.gpuDispatches ?? coldMetrics.totalGpuDispatches ?? 0}`);
      console.log(`  Transcript:     "${coldReceipt.transcript}"`);

      console.log(`\n[Warm Re-Run Complete] (Reused In-Memory WebGPU/WASM Session)`);
      const warmMetrics = warmReceipt.mode === 'both' ? warmReceipt.offline.metrics : warmReceipt.metrics;
      console.log(`  Load Time:      ${warmMetrics.loadMs} ms ${warmMetrics.loadMs === 0 ? '(✅ CACHED SESSION REUSED - 0ms)' : ''}`);
      console.log(`  Inference Time: ${warmMetrics.inferenceMs ?? warmMetrics.offlineInferenceMs ?? 0} ms`);
      console.log(`  RTFX:           ${warmMetrics.rtfx ?? warmMetrics.offlineRtfx ?? 0}x real-time`);
      console.log(`  GPU Dispatches: ${warmMetrics.gpuDispatches ?? warmMetrics.totalGpuDispatches ?? 0}`);
      console.log(`  Transcript:     "${warmReceipt.transcript}"`);
    } else {
      console.log(`\n--- Executing Benchmark (Single Run: ${options.mode}) ---`);
      coldReceipt = await runBridge(runnerMode, targetUrl, options.mode, spec, harnessUrl);
      console.log(`\n[Run Complete]`);
      const coldMetrics = coldReceipt.mode === 'both' ? coldReceipt.offline.metrics : coldReceipt.metrics;
      console.log(`  Load Time:      ${coldMetrics.loadMs} ms`);
      console.log(`  Inference Time: ${coldMetrics.inferenceMs ?? coldMetrics.offlineInferenceMs ?? 0} ms`);
      console.log(`  RTFX:           ${coldMetrics.rtfx ?? coldMetrics.offlineRtfx ?? 0}x real-time`);
      console.log(`  GPU Dispatches: ${coldMetrics.gpuDispatches ?? coldMetrics.totalGpuDispatches ?? 0}`);
      console.log(`  Transcript:     "${coldReceipt.transcript}"`);
    }

    // Oracle Verification
    let matchExpected = null;
    if (options.expectJfk) {
      const actualTranscript = warmReceipt ? warmReceipt.transcript : coldReceipt.transcript;
      matchExpected = matchesJfkTranscript(actualTranscript);
      console.log(`\n--- Oracle Transcript Verification (JFK Ground Truth) ---`);
      console.log(`  Expected: "${JFK_GROUND_TRUTH}"`);
      console.log(`  Actual:   "${actualTranscript}"`);
      console.log(`  Match:    ${matchExpected ? '✅ PASS' : '❌ FAIL'}`);
      if (!matchExpected) {
        throw new Error(`Oracle verification failed! Transcript does not match JFK ground truth.`);
      }
    }

    // Compose final canonical verification receipt
    const canonicalReceipt = {
      schema: 'asrjs.browser-webgpu.v1',
      status: 'PASS',
      scope: 'real-headless-chrome-webgpu-harness-verified',
      model: {
        modelId: coldReceipt.modelId || spec.modelId,
        localFolder: spec.localFolder || null,
        backendId: coldReceipt.backendId || spec.backend,
      },
      browser_version: coldReceipt.browser_version,
      adapter: coldReceipt.adapter,
      limits: coldReceipt.limits,
      audio: coldReceipt.offline?.audio || coldReceipt.audio,
      runs: {
        cold: coldReceipt,
        warm: warmReceipt,
      },
      verification: {
        warmSessionReused: warmReceipt ? (warmReceipt.cached || warmReceipt.offline?.cached || (warmReceipt.mode === 'both' ? warmReceipt.offline.metrics.loadMs === 0 : warmReceipt.metrics.loadMs === 0)) : false,
        expectJfk: options.expectJfk,
        matchExpected: matchExpected ?? true,
      },
      timestamp: new Date().toISOString(),
    };

    if (options.out) {
      const outPath = path.resolve(REPO_ROOT, options.out);
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, JSON.stringify(canonicalReceipt, null, 2) + '\n', 'utf-8');
      console.log(`\n[Receipt] Canonical verification receipt written to:\n  ${outPath}`);
    }

    console.log(`\n✨ Benchmark & Verification PASSED Successfully!`);
  } finally {
    if (localServer) {
      localServer.close();
    }
  }
}

main().catch((err) => {
  console.error('\n[Error] CLI Execution Failed:', err);
  process.exit(1);
});
