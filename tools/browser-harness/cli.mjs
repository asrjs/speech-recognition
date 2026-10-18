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
import { JFK_ORACLE, validateOracle, verifyRuns, verificationExitCode } from './verification.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');

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

export function parseCliArgs(args = process.argv.slice(2)) {
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
    oraclePath: null,
    warmRerun: true,
    matrix: false,
    localDirs: null,
    out: null,
    timeoutMs: 180000,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const flags = ['--start-daemon', '--daemon', '--probe', '--catalog', '--matrix', '--expect-jfk', '--no-warm-rerun', '--warm-rerun'];
    if (!flags.includes(arg) && (!args[i + 1] || args[i + 1].startsWith('--'))) throw new Error(`Missing value: ${arg}`);
    if (arg === '--start-daemon') options.startDaemon = true;
    else if (arg === '--daemon') options.daemon = true;
    else if (arg === '--probe') options.probe = true;
    else if (arg === '--catalog') options.catalog = true;
    else if (arg === '--matrix') options.matrix = true;
    else if (arg === '--local-dirs' && args[i + 1]) options.localDirs = args[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (arg === '--oracle' && args[i + 1]) options.oraclePath = args[++i];
    else if (arg === '--expect-jfk') options.expectJfk = true;
    else if (arg === '--no-warm-rerun') options.warmRerun = false;
    else if (arg === '--warm-rerun') options.warmRerun = true;
    else if (arg === '--port' && args[i + 1]) options.port = Number(args[++i]);
    else if (arg === '--cdp-port' && args[i + 1]) options.cdpPort = Number(args[++i]);
    else if (arg === '--local-dir' && args[i + 1]) options.localDir = args[++i];
    else if (arg === '--model-id' && args[i + 1]) options.modelId = args[++i];
    else if (arg === '--preset' && args[i + 1]) options.preset = args[++i];
    else if (arg === '--family' && args[i + 1]) options.family = args[++i];
    else if (arg === '--backend' && args[i + 1]) options.backend = args[++i];
    else if (arg === '--fixture' && args[i + 1]) options.fixture = args[++i];
    else if (arg === '--mode' && args[i + 1]) options.mode = args[++i];
    else if (arg === '--chunk-ms' && args[i + 1]) options.chunkMs = Number(args[++i]);
    else if (arg === '--out' && args[i + 1]) options.out = args[++i];
    else if (arg === '--timeout-ms' && args[i + 1]) options.timeoutMs = Number(args[++i]);
    else throw new Error(`Unknown option or missing value: ${arg}`);
  }

  if (options.matrix && !options.localDirs) {
    options.localDirs = [
      'models/onnx/parakeet-ctc-110m-en',
      'models/onnx/moonshine-streaming-tiny-fp32',
      'models/onnx/wav2vec2-xls-r-300m-tr',
    ];
  }

  if (!['offline', 'streaming', 'both'].includes(options.mode)) throw new Error('Invalid --mode');
  if (!['wasm', 'webgpu', 'webgpu-hybrid'].includes(options.backend)) throw new Error('Invalid --backend');
  for (const key of ['port', 'cdpPort', 'chunkMs', 'timeoutMs']) {
    if (!Number.isSafeInteger(options[key]) || options[key] <= 0) throw new Error(`Invalid ${key}: expected a positive integer`);
  }
  if (options.port > 65535 || options.cdpPort > 65535) throw new Error('Port exceeds 65535');
  if (options.localDirs && options.localDirs.length === 0) throw new Error('Empty --local-dirs');
  if (options.expectJfk && options.oraclePath) throw new Error('Use either --expect-jfk or --oracle');
  options.oracle = options.expectJfk ? JFK_ORACLE : options.oraclePath
    ? validateOracle(JSON.parse(fs.readFileSync(options.oraclePath, 'utf8'))) : null;
  return options;
}

export function runBridge(mode, targetUrl, action, spec, harnessUrl = null, timeoutMs = 180000) {
  return new Promise((resolve, reject) => {
    const bridgePath = path.join(__dirname, 'bridge.py');
    const bridgeArgs = [
      bridgePath,
      mode,
      targetUrl,
      action,
      JSON.stringify({ ...spec, timeoutMs }),
    ];
    if (harnessUrl) {
      bridgeArgs.push(harnessUrl);
    }

    const py = spawn('python', ['-u', ...bridgeArgs]);
    const timer = setTimeout(() => {
      py.kill();
      reject(new Error(`Bridge action timed out after ${timeoutMs}ms`));
    }, timeoutMs + 2000); // Allow the browser-side timeout to close the owned browser first.
    py.once('error', (error) => { clearTimeout(timer); reject(error); });
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
      clearTimeout(timer);
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
    mode: options.mode,
    oracle: options.oracle,
    warmRerun: options.warmRerun,
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
      const probeResult = await runBridge(runnerMode, targetUrl, 'probe', {}, harnessUrl, options.timeoutMs);
      console.log(`\n=== WebGPU Probe Results ===`);
      console.log(JSON.stringify(probeResult, null, 2));
      return;
    }

    if (options.catalog) {
      console.log(`[Harness] Querying speech recognition catalog in real browser...`);
      const catalogResult = await runBridge(runnerMode, targetUrl, 'catalog', {}, harnessUrl, options.timeoutMs);
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

    const requested = { ...spec };
    let receipt;
    try {
      if (options.localDirs) {
        receipt = await runBridge(runnerMode, targetUrl, 'matrix',
          { ...spec, dirs: options.localDirs }, harnessUrl, options.timeoutMs);
      } else {
        const pair = await runBridge(runnerMode, targetUrl, 'cold_and_warm', spec, harnessUrl, options.timeoutMs);
        const verification = verifyRuns(pair.cold, pair.warm, spec);
        receipt = {
          schema: 'asrjs.browser-verification.v2', status: verification.status,
          scope: 'public-api-execution-and-normalized-text', requested,
          browser_version: pair.browser_version,
          runs: { cold: pair.cold, warm: pair.warm }, verification,
          timestamp: new Date().toISOString(),
        };
      }
    } catch (error) {
      receipt = {
        schema: 'asrjs.browser-verification.v2', status: 'FAIL', requested,
        error: String(error.message ?? error), timestamp: new Date().toISOString(),
      };
    }
    const outPath = writeReceipt(receipt, options.out);
    console.log(`\n[Verification] ${receipt.status}. Receipt: ${outPath}`);
    for (const item of receipt.results ?? []) console.log(`  ${item.status}: ${item.folder} ${item.error ?? ''}`);
    process.exitCode = verificationExitCode(receipt.status);
  } finally {
    if (localServer) {
      localServer.close();
    }
  }
}

export function writeReceipt(receipt, destination) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outPath = path.resolve(REPO_ROOT, destination || `tools/data/results/browser-harness/${stamp}.json`);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(receipt, null, 2) + '\n', { encoding: 'utf8', flag: 'wx' });
  return outPath;
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  main().catch((err) => {
    console.error('\n[Error] CLI Execution Failed:', err);
    process.exitCode = 1;
  });
}
