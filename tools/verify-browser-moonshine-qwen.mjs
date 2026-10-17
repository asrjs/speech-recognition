#!/usr/bin/env node
/**
 * Canonical browser verification runner for Moonshine Streaming & Qwen3-ASR 0.6B fp16.
 *
 * Drives real headless Chrome with WebGPU enabled to execute both models and
 * emits canonical verification receipts under tools/data/results/canonical/.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const CANONICAL_DIR = path.join(ROOT, 'tools', 'data', 'results', 'canonical');
const MOONSHINE_RECEIPT_PATH = path.join(CANONICAL_DIR, 'moonshine-streaming-browser-verify.json');
const QWEN_RECEIPT_PATH = path.join(CANONICAL_DIR, 'qwen3-asr-fp16-browser-verify.json');

const MIME_MAP = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.data': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.npy': 'application/octet-stream',
  '.wav': 'audio/wav',
};

function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  const buffer = fs.readFileSync(filePath);
  hash.update(buffer);
  return hash.digest('hex');
}

function getFileArtifacts(fileList) {
  return fileList.map((relPath) => {
    const fullPath = path.join(ROOT, relPath);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Missing expected artifact: ${fullPath}`);
    }
    const stat = fs.statSync(fullPath);
    return {
      path: fullPath,
      relativePath: relPath.replace(/\\/g, '/'),
      bytes: stat.size,
      sha256: sha256File(fullPath),
    };
  });
}

const MOONSHINE_INPUT_FILES = [
  '.cache/moonshine-streaming-onnx/encoder_chunked_model.onnx',
  '.cache/moonshine-streaming-onnx/encoder_model.onnx',
  '.cache/moonshine-streaming-onnx/encoder_model.onnx.data',
  '.cache/moonshine-streaming-onnx/decoder_prefill_model.onnx',
  '.cache/moonshine-streaming-onnx/decoder_step_model.onnx',
  '.cache/moonshine-streaming-tiny/tokenizer.json',
  'tools/data/fixtures/audio/jfk-short.wav',
];

const QWEN_INPUT_FILES = [
  'models/onnx/qwen3-asr-0.6b-fp16/audio_front_model.onnx',
  'models/onnx/qwen3-asr-0.6b-fp16/audio_front_model.onnx.data',
  'models/onnx/qwen3-asr-0.6b-fp16/decoder_step_model.onnx',
  'models/onnx/qwen3-asr-0.6b-fp16/decoder_step_model.onnx.data',
  'models/onnx/qwen3-asr-0.6b-fp16/embed_tokens_model.onnx',
  'models/onnx/qwen3-asr-0.6b-fp16/embed_tokens_model.onnx.data',
  'models/onnx/qwen3-asr-0.6b-fp16/tokenizer.json',
  'models/onnx/qwen3-asr-0.6b-fp16/jfk_features.npy',
  'models/onnx/qwen3-asr-0.6b-fp16/jfk_features_mask.npy',
  'models/onnx/qwen3-asr-0.6b-fp16/jfk_cu_seqlens.bin',
  'models/onnx/qwen3-asr-0.6b-fp16/prompt_prefix_embeds.npy',
  'models/onnx/qwen3-asr-0.6b-fp16/prompt_suffix_embeds.npy',
];

const MOONSHINE_EXPECTED_TEXT =
  'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.';

const QWEN_EXPECTED_TEXT =
  'And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.';

const MOONSHINE_HTML = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Moonshine Streaming Verification</title>
  <script src="/node_modules/onnxruntime-web/dist/ort.all.min.js"></script>
</head>
<body>
  <h1>Moonshine Streaming Browser Verification</h1>
  <pre id="log"></pre>
  <script type="module">
    import { MoonshineStreamingSession, embeddedFrameCount } from '/dist/models/moonshine/streaming-session.js';
    import { MoonshineTokenizer } from '/dist/models/moonshine/session.js';
    import { decodeWavToMonoPcm } from '/dist/browser-media.js';

    window.testResult = null;
    const log = (msg) => console.log('[Moonshine]', msg);

    async function run() {
      try {
        if (!navigator.gpu) throw new Error('navigator.gpu not available');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('requestAdapter returned null');
        const info = await (adapter.requestAdapterInfo ? adapter.requestAdapterInfo() : adapter.info);

        ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
        ort.env.wasm.numThreads = 1;

        const [tokRes, wavRes] = await Promise.all([
          fetch('/.cache/moonshine-streaming-tiny/tokenizer.json'),
          fetch('/tools/data/fixtures/audio/jfk-short.wav')
        ]);
        const tokJson = await tokRes.json();
        const tokenizer = MoonshineTokenizer.fromTokenizerJson(tokJson, 0);

        const wavBytes = await wavRes.arrayBuffer();
        const { pcm } = decodeWavToMonoPcm(wavBytes);

        const ep = ['webgpu', 'wasm'];
        const sessionOpts = {
          executionProviders: ep,
          graphOptimizationLevel: 'basic',
        };

        const t0 = performance.now();
        const chunkedEncoder = await ort.InferenceSession.create(
          '/.cache/moonshine-streaming-onnx/encoder_chunked_model.onnx',
          sessionOpts
        );
        const wholeEncoder = await ort.InferenceSession.create(
          '/.cache/moonshine-streaming-onnx/encoder_model.onnx',
          {
            ...sessionOpts,
            externalData: [{
              path: 'encoder_model.onnx.data',
              data: '/.cache/moonshine-streaming-onnx/encoder_model.onnx.data'
            }]
          }
        );
        const prefill = await ort.InferenceSession.create(
          '/.cache/moonshine-streaming-onnx/decoder_prefill_model.onnx',
          sessionOpts
        );
        const step = await ort.InferenceSession.create(
          '/.cache/moonshine-streaming-onnx/decoder_step_model.onnx',
          sessionOpts
        );
        const t1 = performance.now();

        const streaming = new MoonshineStreamingSession({
          ort,
          chunkedEncoder,
          wholeEncoder,
          prefill,
          step,
          tokenizer,
          bosTokenId: 1,
          eosTokenId: 2,
          emitFrames: 256,
          dtype: 'float32',
        });

        const chunk = 16000;
        const streamStart = performance.now();
        for (let offset = 0; offset + chunk <= pcm.length; offset += chunk) {
          await streaming.pushAudio(pcm.subarray(offset, offset + chunk));
        }
        if (pcm.length % chunk !== 0) {
          await streaming.pushAudio(pcm.subarray(Math.floor(pcm.length / chunk) * chunk));
        }
        const result = await streaming.finalize();
        const streamEnd = performance.now();

        const durationSec = pcm.length / 16000;
        const totalInferenceMs = streamEnd - streamStart;
        const rtfx = (durationSec * 1000) / totalInferenceMs;

        window.testResult = {
          status: result.text === ${JSON.stringify(MOONSHINE_EXPECTED_TEXT)} ? 'PASS_BROWSER_WEBGPU_MOONSHINE' : 'FAIL_TEXT_MISMATCH',
          adapter: {
            vendor: info.vendor || adapter.info?.vendor,
            architecture: info.architecture || adapter.info?.architecture,
            device: info.device || adapter.info?.device,
            description: info.description || adapter.info?.description,
          },
          executionProviders: {
            encoderChunked: ep,
            encoderWhole: ep,
            decoderPrefill: ep,
            decoderStep: ep,
          },
          sessionCreateMs: t1 - t0,
          inferenceTimeMs: totalInferenceMs,
          audioDurationSec: durationSec,
          rtfx,
          resultText: result.text,
          tokenIds: result.tokenIds,
          tokenCount: result.tokenIds.length,
          encoderFrames: result.encoderFrames,
          expectedFrames: embeddedFrameCount(pcm.length / 80),
          matchExpected: result.text === ${JSON.stringify(MOONSHINE_EXPECTED_TEXT)}
        };
      } catch (err) {
        window.testResult = {
          status: 'ERROR',
          error: err.message,
          stack: err.stack
        };
      }
    }
    run();
  </script>
</body>
</html>`;

const QWEN_HTML = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Qwen3-ASR fp16 Verification</title>
  <script src="/node_modules/onnxruntime-web/dist/ort.all.min.js"></script>
</head>
<body>
  <h1>Qwen3-ASR fp16 Browser Verification</h1>
  <pre id="log"></pre>
  <script type="module">
    import { Qwen3AsrSession, Qwen3AsrTokenizer } from '/dist/models/qwen3-asr/session.js';

    window.testResult = null;
    const log = (msg) => console.log('[Qwen3-ASR]', msg);

    function parseNpy(buffer) {
      const view = new DataView(buffer);
      const headerLen = view.getUint16(8, true);
      const headerStr = new TextDecoder().decode(new Uint8Array(buffer, 10, headerLen));
      const match = /'shape':\\s*\\(([^)]*)\\)/.exec(headerStr);
      const dims = match[1].split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
      const offset = 10 + headerLen;
      const count = dims.reduce((a, b) => a * b, 1);
      const data = new Float32Array(buffer.slice(offset, offset + count * 4));
      return { data, dims };
    }

    function parseInt32(buffer) {
      return new Int32Array(buffer.slice(0));
    }

    async function run() {
      try {
        if (!navigator.gpu) throw new Error('navigator.gpu not available');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('requestAdapter returned null');
        const info = await (adapter.requestAdapterInfo ? adapter.requestAdapterInfo() : adapter.info);

        ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
        ort.env.wasm.numThreads = 1;

        const base = '/models/onnx/qwen3-asr-0.6b-fp16';
        const [tokRes, featRes, maskRes, cuRes, prefixRes, suffixRes] = await Promise.all([
          fetch(\`\${base}/tokenizer.json\`),
          fetch(\`\${base}/jfk_features.npy\`),
          fetch(\`\${base}/jfk_features_mask.npy\`),
          fetch(\`\${base}/jfk_cu_seqlens.bin\`),
          fetch(\`\${base}/prompt_prefix_embeds.npy\`),
          fetch(\`\${base}/prompt_suffix_embeds.npy\`),
        ]);

        const tokenizerJson = await tokRes.json();
        const tokenizer = new Qwen3AsrTokenizer(tokenizerJson);
        const features = parseNpy(await featRes.arrayBuffer());
        const mask = parseNpy(await maskRes.arrayBuffer());
        const cuSeqlens = parseInt32(await cuRes.arrayBuffer());
        const prefix = parseNpy(await prefixRes.arrayBuffer());
        const suffix = parseNpy(await suffixRes.arrayBuffer());

        const sessionOpts = (ep, dataName) => ({
          executionProviders: ep,
          graphOptimizationLevel: 'basic',
          enableMemPattern: false,
          enableCpuMemArena: false,
          externalData: dataName ? [{
            path: dataName,
            data: \`\${base}/\${dataName}\`
          }] : undefined
        });

        const t0 = performance.now();
        const front = await ort.InferenceSession.create(
          \`\${base}/audio_front_model.onnx\`,
          sessionOpts(['webgpu', 'wasm'], 'audio_front_model.onnx.data')
        );
        const embed = await ort.InferenceSession.create(
          \`\${base}/embed_tokens_model.onnx\`,
          sessionOpts(['wasm'], 'embed_tokens_model.onnx.data')
        );
        const decoder = await ort.InferenceSession.create(
          \`\${base}/decoder_step_model.onnx\`,
          sessionOpts(['wasm'], 'decoder_step_model.onnx.data')
        );
        const t1 = performance.now();

        const session = new Qwen3AsrSession({
          ort,
          front,
          decoder,
          embed,
          tokenizer,
          cuSeqlens,
          prefixEmbeds: prefix.data,
          suffixEmbeds: suffix.data,
          layers: 28,
          kvHeads: 8,
          headDim: 128,
        });

        const inf0 = performance.now();
        const result = await session.transcribe(features.data, features.dims, mask.data, mask.dims);
        const inf1 = performance.now();
        const infTimeMs = inf1 - inf0;

        const audioDurationSec = 11.0;
        const rtfx = (audioDurationSec * 1000) / infTimeMs;

        window.testResult = {
          status: result.text === ${JSON.stringify(QWEN_EXPECTED_TEXT)} ? 'PASS_BROWSER_WEBGPU_QWEN3_FP16' : 'FAIL_TEXT_MISMATCH',
          adapter: {
            vendor: info.vendor || adapter.info?.vendor,
            architecture: info.architecture || adapter.info?.architecture,
            device: info.device || adapter.info?.device,
            description: info.description || adapter.info?.description,
          },
          executionProviders: {
            audioFront: ['webgpu', 'wasm'],
            embedTokens: ['wasm'],
            decoderStep: ['wasm'],
          },
          sessionCreateMs: t1 - t0,
          inferenceTimeMs: infTimeMs,
          audioDurationSec,
          rtfx,
          resultText: result.text,
          language: result.language,
          tokenIds: result.tokenIds,
          tokenCount: result.tokenIds.length,
          steps: result.steps,
          audioFrames: result.audioFrames,
          matchExpected: result.text === ${JSON.stringify(QWEN_EXPECTED_TEXT)}
        };
      } catch (err) {
        window.testResult = {
          status: 'ERROR',
          error: err.message,
          stack: err.stack
        };
      }
    }
    run();
  </script>
</body>
</html>`;

function createStaticServer(port) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const parsedUrl = new URL(req.url, `http://127.0.0.1:${port}`);
      const pathname = decodeURIComponent(parsedUrl.pathname);

      if (pathname === '/harness/moonshine.html') {
        res.writeHead(200, {
          'Content-Type': 'text/html; charset=utf-8',
          'Access-Control-Allow-Origin': '*',
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(MOONSHINE_HTML);
        return;
      }

      if (pathname === '/harness/qwen.html') {
        res.writeHead(200, {
          'Content-Type': 'text/html; charset=utf-8',
          'Access-Control-Allow-Origin': '*',
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(QWEN_HTML);
        return;
      }

      const filePath = path.join(ROOT, pathname);
      if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
        res.writeHead(404);
        res.end('Not Found: ' + pathname);
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      const contentType = MIME_MAP[ext] || 'application/octet-stream';
      const stat = fs.statSync(filePath);

      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': stat.size,
        'Access-Control-Allow-Origin': '*',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      fs.createReadStream(filePath).pipe(res);
    });

    server.listen(port, '127.0.0.1', () => {
      resolve(server);
    });
  });
}

function runPlaywrightPage(port, harnessPath, timeoutMs = 180000) {
  return new Promise((resolve, reject) => {
    const pyCode = `
import json, sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(
        channel='chrome',
        headless=True,
        args=[
            '--enable-unsafe-webgpu',
            '--ignore-gpu-blocklist',
            '--use-angle=d3d11',
            '--enable-features=Vulkan',
        ]
    )
    version = browser.version
    context = browser.new_context()
    page = context.new_page()
    page.on('console', lambda m: print(f'[browser console] {m.text}', flush=True))
    page.on('pageerror', lambda err: print(f'[page error] {err}', flush=True))
    page.goto('http://127.0.0.1:${port}${harnessPath}', timeout=60000)
    page.wait_for_function('window.testResult !== null', timeout=${timeoutMs})
    res = page.evaluate('window.testResult')
    res['browser_version'] = version
    print('RESULT_PAYLOAD:' + json.dumps(res))
    browser.close()
`;
    const py = spawn('python', ['-u', '-c', pyCode]);
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
        reject(new Error(`Python Playwright process exited with code ${code}`));
        return;
      }
      const match = /RESULT_PAYLOAD:(.*)/.exec(stdout);
      if (!match) {
        reject(new Error('No RESULT_PAYLOAD received from browser runner'));
        return;
      }
      try {
        resolve(JSON.parse(match[1]));
      } catch (e) {
        reject(new Error(`Failed to parse RESULT_PAYLOAD JSON: ${e.message}`));
      }
    });
  });
}

async function main() {
  console.log('=== Wave 2 Task B: Real Browser WebGPU Verification ===');
  const port = 8998;
  const server = await createStaticServer(port);
  console.log(`Static server running at http://127.0.0.1:${port}`);

  fs.mkdirSync(CANONICAL_DIR, { recursive: true });

  try {
    // 1. Verify Moonshine Streaming
    console.log('\n--- 1. Executing Moonshine Streaming Browser WebGPU Verification ---');
    const moonshineInputs = getFileArtifacts(MOONSHINE_INPUT_FILES);
    const audioArtifact = moonshineInputs.find((i) => i.relativePath.endsWith('jfk-short.wav'));

    const moonshineRawResult = await runPlaywrightPage(port, '/harness/moonshine.html');
    if (moonshineRawResult.status !== 'PASS_BROWSER_WEBGPU_MOONSHINE') {
      throw new Error(`Moonshine verification failed: ${JSON.stringify(moonshineRawResult)}`);
    }

    const moonshineReceipt = {
      status: 'PASS_AUTO_BROWSER_WEBGPU_MOONSHINE_STREAMING',
      scope: 'real-headless-chrome-webgpu-moonshine-streaming-verified',
      model_id: 'moonshine-streaming-tiny',
      model_family: 'moonshine-streaming',
      browser_version: moonshineRawResult.browser_version,
      adapter: moonshineRawResult.adapter,
      execution_providers: moonshineRawResult.executionProviders,
      audio: {
        path: audioArtifact.path,
        sha256: audioArtifact.sha256,
        durationSeconds: moonshineRawResult.audioDurationSec,
      },
      inputs: moonshineInputs,
      result: {
        transcript: moonshineRawResult.resultText,
        expected: MOONSHINE_EXPECTED_TEXT,
        matchExpected: moonshineRawResult.matchExpected,
        tokenIds: moonshineRawResult.tokenIds,
        tokenCount: moonshineRawResult.tokenCount,
        encoderFrames: moonshineRawResult.encoderFrames,
        expectedFrames: moonshineRawResult.expectedFrames,
        metrics: {
          sessionCreateMs: moonshineRawResult.sessionCreateMs,
          inferenceTimeMs: moonshineRawResult.inferenceTimeMs,
          rtfx: moonshineRawResult.rtfx,
        },
      },
      timestamp: new Date().toISOString(),
    };

    fs.writeFileSync(
      MOONSHINE_RECEIPT_PATH,
      JSON.stringify(moonshineReceipt, null, 2) + '\n',
      'utf-8'
    );
    console.log(`[PASS] Moonshine Streaming canonical receipt written to:\n  ${MOONSHINE_RECEIPT_PATH}`);

    // 2. Verify Qwen3-ASR fp16
    console.log('\n--- 2. Executing Qwen3-ASR 0.6B fp16 Browser WebGPU Verification ---');
    const qwenInputs = getFileArtifacts(QWEN_INPUT_FILES);

    const qwenRawResult = await runPlaywrightPage(port, '/harness/qwen.html');
    if (qwenRawResult.status !== 'PASS_BROWSER_WEBGPU_QWEN3_FP16') {
      throw new Error(`Qwen3-ASR verification failed: ${JSON.stringify(qwenRawResult)}`);
    }

    const qwenReceipt = {
      status: 'PASS_AUTO_BROWSER_WEBGPU_QWEN3_ASR_FP16',
      scope: 'real-headless-chrome-webgpu-qwen3-asr-fp16-verified',
      model_id: 'Qwen/Qwen3-ASR-0.6B',
      model_family: 'qwen3-asr',
      precision: 'fp16',
      browser_version: qwenRawResult.browser_version,
      adapter: qwenRawResult.adapter,
      execution_providers: qwenRawResult.executionProviders,
      inputs: qwenInputs,
      result: {
        transcript: qwenRawResult.resultText,
        expected: QWEN_EXPECTED_TEXT,
        matchExpected: qwenRawResult.matchExpected,
        language: qwenRawResult.language,
        tokenIds: qwenRawResult.tokenIds,
        tokenCount: qwenRawResult.tokenCount,
        steps: qwenRawResult.steps,
        audioFrames: qwenRawResult.audioFrames,
        metrics: {
          sessionCreateMs: qwenRawResult.sessionCreateMs,
          inferenceTimeMs: qwenRawResult.inferenceTimeMs,
          rtfx: qwenRawResult.rtfx,
        },
      },
      timestamp: new Date().toISOString(),
    };

    fs.writeFileSync(
      QWEN_RECEIPT_PATH,
      JSON.stringify(qwenReceipt, null, 2) + '\n',
      'utf-8'
    );
    console.log(`[PASS] Qwen3-ASR fp16 canonical receipt written to:\n  ${QWEN_RECEIPT_PATH}`);

    console.log('\n=== All Browser WebGPU Verifications PASSED Successfully ===');
  } finally {
    server.close();
  }
}

main().catch((err) => {
  console.error('\nVerification failed:', err);
  process.exit(1);
});
