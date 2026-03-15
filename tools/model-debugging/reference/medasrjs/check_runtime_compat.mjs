import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');
const onnxPath = path.join(PROJECT_ROOT, 'models', 'medasr', 'model.onnx');

function inspectOnnxOps() {
  try {
    const onnx = require('onnxruntime-node/dist/ort.node.min.js') ? null : null;
    void onnx;
  } catch (_e) {
    // no-op
  }
}

async function checkNodeEPs() {
  let ort;
  try {
    ort = (await import('onnxruntime-node')).default;
  } catch (e) {
    return { error: `onnxruntime-node import failed: ${e?.message || String(e)}` };
  }

  const out = {};
  for (const ep of ['cpu', 'cuda']) {
    try {
      const session = await ort.InferenceSession.create(onnxPath, { executionProviders: [ep] });
      out[ep] = { available: true, message: `session created (${session.inputNames.length} inputs)` };
    } catch (e) {
      out[ep] = { available: false, message: e?.message || String(e) };
    }
  }
  return out;
}

async function checkOrtWebBackends() {
  let ortWeb;
  try {
    ortWeb = (await import('onnxruntime-web')).default;
  } catch (e) {
    return { error: `onnxruntime-web import failed: ${e?.message || String(e)}` };
  }

  const out = {};

  try {
    await ortWeb.InferenceSession.create(onnxPath, { executionProviders: ['wasm'] });
    out.wasm = { available: true, message: 'session created' };
  } catch (e) {
    const message = e?.message || String(e);
    out.wasm = {
      available: false,
      message,
      likely_cause: message.includes('model.onnx.data')
        ? 'external_data_loading_in_node'
        : 'backend_or_model_issue',
    };
  }

  try {
    await ortWeb.InferenceSession.create(onnxPath, {
      executionProviders: [{ name: 'webgpu', deviceType: 'gpu', powerPreference: 'high-performance' }, 'wasm'],
    });
    out.webgpu = { available: true, message: 'session created' };
  } catch (e) {
    const message = e?.message || String(e);
    out.webgpu = {
      available: false,
      message,
      likely_cause: message.includes('model.onnx.data')
        ? 'external_data_loading_in_node_or_missing_webgpu'
        : 'backend_or_model_issue',
    };
  }

  return out;
}

(async () => {
  if (!fs.existsSync(onnxPath)) {
    throw new Error(`Model not found: ${onnxPath}`);
  }

  const nodeEPs = await checkNodeEPs();
  const ortWebBackends = await checkOrtWebBackends();

  console.log('--- medasrjs compatibility check ---');
  console.log(JSON.stringify({
    model: onnxPath,
    node: nodeEPs,
    ort_web: ortWebBackends,
    notes: [
      'onnxruntime-node does not use wasm backend; wasm belongs to onnxruntime-web.',
      'webgpu backend requires environment WebGPU support (typically browser/secure context).',
    ],
  }, null, 2));
})();
