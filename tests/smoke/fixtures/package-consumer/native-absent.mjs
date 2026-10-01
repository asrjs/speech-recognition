import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { realpathSync } from 'node:fs';
import { resolve } from 'node:path';
import { createNativeCpuBackend, createSpeechRuntime, createWasmBackend } from '@asrjs/speech-recognition';
import * as browser from '@asrjs/speech-recognition/browser';

const require = createRequire(import.meta.url);
const expectedRoot = realpathSync(resolve('node_modules/@asrjs/speech-recognition'));
assert.ok(realpathSync(require.resolve('@asrjs/speech-recognition')).startsWith(expectedRoot));
assert.throws(() => require.resolve('onnxruntime-node'), { code: 'MODULE_NOT_FOUND' });
assert.equal(typeof createSpeechRuntime, 'function');
assert.ok(Object.keys(browser).length > 0);
const backend = createNativeCpuBackend();
assert.equal((await backend.probeCapabilities()).available, false);
await assert.rejects(
  backend.createExecutionContext({ modelFamily: 'example', modelId: 'example' }),
  (error) => error.code === 'backend-unavailable',
);
const wasm = createWasmBackend();
assert.equal((await wasm.probeCapabilities()).available, true, 'WASM fallback control must be available');
let modelCreateCalls = 0;
let assetResolveCalls = 0;
const runtime = createSpeechRuntime({
  backends: [backend, wasm],
  modelFamilies: [{
    family: 'native-absent-test',
    supportedBackendIds: ['native-cpu', 'wasm'],
    supports: () => true,
    async createModel() {
      modelCreateCalls += 1;
      throw new Error('Explicit unavailable native request must not create a model or fall back');
    },
  }],
  assetProvider: {
    canResolve: () => true,
    async resolve() {
      assetResolveCalls += 1;
      throw new Error('Unavailable native request must not resolve assets');
    },
  },
});
try {
  await assert.rejects(
    runtime.loadModel({
      family: 'native-absent-test',
      modelId: 'native-test',
      backend: 'native-cpu',
      selectionCriteria: { environment: 'node', allowExperimental: true, requiredPrecision: 'fp32' },
    }),
    (error) => error.code === 'backend-unavailable',
  );
  assert.equal(modelCreateCalls, 0, 'Explicit native selection must not enter the model factory');
  assert.equal(assetResolveCalls, 0, 'Explicit native selection must fail before asset loading');
} finally {
  await runtime.dispose();
}
console.log('[packed-native-absent] PASS: packed imports and public native load rejection without WASM fallback');
