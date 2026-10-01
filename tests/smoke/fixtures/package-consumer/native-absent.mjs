import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { realpathSync } from 'node:fs';
import { resolve } from 'node:path';
import { createNativeCpuBackend, createSpeechRuntime } from '@asrjs/speech-recognition';
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
console.log('[packed-native-absent] PASS: actual packed imports and missing-engine error');
