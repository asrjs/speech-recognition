import assert from 'node:assert/strict';
import Module, { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';

const metadata = JSON.parse(readFileSync(new URL('../../package.json', import.meta.url), 'utf8'));
const nativeVersion = '1.25.0-dev.20260228-6e72d31970';
assert.equal(metadata.dependencies?.['onnxruntime-node'], undefined);
assert.equal(metadata.optionalDependencies?.['onnxruntime-node'], undefined);
assert.equal(metadata.peerDependencies?.['onnxruntime-node'], nativeVersion);
assert.equal(metadata.peerDependenciesMeta?.['onnxruntime-node']?.optional, true);
assert.equal(metadata.devDependencies?.['onnxruntime-node'], nativeVersion);

// Process-local resolution isolation, not a real package-manager installation.
// Block only the optional engine; never move or remove the installed dependency.
const originalResolve = Module._resolveFilename;
let blockedResolutions = 0;
Module._resolveFilename = function (request, ...args) {
  if (request === 'onnxruntime-node') {
    blockedResolutions++;
    const error = new Error("Cannot find module 'onnxruntime-node'");
    error.code = 'MODULE_NOT_FOUND';
    throw error;
  }
  return originalResolve.call(this, request, ...args);
};

try {
  const api = await import('../../dist/index.js');
  await import('../../dist/browser.js');
  assert.equal(blockedResolutions, 0, 'public imports must not load native ORT');
  const backend = api.createNativeCpuBackend();
  const capabilities = await backend.probeCapabilities();
  assert.equal(capabilities.available, false);
  assert.equal(capabilities.fallbackSuitable, false);
  await assert.rejects(
    backend.createExecutionContext({ modelFamily: 'example', modelId: 'example' }),
    (error) => error.code === 'backend-unavailable',
  );
  assert.ok(blockedResolutions >= 2, 'native operations must exercise blocked resolution');
} finally {
  Module._resolveFilename = originalResolve;
}

// Positive binding control: metadata changes must not break explicitly installed ORT.
const ort = createRequire(import.meta.url)('onnxruntime-node');
assert.ok(ort.listSupportedBackends().some((backend) => backend.name === 'cpu' && backend.bundled));
console.log(
  '[native-optional-peer-smoke] PASS: metadata, isolated absent-engine imports/errors, installed CPU binding',
);
