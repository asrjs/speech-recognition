// Resource loading only: the owned Identity graph is not an ASR parity fixture.
import assert from 'node:assert/strict';
import { configureUnifiedOrtEnv, createUnifiedOrtSession, importUnifiedOrtModule } from '../../dist/inference/ort/index.js';

const identity = Buffer.from('CAgSFmFzcmpzLXJlc291cmNlLWZpeHR1cmU6WwoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRITYXNyanNfcmVzb3VyY2Vfb25seVoTCgVpbnB1dBIKCggIARIECgIIAWIUCgZvdXRwdXQSCgoICAESBAoCCAFCBAoAEBE=', 'base64');
const originalFetch = globalThis.fetch;
globalThis.fetch = async () => { throw new Error('Network disabled for offline ORT smoke'); };
let session;
try {
  const ort = await importUnifiedOrtModule('wasm');
  configureUnifiedOrtEnv(ort, { cpuThreads: 1 });
  assert.match(ort.env.wasm.wasmPaths, /^file:/);
  ({ session } = await createUnifiedOrtSession(ort, identity, { backendId: 'wasm' }));
  const input = new ort.Tensor('float32', Float32Array.of(7), [1]);
  try {
    const outputs = await session.run({ input });
    try { assert.deepEqual(Array.from(outputs.output.data), [7]); }
    finally { for (const output of Object.values(outputs)) output.dispose?.(); }
  } finally { input.dispose?.(); }
  console.log('PASS: installed Node WASM runtime executed Identity with network fetch disabled.');
} finally {
  await session?.release();
  globalThis.fetch = originalFetch;
}
