import assert from 'node:assert/strict';

import { createSpeechRuntime, createWasmBackend } from '@asrjs/speech-recognition';
import { createBuiltInSpeechRuntime } from '@asrjs/speech-recognition/builtins';
import { createDefaultNodeAssetProvider } from '@asrjs/speech-recognition/io/node';
import { argmax } from '@asrjs/speech-recognition/inference';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

async function main() {
  assert.equal(typeof createSpeechRuntime, 'function');
  assert.equal(typeof createWasmBackend, 'function');
  assert.equal(typeof createDefaultNodeAssetProvider, 'function');
  assert.equal(typeof argmax, 'function');

  const runtime = createSpeechRuntime({
    assetProvider: createDefaultNodeAssetProvider(),
  });
  runtime.registerBackend(createWasmBackend());
  runtime.registerModelFamily(createNemoTdtModelFamily());
  runtime.registerPreset(createParakeetPresetFactory());

  const model = await runtime.loadModel({
    preset: 'parakeet',
    modelId: 'parakeet-tdt-0.6b-v3',
    backend: 'wasm',
  });
  const session = await model.createSession();
  const envelope = await session.transcribe(new Float32Array(16000), {
    detail: 'words',
    responseFlavor: 'canonical+native',
  });

  assert.equal(model.info.family, 'nemo-tdt');
  assert.equal(model.info.preset, 'parakeet');
  assert.equal(envelope.canonical.meta.backendId, 'wasm');
  assert.equal(typeof envelope.canonical.text, 'string');

  await session.dispose();
  await runtime.dispose();

  const builtins = createBuiltInSpeechRuntime({
    useManifestSources: false,
  });
  assert.ok(builtins.listModelFamilies().length >= 3);
  assert.ok(builtins.listPresets().length >= 3);
  await builtins.dispose();

  console.log('[node-import-smoke] ok');
}

main().catch((error) => {
  console.error('[node-import-smoke] failed:', error);
  process.exitCode = 1;
});
