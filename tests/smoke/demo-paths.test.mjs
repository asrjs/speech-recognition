import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import config, { resolveDemoPaths } from '../../examples/demo/vite.config.js';

for (const directory of ['relocated-checkout', 'checkout with spaces', 'checkout#hash?query']) {
  test(`dev runtime paths follow ${directory}, not the original checkout`, () => {
    const root = resolve(directory);
    const configUrl = pathToFileURL(resolve(root, 'examples/demo/vite.config.js'));
    const paths = resolveDemoPaths(configUrl);
    assert.equal(resolve(paths.repoRoot), root);
    assert.equal(
      decodeURIComponent(paths.wasmBase.slice('/@fs/'.length)),
      `${root.replace(/\\/g, '/')}/node_modules/onnxruntime-web/dist/`,
    );
    assert.equal(new URL(paths.wasmBase, 'http://localhost').search, '');
    assert.equal(new URL(paths.wasmBase, 'http://localhost').hash, '');
    assert.ok(paths.wasmBase.endsWith('/'));
  });
}

test('Vite injects the installed ORT base and allows only the checkout plus existing Canary assets', () => {
  const paths = resolveDemoPaths(new URL('../../examples/demo/vite.config.js', import.meta.url));
  assert.equal(JSON.parse(config.define['import.meta.env.ASRJS_ORT_WASM_BASE']), paths.wasmBase);
  assert.deepEqual(config.server.fs.allow, [paths.repoRoot, 'N:/models']);
  assert.equal(resolve(paths.repoRoot), resolve(fileURLToPath(new URL('../../', import.meta.url))));
});
