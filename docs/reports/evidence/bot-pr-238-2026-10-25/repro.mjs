import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const source = fs.readFileSync(path.join(repo, 'src/presets/parakeet/compat.ts'), 'utf8');
const helperStart = source.indexOf('async function disposeAssetHandles(');
assert.notEqual(helperStart, -1);
const helperEnd = source.indexOf('\n}\n', helperStart);
assert.notEqual(helperEnd, -1);
const helper = source.slice(helperStart, helperEnd + 2);
const previousExpression = 'await Promise.all(handles.map(async (handle) => handle.dispose()));';
const currentExpression = 'await Promise.all(handles.map((handle) => handle.dispose()));';
assert.ok(helper.includes(previousExpression), 'base helper is pre-PR version from diff');
const body = helper.slice(helper.indexOf('{') + 1, helper.lastIndexOf('}'))
  .replace(/:\s*readonly ResolvedAssetHandle\[\]\s*\|\s*undefined/, '')
  .replace(/:\s*Promise<void>/, '');
const previousHelper = new Function(`return async function disposeAssetHandles(handles) {${body}\n}`)();
const afterBody = body.replace(previousExpression, currentExpression);
assert.notEqual(afterBody, body, 'substituted exact PR expression');
const currentHelper = new Function(`return async function disposeAssetHandles(handles) {${afterBody}\n}`)();
async function exercise(fn) {
  const calls = [];
  const handles = [
    { dispose() { calls.push('first'); throw new Error('sync-dispose-failure'); } },
    { dispose() { calls.push('second'); } },
  ];
  let rejected = false, errorMessage = null;
  try { await fn(handles); }
  catch (error) { rejected = true; errorMessage = String(error?.message ?? error); }
  return { calls, rejected, errorMessage };
}
const previous = await exercise(previousHelper);
const current = await exercise(currentHelper);
assert.deepEqual(previous, { calls: ['first', 'second'], rejected: true, errorMessage: 'sync-dispose-failure' });
assert.deepEqual(current, { calls: ['first'], rejected: true, errorMessage: 'sync-dispose-failure' });
const interfaceSource = fs.readFileSync(path.join(repo, 'src/types/io.ts'), 'utf8');
const interfaceMatch = interfaceSource.match(/export interface ResolvedAssetHandle\s*\{([\s\S]*?)\n\}/);
assert.ok(interfaceMatch);
assert.match(interfaceMatch[1], /dispose\(\):\s*Promise<void>\s*\|\s*void/);
const configMatch = source.match(/export interface ParakeetFromUrlsConfig\s*\{([\s\S]*?)\n\}/);
assert.ok(configMatch);
assert.match(configMatch[1], /assetHandles\?:\s*readonly ResolvedAssetHandle\[\]/);
assert.match(source, /async \(\) =>\s*\{\s*await disposeAssetHandles\(config\.assetHandles\);/);
const report = {
 schemaVersion: 1, pr: 238, baseSha: 'e534fb517dcd15a52ac8b52bff85551df89d3673',
 headSha: '37d7b774d0bf86340c2ae8f45b3328605fc79dc9', file: 'src/presets/parakeet/compat.ts',
 currentSourceExpression: previousExpression, proposedPrExpression: currentExpression,
 contract: { disposeAllowsSyncVoid: true, fromUrlsAcceptsCallerSuppliedAssetHandles: true, helperUsedByModelDisposeCallback: true },
 reproduction: { firstDisposeSynchronouslyThrows: true, before: previous, after: current,
 conclusion: 'After PR, the first synchronous disposal exception aborts Array.map; later handles are not disposed.' }
};
const out = path.join(repo, 'docs/reports/evidence/bot-pr-238-2026-10-25/repro.json');
fs.writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
