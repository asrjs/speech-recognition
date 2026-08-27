import { describe, expect, it } from 'vitest';

import { getNodeBuiltin } from '../src/io/node-builtin.js';
import {
  importNodeModule,
  isNodeLikeRuntime,
  resolveNodePackageSubpathUrl,
} from '../src/io/node-compat.js';

describe('Node built-in bridge', () => {
  it('resolves built-ins without a static Node import', () => {
    const fs = getNodeBuiltin<typeof import('node:fs')>('fs');
    expect(typeof fs.readFileSync).toBe('function');
  });

  it('keeps the root-reachable compatibility bridge functional in Node', async () => {
    expect(isNodeLikeRuntime()).toBe(true);
    const path = await importNodeModule<typeof import('node:path')>('node:path');
    expect(path.basename('/tmp/audio.wav')).toBe('audio.wav');

    const packageUrl = await resolveNodePackageSubpathUrl('onnxruntime-web', 'dist');
    expect(packageUrl).toMatch(/^file:/);
  });
});
