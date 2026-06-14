import { describe, expect, it } from 'vitest';

import {
  importNodeModule,
  isNodeLikeRuntime,
  resolveNodePackageSubpathUrl,
} from '../src/io/browser-node-stub.js';

describe('browser-node-stub', () => {
  it('reports that the browser build is not Node-like', () => {
    expect(isNodeLikeRuntime()).toBe(false);
  });

  it('rejects Node module imports with a browser-specific message', async () => {
    await expect(importNodeModule('node:fs')).rejects.toThrowError(
      'Node module imports are unavailable in the browser build: node:fs',
    );
  });

  it('rejects Node package subpath resolution with a browser-specific message', async () => {
    await expect(resolveNodePackageSubpathUrl('onnxruntime-node', 'bin/test')).rejects.toThrowError(
      'Node package resolution is unavailable in the browser build: onnxruntime-node/bin/test',
    );
  });
});
