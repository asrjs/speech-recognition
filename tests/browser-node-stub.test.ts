import { describe, expect, it } from 'vitest';

import {
  importNodeModule,
  isNodeLikeRuntime,
  resolveNodePackageSubpathUrl,
} from '../src/io/browser-node-stub.js';

describe('browser-node-stub', () => {
  describe('isNodeLikeRuntime', () => {
    it('always returns false', () => {
      expect(isNodeLikeRuntime()).toBe(false);
    });
  });

  describe('importNodeModule', () => {
    it('throws an error indicating node module imports are unavailable', async () => {
      await expect(importNodeModule('node:fs')).rejects.toThrowError(
        'Node module imports are unavailable in the browser build: node:fs',
      );
    });
  });

  describe('resolveNodePackageSubpathUrl', () => {
    it('throws an error indicating node package resolution is unavailable', async () => {
      await expect(resolveNodePackageSubpathUrl('onnxruntime-node', 'bin/test')).rejects.toThrowError(
        'Node package resolution is unavailable in the browser build: onnxruntime-node/bin/test',
      );
    });
  });
});
