import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { listLocalEntries, resolveSafePath } from '../tools/browser-harness/server.mjs';

describe('local harness model discovery', () => {
  it('preserves nested tokenizer paths and excludes git internals', () => {
    const root = mkdtempSync(path.join(tmpdir(), 'asr-harness-'));
    try {
      mkdirSync(path.join(root, 'tokenizer'));
      mkdirSync(path.join(root, '.git'));
      writeFileSync(path.join(root, 'tokenizer/tokenizer.json'), '{}');
      writeFileSync(path.join(root, '.git/config'), 'private');
      expect(listLocalEntries(root, '/models/test')).toEqual([{
        name: 'tokenizer.json', path: 'tokenizer/tokenizer.json', relativePath: 'tokenizer/tokenizer.json',
        size: 2, url: '/models/test/tokenizer/tokenizer.json',
      }]);
    } finally { rmSync(root, { recursive: true }); }
  });
  it('rejects escapes into a sibling with the same path prefix', () => {
    const root = path.resolve(tmpdir(), 'asr-models');
    expect(resolveSafePath(root, '../asr-models-private/config')).toBeNull();
    expect(resolveSafePath(root, '../secret')).toBeNull();
    expect(resolveSafePath(root, 'tokenizer/tokenizer.json')).toBe(path.join(root, 'tokenizer/tokenizer.json'));
  });
});
