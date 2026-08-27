import { describe, expect, it } from 'vitest';

import { getNodeBuiltin } from '../src/io/node-builtin.js';

describe('Node built-in bridge', () => {
  it('resolves built-ins without a static Node import', () => {
    const fs = getNodeBuiltin<typeof import('node:fs')>('fs');
    expect(typeof fs.readFileSync).toBe('function');
  });
});
