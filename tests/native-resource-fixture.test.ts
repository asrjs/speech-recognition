import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import {
  resourceGraphSha256,
  writeNativeResourceFixture,
} from './support/native-resource-fixture.js';

it('materializes deterministic resource-only fixtures without a local model or Python', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'asrjs-resource-fixture-'));
  try {
    const { graph, vocab } = await writeNativeResourceFixture(directory);
    const bytes = await readFile(graph);
    expect(bytes.length).toBe(125);
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(resourceGraphSha256);
    expect(await readFile(vocab, 'utf8')).toBe('▁resource 0\n<unk> 1\n');
    await expect(writeNativeResourceFixture(directory)).rejects.toMatchObject({ code: 'EEXIST' });
    expect(await readFile(graph)).toEqual(bytes);
  } finally {
    await rm(directory, { recursive: true });
  }
});
