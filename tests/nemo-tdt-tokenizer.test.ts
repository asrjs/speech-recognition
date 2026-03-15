import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { ParakeetTokenizer } from '@asrjs/speech-recognition/models/nemo-tdt';
import { afterEach, describe, expect, it } from 'vitest';

const tempPaths: string[] = [];

afterEach(async () => {
  await Promise.all(
    tempPaths.splice(0).map(async (filePath) => {
      await fs.rm(filePath, { force: true });
    }),
  );
});

describe('ParakeetTokenizer', () => {
  it('loads vocabularies from file URLs in Node', async () => {
    const filePath = path.join(os.tmpdir(), `parakeet-tokenizer-${Date.now()}.txt`);
    tempPaths.push(filePath);
    await fs.writeFile(filePath, '<blk> 0\nhello 1\nworld 2\n', 'utf8');

    const tokenizer = await ParakeetTokenizer.fromUrl(pathToFileURL(filePath).href);

    expect(tokenizer.blankId).toBe(0);
    expect(tokenizer.vocabSize).toBe(3);
    expect(tokenizer.decode([1, 2])).toBe('helloworld');
  });
});
