import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { describe, expect, it } from 'vitest';

import { createDefaultNodeAssetProvider, createNodeFileSystemAssetProvider } from '@asrjs/speech-recognition/io/node';

async function withTempDir<T>(run: (dir: string) => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), 'asrjs-node-smoke-'));
  try {
    return await run(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

describe('node IO smoke', () => {
  it('reads local assets through the node filesystem provider', async () => {
    await withTempDir(async (dir) => {
      const jsonPath = join(dir, 'fixture.json');
      const textPath = join(dir, 'fixture.txt');
      const bytes = new Uint8Array([1, 2, 3, 4]);

      await writeFile(jsonPath, JSON.stringify({ ok: true, value: 7 }), 'utf8');
      await writeFile(textPath, Buffer.from(bytes));

      const provider = createNodeFileSystemAssetProvider();

      const jsonHandle = await provider.resolve({
        id: 'json-fixture',
        provider: 'path',
        path: jsonPath,
      });
      const textHandle = await provider.resolve({
        id: 'text-fixture',
        provider: 'path',
        path: textPath,
        contentType: 'application/octet-stream',
      });

      expect(await jsonHandle.readJson<{ ok: boolean; value: number }>()).toEqual({
        ok: true,
        value: 7,
      });
      expect(await jsonHandle.getLocator('path')).toBe(jsonPath);
      expect(await textHandle.readBytes()).toEqual(bytes);

      await jsonHandle.dispose();
      await textHandle.dispose();
    });
  });

  it('resolves file URLs through the default asset provider in node', async () => {
    await withTempDir(async (dir) => {
      const filePath = join(dir, 'hello.txt');
      await writeFile(filePath, 'hello from node smoke', 'utf8');

      const provider = createDefaultNodeAssetProvider();
      const handle = await provider.resolve({
        id: 'file-url-fixture',
        url: pathToFileURL(filePath).href,
      });

      expect(await handle.readText()).toBe('hello from node smoke');
      expect(await handle.getLocator('path')).toBe(filePath);

      const locatorUrl = await handle.getLocator('url');
      expect(locatorUrl?.startsWith('file:')).toBe(true);

      await handle.dispose();
      expect(await readFile(filePath, 'utf8')).toBe('hello from node smoke');
    });
  });
});
