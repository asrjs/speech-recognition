import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  requireFixtureDirectory,
  requireFixtureFiles,
  requireFixtureOptIn,
} from './support/fixture-gates.js';

describe('model-neutral fixture prerequisites', () => {
  const directories: string[] = [];
  function fixtureDirectory(): string {
    const directory = mkdtempSync(join(tmpdir(), 'asr-fixture-gate-'));
    directories.push(directory);
    return directory;
  }
  afterEach(() => {
    for (const directory of directories.splice(0)) rmSync(directory, { recursive: true });
  });

  it('reports a disabled integration case as a real Vitest skip', (context) => {
    requireFixtureOptIn(context, false, 'No model fixture requested');
    expect.fail('Disabled fixture test continued execution');
  });

  it('does not skip an explicitly requested integration case', (context) => {
    const skip = vi.spyOn(context, 'skip');
    requireFixtureOptIn(context, true, 'disabled');
    expect(skip).not.toHaveBeenCalled();
    skip.mockRestore();
  });

  it.each([undefined, '', '  '])('rejects absent directory configuration: %s', (value) => {
    expect(() => requireFixtureDirectory(value, 'MODEL_FIXTURE_DIR')).toThrow(
      'Required fixture setting is missing: MODEL_FIXTURE_DIR',
    );
  });

  it('accepts a directory and nonempty model-specific artifact names', () => {
    const directory = fixtureDirectory();
    const graph = join(directory, 'family-owned-name.onnx');
    writeFileSync(graph, 'fixture, not a real ONNX graph');
    expect(requireFixtureDirectory(directory, 'MODEL_FIXTURE_DIR')).toBe(directory);
    expect(() => requireFixtureFiles({ encoder: graph })).not.toThrow();
    expect(() => requireFixtureDirectory(graph, 'MODEL_FIXTURE_DIR')).toThrow('not a directory');
  });

  it('fails on a configured but absent directory or file', () => {
    const missing = join(fixtureDirectory(), 'absent');
    expect(() => requireFixtureDirectory(missing, 'MODEL_FIXTURE_DIR')).toThrow(/ENOENT/);
    expect(() => requireFixtureFiles({ reference: missing })).toThrow(/ENOENT/);
  });

  it('rejects empty discovery and absent required file configuration', () => {
    expect(() => requireFixtureFiles({})).toThrow('file list must not be empty');
    expect(() => requireFixtureFiles({ reference: undefined })).toThrow(
      'Required fixture setting is missing: reference',
    );
  });

  it('rejects empty files and directories masquerading as artifacts', () => {
    const directory = fixtureDirectory();
    const empty = join(directory, 'empty.onnx');
    const fakeGraph = join(directory, 'directory.onnx');
    writeFileSync(empty, '');
    mkdirSync(fakeGraph);
    for (const file of [empty, fakeGraph]) {
      expect(() => requireFixtureFiles({ encoder: file })).toThrow('nonempty file');
    }
  });
});
