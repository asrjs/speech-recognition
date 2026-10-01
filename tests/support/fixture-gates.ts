import { statSync } from 'node:fs';
import type { TestContext } from 'vitest';

/** Disabled integration tests must be reported as SKIP, never as a successful return. */
export function requireFixtureOptIn(
  context: Pick<TestContext, 'skip'>,
  enabled: boolean,
  disabledReason: string,
): void {
  if (!enabled) context.skip(disabledReason);
}

/** Once a test is requested, missing configuration or artifacts are failures. */
export function requireFixtureDirectory(value: string | undefined, setting: string): string {
  if (!value?.trim()) throw new Error(`Required fixture setting is missing: ${setting}`);
  if (!statSync(value).isDirectory()) {
    throw new Error(`Required fixture directory is not a directory: ${setting}=${value}`);
  }
  return value;
}

/** Graph names, reference paths and other model contracts are supplied by the caller. */
export function requireFixtureFiles(files: Readonly<Record<string, string | undefined>>): void {
  const entries = Object.entries(files);
  if (entries.length === 0) throw new Error('Required fixture file list must not be empty');
  for (const [label, file] of entries) {
    if (!file?.trim()) throw new Error(`Required fixture setting is missing: ${label}`);
    const stat = statSync(file);
    if (!stat.isFile() || stat.size === 0) {
      throw new Error(`Required fixture must be a nonempty file: ${label}=${file}`);
    }
  }
}
