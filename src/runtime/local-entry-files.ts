import type { QuantizationMode } from './huggingface.js';
import type { AbortSignalLike } from '../types/index.js';
import type { SpeechModelLocalDirectoryHandleLike, SpeechModelLocalEntry } from './local-types.js';

export function normalizeLocalRelativePath(path: string): string {
  return String(path || '')
    .replace(/\\/g, '/')
    .replace(/^\.\//, '');
}

export function localEntryBasename(path: string): string {
  return normalizeLocalRelativePath(path).split('/').pop() || '';
}

/** Converts a flat list of browser `File` objects into local-entry records. */
export function createLocalEntriesFromFiles(files: readonly File[]): SpeechModelLocalEntry[] {
  return files.map((file) => {
    const path = normalizeLocalRelativePath(file.webkitRelativePath || file.name);
    return { path, basename: localEntryBasename(path), file };
  });
}

/** Recursively collects file entries from a local directory handle. */
export async function collectLocalEntriesFromDirectory(
  dirHandle: SpeechModelLocalDirectoryHandleLike,
  prefix = '',
  signal?: AbortSignalLike | null,
  isAborted: (signal: AbortSignalLike | null | undefined) => void = () => undefined,
): Promise<SpeechModelLocalEntry[]> {
  isAborted(signal);
  const entries: SpeechModelLocalEntry[] = [];
  for await (const [name, handle] of dirHandle.entries()) {
    isAborted(signal);
    if (handle.kind === 'directory' && name === '.git') {
      continue;
    }

    const relativePath = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === 'file') {
      const path = normalizeLocalRelativePath(relativePath);
      entries.push({ path, basename: localEntryBasename(path), handle });
      continue;
    }

    if (handle.kind === 'directory') {
      const nested = await collectLocalEntriesFromDirectory(
        handle,
        relativePath,
        signal,
        isAborted,
      );
      entries.push(...nested);
    }
  }

  return entries;
}

/** Resolves the actual `File` or `Blob` behind a local entry. */
export async function getLocalEntryFile(entry: SpeechModelLocalEntry): Promise<File | Blob> {
  if (entry.file) {
    return entry.file;
  }
  if (entry.handle?.kind === 'file') {
    return entry.handle.getFile();
  }
  throw new Error(
    `Could not access local file entry: ${entry.path || entry.basename || 'unknown'}`,
  );
}

export function findLocalEntry(
  entries: readonly SpeechModelLocalEntry[],
  expectedName: string,
): SpeechModelLocalEntry | null {
  const lower = expectedName.toLowerCase();
  return (
    entries.find(
      (entry) =>
        entry.path.toLowerCase() === lower ||
        entry.basename.toLowerCase() === lower ||
        entry.path.toLowerCase().endsWith(`/${lower}`),
    ) ?? null
  );
}

/** Lists the quantizations a local folder actually contains for one graph base name. */
export function detectLocalQuantModes(
  entries: readonly SpeechModelLocalEntry[],
  baseName: string,
): QuantizationMode[] {
  const names = new Set(entries.map((entry) => entry.basename.toLowerCase()));
  const out: QuantizationMode[] = [];
  if (names.has(`${baseName.toLowerCase()}.onnx`)) out.push('fp32');
  if (names.has(`${baseName.toLowerCase()}.fp16.onnx`)) out.push('fp16');
  if (names.has(`${baseName.toLowerCase()}.int8.onnx`)) out.push('int8');
  return out;
}

export function findLocalSidecar(
  entries: readonly SpeechModelLocalEntry[],
  filename: string,
): SpeechModelLocalEntry | null {
  return findLocalEntry(entries, `${filename}.data`);
}
