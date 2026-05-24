import { fetchModelFiles } from '../../runtime/huggingface.js';

function hasModelFile(files: readonly string[], filename: string): boolean {
  return files.some((path) => path === filename || path.endsWith(`/${filename}`));
}

export async function hasHuggingFaceExternalDataFile(
  repoId: string,
  revision: string,
  modelFilename: string | undefined,
): Promise<boolean> {
  if (!modelFilename) {
    return false;
  }

  const dataFilename = `${modelFilename}.data`;
  const files = await fetchModelFiles(repoId, revision);
  return hasModelFile(files, dataFilename);
}
