export function isNodeLikeRuntime(): boolean {
  return false;
}

export async function importNodeModule<T = unknown>(specifier: string): Promise<T> {
  throw new Error(`Node module imports are unavailable in the browser build: ${specifier}`);
}

export async function resolveNodePackageSubpathUrl(
  packageName: string,
  subpath: string,
): Promise<string> {
  throw new Error(
    `Node package resolution is unavailable in the browser build: ${packageName}/${subpath}`,
  );
}
