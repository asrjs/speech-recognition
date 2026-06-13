export function isNodeLikeRuntime() {
  return false;
}

export async function importNodeModule() {
  throw new Error('Node.js modules are unavailable in the browser demo.');
}

export async function resolveNodePackageSubpathUrl() {
  throw new Error('Node.js package path resolution is unavailable in the browser demo.');
}
