/**
 * Resolve a Node built-in without creating a static Node dependency in
 * browser bundles. Node 22+ exposes this through process.getBuiltinModule(),
 * which matches the package engine floor.
 */
export function getNodeBuiltin<T>(specifier: string): T {
  const runtime = (
    globalThis as typeof globalThis & {
      process?: {
        getBuiltinModule?: (moduleName: string) => unknown;
      };
    }
  ).process;
  const getBuiltinModule = runtime?.getBuiltinModule;
  if (typeof getBuiltinModule !== 'function') {
    throw new Error(`Node built-in imports are unavailable outside Node.js: ${specifier}`);
  }
  const module = getBuiltinModule.call(runtime, specifier);
  if (module === undefined) {
    throw new Error(`Unable to resolve Node built-in module: ${specifier}`);
  }
  return module as T;
}
