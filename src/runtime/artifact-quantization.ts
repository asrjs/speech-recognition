export const QUANTIZATION_ORDER = ['fp16', 'int8', 'fp32'] as const;
export type QuantizationMode = 'fp16' | 'int8' | 'fp32';
export function normalizePath(path: string): string {
  return String(path || '')
    .replace(/^\.\/+/, '')
    .replace(/\\/g, '/');
}
export function hasFile(files: readonly string[], filename: string): boolean {
  const target = normalizePath(filename);
  return files.some((path) => path === target || path.endsWith(`/${target}`));
}
export function getAvailableQuantModes(
  files: readonly string[],
  baseName: string,
): QuantizationMode[] {
  const options = QUANTIZATION_ORDER.filter((quant) => {
    if (quant === 'fp32') {
      return hasFile(files, `${baseName}.onnx`);
    }
    if (quant === 'fp16') {
      return hasFile(files, `${baseName}.fp16.onnx`);
    }
    return hasFile(files, `${baseName}.int8.onnx`);
  });

  return options.length > 0 ? [...options] : ['fp32'];
}
